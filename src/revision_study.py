"""Independent development extension. NEVER reads/predicts original test records.

Run: python src/revision_study.py --root . [--skip-deep]
Original model selection, test guard, results and artifacts remain immutable.
The validation set has been used previously: these results are exploratory,
not a new unbiased test estimate. No external validation is claimed.
"""
from pathlib import Path
import argparse,json,time,hashlib,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/maternal-matplotlib')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import (confusion_matrix,balanced_accuracy_score,matthews_corrcoef,log_loss,
    precision_recall_curve,roc_auc_score,average_precision_score)
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from project import Config,load_dev,metrics,choose_threshold,write_json
from readable_features import readable_features,readable_names


class NativeCategories(TransformerMixin,BaseEstimator):
    """Fit category vocabularies on training rows; HGB handles numeric NaN natively."""
    def fit(self,X,y=None):
        self.numeric_,self.categorical_=readable_names()
        self.categories_={c:sorted(X[c].dropna().unique().tolist()) for c in self.categorical_}
        self.feature_names_in_=np.array(X.columns,dtype=object)
        return self
    def transform(self,X):
        out=X.copy()
        for c,v in self.categories_.items():out[c]=pd.Categorical(out[c],categories=v)
        return out
    def get_feature_names_out(self,input_features=None):return self.feature_names_in_

def readable_preprocessor():
    num,cat=readable_names()
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),num),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat)])

class SigmoidProbabilityCalibrator:
    """Platt scaling on an independent TRAINING calibration partition."""
    def fit(self,p,y):
        self.model=LogisticRegression(C=1e6,max_iter=500).fit(self._logit(p),y)
        return self
    @staticmethod
    def _logit(p):
        p=np.clip(np.asarray(p),1e-7,1-1e-7)
        return np.log(p/(1-p)).reshape(-1,1)
    def predict(self,p):return self.model.predict_proba(self._logit(p))[:,1]

def rich_metrics(y,p,threshold):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    if len(np.unique(y))==2:
        result=metrics(y,p,threshold)
    else:
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else np.nan
        result=dict(roc_auc=np.nan,average_precision=np.nan,precision=precision,recall=recall,
          f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
          accuracy=float((pred==y).mean()),brier=float(np.mean((p-y)**2)),threshold=float(threshold))
    result.update(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
      specificity=tn/(tn+fp) if tn+fp else np.nan,negative_predictive_value=tn/(tn+fn) if tn+fn else np.nan,
      balanced_accuracy=balanced_accuracy_score(y,pred) if len(np.unique(y))==2 else np.nan,
      mcc=matthews_corrcoef(y,pred) if len(np.unique(y))==2 else np.nan,
      log_loss=log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),alert_fraction=float(pred.mean()))
    return result

def threshold_for_recall(y,p,target):
    precision,recall,threshold=precision_recall_curve(y,p)
    eligible=np.flatnonzero(recall[:-1]>=target)
    return float(threshold[eligible[-1]]) if len(eligible) else 0.

def run_revision(cfg,skip_deep=False):
    # These outputs have separate names; never call original stage5/6 or remove a test guard.
    folder=cfg.root/'reports/revision';tables=folder/'tables';figures=folder/'figures'
    model_dir=cfg.models/'revision'
    for p in [tables,figures,model_dir]:p.mkdir(parents=True,exist_ok=True)
    source_guard=cfg.state/'test_evaluation_guard.json'
    guard_hash=hashlib.sha256(source_guard.read_bytes()).hexdigest() if source_guard.exists() else None
    train,val=load_dev(cfg)
    # Only original training rows are split further. Existing validation is external to fitting.
    fit_ids,cal_ids=train_test_split(np.arange(len(train)),test_size=.15,stratify=train.target,random_state=cfg.seed+101)
    core_ids,stop_ids=train_test_split(fit_ids,test_size=.15,stratify=train.target.iloc[fit_ids],random_state=cfg.seed+102)
    x=readable_features(train);xv=readable_features(val)
    y=train.target.to_numpy();yv=val.target.to_numpy()
    np.savez_compressed(model_dir/'training_partition_indices.npz',fit=fit_ids,calibration=cal_ids,deep_fit=core_ids,deep_early_stop=stop_ids)
    sub,_=train_test_split(fit_ids,train_size=min(cfg.tuning_size,len(fit_ids)-1),stratify=y[fit_ids],random_state=cfg.seed)
    cv=StratifiedKFold(cfg.folds,shuffle=True,random_state=cfg.seed)
    rows=[];predictions={};cal_predictions={};thresholds={}
    def record(name,pc,pv,seconds,fit_rows,params):
        threshold=choose_threshold(y[cal_ids],pc)
        result=dict(model=name,scope='development_validation_only',fit_rows=fit_rows,calibration_rows=len(cal_ids),
            training_seconds=seconds,parameters=json.dumps(params),threshold_source='original_training_calibration_partition',
            **rich_metrics(yv,pv,threshold))
        rows.append(result);predictions[name]=pv;cal_predictions[name]=pc;thresholds[name]=threshold
        if result['roc_auc']>.85:raise RuntimeError('Unexpected core AUC: stop and investigate leakage.')
        print('REVISION',name,'AP',result['average_precision'],'AUC',result['roc_auc'],'Brier',result['brier'],flush=True)
    candidates={
       'native_hist_boosting':(Pipeline([('categories',NativeCategories()),('model',HistGradientBoostingClassifier(categorical_features='from_dtype',max_iter=350,early_stopping=True,random_state=cfg.seed))]),
          {'model__learning_rate':[.04,.08],'model__max_leaf_nodes':[15,31],'model__l2_regularization':[1.,10.], 'model__min_samples_leaf':[30,80]},6),
       'extra_trees':(Pipeline([('preprocessing',readable_preprocessor()),('model',ExtraTreesClassifier(n_estimators=180,n_jobs=cfg.jobs,random_state=cfg.seed))]),
          {'model__min_samples_leaf':[10,30],'model__max_features':['sqrt',.7],'model__class_weight':[None,'balanced']},4)}
    write_json(tables/'search_plan.json',dict(models={n:dict(space=s,configurations=k) for n,(_,s,k) in candidates.items()},cv_folds=cfg.folds,search_rows=len(sub),scoring='average_precision',seed=cfg.seed))
    for name,(estimator,space,budget) in candidates.items():
        start=time.perf_counter()
        with threadpool_limits(limits=4):
            search=RandomizedSearchCV(estimator,space,n_iter=budget,cv=cv,scoring={'AP':'average_precision','AUC':'roc_auc','Brier':'neg_brier_score'},refit='AP',n_jobs=1,random_state=cfg.seed,error_score='raise',return_train_score=True)
            search.fit(x.iloc[sub],y[sub])
            pd.DataFrame(search.cv_results_).to_csv(tables/f'cv_{name}.csv',index=False)
            model=search.best_estimator_;model.fit(x.iloc[fit_ids],y[fit_ids])
            pc=model.predict_proba(x.iloc[cal_ids])[:,1];pv=model.predict_proba(xv)[:,1]
        elapsed=time.perf_counter()-start
        joblib.dump(model,model_dir/f'{name}.joblib',compress=3)
        record(name,pc,pv,elapsed,len(fit_ids),search.best_params_)
        calibrator=SigmoidProbabilityCalibrator().fit(pc,y[cal_ids]);joblib.dump(calibrator,model_dir/f'{name}_sigmoid.joblib')
        # Threshold selection on calibration is development fitting, not unbiased calibration assessment.
        record(name+'_sigmoid',calibrator.predict(pc),calibrator.predict(pv),elapsed,len(fit_ids),{'base':search.best_params_,'calibration':'sigmoid on training holdout'})
    if not skip_deep:
        import tensorflow as tf
        from deep_learning import architecture,dataset,seed_tensorflow
        seed_tensorflow(cfg)
        # Existing embedding preprocessor uses CDC feature keys; fit only on deep training.
        from deep_learning import EmbeddingPreprocessor
        from project import features
        rawx=features(train);rawv=features(val)
        params=dict(depth=1,units=96,dropout=.3,learning_rate=.0005,batch_size=512,optimizer='rmsprop',l2=.0001)
        for kind,weighted in [('sequential',False),('functional',False),('functional',True)]:
            name=f'{kind}_'+('weighted' if weighted else 'unweighted')
            tf.keras.backend.clear_session();seed_tensorflow(cfg)
            if kind=='functional':
                prep=EmbeddingPreprocessor().fit(rawx.iloc[core_ids]);a=prep.transform(rawx);b=prep.transform(rawv)
            else:
                prep=readable_preprocessor().fit(x.iloc[core_ids]);a=prep.transform(x).astype('float32');b=prep.transform(xv).astype('float32')
            def select(ids):return {k:v[ids] for k,v in a.items()} if isinstance(a,dict) else a[ids]
            model=architecture(kind,prep,params,input_dim=None if isinstance(a,dict) else a.shape[1])
            weights={0:len(core_ids)/(2*(y[core_ids]==0).sum()),1:len(core_ids)/(2*(y[core_ids]==1).sum())} if weighted else None
            start=time.perf_counter();path=model_dir/f'{name}.keras'
            history=model.fit(dataset(select(core_ids),y[core_ids],512,True,cfg.seed),validation_data=dataset(select(stop_ids),y[stop_ids],512),epochs=24,class_weight=weights,verbose=0,shuffle=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc',mode='max',patience=2,factor=.5),tf.keras.callbacks.ModelCheckpoint(path,monitor='val_pr_auc',mode='max',save_best_only=True)])
            pc=model.predict(dataset(select(cal_ids),batch_size=1024),verbose=0).ravel();pv=model.predict(dataset(b,batch_size=1024),verbose=0).ravel()
            elapsed=time.perf_counter()-start;joblib.dump(prep,model_dir/f'{name}_preprocessing.joblib')
            pd.DataFrame(history.history).to_csv(tables/f'history_{name}.csv',index=False)
            record(name,pc,pv,elapsed,len(core_ids),dict(**params,class_weight=weights,early_stopping_rows=len(stop_ids),epochs=len(history.history['loss'])))
            cal=SigmoidProbabilityCalibrator().fit(pc,y[cal_ids]);joblib.dump(cal,model_dir/f'{name}_sigmoid.joblib')
            record(name+'_sigmoid',cal.predict(pc),cal.predict(pv),elapsed,len(core_ids),{'calibration':'sigmoid on training holdout','weighted':weighted})
    # Fixed equal-weight pooling: no validation-fitted stacking or optimized weights.
    members=['native_hist_boosting_sigmoid','functional_unweighted_sigmoid']
    if all(m in predictions for m in members):
        record('equal_weight_tree_neural_ensemble',np.mean([cal_predictions[m] for m in members],axis=0),np.mean([predictions[m] for m in members],axis=0),sum(r['training_seconds'] for r in rows if r['model'] in members),len(fit_ids),{'members':members,'weights':[.5,.5]})
    results=pd.DataFrame(rows).sort_values('average_precision',ascending=False)
    results.to_csv(tables/'validation_metrics.csv',index=False)
    np.savez_compressed(model_dir/'validation_predictions.npz',y=yv,**predictions)
    chosen=results.iloc[0]
    write_json(model_dir/'selected_candidate.json',dict(selected_candidate=chosen['model'],threshold=float(chosen['threshold']),
        scope='development-only; never evaluated on original test',input_schema='readable core features; raw CDC records also accepted',
        ensemble_members=members,ensemble_weights=[.5,.5],selection='existing development-validation AP; no independent revised test'))
    trades=[];curves=[];intervals=[]
    rng=np.random.default_rng(cfg.seed)
    # Same bootstrap indices for every model permit paired, conditional comparisons.
    positive=np.flatnonzero(yv==1);negative=np.flatnonzero(yv==0)
    draws=[np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)] for _ in range(200)]
    for name,p in predictions.items():
        for target in [.5,.7,.8,.9]:
            threshold=threshold_for_recall(y[cal_ids],cal_predictions[name],target)
            trades.append(dict(model=name,requested_calibration_recall=target,**rich_metrics(yv,p,threshold)))
        auc=[roc_auc_score(yv[i],p[i]) for i in draws]
        ap=[average_precision_score(yv[i],p[i]) for i in draws]
        intervals.append(dict(model=name,auc_low=np.quantile(auc,.025),auc_high=np.quantile(auc,.975),ap_low=np.quantile(ap,.025),ap_high=np.quantile(ap,.975),bootstrap_reps=len(draws),scope='conditional validation uncertainty, not final test'))
        for threshold in np.linspace(.02,.4,40):
            m=rich_metrics(yv,p,threshold)
            curves.append(dict(model=name,risk_threshold=threshold,net_benefit=m['tp']/len(yv)-m['fp']/len(yv)*threshold/(1-threshold),treat_all=yv.mean()-(1-yv.mean())*threshold/(1-threshold),treat_none=0))
    pd.DataFrame(trades).to_csv(tables/'recall_threshold_tradeoffs.csv',index=False)
    pd.DataFrame(intervals).to_csv(tables/'validation_confidence_intervals.csv',index=False)
    decision=pd.DataFrame(curves);decision.to_csv(tables/'decision_curve.csv',index=False)
    selected=results.iloc[0]['model']
    if guard_hash is not None:assert hashlib.sha256(source_guard.read_bytes()).hexdigest()==guard_hash
    write_json(tables/'study_manifest.json',dict(status='complete',date='2026-10-18',selection_metric='development validation AP',selected_candidate=selected,
        original_test_predictions=0,original_test_guard_sha256=guard_hash,training_rows=len(train),fit_rows=len(fit_ids),calibration_rows=len(cal_ids),deep_fit_rows=len(core_ids),deep_early_stopping_rows=len(stop_ids),validation_rows=len(val),
        caveat='Original validation has already supported earlier selection. Revision estimates are exploratory. Original test was previously opened and is never reused; a new external cohort is needed for revised final claims.',skip_deep=skip_deep))
    fig,axes=plt.subplots(1,2,figsize=(12,5));view=results.iloc[::-1]
    axes[0].barh(view.model,view.average_precision);axes[0].set_title('Development validation AP (higher better)')
    axes[1].barh(view.model,view.brier);axes[1].set_title('Development validation Brier (lower better)');fig.tight_layout();fig.savefig(figures/'revision_model_comparison.png',dpi=300,bbox_inches='tight');plt.close(fig)
    plt.figure(figsize=(8,5))
    for name in [selected,'native_hist_boosting_sigmoid']:
        part=decision[decision.model.eq(name)];plt.plot(part.risk_threshold,part.net_benefit,label=name)
    plt.plot(part.risk_threshold,part.treat_all,'--',label='flag all');plt.axhline(0,color='black',label='flag none');plt.xlabel('Assumed risk threshold');plt.ylabel('Exploratory net benefit');plt.legend();plt.tight_layout();plt.savefig(figures/'revision_decision_curve.png',dpi=300);plt.close()
    # Reload proof is integral to a hosted run, not dependent on an external script.
    from revision_inference import RevisedPredictor
    positions=np.r_[np.flatnonzero(yv==0)[:5],np.flatnonzero(yv==1)[:5]]
    proof=[]
    for name in list(dict.fromkeys([selected,'native_hist_boosting_sigmoid'])):
        actual=RevisedPredictor(model_dir,name).predict_proba(xv.iloc[positions])[:,1]
        expected=predictions[name][positions]
        difference=float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        proof.append(dict(model=name,rows=len(positions),source='development validation reload examples',max_absolute_prediction_difference=difference,passed=True))
    pd.DataFrame(proof).to_csv(tables/'reload_proof.csv',index=False)
    print(results[['model','roc_auc','average_precision','recall','brier']].to_string(index=False),flush=True)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.');parser.add_argument('--skip-deep',action='store_true')
    args=parser.parse_args()
    # Canonical module identity makes custom joblib transformers reloadable.
    from revision_study import run_revision as run_importable_revision
    run_importable_revision(Config(Path(args.root).resolve()).setup(),args.skip_deep)
