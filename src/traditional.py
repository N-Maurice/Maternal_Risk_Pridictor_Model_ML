"""Stage 5: training-only CV, validation thresholds, learning curves and ablations."""
import time,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV,train_test_split,learning_curve
from threadpoolctl import threadpool_limits
from project import *
from data_stages import pipeline


def stratified_subset(y,n,seed):
    ids=np.arange(len(y))
    if len(ids)<=n: return ids
    return train_test_split(ids,train_size=n,stratify=y,random_state=seed)[0]


def stage5(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        raise RuntimeError('Final test has been opened; training/tuning is frozen. Use saved results or start a separate explicitly documented study.')
    train,val=load_dev(cfg)
    xt,xv=features(train),features(val);yt=train.target;yv=val.target
    sub=stratified_subset(yt,cfg.tuning_size,cfg.seed)
    svmids=stratified_subset(yt,cfg.svm_size,cfg.seed)
    cv=StratifiedKFold(cfg.folds,shuffle=True,random_state=cfg.seed)
    models={
      'logistic':(LogisticRegression(max_iter=700,random_state=cfg.seed),{'model__C':[.1,1,10],'model__class_weight':[None,'balanced']},4),
      'random_forest':(RandomForestClassifier(n_estimators=120,max_depth=14,n_jobs=1,random_state=cfg.seed),{'model__min_samples_leaf':[10,30],'model__max_features':['sqrt',.5],'model__class_weight':[None,'balanced_subsample']},4),
      'hist_gradient_boosting':(HistGradientBoostingClassifier(max_iter=160,early_stopping=True,random_state=cfg.seed),{'model__learning_rate':[.05,.1],'model__max_leaf_nodes':[15,31],'model__l2_regularization':[1.,10.],'model__class_weight':[None,'balanced']},4),
      'svm':(SVC(probability=False,cache_size=256,random_state=cfg.seed),{'model__C':[.5,2.],'model__gamma':['scale',.01],'model__class_weight':[None,'balanced']},3)}
    spaces={name:dict(space=space,n_iter=n,cv_folds=cfg.folds,scoring='average_precision',training_subsample=cfg.svm_size if name=='svm' else len(sub)) for name,(_,space,n) in models.items()}
    write_json(cfg.tables/'stage5_search_spaces.json',spaces)
    results=[]; fitted={}; predictions={}; tuning=[]
    def evaluate(name,model,seconds,fit_rows,params):
        p=model.predict_proba(xv)[:,1];threshold=.5 if name=='dummy' else choose_threshold(yv,p)
        row=dict(model=name,training_seconds=seconds,fit_rows=fit_rows,best_parameters=json.dumps(params,default=str),**metrics(yv,p,threshold))
        results.append(row);fitted[name]=model;predictions[name]=p
        joblib.dump(model,cfg.models/f'{name}.joblib',compress=3)
        print('VALIDATION',name,row,flush=True)
        if name!='dummy' and row['roc_auc']>.85: raise RuntimeError(f'Core {name} exceeds leakage trigger; investigate before proceeding.')
    with threadpool_limits(limits=4):
        for name,base in [('dummy',DummyClassifier(strategy='most_frequent')),('plain_logistic',LogisticRegression(max_iter=700,random_state=cfg.seed))]:
            start=time.perf_counter();m=Pipeline([('preprocessing',pipeline()),('model',base)])
            m.fit(xt,yt);evaluate(name,m,time.perf_counter()-start,len(train),base.get_params())
        for name,(base,space,niter) in models.items():
            idx=svmids if name=='svm' else sub
            estimator=Pipeline([('preprocessing',pipeline()),('model',base)])
            search=RandomizedSearchCV(estimator,space,n_iter=niter,scoring='average_precision',cv=cv,n_jobs=cfg.jobs,random_state=cfg.seed,refit=False,error_score='raise',return_train_score=True)
            start=time.perf_counter();search.fit(xt.iloc[idx],yt.iloc[idx])
            cvtable=pd.DataFrame(search.cv_results_);cvtable['model']=name
            cvtable.to_csv(cfg.tables/f'stage5_cv_{name}.csv',index=False)
            tuning.append(cvtable)
            winner=search.best_params_
            if name=='svm':
                # Platt/sigmoid calibration fitted only inside training folds.
                m=CalibratedClassifierCV(estimator=clone(estimator).set_params(**winner),method='sigmoid',cv=cv,n_jobs=cfg.jobs)
                m.fit(xt.iloc[svmids],yt.iloc[svmids]);fit_rows=len(svmids)
            else:
                m=clone(estimator).set_params(**winner)
                if name=='random_forest':m.set_params(model__n_jobs=cfg.jobs)
                m.fit(xt,yt);fit_rows=len(train)
            evaluate(name,m,time.perf_counter()-start,fit_rows,winner)
    joblib.dump(fitted['random_forest'].named_steps['preprocessing'],cfg.models/'core_preprocessing.joblib')
    comparison=pd.DataFrame(results).sort_values('average_precision',ascending=False)
    comparison.to_csv(cfg.tables/'stage5_validation_models.csv',index=False)
    cvall=pd.concat(tuning,ignore_index=True);cvall.to_csv(cfg.tables/'stage5_tuning.csv',index=False)
    imbalance=cvall[['model','params','mean_test_score','std_test_score']].copy()
    imbalance.to_csv(cfg.tables/'stage5_imbalance_strategies.csv',index=False)
    np.savez_compressed(cfg.state/'traditional_validation_predictions.npz',y=yv.to_numpy(),**predictions)
    eligible=comparison[~comparison.model.isin(['dummy'])]
    best=eligible.iloc[0];bestname=best.model
    tree=comparison[comparison.model.isin(['random_forest','hist_gradient_boosting'])].iloc[0].model
    finalists=dict(traditional=bestname,traditional_threshold=float(best.threshold),best_tree=tree,selection_metric='validation average_precision',test_opened=False)
    write_json(cfg.state/'traditional_selection.json',finalists)
    # Learning curves use only the training subset and refit preprocessing within CV folds.
    curves=[]
    for name in eligible[eligible.model.ne('svm')].head(2).model:
        estimator=clone(fitted[name])
        if name=='random_forest':estimator.set_params(model__n_jobs=1)
        with threadpool_limits(limits=4):
            sizes,tr,va=learning_curve(estimator,xt.iloc[sub],yt.iloc[sub],train_sizes=[.1,.4,1.],cv=cv,scoring='roc_auc',n_jobs=cfg.jobs,shuffle=True,random_state=cfg.seed,error_score='raise')
        for size,t,v in zip(sizes,tr,va):curves.append(dict(model=name,training_rows=int(size),train_auc_mean=t.mean(),train_auc_std=t.std(),cv_auc_mean=v.mean(),cv_auc_std=v.std()))
        plt.figure(figsize=(6,4));plt.plot(sizes,tr.mean(axis=1),'o-',label='Training');plt.plot(sizes,va.mean(axis=1),'o-',label='Training-fold validation');plt.fill_between(sizes,va.mean(axis=1)-va.std(axis=1),va.mean(axis=1)+va.std(axis=1),alpha=.2);plt.xlabel('Training records');plt.ylabel('ROC-AUC');plt.legend();plt.title(name);savefig(cfg,f'stage5_learning_{name}')
    pd.DataFrame(curves).to_csv(cfg.tables/'stage5_learning_curves.csv',index=False)
    # Matched-family ablations; these are diagnostic, never eligible for final selection.
    reference=fitted[bestname]
    if bestname=='svm':reference=fitted[tree]
    estimator=clone(reference.named_steps['model'])
    ablations=[]
    for label,eng,td,singleton in [('baseline_features',False,False,False),('engineered_core',True,False,False),('time_dependent_added',True,True,False),('singleton_only',True,False,True)]:
        tr=train[train.DPLURAL.eq(1)] if singleton else train
        vl=val[val.DPLURAL.eq(1)] if singleton else val
        a,b=features(tr,eng,td),features(vl,eng,td)
        m=Pipeline([('preprocessing',pipeline(engineered=eng,time_features=td)),('model',clone(estimator))])
        start=time.perf_counter()
        with threadpool_limits(limits=4):m.fit(a,tr.target)
        p=m.predict_proba(b)[:,1];threshold=choose_threshold(vl.target,p)
        row=dict(experiment=label,training_rows=len(tr),validation_rows=len(vl),training_seconds=time.perf_counter()-start,**metrics(vl.target,p,threshold))
        if row['roc_auc']>.85:
            if not td:raise RuntimeError('Unexpected core/ablation AUC requires leakage investigation.')
            row['leakage_investigation']='Time-dependent observation-duration and complication fields are not prenatally fixed; this diagnostic is excluded from all selection and final test candidates.'
        ablations.append(row)
    # Target-definition sensitivity, on the existing development split only.
    a=train.GESTREC3.notna();b=val.GESTREC3.notna()
    m=Pipeline([('preprocessing',pipeline()),('model',clone(estimator))]);start=time.perf_counter()
    with threadpool_limits(limits=4):m.fit(xt.loc[a],train.loc[a,'GESTREC3'].eq(1).astype(int))
    p=m.predict_proba(xv.loc[b])[:,1];y=val.loc[b,'GESTREC3'].eq(1).astype(int)
    ablations.append(dict(experiment='GESTREC3_sensitivity',training_rows=int(a.sum()),validation_rows=int(b.sum()),training_seconds=time.perf_counter()-start,**metrics(y,p,choose_threshold(y,p))))
    abl=pd.DataFrame(ablations);base_ap=float(abl.loc[abl.experiment.eq('engineered_core'),'average_precision'].iloc[0]);abl['delta_AP_vs_core']=abl.average_precision-base_ap
    abl.to_csv(cfg.tables/'stage5_ablations.csv',index=False)
    print(comparison.to_string(index=False));print(abl.to_string(index=False))
    write_json(cfg.tables/'stage5_training_scope.json',dict(train_rows=len(train),validation_rows=len(val),tuning_rows=len(sub),svm_training_rows=len(svmids),svm_positive_rows=int(yt.iloc[svmids].sum()),fit_on_test=False,threshold_rule='Maximum validation F1; ties use first threshold',imbalance='None versus balanced weights inside training-fold searches; no oversampling',learning_curve_scope='training-only CV',selection=finalists))
    stage_done(cfg,5,finalists)
    return comparison
