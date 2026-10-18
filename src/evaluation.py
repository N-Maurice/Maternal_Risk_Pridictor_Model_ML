"""Stages 7–8: validation explanations and guarded once-only test predictions."""
import json,time,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_curve,precision_recall_curve)
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits
from project import *


def stage7(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        print('Finalists are frozen; returning saved validation results without reselection.')
        return pd.read_csv(cfg.tables/'stage7_validation_comparison.csv')
    from deep_learning import seed_tensorflow
    seed_tensorflow(cfg)
    trad=json.loads((cfg.state/'traditional_selection.json').read_text());deep=json.loads((cfg.state/'deep_selection.json').read_text())
    comparison=pd.concat([pd.read_csv(cfg.tables/'stage5_validation_models.csv'),pd.read_csv(cfg.tables/'stage6_validation_models.csv')],ignore_index=True).sort_values('average_precision',ascending=False)
    comparison.to_csv(cfg.tables/'stage7_validation_comparison.csv',index=False)
    train,val=load_dev(cfg)
    x=features(val);tree=joblib.load(cfg.models/f'{trad["best_tree"]}.joblib')
    sample=val.sample(min(1200,len(val)),random_state=cfg.seed)
    xs=features(sample)
    with threadpool_limits(limits=4):
        perm=permutation_importance(tree,xs,sample.target,scoring='average_precision',n_repeats=3,random_state=cfg.seed,n_jobs=cfg.jobs)
    ptable=pd.DataFrame(dict(feature=xs.columns,importance=perm.importances_mean,std=perm.importances_std)).sort_values('importance',ascending=False)
    ptable.to_csv(cfg.tables/'stage7_permutation_importance.csv',index=False)
    top=ptable.head(15).iloc[::-1];plt.figure(figsize=(8,5));plt.barh(top.feature,top.importance,xerr=top['std']);plt.xlabel('Drop in validation average precision');savefig(cfg,'stage7_permutation')
    import shap
    transform=tree.named_steps['preprocessing'];estimator=tree.named_steps['model']
    shap_frame=xs.iloc[:200];array=transform.transform(shap_frame)
    # Tree-path-dependent SHAP: no test data and no reference background fitted.
    explainer=shap.TreeExplainer(estimator)
    values=explainer.shap_values(array,check_additivity=False)
    if isinstance(values,list):values=values[-1]
    values=np.asarray(values)
    if values.ndim==3:values=values[:,:,-1]
    names=transform.get_feature_names_out()
    stable=pd.DataFrame(dict(feature=names,mean_absolute_shap=np.abs(values).mean(axis=0))).sort_values('mean_absolute_shap',ascending=False)
    stable.to_csv(cfg.tables/'stage7_shap_importance.csv',index=False)
    top=stable.head(15).iloc[::-1];plt.figure(figsize=(8,5));plt.barh(top.feature,top.mean_absolute_shap);plt.xlabel('Mean absolute SHAP (model raw output units)');savefig(cfg,'stage7_shap')
    write_json(cfg.tables/'stage7_explanation_scope.json',dict(best_tree=trad['best_tree'],permutation_rows=len(xs),permutation_repeats=3,shap_rows=len(array),shap_method='TreeExplainer, tree_path_dependent',shap_output='raw; classifier-specific probability or log-odds units',test_rows_used=0,clinical_reference='https://www.cdc.gov/maternal-infant-health/preterm-birth/',interpretation='Compare history of preterm birth, plurality, prepregnancy hypertension and smoking with CDC associations. Ranking is predictive, not causal; race is a social grouping, not a biological mechanism.'))
    tpred=np.load(cfg.state/'traditional_validation_predictions.npz');dpred=np.load(cfg.state/'deep_validation_predictions.npz')
    assert np.array_equal(tpred['y'],val.target.to_numpy()) and np.array_equal(dpred['y'],val.target.to_numpy()), 'Validation prediction/row alignment mismatch'
    selected={trad['traditional']:(tpred[trad['traditional']],trad['traditional_threshold']),deep['deep']:(dpred[deep['deep']],deep['threshold'])}
    groupings={'age_band':pd.cut(val.MAGER,[-np.inf,19,24,29,34,39,np.inf]).astype(str),'race_ethnicity':val.MRACEHISP.fillna(-1).astype(str),'insurance':val.PAY_REC.fillna(-1).astype(str),'plurality':val.DPLURAL.fillna(-1).astype(str)}
    groups=[]
    for model,(p,threshold) in selected.items():
        overall=metrics(val.target,p,threshold)
        for variable,g in groupings.items():
            for value in sorted(g.unique()):
                mask=(g==value).to_numpy();yy=val.target.to_numpy()[mask];pp=p[mask]
                m=metrics(yy,pp,threshold);gap=m['recall']-overall['recall']
                groups.append(dict(model=model,variable=variable,group=value,rows=int(mask.sum()),positive_rows=int(yy.sum()),recall_difference_vs_overall=gap,disparity_flag=abs(gap)>.1,small_group_flag=len(yy)<100 or yy.sum()<20,**m))
    groups=pd.DataFrame(groups);groups.to_csv(cfg.tables/'stage7_subgroups.csv',index=False)
    finalists=dict(traditional=trad['traditional'],traditional_threshold=trad['traditional_threshold'],deep=deep['deep'],deep_threshold=deep['threshold'],selection_metric='validation average_precision; threshold maximum validation F1',frozen=True,core_only=True,test_opened=False)
    write_json(cfg.state/'frozen_finalists.json',finalists);write_json(cfg.tables/'stage7_frozen_finalists.json',finalists)
    print(comparison.to_string(index=False));print(groups.to_string(index=False));print(finalists)
    stage_done(cfg,7,finalists)
    return comparison


def stage8(cfg):
    guard=cfg.state/'test_evaluation_guard.json'
    if guard.exists():
        prior=json.loads(guard.read_text())
        if prior.get('status')=='complete':
            print('Final test predictions already evaluated. Returning frozen saved results without loading/predicting test rows.')
            return pd.read_csv(cfg.tables/'stage8_final_test_metrics.csv')
        saved_traditional=cfg.state/'test_traditional_predictions.npz'
        saved_deep=cfg.state/'test_deep_predictions.npz'
        if saved_traditional.exists() and saved_deep.exists():
            a=np.load(saved_traditional);b=np.load(saved_deep)
            assert np.array_equal(a['y'],b['y']) and np.array_equal(a['row_id'],b['row_id'])
            print('Recovering evaluation from both cached prediction arrays; no model reload or test prediction.')
            frozen=json.loads((cfg.state/'frozen_finalists.json').read_text())
            return finish_test_metrics(cfg,frozen,a['y'],a['p'],b['p'])
        raise RuntimeError('An earlier final-test run began but not all prediction caches exist. Do not retune, clear the guard, or repeat a completed prediction; inspect the partial run.')
    final=json.loads((cfg.state/'frozen_finalists.json').read_text());assert final['frozen']
    with guard.open('x') as f:json.dump(dict(status='started',models=final,seed=cfg.seed),f,indent=2)
    from deep_learning import seed_tensorflow,predict_model
    import tensorflow as tf
    seed_tensorflow(cfg)
    frame=pd.read_parquet(cfg.sample);ids=np.load(cfg.state/'split_indices.npz')['test'];test=frame.iloc[ids].copy();del frame
    y=test.target.to_numpy()
    traditional=joblib.load(cfg.models/f'{final["traditional"]}.joblib')
    p_traditional=traditional.predict_proba(features(test))[:,1]
    np.savez_compressed(cfg.state/'test_traditional_predictions.npz',y=y,p=p_traditional,row_id=test.row_id.to_numpy())
    keras=tf.keras.models.load_model(cfg.models/f'{final["deep"]}.keras')
    prep=joblib.load(cfg.models/f'{final["deep"]}_preprocessing.joblib')
    p_deep=predict_model(keras,prep,test)
    np.savez_compressed(cfg.state/'test_deep_predictions.npz',y=y,p=p_deep,row_id=test.row_id.to_numpy())
    return finish_test_metrics(cfg,final,y,p_traditional,p_deep)


def finish_test_metrics(cfg,final,y,p_traditional,p_deep):
    """Evaluate cached arrays; this helper never loads a model or test predictors."""
    guard=cfg.state/'test_evaluation_guard.json'
    predictions={final['traditional']:(p_traditional,final['traditional_threshold']),final['deep']:(p_deep,final['deep_threshold'])}
    rows=[]
    for name,(p,threshold) in predictions.items():
        result=dict(model=name,split='test',rows=len(y),**metrics(y,p,threshold));rows.append(result)
        pred=(p>=threshold).astype(int)
        report=pd.DataFrame(classification_report(y,pred,labels=[0,1],target_names=['term','preterm'],output_dict=True,zero_division=0)).T
        report.to_csv(cfg.tables/f'stage8_classification_{name}.csv')
        cm=confusion_matrix(y,pred,labels=[0,1]);pd.DataFrame(cm,index=['actual_term','actual_preterm'],columns=['predicted_term','predicted_preterm']).to_csv(cfg.tables/f'stage8_confusion_{name}.csv')
        norm=cm/cm.sum(axis=1,keepdims=True);pd.DataFrame(norm).to_csv(cfg.tables/f'stage8_confusion_normalized_{name}.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        ConfusionMatrixDisplay(cm,display_labels=['term','preterm']).plot(ax=axes[0],colorbar=False,values_format='d');axes[0].set_title(name+' counts')
        ConfusionMatrixDisplay(norm,display_labels=['term','preterm']).plot(ax=axes[1],colorbar=False,values_format='.3f');axes[1].set_title('Row-normalized');savefig(cfg,f'stage8_confusion_{name}')
    for plot in ['roc','precision_recall','calibration']:
        plt.figure(figsize=(6,5))
        for name,(p,threshold) in predictions.items():
            if plot=='roc':
                a,b,_=roc_curve(y,p);plt.plot(a,b,label=f'{name} AUC={roc_auc_score(y,p):.3f}');plt.xlabel('False positive rate');plt.ylabel('True positive rate')
            elif plot=='precision_recall':
                a,b,_=precision_recall_curve(y,p);plt.plot(b,a,label=f'{name} AP={average_precision_score(y,p):.3f}');plt.xlabel('Recall');plt.ylabel('Precision')
            else:
                observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile');plt.plot(predicted,observed,'o-',label=name);plt.xlabel('Mean predicted probability');plt.ylabel('Observed preterm proportion')
        if plot in ['roc','calibration']:plt.plot([0,1],[0,1],'--',color='gray')
        else:plt.axhline(y.mean(),ls='--',color='gray',label='Prevalence')
        plt.legend();savefig(cfg,f'stage8_{plot}')
    rng=np.random.default_rng(cfg.seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    boot={name:[] for name in predictions}
    for _ in range(cfg.bootstrap_reps):
        sample=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        for name,(p,_) in predictions.items():boot[name].append(roc_auc_score(y[sample],p[sample]))
    bframe=pd.DataFrame(boot);bframe.to_csv(cfg.tables/'stage8_bootstrap_auc_draws.csv',index=False)
    for row in rows:
        low,high=np.quantile(boot[row['model']],[.025,.975]);row.update(auc_ci_lower=float(low),auc_ci_upper=float(high),bootstrap_reps=cfg.bootstrap_reps)
    results=pd.DataFrame(rows);results.to_csv(cfg.tables/'stage8_final_test_metrics.csv',index=False)
    comparison=pd.read_csv(cfg.tables/'stage7_validation_comparison.csv')
    merged=comparison.merge(results.drop(columns=['split','rows','threshold']),on='model',how='left',suffixes=('_validation','_test'))
    merged['test_status']=np.where(merged.model.isin(predictions),'evaluated_once','not_selected_not_evaluated')
    merged.to_csv(cfg.tables/'stage8_all_models.csv',index=False)
    write_json(guard,dict(status='complete',test_rows=len(y),test_predictions_per_model=1,models=list(predictions),no_post_test_tuning=True,bootstrap='500 stratified percentile resamples of cached predictions; conditional on sample; ignores unknown family clustering'))
    print(results.to_string(index=False))
    stage_done(cfg,8,results.to_dict('records'))
    return results
