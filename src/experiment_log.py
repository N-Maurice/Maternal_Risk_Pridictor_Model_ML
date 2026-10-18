"""Unified experiment registry from verified results; never fits or predicts a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

def build_registry(cfg,tensorboard=False):
    rows=[]
    existing_path=cfg.tables/'experiment_registry.csv'
    existing=pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()

    definitions=[('baseline_and_classical',cfg.tables/'stage5_validation_models.csv','best_parameters'),
                 ('neural_networks',cfg.tables/'stage6_validation_models.csv','best_parameters'),
                 ('development_experiments',cfg.root/'reports/revision/tables/validation_metrics.csv','parameters')]
    for family,path,params in definitions:
        if not path.exists():continue
        for result in pd.read_csv(path).to_dict('records'):
            result.update(experiment_family=family,evaluation_scope='development_validation',parameter_source=params,
                          evidence_file=str(path.relative_to(cfg.root)),record_type='saved_experiment_result')
            rows.append(result)
    registry=pd.DataFrame(rows)
    # F2 is a supplementary recall-weighted summary. It does not change any threshold.
    predictions={}
    for name in ['traditional_validation_predictions.npz','deep_validation_predictions.npz']:
        file=cfg.state/name
        if file.exists():
            values=np.load(file)
            for model in values.files:
                if model!='y':predictions[('baseline_and_classical' if name.startswith('traditional') else 'neural_networks',model)]=(values['y'],values[model])
    file=cfg.models/'revision/validation_predictions.npz'
    if file.exists():
        values=np.load(file)
        for model in values.files:
            if model!='y':predictions[('development_experiments',model)]=(values['y'],values[model])
    f2=[];f2_sources=[]
    for result in registry.itertuples():
        pair=predictions.get((result.experiment_family,result.model))
        if pair is not None:
            value=fbeta_score(pair[0],pair[1]>=result.threshold,beta=2,zero_division=0)
            source='cached_validation_predictions'
        else:
            matched=existing[(existing.model==result.model)&(existing.experiment_family==result.experiment_family)] if len(existing) else pd.DataFrame()
            if len(matched) and np.isclose(float(matched.iloc[0].threshold),result.threshold,atol=1e-12,rtol=0) and np.isclose(float(matched.iloc[0].average_precision),result.average_precision,atol=1e-12,rtol=0):
                value=float(matched.iloc[0].f2);source='saved_registry_matching_result_and_threshold'
            else:value=np.nan;source='unavailable_without_prediction_cache'
        f2.append(value);f2_sources.append(source)
    registry['f2']=f2
    registry['f2_source']=f2_sources
    directory=cfg.root/'reports/tables';directory.mkdir(parents=True,exist_ok=True)
    registry.to_csv(directory/'experiment_registry.csv',index=False)
    (directory/'experiment_registry.json').write_text(registry.to_json(orient='records',indent=2)+'\n')
    if tensorboard:
        import importlib.util
        if importlib.util.find_spec("tensorboard") is None:
            raise RuntimeError("CSV/JSON saved. Install tensorboard to enable the optional dashboard, then restart the kernel.")
        import tensorflow as tf
        for result in registry.to_dict('records'):
            path=cfg.root/'runs'/result['experiment_family']/result['model']
            writer=tf.summary.create_file_writer(str(path))
            with writer.as_default():
                for name in ['roc_auc','average_precision','precision','recall','f1','f2','brier','threshold','training_seconds']:
                    value=result.get(name)
                    if value is not None and np.isfinite(value):tf.summary.scalar('validation/'+name,value,step=0)
                tf.summary.text('scope','Verified saved development results; event creation does not retrain or reevaluate test data.',step=0)
                tf.summary.text('parameters',str(result.get(result['parameter_source'],'')),step=0)
            writer.flush();writer.close()
        for folder,pattern in [(cfg.tables,'stage6_history_*.csv'),(cfg.root/'reports/revision/tables','history_*.csv')]:
            for source in folder.glob(pattern):
                history=pd.read_csv(source)
                writer=tf.summary.create_file_writer(str(cfg.root/'runs/training_histories'/source.stem))
                with writer.as_default():
                    tf.summary.text('scope','Recorded epoch history; events replay saved measurements without retraining.',step=0)
                    for epoch,row in enumerate(history.to_dict('records'),start=1):
                        for metric,value in row.items():
                            if metric!='epoch' and np.isfinite(value):tf.summary.scalar(metric,value,step=epoch)
                writer.flush();writer.close()
    return registry
