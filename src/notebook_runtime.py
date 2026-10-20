"""Portable orchestration for the comprehensive notebook; no implicit installation."""
from pathlib import Path
import json,os,sys,shutil,hashlib

def configure_inputs(root,input_root=None):
    """Link uploaded inputs into writable workspace; never modify original data."""
    root=Path(root).resolve()
    if input_root is None:return
    source=Path(input_root).resolve()
    names={
      'Nat2024PublicUS.c20250512.r20250708.txt':root/'Data/Nat2024PublicUS.c20250512.r20250708.txt',
      'UserGuide2024.pdf':root/'Data/UserGuide2024.pdf',
      'natality2024_variable_layout.csv':root/'Data/natality2024_variable_layout.csv',
      'modeling_sample.parquet':root/'data/processed/modeling_sample.parquet',
      'split_indices.npz':root/'data/processed/state/split_indices.npz',
      'input_provenance.json':root/'data/processed/input_provenance.json',
    }
    for name,destination in names.items():
        if destination.exists():continue
        matches=list(source.rglob(name))
        if len(matches)>1:raise ValueError(f'Multiple {name} inputs found. Point INPUT_ROOT to one dataset folder.')
        if matches:
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.symlink_to(matches[0])
    provenance=root/'data/processed/input_provenance.json'
    if provenance.exists():
        info=json.loads(provenance.read_text())
        for label,path in [('sample',root/'data/processed/modeling_sample.parquet'),('split',root/'data/processed/state/split_indices.npz')]:
            if not path.exists():raise FileNotFoundError(f'Missing {label} input from attached provenance.')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info[label+'_sha256']:
                raise ValueError(f'{label} checksum differs from uploaded provenance; do not mix sample and split versions.')

def ensure_splits(cfg):
    """Create the documented split only on a new workspace; never replace it."""
    import numpy as np
    import pandas as pd
    from project import partitions
    path=cfg.state/'split_indices.npz'
    if not path.exists():
        if (cfg.state/'test_evaluation_guard.json').exists():raise RuntimeError('Missing frozen split: restore the original manifest.')
        frame=pd.read_parquet(cfg.sample)
        train,validation,test=partitions(frame,cfg)
        np.savez_compressed(path,train=train,validation=validation,test=test)
    return path

def check_environment(include_deep=False):
    import importlib.metadata as metadata
    required=['numpy','pandas','scikit-learn','pyarrow','matplotlib','joblib']
    if include_deep:required+=['tensorflow','keras']
    versions={'python':sys.version.split()[0]}
    for package in required:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:
            if package=='tensorflow':versions[package]=metadata.version('tensorflow_cpu')
            else:raise RuntimeError(f'Missing {package}. Run the optional installation cell, then restart the kernel.')
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__)<Version('1.4'):
        raise RuntimeError('scikit-learn >=1.4 is required for native categorical boosting and current encoder APIs.')
    if include_deep:
        import keras
        if Version(keras.__version__)<Version('3'):raise RuntimeError('Keras 3 is required for this export workflow.')
        import tensorflow as tf
        versions['physical_gpus']=[device.name for device in tf.config.list_physical_devices('GPU')]
    return versions


def verify_final_test_scope(cfg):
    """Do not mistake a copied original sample for an untouched new cohort."""
    guard=cfg.state/'test_evaluation_guard.json'
    if guard.exists():return  # stage8 itself safely returns caches or refuses partial runs.
    uploaded=cfg.data/'input_provenance.json'
    if uploaded.exists() and json.loads(uploaded.read_text()).get('original_test_already_evaluated'):
        raise RuntimeError('This input bundle contains the previously evaluated original cohort. Revised final testing is prohibited.')
    references=[cfg.root/'reference_outputs/reports/revision/tables/hosted_input_provenance.json',cfg.root/'reports/revision/tables/hosted_input_provenance.json']
    for reference in references:
        if not reference.exists():continue
        info=json.loads(reference.read_text());expected=info.get('original_test_row_ids_sha256')
        if expected:
            import numpy as np
            import pandas as pd
            frame=pd.read_parquet(cfg.sample,columns=['row_id'])
            test=np.load(cfg.state/'split_indices.npz')['test']
            actual=hashlib.sha256(np.sort(frame.row_id.iloc[test].to_numpy()).tobytes()).hexdigest()
            if actual==expected:
                raise RuntimeError('Copied original test membership is already known. Keep revision evaluation on development data; obtain an independent future cohort.')

def ingest_fresh(cfg):
    """Full raw audit for a new run, preserving the original Stage 1 procedures."""
    import pandas as pd
    from ingestion import verify_layout,build_dictionary,stream_raw_audit,parse_audit
    from finalize_ingestion import finalize
    raw=cfg.root/'Data/Nat2024PublicUS.c20250512.r20250708.txt'
    guide=cfg.root/'Data/UserGuide2024.pdf';layout=cfg.root/'Data/natality2024_variable_layout.csv'
    for p in [raw,guide,layout]:
        if not p.is_file():raise FileNotFoundError(f'Required raw-ingestion input: {p}. The raw file is stored in Data/Nat2024us.zip; extract it with: unzip Data/Nat2024us.zip -d Data')
    if shutil.which('pdftotext') is None:
        raise RuntimeError('Raw audit requires pdftotext (Poppler). Install it in your environment or use the audited parquet mode.')
    audit=cfg.data/'stage1_audit';audit.mkdir(parents=True,exist_ok=True)
    selected,evidence,control=verify_layout(layout,guide,audit)
    dictionary,rules=build_dictionary(selected,evidence)
    dictionary.to_csv(cfg.tables/'stage1_data_dictionary.csv',index=False)
    summary=stream_raw_audit(raw,audit/'raw_stream_audit.json')
    assert summary['raw_rows']==control['guide_us_record_count']
    parse_audit(raw,selected,rules,summary,audit,cfg.cache,25000)
    shutil.copyfile(audit/'raw_stream_audit.json',cfg.tables/'stage1_audit_summary.json')
    for name in ['missingness','guide_position_verification','reporting_flags','raw_code_counts']:
        shutil.copyfile(audit/f'{name}.csv',cfg.tables/f'stage1_{name}.csv')
    return finalize(cfg.root)


def export_fitted_models(cfg):
    """Export selected models from parquet-only training; no raw audit/test required."""
    import numpy as np
    import pandas as pd
    import joblib
    import tensorflow as tf
    from project import load_dev,features
    from deep_learning import predict_model
    selection=cfg.state/'frozen_finalists.json'
    if not selection.exists():raise RuntimeError('Complete training and validation selection before exporting models.')
    finalists=json.loads(selection.read_text());train,_=load_dev(cfg)
    examples=pd.concat([train[train.target.eq(0)].head(5),train[train.target.eq(1)].head(5)])
    traditional=joblib.load(cfg.models/f'{finalists["traditional"]}.joblib')
    p=traditional.predict_proba(features(examples))[:,1]
    joblib.dump(traditional,cfg.models/'best_traditional.joblib',compress=3)
    check=joblib.load(cfg.models/'best_traditional.joblib').predict_proba(features(examples))[:,1]
    proof=[dict(artifact='best_traditional.joblib',rows=len(examples),max_absolute_difference=float(np.max(np.abs(p-check))),passed=bool(np.allclose(p,check)))]
    model=tf.keras.models.load_model(cfg.models/f'{finalists["deep"]}.keras')
    prep=joblib.load(cfg.models/f'{finalists["deep"]}_preprocessing.joblib')
    joblib.dump(prep,cfg.models/'best_deep_preprocessing.joblib')
    expected=predict_model(model,prep,examples)
    for name in ['best_deep.keras','best_deep.h5']:
        model.save(cfg.models/name)
        loaded=tf.keras.models.load_model(cfg.models/name);actual=predict_model(loaded,prep,examples)
        proof.append(dict(artifact=name,rows=len(examples),max_absolute_difference=float(np.max(np.abs(expected-actual))),passed=bool(np.allclose(expected,actual,atol=1e-6))))
    exported=cfg.models/'best_deep_savedmodel'
    if exported.exists():shutil.rmtree(exported)
    model.export(exported,format='tf_saved_model',verbose=False)
    values=prep.transform(features(examples))
    inputs={k:tf.convert_to_tensor(v) for k,v in values.items()} if isinstance(values,dict) else tf.convert_to_tensor(np.asarray(values,dtype='float32'))
    actual=np.asarray(tf.saved_model.load(exported).serve(inputs)).ravel()
    proof.append(dict(artifact='best_deep_savedmodel',rows=len(examples),max_absolute_difference=float(np.max(np.abs(expected-actual))),passed=bool(np.allclose(expected,actual,atol=1e-6))))
    result=pd.DataFrame(proof);assert result.passed.all()
    result.to_csv(cfg.tables/'portable_reload_proof.csv',index=False)
    (cfg.models/'portable_export_scope.json').write_text(json.dumps(dict(selection='validation-selected candidates',test_predictions_performed_here=0,reload_rows='original training examples',finalists=finalists),indent=2)+'\n')
    return result
