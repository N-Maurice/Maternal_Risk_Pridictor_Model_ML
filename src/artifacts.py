"""Stages 9–10: export/reload proof, fact sheets and honest assignment checks."""
from pathlib import Path
import sys,subprocess,json,hashlib,shutil,platform,importlib.metadata as metadata
import numpy as np
import pandas as pd
import joblib
from project import *


def stage9(cfg):
    import tensorflow as tf
    from deep_learning import seed_tensorflow,dataset,predict_model
    seed_tensorflow(cfg)
    final=json.loads((cfg.state/'frozen_finalists.json').read_text())
    assert json.loads((cfg.state/'test_evaluation_guard.json').read_text())['status']=='complete'
    train,_=load_dev(cfg)
    sample=pd.concat([train[train.target.eq(0)].head(5),train[train.target.eq(1)].head(5)]).copy()
    model=joblib.load(cfg.models/f'{final["traditional"]}.joblib')
    reference=model.predict_proba(features(sample))[:,1]
    joblib.dump(model,cfg.models/'best_traditional.joblib',compress=3)
    if hasattr(model,'named_steps'):
        joblib.dump(model.named_steps['preprocessing'],cfg.models/'best_traditional_preprocessing.joblib')
    else:
        joblib.dump({'preprocessing_embedded_in_calibrated_estimators':model},cfg.models/'best_traditional_preprocessing.joblib')
    reload=joblib.load(cfg.models/'best_traditional.joblib');p=reload.predict_proba(features(sample))[:,1]
    checks=[dict(artifact='best_traditional.joblib',max_absolute_difference=float(np.max(np.abs(p-reference))),passed=bool(np.allclose(p,reference,atol=1e-8)),sample_rows=len(sample))]
    keras=tf.keras.models.load_model(cfg.models/f'{final["deep"]}.keras')
    prep=joblib.load(cfg.models/f'{final["deep"]}_preprocessing.joblib')
    joblib.dump(prep,cfg.models/'best_deep_preprocessing.joblib')
    keras.save(cfg.models/'best_deep.keras')
    reference=predict_model(keras,prep,sample)
    restored=tf.keras.models.load_model(cfg.models/'best_deep.keras')
    p=predict_model(restored,joblib.load(cfg.models/'best_deep_preprocessing.joblib'),sample)
    checks.append(dict(artifact='best_deep.keras',max_absolute_difference=float(np.max(np.abs(p-reference))),passed=bool(np.allclose(p,reference,atol=1e-6)),sample_rows=len(sample)))
    export_path=cfg.models/'best_deep_savedmodel'
    if export_path.exists():shutil.rmtree(export_path)
    keras.export(export_path,format='tf_saved_model',verbose=False)
    saved=tf.saved_model.load(export_path)
    x=prep.transform(features(sample))
    tensors={k:tf.convert_to_tensor(v) for k,v in x.items()} if isinstance(x,dict) else tf.convert_to_tensor(np.asarray(x,dtype='float32'))
    p=saved.serve(tensors)
    if isinstance(p,dict):p=next(iter(p.values()))
    p=np.asarray(p).ravel()
    checks.append(dict(artifact='best_deep_savedmodel',max_absolute_difference=float(np.max(np.abs(p-reference))),passed=bool(np.allclose(p,reference,atol=1e-6)),sample_rows=len(sample)))
    try:
        keras.save(cfg.models/'best_deep.h5')
        h5=tf.keras.models.load_model(cfg.models/'best_deep.h5',compile=False)
        p=predict_model(h5,prep,sample)
        checks.append(dict(artifact='best_deep.h5',max_absolute_difference=float(np.max(np.abs(p-reference))),passed=bool(np.allclose(p,reference,atol=1e-6)),sample_rows=len(sample)))
    except Exception as e:
        checks.append(dict(artifact='best_deep.h5',max_absolute_difference=np.nan,passed=False,sample_rows=len(sample),unsupported_reason=str(e)))
    proof=pd.DataFrame(checks);proof.to_csv(cfg.tables/'stage9_reload_proof.csv',index=False)
    assert proof[proof.artifact.ne('best_deep.h5')].passed.all(),'A required artifact reload did not reproduce predictions.'
    write_json(cfg.models/'inference_metadata.json',dict(target='preterm <37 completed weeks; positive class 1. Deep sigmoid scores use class-weighted training and are not clinically calibrated risks.',traditional=final['traditional'],deep=final['deep'],thresholds=final,feature_engineering='src/project.py features(); raw ILP_R numeric interval excluded; only 888 history flag allowed',preprocessing='Saved fitted objects; never refit at inference',reporting_flags='Apply guide-backed flag-zero overrides before inference on raw records',python=platform.python_version()))
    files=[]
    for p in sorted(cfg.models.rglob('*')):
        if p.is_file() and p.name!='.gitkeep':files.append(dict(path=str(p.relative_to(cfg.root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(files).to_csv(cfg.tables/'stage9_artifact_manifest.csv',index=False)
    # Freeze the actual modeling environment, including transitive dependencies.
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (cfg.root/'requirements.txt').write_text('# Tested modeling environment: Python '+platform.python_version()+'\n'+freeze)
    (cfg.root/'.python-version').write_text('3.13\n')
    write_json(cfg.tables/'stage9_environment.json',dict(python=platform.python_version(),packages={p:metadata.version(p) for p in ['numpy','pandas','scikit-learn','tensorflow-cpu','keras','shap','pyarrow','nbclient']}))
    facts(cfg)
    finish_readme(cfg)
    import zipfile
    archive=cfg.models/'submission_models.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        candidates=list(cfg.models.glob('best_*'))+[cfg.models/'inference_metadata.json']
        for candidate in candidates:
            paths=candidate.rglob('*') if candidate.is_dir() else [candidate]
            for path in paths:
                if path.is_file():bundle.write(path,path.relative_to(cfg.root))
        for path in list((cfg.root/'src').glob('*.py'))+[cfg.root/'README.md',cfg.root/'requirements.txt',cfg.tables/'stage1_data_dictionary.csv',cfg.tables/'stage1_corrected_variable_layout.csv']:
            bundle.write(path,path.relative_to(cfg.root))
    files=[]
    for path in sorted(cfg.models.rglob('*')):
        if path.is_file() and path.name!='.gitkeep':
            files.append(dict(path=str(path.relative_to(cfg.root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(files).to_csv(cfg.tables/'stage9_artifact_manifest.csv',index=False)
    from PIL import Image
    figures=[]
    for p in sorted(cfg.figures.glob('*.png')):
        with Image.open(p) as im:
            dpi=im.info.get('dpi',(0,0));figures.append(dict(file=p.name,width=im.width,height=im.height,dpi_x=dpi[0],dpi_y=dpi[1]))
    figtable=pd.DataFrame(figures);figtable.to_csv(cfg.tables/'stage9_figure_manifest.csv',index=False)
    assert (figtable.dpi_x>=299).all() and (figtable.dpi_y>=299).all()
    print(proof.to_string(index=False));print('Saved figures:',len(figtable))
    stage_done(cfg,9,dict(reload_checks=checks,figure_count=len(figtable),saved_model_files=len(files),python=platform.python_version()))
    return proof


def facts(cfg):
    raw=json.loads((cfg.data/'stage1_audit/final_summary.json').read_text())
    sample=json.loads((cfg.tables/'stage2_sampling.json').read_text())
    split=pd.read_csv(cfg.tables/'stage4_split_manifest.csv')
    val=pd.read_csv(cfg.tables/'stage7_validation_comparison.csv')
    test=pd.read_csv(cfg.tables/'stage8_final_test_metrics.csv')
    miss=pd.read_csv(cfg.tables/'stage1_missingness_final.csv').sort_values('missing_pct',ascending=False)
    top=pd.read_csv(cfg.tables/'stage7_permutation_importance.csv').head(10)
    lines=['# Verified report facts','',f'- Raw records: {raw["raw_rows"]}',f'- Raw SHA-256: {raw["raw_sha256"]}',f'- Unknown primary targets dropped: {sample["target_code3_dropped"]}',f'- Known targets: {sample["known_target_rows"]}',f'- Sample rows: {sample["sample_rows"]}',f'- Seed: {cfg.seed}',f'- Chunk size: 25000',f'- Sample method: {sample["sampling"]}',f'- Sample class-ratio absolute difference: {sample["ratio_absolute_difference"]}','- Full balance: reports/tables/stage2_class_balance.csv','- Final missingness: reports/tables/stage1_missingness_final.csv','- Hyperparameter search spaces and results: reports/tables/stage5_search_spaces.json, stage5_tuning.csv, stage6_search_space.json, stage6_tuning.csv','- No target gestation used to reconstruct an interval predictor.','- Short interpregnancy interval: unavailable without leakage; not engineered.','']
    for r in split.itertuples():lines.append(f'- {r.split} rows: {r.rows}')
    lines+=['','## Class balance','']
    balance=pd.read_csv(cfg.tables/'stage2_class_balance.csv')
    for r in balance.itertuples():lines.append(f'- {r.population}: rows={r.rows}; preterm={r.preterm}; term={r.term}; preterm proportion={r.preterm_ratio:.8f}')
    lines+=['','## Largest missingness values','']
    for r in miss.head(12).itertuples():lines.append(f'- {r.field}: {r.missing_count} missing; {r.missing_pct:.6f}%')
    lines+=['','## Validation models','']
    for r in val.itertuples():lines.append(f'- {r.model}: AUC={r.roc_auc:.6f}; AP={r.average_precision:.6f}; F1={r.f1:.6f}; recall={r.recall:.6f}; precision={r.precision:.6f}; threshold={r.threshold:.8f}; training seconds={r.training_seconds:.3f}; parameters={r.best_parameters}')
    lines+=['','## Final test','']
    for r in test.itertuples():lines.append(f'- {r.model}: AUC={r.roc_auc:.6f}; 95% bootstrap CI=[{r.auc_ci_lower:.6f}, {r.auc_ci_upper:.6f}]; AP={r.average_precision:.6f}; precision={r.precision:.6f}; recall={r.recall:.6f}; F1={r.f1:.6f}; accuracy={r.accuracy:.6f}; Brier={r.brier:.6f}; threshold={r.threshold:.8f}')
    lines+=['','## Development ablations','']
    ablations=pd.read_csv(cfg.tables/'stage5_ablations.csv')
    for r in ablations.itertuples():lines.append(f'- {r.experiment}: AUC={r.roc_auc:.6f}; AP={r.average_precision:.6f}; delta AP versus core={r.delta_AP_vs_core:+.6f}; F1={r.f1:.6f}; training rows={r.training_rows}; validation rows={r.validation_rows}')
    lines+=['','## Leading permutation drivers','']
    for r in top.itertuples():lines.append(f'- {r.feature}: mean AP drop={r.importance:.8f}; SD={r.std:.8f}')
    lines+=['','## Limitations and submission facts','', '- US births in the public-use file; no state/county analysis.','- Live-birth registry only; not all pregnancies or fetal losses.','- Administrative/self-reported and edited data; underreporting and missing values.','- No clinical biomarkers or ultrasound measurements.','- Retrospective prenatal proxies; no timestamped prospective validation.','- True interpregnancy interval unavailable; birth-date-derived ILP_R and gestation-derived PRECARE excluded.','- Sampling and minority-class imbalance; subgroup disparities.','- Missingness indicators may encode reporting or healthcare-access patterns.','- Structural ILP_R coding is a retrospective history proxy and can differ within a plural delivery.','- No maternal/pregnancy identifiers: sibling/repeated-pregnancy overlap across splits cannot be ruled out.','- Singleton sensitivity uses the same development-only experiment; it does not remove all possible family overlap.','- Bootstrap intervals condition on the chosen sample and assume independent rows; no family-cluster correction is possible.','- Single-year internal evaluation; no external/temporal validation.','- Validation thresholds maximize F1, not an established clinical utility/cost target.','- Class-weighted probability calibration may be imperfect; inspect saved calibration plots.','- No causal or clinical deployment claim.','- Student report, public Drive upload, GitHub publication, and video remain student-owned.','- AI-generated report percentage: not measured or certified.']
    (cfg.root/'reports/FACTS.md').write_text('\n'.join(lines)+'\n')


def finish_readme(cfg):
    test=pd.read_csv(cfg.tables/'stage8_final_test_metrics.csv')
    rows='\n'.join(f'| {r.model} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |' for r in test.itertuples())
    text=f'''# Premature birth prediction — US Natality 2024

University ML summative project. Predict birth before 37 completed weeks using conservative prenatal/history proxies. Positive target: OEGest_R3=1; negative: code 2; unknown code 3 is dropped. Retrospective research only.

## Data and source

[CDC/NCHS natality data directory](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/) · [official guide directory](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/).

Place these unchanged files under capitalized `Data/`:

- `Nat2024PublicUS.c20250512.r20250708.txt`
- `UserGuide2024.pdf`
- `natality2024_variable_layout.csv`

The supplied raw file contains 3,638,436 records. All records have 1,345 bytes excluding CRLF; bytes after the guide's stated 1,330 are spaces. The audit records the exact SHA-256. The original CSV and guide remain unchanged. An additional reporting flag at position 329 is added only to a derived layout. Valid-looking values are masked where their verified reporting flag is zero. The original ZIP and duplicate extracted copy are preserved locally.

Both `Data/` and `data/`, TXT/ZIP/Parquet data and model artifacts are ignored by Git. Derived samples/cache go to `data/processed/`. No automatic data download or raw conversion occurs.

## Environment and execution

Tested modeling environment: Python {platform.python_version()}; exact package versions are in `requirements.txt`. Install Poppler's `pdftotext` separately. For a fresh environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name maternal-risk
```

Open notebooks in VS Code using this environment. All paths and seeds are in one top configuration cell per notebook. Change `PROJECT_ROOT` for another checkout or mounted Google Drive. Stage 1 runs locally on the raw file. Modeling can run in Colab with the processed sample, `src/`, artifacts directory and matching configuration; a separate GPU environment has not been tested. The local final run uses CPU, with bounded CV parallelism and tf.data thread pools.

Run in order:

1. `01_ingestion_audit.ipynb`: exact count, guide checks, chunked parse, approved reporting-flag correction.
2. `02_eda_preprocessing.ipynb`: target, stratified sample, development-only EDA/leakage audit, split and train-only preprocessing.
3. `03_traditional_ml.ipynb`: baselines, LR/RF/HistGradientBoosting/SVM tuning, learning curves and ablations.
4. `04_deep_learning.ipynb`: Sequential MLP and Functional embeddings, tf.data and documented random search.
5. `05_comparison_evaluation.ipynb`: validation explanation/subgroups, frozen candidates, guarded once-only test evaluation.
6. `06_artifacts_audit.ipynb`: export/reload, verified facts and final assignment audit.

Batch execution is supported by `python src/run_notebooks.py NOTEBOOK_NAME ...`. Jupyter needs local sockets. Do not blindly rerun training notebooks after final test evaluation. The test guard returns saved results on a completed rerun; it does not predict again. A failed partial final run must be inspected and recovered from cached predictions, not retuned or reset.

The 300,000-row stratified modeling sample uses seed 42. Train/validation/test sizes are 210,000/45,000/45,000. CV tuning uses 60,000 training rows; SVM uses 6,000. All preprocessing is fitted inside training data/folds. Thresholds maximize validation F1. Compare models by validation average precision; only the chosen traditional and deep model are evaluated on test.

## Final test results

| Model | ROC-AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{rows}

Confidence intervals, confusion matrices, reports and training times: `reports/tables/`. Figures are PNG at 300 dpi in `reports/figures/`. `reports/FACTS.md` contains verified report facts, not a drafted report.

## Leakage and availability decisions

Numeric ILP_R is computed from the current infant's birth date; PRECARE uses the obstetric gestation estimate. Both are excluded as numeric predictors, even though the supplied layout initially marked them KEEP. Only ILP_R's no-prior-pregnancy structural flag is retained. A true short interpregnancy interval cannot be reconstructed safely. Delivery payment and recorded infant sex are reporting-only. Target/gestation fields, reporting flags, birth dates, birth weight and all delivery/newborn/morbidity/congenital fields never enter predictors. WIC, infection flags and time-dependent complications/accumulations are separate diagnostic ablations. Plurality stays in the core; a singleton-only sensitivity analysis is saved.

## Saved artifacts and inference

`models/best_traditional.joblib` contains its fitted preprocessing and classifier. `models/best_deep.keras`, `best_deep_savedmodel/`, `.h5` where supported, and `best_deep_preprocessing.joblib` are accompanied by reload proofs. Run:

```bash
python src/reload_models.py
```

A ready-to-upload bundle is saved as `models/submission_models.zip`; the original raw archive remains `Data/Nat2024us.zip`.

This predicts on training sample rows and compares formats. It never fits or accesses test rows. Preprocessing includes the stateless feature transformations in `src/project.py`; do not feed raw fixed-width codes directly into a fitted model.

## Limits and student submission

US-only single-year data; no geography, biomarkers or ultrasound; administrative/self-reported quality and edited variables; imbalance and subgroup differences; retrospective availability and sampling. Maternal/pregnancy identifiers are unavailable, so related infants or repeated pregnancies across splits cannot be ruled out. Bootstrap intervals assume independent rows. No prospective, external or clinical validation is claimed. Class-weighted scores may be poorly calibrated. Race/ethnicity is a social grouping, not a biological cause.

The student writes the 3,500–5,000-word IEEE report, provides GitHub/Drive/video links on its first page, uploads the raw ZIP and saved models to Drive with link-view access, and records a 5–10-minute video. AI-generated percentage cannot be measured or certified here. See `reports/SUBMISSION_CHECKLIST.md` and `reports/FINAL_AUDIT.md`.

- GitHub: TODO
- Drive: TODO
- Video: TODO

Technical references: [tf.data](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), [Keras model export](https://keras.io/api/models/model_saving_apis/export/), [CDC preterm associations](https://www.cdc.gov/maternal-infant-health/preterm-birth/).
'''
    (cfg.root/'README.md').write_text(text)


def stage10(cfg):
    raw=json.loads((cfg.data/'stage1_audit/final_summary.json').read_text())
    tests=pd.read_csv(cfg.tables/'stage8_final_test_metrics.csv')
    reload=pd.read_csv(cfg.tables/'stage9_reload_proof.csv')
    guard=json.loads((cfg.state/'test_evaluation_guard.json').read_text())
    git=subprocess.run(['git','ls-files','-z'],cwd=cfg.root,capture_output=True,text=True)
    tracked=git.stdout.split('\0') if git.returncode==0 else []
    tracked_data=[p for p in tracked if p.startswith(('Data/','data/','models/')) or p.lower().endswith(('.zip','.parquet','.txt')) and p!='requirements.txt']
    ignored=subprocess.run(['git','check-ignore','Data/Nat2024PublicUS.c20250512.r20250708.txt','Data/Nat2024us.zip','data/processed/modeling_sample.parquet','models/best_traditional.joblib'],cwd=cfg.root,capture_output=True,text=True)
    git_ok=git.returncode==0 and not tracked_data and ignored.returncode==0 and len(ignored.stdout.splitlines())==4
    code_notebooks=list((cfg.root/'notebooks').glob('*.ipynb'))
    import nbformat
    notebook_ok=True;nb_evidence=[]
    for p in code_notebooks:
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        cells=[c for c in n.cells if c.cell_type=='code']
        bad=sum(o.output_type=='error' for c in cells for o in c.get('outputs',[]))
        # The active final audit cell has no persisted output until it finishes.
        if p.name!='06_artifacts_audit.ipynb':notebook_ok &= all(c.execution_count is not None for c in cells) and bad==0
        notebook_ok &= not any('Interpretation placeholder:' in c.source for c in n.cells if c.cell_type=='markdown')
        nb_evidence.append(dict(notebook=p.name,code_cells=len(cells),error_outputs=bad))
    pd.DataFrame(nb_evidence).to_csv(cfg.tables/'stage10_notebook_validation.csv',index=False)
    rules=[
      (1,True,'Real CDC fixed-width public-use file; raw hash and guide verification in notebook 01.'),
      (2,raw['raw_rows']>=20000,f"Executed streaming raw count {raw['raw_rows']}; Stage 1 audit."),
      (3,True,'Binary target mapping and full/sample balance in stage2_class_balance.csv.'),
      (4,(cfg.tables/'stage5_validation_models.csv').is_file(),'Notebook 03; majority/LR/RF/HGB/SVM, CV spaces/results and training-only SVM subsample.'),
      (5,(cfg.models/'dl_sequential.keras').exists() and (cfg.models/'dl_functional.keras').exists(),'Notebook 04; both APIs, embeddings and tf.data operations in src/deep_learning.py.'),
      (6,all((cfg.tables/f).exists() for f in ['stage5_tuning.csv','stage5_ablations.csv','stage5_learning_curves.csv','stage6_tuning.csv','stage8_final_test_metrics.csv']),'Tuning both paradigms, engineering, imbalance, learning curves, confusion/ROC/PR/calibration and bootstrap tables/figures.'),
      (7,notebook_ok,'Notebooks 01–06: technical notes before code and interpretations after outputs; nbformat and execution checks.'),
      (8,git_ok,'README, exact requirements; git ls-files/check-ignore audit; raw/model paths ignored; no raw data tracked.'),
      (9,bool(reload[reload.artifact.ne('best_deep.h5')].passed.all()),'stage9_reload_proof.csv; joblib, .keras, SavedModel and optional .h5 with preprocessing.'),
      (10,False,'Facts/figures/tables are supplied. Student report, authorship/AI-percentage compliance and submission links are not complete or certifiable here.')]
    audit=pd.DataFrame([dict(rule=i,status='PASS' if passed else 'FAIL',evidence=evidence) for i,passed,evidence in rules])
    audit.to_csv(cfg.tables/'stage10_assignment_audit.csv',index=False)
    write_json(cfg.tables/'stage10_git_audit.json',dict(git_repository=git.returncode==0,tracked_raw_paths=tracked_data,ignored_paths=ignored.stdout.splitlines(),raw_not_tracked=git_ok,committed_or_published=False))
    lines=['# Final assignment audit','']+[f'- Rule {r.rule}: **{r.status}** — {r.evidence}' for r in audit.itertuples()]
    lines+=['','- Final-test guard: '+json.dumps(guard),'- No model selection or retuning follows test evaluation.','- Code work is complete; rule 10 remains FAIL until student-authored work/submission is complete.','- Parent task approval gates were waived by the user; stages executed in order.']
    (cfg.root/'reports/FINAL_AUDIT.md').write_text('\n'.join(lines)+'\n')
    (cfg.root/'reports/SUBMISSION_CHECKLIST.md').write_text('''# Student submission checklist

- [ ] Write your own 3,500–5,000-word IEEE-style report using verified FACTS.md, tables and figures.
- [ ] Confirm the university's AI-use policy and authorship requirement; no automated percentage is certified.
- [ ] Put working GitHub, Drive and video links on the report's first page and replace README placeholders.
- [ ] Review code and create/publish the GitHub repository; raw data and large models must remain untracked.
- [ ] Upload the raw ZIP and required saved models/preprocessors to Drive; set Anyone with the link can view.
- [ ] Record a 5–10-minute video: problem/data, leakage decisions, notebooks, comparisons, honest limitations, reload demo.
- [ ] Verify every link in a signed-out browser and package the university-required deliverables.

Report coverage prompts (student-written prose): problem and intended prenatal use; data provenance and codebook; ethical/availability audit; sampling/splits; preprocessing and engineered features; baselines and tuning for both paradigms; validation selection and thresholds; ablations/learning dynamics; explanation/subgroups; frozen test results and uncertainty; reproducibility/deployment limits; limitations and future prospective evaluation; IEEE references. Do not describe unavailable interval features as successfully engineered.
''')
    stage_done(cfg,10,audit.to_dict('records'))
    progress=cfg.root/'PROGRESS.md';text=progress.read_text()
    for r in audit.itertuples():
        if r.status=='PASS':text=text.replace(f'- [ ] Rule {r.rule}:',f'- [x] Rule {r.rule}:')
    text+='\n## Final status\n\nAll computational stages executed. See reports/FINAL_AUDIT.md. Student-authored report and submission tasks remain; no claim that all assignment rules pass.\n'
    progress.write_text(text)
    print(audit.to_string(index=False));print('Tracked raw data:',tracked_data)
    return audit
