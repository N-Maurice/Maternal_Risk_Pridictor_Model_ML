# Maternal Risk Predictor — Preterm Birth Classification

I built and compared traditional machine-learning (scikit-learn) and deep-learning (TensorFlow/Keras) classifiers that predict whether a pregnancy ends in **preterm birth (before 37 completed weeks)**, using only information available before delivery. The data come from the official **2024 US Natality Public Use File** (CDC/NCHS), with 3,638,436 real birth records.

**Author:** Maurice Nshimyumukiza — African Leadership University, Software Engineering, Intro to Machine Learning

**Links:** [GitHub repository](https://github.com/N-Maurice/Maternal_Risk_Pridictor_Model_ML) · [Google Drive (raw dataset and saved models)](https://drive.google.com/drive/folders/1vB0R_G9nah5cxa6Us7XRh2gRk-uRVlCp?usp=sharing) · [Video presentation](https://drive.google.com/drive/folders/1A98fd_4lTyoTX-I0Uln0CAJ2CJ792L6l?usp=sharing)

## Problem and dataset

About one US birth in ten is preterm, and preterm birth is a major cause of newborn death and disability. Birth certificates record the outcome alongside many fields that are only known at delivery (birth weight, labour events, neonatal care). A useful model must therefore be built on pre-delivery information only. I audited every field for when it becomes known, excluded outcome-derived and late-pregnancy variables, and kept 23 source predictors: parental demographics, obstetric history, pre-pregnancy health, smoking and BMI, infertility treatment and plurality. I then engineered 9 more.

| Item | Value |
|---|---|
| Source | CDC/NCHS 2024 Natality Public Use File (`Nat2024PublicUS.c20250512.r20250708.txt`) and its official user guide |
| Raw records | 3,638,436 (4.9 GB fixed-width text); 4,609 with unknown gestation excluded |
| Target | Binary: preterm (<37 weeks) = 1, term = 0; 10.4% preterm |
| Modelling sample | 300,000 records, stratified; train / validation / test = 210,000 / 45,000 / 45,000 |
| Inputs | 32 features (17 numeric, 15 categorical) → 109 columns after one-hot encoding |

## Results

All models use the same splits, training-only preprocessing and scikit-learn metrics. I selected finalists by validation average precision (AP), froze their thresholds on validation, and evaluated them once on the untouched test set.

| Model | Validation ROC-AUC | Validation AP | Brier |
|---|---:|---:|---:|
| Majority class (baseline) | 0.500 | 0.104 | 0.104 |
| Logistic regression (plain = tuned) | 0.708 | 0.336 | 0.081 |
| Random forest | 0.711 | 0.335 | 0.081 |
| **HistGradientBoosting** (traditional finalist) | 0.712 | **0.341** | **0.081** |
| SVM (RBF, 6,000-row subsample, calibrated) | 0.645 | 0.286 | 0.085 |
| Keras Sequential MLP | 0.713 | 0.334 | 0.201 |
| **Keras Functional with embeddings** (deep finalist) | 0.713 | **0.339** | 0.199 |

**Test set (45,000 births, evaluated once):** HistGradientBoosting reached ROC-AUC 0.706 (95% CI 0.698–0.715) and AP 0.331. The Functional network reached ROC-AUC 0.708 (0.699–0.716) and AP 0.332. The two paradigms rank births equally well, but gradient boosting gives far better-calibrated probabilities (Brier 0.082 vs 0.200) and trains in a fraction of the time.

**Error analysis:** multiple pregnancies (3% of births, 63% preterm) are detected almost perfectly and supply about 60% of correct detections. For singleton pregnancies, recall at the frozen threshold is only 0.16–0.20 (ROC-AUC ≈0.66), and first pregnancies are hardest. Pre-delivery birth-certificate data support modest retrospective risk ranking; this is a research benchmark, not a clinical screening tool.

## Notebooks

Run them in order; each also works from a fresh kernel.

1. [01 — Data Preparation and EDA](01_data_preparation_and_eda.ipynb): raw-file audit, codebook-based cleaning, feature selection and leakage audit, feature engineering, class balance, EDA, stratified splits, training-only preprocessing.
2. [02 — Models and Evaluation](02_models_and_evaluation.ipynb): baselines; tuned Logistic Regression, Random Forest, HistGradientBoosting and SVM; Keras Sequential and Functional models with `tf.data`; learning curves; ablations; model comparison; screening thresholds; permutation importance; subgroup and error analysis; frozen test evaluation; model saving and reload checks; TensorBoard.
3. [03 — Lean Feature Evaluation](03_lean_feature_evaluation.ipynb): an extra ablation comparing a 16-feature set with the 32 core features.

Each notebook has a markdown explanation before its code cells and an interpretation after key outputs.

**Review mode (default)** reloads the predictions saved by my completed study and recomputes all metrics, figures and TensorBoard logs from them, without retraining and without touching the test set again. **Training mode** (`RUN_TRAINING=True`, plus `RUN_DEEP_LEARNING=True` for the networks) retrains every model on the training/validation partitions, with live TensorBoard logging. A training run writes its models, tables and figures to `experiments/core/<run-id>/` and never overwrites the completed study.

## TensorBoard: Keras training curves and scikit-learn metrics together

Keras logs only its own streaming metrics, so I linked scikit-learn's evaluation into TensorBoard ([src/tensorboard_metrics.py](src/tensorboard_metrics.py), [src/live_logging.py](src/live_logging.py)):

- **During neural training**, a Keras callback predicts the validation set after every epoch and writes scikit-learn ROC-AUC, average precision, precision, recall, F1, F2, Brier score and log loss. These go into the same TensorBoard run as Keras's `epoch_loss`, `epoch_roc_auc` and `epoch_pr_auc`, under the `sklearn_validation` sub-run.
- **For every model**, classical and neural: scikit-learn scalars, a PR curve, confusion-matrix and ROC/PR/calibration images, the classification report, a threshold sweep (precision, recall, F1, F2 and alerts per 1,000 at thresholds 0.01–0.99), and HParams, so the **HParams** tab ranks all models by the same metrics.
- **Randomized searches**: one step per candidate. **Learning curves**: step = number of training records. **Frozen test** and **singleton-only** evaluations are written to their own sections.

```bash
source .venv/bin/activate
tensorboard --logdir runs        # then open http://localhost:6006
```

`runs/live/` holds logs from training runs; `runs/review/` holds logs computed from the saved predictions of my completed study (each run's `manifest.json` and `events.jsonl` record exactly what was executed). Useful tabs: **Scalars** (filter `sklearn_` or `epoch_`), **PR Curves**, **Images**, **Text** and **HParams**. Inside notebook 02, set `SHOW_TENSORBOARD=True` to open the dashboard inline.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` lists every dependency with the versions I tested (Python 3.13.15, scikit-learn 1.9.1, TensorFlow 2.21.0, Keras 3.15.1, TensorBoard 2.21.0, SHAP 0.54.1).

**Data.** Raw data are not committed to Git. Download `Nat2024us.zip` and `UserGuide2024.pdf` from the [CDC natality data page](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/), or from my Google Drive folder, and place them in `Data/`. The Drive folder also contains the processed sample and split files needed to run the notebooks without re-parsing the raw file. The 4.9 GB raw text file is kept only inside the zip; the notebooks do not need it, and the optional raw audit needs it extracted first: `unzip Data/Nat2024us.zip -d Data`.

**Kaggle / Colab.** All paths are set in each notebook's first code cell. Set `INPUT_ROOT_OVERRIDE` to the uploaded project folder; outputs go to `/kaggle/working/maternal-risk-results` or `/content/maternal-risk-results`. On Colab, `MOUNT_GOOGLE_DRIVE=True` mounts Drive. Details are in [reports/PORTABILITY.md](reports/PORTABILITY.md).

## Commands

```bash
python scripts/run_analysis.py                                 # review notebooks 01 and 02 in fresh kernels
python scripts/run_analysis.py --notebook 2 --train --deep-learning   # new training run with live TensorBoard logs
python scripts/run_analysis.py --notebook 3 --train-feature-study     # retrain the 16- vs 32-feature study
python -m unittest discover -s tests -v                        # automated checks
python scripts/audit_project.py                                # notebook, data-tracking and report audit
tensorboard --logdir runs
```

## Saved models

| File | Contents |
|---|---|
| `models/best_traditional.joblib` | HistGradientBoosting pipeline, including its fitted preprocessing |
| `models/best_deep.keras`, `models/best_deep.h5`, `models/best_deep_savedmodel/` | Functional Keras network in native Keras, HDF5 and SavedModel formats |
| `models/best_deep_preprocessing.joblib` | Training-fitted numeric scaler and category encoder for the network |
| `models/inference_metadata.json` | Frozen thresholds, target definition and input description |

Reloading each format reproduces the original predictions exactly ([reports/tables/stage9_reload_proof.csv](reports/tables/stage9_reload_proof.csv)). Model files are gitignored and shared through the Google Drive folder.

## Repository layout

| Path | Contents |
|---|---|
| `0*_*.ipynb` (project root) | The three notebooks above |
| `src/` | Reusable code: ingestion, features, models, evaluation, TensorBoard/scikit-learn logging |
| `scripts/` | Notebook runner and project audit |
| `tests/` | Leakage boundaries, training-only preprocessing, frozen-test guard, metrics and logging checks |
| `reports/` | Final report, figures (300 dpi), result tables, verified facts and portability notes |
| `Data/`, `data/`, `models/`, `runs/`, `experiments/` | Raw data, processed data, model files and logs (gitignored) |

## Limitations

The birth-certificate data have no fixed prediction time, no mother identifier (so co-twins and siblings can fall into different partitions), and none of the clinical measurements that predict preterm birth best, such as cervical length or biomarkers. All results come from a single year in a single country. The outcome combines spontaneous and medically indicated preterm births. Threshold choices are illustrations on validation data, not clinical cutoffs. The report discusses these points in detail.
