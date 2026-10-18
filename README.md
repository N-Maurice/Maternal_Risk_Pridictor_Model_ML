# Maternal Risk Predictor — preterm birth

The target is recorded birth before 37 completed weeks. The project does not predict every pregnancy complication or create a separate intermediate-risk diagnosis.

Run these files in order; each also works in a fresh kernel:

1. [Data Preparation and EDA](notebooks/01_data_preparation_and_eda.ipynb): real-data preview, codebook cleaning, selected columns, missingness, features, EDA and training-only preprocessing.
2. [Models and Evaluation](notebooks/02_models_and_evaluation.ipynb): baselines, LR/RF/HGB/SVM, both Keras APIs, tuning, learning curves, threshold analysis and frozen test evidence.

**Separate feature experiment:** [03 — Lean Feature Evaluation](notebooks/03_lean_feature_evaluation.ipynb) compares the proposed 16-feature set with the existing 32 using matched HGB and Logistic Regression models. It is an optional development ablation, not a replacement for the core analysis. [Measured findings](reports/FEATURE_CHALLENGER.md).

```bash
python scripts/run_analysis.py --notebook 3
python scripts/run_analysis.py --notebook 3 --train-feature-study
```

The main comparison has eight entries: dummy, plain LR, tuned LR, RF, HGB, SVM, Sequential and Functional. Plain/tuned LR selected the same configuration. Extra model families and model-blending demonstrations are outside the main workflow. Previous results remain supporting history; they are not current principal candidates.

## Local setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` is the single dependency file; it retains the recorded Python 3.13 environment pins and includes SHAP for model explanations.

Select the environment in VS Code. Keep training switches false for review. Review computes figures/metrics from saved evidence, without training or new test predictions.

## Kaggle and Colab

All paths are in each notebook's first code cell. Set `INPUT_ROOT_OVERRIDE` to the uploaded project folder. Kaggle outputs default to `/kaggle/working/maternal-risk-results`; Colab outputs default to `/content/maternal-risk-results`. Local outputs stay in this project.

On Colab, `MOUNT_GOOGLE_DRIVE=True` enables the optional Drive mount; set the input path to the project in Drive. Kaggle can locate a single uploaded project automatically. With multiple candidates, set the path explicitly. Dependencies and upload structure are in [PORTABILITY.md](reports/PORTABILITY.md). No remote account run is claimed.

The input folder needs the sample, split/prediction state, `src/`, `reports/tables/`, and the official guide. It does not need the large raw TXT for review. Historical development model files are no longer required by the notebooks.

## Execution switches

- `RUN_TRAINING=True`: bounded training-fold searches for all four classical families and development comparisons.
- `RUN_DEEP_LEARNING=True`: also tune/refit Sequential and Functional models; requires training enabled.
- `RUN_RAW_AUDIT=True`: stream the raw count and checksum.
- `RUN_INGESTION=True`: parse verified selected fields in chunks into a new parquet.
- `RECALL_TARGETS=(0.70,0.80)`: illustrative validation policies, not clinical standards.

New model outputs go to `experiments/core/`. Core tables go to `reports/tables/core/`; figures are saved at 300 dpi. The original final-test guard and predictions remain unchanged.

```bash
python scripts/run_analysis.py
python scripts/run_analysis.py --notebook 2
python -m unittest discover -s tests -v
python scripts/audit_project.py
```

## Thresholds and interpretation

The completed HGB and Functional test results retain the original validation-F1 thresholds. The new table `reports/tables/core/validation_threshold_tradeoffs.csv` compares those thresholds with validation-F2 and illustrative recall targets. It shows missed births, false alerts, specificity, NPV and alert volume alongside precision and recall. New thresholds have no independent test evaluation and are not approved clinical cutoffs.

Increasing recall may require many more assessments. No cutoff establishes clinical severity or proves that a mother has a complication. Better identification requires a defined prediction time, appropriate clinical inputs and external prospective evaluation.

## Files to retain

- `Data/`: unchanged raw TXT/ZIP, official UserGuide2024.pdf and variable layout.
- `data/processed/`: audited selected-field cache, sample and protected evaluation state.
- `notebooks/`: the two main maternal notebooks and optional feature evaluation notebook.
- `src/`: supporting reproducibility code and saved-object dependencies.
- `models/`: saved models and preprocessors; historical variants are supporting artifacts.
- `reports/`: report, facts, figures and tables. `reports/revision/` is historical supporting work, not required notebook input.
- `tests/` and `scripts/`: execution and integrity checks.

Raw data, processed data, model weights and experiment outputs are gitignored. The raw filename remains `Nat2024PublicUS.c20250512.r20250708.txt`.

Credentials (`.env` files, private keys and common credential JSON files), local editor settings, caches, logs, prediction arrays and record-level CSV previews are also ignored. Keep actual credentials outside source files; use `.env.example` with placeholder values if configuration needs documenting. The ignored datasets and fitted artifacts must be supplied separately to reproduce the saved analysis after cloning. Executed notebooks retain their existing outputs, including previews of the public source dataset.

## Report and submission

The final report (`reports/REPORT.md`, `reports/REPORT.docx`, and `reports/REPORT.pdf`) is kept locally and temporarily excluded from GitHub. Remove the report rules in `.gitignore` when ready to publish it.

The report contains the complete rationale and measured limitations. It remains an AI-assisted draft requiring student revision and authorship review; no AI-generated percentage is certified.

Student: **[FILL IN]**  
GitHub: **[FILL IN]**  
Drive: **[FILL IN]**  
Video: **[FILL IN]**
