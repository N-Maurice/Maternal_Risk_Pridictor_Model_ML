# Local, Kaggle and Colab execution

Both maternal notebooks are independently runnable. Their first code cell contains all paths and switches. Review is the default and never fits a model or repeats test predictions.

## Input folder

Preserve these relative paths in your uploaded project:

```text
maternal-project/
  notebooks/01_data_preparation_and_eda.ipynb
  notebooks/02_models_and_evaluation.ipynb
  src/
  reports/tables/
  Data/UserGuide2024.pdf
  data/processed/modeling_sample.parquet
  data/processed/state/
```

Upload privately. Do not upload `.venv/` or `.git/`. The raw TXT/ZIP and full selected cache are optional for review; retain them locally for auditing and raw ingestion. No `models/revision/` or `reports/revision/` files are required for the main notebooks. Actual saved-model inference additionally needs the original model/preprocessing artifacts and matching package versions.

## VS Code

Install `requirements.txt` in Python 3.13 and choose that kernel. Open either notebook and Run All. Input discovery supports both the project root and the notebooks folder. The preview helper displays real training rows, column types and missingness before modeling.

## Kaggle

Import either notebook and set:

```python
INPUT_ROOT_OVERRIDE = '/kaggle/input/YOUR_DATASET/maternal-project'
OUTPUT_ROOT_OVERRIDE = '/kaggle/working/maternal-risk-results'
```

The notebook can discover a single compatible uploaded project; set the input path explicitly when there are several. Input files stay read-only. Figures and current results go to the writable output folder.

## Colab

In the first code cell set `MOUNT_GOOGLE_DRIVE=True` if you want the optional Google Drive mount. Then set:

```python
INPUT_ROOT_OVERRIDE = '/content/drive/MyDrive/Maternal_Risk_Pridictor_Model'
OUTPUT_ROOT_OVERRIDE = '/content/maternal-risk-results'
```

Alternatively upload the project without mounting Drive and use its uploaded path. Authentication is performed by the student through Colab. Use an output path in Drive if you want outputs to persist.

## Packages and experiment scope

Use the platform's compatible TensorFlow runtime where available. The commented dependency cell provides installs for missing packages. Restart after changing core dependencies. `requirements.txt` is the only dependency file and retains the recorded local version pins for saved-model loading. Hosted runtimes may use different Python/package versions; installing the local pins there has not been verified.

`RUN_TRAINING=True` enables classical tuning; `RUN_DEEP_LEARNING=True` additionally enables both neural APIs. These are new development experiments. Searches use training folds, thresholds use validation, and the previously evaluated test cohort remains closed. Threshold F2 and recall targets are research comparisons, not clinical policy. No connection to a remote Kaggle/Colab account is claimed by local input-path testing.

## Optional notebook 03 — feature challenger

For cached review add reports/tables/feature_challenger/ and models/feature_challenger/validation_predictions.npz to the same read-only input folder. It needs neither raw TXT nor historical extension assets. To reproduce training, set RUN_STUDY=True in notebook 03; output pipelines, caches and tables go only to the separate feature_challenger directories. Default core runner execution remains notebooks 01/02; use --notebook 3 explicitly. The study uses the same saved training/validation split and never evaluates the closed test cohort.
