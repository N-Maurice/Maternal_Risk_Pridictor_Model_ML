# Matched 16-feature challenger evaluation

The separate notebook is `notebooks/03_lean_feature_evaluation.ipynb`. It compares the proposed 16 inputs with the existing 32 using the same 210,000 training rows, 45,000 validation rows, fixed model parameters, training-only preprocessing, CV folds and threshold-selection rules. HGB is primary; Logistic Regression is a control. This is a feature ablation, not an additional algorithm search.

The proposal was informed by previously examined validation results. All new results are exploratory; the original test cohort, predictions and thresholds remain unchanged.

| Model | Inputs | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|
| hist_gradient_boosting | 32 | 0.712484 | 0.341159 | 0.081128 |
| hist_gradient_boosting | 16 | 0.710337 | 0.339971 | 0.081188 |
| logistic | 32 | 0.708295 | 0.335889 | 0.081307 |
| logistic | 16 | 0.699671 | 0.332528 | 0.081479 |

The challenger did not outperform the full set on observed AP/AUC in either family. HGB AP difference (16 minus 32) was -0.001188; its conditional paired-bootstrap interval was [-0.002912, 0.000600], including zero. This does not establish superiority or formal equivalence. Logistic AP difference was -0.003361.

At matched 70% HGB validation recall, the full set produced 17,113 false alerts and 453.089 alerts per thousand records. The lean set produced 17,393 false alerts and 459.311 alerts per thousand. Higher lean precision at its own F2 optimum should not be mistaken for improved case finding: its optimized operating point has lower recall.

The lean pipelines fitted faster and produced fewer encoded columns. The full-set refits reproduced the original predictions exactly. All four new pipelines reloaded and agreed on training-example predictions. The experiment jointly removes source and engineered inputs, so it cannot identify which individual removal caused the difference or prove that every original input is necessary. The fixed hyperparameters were originally selected for the full set; no claim is made about an independently retuned lean model, other algorithms or neural networks.

## Evidence

- `reports/tables/feature_challenger/`: CV, score, threshold, uncertainty, learning-curve and reload tables.
- `reports/figures/feature_challenger/`: comparison plots at 300 dpi.
- `models/feature_challenger/`: four new fitted pipelines and aligned validation prediction cache.
- The original primary study remains in notebooks 01/02 and models/best_*.

## Run

```bash
python scripts/run_analysis.py --notebook 3
python scripts/run_analysis.py --notebook 3 --train-feature-study
```

Review is the default. Training executions are saved under experiments/feature_challenger/executed/ so the delivered notebook keeps its review default. On Kaggle/Colab set the notebook input/output paths as usual; review also needs the challenger tables and models/feature_challenger/validation_predictions.npz. Full model weights are required only for new inference/reload checks, not cached review.
