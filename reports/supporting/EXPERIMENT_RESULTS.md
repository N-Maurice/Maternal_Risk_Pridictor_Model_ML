> Supporting historical experiments; excluded from the current main model comparison.

# Development experiments: measured results

Date: 18 October 2026. Original final-test results remain unchanged. These candidates use only original training and development-validation records; they have **no development final-test result**. Existing validation has already influenced development and is not an independent external cohort.

## Selected development candidate

- Candidate: `equal_weight_tree_neural_ensemble`.
- Fixed equal average of calibrated native categorical gradient boosting and calibrated unweighted Functional network scores.
- Development ROC-AUC: 0.713907177; AP: 0.343445704.
- Precision: 0.366503776; recall: 0.352564103; F1: 0.359398824.
- Brier: 0.080958564; log loss: 0.291730989; specificity: 0.929265873; MCC: 0.286715280.
- Threshold: 0.169901429, chosen from the training calibration partition.
- Confusion counts: TN=37468, FP=2852, FN=3030, TP=1650.
- Original HGB validation AP: 0.341158565; observed AP difference: +0.002287139. Different fitting/calibration budgets mean this is not an isolated architecture effect.

## All development candidates

| model | roc_auc | average_precision | precision | recall | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- |
| equal_weight_tree_neural_ensemble | 0.713907 | 0.343446 | 0.366504 | 0.352564 | 0.359399 | 0.080959 |
| native_hist_boosting_sigmoid | 0.712546 | 0.342343 | 0.371422 | 0.349359 | 0.360053 | 0.081038 |
| native_hist_boosting | 0.712546 | 0.342343 | 0.371422 | 0.349359 | 0.360053 | 0.081075 |
| functional_unweighted | 0.711544 | 0.340271 | 0.382266 | 0.331624 | 0.355149 | 0.081173 |
| functional_unweighted_sigmoid | 0.711544 | 0.340271 | 0.382266 | 0.331624 | 0.355149 | 0.081114 |
| functional_weighted | 0.711511 | 0.336799 | 0.374007 | 0.341880 | 0.357223 | 0.197302 |
| functional_weighted_sigmoid | 0.711511 | 0.336799 | 0.374007 | 0.341880 | 0.357223 | 0.081298 |
| sequential_unweighted_sigmoid | 0.709355 | 0.330902 | 0.388482 | 0.328632 | 0.356060 | 0.081498 |
| sequential_unweighted | 0.709355 | 0.330902 | 0.388482 | 0.328632 | 0.356060 | 0.081569 |
| extra_trees | 0.702936 | 0.329490 | 0.385839 | 0.331838 | 0.356806 | 0.081596 |
| extra_trees_sigmoid | 0.702936 | 0.329490 | 0.385839 | 0.331838 | 0.356806 | 0.081571 |

## What improved, and what did not

Native categorical boosting improved AP slightly. Extra Trees did not outperform the original HGB. Unweighted neural training greatly improved probability quality relative to weighted raw scores. Within the matched weighted Functional model, sigmoid calibration changed development Brier from 0.197301716 to 0.081298359; AP was essentially preserved, as expected for monotonic calibration. The selected ensemble's ranking improvement is small and does not remove the substantial missed-case problem.

## Paired uncertainty versus original HGB

| metric | point_difference | ci_low | ci_high | bootstrap_reps |
| --- | --- | --- | --- | --- |
| average_precision | 0.002287 | -0.000075 | 0.004674 | 200 |
| roc_auc | 0.001423 | -0.000430 | 0.003289 | 200 |

These intervals condition on fitted models and existing validation rows, ignore unknown family dependence and do not account for model selection or multiple comparisons. They are exploratory, not confirmatory proof of superiority.

## Recall versus workload for the selected candidate

| requested_calibration_recall | threshold | precision | recall | specificity | alert_fraction |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.105718 | 0.234141 | 0.497650 | 0.811062 | 0.221044 |
| 0.700000 | 0.077614 | 0.161941 | 0.701709 | 0.578497 | 0.450644 |
| 0.800000 | 0.068132 | 0.140668 | 0.788462 | 0.440923 | 0.582933 |
| 0.900000 | 0.056810 | 0.123100 | 0.896368 | 0.258854 | 0.757289 |

Requested recall is fitted on calibration rows; achieved recall is measured on development validation. Higher recall comes with more false alerts and a larger fraction flagged. No clinical utility threshold has been established.

## Data and fitting provenance

- Original training: 210,000; original development validation: 45,000.
- Development tree fitting: 178,500; independent training calibration: 31,500.
- Development neural fitting: 151,725; training-only early stopping: 26,775; calibration: 31,500.
- Tree search: three-fold stratified CV on 60,000 fitting rows, scoring AP/AUC/Brier, selection by AP.
- Native HGB sampled six configurations; Extra Trees sampled four.
- Neural comparisons retain the selected architecture and vary weighting, with training-only early stopping; original notebook retains both API hyperparameter searches.
- Original test predictions during this revision: zero. Original guard checksum verified unchanged.
- Saved components, preprocessing, calibrators and selected-candidate manifest: `models/revision/`.
- Reload proof: `reports/revision/tables/reload_proof.csv`; checks use development examples, never the original test.

## Reading the results honestly

The development development candidate is a promising small improvement, not a clinically validated replacement. Retain the original finalists and frozen test results as the completed assignment's final evaluation. Independent future data are needed before making new final performance claims. The one-file notebook displays both scopes clearly.
