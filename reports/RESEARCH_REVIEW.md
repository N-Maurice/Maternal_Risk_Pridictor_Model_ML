# Related research and decisions for this project

Reviewed 18 October 2026. This is a targeted comparison, not a systematic review. Studies differ in outcome definition, population, pregnancy landmark and input information. Their published scores are not a leaderboard for our dataset. The revised experiments remain development analyses because the original test set has already been evaluated.

| Original study | What it did well | Limitation or lesson | Action in this project |
|---|---|---|---|
| Meertens et al., 2018: maternal-characteristic models for spontaneous preterm birth | Independent prospective validation, calibration and decision curves, with early predictor availability | Externally evaluated models offered limited discrimination; nulliparous prediction was especially difficult. This does not identify an algorithm that will solve missing information. | Add exploratory decision curves, distinguish probability calibration from ranking, and retain the requirement for an independent future cohort. |
| Hornaday et al., 2025: maternal blood gene-expression prediction | Reassessment, replication and external validation; explicit discussion of leakage | Whole-dataset biomarker preselection contaminated validation. Greater neural-network complexity did not overcome weak generalization. | Keep all fitted transformations and model searches inside training; do not perform outcome-based feature screening across the sample or recycle the original test set. |
| Huang et al., 2024: multiple prenatal visits in nulliparous women | Defined prenatal visit windows, regularized regression and visit-specific evaluation | Later visits and ultrasound inputs supplied information unavailable in this static natality file. Its results are not directly comparable with our mixed-parity all-preterm target. | Keep time-dependent features separate, make the missing prediction landmark explicit, and do not invent ultrasound or biomarker features. |
| Yang et al., published online 2022 / journal issue 2023: reporting and bias assessment | Structured TRIPOD/PROBAST assessment | Identified weaknesses in effective sample size, predictor selection and calibration reporting across reviewed studies. These are aggregate findings, not accusations against every study. | Publish feature definitions, partition provenance, search budgets, calibration and uncertainty; avoid relying on accuracy or a single favorable score. |

## What was already sound

The original study used a genuine large public file, a streaming audit, codebook-specific missingness, an explicit leakage audit, training-only preprocessing/CV, a dummy and logistic baseline, both Keras interfaces, an isolated test evaluation and reloadable artifacts. Those protections are retained.

## Improvements implemented

- A single comprehensive notebook, including an embedded portable source/results bundle, readable inputs, visible transformations and short explanations.
- An explicit raw-to-feature funnel; training-only head and tail previews; numeric/categorical missing-value handling and feature-selection tables.
- Native categorical gradient boosting and Extra Trees experiments; bounded, multi-metric training cross-validation.
- A calibration partition drawn only from original training data. Revised thresholds come from this partition rather than the external development-validation rows.
- Unweighted versus weighted neural training, and sigmoid probability calibration, to investigate the original network's poor Brier score.
- A fixed equal-weight tree/neural ensemble; no validation-optimized blend weights.
- Specificity, negative predictive value, balanced accuracy, MCC, log loss, alert burden, recall-target trade-offs, calibration tables and exploratory decision curves, alongside AUC/AP/F1.
- Separate revised artifacts and development estimates. The original final-test results are preserved and never recalculated using new models.

## Gaps that code cannot repair

There are no reliable per-variable observation times, family identifiers, geographical fields, ultrasound measurements, biomarkers or independent external cohort in the supplied file. All-preterm birth is not the same endpoint as spontaneous preterm birth. More tuning cannot establish prospective clinical utility or remove these information limits. Reusing an already opened test set to advertise a stronger revised model would create another methodological error.

## References

1. L. J. E. Meertens et al., “Prediction models for the risk of spontaneous preterm birth based on maternal characteristics: a systematic review and independent external validation,” *Acta Obstetricia et Gynecologica Scandinavica*, 2018, doi: 10.1111/aogs.13358. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6099449/).
2. K. K. Hornaday et al., “Machine learning for the prediction of spontaneous preterm birth using early second and third trimester maternal blood gene expression: A cautionary tale,” *PLOS ONE*, 2025, doi: 10.1371/journal.pone.0310937. [Original article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310937).
3. C. Huang et al., “Predicting preterm birth using electronic medical records from multiple prenatal visits,” *BMC Pregnancy and Childbirth*, 2024, doi: 10.1186/s12884-024-07049-y. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11662559/).
4. Q. Yang et al., “Reporting and risk of bias of prediction models based on machine learning methods in preterm birth: A systematic review,” *Acta Obstetricia et Gynecologica Scandinavica*, vol. 102, pp. 7–14, 2023; online 2022, doi: 10.1111/aogs.14475. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9780725/).

Sources were read directly; conclusions above are methodological interpretations, not evidence that our revised candidates have external validity. Original published numeric scores are deliberately not used to imply matched-cohort superiority.
