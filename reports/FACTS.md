# Verified report facts

- Raw records: 3638436
- Raw SHA-256: ff8b58eb90c440198ba3bc8a8c7b446a3714f75fa95fae391bd10b601ede87b2
- Unknown primary targets dropped: 4609
- Known targets: 3633827
- Sample rows: 300000
- Seed: 42
- Chunk size: 25000
- Sample method: Exact per-class random-priority reservoir, uniform without replacement
- Sample class-ratio absolute difference: 1.586890808688568e-06
- Full balance: reports/tables/stage2_class_balance.csv
- Final missingness: reports/tables/stage1_missingness_final.csv
- Hyperparameter search spaces and results: reports/tables/stage5_search_spaces.json, stage5_tuning.csv, stage6_search_space.json, stage6_tuning.csv
- No target gestation used to reconstruct an interval predictor.
- Short interpregnancy interval: unavailable without leakage; not engineered.

- train rows: 210000
- validation rows: 45000
- test_locked rows: 45000

## Class balance

- full_known_target: rows=3633827; preterm=377948; term=3255879; preterm proportion=0.10400825
- stratified_sample: rows=300000; preterm=31202; term=268798; preterm proportion=0.10400667

## Largest missingness values

- DMAR: 402966 missing; 11.075253%
- ILP_R: 394332 missing; 10.837953%
- FAGECOMB: 363459 missing; 9.989430%
- RF_FEDRG: 267569 missing; 7.353956%
- RF_ARTEC: 267569 missing; 7.353956%
- RF_INFTR: 265458 missing; 7.295937%
- WTGAIN: 117918 missing; 3.240898%
- WTGAIN_REC: 117918 missing; 3.240898%
- BMI: 97797 missing; 2.687886%
- BMI_R: 97797 missing; 2.687886%
- PWgt_R: 86553 missing; 2.378852%
- PRECARE5: 79226 missing; 2.177474%

## Validation models

- hist_gradient_boosting: AUC=0.712484; AP=0.341159; F1=0.359717; recall=0.315598; precision=0.418177; threshold=0.20067759; training seconds=24.497; parameters={"model__max_leaf_nodes": 15, "model__learning_rate": 0.05, "model__l2_regularization": 1.0, "model__class_weight": null}
- dl_functional: AUC=0.712922; AP=0.338774; F1=0.359561; recall=0.346154; precision=0.374048; threshold=0.63497347; training seconds=182.521; parameters={"depth": 1, "units": 96, "dropout": 0.3, "learning_rate": 0.0005, "batch_size": 512, "optimizer": "rmsprop", "l2": 0.0001}
- logistic: AUC=0.708295; AP=0.335889; F1=0.358838; recall=0.331197; precision=0.391513; threshold=0.16778695; training seconds=20.122; parameters={"model__class_weight": null, "model__C": 1}
- plain_logistic: AUC=0.708295; AP=0.335889; F1=0.358838; recall=0.331197; precision=0.391513; threshold=0.16778695; training seconds=5.816; parameters={"C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true, "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 700, "n_jobs": null, "penalty": "deprecated", "random_state": 42, "solver": "lbfgs", "tol": 0.0001, "verbose": 0, "warm_start": false}
- random_forest: AUC=0.710725; AP=0.335309; F1=0.362398; recall=0.335256; precision=0.394320; threshold=0.17657479; training seconds=95.349; parameters={"model__min_samples_leaf": 10, "model__max_features": "sqrt", "model__class_weight": null}
- dl_sequential: AUC=0.713200; AP=0.334237; F1=0.360732; recall=0.347436; precision=0.375087; threshold=0.63751614; training seconds=96.704; parameters={"depth": 1, "units": 96, "dropout": 0.3, "learning_rate": 0.0005, "batch_size": 512, "optimizer": "rmsprop", "l2": 0.0001}
- svm: AUC=0.644714; AP=0.285950; F1=0.331809; recall=0.279060; precision=0.409148; threshold=0.12092046; training seconds=11.267; parameters={"model__gamma": "scale", "model__class_weight": null, "model__C": 0.5}
- dummy: AUC=0.500000; AP=0.104000; F1=0.000000; recall=0.000000; precision=0.000000; threshold=0.50000000; training seconds=1.811; parameters={"constant": null, "random_state": null, "strategy": "most_frequent"}

## Final test

- hist_gradient_boosting: AUC=0.706275; 95% bootstrap CI=[0.698363, 0.714657]; AP=0.331385; precision=0.409983; recall=0.303632; F1=0.348883; accuracy=0.882133; Brier=0.081585; threshold=0.20067759
- dl_functional: AUC=0.707848; 95% bootstrap CI=[0.699443, 0.716155]; AP=0.332176; precision=0.368843; recall=0.334402; F1=0.350779; accuracy=0.871267; Brier=0.199597; threshold=0.63497347

## Development ablations

- baseline_features: AUC=0.712881; AP=0.340928; delta AP versus core=-0.000231; F1=0.361459; training rows=210000; validation rows=45000
- engineered_core: AUC=0.712484; AP=0.341159; delta AP versus core=+0.000000; F1=0.359717; training rows=210000; validation rows=45000
- time_dependent_added: AUC=0.804076; AP=0.461132; delta AP versus core=+0.119973; F1=0.432619; training rows=210000; validation rows=45000
- singleton_only: AUC=0.653130; AP=0.177478; delta AP versus core=-0.163681; F1=0.238019; training rows=203568; validation rows=43600
- GESTREC3_sensitivity: AUC=0.683098; AP=0.313878; delta AP versus core=-0.027281; F1=0.329891; training rows=210000; validation rows=45000

## Leading permutation drivers

- DPLURAL: mean AP drop=0.13206864; SD=0.01141153
- prior_preterm: mean AP drop=0.03303753; SD=0.00572334
- PRIORLIVE: mean AP drop=0.01284175; SD=0.00186057
- RF_PHYPE: mean AP drop=0.00987916; SD=0.00114244
- MAGER: mean AP drop=0.00924854; SD=0.00176583
- MRACEHISP: mean AP drop=0.00823280; SD=0.00177896
- M_Ht_In: mean AP drop=0.00754599; SD=0.00258752
- BMI: mean AP drop=0.00634473; SD=0.00294215
- FAGECOMB: mean AP drop=0.00600201; SD=0.00241551
- RF_CESARN: mean AP drop=0.00536570; SD=0.00101345

## Limitations and submission facts

- US births in the public-use file; no state/county analysis.
- Live-birth registry only; not all pregnancies or fetal losses.
- Administrative/self-reported and edited data; underreporting and missing values.
- No clinical biomarkers or ultrasound measurements.
- Retrospective prenatal proxies; no timestamped prospective validation.
- True interpregnancy interval unavailable; birth-date-derived ILP_R and gestation-derived PRECARE excluded.
- Sampling and minority-class imbalance; subgroup disparities.
- Missingness indicators may encode reporting or healthcare-access patterns.
- Structural ILP_R coding is a retrospective history proxy and can differ within a plural delivery.
- No maternal/pregnancy identifiers: sibling/repeated-pregnancy overlap across splits cannot be ruled out.
- Singleton sensitivity uses the same development-only experiment; it does not remove all possible family overlap.
- Bootstrap intervals condition on the chosen sample and assume independent rows; no family-cluster correction is possible.
- Single-year internal evaluation; no external/temporal validation.
- Validation thresholds maximize F1, not an established clinical utility/cost target.
- Class-weighted probability calibration may be imperfect; inspect saved calibration plots.
- No causal or clinical deployment claim.

## Current core scope and validation-only operating points

- Main comparison entries: 8.
- Current tests passed: 21.
- New test model predictions: 0.
- HistGradientBoosting exploratory validation-F2 threshold: 0.080211580; recall: 0.686538462; precision: 0.165048544; alert fraction: 0.432600000.
- Deep functional exploratory validation-F2 threshold: 0.452131867; recall: 0.630982906; precision: 0.182149025; alert fraction: 0.360266667.
- Operating-point evidence: reports/tables/core/validation_threshold_tradeoffs.csv.
- Data notebook: 01_data_preparation_and_eda.ipynb.
- Modeling notebook: 02_models_and_evaluation.ipynb.

## Separate matched feature experiment

- Evaluation notebook: 03_lean_feature_evaluation.ipynb.
- Feature sets: 32 versus 16 model inputs.
- Matched families: HistGradientBoosting and Logistic Regression.
- Primary models and original test predictions: unchanged.
- Results: reports/tables/feature_challenger/validation_scores.csv.
- Conditional paired intervals: reports/tables/feature_challenger/paired_bootstrap_intervals.csv.
- Observed AP: lower for the lean set in both families.
- HGB AP-difference interval: includes zero.
- No fresh test evaluation or independent lean hyperparameter search.
