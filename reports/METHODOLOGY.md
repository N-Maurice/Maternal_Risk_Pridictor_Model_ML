# Methodology

I framed the task as binary prediction of recorded birth before 37 completed weeks. I used the obstetric gestation recode as the target, counted unknown outcomes and kept every source field traceable to the official CDC guide.

I streamed the raw file, checked record alignment, parsed selected fields and validated codes. I converted field-specific unknown values and nonreporting flags to missing. I preserved real zero and structural not-applicable categories. I selected prenatal-history inputs before modeling, excluding outcome proxies and later-pregnancy information from the core analysis.

I created a stratified sample and an immutable training/validation/test split. I inspected only development rows during EDA. I engineered age indicators/bands, BMI categories, prior-preterm history, a missing-aware risk composite and selected interactions. I did not reconstruct a pregnancy interval from outcome gestation.

I fitted numeric imputation/scaling and categorical encoders only on training data. I used stratified training cross-validation for classical tuning, compared a majority dummy and plain logistic baseline with tree ensembles and a subsampled SVM, and developed both Sequential and Functional Keras models using tf.data. I recorded the actual search budgets and retained unsuccessful experiments.

I selected the original finalists using validation average precision and selected their thresholds by validation F1. I generated final test predictions once and saved them. I computed subsequent plots and uncertainty summaries from the cached predictions; I did not reopen that cohort for new model selection.

I keep the main comparison limited to the required algorithms and two Keras APIs. Additional model families and ensemble experiments remain supporting history, not principal candidates. I compare recorded F1 thresholds with validation-only F2 and illustrative recall-target policies, reporting detection, missed births, false alerts and workload. These new policies are post-hoc exploration and have no independent test result.

I report ranking, decision and probability metrics separately. I interpret higher recall alongside false alerts and workload. I use subgroup checks and predictive explanations without claiming causal risk effects. I save preprocessing with each model and verify artifact reloads. The lack of timestamps, family identifiers, geography, biomarkers, external validation and prospective clinical outcomes remains explicit.

I present preparation/EDA in notebooks/01_data_preparation_and_eda.ipynb and modeling/evaluation in notebooks/02_models_and_evaluation.ipynb. Both load the same immutable sample and split independently. Computational code is preserved; lengthy discussion belongs in REPORT.md. The source-hash split manifest and notebook_report_map.csv document the reorganization.
