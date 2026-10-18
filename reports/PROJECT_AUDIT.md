# Current core-project audit

The two core notebooks contain the required comparison and relevant prenatal checks. Notebook 03 separately evaluates the requested lean-feature challenger. The project keeps short notebook notes and a separate full report.

| Item | Evidence |
|---|---|
| Main inventory | Eight entries: dummy, plain/tuned LR, RF, HGB, SVM, Sequential and Functional |
| Duplicate configuration | Plain and tuned LR have identical selected parameters; they remain baseline/search evidence |
| Removed main extensions | Extra RF demos, resampling, LightGBM, Extra Trees/native-tree extension, blending, SHAP and dashboard sections |
| Data preview | Real training head/tail, types and missingness before model experiments |
| Input integrity | Sample source-row hashes match the saved train/validation/test manifest |
| Training | All four classical search pipelines and both neural architectures have real-training-subset smoke checks |
| Thresholds | Recorded F1 versus validation-only F2/recall policies; confusion counts and alert volume exported |
| Evaluation scope | New thresholds are exploratory validation results; original test thresholds/predictions remain unchanged |
| Tests | 21 tests pass; output and protected hash checks in current_validation.json and core_scope_provenance.json |
| Hosted execution | Both core notebooks pass with 118 read-only input files unchanged and no extension assets; no remote account run is claimed |
| Report | Updated Markdown, Word and PDF; student revision, final formatting and submission placeholders remain |

Historical model variants remain supporting evidence, not principal candidates. Lowering a threshold improves sensitivity at a false-alert cost and does not improve underlying AUC/AP. The binary target is preterm birth, not every pregnancy complication or an intermediate-risk diagnosis.

The feature challenger trains matched HGB and LR pipelines with 16 versus 32 inputs. The current full-feature predictions are reproduced exactly; no original model or prediction artifact is replaced. Observed AP is lower for the challenger in both families, with the HGB conditional paired interval including zero. The result does not establish that every omitted feature is useful. Notebook 03 passes cached review with read-only inputs and no model weights. Results are in FEATURE_CHALLENGER.md.
