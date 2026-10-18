# Superseded initial Stage 1 audit

The reporting-flag policy below was subsequently approved and applied. Final results are in `tables/stage1_final_summary.json` and `tables/stage1_missingness_final.csv`; the source CSV is preserved. This document retains the initial investigation trail.

# Stage 1 audit — gate pending cleaning clarification

- Raw rows counted by streaming: 3,638,436.
- Parsed rows: 3,638,436; dropped rows: 0.
- Selected fields: 117; all position/name/width checks passed against the local guide.
- Observed out-of-guide codes: 0.
- Chunk size: 25,000.
- Selected-column cache bytes: 54,990,681; parse runtime seconds: 391.049.
- Raw SHA-256: `ff8b58eb90c440198ba3bc8a8c7b446a3714f75fa95fae391bd10b601ede87b2`.
- All records: 1,345 bytes plus CRLF; guide declares 1,330. Every byte beyond 1,330 is a space. Source unchanged.
- No sampling, binary target construction, row removal, fitted preprocessing, or modeling.

## Five-row parsed preview (selected fields)

| MAGER | BMI | ILP_R | WIC | RF_PPTERM | DPLURAL | OEGest_R3 |
| --- | --- | --- | --- | --- | --- | --- |
| 30.0 | 20.8 | 30.0 | N | N | 1.0 | 2.0 |
| 28.0 | 20.4 | 888.0 | N | N | 1.0 | 2.0 |
| 41.0 | 22.7 | NaN | N | Y | 1.0 | 2.0 |
| 29.0 | 26.4 | 49.0 | N | N | 1.0 | 2.0 |
| 28.0 | 21.5 | 11.0 | N | Y | 1.0 | 1.0 |

`ILP_R=888` remains a real category; `999` becomes NaN. Original code counts are retained. `OEGest_R3` is an outcome audit field and not a predictor.

## Missingness from unknown codes and blanks (provisional)

| field | missing_count | missing_pct |
| --- | --- | --- |
| DMAR | 402966 | 11.0753 |
| ILP_R | 385882 | 10.6057 |
| FAGECOMB | 363459 | 9.9894 |
| WTGAIN | 109016 | 2.9962 |
| WTGAIN_REC | 109016 | 2.9962 |
| BMI | 88890 | 2.4431 |
| BMI_R | 88890 | 2.4431 |
| PWgt_R | 77596 | 2.1327 |
| PRECARE | 70222 | 1.93 |
| PRECARE5 | 70222 | 1.93 |
| MEDUC | 69489 | 1.9099 |
| PREVIS | 67725 | 1.8614 |

Full table: `tables/stage1_missingness.csv`. Reporting flag `0` has not yet overridden populated values. Therefore this table is explicitly provisional for cleaning sign-off.

## Gestation consistency

| recode | known_pairs_checked | mismatches |
| --- | --- | --- |
| GESTREC3 | 3633827 | 0 |
| OEGest_R3 | 3633827 | 0 |

## Unexpected reporting-flag combinations

| field | flag_zero_nonmissing_count | code_missing_pct | proposed_missing_pct |
| --- | --- | --- | --- |
| MEDUC | 8852 | 1.9099 | 2.1532 |
| ILP_R | 8450 | 10.6057 | 10.838 |
| BMI | 8907 | 2.4431 | 2.6879 |
| RF_INFTR | 257964 | 0.206 | 7.2959 |
| RF_ARTEC | 257956 | 0.2642 | 7.354 |
| PAY_REC | 9182 | 0.8313 | 1.0837 |

`flag_zero_nonmissing_count` counts additional populated values that would become NaN if non-reporting flags override their contents. Proposed missing percentages are computed as the union of current missing values and flag zero, avoiding double counting. No proposed override has been applied.

Examples: `RF_INFTR` has 258,727 rows with flag zero, including 257,964 nonmissing values. This contradicts a blanket assumption that non-reporting fields are always blank. Do not infer a region or resident population from these flags.

The PDF defines `F_RF_INF_DRG` at position 329 (one-based), but the supplied CSV has no entry for it. It is not a REVIEW field: it is absent. The ingestion module leaves the RF_FEDRG flag mapping empty rather than substitute a different flag or silently add a column.

Recommendation for confirmation: mask values where their verified reporting flag is zero; add the omitted drug-treatment flag to a derived ingestion layout after guide verification, preserving the supplied CSV. Recompute the cleaned cache and missingness within Stage 1. Until resolved, the cache is an audit intermediate and not ready for modeling.

## Other guide/availability notes

- Preserve fertility-treatment X as not applicable, distinct from unknown U; preserve ILP_R=888 and plural codes 000–003.
- Target unknown code 3 becomes NaN; raw code counts preserve the explicit Stage 2 drop audit.
- The guide's BMI_R Obesity I label reads 35.0–34.9. Do not infer engineered bin boundaries from this typo; revisit before Stage 4.
- PRECARE/no-care status, PAY_REC delivery payment, recorded sex, and pregnancy-interval timing require Stage 3 availability review.
- All targets and reporting flags are audit-only columns, excluded from future predictors.

## Reproduction and gate

Notebook: `notebooks/01_ingestion_audit.ipynb`; parsing functions: `src/ingestion.py`.
Supplemental flag audit: `python src/audit_reporting_flags.py --root .`.
Dictionary: `tables/stage1_data_dictionary.csv`; PDF extracts: `STAGE1_GUIDE_CODES.md`.

STOP. Stage 2 has not begun. Clarify the reporting-flag policy and omitted flag before Stage 1 cleaning sign-off, then obtain APPROVED for Stage 2.
