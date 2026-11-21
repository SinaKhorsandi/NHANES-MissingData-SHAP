# SINA / NHANES: Stage 4 MICE extension and Stage 11 reproducibility record

**Status:** PROVISIONAL -- advisor approval pending.  
**Scope:** Four-method extension of the original, three-method Stage 3 NHANES 2017-2018 analysis.  
**Frozen computational source revision:** `2f7bca74b4a1972c6209ed581d700fd1a842b086` (`stage3-revision`).  
**Previously registered code revision:** `4e054df` (Stage 4 MICE and original verification scripts).  
**Record assembled:** 2026-11-21.  
**Privacy:** Keep participant-level data, predictions, SHAP matrices, fitted models, split identifiers, and original raw data outside the public Git repository.

## 1. Scope and interpretation

The target is **self-reported history of having been told one has diabetes**, derived from NHANES `DIQ010` responses 1/2; it is **not** laboratory-confirmed diabetes, incident diabetes, or clinical risk prediction. The 2017-2018 analytic cohort comprises 5,098 examined adults, of whom 4,031 have complete values for the 11 predictors. The fixed 25 repeated outer splits, the shared complete-case test participants in each split, and the 100-person SHAP background are inherited from the historically verified Stage 3 analysis.

The four methods are complete-case analysis (CCA), Median/Mode imputation, KNN imputation (`k=5`), and MICE. The prediction models are logistic regression (LR) and random forest (RF). A shared test cohort supports within-split comparisons; evaluation on complete-case test participants does not establish performance on people with missing predictors.

**Stage 11 was an independent *rerun* of Stage 4's MICE and SHAP computations, not a fully independent external implementation or a new validation cohort.** Downstream stages were recalculated from the rerun. The original Stage 3 source analysis was treated as a verified historical reference and was **not re-executed from raw data** during Stage 11.

## 2. Frozen code, environment, and data handling

The source snapshot is identified by the full Git commit above. Six Stage 11 replay scripts were committed at `2f7bca7`; the Stage 4 pipeline and verification scripts were committed at `4e054df`.

Recorded execution environment:

| Component | Recorded version |
| --- | --- |
| OS | Linux, kernel `6.14.0-37-generic` |
| Python | `3.13.3` |
| numpy | `2.5.4` |
| pandas | `3.1.1` |
| scipy | `1.18.1` |
| scikit-learn | `1.9.1` |
| shap | `0.54.1` |
| matplotlib | `3.11.2` |
| joblib | `1.6.0` |
| Pillow | `13.0.0` |

The repository has a pinned `requirements.txt`. On 2026-11-21, the active Python environment was checked against **all 21/21 pinned requirements** with no mismatches; `python -m pip check` reported **no broken requirements**. This confirms consistency of the recorded local environment, not portability to other operating systems or Python versions. Keep private environment dumps and filesystem paths in ignored local output directories.

The MICE specification in `revision_protocol.md` is: `IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10)`, with five imputations per outer split. Its per-draw seed is `2026 + 100 * split_id + draw_id`, for draw IDs 1-5. LR and RF are trained per completed imputation, and the five positive-class probabilities are averaged for evaluation. SHAP contributions for the five completed-data models are aligned and averaged for the same test individuals and common background. The correct RF permutation-SHAP seed is `2026 + split_id`; the corrected Stage 6 output is canonical. SHAP values for LR are on the **log-odds** scale, while RF values are on the **positive-class probability** scale. Do not compare absolute SHAP MSE magnitudes across these model scales.

With `sample_posterior=True`, a reported MICE `n_iter_=10` and zero early stops are **not evidence of convergence**.

## 3. Pinned original and independent-rerun directories

All paths below are relative to the project root; directories under `output/` are intentionally excluded from Git. Preserve the original outputs as read-only references. The directory names identify the audited runs, not portable default locations.

| Stage | Canonical original/reference directory | Stage 11 replay directory |
| --- | --- | --- |
| 3 | `output/stage3_final` | Not rerun in Stage 11 |
| 5 (MICE/model fitting) | `output/stage4_mice/local_rechecks/stage5_cv25_20261116T064635Z_2203846` | `output/stage4_mice/local_rechecks/stage11_replay_stage5_20261117T151930Z_9784` |
| 6 (SHAP) | `output/stage4_mice/local_rechecks/stage6_seedfix_20261117T050734Z_9784` | `output/stage4_mice/local_rechecks/stage11_replay_shap_20261119T045429Z_9784` |
| 7 (SHAP stability) | `output/stage4_mice/local_rechecks/stage7_stability_20261117T132020Z_9784` | `output/stage4_mice/local_rechecks/stage11_replay_step7_20261121T051829Z_9784` |
| 8 (MICE diagnostics) | `output/stage4_mice/local_rechecks/stage8_diagnostics_20261117T134348Z_9784` | `output/stage4_mice/local_rechecks/stage11_replay_step8_20261121T052236Z_9784` |
| 9 (inference) | `output/stage4_mice/local_rechecks/stage9_inference_20261117T140157Z_9784` | `output/stage4_mice/local_rechecks/stage11_replay_step9_20261121T052808Z_9784` |
| 10 (tables and figures) | `output/stage4_mice/local_rechecks/stage10_reporting_20261117T140157Z_9784` | `output/stage4_mice/local_rechecks/stage11_replay_step10_20261121T053359Z_9784` |

The superseded **uncorrected** Stage 6 run must remain separately retained for audit; it is **not** a canonical scientific source. Avoid recomputing it when reproducing the corrected Stage 6.

## 4. Reproducibility checks and independent audit evidence

| Control | Evidence | Result |
| --- | --- | --- |
| Original/replay Stage 5 checkpoint integrity | 25 + 25 manifests covering 900 referenced files | PASS in local verifier |
| Fitted MICE artifacts | 375 / 375 original/replay model and imputer artifacts have identical SHA-256 | PASS in local verifier |
| MICE performance | 50 / 50 LR/RF split-level records agree in scientific fields | PASS |
| MICE diagnostics | 125 / 125 draws agree in all 20 scientific fields; runtime excluded | PASS |
| MICE inner tuning | Selected parameters agree for all split/model combinations | PASS, four small numeric cells differ in two losing RF candidates |
| Fresh SHAP replay | 125 LR + 125 RF draw-level SHAP computations | PASS in local logs |
| SHAP independent aggregate review | Global importance 550 rows and pooled results 50 rows byte-identical to the corrected reference | PASS |
| SHAP private comparison | Local replay reports zero difference in 25 / 25 splits | PASS according to local verifier; private matrices not redistributed |
| Stage 7 stability | 300 paired rows recomputed; original 150 historical comparisons preserved | PASS, local maximum difference zero |
| Stage 8 diagnostics | 3 tables, 16 summary rows and summary JSON agree bytewise | PASS, local verifier |
| Stage 9 corrected inference | 48 rows / 456 numeric cells recalculated; scientific CSV identical to reference | PASS, local verifier |
| Stage 10 reporting | 9 scientific CSV tables (255 rows) byte-identical | PASS, independently audited archive |
| Stage 10 figures | 4 PNG byte-identical; 4 PDFs visually/textually identical, with different file metadata | PASS, independently audited archive |

**Independent aggregate-audit scope:** Selected original/replay aggregate archives, manifests, statistical tables, and rendered figures were checked outside the local project. Full participant-level data, saved models, and private SHAP matrices were not exported in these review archives; their integrity and comparison are evidenced by local verifier logs. This distinction is necessary when claiming independent audit.

### Documented non-exact or historical components

- Stage 5 inner tuning: four numeric cells differ between original and replay in *nonwinning* RF candidate 7 at splits 1 and 20. The largest absolute difference is approximately `1.09e-6`; the selected hyperparameters, 375 fitted artifacts, and final AUC/Brier results agree.
- Timings in Stage 5/6 per-split tables differ between runs as expected; scientific values match.
- Four PDF hashes differ because of document metadata (including creation timestamps); extracted text and renderings match pixel-for-pixel at 72 and 144 DPI. The four PNGs are byte-identical.
- Stage 10 Tables 8 and 9 reuse the original complete-case collinearity analyses; they are **historical references**, not new MICE collinearity estimates.
- The Stage 10 context table has one documented missing descriptive SD in a complete-case summary. It is not evidence of a missing predictive metric.
- Stage 3, including the NHANES raw-data preparation and all original three-arm controls, was **not rerun from scratch** in Stage 11. Stage 11 is therefore **not** a claim of full end-to-end rerun from the CDC source files.

## 5. Research results, qualifications, and statistical limitations

Across the 25 outer splits, the MICE pooled means are:

| Model | Mean ROC-AUC | Mean Brier score |
| --- | ---: | ---: |
| LR | `0.8029547704` | `0.1116765888` |
| RF | `0.7968548195` | `0.1134458325` |

Stage 9 applied a Nadeau-Bengio corrected standard error and a t reference distribution with **24 degrees of freedom** for aligned 25-split comparisons. Holm adjustment was applied **separately** within the 12 AUC tests and the 12 Brier tests. Two adjusted tests passed the stated significance threshold: for LR, Median/Mode minus KNN `k=5` in AUC (difference `-0.0022078111`, Holm p `0.022774`) and Brier (difference `+0.0003915586`, Holm p `0.029736`). **No Holm-significant comparison involved MICE.** These are conditional, approximate repeated-CV inferences, not proof of clinical or population-level superiority.

The corrected 95% intervals are approximate because repeated-CV splits overlap. The displayed per-comparison intervals are not simultaneous Holm-adjusted confidence intervals. Corrected descriptive intervals for Spearman or Jaccard@5 have no inferential p-values; a t approximation can place bounds outside the natural `[0, 1]` scale. Do not interpret those bounds as valid attainable similarity values. The original Stage 3 sensitivity checks do not imply a comprehensive weighted NHANES complex-survey design analysis or external validation.

**No clinical claim:** These experiments study prediction of a self-reported outcome and explanation stability under missing-data handling choices. They do not establish prospective risk prediction or medical utility.

## 6. Protocol status and release gates

The committed `revision_protocol.md` stipulated **advisor approval before running Stage 4**. The computational runs described here were performed while advisor approval was **pending**. Accordingly, the scientific outputs and decisions remain explicitly **PROVISIONAL**. The preapproval requirement is a protocol/process deviation that must be disclosed to, and resolved by, the advisor before describing the study as approved or finalized; preserve `revision_protocol.md` unchanged as the preregistered reference.

Release checks (completed and pending):

1. **PASS (2026-11-21):** All 21/21 pinned versions in tracked `requirements.txt` matched the active environment, and `python -m pip check` passed. The exact version list is preserved in `requirements.txt` and the environment summary above; do not publish credentials or private absolute paths.
2. Review and commit this reproducibility note plus a minimal Stage 4 link in `README.md`; retain `README_FA.md` as historical Stage 1 documentation.
3. Generate public SHA-256 checksums only for selected **tracked distributable source/document files**, excluding the checksum file itself; independently run `sha256sum -c` after the last content changes.
4. Maintain a **separate local/private** output manifest for ignored `output/` artifacts and local integrity checks. Never commit private participant identifiers, split manifests, fitted models, SHAP matrices, full filesystem paths, or raw NHANES files.
5. Record the final documentation commit and require clean `git status`. The precise documentation commit cannot be recorded inside the same commit without creating a self-reference; use Git history for it.
6. Obtain explicit advisor review of the analysis scope, MICE strategy, corrected inference, protocol deviation, and remaining scientific limitations before final paper claims.

For a local verification that does not alter results, use the existing Stage 11 scripts in `--mode preflight` and the pinned original/replay paths above. Do not use `--mode run` on already populated reference directories, and do not bypass the original runtime/approval gates for long jobs. See the original scripts' `--help` and `revision_protocol.md` for required arguments and restrictions.

**Release status at this document revision: COMPUTATIONAL REPLAY PASS; ENVIRONMENT MATCH PASS (21/21; pip check); DOCUMENTATION/INTEGRITY RELEASE GATES PENDING; ADVISOR APPROVAL PENDING.**
