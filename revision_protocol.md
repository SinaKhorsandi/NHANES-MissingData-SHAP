# Revision Protocol: MICE and Corrected Confidence Intervals

**Status:** For advisor approval; **no runs before approval**. **Branch:** `stage3-revision`, created from `stage3-followup` at `9265dd9`; leave `stage3-final` and `stage3-followup` unchanged. **Output:** `output/stage4_mice/`; publish aggregate CSVs only, never participant-level data. Use the exact Stage 3 `split_manifest_final.csv` (25 outer splits; base seed 2026).

## 1. Multiple imputation (MICE)

- **Estimator:** scikit-learn `IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10)`.
- **Imputations:** `m = 5`; for split `s` and imputation `j = 1,...,5`, use `random_state = 2026 + 100*s + j`.
- **MICE targets:** missing numerical `BP_SYS_MEAN`, `BP_DIA_MEAN`, `BMXBMI`, `BMXWAIST`, `INDFMPIR`; use all remaining predictors (including one-hot columns) as auxiliary variables. Impute the categorical `DMDEDUC2` (11 missing) and `PAQ605` (6 missing) by training-set mode, as previously.
- **Leakage control:** fit imputer, encoding and preprocessing on training data only (including within inner CV). The common complete-case test sample requires no imputation. Keep the Stage 3 shared test samples and 100-person SHAP background.

## 2. Pooling and original-model controls

- Train a separate LR and RF for each of the five imputed datasets. Choose hyperparameters with the **original Stage 3 grid and inner CV on imputation 1**, then hold them fixed for imputations 2-5.
- Average the **five positive-class probabilities** on the same test subjects; compute AUC/Brier from that average. Compute SHAP for each fitted model and average the **five attribution matrices** for identical test subjects and identical 100-person background; keep Stage 3 scales (LR log-odds; RF positive-class probability).
- Also rerun the original CCA, Median/Mode and KNN k=5 arms for LR/RF, unchanged. Split-1 AUC/Brier must match `cv_metrics_final.csv` to six decimals; original three-strategy results must stay identical.

## 3. Nadeau-Bengio corrected intervals

- For each pair of four strategies (6 pairs x 2 models), compute split-aligned `d_i`, mean `d_bar`, and sample variance `s^2` over `J=25` splits.
- `sigma_corr^2 = (1/J + n_test/n_train) * s^2`; `n_test` = mean shared test size (~806.2), `n_train` = mean training size of the **smaller strategy** in each pair (conservative).
- With **24 df**, report `95% CI = d_bar +/- t(0.975,24)*sigma_corr`, `t_stat=d_bar/sigma_corr` and two-sided raw p. Apply **Holm** within separate AUC and Brier families, **12 pairwise tests per family** (6 pairs x 2 models).
- For mean Spearman and Jaccard@5, report **descriptive corrected 95% CIs without p-values**. Include `sigma_corr`, `n_train`, and `n_test` columns.

## 4. Execution gates

Before a long run, confirm no data leakage; run **split 1 only**, with both models, all five imputations and SHAP, and estimate the 25-split duration. If **>=12 hours**, stop and ask the advisor before changing `m` or SHAP scope. After approval and a passing benchmark, run 25 splits, record each imputer's `n_iter_`, verify no NaNs and SHAP local additivity (`error < 1e-8`), generate four-strategy tables/figures, and independently rerun and checksum all outputs. **Commit this protocol and submit it for approval before any Stage 4 computation.**
