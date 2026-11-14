# NHANES Missing-Data & SHAP Stability

**How do missing-data handling strategies affect the stability of SHAP explanations when predicting self-reported history of a diabetes diagnosis?**

This repository contains Python code for a reproducible analysis of **NHANES 2017–2018**. It compares three missing-data handling strategies across two classification models and evaluates both predictive performance and the agreement of their SHAP explanations.

> **Outcome definition:** The binary target `diabetes_reported` is derived from the survey question `DIQ010` (`1 = yes`, `2 = no`). It represents **a participant's self-reported history of being told they have diabetes**, not laboratory-confirmed diabetes, a clinical diagnosis established by this study, or prospective diabetes risk.

## Research questions

1. How do complete-case analysis, median/mode imputation, and KNN-based imputation affect ROC-AUC and Brier score?
2. How much do global feature rankings, top-five feature sets, and individual SHAP attributions agree across missing-data strategies?
3. Are the findings sensitive to training-set size, the number of permutation-SHAP cycles, or NHANES MEC examination weights?

## Study population and variables

The analytic cohort contains **5,098 participants**, restricted to adults aged **20 years or older** who were examined at an NHANES mobile examination center (`RIDSTATR = 2`) and had `DIQ010` coded as `1` or `2`. Of these, **4,031** have observed values for all 11 predictors; **1,067** have at least one missing predictor. The constructed mean systolic and diastolic blood pressure variables each have **258 missing observations**; means use at least one valid recorded reading.

The predictors are:

| NHANES/derived field | Description |
| --- | --- |
| `RIDAGEYR` | Age in years |
| `RIAGENDR` | Sex variable recorded by NHANES |
| `RIDRETH3` | Race/Hispanic-origin category |
| `DMDEDUC2` | Adult education level |
| `INDFMPIR` | Family income-to-poverty ratio |
| `BMXBMI` | Body mass index |
| `BMXWAIST` | Waist circumference |
| `BP_SYS_MEAN` | Mean valid systolic blood pressure reading |
| `BP_DIA_MEAN` | Mean valid diastolic blood pressure reading |
| `SMQ020` | Ever smoked at least 100 cigarettes |
| `PAQ605` | Vigorous-intensity work activity |

**Data provenance:** NHANES public-use 2017–2018 source files include `DEMO_J`, `DIQ_J`, `BMX_J`, `BPX_J`, `SMQ_J`, and `PAQ_J`. Stage 3 requires the previously prepared dataset `output/nhanes_examined_clean_with_missing.csv` and `output/preparation_notes.txt` from the Stage 1 preparation workflow. Raw source files and analytic participant-level outputs are deliberately excluded from Git.

## Study design

### Missing-data strategies

- **CCA (complete-case analysis):** fit using only complete participants in the training portion of each fold. CCA is a comparator, **not ground truth**.
- **Median/Mode:** median imputation for numeric predictors and most-frequent-category imputation for categorical predictors.
- **KNN (`k = 5`):** numeric predictors are standardized for neighbor-distance calculation, then imputed using five neighbors. Categorical predictors are imputed using their most frequent training-fold category (not KNN).

Imputation, feature scaling, and one-hot encoding are fitted **within the relevant training fold**, avoiding test-to-train information leakage.

### Classification and evaluation

- **Models:** logistic regression (**LR**) and random forest (**RF**, 500 trees).
- **Outer validation:** stratified 5-fold cross-validation repeated 5 times (**25 outer splits**; base seed `2026`).
- **Hyperparameter selection:** 3-fold inner stratified cross-validation, optimizing ROC-AUC.
- **Fair test comparison:** for each outer split, all six model/strategy combinations are evaluated on **the same complete-case participants in the outer test fold**. The mean common test-set size is **806.2** participants per split.
- **Primary predictive metrics:** ROC-AUC (higher is better) and Brier score (lower is better).
- **Primary analysis:** unweighted; NHANES examination weights are considered separately in a sensitivity analysis.

### SHAP explanations and stability

- **LR:** `shap.LinearExplainer`, expressed on the **log-odds** scale.
- **RF:** `shap.PermutationExplainer`, expressed on the **positive-class probability** scale; **5 permutation cycles** in the primary analysis.
- **Shared background:** 100 complete outer-training participants per split, shared across missing-data methods and selected using seed `2026 + split_id`.
- **Feature alignment:** signed SHAP contributions from one-hot encoded columns are **summed before** returning to the 11 original predictors.
- **Quality control:** all **150 model/strategy/split SHAP runs** passed the local additivity audit (`maximum absolute error < 1e-8`).
- **Pairwise agreement:** compare CCA vs Median/Mode, CCA vs KNN, and Median/Mode vs KNN, separately for LR and RF within each split (**150 model-specific pairwise comparisons**).

Stability measures are **Spearman correlation of global feature-importance ranks**, **Jaccard similarity of the top-five features**, and **MSE-SHAP** between aligned, participant-level attributions. MAE-SHAP is also recorded. **Do not directly compare MSE-SHAP values between LR and RF**: the explainers operate on different output scales.

## Main results

### Predictive performance (25 outer splits; mean ± SD)

| Model | Missing-data strategy | ROC-AUC | Brier score |
| --- | --- | ---: | ---: |
| LR | CCA | 0.8027 ± 0.0141 | 0.1121 ± 0.0048 |
| LR | Median/Mode | 0.7998 ± 0.0142 | 0.1123 ± 0.0046 |
| LR | KNN (k=5) | 0.8020 ± 0.0138 | 0.1119 ± 0.0046 |
| RF | CCA | 0.7960 ± 0.0137 | 0.1132 ± 0.0045 |
| RF | Median/Mode | 0.7955 ± 0.0146 | 0.1137 ± 0.0046 |
| RF | KNN (k=5) | 0.7963 ± 0.0154 | 0.1136 ± 0.0046 |

### Global SHAP agreement (25 splits; mean)

| Model | Pair of strategies | Spearman | Top-5 Jaccard | MSE-SHAP (within-model scale) |
| --- | --- | ---: | ---: | ---: |
| LR | CCA vs Median/Mode | 0.9076 | 0.7067 | 0.04439 |
| LR | CCA vs KNN | 0.8160 | 0.6686 | 0.01362 |
| LR | Median/Mode vs KNN | 0.8785 | 0.7371 | 0.01057 |
| RF | CCA vs Median/Mode | 0.9258 | 0.7467 | 0.00004526 |
| RF | CCA vs KNN | 0.9345 | 0.7867 | 0.00003135 |
| RF | Median/Mode vs KNN | 0.9695 | 0.9200 | 0.00002297 |

**Interpretation:** Predictive performance is similar across the three missing-data strategies, but agreement between SHAP explanations is not perfect. Predictive similarity must not be interpreted as explanation equivalence. The repeated-CV results are descriptive; the 25 overlapping splits are **not 25 independent samples**.

### Sensitivity analyses

1. **Training-set size matching:** imputation-model training sets were subsampled to the size of the corresponding CCA training set in each split (mean **3,224.8**). Prediction gaps did **not consistently shrink**. This check matches sample size, not participant composition or missingness patterns, so it does not establish a causal effect of imputation alone.
2. **RF SHAP permutation cycles:** increasing from **5 to 10 cycles on split 1** left all 11 global feature ranks and all top-five sets unchanged for each strategy (Spearman rank correlation and top-five Jaccard both **1.0**); individual attributions differed slightly. This does not prove convergence on all 25 splits.
3. **MEC examination weights:** `WTMEC2YR` from `DEMO_J` was applied via classifier `sample_weight` in all 25 splits, with training weights normalized to mean 1. Mean ROC-AUC decreased and Brier score increased slightly relative to the unweighted primary models on a like-for-like test-evaluation basis. Preprocessing was not weighted; this is **not** full NHANES survey-design inference (strata/PSUs were not modeled).

## Installation

The analysis was developed with **Python 3.13** and pinned dependencies in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the following commands from the **repository root**. Before starting Stage 3, prepare the Stage 1 inputs at:

```text
output/nhanes_examined_clean_with_missing.csv
output/preparation_notes.txt
```

The preparation notes must include the accepted Stage 3 blood-pressure rule. The raw NHANES files may be stored under `raw_xpt/` for the separate data-preparation workflow. The weight-sensitivity script specifically expects `raw_xpt/DEMO_J.XPT`.

## Reproducing the analysis

The scripts protect existing outputs; **use a fresh output directory** rather than rerunning into a nonempty one. For a first run, test one outer split before starting the full analysis.

```bash
# Optional smoke test (does not overwrite the final run)
python stage3_cv.py --limit-splits 1 --out output/stage3_cv_smoke

# 1. Primary repeated nested cross-validation
python stage3_cv.py --out output/stage3_final

# 2. SHAP attributions for the 25 outer splits
python stage3_shap.py --cv-out output/stage3_final

# 3. Pairwise SHAP stability metrics
python stage3_stability.py --shap-dir output/stage3_final/shap

# 4. Training-set size sensitivity
python stage3_sensitivity_sample_size.py \
  --cv-out output/stage3_final \
  --out output/stage3_sensitivity/sample_size_matched_full

# 5. RF permutation-cycle sensitivity, split 1
python stage3_sensitivity_shap_cycles.py \
  --cv-out output/stage3_final \
  --out output/stage3_sensitivity/shap_cycles_split01

# 6. Weighted classifier-fitting sensitivity
python stage3_sensitivity_weights.py \
  --cv-out output/stage3_final \
  --out output/stage3_sensitivity/weights_full_v1

# 7. Independent audit of the saved weight-sensitivity results
python audit_stage3_weight_sensitivity.py

# 8. Manuscript tables and original figures
python stage3_reporting.py \
  --cv-out output/stage3_final \
  --out output/stage3_reporting_v1

# 9. Improved, readable rank-bump figures (separate LR and RF plots)
python stage3_figure2_clean.py \
  --importance output/stage3_final/shap/shap_global_importance_final.csv \
  --out output/stage3_reporting_v2_figures
```

`audit_stage3_weight_sensitivity.py` uses the saved full-run directory `output/stage3_sensitivity/weights_full_v1`. If you choose a different directory for the weighted run, adjust that path in the audit script.

### Important saved outputs

| Output | Purpose |
| --- | --- |
| `output/stage3_final/cv_metrics_final.csv` | ROC-AUC and Brier score for 150 model/strategy/split runs |
| `output/stage3_final/test_predictions_final.csv` | Predictions on common complete-case test participants |
| `output/stage3_final/split_manifest_final.csv` | Participant assignments to outer train/test splits |
| `output/stage3_final/tuning_results_final.csv` | Inner-CV hyperparameter search |
| `output/stage3_final/shap/additivity_audit_final.csv` | SHAP local additivity checks |
| `output/stage3_final/shap/shap_global_importance_final.csv` | Global importance and ranks of the 11 original predictors |
| `output/stage3_final/shap/shap_stability_metrics_by_split_final.csv` | 150 pairwise SHAP-stability results |
| `output/stage3_reporting_v1/table1_predictive_performance.csv` | Manuscript performance summary |
| `output/stage3_reporting_v1/table2_shap_stability.csv` | Manuscript SHAP-stability summary |
| `output/stage3_reporting_v2_figures/` | Revised rank-bump figures |

## Reproducibility and data availability

- **Primary random seed:** `2026` (background and permutation seeds depend on split ID).
- **Input SHA-256:** `a017e04e8c9cbbf5d1d5a85e7bbc241629cc8679bdd21b1a3f96cbf0a2338752` for the analyzed Stage 1 CSV.
- **Locally verified Git revision for Stage 3 scripts:** `4032bdb` on branch `stage3-final`.
- **Git exclusions:** `.venv/`, `raw_xpt/`, and `output/` are excluded from version control.
- The repo contains code and analysis instructions. The preprocessed Stage 1 CSV, source XPT files, fitted models, and participant-level analysis outputs are **not bundled**; obtain the public NHANES source files from the CDC and prepare the Stage 1 inputs before rerunning.
- **Verification status (updated 2026-11-14):** The 25-split CV was rerun on the original Linux host. SHAP was recomputed on Windows using fitted models from that Linux CV rerun. Key-aligned numerical comparisons of CV performance, SHAP importance, SHAP stability, and manuscript Tables 1 and 2 passed. This does not establish a single-platform, end-to-end rerun starting from raw NHANES XPT files.


### Stage 3 follow-up verification (2026-11-14)

The follow-up work was performed on branch `stage3-followup`.
The original Stage 3 analysis commit remains `4032bdb`.

- **Primary CV:** a fresh 25-split run on the original Linux
  host reproduced all 150 split-level performance results.
- **SHAP:** SHAP values were recomputed on Windows using the
  models saved by the Linux CV rerun. All 150 additivity
  checks passed. The original and rerun global-importance
  tables matched across 1,650 rows.
- **Stability and reporting:** the 150 split-level SHAP
  stability results and manuscript Tables 1 and 2 matched
  the original outputs in key-aligned comparisons.
- **Sample-size sensitivity:** a fresh 25-split run on the
  original Linux host reproduced the original results,
  including 100 split-level metric rows.
- **RF permutation-cycle sensitivity:** the split-1
  comparison of 5 versus 10 cycles was reproduced;
  all three 10-cycle additivity checks passed.
- **WTMEC2YR sensitivity:** the 25-split weighted-fitting
  analysis reproduced all 150 metric rows. An independent
  audit recalculated the saved performance metrics and
  verified the participant assignments and evaluation sets.
- **Figures:** Figure 1 and separate LR/RF rank-bump figures
  were regenerated from the rerun outputs at 300 DPI.
  Visual review found no problematic label overlap.
- **Cross-platform qualification:** Windows refitting of
  the KNN arm in the sample-size-matched sensitivity
  produced small numerical differences from the original
  Linux results. Median/Mode matched; a fresh Linux
  rerun reproduced the original KNN results. The exact
  platform-dependent numerical mechanism was not
  established.
- **Scope:** these checks do not constitute external
  validation or a full complex-survey inferential analysis.
  Participant-level split and prediction files remain private
  and are not included in Git.


## Scientific limitations

- The target is **self-reported prior diabetes diagnosis**, not measured clinical diabetes or incident diabetes.
- Evaluating only participants with complete predictors gives a common comparison set, but limits generalization to people with missing predictors.
- Missingness mechanisms were not established; none of the three strategies should be treated as a reference truth.
- Training-set size matching does not match participant identity, predictor distribution, or missingness pattern.
- The 10-cycle SHAP check covers **split 1 only**; SHAP additivity checks accuracy of the output decomposition, not necessarily attribution convergence.
- Sample weighting only changes classifier fitting, not imputation or other preprocessing, and is not complex-survey population inference.
- No external cohort validation is reported.

## Responsible interpretation and citation

This research evaluates explanation stability under alternative analytical choices; it does **not** provide a clinically validated diabetes screening tool or treatment advice.

If you use or adapt this repository, cite the associated paper or thesis when its bibliographic details become available. For the data source, acknowledge **National Health and Nutrition Examination Survey (NHANES), 2017–2018, U.S. National Center for Health Statistics / CDC**.

A software license has not yet been specified. Add a `LICENSE` file after choosing the intended reuse permissions.
