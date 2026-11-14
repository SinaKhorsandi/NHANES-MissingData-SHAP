#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent quality audit for Stage 3 WTMEC2YR sensitivity outputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "stage3_rerun" / "sensitivity" / "weights_full_v1"
PRIMARY = ROOT / "output" / "stage3_rerun"

FILES = {
    "metrics": OUT / "weight_sensitivity_metrics_by_split.csv",
    "predictions": OUT / "weight_sensitivity_predictions.csv",
    "manifest": OUT / "weight_sensitivity_training_manifest.csv",
    "runtime": OUT / "weight_sensitivity_runtime.csv",
    "summary": OUT / "weight_sensitivity_summary.csv",
    "weight_distribution": OUT / "wtmec2yr_distribution_summary.csv",
    "metadata": OUT / "weight_sensitivity_metadata.json",
    "primary_predictions": PRIMARY / "test_predictions_final.csv",
    "primary_metrics": PRIMARY / "cv_metrics_final.csv",
    "primary_manifest": PRIMARY / "split_manifest_final.csv",
    "primary_metadata": PRIMARY / "stage3_cv_metadata.json",
}

for label, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} file: {path}")

metrics = pd.read_csv(FILES["metrics"])
pred = pd.read_csv(FILES["predictions"])
manifest = pd.read_csv(FILES["manifest"])
runtime = pd.read_csv(FILES["runtime"])
summary = pd.read_csv(FILES["summary"])
weight_dist = pd.read_csv(FILES["weight_distribution"])
metadata = json.loads(FILES["metadata"].read_text(encoding="utf-8"))
primary_pred = pd.read_csv(FILES["primary_predictions"])
primary_metrics = pd.read_csv(FILES["primary_metrics"])
primary_manifest = pd.read_csv(FILES["primary_manifest"])
primary_metadata = json.loads(FILES["primary_metadata"].read_text(encoding="utf-8"))

checks = {}

def add(name, condition):
    checks[name] = bool(condition)

expected_models = {"LR", "RF"}
expected_methods = {"CCA", "Median/Mode", "KNN k=5"}
metric_cols = [
    "primary_unweighted_roc_auc",
    "primary_unweighted_brier_score",
    "weighted_fit_unweighted_eval_roc_auc",
    "weighted_fit_unweighted_eval_brier_score",
    "primary_weighted_eval_roc_auc",
    "primary_weighted_eval_brier_score",
    "weighted_fit_weighted_eval_roc_auc",
    "weighted_fit_weighted_eval_brier_score",
    "delta_unweighted_eval_roc_auc",
    "delta_unweighted_eval_brier_score",
    "delta_weighted_eval_roc_auc",
    "delta_weighted_eval_brier_score",
]
pred_cols = [
    "y_true",
    "WTMEC2YR",
    "primary_predicted_probability",
    "weighted_fit_predicted_probability",
]

add("150 weighted model results", len(metrics) == 150)
add("25 unique outer splits", metrics["split_id"].nunique() == 25)
add("Both models present", set(metrics["model"]) == expected_models)
add("All three methods present", set(metrics["method"]) == expected_methods)
add("Six model-method results per split",
    metrics.groupby("split_id").size().eq(6).all())
add("No duplicate metric rows",
    not metrics.duplicated(["split_id", "model", "method"]).any())
add("No missing metric values",
    not metrics[metric_cols].isna().any().any())
add("All metric values finite",
    np.isfinite(metrics[metric_cols].to_numpy(dtype=float)).all())
add("Six summary rows", len(summary) == 6)
add("25 runtime rows", len(runtime) == 25)
add("Metadata marks run complete", metadata.get("run_complete") is True)
add("Metadata reports 25 completed splits",
    metadata.get("n_outer_splits_completed") == 25)
add("Metadata reports 150 metric rows",
    metadata.get("n_metric_rows") == 150)
add("Input SHA-256 matches primary run",
    metadata.get("input_dataset_sha256")
    == primary_metadata.get("input_dataset_sha256"))
add("WTMEC2YR distribution covers 5098 participants",
    int(weight_dist.loc[0, "n_participants"]) == 5098)
add("No missing WTMEC2YR values",
    int(weight_dist.loc[0, "n_missing"]) == 0)
add("No nonpositive WTMEC2YR values",
    int(weight_dist.loc[0, "n_nonpositive"]) == 0)

# Validate manifest structure and per-split cohort counts.
add("Manifest has 25 x 5098 participant-role rows", len(manifest) == 25 * 5098)
add("No duplicate SEQN within a split",
    not manifest.duplicated(["split_id", "SEQN"]).any())
add("Each manifest split contains 5098 people",
    manifest.groupby("split_id").size().eq(5098).all())
add("All manifest weights are positive and finite",
    np.isfinite(manifest["WTMEC2YR"].to_numpy(dtype=float)).all()
    and (manifest["WTMEC2YR"] > 0).all())

# Verify that training cohort sizes match the intended CCA / imputation cohorts,
# and that common complete-case test sizes match the saved split manifest.
train_counts = manifest[manifest["role"].eq("train")].groupby("split_id").size()
cca_counts = manifest[manifest["used_for_CCA_training"]].groupby("split_id").size()
imputation_counts = manifest[manifest["used_for_imputation_training"]].groupby("split_id").size()
common_test_counts = (
    manifest[manifest["included_in_common_complete_case_test"]]
    .groupby("split_id").size()
)

train_size_ok = True
test_size_ok = True
for row in metrics.itertuples(index=False):
    sid = int(row.split_id)
    expected_train = int(cca_counts[sid]) if row.method == "CCA" else int(imputation_counts[sid])
    if int(row.train_n) != expected_train:
        train_size_ok = False
    if int(row.test_n) != int(common_test_counts[sid]):
        test_size_ok = False

add("CCA and imputation train sizes match manifest", train_size_ok)
add("All methods use common complete-case test size", test_size_ok)

# Check saved prediction rows, participant identity, and recompute metrics.
pred_key = ["split_id", "model", "method", "SEQN"]
add("No duplicate prediction participant rows",
    not pred.duplicated(pred_key).any())
add("Prediction fields contain no missing values",
    not pred[pred_cols].isna().any().any())
add("All prediction weights positive",
    (pred["WTMEC2YR"] > 0).all())
add("All predicted probabilities are in [0, 1]",
    pred[["primary_predicted_probability",
          "weighted_fit_predicted_probability"]].apply(
              lambda col: col.between(0, 1).all()
          ).all())
add("Prediction row count equals sum of test_n",
    len(pred) == int(metrics["test_n"].sum()))

prediction_counts = pred.groupby(["split_id", "model", "method"]).size()
prediction_sizes_ok = True
for row in metrics.itertuples(index=False):
    key = (int(row.split_id), row.model, row.method)
    if int(prediction_counts.loc[key]) != int(row.test_n):
        prediction_sizes_ok = False
add("Prediction counts match each metric test_n", prediction_sizes_ok)

# Compare weighted-run test participants and primary probabilities exactly.
merged = pred.merge(
    primary_pred[
        ["split_id", "model", "method", "SEQN", "y_true",
         "predicted_probability_class_1"]
    ],
    on=pred_key,
    how="outer",
    validate="one_to_one",
    indicator=True,
)
same_people = merged["_merge"].eq("both").all()
same_y = (
    merged["y_true_x"].astype(int).to_numpy()
    == merged["y_true_y"].astype(int).to_numpy()
).all()
same_primary_prob = np.allclose(
    merged["primary_predicted_probability"].to_numpy(dtype=float),
    merged["predicted_probability_class_1"].to_numpy(dtype=float),
    rtol=0,
    atol=1e-12,
)
add("Weighted and primary runs use identical test participants", same_people)
add("Weighted and primary runs use identical test outcomes", same_y)
add("Primary probabilities are preserved exactly", same_primary_prob)

# Recompute all four evaluation metric variants for each split/model/method.
recalculation_ok = True
for row in metrics.itertuples(index=False):
    key = (int(row.split_id), row.model, row.method)
    g = pred[
        (pred["split_id"] == key[0])
        & (pred["model"] == key[1])
        & (pred["method"] == key[2])
    ].sort_values("SEQN")
    y = g["y_true"].to_numpy(dtype=int)
    w = g["WTMEC2YR"].to_numpy(dtype=float)
    p_primary = g["primary_predicted_probability"].to_numpy(dtype=float)
    p_weighted = g["weighted_fit_predicted_probability"].to_numpy(dtype=float)

    observed = {
        "primary_unweighted_roc_auc": roc_auc_score(y, p_primary),
        "primary_unweighted_brier_score": brier_score_loss(y, p_primary),
        "weighted_fit_unweighted_eval_roc_auc": roc_auc_score(y, p_weighted),
        "weighted_fit_unweighted_eval_brier_score": brier_score_loss(y, p_weighted),
        "primary_weighted_eval_roc_auc": roc_auc_score(y, p_primary, sample_weight=w),
        "primary_weighted_eval_brier_score": brier_score_loss(y, p_primary, sample_weight=w),
        "weighted_fit_weighted_eval_roc_auc": roc_auc_score(y, p_weighted, sample_weight=w),
        "weighted_fit_weighted_eval_brier_score": brier_score_loss(y, p_weighted, sample_weight=w),
    }
    for col, value in observed.items():
        if not np.isclose(float(getattr(row, col)), float(value), rtol=0, atol=1e-10):
            recalculation_ok = False
            break
add("All saved evaluation metrics independently reproduce", recalculation_ok)

print("=== WTMEC2YR FULL-RUN QUALITY AUDIT ===")
for name, passed in checks.items():
    print(("PASS" if passed else "FAIL") + " | " + name)

print("\n=== SUMMARY FROM SAVED CSV ===")
print(summary.to_string(index=False))

print("\n=== KEY COUNTS ===")
print("Metric rows:", len(metrics))
print("Prediction rows:", len(pred))
print("Manifest rows:", len(manifest))
print("Summary rows:", len(summary))
print("Runtime rows:", len(runtime))

if not all(checks.values()):
    raise SystemExit("\nWTMEC2YR QUALITY AUDIT: FAIL")

print("\nWTMEC2YR QUALITY AUDIT: PASS")
