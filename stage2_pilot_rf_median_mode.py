#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES Stage 2 pilot: Random Forest + median/mode.

This script MUST be placed beside stage2_pilot_lr_median_mode.py, in the
verified Stage 1 project directory. Run the LR script first. The RF pilot
REUSES the saved train/test split and the exact LR preprocessing specification.
No Stage 1 input files, LR predictions, or LR outputs are overwritten.

Run:
    python stage2_pilot_rf_median_mode.py
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 42
PREDICTORS = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST",
    "BP_SYS_MEAN", "BP_DIA_MEAN", "RIAGENDR", "RIDRETH3",
    "DMDEDUC2", "SMQ020", "PAQ605",
]
TARGET = "diabetes_reported"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(y_true, prob, group_name, baseline_probability):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError(f"The group {group_name} is empty or has only one target class")
    return {
        "evaluation_group": group_name,
        "test_n": int(len(y_true)),
        "test_positive_n": int(np.sum(y_true)),
        "test_positive_percent": round(float(np.mean(y_true)) * 100, 2),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "baseline_brier_score": float(brier_score_loss(
            y_true, np.full(len(y_true), baseline_probability)
        )),
    }


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    out = root / "output" / "stage2_pilot_v1"
    manifest_path = out / "split_manifest.csv"
    lr_metadata_path = out / "pilot_metadata.json"
    lr_pipeline_path = out / "lr_median_mode_pipeline.joblib"
    lr_predictions_path = out / "test_predictions.csv"
    lr_metrics_path = out / "pilot_metrics.csv"

    required_files = [data_path, manifest_path, lr_metadata_path,
                      lr_pipeline_path, lr_predictions_path, lr_metrics_path]
    missing_files = [str(p.relative_to(root)) for p in required_files if not p.is_file()]
    if missing_files:
        sys.exit(
            "Required files are missing:\n- " + "\n- ".join(missing_files)
            + "\nRun stage2_pilot_lr_median_mode.py first, "
            "and keep this script next to prepare_nhanes.py."
        )

    df = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    old_predictions = pd.read_csv(lr_predictions_path)
    old_metrics = pd.read_csv(lr_metrics_path)
    with lr_metadata_path.open(encoding="utf-8") as f:
        lr_metadata = json.load(f)

    missing_cols = sorted(set(["SEQN", TARGET] + PREDICTORS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Data missing columns: {missing_cols}")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("Invalid or duplicated SEQN in input data")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes, 0 and 1, and no missing values")
    if manifest["SEQN"].isna().any() or manifest["SEQN"].duplicated().any():
        raise ValueError("Invalid or duplicated SEQN in split_manifest.csv")

    # A strict guard makes comparisons with the preceding LR experiment honest.
    actual_hash = sha256(data_path)
    if actual_hash != lr_metadata.get("input_sha256"):
        raise ValueError(
            "Input dataset differs from the LR pilot dataset. "
            "Restore the same Stage 1 dataset or rerun the LR pilot first."
        )
    if len(df) != len(manifest) or set(df["SEQN"]) != set(manifest["SEQN"]):
        raise ValueError("Dataset participant IDs differ from the original LR split")
    if list(lr_metadata.get("numeric_predictors", [])) + list(
            lr_metadata.get("categorical_predictors", [])) != PREDICTORS:
        raise ValueError("Feature list differs from the previous LR pilot")
    if lr_metadata.get("seed") != SEED:
        raise ValueError("Unexpected LR random seed")

    df = df.sort_values("SEQN").reset_index(drop=True)
    split = df[["SEQN", TARGET]].merge(
        manifest[["SEQN", "split", "diabetes_reported", "has_all_11_predictors"]],
        on="SEQN", how="left", validate="one_to_one", suffixes=("_data", "_manifest"),
    )
    if split["split"].isna().any() or not set(split["split"]).issubset({"train", "test"}):
        raise ValueError("Manifest must assign every participant to train or test")
    if not (split[f"{TARGET}_data"] == split[f"{TARGET}_manifest"]).all():
        raise ValueError("Target labels differ from the LR pilot")
    completeness = ~df[PREDICTORS].isna().any(axis=1)
    if not np.array_equal(completeness.to_numpy(), split["has_all_11_predictors"].to_numpy()):
        raise ValueError("Complete-case flags differ from LR pilot")

    train_mask = split["split"].eq("train").to_numpy()
    test_mask = split["split"].eq("test").to_numpy()
    X_train = df.loc[train_mask, PREDICTORS]
    y_train = df.loc[train_mask, TARGET].astype(int).to_numpy()
    X_test = df.loc[test_mask, PREDICTORS]
    y_test = df.loc[test_mask, TARGET].astype(int).to_numpy()
    seqn_test = df.loc[test_mask, "SEQN"].to_numpy()
    complete_test = completeness.to_numpy()[test_mask]
    if (len(y_train) != lr_metadata.get("train_n")
            or len(y_test) != lr_metadata.get("test_n")
            or int(np.sum(y_train)) != lr_metadata.get("train_positive_n")
            or int(np.sum(y_test)) != lr_metadata.get("test_positive_n")):
        raise ValueError("Train/test counts differ from the original LR pilot")

    old_predictions_sorted = old_predictions.sort_values("SEQN").reset_index(drop=True)
    current_test_sorted = pd.DataFrame({
        "SEQN": seqn_test, "y_true": y_test,
        "is_complete_case_test": complete_test,
    }).sort_values("SEQN").reset_index(drop=True)
    for col in ["SEQN", "y_true", "is_complete_case_test"]:
        if not np.array_equal(old_predictions_sorted[col].to_numpy(),
                              current_test_sorted[col].to_numpy()):
            raise ValueError(f"Test-set discrepancy compared to LR outputs: {col}")

    # Clone the *specification* of the original LR preprocessing pipeline.
    # Its numeric imputer, scaler, categorical mode imputer and one-hot encoder
    # will all be fitted again ONLY to these exact same training participants.
    lr_model = joblib.load(lr_pipeline_path)
    preprocessing = clone(lr_model.named_steps["preprocess"])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    train_prevalence = float(y_train.mean())

    rows = []
    for group_name, mask in [
        ("entire_held_out_test", np.ones(len(y_test), dtype=bool)),
        ("common_complete_case_test", complete_test),
    ]:
        rows.append(summarize(y_test[mask], probs[mask], group_name, train_prevalence))
    rf_metrics = pd.DataFrame(rows)

    rf_predictions = pd.DataFrame({
        "SEQN": seqn_test,
        "y_true": y_test,
        "predicted_probability_diabetes_reported_1": probs,
        "is_complete_case_test": complete_test,
        "baseline_probability_train_prevalence": train_prevalence,
    }).sort_values("SEQN")

    # Keep the Logistic Regression results intact; write only new RF files.
    rf_metrics.to_csv(out / "rf_pilot_metrics.csv", index=False)
    rf_predictions.to_csv(out / "rf_test_predictions.csv", index=False)
    joblib.dump(model, out / "rf_median_mode_pipeline.joblib")

    comparison = old_metrics[["evaluation_group", "test_n", "roc_auc", "brier_score"]].merge(
        rf_metrics[["evaluation_group", "test_n", "roc_auc", "brier_score"]],
        on=["evaluation_group", "test_n"], how="inner", validate="one_to_one",
        suffixes=("_lr", "_rf"),
    )
    if len(comparison) != len(rf_metrics):
        raise ValueError("LR and RF evaluation groups differ")
    comparison["auc_rf_minus_lr"] = comparison["roc_auc_rf"] - comparison["roc_auc_lr"]
    comparison["brier_rf_minus_lr"] = comparison["brier_score_rf"] - comparison["brier_score_lr"]
    comparison.to_csv(out / "lr_rf_pilot_comparison.csv", index=False)

    metadata = {
        "study_stage": "pilot_only_not_final_analysis",
        "model": "RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)",
        "outcome": "DIQ010 self-reported prior diabetes diagnosis; not clinical confirmation",
        "input_sha256": actual_hash,
        "train_test_assignment": "Read unchanged split_manifest.csv created by LR pilot",
        "preprocessing": "Clone LR numeric median + scale and categorical mode + one-hot specification; fit on train only",
        "predictors": PREDICTORS,
        "train_n": len(y_train), "test_n": len(y_test),
        "common_complete_case_test_n": int(np.sum(complete_test)),
        "random_state": SEED,
        "survey_weighting_applied": False,
        "interpretation": "One held-out split; not a final statistically validated model comparison",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }
    (out / "rf_pilot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=== STAGE 2 PILOT v1: RF + MEDIAN/MODE ===")
    print("Input participants:", len(df))
    print("Train participants:", len(y_train), "| positive:", int(np.sum(y_train)))
    print("Test participants:", len(y_test), "| positive:", int(np.sum(y_test)))
    print("Common complete-case test participants:", int(np.sum(complete_test)))
    print("\nHeld-out test results (same split as Logistic Regression):")
    for row in rows:
        print("-", row["evaluation_group"], "| N =", row["test_n"],
              "| ROC-AUC =", round(row["roc_auc"], 4),
              "| Brier =", round(row["brier_score"], 4),
              "| baseline Brier =", round(row["baseline_brier_score"], 4))
    print("\nLR vs RF: ROC-AUC and Brier on the same test participants")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNew files saved to:", out)
    print("NOTE: Pilot only; no SHAP, no tuning, no repeated CV, no survey weighting.")


if __name__ == "__main__":
    main()
