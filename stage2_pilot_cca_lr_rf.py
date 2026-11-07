#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES Stage 2 pilot: complete-case analysis (CCA) for LR and RF.

Place this script beside stage2_pilot_lr_median_mode.py and
stage2_pilot_rf_median_mode.py, after running those two scripts.

CCA uses ONLY training participants with all 11 predictors observed.
Evaluation is ONLY on the SAME 798 complete-case held-out participants
used in the previous median/mode models, identified by saved SEQN split.

This is a methodological pilot, not final analysis. CCA vs imputation
is confounded by training-sample size/composition until sensitivity
analysis is done. All previously saved files are left untouched.
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST",
    "BP_SYS_MEAN", "BP_DIA_MEAN",
]
CATEGORICAL = [
    "RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605",
]
PREDICTORS = NUMERIC + CATEGORICAL
TARGET = "diabetes_reported"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score(y_true, probs):
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
    }


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    out = root / "output" / "stage2_pilot_v1"
    required_files = [
        data_path,
        out / "split_manifest.csv",
        out / "pilot_metadata.json",
        out / "pilot_metrics.csv",
        out / "rf_pilot_metrics.csv",
        out / "test_predictions.csv",
        out / "rf_test_predictions.csv",
    ]
    absent = [str(p.relative_to(root)) for p in required_files if not p.is_file()]
    if absent:
        sys.exit("Missing required files:\n- " + "\n- ".join(absent)
                 + "\nRun the existing LR and RF median/mode pilots first.")

    df = pd.read_csv(data_path)
    manifest = pd.read_csv(out / "split_manifest.csv")
    old_lr = pd.read_csv(out / "pilot_metrics.csv")
    old_rf = pd.read_csv(out / "rf_pilot_metrics.csv")
    lr_test_predictions = pd.read_csv(out / "test_predictions.csv")
    rf_test_predictions = pd.read_csv(out / "rf_test_predictions.csv")
    metadata = json.loads((out / "pilot_metadata.json").read_text(encoding="utf-8"))

    if sha256(data_path) != metadata.get("input_sha256"):
        raise ValueError("Stage 1 input has changed since the previous LR pilot; rerun that pilot first.")
    if metadata.get("seed") != SEED:
        raise ValueError("Random seed differs from the earlier pilot")
    if metadata.get("numeric_predictors") != NUMERIC or metadata.get("categorical_predictors") != CATEGORICAL:
        raise ValueError("Predictor definitions differ from the earlier pilot")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("Invalid SEQN in input data")
    if manifest["SEQN"].isna().any() or manifest["SEQN"].duplicated().any():
        raise ValueError("Invalid SEQN in split manifest")
    if len(df) != len(manifest) or set(df["SEQN"]) != set(manifest["SEQN"]):
        raise ValueError("Input IDs and original train/test split disagree")
    if not set(df[TARGET].unique()) == {0, 1} or df[TARGET].isna().any():
        raise ValueError("Outcome must contain both classes, 0 and 1, without missing values")
    missing_cols = sorted(set(PREDICTORS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Predictors missing from input: {missing_cols}")

    df = df.sort_values("SEQN").reset_index(drop=True)
    split = df[["SEQN", TARGET]].merge(
        manifest[["SEQN", "split", "diabetes_reported", "has_all_11_predictors"]],
        on="SEQN", how="left", validate="one_to_one", suffixes=("_data", "_manifest"),
    )
    if split["split"].isna().any() or not set(split["split"]).issubset({"train", "test"}):
        raise ValueError("Invalid split labels")
    if not (split[f"{TARGET}_data"] == split[f"{TARGET}_manifest"]).all():
        raise ValueError("Outcome values differ from the original pilot")
    complete = ~df[PREDICTORS].isna().any(axis=1)
    if not np.array_equal(complete.to_numpy(), split["has_all_11_predictors"].to_numpy()):
        raise ValueError("Complete-case membership differs from the earlier pilot")

    train_all = split["split"].eq("train").to_numpy()
    test_all = split["split"].eq("test").to_numpy()
    train_cca = train_all & complete.to_numpy()
    test_cca = test_all & complete.to_numpy()
    X_train = df.loc[train_cca, PREDICTORS]
    y_train = df.loc[train_cca, TARGET].astype(int).to_numpy()
    X_test = df.loc[test_cca, PREDICTORS]
    y_test = df.loc[test_cca, TARGET].astype(int).to_numpy()
    seqn_test = df.loc[test_cca, "SEQN"].to_numpy()
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("CCA dataset cannot contain missing predictors")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("CCA train/test must each contain both outcome classes")

    # Verify the *exact* test individuals and their labels match the previous models.
    expected = pd.DataFrame({"SEQN": seqn_test, "y_true": y_test}).sort_values("SEQN").reset_index(drop=True)
    for name, old_predictions in [("LR", lr_test_predictions), ("RF", rf_test_predictions)]:
        original_complete = old_predictions.loc[
            old_predictions["is_complete_case_test"].astype(bool), ["SEQN", "y_true"]
        ].sort_values("SEQN").reset_index(drop=True)
        if not original_complete.equals(expected):
            raise ValueError(f"CCA test people/labels differ from previous {name} pilot")

    # CCA deliberately does NOT include imputers. Scaling and category encoding
    # are trained ONLY on the complete-case training participants.
    def make_preprocess():
        return ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ], remainder="drop")

    models = {
        "LR": Pipeline([
            ("preprocess", make_preprocess()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "RF": Pipeline([
            ("preprocess", make_preprocess()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1,
            )),
        ]),
    }

    results = []
    preds = pd.DataFrame({"SEQN": seqn_test, "y_true": y_test})
    baseline_cca = float(y_train.mean())
    baseline_brier = float(brier_score_loss(y_test, np.full(len(y_test), baseline_cca)))
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        scores = score(y_test, probs)
        results.append({
            "method": "CCA", "model": name,
            "train_n": int(len(y_train)), "train_positive_n": int(y_train.sum()),
            "test_n": int(len(y_test)), "test_positive_n": int(y_test.sum()),
            **scores,
            "baseline_brier_score_cca_train_prevalence": baseline_brier,
        })
        preds[f"cca_{name.lower()}_probability"] = probs
        joblib.dump(model, out / f"cca_{name.lower()}_pipeline.joblib")

    results = pd.DataFrame(results)
    results.to_csv(out / "cca_pilot_metrics.csv", index=False)
    preds.sort_values("SEQN").to_csv(out / "cca_test_predictions.csv", index=False)
    pd.DataFrame({
        "SEQN": df["SEQN"],
        "original_split": split["split"],
        "complete_case": complete,
        "used_for_cca_training": train_cca,
        "used_for_common_complete_case_test": test_cca,
    }).to_csv(out / "cca_membership.csv", index=False)

    # Read the 798-person rows from existing metrics; no need to rerun prior models.
    comparison = []
    for model_name, prior in [("LR", old_lr), ("RF", old_rf)]:
        prior_row = prior.loc[prior["evaluation_group"] == "common_complete_case_test"]
        if len(prior_row) != 1 or int(prior_row.iloc[0]["test_n"]) != len(y_test):
            raise ValueError(f"No matching complete-case test score from prior {model_name}")
        cca_row = results.loc[results["model"] == model_name].iloc[0]
        previous = prior_row.iloc[0]
        comparison.append({
            "model": model_name,
            "test_n_same_complete_cases": int(len(y_test)),
            "train_n_imputed": int(train_all.sum()),
            "train_n_cca": int(len(y_train)),
            "auc_median_mode": float(previous["roc_auc"]),
            "auc_cca": float(cca_row["roc_auc"]),
            "auc_cca_minus_imputed": float(cca_row["roc_auc"] - previous["roc_auc"]),
            "brier_median_mode": float(previous["brier_score"]),
            "brier_cca": float(cca_row["brier_score"]),
            "brier_cca_minus_imputed": float(cca_row["brier_score"] - previous["brier_score"]),
        })
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out / "cca_vs_median_mode_pilot_comparison.csv", index=False)

    (out / "cca_pilot_metadata.json").write_text(json.dumps({
        "analysis_stage": "pilot_only_not_final_results",
        "source_sha256": sha256(data_path),
        "same_original_split_as_lr_rf_median_mode": True,
        "evaluation_group": "common_complete_case_test_only",
        "preprocessing": "no imputation; standard scaling and one-hot encoding train only",
        "predictors": PREDICTORS,
        "train_n_full_median_mode": int(train_all.sum()),
        "train_n_cca": int(len(y_train)),
        "test_n_common_complete_case": int(len(y_test)),
        "uncontrolled_difference": "CCA uses fewer and potentially different training participants; NOT an isolated imputation effect",
        "tuning_repeated_cv_shap_survey_weights": "NOT YET DONE",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== STAGE 2 PILOT v1: CCA + LR/RF ===")
    print("All participants:", len(df))
    print("Original train participants:", int(train_all.sum()))
    print("CCA train participants:", len(y_train), "| positive:", int(y_train.sum()))
    print("Common complete-case test participants:", len(y_test), "| positive:", int(y_test.sum()))
    print("\nCCA results (same 798 test individuals as previous LR/RF models):")
    print(results[["model", "train_n", "test_n", "roc_auc", "brier_score"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCCA versus median/mode (same TEST people, different TRAIN cohorts):")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNew outputs saved to:", out)
    print("NOTE: Pilot only; SHAP, repeated CV, survey weights, and matched-train-size sensitivity are still pending.")


if __name__ == "__main__":
    main()
