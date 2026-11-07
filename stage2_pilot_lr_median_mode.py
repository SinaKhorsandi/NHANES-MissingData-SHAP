#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES 2017-2018 | Stage 2 pilot v1.

Binary classification of SELF-REPORTED prior physician-diagnosed diabetes,
not clinical diabetes diagnosis. Uses 11 provisional predictors and the
examined subgroup. Imputation is fit ONLY on the training split.

Run from the Stage 1 project root:
    python stage2_pilot_lr_median_mode.py

Optional:
    python stage2_pilot_lr_median_mode.py --data output/nhanes_examined_clean_with_missing.csv

Outputs are written under output/stage2_pilot_v1/; Stage 1 files are unchanged.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST",
    "BP_SYS_MEAN", "BP_DIA_MEAN",
]
CATEGORICAL = [
    "RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605",
]
PREDICTORS = NUMERIC + CATEGORICAL
TARGET = "diabetes_reported"
SEED = 42
TEST_FRACTION = 0.20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(y_true, probs, name):
    """ROC-AUC and Brier; a single-class held-out set has undefined AUC."""
    if len(y_true) == 0:
        raise ValueError(f"Empty evaluation group: {name}")
    return {
        "evaluation_group": name,
        "test_n": int(len(y_true)),
        "test_positive_n": int(np.sum(y_true)),
        "test_positive_percent": round(100 * float(np.mean(y_true)), 2),
        "roc_auc": (float(roc_auc_score(y_true, probs))
                    if len(np.unique(y_true)) == 2 else None),
        "brier_score": float(brier_score_loss(y_true, probs)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    p.add_argument("--data", type=Path,
                   default=root / "output" / "nhanes_examined_clean_with_missing.csv")
    p.add_argument("--out", type=Path,
                   default=root / "output" / "stage2_pilot_v1")
    args = p.parse_args()
    data_path = args.data.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not data_path.is_file():
        sys.exit(f"INPUT NOT FOUND: {data_path}\n"
                 "Put this script next to prepare_nhanes.py, or pass --data PATH")

    df = pd.read_csv(data_path)
    required = ["SEQN", TARGET] + PREDICTORS
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if len(df) != 5098:
        print(f"NOTE: input has {len(df)} rows; Stage 2 v1 expected 5098 examined adults.")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("SEQN must be present and unique")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1 without missing values")
    if not set(df[TARGET].unique()) == {0, 1}:
        raise ValueError("Both outcome classes are required")

    # Sorting makes the seeded split invariant to CSV row order.
    df = df.sort_values("SEQN").reset_index(drop=True)
    X, y = df[PREDICTORS], df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_FRACTION,
        random_state=SEED, stratify=y,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Imputers, scaler, encoder and LR are a SINGLE pipeline:
    # ALL fit operations see training data only, never the test data.
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000, random_state=SEED,
        )),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    # This is the common test set for *later* CCA-vs-imputation comparisons.
    test_complete_mask = (~X_test.isna().any(axis=1)).to_numpy()
    if sum(test_complete_mask) < 2:
        raise ValueError("Not enough complete test cases for common evaluation")

    # Prevalence-only baseline helps interpret Brier score in an imbalanced cohort.
    baseline_probs = np.full(len(X_test), float(y_train.mean()))
    results = []
    for label, mask in [
        ("entire_held_out_test", np.ones(len(X_test), dtype=bool)),
        ("common_complete_case_test", test_complete_mask),
    ]:
        row = summarize(y_test.to_numpy()[mask], probabilities[mask], label)
        row["baseline_brier_score"] = float(brier_score_loss(
            y_test.to_numpy()[mask], baseline_probs[mask]
        ))
        results.append(row)

    out.mkdir(parents=True, exist_ok=True)
    # Persist each participant's assignment for future comparisons.
    split_labels = np.full(len(df), "train", dtype="<U5")
    split_labels[test_idx] = "test"
    split_file = pd.DataFrame({
        "SEQN": df["SEQN"],
        "split": split_labels,
        "diabetes_reported": y,
        "has_all_11_predictors": ~X.isna().any(axis=1),
    })
    split_file.to_csv(out / "split_manifest.csv", index=False)

    predictions = pd.DataFrame({
        "SEQN": df.loc[test_idx, "SEQN"].to_numpy(),
        "y_true": y_test.to_numpy(),
        "predicted_probability_diabetes_reported_1": probabilities,
        "is_complete_case_test": test_complete_mask,
        "baseline_probability_train_prevalence": baseline_probs,
    }).sort_values("SEQN")
    predictions.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(out / "pilot_metrics.csv", index=False)
    joblib.dump(model, out / "lr_median_mode_pipeline.joblib")

    metadata = {
        "study_stage": "pilot_only_not_final_analysis",
        "outcome_definition": "DIQ010 yes/no; self-reported prior diagnosis",
        "input_path": str(data_path),
        "input_sha256": sha256(data_path),
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "train_n": int(len(train_idx)),
        "train_positive_n": int(y_train.sum()),
        "test_n": int(len(test_idx)),
        "test_positive_n": int(y_test.sum()),
        "common_complete_case_test_n": int(test_complete_mask.sum()),
        "train_any_missing_n": int(X_train.isna().any(axis=1).sum()),
        "numeric_predictors": NUMERIC,
        "categorical_predictors": CATEGORICAL,
        "preprocessing": "train-only numeric median+standardization; categorical mode+one-hot",
        "estimator": "LogisticRegression(max_iter=2000, default regularization)",
        "weights_or_survey_design_used": False,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (out / "pilot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=== STAGE 2 PILOT v1: LR + MEDIAN/MODE ===")
    print("Input participants:", len(df))
    print("Train participants:", len(train_idx), "| positive:", int(y_train.sum()))
    print("Test participants:", len(test_idx), "| positive:", int(y_test.sum()))
    print("Common complete-case test participants:", int(test_complete_mask.sum()))
    print("\nHeld-out test results (no imputation leakage):")
    for row in results:
        print("-", row["evaluation_group"],
              "| N =", row["test_n"],
              "| ROC-AUC =", round(row["roc_auc"], 4) if row["roc_auc"] is not None else "NA",
              "| Brier =", round(row["brier_score"], 4),
              "| baseline Brier =", round(row["baseline_brier_score"], 4))
    print("\nSaved to:", out)
    print("NOTE: This is a pilot. No SHAP or model comparison has been performed.")


if __name__ == "__main__":
    main()
