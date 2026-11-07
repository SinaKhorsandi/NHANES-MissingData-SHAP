#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES 2017-2018 Stage 2 pilot v1: KNN imputation with LR and RF.

Put this script BESIDE prepare_nhanes.py and the existing Stage 2 pilot scripts.
Run the LR, RF and CCA pilots first, then:
    python stage2_pilot_knn_lr_rf.py

Numeric predictors: StandardScaler -> KNNImputer(k=5), both trained only
on the ORIGINAL 4,078-person training set. Scaling is performed BEFORE
KNN so differing measurement units do not dominate neighbor distances.
Categorical predictors: training-set mode -> one-hot encoding.
The same unchanged 1,020-person test set is used for both models; the
798 complete-case test people form the common comparison group.

Outputs are placed in output/stage2_pilot_v1/ with 'knn_' prefixes.
Stage 1 files and previous Stage 2 outputs are never overwritten.
This is a pilot, not a validated final analysis or clinical diagnosis.
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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
K_NEIGHBORS = 5
TARGET = "diabetes_reported"
NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN",
]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605"]
PREDICTORS = NUMERIC + CATEGORICAL


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_for_group(y, proba, label, baseline_probability):
    if len(y) == 0 or len(set(y)) != 2:
        raise ValueError(f"Evaluation group {label} needs both classes")
    return {
        "evaluation_group": label,
        "test_n": int(len(y)),
        "test_positive_n": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier_score": float(brier_score_loss(y, proba)),
        "baseline_brier_score": float(brier_score_loss(
            y, np.full(len(y), baseline_probability)
        )),
    }


def build_preprocessor():
    # StandardScaler handles NaNs (ignoring them in fit). KNN then computes
    # neighbor distances on the standardized numeric feature scales.
    # Everything is fitted within the training fold ONLY.
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("scaler", StandardScaler()),
            ("knn_imputer", KNNImputer(n_neighbors=K_NEIGHBORS, weights="uniform")),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("mode_imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ], remainder="drop")


def validate_inputs(df, manifest, original_metadata, previous_predictions, source_hash):
    needed = ["SEQN", TARGET, *PREDICTORS]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Input dataset missing columns: {missing}")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("Input data have missing or duplicate SEQN")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must be valid 0/1 with both classes")
    if manifest["SEQN"].isna().any() or manifest["SEQN"].duplicated().any():
        raise ValueError("Split manifest has invalid participant IDs")
    if source_hash != original_metadata.get("input_sha256"):
        raise ValueError(
            "Stage 1 dataset differs from the original LR pilot. "
            "Do not mix different data versions; rerun all pilots for the new dataset."
        )
    if original_metadata.get("seed") != SEED:
        raise ValueError("Pilot random seed differs from earlier runs")
    if original_metadata.get("numeric_predictors") != NUMERIC:
        raise ValueError("Numeric predictors differ from previous pilots")
    if original_metadata.get("categorical_predictors") != CATEGORICAL:
        raise ValueError("Categorical predictors differ from previous pilots")
    if len(df) != len(manifest) or set(df["SEQN"]) != set(manifest["SEQN"]):
        raise ValueError("Participant IDs differ from saved train/test split")

    df = df.sort_values("SEQN").reset_index(drop=True)
    m = df[["SEQN", TARGET]].merge(
        manifest[["SEQN", "split", "diabetes_reported", "has_all_11_predictors"]],
        on="SEQN", how="left", validate="one_to_one", suffixes=("_data", "_manifest"),
    )
    if m["split"].isna().any() or set(m["split"]) != {"train", "test"}:
        raise ValueError("All people must be assigned train or test")
    if not (m[TARGET + "_data"] == m[TARGET + "_manifest"]).all():
        raise ValueError("Outcome labels differ from previous pilots")
    complete = ~df[PREDICTORS].isna().any(axis=1)
    if not np.array_equal(complete.to_numpy(), m["has_all_11_predictors"].to_numpy()):
        raise ValueError("Complete-case membership differs from previous pilots")
    train_mask = m["split"].eq("train").to_numpy()
    test_mask = m["split"].eq("test").to_numpy()
    if (int(train_mask.sum()) != original_metadata.get("train_n")
            or int(test_mask.sum()) != original_metadata.get("test_n")):
        raise ValueError("Train/test sample counts differ from previous pilots")

    now_test = pd.DataFrame({
        "SEQN": df.loc[test_mask, "SEQN"].to_numpy(),
        "y_true": df.loc[test_mask, TARGET].astype(int).to_numpy(),
        "is_complete_case_test": complete.to_numpy()[test_mask],
    }).sort_values("SEQN").reset_index(drop=True)
    before = previous_predictions.sort_values("SEQN").reset_index(drop=True)
    for col in ["SEQN", "y_true", "is_complete_case_test"]:
        if not np.array_equal(now_test[col].to_numpy(), before[col].to_numpy()):
            raise ValueError(f"Test participant or label mismatch: {col}")
    return df, train_mask, test_mask, complete.to_numpy()


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    out = root / "output" / "stage2_pilot_v1"
    required = [
        data_path,
        out / "split_manifest.csv",
        out / "pilot_metadata.json",
        out / "test_predictions.csv",
        out / "pilot_metrics.csv",
        out / "rf_pilot_metrics.csv",
        out / "cca_pilot_metrics.csv",
        out / "cca_test_predictions.csv",
    ]
    absent = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if absent:
        sys.exit("Missing required files:\n- " + "\n- ".join(absent)
                 + "\nRun earlier LR, RF and CCA pilots first.")

    df = pd.read_csv(data_path)
    manifest = pd.read_csv(out / "split_manifest.csv")
    old_preds = pd.read_csv(out / "test_predictions.csv")
    old_lr = pd.read_csv(out / "pilot_metrics.csv")
    old_rf = pd.read_csv(out / "rf_pilot_metrics.csv")
    old_cca = pd.read_csv(out / "cca_pilot_metrics.csv")
    old_cca_preds = pd.read_csv(out / "cca_test_predictions.csv")
    original_metadata = json.loads((out / "pilot_metadata.json").read_text(encoding="utf-8"))
    data_hash = sha256(data_path)
    df, train_mask, test_mask, complete = validate_inputs(
        df, manifest, original_metadata, old_preds, data_hash
    )
    X_train = df.loc[train_mask, PREDICTORS]
    y_train = df.loc[train_mask, TARGET].astype(int).to_numpy()
    X_test = df.loc[test_mask, PREDICTORS]
    y_test = df.loc[test_mask, TARGET].astype(int).to_numpy()
    seqn_test = df.loc[test_mask, "SEQN"].to_numpy()
    complete_test = complete[test_mask]
    baseline = float(y_train.mean())
    if not np.array_equal(np.unique(y_train), np.array([0, 1])):
        raise ValueError("Training set needs both outcome classes")
    if (len(y_train) != 4078 or len(y_test) != 1020 or int(complete_test.sum()) != 798):
        print("NOTE: sample counts differ from original pilot; using verified split manifest.")
    if old_cca_preds.shape[0] != int(complete_test.sum()):
        raise ValueError("CCA test sample count does not match complete-case group")
    correct_common = pd.DataFrame({
        "SEQN": seqn_test[complete_test], "y_true": y_test[complete_test]
    }).sort_values("SEQN").reset_index(drop=True)
    cca_common = old_cca_preds[["SEQN", "y_true"]].sort_values("SEQN").reset_index(drop=True)
    if not correct_common.equals(cca_common):
        raise ValueError("CCA test participant IDs/labels differ from this KNN pilot")

    estimators = {
        "LR": LogisticRegression(max_iter=2000, random_state=SEED),
        "RF": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1,
        ),
    }
    metrics_rows = []
    predictions = pd.DataFrame({"SEQN": seqn_test, "y_true": y_test,
                                "is_complete_case_test": complete_test})
    for label, estimator in estimators.items():
        model = Pipeline([("preprocess", build_preprocessor()), ("classifier", estimator)])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        predictions[f"knn_{label.lower()}_probability"] = probs
        for group, mask in [
            ("entire_held_out_test", np.ones(len(y_test), dtype=bool)),
            ("common_complete_case_test", complete_test),
        ]:
            metrics_rows.append({"model": label, "method": "KNN_numeric_k5_categorical_mode",
                                 "train_n": int(len(y_train)),
                                 "train_positive_n": int(np.sum(y_train)),
                                 **metrics_for_group(y_test[mask], probs[mask], group, baseline)})
        joblib.dump(model, out / f"knn_{label.lower()}_pipeline.joblib")

    knn_results = pd.DataFrame(metrics_rows)
    knn_results.to_csv(out / "knn_pilot_metrics.csv", index=False)
    predictions.sort_values("SEQN").to_csv(out / "knn_test_predictions.csv", index=False)

    # Shared 798-person test, so group performance is comparable. CCA uses
    # fewer training people; KNN/Median use the same 4,078 train participants.
    comparison_rows = []
    for label, source in [
        ("Median/Mode", old_lr.assign(model="LR")),
        ("Median/Mode", old_rf.assign(model="RF")),
        ("CCA", old_cca),
        ("KNN k=5", knn_results),
    ]:
        subset = source.copy()
        if "evaluation_group" in subset.columns:
            subset = subset[subset["evaluation_group"] == "common_complete_case_test"]
        for _, row in subset.iterrows():
            comparison_rows.append({
                "model": row["model"],
                "method": label,
                "train_n": int(row["train_n"]) if "train_n" in row else int(original_metadata["train_n"]),
                "test_n_same_complete_cases": int(row["test_n"]),
                "roc_auc": float(row["roc_auc"]),
                "brier_score": float(row["brier_score"]),
            })
    comparison = pd.DataFrame(comparison_rows)
    expected = {(x, y) for x in ["LR", "RF"] for y in ["Median/Mode", "CCA", "KNN k=5"]}
    if (len(comparison) != 6 or
            set(zip(comparison["model"], comparison["method"])) != expected or
            set(comparison["test_n_same_complete_cases"]) != {int(sum(complete_test))}):
        raise ValueError("The previous pilot comparison rows are incomplete or incompatible")
    comparison = comparison.sort_values(["model", "method"]).reset_index(drop=True)
    comparison.to_csv(out / "all_six_pilot_comparison_common_test.csv", index=False)

    (out / "knn_pilot_metadata.json").write_text(json.dumps({
        "stage": "pilot_only_not_final_analysis",
        "outcome_definition": "DIQ010 prior physician-diagnosed diabetes self-report, not clinical diagnosis",
        "data_file_sha256": data_hash,
        "source_split_file": "output/stage2_pilot_v1/split_manifest.csv",
        "seed": SEED,
        "train_n": int(len(y_train)),
        "train_positive_n": int(y_train.sum()),
        "test_n": int(len(y_test)),
        "test_positive_n": int(y_test.sum()),
        "complete_case_test_n": int(sum(complete_test)),
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "numeric_imputation": "StandardScaler fit on train -> KNNImputer(k=5, weights=uniform) fit on train",
        "categorical_imputation": "SimpleImputer(most_frequent) fit on train -> OneHotEncoder fit on train",
        "random_forest": "RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)",
        "logistic_regression": "LogisticRegression(max_iter=2000, random_state=42)",
        "same_test_ids_and_labels_checked": True,
        "same_training_people_as_median_mode": True,
        "CCA_uses_fewer_training_people": True,
        "repeated_cross_validation_tuning_SHAP_survey_weights": "NOT DONE",
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== STAGE 2 PILOT v1: KNN(k=5) + LR/RF ===")
    print("Input participants:", len(df))
    print("Train participants:", len(y_train), "| positive:", int(np.sum(y_train)))
    print("Test participants:", len(y_test), "| positive:", int(np.sum(y_test)))
    print("Common complete-case test participants:", int(sum(complete_test)))
    print("\nKNN results:")
    print(knn_results[["model", "evaluation_group", "train_n", "test_n", "roc_auc", "brier_score"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nAll SIX combinations on the same complete-case test participants:")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNew files saved to:", out)
    print("NOTE: Pilot only; no SHAP, tuning, repeated CV, or NHANES survey weighting.")


if __name__ == "__main__":
    main()
