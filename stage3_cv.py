#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 outer cross-validation for the NHANES SHAP stability study.

This script implements the advisor-approved 5-fold x 5-repeat outer CV,
three missing-data strategies, two classifiers, and inner 3-fold tuning.
All preprocessing is fitted within each training fold only.

First timing run (one outer split):
    python stage3_cv.py --limit-splits 1 --out output/stage3_timing_run1

Full 25-split run (only after the timing gate is passed):
    python stage3_cv.py --out output/stage3_final

The output directory must be new or empty. Models are saved for the later
stage3_shap.py step. No survey weights are used in the primary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "diabetes_reported"
PREDICTORS = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "BP_SYS_MEAN",
    "BP_DIA_MEAN",
    "SMQ020",
    "PAQ605",
]
NUMERIC = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "BP_SYS_MEAN",
    "BP_DIA_MEAN",
]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605"]
METHODS = ["CCA", "Median/Mode", "KNN k=5"]
MODELS = ["LR", "RF"]

EXPECTED_N = 5098
EXPECTED_BP_MISSING = 258
OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 3
BASE_SEED = 2026
KNN_NEIGHBORS = 5

LR_C_GRID = [0.01, 0.1, 1.0, 10.0]
RF_MAX_DEPTH_GRID = [5, 10, None]
RF_MIN_SAMPLES_LEAF_GRID = [2, 5, 10]
RF_N_ESTIMATORS = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out_dir / "stage3_cv.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}\n"
            "Choose a new --out directory. Existing results will not be overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_input(df: pd.DataFrame, data_path: Path, root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = ["SEQN", "RIDAGEYR", "RIDSTATR", "DIQ010", TARGET, *PREDICTORS]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input dataset is missing required columns: {missing_columns}")

    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected exactly {EXPECTED_N} participants; found {len(df)}")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("SEQN must be non-missing and unique")
    if not (pd.to_numeric(df["RIDAGEYR"], errors="coerce") >= 20).all():
        raise ValueError("All participants must be age 20 or older")
    if not (pd.to_numeric(df["RIDSTATR"], errors="coerce") == 2).all():
        raise ValueError("All participants must have RIDSTATR = 2")
    if not set(pd.to_numeric(df["DIQ010"], errors="coerce").dropna().unique()).issubset({1, 2}):
        raise ValueError("DIQ010 must contain only valid codes 1 and 2")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("diabetes_reported must contain both classes 0 and 1 without missing values")
    expected_target = (pd.to_numeric(df["DIQ010"], errors="coerce") == 1).astype(int)
    if not np.array_equal(expected_target.to_numpy(), df[TARGET].astype(int).to_numpy()):
        raise ValueError("diabetes_reported does not match DIQ010 coding (1=yes, 2=no)")

    for col in ["BP_SYS_MEAN", "BP_DIA_MEAN"]:
        missing_n = int(df[col].isna().sum())
        if missing_n != EXPECTED_BP_MISSING:
            raise ValueError(
                f"{col} has {missing_n} missing values; expected {EXPECTED_BP_MISSING} "
                "under the accepted minimum-one-reading rule. Review Stage 1 before proceeding."
            )

    notes_path = root / "output" / "preparation_notes.txt"
    if not notes_path.is_file():
        raise FileNotFoundError(f"Missing preparation notes: {notes_path}")
    notes_text = notes_path.read_text(encoding="utf-8")
    if "BP summary rule (Stage 3)" not in notes_text:
        raise ValueError(
            "The accepted blood-pressure rule is not recorded in output/preparation_notes.txt"
        )

    # Sort by participant ID before creating seeded splits so row order cannot
    # silently change the CV assignments.
    df = df.sort_values("SEQN").reset_index(drop=True)
    X = df[PREDICTORS].copy()
    y = df[TARGET].astype(int).copy()
    complete = (~X.isna().any(axis=1)).to_numpy(dtype=bool)
    return df, y.to_numpy(dtype=int), complete


def build_preprocessor(method: str, model_name: str) -> ColumnTransformer:
    if method == "CCA":
        if model_name == "LR":
            numeric_transformer: Any = StandardScaler()
        else:
            numeric_transformer = "passthrough"
        categorical_transformer: Any = OneHotEncoder(handle_unknown="ignore")

    elif method == "Median/Mode":
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if model_name == "LR":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(numeric_steps)
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

    elif method == "KNN k=5":
        # Numeric variables are standardized before distance calculation.
        # The scaler and KNN imputer are both fitted on the current training fold.
        numeric_transformer = Pipeline([
            ("scale_for_distance", StandardScaler()),
            ("knn_imputer", KNNImputer(
                n_neighbors=KNN_NEIGHBORS,
                weights="uniform",
                keep_empty_features=True,
            )),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

    else:
        raise ValueError(f"Unknown missing-data method: {method}")

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC),
            ("categorical", categorical_transformer, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(method: str, model_name: str, params: dict[str, Any], random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(method, model_name)
    if model_name == "LR":
        classifier = LogisticRegression(
            solver="lbfgs",
            l1_ratio=0.0,
            C=float(params["C"]),
            max_iter=5000,
            random_state=random_state,
        )
    elif model_name == "RF":
        classifier = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ])


def parameter_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "LR":
        return [{"C": value} for value in LR_C_GRID]
    if model_name == "RF":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth, leaf in product(RF_MAX_DEPTH_GRID, RF_MIN_SAMPLES_LEAF_GRID)
        ]
    raise ValueError(f"Unknown model: {model_name}")


def tune_hyperparameters(
    X_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    complete_outer_train: np.ndarray,
    method: str,
    model_name: str,
    split_id: int,
    random_state: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], float]:
    """Tune one method/model using common complete-case inner validation rows.

    Inner folds are generated once from the full outer-training cohort. In each
    fold, all methods are scored on the complete-case members of that inner
    validation fold. CCA trains only on complete inner-training rows; imputation
    methods train on all inner-training rows. This keeps the inner validation
    participants common while respecting each strategy's training rule.
    """
    started = time.perf_counter()
    inner_seed = random_state + 1000 + split_id
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=inner_seed)
    inner_folds = list(inner_cv.split(np.zeros(len(y_outer_train)), y_outer_train))

    fold_specs: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_fold_no, (inner_train_rel, inner_val_rel) in enumerate(inner_folds, start=1):
        val_complete_rel = inner_val_rel[complete_outer_train[inner_val_rel]]
        y_val = y_outer_train.iloc[val_complete_rel].to_numpy(dtype=int)
        if len(y_val) == 0 or len(np.unique(y_val)) != 2:
            raise ValueError(
                f"Split {split_id}, inner fold {inner_fold_no}: common complete-case "
                "validation subset is empty or lacks one outcome class."
            )
        if method == "CCA":
            fit_rel = inner_train_rel[complete_outer_train[inner_train_rel]]
        else:
            fit_rel = inner_train_rel
        y_fit = y_outer_train.iloc[fit_rel].to_numpy(dtype=int)
        if len(y_fit) == 0 or len(np.unique(y_fit)) != 2:
            raise ValueError(
                f"Split {split_id}, inner fold {inner_fold_no}: training subset "
                "is empty or lacks one outcome class."
            )
        fold_specs.append((fit_rel, val_complete_rel))

    candidate_rows: list[dict[str, Any]] = []
    candidates = parameter_grid(model_name)
    for candidate_no, params in enumerate(candidates, start=1):
        fold_scores: list[float] = []
        for inner_fold_no, (fit_rel, val_rel) in enumerate(fold_specs, start=1):
            estimator = build_pipeline(
                method=method,
                model_name=model_name,
                params=params,
                random_state=random_state,
            )
            estimator.fit(X_outer_train.iloc[fit_rel], y_outer_train.iloc[fit_rel])
            probabilities = estimator.predict_proba(X_outer_train.iloc[val_rel])[:, 1]
            score = float(roc_auc_score(y_outer_train.iloc[val_rel], probabilities))
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        row = {
            "split_id": split_id,
            "model": model_name,
            "method": method,
            "candidate_id": candidate_no,
            "params_json": json.dumps(json_safe(params), sort_keys=True),
            "inner_fold_1_roc_auc": fold_scores[0],
            "inner_fold_2_roc_auc": fold_scores[1],
            "inner_fold_3_roc_auc": fold_scores[2],
            "inner_mean_roc_auc": mean_score,
            "inner_sd_roc_auc": float(np.std(fold_scores, ddof=1)),
        }
        candidate_rows.append(row)

    # Stable tie-breaking: the first candidate in the predeclared grid wins.
    best_index = int(np.argmax([row["inner_mean_roc_auc"] for row in candidate_rows]))
    best_row = candidate_rows[best_index]
    best_params = json.loads(best_row["params_json"])
    logger.info(
        "Split %02d | %s | %s | best inner ROC-AUC=%.6f | params=%s",
        split_id, model_name, method, best_row["inner_mean_roc_auc"], best_row["params_json"],
    )
    return best_params, float(best_row["inner_mean_roc_auc"]), candidate_rows, time.perf_counter() - started


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=root / "output" / "nhanes_examined_clean_with_missing.csv",
        help="Stage 1 examined-cohort CSV (default: output/nhanes_examined_clean_with_missing.csv)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=root / "output" / "stage3_final",
        help="New or empty output directory (default: output/stage3_final)",
    )
    parser.add_argument(
        "--limit-splits", type=int, default=OUTER_SPLITS * OUTER_REPEATS,
        help="Run only the first N outer splits for testing/timing (1..25)",
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help="Fixed random seed for outer CV and estimators (default: 2026)",
    )
    args = parser.parse_args()

    if not (1 <= args.limit_splits <= OUTER_SPLITS * OUTER_REPEATS):
        parser.error("--limit-splits must be between 1 and 25")

    data_path = args.data.expanduser().resolve()
    out_dir = args.out.expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")
    prepare_output_dir(out_dir)
    logger = make_logger(out_dir)

    started_at = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    logger.info("Starting Stage 3 CV. Requested outer splits: %d", args.limit_splits)
    logger.info("Input dataset: %s", data_path)

    df = pd.read_csv(data_path)
    df, y_array, complete = validate_input(df, data_path, root)
    X = df[PREDICTORS].copy()
    y = pd.Series(y_array, name=TARGET)
    input_hash = sha256_file(data_path)

    logger.info("Validated participants: %d", len(df))
    logger.info("Complete cases across 11 predictors: %d", int(complete.sum()))
    logger.info("Participants with >=1 missing predictor: %d", int((~complete).sum()))
    logger.info("BP_SYS_MEAN missing: %d | BP_DIA_MEAN missing: %d",
                int(df["BP_SYS_MEAN"].isna().sum()), int(df["BP_DIA_MEAN"].isna().sum()))
    logger.info("Input SHA-256: %s", input_hash)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=args.seed,
    )
    all_outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
    if len(all_outer_splits) != OUTER_SPLITS * OUTER_REPEATS:
        raise RuntimeError(f"Expected 25 outer splits; generated {len(all_outer_splits)}")
    selected_splits = all_outer_splits[:args.limit_splits]

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    fold_summary_rows: list[dict[str, Any]] = []
    models_root = out_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    for split_id, (train_idx, test_idx) in enumerate(selected_splits, start=1):
        split_started = time.perf_counter()
        repeat_no = (split_id - 1) // OUTER_SPLITS + 1
        fold_no = (split_id - 1) % OUTER_SPLITS + 1
        estimator_seed = args.seed + split_id

        train_complete_mask = complete[train_idx]
        test_complete_idx = test_idx[complete[test_idx]]
        if len(test_complete_idx) == 0:
            raise ValueError(f"Split {split_id}: no complete-case test participants")
        y_test_common = y.iloc[test_complete_idx].to_numpy(dtype=int)
        if len(np.unique(y_test_common)) != 2:
            raise ValueError(f"Split {split_id}: common complete-case test subset lacks one class")

        X_outer_train = X.iloc[train_idx].reset_index(drop=True)
        y_outer_train = y.iloc[train_idx].reset_index(drop=True)
        X_test_common = X.iloc[test_complete_idx].reset_index(drop=True)
        y_test_series = y.iloc[test_complete_idx].reset_index(drop=True)
        complete_outer_train = train_complete_mask.astype(bool)

        split_manifest = pd.DataFrame({
            "split_id": split_id,
            "repeat": repeat_no,
            "fold": fold_no,
            "SEQN": df["SEQN"].to_numpy(),
            "role": "train",
            "diabetes_reported": y_array,
            "has_all_11_predictors": complete,
        })
        split_manifest.loc[test_idx, "role"] = "test"
        split_manifest["included_in_common_complete_case_test"] = (
            split_manifest["role"].eq("test") & split_manifest["has_all_11_predictors"]
        )
        split_manifest["used_for_CCA_training"] = (
            split_manifest["role"].eq("train") & split_manifest["has_all_11_predictors"]
        )
        split_manifest["used_for_imputation_training"] = split_manifest["role"].eq("train")
        manifest_frames.append(split_manifest)

        split_model_dir = models_root / f"split_{split_id:03d}"
        split_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "=== SPLIT %02d/25 (repeat=%d, fold=%d) | train=%d (%d complete), "
            "test=%d (%d common complete) ===",
            split_id, repeat_no, fold_no, len(train_idx), int(train_complete_mask.sum()),
            len(test_idx), len(test_complete_idx),
        )

        for model_name in MODELS:
            for method in METHODS:
                method_started = time.perf_counter()
                best_params, best_inner_auc, candidate_rows, tuning_seconds = tune_hyperparameters(
                    X_outer_train=X_outer_train,
                    y_outer_train=y_outer_train,
                    complete_outer_train=complete_outer_train,
                    method=method,
                    model_name=model_name,
                    split_id=split_id,
                    random_state=estimator_seed,
                    logger=logger,
                )
                tuning_rows.extend(candidate_rows)

                if method == "CCA":
                    final_train_mask = complete_outer_train
                else:
                    final_train_mask = np.ones(len(X_outer_train), dtype=bool)
                X_fit = X_outer_train.loc[final_train_mask].copy()
                y_fit = y_outer_train.loc[final_train_mask].copy()
                if y_fit.nunique() != 2:
                    raise ValueError(f"Split {split_id}: final {model_name}/{method} training lacks a class")

                final_model = build_pipeline(
                    method=method,
                    model_name=model_name,
                    params=best_params,
                    random_state=estimator_seed,
                )
                final_fit_started = time.perf_counter()
                final_model.fit(X_fit, y_fit)
                final_fit_seconds = time.perf_counter() - final_fit_started

                probabilities = final_model.predict_proba(X_test_common)[:, 1]
                auc = float(roc_auc_score(y_test_series, probabilities))
                brier = float(brier_score_loss(y_test_series, probabilities))

                model_slug = f"{model_name}__{method.lower().replace('/', '_').replace(' ', '_').replace('=', '')}"
                model_path = split_model_dir / f"{model_slug}.joblib"
                joblib.dump(final_model, model_path, compress=3)

                metrics_rows.append({
                    "split_id": split_id,
                    "repeat": repeat_no,
                    "fold": fold_no,
                    "model": model_name,
                    "method": method,
                    "train_n": int(len(y_fit)),
                    "train_positive_n": int(y_fit.sum()),
                    "test_n": int(len(y_test_series)),
                    "test_positive_n": int(y_test_series.sum()),
                    "roc_auc": auc,
                    "brier_score": brier,
                    "inner_best_roc_auc": best_inner_auc,
                    "best_params_json": json.dumps(json_safe(best_params), sort_keys=True),
                    "tuning_seconds": float(tuning_seconds),
                    "final_fit_seconds": float(final_fit_seconds),
                    "total_model_seconds": float(time.perf_counter() - method_started),
                    "model_artifact": str(model_path.relative_to(out_dir)),
                })

                prediction_frames.append(pd.DataFrame({
                    "split_id": split_id,
                    "repeat": repeat_no,
                    "fold": fold_no,
                    "model": model_name,
                    "method": method,
                    "SEQN": df.loc[test_complete_idx, "SEQN"].to_numpy(),
                    "y_true": y_test_series.to_numpy(dtype=int),
                    "predicted_probability_class_1": probabilities.astype(float),
                }))
                logger.info(
                    "Split %02d | %s | %s | test_n=%d | AUC=%.6f | Brier=%.6f | "
                    "train_n=%d | elapsed_model=%.1fs",
                    split_id, model_name, method, len(y_test_series), auc, brier,
                    len(y_fit), time.perf_counter() - method_started,
                )

        fold_summary_rows.append({
            "split_id": split_id,
            "repeat": repeat_no,
            "fold": fold_no,
            "train_all_n": int(len(train_idx)),
            "train_complete_n": int(train_complete_mask.sum()),
            "test_all_n": int(len(test_idx)),
            "test_common_complete_n": int(len(test_complete_idx)),
            "test_common_positive_n": int(y_test_series.sum()),
            "estimator_seed": int(estimator_seed),
            "background_seed_for_stage3_shap": int(2026 + split_id),
            "split_seconds": float(time.perf_counter() - split_started),
        })
        logger.info("Completed split %02d in %.1f seconds", split_id, time.perf_counter() - split_started)

        # Incremental writes protect completed folds if a later fold is interrupted.
        pd.DataFrame(metrics_rows).to_csv(out_dir / "cv_metrics_final.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            out_dir / "test_predictions_final.csv", index=False
        )
        pd.concat(manifest_frames, ignore_index=True).to_csv(
            out_dir / "split_manifest_final.csv", index=False
        )
        pd.DataFrame(tuning_rows).to_csv(out_dir / "tuning_results_final.csv", index=False)
        pd.DataFrame(fold_summary_rows).to_csv(out_dir / "fold_runtime_summary.csv", index=False)

    total_seconds = time.perf_counter() - run_started
    n_completed = len(selected_splits)
    estimated_25_split_seconds = total_seconds * (OUTER_SPLITS * OUTER_REPEATS / n_completed)

    metrics_df = pd.DataFrame(metrics_rows)
    expected_metric_rows = n_completed * len(MODELS) * len(METHODS)
    if len(metrics_df) != expected_metric_rows:
        raise RuntimeError(f"Expected {expected_metric_rows} metric rows; found {len(metrics_df)}")
    if metrics_df[["roc_auc", "brier_score", "inner_best_roc_auc"]].isna().any().any():
        raise RuntimeError("NaN found in CV metrics")

    metadata = {
        "analysis_stage": "stage3_cv_partial_timing" if n_completed < 25 else "stage3_cv_full",
        "run_started_utc": started_at,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_path": str(data_path),
        "input_dataset_sha256": input_hash,
        "n_participants": int(len(df)),
        "n_complete_cases": int(complete.sum()),
        "n_participants_with_missing_predictors": int((~complete).sum()),
        "blood_pressure_rule": "Arithmetic mean of all valid readings 1-4; at least one valid reading required; otherwise missing.",
        "bp_sys_missing_n": int(df["BP_SYS_MEAN"].isna().sum()),
        "bp_dia_missing_n": int(df["BP_DIA_MEAN"].isna().sum()),
        "predictors_in_original_article_order": PREDICTORS,
        "numeric_predictors": NUMERIC,
        "categorical_predictors": CATEGORICAL,
        "methods": METHODS,
        "models": MODELS,
        "outer_cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=5)",
        "outer_random_state": int(args.seed),
        "inner_cv": "StratifiedKFold(n_splits=3, shuffle=True)",
        "inner_seed_rule": "outer_estimator_seed + 1000 + split_id",
        "inner_validation_rule": "Within each inner fold, all methods are scored on the same complete-case validation participants; CCA trains on complete inner-training rows, imputation methods train on all inner-training rows.",
        "hyperparameter_selection": "Mean inner-fold ROC-AUC; stable first-candidate tie break.",
        "lr_grid": {"regularization": "L2 (l1_ratio=0.0)", "C": LR_C_GRID, "solver": "lbfgs", "max_iter": 5000},
        "rf_grid": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH_GRID,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF_GRID,
        },
        "knn": {"n_neighbors": KNN_NEIGHBORS, "weights": "uniform", "numeric_scaling_before_distance": "StandardScaler fitted within training fold"},
        "common_outer_test": "Only complete-case participants in each outer test fold; identical SEQN set across all six combinations.",
        "survey_weights_used": False,
        "models_saved_for_stage3_shap": True,
        "n_outer_splits_total": OUTER_SPLITS * OUTER_REPEATS,
        "n_outer_splits_completed": n_completed,
        "run_complete": bool(n_completed == 25),
        "total_runtime_seconds": float(total_seconds),
        "estimated_25_split_runtime_seconds_linear": float(estimated_25_split_seconds),
        "estimated_25_split_runtime_hours_linear": float(estimated_25_split_seconds / 3600.0),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    (out_dir / "stage3_cv_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps({
            "n_splits_completed": n_completed,
            "n_metric_rows": len(metrics_df),
            "total_runtime_seconds": total_seconds,
            "estimated_25_split_runtime_hours_linear": estimated_25_split_seconds / 3600.0,
            "output_directory": str(out_dir),
            "input_dataset_sha256": input_hash,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Completed %d/%d outer splits", n_completed, OUTER_SPLITS * OUTER_REPEATS)
    logger.info("Metric rows: %d (expected %d)", len(metrics_df), expected_metric_rows)
    logger.info("Total runtime: %.1f seconds", total_seconds)
    logger.info("Linear estimate for 25 splits: %.2f hours", estimated_25_split_seconds / 3600.0)
    logger.info("Output directory: %s", out_dir)
    logger.info("Run status: %s", "COMPLETE" if n_completed == 25 else "PARTIAL / TIMING RUN")


if __name__ == "__main__":
    main()
