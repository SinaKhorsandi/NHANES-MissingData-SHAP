#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity analysis: compare RF Permutation SHAP at 5 vs 10 cycles.

Uses split 1 only, reuses the exact fitted RF models, common test participants,
100-person background sample, and permutation seed from the primary Stage 3 run.
Primary 5-cycle outputs are read-only and are never overwritten.
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
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import shap
import sklearn

TARGET = "diabetes_reported"
PREDICTORS = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN", "SMQ020", "PAQ605",
]
NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN",
]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605"]
SPLIT_ID = 1
BACKGROUND_N = 100
BASE_BACKGROUND_SEED = 2026
BASE_PERMUTATION_SEED = 2026
BASELINE_CYCLES = 5
SENSITIVITY_CYCLES = 10
ADD_TOLERANCE = 1e-8


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


def dense_float(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_sensitivity_shap_cycles")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalize_scalar_shap(values: Any, base_values: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    base_values = np.asarray(base_values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected RF SHAP values shape: {values.shape}")
    if base_values.ndim == 0:
        base_values = np.full(n_rows, float(base_values), dtype=np.float64)
    else:
        base_values = base_values.reshape(-1)
    if len(base_values) == 1 and n_rows > 1:
        base_values = np.repeat(base_values, n_rows)
    if len(base_values) != n_rows:
        raise ValueError("RF SHAP base-value count does not match test rows")
    return values, base_values


def method_slug(method: str) -> str:
    return method.lower().replace("/", "_").replace(" ", "_").replace("=", "")


def map_original_features(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {feature: [] for feature in PREDICTORS}
    for j, name in enumerate(feature_names):
        if name.startswith("numeric__"):
            original = name.split("__", 1)[1]
            if original not in NUMERIC:
                raise ValueError(f"Unknown transformed numeric feature: {name}")
            groups[original].append(j)
        elif name.startswith("categorical__"):
            suffix = name.split("__", 1)[1]
            matches = [feature for feature in CATEGORICAL
                       if suffix == feature or suffix.startswith(feature + "_")]
            if len(matches) != 1:
                raise ValueError(f"Cannot map transformed categorical feature: {name}")
            groups[matches[0]].append(j)
        else:
            raise ValueError(f"Unexpected transformed feature name: {name}")
    empty = [feature for feature, indices in groups.items() if not indices]
    if empty:
        raise ValueError(f"No transformed columns for original predictors: {empty}")
    return groups


def aggregate_to_original(encoded_values: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    grouped = np.zeros((encoded_values.shape[0], len(PREDICTORS)), dtype=np.float64)
    for j, feature in enumerate(PREDICTORS):
        grouped[:, j] = encoded_values[:, groups[feature]].sum(axis=1)
    if not np.allclose(grouped.sum(axis=1), encoded_values.sum(axis=1), atol=1e-10, rtol=1e-10):
        raise ValueError("One-hot aggregation changed total signed SHAP contribution")
    return grouped


def rank_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    importance = np.mean(np.abs(values), axis=0)
    # Stable sorting makes ties deterministic according to PREDICTORS order.
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty(len(PREDICTORS), dtype=int)
    ranks[order] = np.arange(1, len(PREDICTORS) + 1)
    return importance, ranks


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-out", type=Path, default=root / "output" / "stage3_final")
    parser.add_argument("--data", type=Path, default=root / "output" / "nhanes_examined_clean_with_missing.csv")
    parser.add_argument(
        "--out", type=Path,
        default=root / "output" / "stage3_sensitivity" / "shap_cycles_split01",
        help="New, separate output directory; must not already contain files.",
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    cv_out = resolve(args.cv_out)
    data_path = resolve(args.data)
    out_dir = resolve(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Choose a new --out path.")

    shap_dir = cv_out / "shap"
    values_dir = shap_dir / "values_by_split"
    required = [
        data_path,
        cv_out / "stage3_cv_metadata.json",
        cv_out / "split_manifest_final.csv",
        cv_out / "cv_metrics_final.csv",
        shap_dir / "stage3_shap_metadata.json",
        shap_dir / "additivity_audit_final.csv",
        shap_dir / "shap_background_manifest_final.csv",
        values_dir / "shap_values_split_001.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required primary analysis files:\n- " + "\n- ".join(missing))

    cv_meta = json.loads((cv_out / "stage3_cv_metadata.json").read_text(encoding="utf-8"))
    shap_meta = json.loads((shap_dir / "stage3_shap_metadata.json").read_text(encoding="utf-8"))
    input_hash = sha256_file(data_path)
    if input_hash != cv_meta.get("input_dataset_sha256") or input_hash != shap_meta.get("input_dataset_sha256"):
        raise ValueError("Input dataset SHA-256 does not match primary CV and SHAP metadata")
    if int(shap_meta.get("rf_permutation_cycles", -1)) != BASELINE_CYCLES:
        raise ValueError("Primary SHAP metadata does not record the expected 5-cycle RF baseline")

    df = pd.read_csv(data_path).sort_values("SEQN").reset_index(drop=True)
    needed = ["SEQN", TARGET, *PREDICTORS]
    missing_cols = sorted(set(needed) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input data missing columns: {missing_cols}")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("SEQN must be unique and non-missing")

    manifest = pd.read_csv(cv_out / "split_manifest_final.csv")
    split_manifest = manifest.loc[manifest["split_id"].eq(SPLIT_ID)].copy()
    if split_manifest.empty or split_manifest["SEQN"].duplicated().any():
        raise ValueError("Split 1 manifest is missing or has duplicate SEQN")
    role_by_id = split_manifest.set_index("SEQN")["role"]
    complete_by_id = split_manifest.set_index("SEQN")["has_all_11_predictors"].astype(bool)
    train_ids = set(role_by_id.index[role_by_id.eq("train")])
    test_ids = set(role_by_id.index[role_by_id.eq("test") & complete_by_id])

    baseline_values_path = values_dir / "shap_values_split_001.csv"
    baseline_values = pd.read_csv(baseline_values_path)
    if baseline_values["SEQN"].duplicated().any():
        raise ValueError("Baseline split-1 SHAP file has duplicate SEQN")
    baseline_values = baseline_values.sort_values("SEQN").reset_index(drop=True)
    baseline_ids = set(baseline_values["SEQN"].astype(int))
    if baseline_ids != test_ids:
        raise ValueError("Baseline SHAP participants do not match split-1 common complete-case test set")

    background_manifest = pd.read_csv(shap_dir / "shap_background_manifest_final.csv")
    bg_rows = background_manifest.loc[background_manifest["split_id"].eq(SPLIT_ID)].copy()
    if len(bg_rows) != BACKGROUND_N or bg_rows["SEQN"].nunique() != BACKGROUND_N:
        raise ValueError("Expected exactly 100 unique split-1 background participants")
    background_ids = bg_rows["SEQN"].astype(int).tolist()
    if not set(background_ids).issubset(train_ids):
        raise ValueError("At least one background participant is not in split-1 training data")
    expected_bg_seed = BASE_BACKGROUND_SEED + SPLIT_ID
    if not bg_rows["background_seed"].eq(expected_bg_seed).all():
        raise ValueError("Background seed does not match primary Stage 3 rule")

    baseline_audit = pd.read_csv(shap_dir / "additivity_audit_final.csv")
    baseline_rf_audit = baseline_audit.loc[
        baseline_audit["split_id"].eq(SPLIT_ID) & baseline_audit["model"].eq("RF")
    ].copy()
    if len(baseline_rf_audit) != 3 or not baseline_rf_audit["audit_status"].eq("PASS").all():
        raise ValueError("Primary 5-cycle RF additivity audit for split 1 is incomplete or failed")
    if not baseline_rf_audit["permutation_cycles"].eq(BASELINE_CYCLES).all():
        raise ValueError("Primary split-1 RF audit does not show 5 permutation cycles")

    test = df.loc[df["SEQN"].isin(test_ids)].sort_values("SEQN").reset_index(drop=True)
    background = df.loc[df["SEQN"].isin(background_ids)].sort_values("SEQN").reset_index(drop=True)
    if len(test) != len(baseline_values) or not np.array_equal(test["SEQN"].to_numpy(), baseline_values["SEQN"].to_numpy()):
        raise ValueError("Test participant ordering differs from the saved 5-cycle baseline")
    if background[PREDICTORS].isna().any().any() or test[PREDICTORS].isna().any().any():
        raise ValueError("Background and test participants must be complete cases")

    cv_metrics = pd.read_csv(cv_out / "cv_metrics_final.csv")
    rf_metrics = cv_metrics.loc[cv_metrics["split_id"].eq(SPLIT_ID) & cv_metrics["model"].eq("RF")].copy()
    if len(rf_metrics) != 3:
        raise ValueError("Expected exactly three fitted RF artifacts for split 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "rf_shap_cycles_sensitivity.log")
    logger.info("Starting RF SHAP cycle sensitivity: split 1, %d vs %d cycles", BASELINE_CYCLES, SENSITIVITY_CYCLES)
    logger.info("Primary CV directory: %s", cv_out)
    logger.info("Input SHA-256: %s", input_hash)
    logger.info("Test participants: %d | shared background participants: %d", len(test), len(background))
    logger.info("Background seed: %d | permutation seed: %d", expected_bg_seed, BASE_PERMUTATION_SEED + SPLIT_ID)

    methods = ["CCA", "Median/Mode", "KNN k=5"]
    new_values = pd.DataFrame({
        "split_id": SPLIT_ID,
        "repeat": int(split_manifest["repeat"].iloc[0]),
        "fold": int(split_manifest["fold"].iloc[0]),
        "SEQN": test["SEQN"].to_numpy(),
        "y_true": test[TARGET].astype(int).to_numpy(),
    })
    audit_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    permutation_seed = BASE_PERMUTATION_SEED + SPLIT_ID

    for method in methods:
        metric_row = rf_metrics.loc[rf_metrics["method"].eq(method)]
        if len(metric_row) != 1:
            raise ValueError(f"Expected one split-1 RF artifact for {method}")
        artifact_rel = Path(str(metric_row.iloc[0]["model_artifact"]))
        model_path = cv_out / artifact_rel
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fitted RF model: {model_path}")

        logger.info("RF | %s | computing 10-cycle SHAP", method)
        model_started = time.perf_counter()
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps["preprocess"]
        classifier = pipeline.named_steps["classifier"]
        if list(classifier.classes_) != [0, 1]:
            raise ValueError(f"Unexpected RF class order for {method}: {classifier.classes_}")
        if "n_jobs" in classifier.get_params(deep=False):
            classifier.set_params(n_jobs=1)

        X_bg = dense_float(preprocessor.transform(background[PREDICTORS].copy()))
        X_eval = dense_float(preprocessor.transform(test[PREDICTORS].copy()))
        feature_names = list(preprocessor.get_feature_names_out())
        if X_bg.shape[1] != len(feature_names) or X_eval.shape[1] != len(feature_names):
            raise ValueError("Transformed matrices and feature names do not match")
        groups = map_original_features(feature_names)
        masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))

        def positive_probability(x: Any, clf=classifier) -> np.ndarray:
            return clf.predict_proba(dense_float(x))[:, 1]

        explainer = shap.PermutationExplainer(
            positive_probability,
            masker,
            feature_names=feature_names,
            seed=permutation_seed,
        )
        max_evals = 2 * X_eval.shape[1] * SENSITIVITY_CYCLES + 1
        explanation = explainer(
            X_eval,
            max_evals=max_evals,
            batch_size=256,
            silent=False,
        )
        encoded_values, base_values = normalize_scalar_shap(
            explanation.values, explanation.base_values, len(X_eval)
        )
        if encoded_values.shape != X_eval.shape:
            raise ValueError(f"Unexpected SHAP shape {encoded_values.shape}; expected {X_eval.shape}")
        model_output = classifier.predict_proba(X_eval)[:, 1]
        errors = np.abs(base_values + encoded_values.sum(axis=1) - model_output)
        max_error = float(errors.max())
        mean_error = float(errors.mean())
        if not np.isfinite(max_error) or max_error >= ADD_TOLERANCE:
            raise ValueError(f"{method}: additivity error {max_error:.6g} is not below {ADD_TOLERANCE:.0e}")

        grouped_values = aggregate_to_original(encoded_values, groups)
        if grouped_values.shape != (len(test), len(PREDICTORS)):
            raise ValueError("Grouped SHAP matrix is not n_test x 11 predictors")

        slug = method_slug(method)
        new_values[f"RF__{slug}__base_value"] = base_values
        for j, feature in enumerate(PREDICTORS):
            new_values[f"RF__{slug}__{feature}"] = grouped_values[:, j]
            global_rows.append({
                "split_id": SPLIT_ID,
                "model": "RF",
                "method": method,
                "feature": feature,
                "mean_abs_shap_10_cycles": float(np.mean(np.abs(grouped_values[:, j]))),
                "mean_shap_10_cycles": float(np.mean(grouped_values[:, j])),
                "shap_scale": "positive_class_probability",
            })

        baseline_cols = [f"RF__{slug}__{feature}" for feature in PREDICTORS]
        missing_baseline_cols = [col for col in baseline_cols if col not in baseline_values.columns]
        if missing_baseline_cols:
            raise ValueError(f"Missing 5-cycle baseline columns for {method}: {missing_baseline_cols}")
        baseline_matrix = baseline_values[baseline_cols].to_numpy(dtype=np.float64)
        baseline_audit_row = baseline_rf_audit.loc[baseline_rf_audit["method"].eq(method)]
        if len(baseline_audit_row) != 1:
            raise ValueError(f"Expected one 5-cycle baseline audit row for {method}")
        baseline_max_error = float(
            baseline_audit_row.iloc[0]["max_abs_additivity_error_probability"]
        )
        sensitivity_matrix = grouped_values
        if baseline_matrix.shape != sensitivity_matrix.shape:
            raise ValueError(f"5-cycle and 10-cycle SHAP shapes differ for {method}")

        importance_5 = np.mean(np.abs(baseline_matrix), axis=0)
        importance_10, ranks_10 = rank_features(sensitivity_matrix)
        _, ranks_5 = rank_features(baseline_matrix)
        top5_5 = set(np.argsort(-importance_5, kind="stable")[:5])
        top5_10 = set(np.argsort(-importance_10, kind="stable")[:5])
        union = top5_5 | top5_10
        jaccard = len(top5_5 & top5_10) / len(union) if union else 1.0
        flat_corr = float(spearmanr(baseline_matrix.ravel(), sensitivity_matrix.ravel()).statistic)
        rank_corr = float(pd.Series(ranks_5).corr(pd.Series(ranks_10), method="spearman"))
        delta = sensitivity_matrix - baseline_matrix
        comparison_rows.append({
            "split_id": SPLIT_ID,
            "model": "RF",
            "method": method,
            "test_n": int(len(test)),
            "background_n": int(len(background)),
            "baseline_cycles": BASELINE_CYCLES,
            "sensitivity_cycles": SENSITIVITY_CYCLES,
            "permutation_seed": permutation_seed,
            "spearman_attribution_values_flattened": flat_corr,
            "spearman_global_feature_rank": rank_corr,
            "top5_jaccard": float(jaccard),
            "top5_overlap_n": int(len(top5_5 & top5_10)),
            "mse_shap_5_vs_10": float(np.mean(np.square(delta))),
            "mae_shap_5_vs_10": float(np.mean(np.abs(delta))),
            "rmse_shap_5_vs_10": float(np.sqrt(np.mean(np.square(delta)))),
            "max_abs_shap_difference": float(np.max(np.abs(delta))),
            "max_additivity_error_5_cycles": baseline_max_error,
            "max_additivity_error_10_cycles": max_error,
            "audit_status_5_cycles": "PASS",
            "audit_status_10_cycles": "PASS",
            "shap_scale": "positive_class_probability",
        })
        for j, feature in enumerate(PREDICTORS):
            feature_rows.append({
                "split_id": SPLIT_ID,
                "model": "RF",
                "method": method,
                "feature": feature,
                "mean_abs_shap_5_cycles": float(importance_5[j]),
                "mean_abs_shap_10_cycles": float(importance_10[j]),
                "absolute_change_mean_abs_shap": float(abs(importance_10[j] - importance_5[j])),
                "rank_5_cycles": int(ranks_5[j]),
                "rank_10_cycles": int(ranks_10[j]),
                "rank_change_10_minus_5": int(ranks_10[j] - ranks_5[j]),
            })

        audit_rows.append({
            "split_id": SPLIT_ID,
            "model": "RF",
            "method": method,
            "n_explained": int(len(test)),
            "background_n": int(len(background)),
            "encoded_feature_count": int(X_eval.shape[1]),
            "original_feature_count": int(len(PREDICTORS)),
            "output_scale": "positive_class_probability",
            "permutation_cycles": SENSITIVITY_CYCLES,
            "max_evals": int(max_evals),
            "permutation_seed": permutation_seed,
            "background_seed": expected_bg_seed,
            "max_abs_additivity_error_probability": max_error,
            "mean_abs_additivity_error_probability": mean_error,
            "tolerance": ADD_TOLERANCE,
            "audit_status": "PASS",
            "model_artifact": str(artifact_rel),
        })
        runtime_rows.append({
            "split_id": SPLIT_ID,
            "model": "RF",
            "method": method,
            "cycles": SENSITIVITY_CYCLES,
            "runtime_seconds": float(time.perf_counter() - model_started),
        })
        logger.info("RF | %s | n=%d | max additivity error=%.3g | PASS | %.1fs",
                    method, len(test), max_error, time.perf_counter() - model_started)

    outputs = {
        "rf_shap_values_10_cycles_split_001.csv": new_values,
        "rf_shap_global_importance_10_cycles_split_001.csv": pd.DataFrame(global_rows),
        "rf_shap_additivity_audit_10_cycles.csv": pd.DataFrame(audit_rows),
        "rf_shap_cycles_comparison.csv": pd.DataFrame(comparison_rows),
        "rf_shap_cycles_feature_comparison.csv": pd.DataFrame(feature_rows),
        "rf_shap_cycles_runtime.csv": pd.DataFrame(runtime_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df) != 3 or not audit_df["audit_status"].eq("PASS").all():
        raise RuntimeError("Expected three successful 10-cycle RF additivity audits")
    if len(comparison_df) != 3 or comparison_df.isna().any().any():
        raise RuntimeError("Expected three complete 5-vs-10-cycle comparison rows without NaN")
    if not np.isfinite(comparison_df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Non-finite numeric value found in cycle comparison results")

    runtime_seconds = time.perf_counter() - run_started
    metadata = {
        "analysis_stage": "stage3_sensitivity_rf_permutation_cycles",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_sha256": input_hash,
        "cv_output_directory": str(cv_out),
        "split_id": SPLIT_ID,
        "repeat": int(split_manifest["repeat"].iloc[0]),
        "fold": int(split_manifest["fold"].iloc[0]),
        "model": "RF",
        "methods": methods,
        "baseline_cycles": BASELINE_CYCLES,
        "sensitivity_cycles": SENSITIVITY_CYCLES,
        "test_n": int(len(test)),
        "background_n": int(len(background)),
        "background_seed": expected_bg_seed,
        "permutation_seed": permutation_seed,
        "max_evals_rule": "2 * n_transformed_features * cycles + 1",
        "max_evals": int(2 * len(feature_names) * SENSITIVITY_CYCLES + 1),
        "same_fitted_models_as_primary": True,
        "same_test_participants_as_primary": True,
        "same_background_participants_as_primary": True,
        "same_permutation_seed_as_primary": True,
        "original_predictor_order": PREDICTORS,
        "output_scale": "positive_class_probability",
        "all_additivity_rows_pass": bool(audit_df["audit_status"].eq("PASS").all()),
        "runtime_seconds": float(runtime_seconds),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
        "interpretation_note": "PermutationExplainer local accuracy is checked at both cycle counts; finite-cycle feature allocation remains approximate. The 10-cycle run is a sensitivity analysis for split 1 only, not a replacement for the primary 5-cycle results.",
    }
    (out_dir / "rf_shap_cycles_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Comparison rows: %d (expected 3)", len(comparison_df))
    logger.info("10-cycle additivity rows: %d; all PASS", len(audit_df))
    logger.info("Runtime: %.1f seconds", runtime_seconds)
    logger.info("Output directory: %s", out_dir)
    logger.info("Run status: COMPLETE")


if __name__ == "__main__":
    main()
