#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 SHAP calculations for the NHANES repeated-CV analysis.

Reads fitted pipelines and split manifests produced by stage3_cv.py.
For every selected outer split, computes explanations for all six
(model x missing-data strategy) combinations on the shared complete-case
test participants, using one shared sample of 100 complete-case training
participants across all six combinations in that split.

LR: shap.LinearExplainer on LogisticRegression.decision_function (log-odds).
RF: shap.PermutationExplainer on positive-class probability, 5 cycles.
One-hot encoded attributions are summed algebraically to the 11 original
predictors before saving. Local additivity must be < 1e-8 for every run.

Timing run:
    python stage3_shap.py --cv-out output/stage3_timing_run1 --limit-splits 1

Full run (after timing gate is accepted):
    python stage3_shap.py --cv-out output/stage3_final
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
MODELS = ["LR", "RF"]
METHODS = ["CCA", "Median/Mode", "KNN k=5"]
BACKGROUND_N = 100
BASE_BACKGROUND_SEED = 2026
BASE_PERMUTATION_SEED = 2026
PERMUTATION_CYCLES = 5
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
    logger = logging.getLogger("stage3_shap")
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


def normalize_lr_shap(values: Any, base_values: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    base_values = np.asarray(base_values, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[-1] != 2:
            raise ValueError(f"Unexpected LR SHAP values shape: {values.shape}")
        values = values[:, :, 1]
    elif values.ndim != 2:
        raise ValueError(f"Unexpected LR SHAP values shape: {values.shape}")

    if base_values.ndim == 2:
        if base_values.shape[-1] != 2:
            raise ValueError(f"Unexpected LR base-value shape: {base_values.shape}")
        base_values = base_values[:, 1]
    elif base_values.ndim == 0:
        base_values = np.full(n_rows, float(base_values), dtype=np.float64)
    else:
        base_values = base_values.reshape(-1)
    if len(base_values) == 1 and n_rows > 1:
        base_values = np.repeat(base_values, n_rows)
    if len(base_values) != n_rows:
        raise ValueError("LR SHAP base-value count does not match test rows")
    return values, base_values


def normalize_scalar_shap(values: Any, base_values: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    base_values = np.asarray(base_values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected scalar-output SHAP values shape: {values.shape}")
    if base_values.ndim == 0:
        base_values = np.full(n_rows, float(base_values), dtype=np.float64)
    else:
        base_values = base_values.reshape(-1)
    if len(base_values) == 1 and n_rows > 1:
        base_values = np.repeat(base_values, n_rows)
    if len(base_values) != n_rows:
        raise ValueError("RF SHAP base-value count does not match test rows")
    return values, base_values


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
    if not np.allclose(
        grouped.sum(axis=1), encoded_values.sum(axis=1), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("One-hot aggregation changed the total signed SHAP contribution")
    return grouped


def method_slug(model_name: str, method: str) -> str:
    """Return a stable column-name prefix for one model/imputation pair."""
    method_part = method.lower().replace("/", "_").replace(" ", "_").replace("=", "")
    return f"{model_name}__{method_part}"


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-out", type=Path, default=root / "output" / "stage3_final",
        help="Output directory created by stage3_cv.py",
    )
    parser.add_argument(
        "--data", type=Path, default=root / "output" / "nhanes_examined_clean_with_missing.csv",
        help="Stage 1 examined-cohort CSV",
    )
    parser.add_argument(
        "--limit-splits", type=int, default=None,
        help="Optional number of initial splits for timing/smoke test",
    )
    args = parser.parse_args()

    cv_out = args.cv_out.expanduser()
    if not cv_out.is_absolute():
        cv_out = (root / cv_out).resolve()
    else:
        cv_out = cv_out.resolve()
    data_path = args.data.expanduser()
    if not data_path.is_absolute():
        data_path = (root / data_path).resolve()
    else:
        data_path = data_path.resolve()

    required = [
        data_path,
        cv_out / "stage3_cv_metadata.json",
        cv_out / "split_manifest_final.csv",
        cv_out / "cv_metrics_final.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Stage 3 CV files:\n- " + "\n- ".join(missing))

    cv_meta = json.loads((cv_out / "stage3_cv_metadata.json").read_text(encoding="utf-8"))
    input_hash = sha256_file(data_path)
    if input_hash != cv_meta.get("input_dataset_sha256"):
        raise ValueError("Input dataset SHA-256 does not match stage3_cv metadata")

    df = pd.read_csv(data_path).sort_values("SEQN").reset_index(drop=True)
    required_columns = ["SEQN", TARGET, *PREDICTORS]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input dataset is missing columns: {missing_columns}")
    if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
        raise ValueError("SEQN must be non-missing and unique")
    complete = ~df[PREDICTORS].isna().any(axis=1).to_numpy()

    manifest = pd.read_csv(cv_out / "split_manifest_final.csv")
    metrics = pd.read_csv(cv_out / "cv_metrics_final.csv")
    if not np.array_equal(manifest.groupby("split_id")["SEQN"].nunique().to_numpy(),
                          np.repeat(len(df), manifest["split_id"].nunique())):
        raise ValueError("Each split manifest must contain exactly one row per participant")

    split_ids = sorted(manifest["split_id"].unique().astype(int).tolist())
    if args.limit_splits is not None:
        if args.limit_splits < 1 or args.limit_splits > len(split_ids):
            raise SystemExit(f"--limit-splits must be between 1 and {len(split_ids)}")
        split_ids = split_ids[:args.limit_splits]
    if not split_ids:
        raise ValueError("No outer splits found in split manifest")

    shap_dir = cv_out / "shap"
    if shap_dir.exists() and any(shap_dir.iterdir()):
        raise FileExistsError(
            f"SHAP output directory already exists and is not empty: {shap_dir}\n"
            "Use a fresh Stage 3 CV output directory to avoid overwriting prior results."
        )
    values_dir = shap_dir / "values_by_split"
    values_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(shap_dir / "stage3_shap.log")

    logger.info("Starting Stage 3 SHAP for %d outer splits", len(split_ids))
    logger.info("CV output directory: %s", cv_out)
    logger.info("Input dataset SHA-256: %s", input_hash)
    logger.info("SHAP version: %s", shap.__version__)
    logger.info("LR explainer: LinearExplainer; output=decision_function log-odds")
    logger.info("RF explainer: PermutationExplainer; output=positive-class probability; cycles=%d",
                PERMUTATION_CYCLES)

    global_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    split_runtime_rows: list[dict[str, Any]] = []
    shap_started = time.perf_counter()

    for split_id in split_ids:
        split_started = time.perf_counter()
        split_manifest = manifest.loc[manifest["split_id"].eq(split_id)].copy()
        if split_manifest["SEQN"].duplicated().any():
            raise ValueError(f"Split {split_id}: duplicate SEQN in manifest")
        if not np.array_equal(
            split_manifest.sort_values("SEQN")["SEQN"].to_numpy(), df["SEQN"].to_numpy()
        ):
            raise ValueError(f"Split {split_id}: manifest IDs do not match input dataset")

        role_by_id = split_manifest.set_index("SEQN")["role"]
        complete_by_id = split_manifest.set_index("SEQN")["has_all_11_predictors"].astype(bool)
        train_ids = role_by_id.index[role_by_id.eq("train")]
        test_ids = role_by_id.index[role_by_id.eq("test")]
        common_test_ids = role_by_id.index[
            role_by_id.eq("test") & complete_by_id
        ]

        train_complete = df.loc[df["SEQN"].isin(train_ids) & complete].copy()
        if len(train_complete) < BACKGROUND_N:
            raise ValueError(f"Split {split_id}: fewer than {BACKGROUND_N} complete training rows")
        background_seed = BASE_BACKGROUND_SEED + split_id
        background = train_complete.sample(n=BACKGROUND_N, random_state=background_seed)
        background = background.sort_values("SEQN").reset_index(drop=True)
        test = df.loc[df["SEQN"].isin(common_test_ids)].copy()
        test = test.sort_values("SEQN").reset_index(drop=True)
        if len(test) == 0 or test[TARGET].nunique() != 2:
            raise ValueError(f"Split {split_id}: common complete-case test set is empty or lacks a class")
        if background[PREDICTORS].isna().any().any() or test[PREDICTORS].isna().any().any():
            raise ValueError(f"Split {split_id}: background and common test must be complete")

        X_background_raw = background[PREDICTORS].copy()
        X_test_raw = test[PREDICTORS].copy()
        split_values = pd.DataFrame({
            "split_id": split_id,
            "repeat": int(split_manifest["repeat"].iloc[0]),
            "fold": int(split_manifest["fold"].iloc[0]),
            "SEQN": test["SEQN"].to_numpy(),
            "y_true": test[TARGET].astype(int).to_numpy(),
        })

        logger.info(
            "=== SPLIT %02d/%02d | train complete background pool=%d | background=%d | common test=%d ===",
            split_id, len(split_ids), len(train_complete), BACKGROUND_N, len(test),
        )
        for row in background[["SEQN"]].itertuples(index=False):
            background_rows.append({
                "split_id": split_id,
                "repeat": int(split_manifest["repeat"].iloc[0]),
                "fold": int(split_manifest["fold"].iloc[0]),
                "SEQN": int(row.SEQN),
                "background_seed": background_seed,
                "role": "complete_case_training_background",
            })

        for model_name in MODELS:
            for method in METHODS:
                artifact_rows = metrics.loc[
                    metrics["split_id"].eq(split_id)
                    & metrics["model"].eq(model_name)
                    & metrics["method"].eq(method)
                ]
                if len(artifact_rows) != 1:
                    raise ValueError(
                        f"Split {split_id}: expected one metric/artifact row for {model_name}/{method}; "
                        f"found {len(artifact_rows)}"
                    )
                artifact_rel = Path(str(artifact_rows.iloc[0]["model_artifact"]))
                model_path = cv_out / artifact_rel
                if not model_path.is_file():
                    raise FileNotFoundError(f"Missing fitted model artifact: {model_path}")

                logger.info("Split %02d | %s | %s | computing SHAP", split_id, model_name, method)
                pipeline = joblib.load(model_path)
                preprocessor = pipeline.named_steps["preprocess"]
                classifier = pipeline.named_steps["classifier"]
                X_bg = dense_float(preprocessor.transform(X_background_raw))
                X_eval = dense_float(preprocessor.transform(X_test_raw))
                feature_names = list(preprocessor.get_feature_names_out())
                if X_bg.shape[1] != len(feature_names) or X_eval.shape[1] != len(feature_names):
                    raise ValueError("Transformed feature matrix and feature names do not match")
                groups = map_original_features(feature_names)

                if model_name == "LR":
                    masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
                    explainer = shap.LinearExplainer(classifier, masker)
                    explanation = explainer(X_eval)
                    encoded_values, base_values = normalize_lr_shap(
                        explanation.values, explanation.base_values, len(X_eval)
                    )
                    model_output = np.asarray(classifier.decision_function(X_eval)).reshape(-1)
                    output_scale = "log_odds"
                    error_field = "max_abs_additivity_error_log_odds"
                    mean_error_field = "mean_abs_additivity_error_log_odds"
                    permutation_seed = None
                    max_evals = None
                else:
                    if list(classifier.classes_) != [0, 1]:
                        raise ValueError(
                            f"Split {split_id} {method}: expected RF class order [0, 1], "
                            f"got {classifier.classes_}"
                        )
                    if "n_jobs" in classifier.get_params(deep=False):
                        classifier.set_params(n_jobs=1)
                    masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))

                    def positive_probability(x: Any, clf=classifier) -> np.ndarray:
                        return clf.predict_proba(dense_float(x))[:, 1]

                    permutation_seed = BASE_PERMUTATION_SEED + split_id
                    explainer = shap.PermutationExplainer(
                        positive_probability,
                        masker,
                        feature_names=feature_names,
                        seed=permutation_seed,
                    )
                    max_evals = 2 * X_eval.shape[1] * PERMUTATION_CYCLES + 1
                    explanation = explainer(
                        X_eval,
                        max_evals=max_evals,
                        batch_size=256,
                        silent=False,
                    )
                    encoded_values, base_values = normalize_scalar_shap(
                        explanation.values, explanation.base_values, len(X_eval)
                    )
                    model_output = classifier.predict_proba(X_eval)[:, 1]
                    output_scale = "positive_class_probability"
                    error_field = "max_abs_additivity_error_probability"
                    mean_error_field = "mean_abs_additivity_error_probability"

                if encoded_values.shape != X_eval.shape:
                    raise ValueError(
                        f"Unexpected SHAP shape {encoded_values.shape}; expected {X_eval.shape}"
                    )
                reconstructed = base_values + encoded_values.sum(axis=1)
                errors = np.abs(reconstructed - model_output)
                max_error = float(errors.max())
                mean_error = float(errors.mean())
                if not np.isfinite(max_error) or max_error >= ADD_TOLERANCE:
                    raise ValueError(
                        f"Split {split_id} {model_name}/{method}: additivity error "
                        f"{max_error:.6g} is not below {ADD_TOLERANCE:.0e}"
                    )

                grouped_values = aggregate_to_original(encoded_values, groups)
                if grouped_values.shape != (len(test), len(PREDICTORS)):
                    raise ValueError("Grouped SHAP matrix is not n_test x 11 predictors")
                split_values[f"{method_slug(model_name, method)}__base_value"] = base_values
                for j, feature in enumerate(PREDICTORS):
                    split_values[f"{method_slug(model_name, method)}__{feature}"] = grouped_values[:, j]
                    global_rows.append({
                        "split_id": split_id,
                        "repeat": int(split_manifest["repeat"].iloc[0]),
                        "fold": int(split_manifest["fold"].iloc[0]),
                        "model": model_name,
                        "method": method,
                        "feature": feature,
                        "mean_abs_shap": float(np.mean(np.abs(grouped_values[:, j]))),
                        "mean_shap": float(np.mean(grouped_values[:, j])),
                        "rank": None,
                        "shap_scale": output_scale,
                    })

                audit_row = {
                    "split_id": split_id,
                    "repeat": int(split_manifest["repeat"].iloc[0]),
                    "fold": int(split_manifest["fold"].iloc[0]),
                    "model": model_name,
                    "method": method,
                    "n_explained": int(len(test)),
                    "background_n": int(len(background)),
                    "encoded_feature_count": int(X_eval.shape[1]),
                    "original_feature_count": int(len(PREDICTORS)),
                    "output_scale": output_scale,
                    "max_evals": max_evals,
                    "permutation_cycles": PERMUTATION_CYCLES if model_name == "RF" else None,
                    "permutation_seed": permutation_seed,
                    "background_seed": background_seed,
                    error_field: max_error,
                    mean_error_field: mean_error,
                    "tolerance": ADD_TOLERANCE,
                    "audit_status": "PASS",
                    "model_artifact": str(artifact_rel),
                }
                audit_rows.append(audit_row)
                logger.info(
                    "Split %02d | %s | %s | n=%d | max additivity error=%.3g | PASS",
                    split_id, model_name, method, len(test), max_error,
                )

        split_values_path = values_dir / f"shap_values_split_{split_id:03d}.csv"
        split_values.to_csv(split_values_path, index=False)
        split_runtime_rows.append({
            "split_id": split_id,
            "repeat": int(split_manifest["repeat"].iloc[0]),
            "fold": int(split_manifest["fold"].iloc[0]),
            "n_test": int(len(test)),
            "background_n": int(len(background)),
            "split_shap_seconds": float(time.perf_counter() - split_started),
            "shap_values_file": str(split_values_path.relative_to(shap_dir)),
        })

        # Incremental files allow completed splits to be audited if a later split fails.
        global_df = pd.DataFrame(global_rows)
        global_df["rank"] = global_df.groupby(["split_id", "model", "method"])["mean_abs_shap"].rank(
            method="min", ascending=False
        ).astype(int)
        global_df.to_csv(shap_dir / "shap_global_importance_final.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(shap_dir / "additivity_audit_final.csv", index=False)
        pd.DataFrame(background_rows).to_csv(shap_dir / "shap_background_manifest_final.csv", index=False)
        pd.DataFrame(split_runtime_rows).to_csv(shap_dir / "shap_runtime_summary.csv", index=False)
        logger.info("Completed SHAP split %02d in %.1f seconds", split_id, time.perf_counter() - split_started)

    total_seconds = time.perf_counter() - shap_started
    completed = len(split_ids)
    estimate_shap_25 = total_seconds * (25 / completed)
    cv_seconds_per_split = float(cv_meta.get("total_runtime_seconds", 0.0)) / max(
        int(cv_meta.get("n_outer_splits_completed", 1)), 1
    )
    estimate_cv_25 = cv_seconds_per_split * 25
    combined_estimate_hours = (estimate_shap_25 + estimate_cv_25) / 3600.0

    final_global = pd.read_csv(shap_dir / "shap_global_importance_final.csv")
    final_audit = pd.read_csv(shap_dir / "additivity_audit_final.csv")
    expected_audit_rows = completed * len(MODELS) * len(METHODS)
    if len(final_audit) != expected_audit_rows:
        raise RuntimeError(
            f"Expected {expected_audit_rows} additivity rows; found {len(final_audit)}"
        )
    if not final_audit["audit_status"].eq("PASS").all():
        raise RuntimeError("At least one SHAP additivity audit did not PASS")
    if final_global.duplicated(["split_id", "model", "method", "feature"]).any():
        raise RuntimeError("Duplicate global importance rows found")

    metadata = {
        "analysis_stage": "stage3_shap_partial_timing" if completed < 25 else "stage3_shap_full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_path": str(data_path),
        "input_dataset_sha256": input_hash,
        "cv_output_directory": str(cv_out),
        "n_participants": int(len(df)),
        "n_splits_completed": int(completed),
        "n_shap_model_runs": int(len(final_audit)),
        "models": MODELS,
        "methods": METHODS,
        "original_predictor_order": PREDICTORS,
        "background_n": BACKGROUND_N,
        "background_seed_rule": "2026 + split_id; one shared sample across all six model/method combinations within each split",
        "common_test_rule": "complete-case participants in each outer test fold; same SEQN set across all six combinations",
        "lr_explainer": "shap.LinearExplainer on LogisticRegression.decision_function",
        "lr_output_scale": "log_odds",
        "rf_explainer": "shap.PermutationExplainer on RandomForestClassifier.predict_proba[:, 1]",
        "rf_output_scale": "positive_class_probability",
        "rf_permutation_cycles": PERMUTATION_CYCLES,
        "rf_permutation_seed_rule": "2026 + split_id; shared across methods within a split",
        "rf_max_evals_rule": "2 * n_transformed_features * permutation_cycles + 1",
        "categorical_shap_aggregation": "Signed one-hot column SHAP values are algebraically summed per participant to the original 11 predictors before global importance and stability metrics.",
        "local_additivity_tolerance": ADD_TOLERANCE,
        "all_additivity_rows_pass": bool(final_audit["audit_status"].eq("PASS").all()),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
        "runtime_seconds": float(total_seconds),
        "estimated_shap_25_split_hours_linear": float(estimate_shap_25 / 3600.0),
        "estimated_cv_25_split_hours_linear": float(estimate_cv_25 / 3600.0),
        "estimated_cv_plus_shap_25_split_hours_linear": float(combined_estimate_hours),
        "caveats": [
            "PermutationExplainer has exact local accuracy by permutation construction, but finite-cycle feature attribution remains an approximation for models with interactions.",
            "SHAP MSE values must only be compared within the same model because LR uses log-odds and RF uses probability scale.",
            "The grouped one-hot attributions follow the advisor-specified signed-sum aggregation to 11 original predictors.",
        ],
    }
    (shap_dir / "stage3_shap_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Completed SHAP splits: %d", completed)
    logger.info("Additivity audit rows: %d; all PASS", len(final_audit))
    logger.info("SHAP runtime: %.1f seconds", total_seconds)
    logger.info("Estimated SHAP runtime for 25 splits: %.2f hours", estimate_shap_25 / 3600.0)
    logger.info("Estimated CV runtime for 25 splits: %.2f hours", estimate_cv_25 / 3600.0)
    logger.info("Estimated combined CV + SHAP runtime: %.2f hours", combined_estimate_hours)
    logger.info("SHAP outputs: %s", shap_dir)
    logger.info("Run status: %s", "COMPLETE" if completed == 25 else "PARTIAL / TIMING RUN")


if __name__ == "__main__":
    main()
