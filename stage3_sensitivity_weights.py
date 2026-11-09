#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES Stage 3 sensitivity analysis using WTMEC2YR sample weights.

Runs the same 25 outer splits and six model/imputation configurations as the
primary Stage 3 analysis. Primary hyperparameters are held fixed to isolate
the sensitivity to weighted classifier fitting.

Important design details:
- WTMEC2YR is merged from DEMO_J.xpt by SEQN.
- CCA is fitted on complete outer-training participants only.
- Imputation methods are fitted on all outer-training participants.
- Preprocessing is fitted without sample weights; sample_weight is passed to
  the final classifier step only.
- Training weights are rescaled to mean 1 within each fitted training cohort
  to preserve relative weights without changing the overall loss scale.
- Both unweighted and WTMEC2YR-weighted test metrics are reported.
- Primary model predictions are re-scored with WTMEC2YR on the same test
  participants, allowing a like-for-like weighted-evaluation comparison.
- This is a sensitivity analysis, not full NHANES complex-survey inference.
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
import sklearn
import shap
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import stage3_cv as primary


WEIGHT = "WTMEC2YR"


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
    logger = logging.getLogger("stage3_weight_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        out_dir / "weight_sensitivity.log", encoding="utf-8"
    )
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
            "Choose a new output directory; existing results will not be overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)


def weighted_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    return (
        float(roc_auc_score(y_true, probabilities, sample_weight=weights)),
        float(brier_score_loss(y_true, probabilities, sample_weight=weights)),
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-out",
        type=Path,
        default=root / "output" / "stage3_final",
        help="Completed primary Stage 3 CV output directory.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "output" / "nhanes_examined_clean_with_missing.csv",
        help="Stage 1 examined-cohort CSV.",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=root / "raw_xpt" / "DEMO_J.XPT",
        help="NHANES 2017-2018 DEMO_J XPT file containing WTMEC2YR.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "output" / "stage3_sensitivity" / "weights_pilot",
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--limit-splits",
        type=int,
        default=25,
        help="Run first N primary outer splits (1..25); use 1 for pilot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=primary.BASE_SEED,
        help="Must match the primary outer-CV seed (default 2026).",
    )
    args = parser.parse_args()

    if not (1 <= args.limit_splits <= 25):
        parser.error("--limit-splits must be between 1 and 25")
    if args.seed != primary.BASE_SEED:
        parser.error(
            f"Primary Stage 3 seed is fixed at {primary.BASE_SEED}; "
            "use the same seed to preserve split assignments."
        )

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

    demo_path = args.demo.expanduser()
    if not demo_path.is_absolute():
        demo_path = (root / demo_path).resolve()
    else:
        demo_path = demo_path.resolve()

    out_dir = args.out.expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    for required_path in [cv_out, data_path, demo_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    required_cv_files = [
        cv_out / "cv_metrics_final.csv",
        cv_out / "test_predictions_final.csv",
        cv_out / "split_manifest_final.csv",
        cv_out / "stage3_cv_metadata.json",
    ]
    missing = [str(p) for p in required_cv_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Primary CV outputs are incomplete: {missing}")

    prepare_output_dir(out_dir)
    logger = make_logger(out_dir)

    run_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Starting WTMEC2YR sensitivity for %d outer splits",
        args.limit_splits,
    )
    logger.info("Primary CV directory: %s", cv_out)
    logger.info("Input dataset: %s", data_path)
    logger.info("DEMO_J file: %s", demo_path)

    # Load and validate the same participant cohort and row ordering as Stage 3.
    df_raw = pd.read_csv(data_path)
    df, y_array, complete = primary.validate_input(df_raw, data_path, root)
    df = df.sort_values("SEQN").reset_index(drop=True)
    input_hash = sha256_file(data_path)

    # Read the official MEC examination weight from DEMO_J and merge by SEQN.
    demo = pd.read_sas(
        demo_path,
        format="xport",
        encoding="infer",
    )
    needed_demo_cols = ["SEQN", "RIDSTATR", WEIGHT]
    missing_demo_cols = sorted(set(needed_demo_cols) - set(demo.columns))
    if missing_demo_cols:
        raise ValueError(f"DEMO_J is missing required columns: {missing_demo_cols}")
    demo_weight = demo[needed_demo_cols].copy().rename(
        columns={"RIDSTATR": "RIDSTATR_demo"}
    )
    if demo_weight["SEQN"].isna().any() or demo_weight["SEQN"].duplicated().any():
        raise ValueError("DEMO_J SEQN must be non-missing and unique")
    demo_weight[WEIGHT] = pd.to_numeric(demo_weight[WEIGHT], errors="coerce")

    # The Stage 1 analytic CSV may already contain WTMEC2YR. Preserve that
    # column temporarily so the authoritative DEMO_J value can be merged
    # without pandas creating WTMEC2YR_x / WTMEC2YR_y suffixes.
    stage1_weight_col = f"{WEIGHT}_stage1"
    had_stage1_weight = WEIGHT in df.columns
    if had_stage1_weight:
        if stage1_weight_col in df.columns:
            raise ValueError(
                f"Unexpected pre-existing column {stage1_weight_col}; "
                "resolve the duplicate weight columns before proceeding."
            )
        df = df.rename(columns={WEIGHT: stage1_weight_col})

    df = df.merge(
        demo_weight,
        on="SEQN",
        how="left",
        validate="one_to_one",
    ).sort_values("SEQN").reset_index(drop=True)

    # If Stage 1 already carried a weight, verify it against DEMO_J rather
    # than silently trusting or overwriting it.
    if had_stage1_weight:
        stage1_weights = pd.to_numeric(
            df[stage1_weight_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        demo_weights = pd.to_numeric(
            df[WEIGHT], errors="coerce"
        ).to_numpy(dtype=np.float64)
        comparable = np.isfinite(stage1_weights) & np.isfinite(demo_weights)
        mismatches = comparable & ~np.isclose(
            stage1_weights, demo_weights, rtol=1e-10, atol=1e-12
        )
        if mismatches.any():
            mismatch_n = int(mismatches.sum())
            raise ValueError(
                f"Stage 1 {WEIGHT} differs from authoritative DEMO_J values "
                f"for {mismatch_n} participants."
            )
        logger.info(
            "Pre-existing Stage 1 %s checked against DEMO_J: %d comparable values, all agree.",
            WEIGHT,
            int(comparable.sum()),
        )
        df = df.drop(columns=[stage1_weight_col])

    if len(df) != primary.EXPECTED_N:
        raise ValueError(f"Expected {primary.EXPECTED_N} participants after weight merge")
    if not df["RIDSTATR"].eq(2).all() or not df["RIDSTATR_demo"].eq(2).all():
        raise ValueError("The analytic cohort must have RIDSTATR=2 in both Stage 1 and DEMO_J")
    if df[WEIGHT].isna().any():
        raise ValueError(f"{WEIGHT} contains missing values in the examined cohort")
    if not np.isfinite(df[WEIGHT].to_numpy(dtype=float)).all():
        raise ValueError(f"{WEIGHT} contains non-finite values")
    if not (df[WEIGHT] > 0).all():
        bad_n = int((df[WEIGHT] <= 0).sum())
        raise ValueError(f"{WEIGHT} must be positive for all examined participants; bad_n={bad_n}")

    weights_all = df[WEIGHT].to_numpy(dtype=np.float64)
    X = df[primary.PREDICTORS].copy()
    y = pd.Series(y_array, name=primary.TARGET)

    # Confirm the primary analysis used the same input file hash.
    primary_metadata = json.loads(
        (cv_out / "stage3_cv_metadata.json").read_text(encoding="utf-8")
    )
    if primary_metadata.get("input_dataset_sha256") != input_hash:
        raise ValueError("Input dataset SHA-256 differs from the primary CV run")
    if primary_metadata.get("survey_weights_used") is not False:
        raise ValueError("Primary analysis metadata does not indicate an unweighted analysis")

    # Load the primary hyperparameters, metrics, predictions, and split manifest.
    primary_metrics = pd.read_csv(cv_out / "cv_metrics_final.csv")
    primary_predictions = pd.read_csv(cv_out / "test_predictions_final.csv")
    primary_manifest = pd.read_csv(cv_out / "split_manifest_final.csv")

    expected_primary_rows = 25 * len(primary.MODELS) * len(primary.METHODS)
    if len(primary_metrics) != expected_primary_rows:
        raise ValueError(
            f"Expected {expected_primary_rows} primary metric rows; found {len(primary_metrics)}"
        )
    if primary_metrics.duplicated(["split_id", "model", "method"]).any():
        raise ValueError("Primary metrics contain duplicate split/model/method rows")
    if primary_predictions.duplicated(
        ["split_id", "model", "method", "SEQN"]
    ).any():
        raise ValueError("Primary predictions contain duplicate participant rows")

    metric_lookup = primary_metrics.set_index(
        ["split_id", "model", "method"]
    )
    if not metric_lookup.index.is_unique:
        raise ValueError("Primary metrics contain duplicate split/model/method rows")
    prediction_lookup = {
        key: group.sort_values("SEQN").reset_index(drop=True)
        for key, group in primary_predictions.groupby(
            ["split_id", "model", "method"], sort=False
        )
    }

    outer_cv = RepeatedStratifiedKFold(
        n_splits=primary.OUTER_SPLITS,
        n_repeats=primary.OUTER_REPEATS,
        random_state=args.seed,
    )
    all_splits = list(outer_cv.split(np.zeros(len(y)), y))
    if len(all_splits) != 25:
        raise RuntimeError(f"Expected 25 generated splits; found {len(all_splits)}")

    # Independently verify generated splits against the saved primary manifest.
    manifest_groups = {
        int(split_id): group
        for split_id, group in primary_manifest.groupby("split_id", sort=False)
    }
    if set(manifest_groups) != set(range(1, 26)):
        raise ValueError("Primary split manifest does not contain exactly split IDs 1..25")

    for split_id, (train_idx, test_idx) in enumerate(all_splits, start=1):
        saved = manifest_groups[split_id]
        saved_train = set(saved.loc[saved["role"].eq("train"), "SEQN"].astype(int))
        saved_test = set(saved.loc[saved["role"].eq("test"), "SEQN"].astype(int))
        generated_train = set(df.iloc[train_idx]["SEQN"].astype(int))
        generated_test = set(df.iloc[test_idx]["SEQN"].astype(int))
        if saved_train != generated_train or saved_test != generated_test:
            raise ValueError(f"Generated split {split_id} does not match primary split manifest")

    logger.info("Validated participants: %d", len(df))
    logger.info("Complete cases: %d", int(complete.sum()))
    logger.info("WTMEC2YR: min=%.6f | median=%.6f | mean=%.6f | max=%.6f",
                float(np.min(weights_all)), float(np.median(weights_all)),
                float(np.mean(weights_all)), float(np.max(weights_all)))
    logger.info("All examined participants have positive WTMEC2YR: PASS")
    logger.info("Primary CV SHA-256: %s", input_hash)
    logger.info("Primary splits and hyperparameters will be reused; no re-tuning.")
    logger.info(
        "sample_weight is passed to classifier only; preprocessing remains fitted unweighted."
    )
    logger.info(
        "Training weights are normalized to mean 1 within each fitted training cohort."
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []

    selected_splits = all_splits[:args.limit_splits]

    for split_id, (train_idx, test_idx) in enumerate(selected_splits, start=1):
        split_started = time.perf_counter()
        repeat_no = (split_id - 1) // primary.OUTER_SPLITS + 1
        fold_no = (split_id - 1) % primary.OUTER_SPLITS + 1
        estimator_seed = args.seed + split_id

        train_complete = complete[train_idx]
        test_complete_idx = test_idx[complete[test_idx]]
        test_complete_idx = np.array(
            sorted(test_complete_idx, key=lambda idx: int(df.iloc[idx]["SEQN"])),
            dtype=int,
        )
        if len(test_complete_idx) == 0:
            raise ValueError(f"Split {split_id}: no common complete-case test participants")

        X_outer_train = X.iloc[train_idx].reset_index(drop=True)
        y_outer_train = y.iloc[train_idx].reset_index(drop=True)
        w_outer_train = weights_all[train_idx]
        X_test = X.iloc[test_complete_idx].reset_index(drop=True)
        y_test = y.iloc[test_complete_idx].to_numpy(dtype=int)
        w_test = weights_all[test_complete_idx]
        test_seqn = df.iloc[test_complete_idx]["SEQN"].to_numpy(dtype=int)

        split_manifest = pd.DataFrame({
            "split_id": split_id,
            "repeat": repeat_no,
            "fold": fold_no,
            "SEQN": np.concatenate([
                df.iloc[train_idx]["SEQN"].to_numpy(),
                df.iloc[test_idx]["SEQN"].to_numpy(),
            ]),
            "role": np.concatenate([
                np.repeat("train", len(train_idx)),
                np.repeat("test", len(test_idx)),
            ]),
        })
        split_manifest = split_manifest.merge(
            df[["SEQN", WEIGHT, primary.TARGET]].assign(
                has_all_11_predictors=complete
            ),
            on="SEQN",
            how="left",
            validate="many_to_one",
        )
        split_manifest["included_in_common_complete_case_test"] = (
            split_manifest["role"].eq("test")
            & split_manifest["has_all_11_predictors"]
        )
        split_manifest["used_for_CCA_training"] = (
            split_manifest["role"].eq("train")
            & split_manifest["has_all_11_predictors"]
        )
        split_manifest["used_for_imputation_training"] = split_manifest["role"].eq("train")
        manifest_frames.append(split_manifest)

        logger.info(
            "=== SPLIT %02d/25 repeat=%d fold=%d | train=%d (%d complete) "
            "test=%d common_complete=%d ===",
            split_id, repeat_no, fold_no, len(train_idx),
            int(train_complete.sum()), len(test_idx), len(test_complete_idx),
        )

        for model_name in primary.MODELS:
            for method in primary.METHODS:
                method_started = time.perf_counter()
                key = (split_id, model_name, method)
                if key not in metric_lookup.index:
                    raise ValueError(f"Missing primary metric row: {key}")
                primary_row = metric_lookup.loc[key]
                best_params = json.loads(primary_row["best_params_json"])

                if method == "CCA":
                    fit_mask = train_complete
                else:
                    fit_mask = np.ones(len(train_idx), dtype=bool)

                X_fit = X_outer_train.loc[fit_mask].copy()
                y_fit = y_outer_train.loc[fit_mask].copy()
                raw_fit_weights = w_outer_train[fit_mask].astype(np.float64)
                if len(y_fit) == 0 or y_fit.nunique() != 2:
                    raise ValueError(f"Split {split_id} {model_name}/{method}: invalid training classes")
                if not np.isfinite(raw_fit_weights).all() or not (raw_fit_weights > 0).all():
                    raise ValueError(f"Split {split_id} {model_name}/{method}: invalid training weights")

                # Normalize to mean 1 so the absolute scale of NHANES weights
                # does not inadvertently change regularization strength.
                normalized_fit_weights = raw_fit_weights / float(raw_fit_weights.mean())

                estimator = primary.build_pipeline(
                    method=method,
                    model_name=model_name,
                    params=best_params,
                    random_state=estimator_seed,
                )
                estimator.fit(
                    X_fit,
                    y_fit,
                    classifier__sample_weight=normalized_fit_weights,
                )
                probabilities = estimator.predict_proba(X_test)[:, 1]

                auc_unweighted = float(roc_auc_score(y_test, probabilities))
                brier_unweighted = float(brier_score_loss(y_test, probabilities))
                auc_weighted, brier_weighted = weighted_metrics(
                    y_test, probabilities, w_test
                )

                baseline_predictions = prediction_lookup.get(key)
                if baseline_predictions is None:
                    raise ValueError(f"Missing primary prediction rows: {key}")
                if not np.array_equal(
                    baseline_predictions["SEQN"].astype(int).to_numpy(),
                    test_seqn,
                ):
                    raise ValueError(
                        f"Primary and weighted test SEQN differ for split/model/method {key}"
                    )
                if not np.array_equal(
                    baseline_predictions["y_true"].astype(int).to_numpy(),
                    y_test,
                ):
                    raise ValueError(f"Primary and weighted y_true differ for {key}")

                baseline_prob = baseline_predictions[
                    "predicted_probability_class_1"
                ].to_numpy(dtype=np.float64)
                primary_auc_unweighted = float(roc_auc_score(y_test, baseline_prob))
                primary_brier_unweighted = float(brier_score_loss(y_test, baseline_prob))
                primary_auc_weighted, primary_brier_weighted = weighted_metrics(
                    y_test, baseline_prob, w_test
                )

                metric_rows.append({
                    "split_id": split_id,
                    "repeat": repeat_no,
                    "fold": fold_no,
                    "model": model_name,
                    "method": method,
                    "train_n": int(len(y_fit)),
                    "train_positive_n": int(y_fit.sum()),
                    "train_weight_sum_raw": float(raw_fit_weights.sum()),
                    "train_weight_mean_raw": float(raw_fit_weights.mean()),
                    "train_weight_min_raw": float(raw_fit_weights.min()),
                    "train_weight_max_raw": float(raw_fit_weights.max()),
                    "train_weight_sum_normalized": float(normalized_fit_weights.sum()),
                    "test_n": int(len(y_test)),
                    "test_positive_n": int(y_test.sum()),
                    "test_weight_sum_raw": float(w_test.sum()),
                    "primary_unweighted_roc_auc": primary_auc_unweighted,
                    "primary_unweighted_brier_score": primary_brier_unweighted,
                    "weighted_fit_unweighted_eval_roc_auc": auc_unweighted,
                    "weighted_fit_unweighted_eval_brier_score": brier_unweighted,
                    "primary_weighted_eval_roc_auc": primary_auc_weighted,
                    "primary_weighted_eval_brier_score": primary_brier_weighted,
                    "weighted_fit_weighted_eval_roc_auc": auc_weighted,
                    "weighted_fit_weighted_eval_brier_score": brier_weighted,
                    "delta_unweighted_eval_roc_auc": auc_unweighted - primary_auc_unweighted,
                    "delta_unweighted_eval_brier_score": brier_unweighted - primary_brier_unweighted,
                    "delta_weighted_eval_roc_auc": auc_weighted - primary_auc_weighted,
                    "delta_weighted_eval_brier_score": brier_weighted - primary_brier_weighted,
                    "best_params_json_from_primary": json.dumps(best_params, sort_keys=True),
                    "preprocessing_weighted": False,
                    "classifier_fit_weighted": True,
                    "training_weight_normalization": "divide by training-cohort mean; mean=1",
                    "elapsed_seconds": float(time.perf_counter() - method_started),
                })

                prediction_frames.append(pd.DataFrame({
                    "split_id": split_id,
                    "repeat": repeat_no,
                    "fold": fold_no,
                    "model": model_name,
                    "method": method,
                    "SEQN": test_seqn,
                    "y_true": y_test,
                    "WTMEC2YR": w_test,
                    "primary_predicted_probability": baseline_prob,
                    "weighted_fit_predicted_probability": probabilities.astype(float),
                }))

                logger.info(
                    "Split %02d | %s | %s | train_n=%d | test_n=%d | "
                    "unweighted AUC=%.6f | weighted AUC=%.6f | "
                    "unweighted Brier=%.6f | weighted Brier=%.6f",
                    split_id, model_name, method, len(y_fit), len(y_test),
                    auc_unweighted, auc_weighted, brier_unweighted, brier_weighted,
                )

        runtime_rows.append({
            "split_id": split_id,
            "repeat": repeat_no,
            "fold": fold_no,
            "split_runtime_seconds": float(time.perf_counter() - split_started),
        })

        # Incremental writes preserve completed splits if execution is interrupted.
        pd.DataFrame(metric_rows).to_csv(
            out_dir / "weight_sensitivity_metrics_by_split.csv", index=False
        )
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            out_dir / "weight_sensitivity_predictions.csv", index=False
        )
        pd.concat(manifest_frames, ignore_index=True).to_csv(
            out_dir / "weight_sensitivity_training_manifest.csv", index=False
        )
        pd.DataFrame(runtime_rows).to_csv(
            out_dir / "weight_sensitivity_runtime.csv", index=False
        )
        logger.info(
            "Completed weighted sensitivity split %02d in %.1fs",
            split_id, time.perf_counter() - split_started,
        )

    metrics_df = pd.DataFrame(metric_rows)
    expected_rows = args.limit_splits * len(primary.MODELS) * len(primary.METHODS)
    if len(metrics_df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} weighted metric rows; found {len(metrics_df)}")

    metric_columns = [
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
    if metrics_df[metric_columns].isna().any().any():
        raise RuntimeError("NaN found in weighted sensitivity metrics")
    if not np.isfinite(metrics_df[metric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite values found in weighted sensitivity metrics")

    summary_rows: list[dict[str, Any]] = []
    for (model_name, method), group in metrics_df.groupby(["model", "method"], sort=False):
        row: dict[str, Any] = {
            "model": model_name,
            "method": method,
            "n_splits": int(len(group)),
        }
        for col in metric_columns:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"sd_{col}"] = float(group[col].std(ddof=1)) if len(group) > 1 else float("nan")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "weight_sensitivity_summary.csv", index=False)

    # Overall analytic-cohort weight distribution.
    weight_summary = pd.DataFrame([{
        "n_participants": int(len(weights_all)),
        "weight_variable": WEIGHT,
        "minimum": float(np.min(weights_all)),
        "p25": float(np.quantile(weights_all, 0.25)),
        "median": float(np.median(weights_all)),
        "mean": float(np.mean(weights_all)),
        "p75": float(np.quantile(weights_all, 0.75)),
        "maximum": float(np.max(weights_all)),
        "sum": float(np.sum(weights_all)),
        "n_missing": int(pd.isna(weights_all).sum()),
        "n_nonpositive": int((weights_all <= 0).sum()),
    }])
    weight_summary.to_csv(out_dir / "wtmec2yr_distribution_summary.csv", index=False)

    runtime_df = pd.DataFrame(runtime_rows)
    total_seconds = time.perf_counter() - run_started
    metadata = {
        "analysis_stage": "stage3_sensitivity_WTMEC2YR",
        "run_started_utc": started_utc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_sha256": input_hash,
        "input_dataset_path": str(data_path),
        "demo_j_path": str(demo_path),
        "primary_cv_directory": str(cv_out),
        "weight_variable": WEIGHT,
        "weight_definition": "Full sample 2-year MEC examination weight from NHANES 2017-2018 DEMO_J.",
        "preexisting_stage1_weight_validated_against_demo_j": bool(had_stage1_weight),
        "n_participants": int(len(df)),
        "n_complete_cases": int(complete.sum()),
        "n_outer_splits_requested": int(args.limit_splits),
        "n_outer_splits_completed": int(metrics_df["split_id"].nunique()),
        "n_metric_rows": int(len(metrics_df)),
        "models": primary.MODELS,
        "methods": primary.METHODS,
        "outer_cv": "Same RepeatedStratifiedKFold 5-fold x 5-repeat splits as primary Stage 3",
        "outer_seed": int(args.seed),
        "primary_hyperparameters_reused": True,
        "retuning_performed": False,
        "training_rule": "CCA fits complete-case outer-training rows; Median/Mode and KNN fit all outer-training rows.",
        "sample_weight_rule": "Raw WTMEC2YR divided by its mean within each model/method training cohort; relative weights preserved and mean normalized weight equals 1.",
        "preprocessing_weighted": False,
        "classifier_fit_weighted": True,
        "test_metrics": [
            "unweighted ROC-AUC and Brier",
            "WTMEC2YR-weighted ROC-AUC and Brier",
        ],
        "primary_predictions_rescored_with_WTMEC2YR": True,
        "complex_survey_design_variance_estimation": False,
        "interpretation_limit": "Sensitivity analysis only; sample_weight in scikit-learn does not implement full NHANES strata/PSU design-based population inference.",
        "weight_distribution": json_safe(weight_summary.iloc[0].to_dict()),
        "n_rows_expected": expected_rows,
        "run_complete": bool(args.limit_splits == 25),
        "total_runtime_seconds": float(total_seconds),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
    }
    (out_dir / "weight_sensitivity_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Weighted sensitivity metric rows: %d (expected %d)", len(metrics_df), expected_rows)
    logger.info("Summary rows: %d (expected 6)", len(summary_df))
    logger.info("Total runtime: %.1f seconds", total_seconds)
    logger.info("Output directory: %s", out_dir)
    logger.info(
        "Run status: %s",
        "COMPLETE" if args.limit_splits == 25 else "PARTIAL / PILOT",
    )


if __name__ == "__main__":
    main()
