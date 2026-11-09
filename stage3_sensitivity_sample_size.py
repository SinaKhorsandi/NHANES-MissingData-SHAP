#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 sensitivity analysis: match imputation-model training N to CCA.

For each outer split, the Median/Mode and KNN training cohorts are randomly
subsampled to the exact CCA training size. Stratified sampling preserves
full outer-training outcome prevalence as closely as possible. The same sampled
participants are used for both imputation methods. Hyperparameters selected
in the primary analysis are held fixed to isolate the effect of training
sample size. Preprocessing is re-fitted only on the downsampled training data.

This script re-runs predictive modeling (ROC-AUC and Brier) on the same common
complete-case test participants. It saves fitted sensitivity models so a
separate SHAP sensitivity extension can be run if requested/required.

Pilot:
  python stage3_sensitivity_sample_size.py --cv-out output/stage3_final \
      --limit-splits 1 --out output/stage3_sensitivity/sample_size_pilot
Full:
  python stage3_sensitivity_sample_size.py --cv-out output/stage3_final \
      --out output/stage3_sensitivity/sample_size_matched
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import stage3_cv as primary

TARGET = primary.TARGET
PREDICTORS = primary.PREDICTORS
METHODS = ["Median/Mode", "KNN k=5"]
MODELS = primary.MODELS
BASE_SAMPLE_SEED = 2026


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_sensitivity_sample_size")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out_dir / "sample_size_sensitivity.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def prepare_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a fresh --out directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def choose_matched_sample(
    train_all: pd.DataFrame,
    cca_train: pd.DataFrame,
    split_id: int,
) -> tuple[pd.DataFrame, int]:
    """Downsample outer-train rows to CCA N, stratifying on the full-train outcome."""
    seed = BASE_SAMPLE_SEED + split_id
    if train_all[TARGET].nunique() != 2:
        raise ValueError(f"Split {split_id}: full outer-training cohort lacks one class")
    if len(train_all) < len(cca_train):
        raise ValueError(
            f"Split {split_id}: full outer-training N ({len(train_all)}) is smaller "
            f"than CCA training N ({len(cca_train)})."
        )
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=len(cca_train), random_state=seed
    )
    selected_idx, _ = next(
        splitter.split(np.zeros(len(train_all)), train_all[TARGET].astype(int).to_numpy())
    )
    sampled = train_all.iloc[selected_idx].sort_index().reset_index(drop=True)
    if len(sampled) != len(cca_train):
        raise RuntimeError(
            f"Split {split_id}: matched sample N={len(sampled)} != CCA N={len(cca_train)}"
        )
    if sampled[TARGET].nunique() != 2:
        raise RuntimeError(f"Split {split_id}: matched sample lost an outcome class")
    return sampled, seed


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method), group in frame.groupby(["model", "method"], sort=False):
        rows.append({
            "model": model,
            "method": method,
            "n_splits": int(group["split_id"].nunique()),
            "mean_roc_auc": float(group["roc_auc"].mean()),
            "sd_roc_auc": float(group["roc_auc"].std(ddof=1)),
            "mean_brier_score": float(group["brier_score"].mean()),
            "sd_brier_score": float(group["brier_score"].std(ddof=1)),
            "mean_train_n": float(group["train_n"].mean()),
        })
    return pd.DataFrame(rows)


def build_pairwise_gap_tables(
    primary_metrics: pd.DataFrame,
    matched_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [
        ("CCA", "Median/Mode"),
        ("CCA", "KNN k=5"),
        ("Median/Mode", "KNN k=5"),
    ]
    primary_lookup = {
        (int(r.split_id), r.model, r.method): r
        for r in primary_metrics.itertuples(index=False)
    }
    matched_lookup = {
        (int(r.split_id), r.model, r.method): r
        for r in matched_metrics.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    split_ids = sorted(matched_metrics["split_id"].astype(int).unique())
    for split_id in split_ids:
        for model in MODELS:
            for method_a, method_b in pairs:
                a_primary = primary_lookup[(split_id, model, method_a)]
                b_primary = primary_lookup[(split_id, model, method_b)]
                # CCA remains unchanged; imputation methods use the matched-N refit.
                a_matched = (
                    primary_lookup[(split_id, model, method_a)]
                    if method_a == "CCA"
                    else matched_lookup[(split_id, model, method_a)]
                )
                b_matched = (
                    primary_lookup[(split_id, model, method_b)]
                    if method_b == "CCA"
                    else matched_lookup[(split_id, model, method_b)]
                )
                auc_gap_primary = abs(float(a_primary.roc_auc) - float(b_primary.roc_auc))
                auc_gap_matched = abs(float(a_matched.roc_auc) - float(b_matched.roc_auc))
                brier_gap_primary = abs(float(a_primary.brier_score) - float(b_primary.brier_score))
                brier_gap_matched = abs(float(a_matched.brier_score) - float(b_matched.brier_score))
                rows.append({
                    "split_id": split_id,
                    "repeat": int(a_primary.repeat),
                    "fold": int(a_primary.fold),
                    "model": model,
                    "method_a": method_a,
                    "method_b": method_b,
                    "primary_abs_auc_gap": auc_gap_primary,
                    "matched_abs_auc_gap": auc_gap_matched,
                    "delta_abs_auc_gap_matched_minus_primary": auc_gap_matched - auc_gap_primary,
                    "auc_gap_decreased": bool(auc_gap_matched < auc_gap_primary),
                    "primary_abs_brier_gap": brier_gap_primary,
                    "matched_abs_brier_gap": brier_gap_matched,
                    "delta_abs_brier_gap_matched_minus_primary": brier_gap_matched - brier_gap_primary,
                    "brier_gap_decreased": bool(brier_gap_matched < brier_gap_primary),
                })
    by_split = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (model, method_a, method_b), group in by_split.groupby(
        ["model", "method_a", "method_b"], sort=False
    ):
        summary_rows.append({
            "model": model,
            "method_a": method_a,
            "method_b": method_b,
            "n_splits": int(group["split_id"].nunique()),
            "mean_primary_abs_auc_gap": float(group["primary_abs_auc_gap"].mean()),
            "mean_matched_abs_auc_gap": float(group["matched_abs_auc_gap"].mean()),
            "mean_delta_abs_auc_gap": float(group["delta_abs_auc_gap_matched_minus_primary"].mean()),
            "fraction_splits_auc_gap_decreased": float(group["auc_gap_decreased"].mean()),
            "mean_primary_abs_brier_gap": float(group["primary_abs_brier_gap"].mean()),
            "mean_matched_abs_brier_gap": float(group["matched_abs_brier_gap"].mean()),
            "mean_delta_abs_brier_gap": float(group["delta_abs_brier_gap_matched_minus_primary"].mean()),
            "fraction_splits_brier_gap_decreased": float(group["brier_gap_decreased"].mean()),
        })
    return by_split, pd.DataFrame(summary_rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-out", type=Path, default=root / "output" / "stage3_final",
        help="Completed primary Stage 3 CV output directory",
    )
    parser.add_argument(
        "--data", type=Path, default=root / "output" / "nhanes_examined_clean_with_missing.csv",
        help="Stage 1 examined-cohort dataset",
    )
    parser.add_argument(
        "--out", type=Path, default=root / "output" / "stage3_sensitivity" / "sample_size_matched",
        help="New output directory for this sensitivity analysis",
    )
    parser.add_argument(
        "--limit-splits", type=int, default=None,
        help="Optional number of initial outer splits for a pilot run",
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
    out_dir = args.out.expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    required = [
        data_path,
        cv_out / "stage3_cv_metadata.json",
        cv_out / "split_manifest_final.csv",
        cv_out / "cv_metrics_final.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required primary analysis files:\n- " + "\n- ".join(missing))
    prepare_out_dir(out_dir)
    logger = make_logger(out_dir)

    cv_meta = json.loads((cv_out / "stage3_cv_metadata.json").read_text(encoding="utf-8"))
    input_hash = sha256_file(data_path)
    if input_hash != cv_meta.get("input_dataset_sha256"):
        raise ValueError("Input dataset SHA-256 does not match primary Stage 3 metadata")

    raw = pd.read_csv(data_path)
    df, y_array, complete = primary.validate_input(raw, data_path, root)
    df = df.sort_values("SEQN").reset_index(drop=True)
    df_index = df.set_index("SEQN", drop=False)
    manifest = pd.read_csv(cv_out / "split_manifest_final.csv")
    primary_metrics = pd.read_csv(cv_out / "cv_metrics_final.csv")

    expected_primary_rows = 25 * len(MODELS) * len(primary.METHODS)
    if len(primary_metrics) != expected_primary_rows:
        raise ValueError(
            f"Primary metrics should have {expected_primary_rows} rows; found {len(primary_metrics)}"
        )
    if manifest.duplicated(["split_id", "SEQN"]).any():
        raise ValueError("Duplicate split_id/SEQN rows in primary split manifest")

    all_split_ids = sorted(manifest["split_id"].astype(int).unique().tolist())
    if args.limit_splits is not None:
        if args.limit_splits < 1 or args.limit_splits > len(all_split_ids):
            parser.error(f"--limit-splits must be between 1 and {len(all_split_ids)}")
        split_ids = all_split_ids[:args.limit_splits]
    else:
        split_ids = all_split_ids
    if not split_ids:
        raise ValueError("No split IDs found")

    logger.info("Starting sample-size sensitivity for %d outer splits", len(split_ids))
    logger.info("Primary CV directory: %s", cv_out)
    logger.info("Input SHA-256: %s", input_hash)
    logger.info("Sampling rule: match CCA training N; stratify by full outer-training outcome prevalence")
    logger.info("Same downsampled participant set is used for Median/Mode and KNN within each split")
    logger.info("Primary best hyperparameters are held fixed; no re-tuning in sensitivity analysis")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    sample_manifest_frames: list[pd.DataFrame] = []
    model_root = out_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for split_id in split_ids:
        split_started = time.perf_counter()
        split_manifest = manifest.loc[manifest["split_id"].astype(int).eq(split_id)].copy()
        role = split_manifest.set_index("SEQN")["role"]
        complete_flag = split_manifest.set_index("SEQN")["has_all_11_predictors"].astype(bool)
        train_ids = role.index[role.eq("train")]
        test_ids = role.index[role.eq("test") & complete_flag]
        train_all = df_index.loc[train_ids].copy()
        test = df_index.loc[test_ids].sort_index().copy()
        cca_train = train_all.loc[train_all[PREDICTORS].notna().all(axis=1)].copy()
        if len(test) == 0 or test[TARGET].nunique() != 2:
            raise ValueError(f"Split {split_id}: common complete-case test is empty or lacks a class")
        if len(cca_train) == 0 or cca_train[TARGET].nunique() != 2:
            raise ValueError(f"Split {split_id}: CCA training cohort is empty or lacks a class")

        sampled, sample_seed = choose_matched_sample(train_all, cca_train, split_id)
        logger.info(
            "Split %02d | full train=%d | CCA train=%d | matched imputation train=%d "
            "(positive=%d; full-train positive prevalence preserved by stratification) | common test=%d",
            split_id, len(train_all), len(cca_train), len(sampled),
            int(sampled[TARGET].sum()), len(test),
        )

        sample_manifest_frames.append(pd.DataFrame({
            "split_id": split_id,
            "repeat": int(split_manifest["repeat"].iloc[0]),
            "fold": int(split_manifest["fold"].iloc[0]),
            "SEQN": sampled["SEQN"].to_numpy(),
            "diabetes_reported": sampled[TARGET].astype(int).to_numpy(),
            "sample_seed": sample_seed,
            "matched_cca_train_n": len(cca_train),
            "cca_train_positive_n": int(cca_train[TARGET].sum()),
            "cca_train_negative_n": int(len(cca_train) - cca_train[TARGET].sum()),
            "matched_sample_positive_n": int(sampled[TARGET].sum()),
            "matched_sample_negative_n": int(len(sampled) - sampled[TARGET].sum()),
        }))

        split_model_dir = model_root / f"split_{split_id:03d}"
        split_model_dir.mkdir(parents=True, exist_ok=True)
        y_test = test[TARGET].astype(int).to_numpy()
        X_test = test[PREDICTORS].copy()

        for model_name in MODELS:
            for method in METHODS:
                primary_row = primary_metrics.loc[
                    primary_metrics["split_id"].astype(int).eq(split_id)
                    & primary_metrics["model"].eq(model_name)
                    & primary_metrics["method"].eq(method)
                ]
                if len(primary_row) != 1:
                    raise ValueError(
                        f"Split {split_id}: expected one primary metrics row for {model_name}/{method}"
                    )
                params = json.loads(primary_row.iloc[0]["best_params_json"])
                estimator_seed = int(cv_meta.get("outer_random_state", primary.BASE_SEED)) + split_id
                estimator = primary.build_pipeline(
                    method=method,
                    model_name=model_name,
                    params=params,
                    random_state=estimator_seed,
                )
                fit_started = time.perf_counter()
                estimator.fit(sampled[PREDICTORS], sampled[TARGET].astype(int))
                fit_seconds = time.perf_counter() - fit_started
                probabilities = estimator.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, probabilities))
                brier = float(brier_score_loss(y_test, probabilities))

                slug = f"{model_name}__{method.lower().replace('/', '_').replace(' ', '_').replace('=', '')}__nmatched"
                model_path = split_model_dir / f"{slug}.joblib"
                joblib.dump(estimator, model_path, compress=3)

                metric_rows.append({
                    "split_id": split_id,
                    "repeat": int(split_manifest["repeat"].iloc[0]),
                    "fold": int(split_manifest["fold"].iloc[0]),
                    "model": model_name,
                    "method": method,
                    "train_n": int(len(sampled)),
                    "train_positive_n": int(sampled[TARGET].sum()),
                    "train_negative_n": int(len(sampled) - sampled[TARGET].sum()),
                    "cca_train_n_reference": int(len(cca_train)),
                    "test_n": int(len(test)),
                    "test_positive_n": int(y_test.sum()),
                    "roc_auc": auc,
                    "brier_score": brier,
                    "primary_full_train_roc_auc": float(primary_row.iloc[0]["roc_auc"]),
                    "primary_full_train_brier_score": float(primary_row.iloc[0]["brier_score"]),
                    "best_params_json_reused": json.dumps(params, sort_keys=True),
                    "sample_seed": sample_seed,
                    "fit_seconds": float(fit_seconds),
                    "model_artifact": str(model_path.relative_to(out_dir)),
                })
                prediction_frames.append(pd.DataFrame({
                    "split_id": split_id,
                    "repeat": int(split_manifest["repeat"].iloc[0]),
                    "fold": int(split_manifest["fold"].iloc[0]),
                    "model": model_name,
                    "method": method,
                    "SEQN": test["SEQN"].to_numpy(),
                    "y_true": y_test,
                    "predicted_probability_class_1": probabilities.astype(float),
                }))
                logger.info(
                    "Split %02d | %s | %s | matched train_n=%d | AUC=%.6f | Brier=%.6f | %.1fs",
                    split_id, model_name, method, len(sampled), auc, brier, fit_seconds,
                )

        logger.info("Completed matched-size split %02d in %.1fs", split_id, time.perf_counter() - split_started)

    matched_metrics = pd.DataFrame(metric_rows)
    expected_metric_rows = len(split_ids) * len(MODELS) * len(METHODS)
    if len(matched_metrics) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} matched-size metric rows; found {len(matched_metrics)}"
        )
    if matched_metrics[["roc_auc", "brier_score"]].isna().any().any():
        raise RuntimeError("NaN found in matched-size metrics")

    selected_primary = primary_metrics.loc[primary_metrics["split_id"].astype(int).isin(split_ids)].copy()
    pairwise_by_split, pairwise_summary = build_pairwise_gap_tables(selected_primary, matched_metrics)

    matched_metrics.to_csv(out_dir / "sample_size_matched_metrics_by_split.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out_dir / "sample_size_matched_predictions.csv", index=False
    )
    pd.concat(sample_manifest_frames, ignore_index=True).to_csv(
        out_dir / "sample_size_matched_training_manifest.csv", index=False
    )
    summarize_metrics(matched_metrics).to_csv(
        out_dir / "sample_size_matched_summary.csv", index=False
    )
    pairwise_by_split.to_csv(out_dir / "sample_size_pairwise_gaps_by_split.csv", index=False)
    pairwise_summary.to_csv(out_dir / "sample_size_pairwise_gap_summary.csv", index=False)

    total_seconds = time.perf_counter() - started
    metadata = {
        "analysis_stage": "stage3_sensitivity_sample_size_pilot" if len(split_ids) < 25 else "stage3_sensitivity_sample_size_full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cv_directory": str(cv_out),
        "input_dataset_path": str(data_path),
        "input_dataset_sha256": input_hash,
        "splits_processed": split_ids,
        "n_splits_processed": len(split_ids),
        "models": MODELS,
        "methods_refit": METHODS,
        "training_size_rule": "For each split, downsample the full outer-training cohort to exactly the CCA training N.",
        "stratification_rule": "Stratified sampling by diabetes_reported preserves the full outer-training cohort outcome prevalence as closely as possible; class counts are not forced to equal CCA.",
        "shared_downsample_rule": "The same downsampled participants are used for Median/Mode and KNN within each split.",
        "sampling_seed_rule": "2026 + split_id.",
        "hyperparameter_rule": "Reuse best_params_json selected in the primary analysis; no re-tuning, to isolate the training-size effect.",
        "preprocessing_rule": "All imputation, scaling, and encoding are re-fitted within each downsampled training cohort only.",
        "test_rule": "Same complete-case outer test participants as the primary analysis.",
        "metrics": ["ROC-AUC", "Brier score"],
        "pairwise_gap_rule": "Absolute primary and matched-size gaps are compared for all three method pairs within each model.",
        "shap_recomputed": False,
        "runtime_seconds": float(total_seconds),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
    }
    (out_dir / "sample_size_sensitivity_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Matched-size metric rows: %d (expected %d)", len(matched_metrics), expected_metric_rows)
    logger.info("Pairwise gap rows: %d (expected %d)", len(pairwise_by_split), len(split_ids) * len(MODELS) * 3)
    logger.info("Runtime: %.1f seconds", total_seconds)
    logger.info("Output directory: %s", out_dir)
    logger.info("Run status: %s", "COMPLETE" if len(split_ids) == 25 else "PARTIAL / PILOT")


if __name__ == "__main__":
    main()
