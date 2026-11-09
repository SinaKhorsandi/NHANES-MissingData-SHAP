#!/usr/bin/env python3
# -*- coding: utf-8
"""Compute pairwise SHAP stability metrics for Stage 3 outer splits.

Expected input directory is the `shap` subdirectory created by stage3_shap.py.
For each split and model, this script computes all three pairwise comparisons:
CCA vs Median/Mode, CCA vs KNN, and Median/Mode vs KNN.

Metrics are calculated on the 11 original predictors after signed one-hot
attributions have already been aggregated by stage3_shap.py:
- Spearman correlation between global mean-absolute-SHAP feature rankings
- Jaccard similarity of the top-five global features
- MSE-SHAP across the shared test participants and 11 predictors

MSE-SHAP is only calculated within the same model. No model-to-model MSE is
computed because LR and RF use different output scales.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDICTORS = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN", "SMQ020", "PAQ605",
]
MODELS = ["LR", "RF"]
METHODS = ["CCA", "Median/Mode", "KNN k=5"]
PAIRS = [
    ("CCA", "Median/Mode"),
    ("CCA", "KNN k=5"),
    ("Median/Mode", "KNN k=5"),
]
ADD_TOLERANCE = 1e-8


def method_slug(model_name: str, method: str) -> str:
    method_part = method.lower().replace("/", "_").replace(" ", "_").replace("=", "")
    return f"{model_name}__{method_part}"


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_stability")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def extract_matrix(frame: pd.DataFrame, model: str, method: str) -> np.ndarray:
    prefix = method_slug(model, method)
    columns = [f"{prefix}__{feature}" for feature in PREDICTORS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing SHAP columns for {model}/{method}: {missing}")
    values = frame[columns].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(frame), len(PREDICTORS)):
        raise ValueError(
            f"Unexpected SHAP matrix shape for {model}/{method}: {values.shape}; "
            f"expected ({len(frame)}, {len(PREDICTORS)})"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite SHAP values found for {model}/{method}")
    return values


def global_importance(values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(values), axis=0)


def stable_top_k(importance: np.ndarray, k: int = 5) -> list[str]:
    # Stable tie-breaking follows the original predictor order in PREDICTORS.
    order = np.argsort(-importance, kind="stable")
    return [PREDICTORS[i] for i in order[:k]]


def validate_audit(audit: pd.DataFrame, split_ids: list[int]) -> None:
    required = {"split_id", "model", "method", "audit_status", "tolerance"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Additivity audit is missing columns: {sorted(missing)}")

    selected = audit[audit["split_id"].astype(int).isin(split_ids)].copy()
    expected = len(split_ids) * len(MODELS) * len(METHODS)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} additivity audit rows; found {len(selected)}")
    if selected.duplicated(["split_id", "model", "method"]).any():
        raise ValueError("Duplicate additivity audit rows found")
    if not selected["audit_status"].eq("PASS").all():
        raise ValueError("At least one additivity audit row is not PASS")

    error_columns = [
        column for column in selected.columns
        if column.startswith("max_abs_additivity_error_")
    ]
    if not error_columns:
        raise ValueError("No maximum additivity error columns found in audit")
    for column in error_columns:
        observed = pd.to_numeric(selected[column], errors="coerce").dropna()
        if len(observed) and (observed >= ADD_TOLERANCE).any():
            raise ValueError(
                f"Additivity error in {column} is not strictly below {ADD_TOLERANCE:.0e}"
            )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shap-dir", type=Path,
        default=root / "output" / "stage3_final" / "shap",
        help="SHAP output directory created by stage3_shap.py",
    )
    parser.add_argument(
        "--limit-splits", type=int, default=None,
        help="Optional number of initial split files to process (for pilot checks)",
    )
    args = parser.parse_args()

    shap_dir = args.shap_dir.expanduser()
    if not shap_dir.is_absolute():
        shap_dir = (root / shap_dir).resolve()
    else:
        shap_dir = shap_dir.resolve()

    values_dir = shap_dir / "values_by_split"
    audit_path = shap_dir / "additivity_audit_final.csv"
    if not values_dir.is_dir():
        raise FileNotFoundError(f"Missing SHAP values directory: {values_dir}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing additivity audit: {audit_path}")

    value_files = sorted(values_dir.glob("shap_values_split_*.csv"))
    if not value_files:
        raise FileNotFoundError(f"No split SHAP files found in {values_dir}")
    if args.limit_splits is not None:
        if args.limit_splits < 1:
            raise SystemExit("--limit-splits must be >= 1")
        value_files = value_files[:args.limit_splits]

    split_ids = []
    for path in value_files:
        try:
            split_ids.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError as exc:
            raise ValueError(f"Cannot parse split id from {path.name}") from exc
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("Duplicate split IDs found in SHAP value filenames")

    output_path = shap_dir / "shap_stability_metrics_by_split_final.csv"
    metadata_path = shap_dir / "shap_stability_metadata.json"
    log_path = shap_dir / "stage3_stability.log"
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Stability output already exists. Preserve it and use a fresh SHAP output directory."
        )

    logger = make_logger(log_path)
    logger.info("Starting SHAP stability calculations for %d splits", len(split_ids))
    logger.info("SHAP directory: %s", shap_dir)

    audit = pd.read_csv(audit_path)
    validate_audit(audit, split_ids)

    rows: list[dict[str, Any]] = []
    for split_id, path in zip(split_ids, value_files):
        frame = pd.read_csv(path)
        required_id_cols = {"split_id", "repeat", "fold", "SEQN", "y_true"}
        missing_id_cols = required_id_cols - set(frame.columns)
        if missing_id_cols:
            raise ValueError(f"{path.name} missing ID columns: {sorted(missing_id_cols)}")
        if frame["SEQN"].isna().any() or frame["SEQN"].duplicated().any():
            raise ValueError(f"{path.name}: SEQN must be non-missing and unique")
        if frame["split_id"].nunique() != 1 or int(frame["split_id"].iloc[0]) != split_id:
            raise ValueError(f"{path.name}: inconsistent split_id")
        if frame["repeat"].nunique() != 1 or frame["fold"].nunique() != 1:
            raise ValueError(f"{path.name}: repeat/fold metadata is not constant")

        n_test = len(frame)
        if n_test == 0 or frame["y_true"].nunique() != 2:
            raise ValueError(f"Split {split_id}: test data is empty or lacks one target class")
        logger.info("Split %02d | test participants=%d", split_id, n_test)

        for model in MODELS:
            matrices = {
                method: extract_matrix(frame, model, method)
                for method in METHODS
            }
            importances = {
                method: global_importance(values)
                for method, values in matrices.items()
            }
            top_sets = {
                method: stable_top_k(importances[method], k=5)
                for method in METHODS
            }

            # The three pairwise comparisons are symmetric; no method is a reference.
            for method_a, method_b in PAIRS:
                rank_a = pd.Series(importances[method_a]).rank(method="average").to_numpy()
                rank_b = pd.Series(importances[method_b]).rank(method="average").to_numpy()
                rho = float(spearmanr(rank_a, rank_b).statistic)
                set_a = set(top_sets[method_a])
                set_b = set(top_sets[method_b])
                union = set_a | set_b
                jaccard = float(len(set_a & set_b) / len(union)) if union else float("nan")

                delta = matrices[method_a] - matrices[method_b]
                mse = float(np.mean(np.square(delta)))
                mae = float(np.mean(np.abs(delta)))

                if not np.isfinite([rho, jaccard, mse, mae]).all():
                    raise ValueError(
                        f"Non-finite stability metric in split {split_id}, {model}, "
                        f"{method_a} vs {method_b}"
                    )

                audit_rows = audit[
                    audit["split_id"].astype(int).eq(split_id)
                    & audit["model"].eq(model)
                    & audit["method"].isin([method_a, method_b])
                ]
                scales = sorted(audit_rows["output_scale"].dropna().unique().tolist()) if "output_scale" in audit_rows else []
                if len(audit_rows) != 2 or len(scales) != 1:
                    raise ValueError(
                        f"Output scales are not consistent for {model}: {method_a} vs {method_b}"
                    )

                rows.append({
                    "split_id": split_id,
                    "repeat": int(frame["repeat"].iloc[0]),
                    "fold": int(frame["fold"].iloc[0]),
                    "model": model,
                    "method_a": method_a,
                    "method_b": method_b,
                    "test_n": n_test,
                    "feature_n": len(PREDICTORS),
                    "spearman_rank_correlation": rho,
                    "top5_jaccard": jaccard,
                    "top5_method_a": ";".join(top_sets[method_a]),
                    "top5_method_b": ";".join(top_sets[method_b]),
                    "top5_overlap_n": len(set_a & set_b),
                    "mse_shap": mse,
                    "mae_shap": mae,
                    "shap_scale": scales[0],
                })

        logger.info("Split %02d | generated %d pairwise rows", split_id, len(MODELS) * len(PAIRS))

    result = pd.DataFrame(rows).sort_values(
        ["split_id", "model", "method_a", "method_b"]
    ).reset_index(drop=True)
    expected_rows = len(split_ids) * len(MODELS) * len(PAIRS)
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} stability rows; found {len(result)}")
    if result.duplicated(["split_id", "model", "method_a", "method_b"]).any():
        raise RuntimeError("Duplicate pairwise comparison rows found")

    # Explicitly guard against any accidental cross-model MSE comparison.
    if result.groupby(["split_id", "model"]).size().ne(3).any():
        raise RuntimeError("Each split/model must have exactly three pairwise comparisons")

    result.to_csv(output_path, index=False)
    metadata = {
        "analysis_stage": "stage3_stability_partial" if len(split_ids) < 25 else "stage3_stability_full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shap_directory": str(shap_dir),
        "splits_processed": split_ids,
        "n_splits_processed": len(split_ids),
        "models": MODELS,
        "methods": METHODS,
        "pairwise_comparisons": [list(pair) for pair in PAIRS],
        "expected_rows": expected_rows,
        "actual_rows": len(result),
        "original_predictors": PREDICTORS,
        "feature_n": len(PREDICTORS),
        "metrics": [
            "spearman_rank_correlation",
            "top5_jaccard",
            "mse_shap",
            "mae_shap",
        ],
        "mse_comparison_rule": "MSE-SHAP is computed only within the same model and same split; LR and RF scales are never compared.",
        "top5_rule": "Top five predictors are ranked by mean absolute signed-sum-aggregated SHAP; ties are resolved by original predictor order.",
        "mse_rule": "Mean squared difference across the shared test participants and 11 original predictors.",
        "spearman_rule": "Spearman correlation between ranks of per-feature mean absolute SHAP across the 11 original predictors.",
        "jaccard_rule": "Intersection-over-union of the two top-five predictor sets.",
        "additivity_audit": "All corresponding SHAP audit rows were checked to be PASS with maximum error below 1e-8.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("Stability rows: %d (expected %d)", len(result), expected_rows)
    logger.info("Models: %s", ", ".join(MODELS))
    logger.info("Pairwise comparisons per model/split: %d", len(PAIRS))
    logger.info("Saved metrics: %s", output_path)
    logger.info("Saved metadata: %s", metadata_path)
    logger.info("Run status: %s", "COMPLETE" if len(split_ids) == 25 else "PARTIAL / PILOT")


if __name__ == "__main__":
    main()
