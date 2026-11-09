#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Stage 3 manuscript tables and figures from completed NHANES outputs.

Inputs:
  output/stage3_final/cv_metrics_final.csv
  output/stage3_final/shap/shap_stability_metrics_by_split_final.csv
  output/stage3_final/shap/shap_global_importance_final.csv
  output/stage3_final/shap/additivity_audit_final.csv

Outputs:
  Table 1: mean +/- SD ROC-AUC and Brier for six model/method combinations.
  Table 1 paired differences: same-split method contrasts within each model.
  Table 2: SHAP stability mean +/- SD by model and pairwise method comparison.
  Figure 1: split-averaged mean absolute SHAP by feature, separate panels for LR/RF.
  Figure 2: bump chart of mean feature ranks, separate panels for LR/RF.

MSE-SHAP is summarized within each model only. LR and RF SHAP values use
different output scales and their MSE-SHAP values must not be compared.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["LR", "RF"]
METHODS = ["CCA", "Median/Mode", "KNN k=5"]
PAIRS = [
    ("CCA", "Median/Mode"),
    ("CCA", "KNN k=5"),
    ("Median/Mode", "KNN k=5"),
]
FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN", "SMQ020", "PAQ605",
]


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_reporting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def mean_sd(series: pd.Series) -> tuple[float, float]:
    return float(series.mean()), float(series.std(ddof=1))


def audit_inputs(metrics: pd.DataFrame, stability: pd.DataFrame,
                 importance: pd.DataFrame, audit: pd.DataFrame) -> dict:
    checks = {
        "cv_metric_rows_150": len(metrics) == 150,
        "cv_splits_25": metrics["split_id"].nunique() == 25,
        "cv_models_LR_RF": set(metrics["model"]) == set(MODELS),
        "cv_methods_complete": set(metrics["method"]) == set(METHODS),
        "cv_six_rows_per_split": metrics.groupby("split_id").size().eq(6).all(),
        "cv_no_missing_auc_brier": not metrics[["roc_auc", "brier_score"]].isna().any().any(),
        "stability_rows_150": len(stability) == 150,
        "stability_models_LR_RF": set(stability["model"]) == set(MODELS),
        "stability_no_missing_metrics": not stability[
            ["spearman_rank_correlation", "top5_jaccard", "mse_shap", "mae_shap"]
        ].isna().any().any(),
        "importance_rows_1650": len(importance) == 25 * 2 * 3 * 11,
        "importance_features_11": set(importance["feature"]) == set(FEATURES),
        "importance_no_missing": not importance["mean_abs_shap"].isna().any(),
        "additivity_rows_150": len(audit) == 150,
        "all_additivity_pass": audit["audit_status"].eq("PASS").all(),
    }
    return {k: bool(v) for k, v in checks.items()}


def build_table1(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, method), group in metrics.groupby(["model", "method"], sort=False):
        auc_mean, auc_sd = mean_sd(group["roc_auc"])
        brier_mean, brier_sd = mean_sd(group["brier_score"])
        rows.append({
            "model": model,
            "method": method,
            "n_splits": int(group["split_id"].nunique()),
            "mean_roc_auc": auc_mean,
            "sd_roc_auc": auc_sd,
            "roc_auc_mean_sd": f"{auc_mean:.4f} ± {auc_sd:.4f}",
            "mean_brier_score": brier_mean,
            "sd_brier_score": brier_sd,
            "brier_mean_sd": f"{brier_mean:.4f} ± {brier_sd:.4f}",
            "mean_train_n": float(group["train_n"].mean()),
            "mean_test_n": float(group["test_n"].mean()),
        })
    table1 = pd.DataFrame(rows)

    paired_rows = []
    for model in MODELS:
        model_df = metrics[metrics["model"].eq(model)]
        for method_a, method_b in PAIRS:
            pair = model_df.pivot(
                index="split_id", columns="method",
                values=["roc_auc", "brier_score"]
            )
            auc_diff = pair["roc_auc"][method_a] - pair["roc_auc"][method_b]
            brier_diff = pair["brier_score"][method_a] - pair["brier_score"][method_b]
            paired_rows.append({
                "model": model,
                "method_a": method_a,
                "method_b": method_b,
                "difference_definition": "method_a minus method_b on the same outer split",
                "n_paired_splits": int(auc_diff.notna().sum()),
                "mean_auc_difference": float(auc_diff.mean()),
                "sd_auc_difference": float(auc_diff.std(ddof=1)),
                "mean_brier_difference": float(brier_diff.mean()),
                "sd_brier_difference": float(brier_diff.std(ddof=1)),
                "fraction_splits_auc_method_a_higher": float((auc_diff > 0).mean()),
                "fraction_splits_brier_method_a_lower": float((brier_diff < 0).mean()),
            })
    return table1, pd.DataFrame(paired_rows)


def build_table2(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "method_a", "method_b"]
    metric_map = {
        "spearman_rank_correlation": "spearman",
        "top5_jaccard": "top5_jaccard",
        "mse_shap": "mse_shap",
        "mae_shap": "mae_shap",
    }
    for keys, group in stability.groupby(group_cols, sort=False):
        model, method_a, method_b = keys
        row = {
            "model": model,
            "method_a": method_a,
            "method_b": method_b,
            "n_splits": int(group["split_id"].nunique()),
            "shap_scale": ";".join(sorted(group["shap_scale"].dropna().unique())),
        }
        for source_col, label in metric_map.items():
            mean, sd = mean_sd(group[source_col])
            row[f"mean_{label}"] = mean
            row[f"sd_{label}"] = sd
            row[f"{label}_mean_sd"] = f"{mean:.4g} ± {sd:.4g}"
        rows.append(row)
    # Each row retains model, so MSE-SHAP is never pooled across LR and RF.
    return pd.DataFrame(rows)


def build_figures(out_dir: Path, importance: pd.DataFrame) -> None:
    # Aggregate the per-split global importance values. The LR and RF panels
    # are deliberately separate because their SHAP output scales differ.
    agg = (
        importance.groupby(["model", "method", "feature"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"),
             sd_abs_shap=("mean_abs_shap", "std"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=False)
    x = np.arange(len(FEATURES))
    width = 0.24
    offsets = [-width, 0.0, width]
    for ax, model in zip(axes, MODELS):
        model_data = agg[agg["model"].eq(model)]
        for method, offset in zip(METHODS, offsets):
            subset = (
                model_data[model_data["method"].eq(method)]
                .set_index("feature")
                .reindex(FEATURES)
            )
            ax.bar(
                x + offset,
                subset["mean_abs_shap"].to_numpy(),
                width=width,
                yerr=subset["sd_abs_shap"].fillna(0).to_numpy(),
                capsize=2,
                label=method,
            )
        ax.set_title(
            "Logistic Regression (log-odds)" if model == "LR"
            else "Random Forest (positive-class probability)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURES, rotation=55, ha="right")
        ax.set_ylabel("Mean absolute SHAP value across outer splits")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Missing-data method")
    fig.suptitle(
        "Figure 1. Global SHAP importance by feature and missing-data method",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_mean_abs_shap_by_model.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure1_mean_abs_shap_by_model.pdf", bbox_inches="tight")
    plt.close(fig)

    # Mean rank across the 25 outer splits. Rank 1 is plotted at the top.
    rank_agg = (
        importance.groupby(["model", "method", "feature"], as_index=False)
        .agg(mean_rank=("rank", "mean"), sd_rank=("rank", "std"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    x = np.arange(len(METHODS))
    for ax, model in zip(axes, MODELS):
        model_data = rank_agg[rank_agg["model"].eq(model)]
        for feature in FEATURES:
            subset = (
                model_data[model_data["feature"].eq(feature)]
                .set_index("method")
                .reindex(METHODS)
            )
            y = subset["mean_rank"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.25)
            if np.isfinite(y[-1]):
                ax.annotate(
                    feature,
                    (x[-1], y[-1]),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
        ax.set_title("Logistic Regression" if model == "LR" else "Random Forest")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=15, ha="right")
        ax.set_xlim(-0.1, 2.65)
        ax.set_ylim(11.5, 0.5)
        ax.set_yticks(range(1, 12))
        ax.set_ylabel("Mean feature rank (1 = most important)")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Figure 2. Bump chart of mean feature ranks across missing-data methods",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_feature_rank_bump_chart.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure2_feature_rank_bump_chart.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-out", type=Path, default=Path("output/stage3_final"))
    parser.add_argument("--shap-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("output/stage3_reporting_v1"))
    args = parser.parse_args()

    cv_out = resolve(root, args.cv_out)
    shap_dir = resolve(root, args.shap_dir) if args.shap_dir else cv_out / "shap"
    out_dir = resolve(root, args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Choose a new --out directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "stage3_reporting.log")

    paths = {
        "metrics": cv_out / "cv_metrics_final.csv",
        "stability": shap_dir / "shap_stability_metrics_by_split_final.csv",
        "importance": shap_dir / "shap_global_importance_final.csv",
        "audit": shap_dir / "additivity_audit_final.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    metrics = pd.read_csv(paths["metrics"])
    stability = pd.read_csv(paths["stability"])
    importance = pd.read_csv(paths["importance"])
    audit = pd.read_csv(paths["audit"])

    checks = audit_inputs(metrics, stability, importance, audit)
    logger.info("=== INPUT QUALITY AUDIT ===")
    for name, passed in checks.items():
        logger.info("%s | %s", "PASS" if passed else "FAIL", name)
    if not all(checks.values()):
        raise RuntimeError("Stage 3 reporting input quality audit failed")

    table1, paired = build_table1(metrics)
    table2 = build_table2(stability)
    table1.to_csv(out_dir / "table1_predictive_performance.csv", index=False)
    paired.to_csv(out_dir / "table1_paired_performance_differences.csv", index=False)
    table2.to_csv(out_dir / "table2_shap_stability.csv", index=False)
    build_figures(out_dir, importance)

    summary = {
        "status": "COMPLETE",
        "n_outer_splits": 25,
        "n_primary_model_results": len(metrics),
        "n_shap_stability_rows": len(stability),
        "n_global_importance_rows": len(importance),
        "n_additivity_audit_rows": len(audit),
        "all_additivity_pass": bool(audit["audit_status"].eq("PASS").all()),
        "table1_rows": len(table1),
        "table1_paired_difference_rows": len(paired),
        "table2_rows": len(table2),
        "figure1_models_separated_due_to_different_shap_scales": True,
        "figure2_models_separated": True,
        "mse_shap_compared_only_within_model": True,
        "input_files": {k: str(v) for k, v in paths.items()},
        "output_files": [
            "table1_predictive_performance.csv",
            "table1_paired_performance_differences.csv",
            "table2_shap_stability.csv",
            "figure1_mean_abs_shap_by_model.png",
            "figure1_mean_abs_shap_by_model.pdf",
            "figure2_feature_rank_bump_chart.png",
            "figure2_feature_rank_bump_chart.pdf",
        ],
        "quality_checks": checks,
    }
    (out_dir / "stage3_reporting_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Table 1 rows: %d", len(table1))
    logger.info("Table 1 paired difference rows: %d", len(paired))
    logger.info("Table 2 rows: %d", len(table2))
    logger.info("Figures: 2 (PNG and PDF versions)")
    logger.info("Output directory: %s", out_dir)
    logger.info("Run status: COMPLETE")


if __name__ == "__main__":
    main()
