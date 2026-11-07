#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES Stage 2 SHAP pilot: Logistic Regression under three missing-data methods.

Run from the Stage 1 project root:
    python stage2_pilot_shap_lr.py

This first SHAP script intentionally handles the LR family only. It uses:
- the saved fixed split from Stage 2;
- the same 798 complete-case held-out participants for all LR methods;
- one shared random background sample of 100 complete-case TRAIN participants;
- LinearExplainer with an Independent masker;
- LR decision_function (log-odds) as the explained output;
- sum of one-hot encoded contributions back to each original predictor.

CCA is the reference within the LR family. This is a pilot, not final analysis.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import shap
import sklearn

TARGET = "diabetes_reported"
NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST",
    "BP_SYS_MEAN", "BP_DIA_MEAN",
]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605"]
PREDICTORS = NUMERIC + CATEGORICAL
BACKGROUND_N = 100
BACKGROUND_SEED = 2026
EXPECTED_TEST_N = 798

MODELS = {
    "Median/Mode": "lr_median_mode_pipeline.joblib",
    "CCA": "cca_lr_pipeline.joblib",
    "KNN k=5": "knn_lr_pipeline.joblib",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dense_float(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def normalize_lr_shap(values, base_values, n_rows):
    values = np.asarray(values)
    base_values = np.asarray(base_values)
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
        base_values = np.full(n_rows, float(base_values))
    else:
        base_values = base_values.reshape(-1)
    if len(base_values) == 1 and n_rows > 1:
        base_values = np.repeat(base_values, n_rows)
    if len(base_values) != n_rows:
        raise ValueError("SHAP base-value count does not match test rows")
    return values, base_values


def map_original_features(feature_names):
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
        raise ValueError(f"No transformed columns for: {empty}")
    return groups


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "output" / "stage2_pilot_v1"
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    manifest_path = output_dir / "split_manifest.csv"
    metadata_path = output_dir / "pilot_metadata.json"

    required = [data_path, manifest_path, metadata_path]
    required += [output_dir / filename for filename in MODELS.values()]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(
            "Required files are missing:\n- " + "\n- ".join(missing)
            + "\nRun all LR/CCA/KNN pilot scripts first."
        )

    df = pd.read_csv(data_path).sort_values("SEQN").reset_index(drop=True)
    manifest = pd.read_csv(manifest_path).sort_values("SEQN").reset_index(drop=True)
    stage2_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256(data_path) != stage2_meta.get("input_sha256"):
        raise ValueError("Input dataset changed since the Stage 2 pilot")
    if not np.array_equal(df["SEQN"], manifest["SEQN"]):
        raise ValueError("Dataset IDs/order differ from split manifest")
    if not np.array_equal(df[TARGET], manifest[TARGET]):
        raise ValueError("Target labels differ from split manifest")
    complete = ~df[PREDICTORS].isna().any(axis=1).to_numpy()
    if not np.array_equal(complete, manifest["has_all_11_predictors"].astype(bool).to_numpy()):
        raise ValueError("Complete-case membership differs from split manifest")

    train_mask = manifest["split"].eq("train").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()
    background_pool = df.loc[train_mask & complete].copy()
    if len(background_pool) < BACKGROUND_N:
        raise ValueError("Not enough complete-case training rows for SHAP background")
    background = background_pool.sample(
        n=BACKGROUND_N, random_state=BACKGROUND_SEED
    ).sort_values("SEQN").reset_index(drop=True)
    X_background_raw = background[PREDICTORS].copy()

    common_test = df.loc[test_mask & complete].copy().sort_values("SEQN").reset_index(drop=True)
    if len(common_test) != EXPECTED_TEST_N:
        print(f"NOTE: expected {EXPECTED_TEST_N} complete-case test rows; found {len(common_test)}")
    X_test_raw = common_test[PREDICTORS].copy()
    y_test = common_test[TARGET].astype(int).to_numpy()
    test_seqn = common_test["SEQN"].to_numpy()

    if X_background_raw.isna().any().any() or X_test_raw.isna().any().any():
        raise ValueError("SHAP background and test data must be complete")

    background[["SEQN", TARGET]].assign(
        role="shared_complete_case_training_background"
    ).to_csv(output_dir / "shap_lr_background_manifest.csv", index=False)

    shap_wide = pd.DataFrame({"SEQN": test_seqn, "y_true": y_test})
    global_rows = []
    additivity_rows = []
    values_by_method = {}
    started = time.time()

    for method, filename in MODELS.items():
        print(f"Computing LR SHAP: {method} ...", flush=True)
        model = joblib.load(output_dir / filename)
        pre = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]
        X_bg = dense_float(pre.transform(X_background_raw))
        X_test = dense_float(pre.transform(X_test_raw))
        feature_names = list(pre.get_feature_names_out())
        if X_bg.shape[1] != len(feature_names) or X_test.shape[1] != len(feature_names):
            raise ValueError("Transformed features and feature names do not match")

        masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
        explainer = shap.LinearExplainer(classifier, masker)
        explanation = explainer(X_test)
        encoded_values, base_values = normalize_lr_shap(
            explanation.values, explanation.base_values, len(X_test)
        )
        margin = np.asarray(classifier.decision_function(X_test)).reshape(-1)
        reconstructed = base_values + encoded_values.sum(axis=1)
        errors = np.abs(reconstructed - margin)
        max_error = float(errors.max())
        mean_error = float(errors.mean())
        if max_error > 1e-4:
            raise ValueError(f"LR SHAP additivity check failed for {method}: {max_error}")

        groups = map_original_features(feature_names)
        grouped = np.zeros((len(X_test), len(PREDICTORS)), dtype=np.float64)
        for j, feature in enumerate(PREDICTORS):
            grouped[:, j] = encoded_values[:, groups[feature]].sum(axis=1)
        if not np.allclose(grouped.sum(axis=1), encoded_values.sum(axis=1), atol=1e-8, rtol=1e-8):
            raise ValueError("Summing one-hot attributions changed total SHAP values")

        values_by_method[method] = grouped
        method_key = method.lower().replace("/", "_").replace(" ", "_").replace("=", "")
        for j, feature in enumerate(PREDICTORS):
            shap_wide[f"LR__{method_key}__{feature}"] = grouped[:, j]
            global_rows.append({
                "model": "LR", "method": method, "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(grouped[:, j]))),
                "mean_shap": float(np.mean(grouped[:, j])),
            })
        additivity_rows.append({
            "model": "LR", "method": method,
            "n_explained": len(X_test),
            "background_n": len(X_bg),
            "encoded_feature_count": X_test.shape[1],
            "max_abs_additivity_error_log_odds": max_error,
            "mean_abs_additivity_error_log_odds": mean_error,
            "output_scale": "log_odds / LogisticRegression decision_function",
        })
        print(f"  done | max additivity error={max_error:.3g}", flush=True)

    importance = pd.DataFrame(global_rows)
    importance["rank"] = importance.groupby(["model", "method"])["mean_abs_shap"].rank(
        method="min", ascending=False
    ).astype(int)
    importance = importance.sort_values(["method", "rank", "feature"])

    ref_imp = importance.query("method == 'CCA'").set_index("feature")[
        "mean_abs_shap"
    ].reindex(PREDICTORS)
    ref_rank = ref_imp.rank(method="average", ascending=False)
    ref_top5 = set(ref_imp.sort_values(ascending=False).head(5).index)
    stability = []
    ref_values = values_by_method["CCA"]
    for method in ["Median/Mode", "KNN k=5"]:
        cur_imp = importance.query("method == @method").set_index("feature")[
            "mean_abs_shap"
        ].reindex(PREDICTORS)
        cur_rank = cur_imp.rank(method="average", ascending=False)
        cur_top5 = set(cur_imp.sort_values(ascending=False).head(5).index)
        union = ref_top5 | cur_top5
        difference = ref_values - values_by_method[method]
        stability.append({
            "model": "LR",
            "reference_method": "CCA",
            "comparison_method": method,
            "test_n": len(X_test),
            "background_n": len(X_bg),
            "spearman_rank_correlation": float(spearmanr(ref_rank, cur_rank).statistic),
            "reference_top5": ";".join([x for x in PREDICTORS if x in ref_top5]),
            "comparison_top5": ";".join([x for x in PREDICTORS if x in cur_top5]),
            "top5_jaccard": float(len(ref_top5 & cur_top5) / len(union)),
            "mse_shap": float(np.mean(difference ** 2)),
            "mae_shap": float(np.mean(np.abs(difference))),
            "shap_scale": "log_odds",
            "note": "Pilot only; CCA and imputation methods use different training cohort sizes/compositions.",
        })

    shap_wide.to_csv(output_dir / "shap_lr_values_common_test.csv", index=False)
    importance.to_csv(output_dir / "shap_lr_global_importance.csv", index=False)
    pd.DataFrame(stability).to_csv(output_dir / "shap_lr_stability_vs_cca.csv", index=False)
    pd.DataFrame(additivity_rows).to_csv(output_dir / "shap_lr_additivity_audit.csv", index=False)

    meta = {
        "stage": "LR SHAP pilot only; not final paper results",
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "input_dataset_sha256": sha256(data_path),
        "common_test_n": int(len(X_test_raw)),
        "common_test_definition": "all held-out complete-case participants (same 798 used in model comparison)",
        "background_n": BACKGROUND_N,
        "background_seed": BACKGROUND_SEED,
        "background_source": "random sample from complete-case training participants; common across LR methods",
        "explainer": "shap.LinearExplainer with Independent masker",
        "output_scale": "LogisticRegression decision_function (log-odds)",
        "categorical_handling": "one-hot encoded at model level; encoded-column SHAP values summed to original predictor",
        "reference": "LR + CCA",
        "metrics": ["Spearman", "Top-5 Jaccard", "MSE-SHAP", "MAE-SHAP"],
        "caveats": [
            "This is the LR family only; RF SHAP will be handled separately.",
            "Summed one-hot contributions are grouped encoded-feature attributions, not a separately fitted group-Shapley game.",
            "CCA training cohort is smaller than imputation-method cohorts.",
            "Background-size and background-seed sensitivity are pending.",
            "No repeated cross-validation, tuning, or survey weighting is included.",
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
    (output_dir / "shap_lr_pilot_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== LR SHAP PILOT COMPLETE ===")
    print("Explained participants:", len(X_test_raw))
    print("Shared background participants:", BACKGROUND_N)
    print("Stability vs LR-CCA:")
    print(pd.DataFrame(stability).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved under:", output_dir)
    print("NOTE: Pilot only; RF SHAP and final repeated-CV analysis are pending.")


if __name__ == "__main__":
    main()
