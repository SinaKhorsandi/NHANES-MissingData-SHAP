#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RF SHAP pilot using PermutationExplainer after TreeExplainer additivity audit.

Run from the Stage 1 project root:
    python stage2_pilot_shap_rf_permutation.py

Optional quick smoke test:
    python stage2_pilot_shap_rf_permutation.py --max-rows 32

More permutation cycles improve the Monte Carlo estimate for higher-order
interactions but increase runtime:
    python stage2_pilot_shap_rf_permutation.py --permutation-cycles 5

This script does NOT alter the fitted models or Stage 1 data. It uses the same
100 complete-case training background participants and the same held-out
complete-case test participants as the LR SHAP pilot. The model function
returns positive-class probabilities. One-hot feature attributions are summed
back to the original 11 predictors, matching the existing LR SHAP grouping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
PERMUTATION_SEED = 2026
ADD_TOLERANCE = 1e-8
MODELS = {
    "Median/Mode": "rf_median_mode_pipeline.joblib",
    "CCA": "cca_rf_pipeline.joblib",
    "KNN k=5": "knn_rf_pipeline.joblib",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional deterministic subset of the common test set for a smoke test.",
    )
    parser.add_argument(
        "--permutation-cycles", type=int, default=1,
        help="Number of forward/reverse permutation cycles (1 is pilot minimum).",
    )
    return parser.parse_args()


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
    args = parse_args()
    if args.permutation_cycles < 1:
        raise SystemExit("--permutation-cycles must be >= 1")

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
            + "\nRun LR, RF, CCA, and KNN pilot scripts first."
        )

    df = pd.read_csv(data_path).sort_values("SEQN").reset_index(drop=True)
    manifest = pd.read_csv(manifest_path).sort_values("SEQN").reset_index(drop=True)
    stage2_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256(data_path) != stage2_meta.get("input_sha256"):
        raise ValueError("Input dataset changed since the Stage 2 pilot")
    if not np.array_equal(df["SEQN"].to_numpy(), manifest["SEQN"].to_numpy()):
        raise ValueError("Dataset IDs/order differ from split manifest")
    if not np.array_equal(df[TARGET].to_numpy(), manifest[TARGET].to_numpy()):
        raise ValueError("Target labels differ from split manifest")

    complete = ~df[PREDICTORS].isna().any(axis=1).to_numpy()
    if not np.array_equal(
        complete, manifest["has_all_11_predictors"].astype(bool).to_numpy()
    ):
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

    common_test = (
        df.loc[test_mask & complete].copy()
        .sort_values("SEQN").reset_index(drop=True)
    )
    if args.max_rows is not None:
        if args.max_rows < 2:
            raise SystemExit("--max-rows must be >= 2")
        if args.max_rows < len(common_test):
            indices = np.linspace(
                0, len(common_test) - 1, args.max_rows, dtype=int
            )
            common_test = common_test.iloc[indices].reset_index(drop=True)

    X_test_raw = common_test[PREDICTORS].copy()
    y_test = common_test[TARGET].astype(int).to_numpy()
    test_seqn = common_test["SEQN"].to_numpy()
    if X_background_raw.isna().any().any() or X_test_raw.isna().any().any():
        raise ValueError("SHAP background and test data must be complete")

    # Verify the same background participants used by the LR SHAP run.
    lr_bg_path = output_dir / "shap_lr_background_manifest.csv"
    if lr_bg_path.is_file():
        lr_bg = pd.read_csv(lr_bg_path).sort_values("SEQN").reset_index(drop=True)
        if not np.array_equal(
            lr_bg["SEQN"].to_numpy(), background["SEQN"].to_numpy()
        ):
            raise ValueError("RF and LR SHAP background participant IDs are not identical")

    background_manifest = background[["SEQN", TARGET]].copy()
    background_manifest["role"] = "shared_complete_case_training_background"
    background_manifest.to_csv(
        output_dir / "shap_rf_perm_background_manifest.csv", index=False
    )
    pd.DataFrame({"SEQN": test_seqn, TARGET: y_test}).to_csv(
        output_dir / "shap_rf_perm_test_manifest.csv", index=False
    )

    shap_wide = pd.DataFrame({"SEQN": test_seqn, "y_true": y_test})
    global_rows = []
    additivity_rows = []
    values_by_method = {}
    started = time.time()

    for method, filename in MODELS.items():
        print(f"Computing RF Permutation SHAP: {method} ...", flush=True)
        pipeline = joblib.load(output_dir / filename)
        pre = pipeline.named_steps["preprocess"]
        classifier = pipeline.named_steps["classifier"]
        if list(classifier.classes_) != [0, 1]:
            raise ValueError(f"Expected binary class order [0, 1], got {classifier.classes_}")

        # Reduce nested parallel overhead during repeated model calls. This is
        # an in-memory prediction setting only; it does not refit or save models.
        if "n_jobs" in classifier.get_params(deep=False):
            classifier.set_params(n_jobs=1)

        X_bg = dense_float(pre.transform(X_background_raw))
        X_test = dense_float(pre.transform(X_test_raw))
        feature_names = list(pre.get_feature_names_out())
        if X_bg.shape[1] != len(feature_names) or X_test.shape[1] != len(feature_names):
            raise ValueError("Transformed features and feature names do not match")

        # Same independent masker/background as the planned interventional game.
        masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))

        def positive_probability(x, clf=classifier):
            x = dense_float(x)
            return clf.predict_proba(x)[:, 1]

        explainer = shap.PermutationExplainer(
            positive_probability,
            masker,
            feature_names=feature_names,
            seed=PERMUTATION_SEED,
        )
        # One cycle = forward and reverse permutations. More cycles improve
        # estimates for higher-order interactions, at increased runtime.
        n_encoded_features = X_test.shape[1]
        max_evals = 2 * n_encoded_features * args.permutation_cycles + 1
        explanation = explainer(
            X_test,
            max_evals=max_evals,
            batch_size=256,
            silent=False,
        )
        encoded_values = np.asarray(explanation.values, dtype=np.float64)
        base_values = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
        predicted_probability = classifier.predict_proba(X_test)[:, 1]
        if encoded_values.shape != X_test.shape:
            raise ValueError(
                f"Unexpected Permutation SHAP values shape {encoded_values.shape}; "
                f"expected {X_test.shape}"
            )
        if len(base_values) != len(X_test):
            raise ValueError("SHAP base-value count does not match test rows")

        reconstructed = base_values + encoded_values.sum(axis=1)
        errors = np.abs(reconstructed - predicted_probability)
        max_error = float(errors.max())
        mean_error = float(errors.mean())
        if not np.isfinite(max_error) or max_error > ADD_TOLERANCE:
            raise ValueError(
                f"Permutation SHAP local-accuracy check failed for {method}: "
                f"max error={max_error:.6g}; stop before interpretation."
            )

        groups = map_original_features(feature_names)
        grouped = np.zeros((len(X_test), len(PREDICTORS)), dtype=np.float64)
        for j, feature in enumerate(PREDICTORS):
            grouped[:, j] = encoded_values[:, groups[feature]].sum(axis=1)
        if not np.allclose(
            grouped.sum(axis=1), encoded_values.sum(axis=1), atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Summing one-hot attributions changed total SHAP values")

        values_by_method[method] = grouped
        method_key = method.lower().replace("/", "_").replace(" ", "_").replace("=", "")
        for j, feature in enumerate(PREDICTORS):
            shap_wide[f"RF__{method_key}__{feature}"] = grouped[:, j]
            global_rows.append({
                "model": "RF", "method": method, "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(grouped[:, j]))),
                "mean_shap": float(np.mean(grouped[:, j])),
            })
        additivity_rows.append({
            "model": "RF", "method": method,
            "n_explained": len(X_test),
            "background_n": len(X_bg),
            "encoded_feature_count": X_test.shape[1],
            "permutation_cycles": args.permutation_cycles,
            "max_evals": max_evals,
            "max_abs_additivity_error_probability": max_error,
            "mean_abs_additivity_error_probability": mean_error,
            "tolerance": ADD_TOLERANCE,
            "audit_status": "PASS",
            "output_scale": "positive-class probability",
        })
        print(
            f"  done | max local-accuracy error={max_error:.3g} | PASS",
            flush=True,
        )

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
            "model": "RF",
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
            "shap_scale": "positive_class_probability",
            "note": (
                "Pilot only; permutation estimates use a fixed seed and limited cycles; "
                "CCA and imputation methods use different training cohort sizes/compositions."
            ),
        })

    shap_wide.to_csv(output_dir / "shap_rf_perm_values_common_test.csv", index=False)
    importance.to_csv(output_dir / "shap_rf_perm_global_importance.csv", index=False)
    pd.DataFrame(stability).to_csv(
        output_dir / "shap_rf_perm_stability_vs_cca.csv", index=False
    )
    pd.DataFrame(additivity_rows).to_csv(
        output_dir / "shap_rf_perm_additivity_audit.csv", index=False
    )

    meta = {
        "stage": "RF SHAP pilot using PermutationExplainer; not final paper results",
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "input_dataset_sha256": sha256(data_path),
        "common_test_n": int(len(X_test_raw)),
        "common_test_definition": "held-out complete-case participants from the shared split manifest",
        "background_n": BACKGROUND_N,
        "background_seed": BACKGROUND_SEED,
        "background_source": "same 100 complete-case training participants used in LR SHAP pilot",
        "explainer": "shap.PermutationExplainer with Independent masker on transformed one-hot matrix",
        "output_scale": "RandomForestClassifier positive-class probability",
        "encoded_feature_count": int(X_test.shape[1]),
        "permutation_seed": PERMUTATION_SEED,
        "permutation_cycles": args.permutation_cycles,
        "max_evals": int(max_evals),
        "local_accuracy_tolerance": ADD_TOLERANCE,
        "categorical_handling": "one-hot encoded columns; attributions summed to original predictors",
        "reference": "RF + CCA",
        "metrics": ["Spearman", "Top-5 Jaccard", "MSE-SHAP", "MAE-SHAP"],
        "caveats": [
            "PermutationExplainer guarantees local accuracy for each evaluated row, but with a finite number of permutation cycles the Shapley allocation is an estimate for models with higher-order interactions.",
            "The pilot uses one permutation cycle by default; final analysis should assess more cycles and seed sensitivity.",
            "Summed one-hot contributions are grouped encoded-feature attributions, not a separately fitted group-Shapley game.",
            "CCA training cohort is smaller than imputation-method cohorts.",
            "No repeated cross-validation, tuning, or survey weighting is included.",
            "Do not compare raw RF SHAP magnitudes with LR SHAP because LR uses log-odds while RF uses probability scale.",
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
    (output_dir / "shap_rf_perm_pilot_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== RF PERMUTATION SHAP PILOT COMPLETE ===")
    print("Explained participants:", len(X_test_raw))
    print("Shared background participants:", BACKGROUND_N)
    print("Permutation cycles:", args.permutation_cycles)
    print("Max evaluations per row:", max_evals)
    print("Output scale: positive-class probability")
    print("Stability vs RF-CCA:")
    print(pd.DataFrame(stability).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nLocal-accuracy audit:")
    print(pd.DataFrame(additivity_rows)[[
        "method", "max_abs_additivity_error_probability", "audit_status"
    ]].to_string(index=False))
    print("\nSaved under:", output_dir)
    print(
        "NOTE: Pilot only. Additivity is exact by permutation construction; "
        "the attribution allocation remains approximate with a finite cycle count."
    )


if __name__ == "__main__":
    main()
