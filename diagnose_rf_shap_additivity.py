#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic only: investigate RF TreeExplainer additivity without changing outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
import sklearn

TARGET = "diabetes_reported"
NUMERIC = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN"]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2", "SMQ020", "PAQ605"]
PREDICTORS = NUMERIC + CATEGORICAL
BG_N = 100
BG_SEED = 2026
DIAG_N = 128


def dense_dtype(x, dtype):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)


def get_class_values(values, base_values, n_rows, n_features):
    values = np.asarray(values)
    base_values = np.asarray(base_values)
    print("  SHAP values shape:", values.shape)
    print("  SHAP base_values shape:", base_values.shape)
    if values.ndim == 3 and values.shape[-1] == 2:
        v = [values[:, :, 0], values[:, :, 1]]
    elif values.ndim == 3 and values.shape[1] == 2:
        v = [values[:, 0, :], values[:, 1, :]]
    elif values.ndim == 2 and values.shape == (n_rows, n_features):
        v = [None, values]
    else:
        raise ValueError("Unexpected SHAP values shape; do not interpret this run.")

    if base_values.ndim == 2 and base_values.shape == (n_rows, 2):
        b = [base_values[:, 0], base_values[:, 1]]
    elif base_values.ndim == 1 and len(base_values) == 2:
        b = [np.full(n_rows, base_values[0]), np.full(n_rows, base_values[1])]
    elif base_values.ndim == 1 and len(base_values) == n_rows:
        b = [None, base_values]
    elif base_values.ndim == 0:
        b = [None, np.full(n_rows, float(base_values))]
    else:
        raise ValueError("Unexpected SHAP base_values shape; do not interpret this run.")
    return v, b


def main():
    root = Path(__file__).resolve().parent
    out = root / "output" / "stage2_pilot_v1"
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    manifest_path = out / "split_manifest.csv"
    model_path = out / "rf_median_mode_pipeline.joblib"
    meta_path = out / "pilot_metadata.json"
    for p in [data_path, manifest_path, model_path, meta_path]:
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")

    df = pd.read_csv(data_path).sort_values("SEQN").reset_index(drop=True)
    manifest = pd.read_csv(manifest_path).sort_values("SEQN").reset_index(drop=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not np.array_equal(df["SEQN"].to_numpy(), manifest["SEQN"].to_numpy()):
        raise ValueError("Dataset and split manifest participant order differ")
    if not np.array_equal(df[TARGET].to_numpy(), manifest[TARGET].to_numpy()):
        raise ValueError("Target differs from split manifest")

    train_mask = manifest["split"].eq("train").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()
    complete = ~df[PREDICTORS].isna().any(axis=1).to_numpy()
    bg_pool = df.loc[train_mask & complete].copy()
    bg = bg_pool.sample(n=BG_N, random_state=BG_SEED).sort_values("SEQN").reset_index(drop=True)
    test = df.loc[test_mask & complete].copy().sort_values("SEQN").reset_index(drop=True)
    # Deterministic diagnostic subset, spread across the ordered common test set.
    ix = np.linspace(0, len(test) - 1, min(DIAG_N, len(test)), dtype=int)
    test = test.iloc[ix].reset_index(drop=True)

    pipe = joblib.load(model_path)
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["classifier"]
    bg_transformed = pre.transform(bg[PREDICTORS])
    test_transformed = pre.transform(test[PREDICTORS])
    names = list(pre.get_feature_names_out())

    print("=== RF SHAP ADDITIVITY DIAGNOSTIC (Median/Mode only) ===")
    print("SHAP:", shap.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("NumPy:", np.__version__)
    print("Classifier:", type(clf).__name__)
    print("Classes:", clf.classes_)
    print("n_features_in_:", clf.n_features_in_)
    print("Transformed feature count:", len(names))
    print("Original transformed matrix type:", type(test_transformed).__name__)
    print("Original transformed matrix dtype:", getattr(test_transformed, "dtype", "n/a"))
    print("Background participants:", len(bg))
    print("Diagnostic test participants:", len(test), "(subset of common complete-case test)")
    depths = [tree.tree_.max_depth for tree in clf.estimators_]
    print("RF trees:", len(depths), "| max depth:", max(depths), "| median depth:", float(np.median(depths)))

    results = {}
    predictions_by_dtype = {}
    for dtype in [np.float64, np.float32]:
        label = np.dtype(dtype).name
        X_bg = dense_dtype(bg_transformed, dtype)
        X_test = dense_dtype(test_transformed, dtype)
        pred = np.asarray(clf.predict_proba(X_test), dtype=float)
        predictions_by_dtype[label] = pred
        print(f"\n--- Input dtype: {label} ---")
        print("X_bg:", X_bg.shape, X_bg.dtype, "| X_test:", X_test.shape, X_test.dtype)
        print("Prediction range class 1:", float(pred[:, 1].min()), float(pred[:, 1].max()))

        explainer = shap.TreeExplainer(
            clf,
            data=X_bg,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=names,
        )
        exp = explainer(X_test)
        vals, bases = get_class_values(exp.values, exp.base_values, len(X_test), X_test.shape[1])
        dtype_results = {}
        for c in [0, 1]:
            if vals[c] is None or bases[c] is None:
                print(f"  class {c}: output layout does not expose both class contributions")
                continue
            reconstructed = bases[c] + vals[c].sum(axis=1)
            err = np.abs(reconstructed - pred[:, c])
            dtype_results[c] = err
            print(
                f"  class {c} additivity: max={err.max():.8g}; "
                f"mean={err.mean():.8g}; p95={np.quantile(err, .95):.8g}"
            )
        results[label] = dtype_results

    if "float64" in predictions_by_dtype and "float32" in predictions_by_dtype:
        delta = np.abs(predictions_by_dtype["float64"] - predictions_by_dtype["float32"])
        print("\nMax prediction difference float64 vs float32:", float(delta.max()))

    print("\nDiagnostic finished. No model, dataset, or existing result file was modified.")
    print("Send the complete terminal output for investigation; do not disable the additivity check.")


if __name__ == "__main__":
    main()
