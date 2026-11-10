#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 follow-up, Step 4B: stored LR coefficients from outer split 001.

Read-only with respect to the primary CV results. Reads the three already
fitted Logistic Regression pipelines from stage3_final/models/split_001/.
Extracts the coefficients for the transformed numeric BMXBMI and BMXWAIST
columns, which are standardized by the training-fold preprocessing pipeline.
No model is retrained and no participant-level data are written.

For KNN, scaling occurs *before* KNN imputation; these coefficients are per
one observed-training SD of the predictor, not necessarily per post-imputation
SD of the transformed column. This distinction is recorded in the output.

Example (from project root):
  python stage3_followup_step4_lr_coefficients.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METHODS = ["CCA", "Median/Mode", "KNN k=5"]
NUMERIC = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN"
]
FEATURES = ["BMXWAIST", "BMXBMI"]
STEP = {"CCA": None, "Median/Mode": "scaler", "KNN k=5": "scale_for_distance"}
EXPECTED_SPLIT = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_fitted_model(path: Path, method: str) -> list[dict]:
    # Only load models created locally by the trusted Stage 3 analysis.
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline)}: {path}")
    if set(["preprocess", "classifier"]) - set(pipeline.named_steps):
        raise TypeError(f"Missing pipeline steps in {path}")

    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression in {path}")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError(f"Unexpected LR class ordering: {classifier.classes_}")
    if classifier.coef_.ndim != 2 or classifier.coef_.shape[0] != 1:
        raise ValueError("Expected a binary LR coefficient vector")

    transformed_names = list(preprocessor.get_feature_names_out())
    coef = classifier.coef_[0].astype(float)
    if len(transformed_names) != len(coef):
        raise ValueError("Transformed-feature count does not equal coef_ length")
    if len(transformed_names) != len(set(transformed_names)):
        raise ValueError("Duplicate transformed feature names")

    numeric_transformer = preprocessor.named_transformers_["numeric"]
    if method == "CCA":
        scaler = numeric_transformer
    elif isinstance(numeric_transformer, Pipeline):
        scaler = numeric_transformer.named_steps[STEP[method]]
    else:
        raise TypeError(f"Unexpected numeric transformer for {method}")
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler for {method}")
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(f"Scaler for {method} has not been fitted")

    transformer_numeric_cols = None
    for name, _, cols in preprocessor.transformers_:
        if name == "numeric":
            transformer_numeric_cols = list(cols)
            break
    if transformer_numeric_cols != NUMERIC:
        raise ValueError(
            f"Unexpected numeric predictor order in {path}: "
            f"{transformer_numeric_cols} != {NUMERIC}"
        )
    if len(scaler.scale_) != len(NUMERIC):
        raise ValueError("Scaler does not match numeric columns")

    rows = []
    for feature in FEATURES:
        key = f"numeric__{feature}"
        matches = [i for i, name in enumerate(transformed_names) if name == key]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one transformed column {key}; got {matches}")
        ix = matches[0]
        numeric_ix = NUMERIC.index(feature)
        value = float(coef[ix])
        fitted_mean = float(scaler.mean_[numeric_ix])
        fitted_scale = float(scaler.scale_[numeric_ix])
        if not all(np.isfinite([value, fitted_mean, fitted_scale])) or fitted_scale <= 0:
            raise ValueError(f"Invalid scaler/coef values for {feature} in {path}")
        rows.append({
            "model": "LR",
            "split_id": EXPECTED_SPLIT,
            "method": method,
            "feature": feature,
            "transformed_feature": key,
            "coefficient_standardized": value,
            "coefficient_unit": "log-odds per one numeric scaling unit",
            "scaler_step": "numeric_StandardScaler" if method == "CCA" else STEP[method],
            "scaler_training_mean": fitted_mean,
            "scaler_training_sd": fitted_scale,
            "lr_regularization_C": float(classifier.C),
            "n_transformed_features": int(len(coef)),
        })
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-out", type=Path, default=Path("output/stage3_final"))
    p.add_argument(
        "--out", type=Path,
        default=Path("output/stage3_followup/step4_coefficients")
    )
    args = p.parse_args()
    cv_dir = resolve(root, args.cv_out)
    out_dir = resolve(root, args.out)
    metrics_file = cv_dir / "cv_metrics_final.csv"
    if not metrics_file.is_file():
        raise FileNotFoundError(metrics_file)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. "
            "Choose a new --out directory; existing files will not be overwritten."
        )

    metrics = pd.read_csv(metrics_file)
    needed_cols = {"split_id", "model", "method", "model_artifact", "best_params_json"}
    if needed_cols - set(metrics.columns):
        raise ValueError(f"CV metrics are missing: {sorted(needed_cols - set(metrics.columns))}")
    subset = metrics.loc[
        metrics["split_id"].eq(EXPECTED_SPLIT) & metrics["model"].eq("LR")
    ]
    if len(subset) != 3 or set(subset["method"]) != set(METHODS):
        raise ValueError("Expected exactly 3 distinct LR methods in split 001")

    detail: list[dict] = []
    model_provenance: list[dict] = []
    for method in METHODS:
        row = subset.loc[subset["method"].eq(method)].iloc[0]
        rel_path = Path(str(row["model_artifact"]))
        artifact = (cv_dir / rel_path).resolve()
        if not artifact.is_relative_to(cv_dir):
            raise ValueError(f"Model artifact outside primary CV directory: {artifact}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing fitted model: {artifact}")
        model_rows = read_fitted_model(artifact, method)
        params = json.loads(row["best_params_json"])
        if not np.isclose(model_rows[0]["lr_regularization_C"], float(params["C"])):
            raise ValueError(f"Stored model C differs from CV metrics: {method}")
        detail.extend(model_rows)
        model_provenance.append({
            "method": method,
            "model_artifact": str(rel_path),
            "model_sha256": sha256(artifact),
            "lr_C_from_primary_tuning": float(params["C"]),
        })

    details = pd.DataFrame(detail)
    compact = (
        details.pivot(index="method", columns="feature", values="coefficient_standardized")
        .reindex(METHODS)[FEATURES]
        .reset_index()
        .rename(columns={
            "BMXWAIST": "BMXWAIST_standardized_LR_coefficient",
            "BMXBMI": "BMXBMI_standardized_LR_coefficient",
        })
    )
    compact.insert(0, "split_id", EXPECTED_SPLIT)
    compact.insert(0, "model", "LR")

    checks = {
        "exactly_three_methods": len(compact) == 3 and list(compact["method"]) == METHODS,
        "six_feature_coefficients": len(details) == 6,
        "no_duplicate_method_feature": not details.duplicated(["method", "feature"]).any(),
        "finite_coefficients": bool(np.isfinite(details["coefficient_standardized"]).all()),
        "finite_scaler_values": bool(np.isfinite(details[["scaler_training_mean", "scaler_training_sd"]]).all().all()),
        "positive_scaler_sd": bool((details["scaler_training_sd"] > 0).all()),
        "three_model_hashes": len(model_provenance) == 3,
    }
    for label, status in checks.items():
        print(("PASS" if status else "FAIL") + " | " + label)
    if not all(checks.values()):
        raise RuntimeError("Step 4B quality gate failed")

    out_dir.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_dir / "step4_lr_standardized_coefficients_split001.csv", index=False)
    details.to_csv(out_dir / "step4_lr_coefficients_detail_split001.csv", index=False)
    metadata = {
        "analysis": "Stage 3 follow-up Step 4B",
        "source": "Three fitted LR pipelines from original primary outer split 001",
        "primary_cv_dir": str(cv_dir),
        "models_refit": False,
        "outcomes_or_participant_data_exported": False,
        "coefficient_scale": "positive-class log-odds per one pipeline-scaled numeric unit",
        "median_scaling": "Median imputation then StandardScaler on fitted training data",
        "knn_scaling": (
            "StandardScaler fitted on observed training numeric values before KNN "
            "imputation; hence a coefficient is per pre-imputation observed SD, "
            "not necessarily one post-imputation SD"
        ),
        "models": model_provenance,
        "quality_checks": checks,
    }
    (out_dir / "step4_lr_coefficient_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== STEP 4B: STORED LR COEFFICIENTS, SPLIT 001 ===")
    print(compact.to_string(index=False, float_format=lambda x: f"{x:.7f}"))
    print("\nSTEP 4B AUDIT: PASS")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
