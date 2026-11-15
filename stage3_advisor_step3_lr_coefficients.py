#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advisor remaining tasks, Step 3B: LR standardized coefficients.

Read original, locally generated Stage 3 LR model artifacts, no refitting.
If ALL 75 LR artifacts exist (25 splits x 3 methods), summarize across 25.
Otherwise use only outer split 001, and state this explicitly in stdout and
in the n_splits_used column of the aggregate CSV.

Coefficients are log-odds per numeric scaling unit from the fitted pipeline.
CCA: StandardScaler on training complete cases.
Median/Mode: imputation followed by StandardScaler.
KNN k=5: StandardScaler BEFORE KNN imputation; therefore scaling is based on
observed training values, not necessarily post-imputation standard deviation.

No participant-level records are read or written; only fitted models and
non-participant-level CV metadata are consumed. Load only trusted local
joblib files that the original analysis produced.

Usage from the Git repository root (on stage3-followup):
    python stage3_advisor_step3_lr_coefficients.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stage3_cv import NUMERIC

METHODS = ("CCA", "Median/Mode", "KNN k=5")
FEATURES = ("BMXWAIST", "BMXBMI")
SCALE_STEPS = {"Median/Mode": "scaler", "KNN k=5": "scale_for_distance"}
OUT_COLS = [
    "model", "method", "feature", "mean_coefficient_standardized",
    "sd_coefficient_standardized", "n_splits_used",
]


def get_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-out", type=Path, default=Path("output/stage3_final"))
    p.add_argument("--out-dir", type=Path, default=Path("advisor_remaining_csv"))
    return p


def read_lr_model(path: Path, method: str, split: int, expected_c: float) -> list[dict]:
    # joblib.load uses pickle; ONLY load files produced by the trusted local CV.
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected Pipeline: {path}")
    if not {"preprocess", "classifier"}.issubset(pipeline.named_steps):
        raise ValueError(f"Missing model pipeline steps: {path}")
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(f"Unexpected preprocessor: {path}")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression: {path}")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError(f"Unexpected target class ordering: {path}")
    if not np.isclose(float(classifier.C), expected_c, rtol=1e-12, atol=1e-12):
        raise ValueError(f"C from artifact disagrees with primary CV metrics: {path}")

    names = list(preprocessor.get_feature_names_out())
    coef = np.asarray(classifier.coef_, dtype=float)
    if coef.shape != (1, len(names)):
        raise ValueError(f"Binary LR coefficients do not match feature columns: {path}")
    if len(names) != len(set(names)) or not np.isfinite(coef).all():
        raise ValueError(f"Duplicate names or non-finite coefficients: {path}")

    actual_numeric = next(
        (list(cols) for name, _, cols in preprocessor.transformers_ if name == "numeric"),
        None,
    )
    if actual_numeric != list(NUMERIC):
        raise ValueError(f"Unexpected numerical feature columns: {path}")

    numeric = preprocessor.named_transformers_["numeric"]
    if method == "CCA":
        scaler = numeric
    else:
        if not isinstance(numeric, Pipeline):
            raise TypeError(f"Expected numeric Pipeline for {method}: {path}")
        scaler = numeric.named_steps.get(SCALE_STEPS[method])
        if method == "Median/Mode" and list(numeric.named_steps) != ["imputer", "scaler"]:
            raise ValueError(f"Unexpected Median/Mode scaling order: {path}")
        if method == "KNN k=5" and list(numeric.named_steps) != ["scale_for_distance", "knn_imputer"]:
            raise ValueError(f"Unexpected KNN scaling order: {path}")
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"No fitted numeric StandardScaler for {method}: {path}")
    if len(scaler.scale_) != len(NUMERIC) or not np.isfinite(scaler.scale_).all():
        raise ValueError(f"Invalid scaling information: {path}")
    if not (np.asarray(scaler.scale_) > 0).all():
        raise ValueError(f"Non-positive numeric scale: {path}")

    result = []
    for feature in FEATURES:
        transformed_name = f"numeric__{feature}"
        ix = [i for i, name in enumerate(names) if name == transformed_name]
        if len(ix) != 1:
            raise ValueError(f"Expected one column {transformed_name}: {path}")
        result.append({
            "model": "LR",
            "split_id": split,
            "method": method,
            "feature": feature,
            "coefficient_standardized": float(coef[0, ix[0]]),
        })
    return result


def main() -> None:
    args = get_argument_parser().parse_args()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != "stage3-followup":
        raise RuntimeError(f"STOP: wrong Git branch: {branch}")

    root = Path.cwd().resolve()
    cv_dir = (root / args.cv_out).resolve()
    out = (root / args.out_dir).resolve()
    metrics_file = cv_dir / "cv_metrics_final.csv"
    target = out / "lr_waist_bmi_coefficients.csv"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"STOP: primary CV metrics missing: {metrics_file}")
    if target.exists():
        raise FileExistsError(f"STOP: output exists, refusing overwrite: {target}")
    if not cv_dir.is_dir():
        raise FileNotFoundError(cv_dir)

    metrics = pd.read_csv(metrics_file)
    mandatory = {"split_id", "model", "method", "model_artifact", "best_params_json"}
    if not mandatory.issubset(metrics.columns):
        raise ValueError(f"Missing columns from primary CV: {mandatory - set(metrics.columns)}")
    lr = metrics.loc[metrics["model"] == "LR"].copy()
    if lr.duplicated(["split_id", "method"]).any():
        raise ValueError("Duplicate LR split-method CV metadata")
    if not set(lr["method"]).issubset(METHODS):
        raise ValueError("Unexpected LR methods")
    if lr["split_id"].isna().any() or not (lr["split_id"] == lr["split_id"].astype(int)).all():
        raise ValueError("Invalid split IDs")
    lr["split_id"] = lr["split_id"].astype(int)

    artifact_map: dict[tuple[int, str], tuple[Path, float]] = {}
    for row in lr.itertuples(index=False):
        split = int(row.split_id)
        method = str(row.method)
        rel = Path(str(row.model_artifact))
        path = (cv_dir / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(cv_dir) or path.suffix != ".joblib":
            raise ValueError(f"Unexpected model artifact path for split {split}, {method}")
        if rel.parent != Path("models") / f"split_{split:03d}":
            raise ValueError(f"Model artifact not in expected split directory: {rel}")
        params = json.loads(row.best_params_json)
        c = float(params["C"])
        if not np.isfinite(c) or c <= 0:
            raise ValueError("Invalid tuned C")
        artifact_map[(split, method)] = (path, c)

    all_keys = [(i, method) for i in range(1, 26) for method in METHODS]
    available = lambda key: key in artifact_map and artifact_map[key][0].is_file()
    if all(available(key) for key in all_keys):
        splits = list(range(1, 26))
        print("PASS | All 75 original LR models available | using 25 splits")
    else:
        missing = sum(not available(key) for key in all_keys)
        print(f"INFO | {missing} of 75 LR artifacts unavailable | using split 001 only")
        splits = [1]
        if not all(available((1, method)) for method in METHODS):
            raise FileNotFoundError("STOP: not all three original LR models in split 001 are available")

    detail_rows = []
    for split in splits:
        for method in METHODS:
            path, expected_c = artifact_map[(split, method)]
            detail_rows.extend(read_lr_model(path, method, split, expected_c))
    detail = pd.DataFrame(detail_rows)
    if len(detail) != len(splits) * len(METHODS) * len(FEATURES):
        raise ValueError("Incorrect number of model-feature coefficients")
    if detail.duplicated(["split_id", "method", "feature"]).any():
        raise ValueError("Duplicate LR coefficient record")
    if not np.isfinite(detail["coefficient_standardized"].to_numpy()).all():
        raise ValueError("Invalid coefficient value")

    grouped = detail.groupby(["model", "method", "feature"], as_index=False).agg(
        mean_coefficient_standardized=("coefficient_standardized", "mean"),
        sd_coefficient_standardized=("coefficient_standardized", "std"),
        n_splits_used=("split_id", "nunique"),
    )
    grouped = grouped[OUT_COLS].sort_values(["model", "method", "feature"])
    if len(grouped) != 6 or not grouped["n_splits_used"].eq(len(splits)).all():
        raise ValueError("Incorrect six-row coefficient summary")
    if len(splits) == 25 and not np.isfinite(grouped["sd_coefficient_standardized"]).all():
        raise ValueError("Invalid coefficient standard deviations")
    if len(splits) == 1 and not grouped["sd_coefficient_standardized"].isna().all():
        raise ValueError("Single-split SD must be undefined, not zero")

    out.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(target, index=False, float_format="%.12g")
    reopened = pd.read_csv(target)
    if len(reopened) != 6 or list(reopened.columns) != OUT_COLS:
        raise ValueError("Output CSV round-trip check failed")
    if not reopened["n_splits_used"].eq(len(splits)).all():
        raise ValueError("Output split count check failed")
    if not np.allclose(
        reopened["mean_coefficient_standardized"].to_numpy(),
        grouped["mean_coefficient_standardized"].to_numpy(),
        rtol=1e-10, atol=1e-10,
    ):
        raise ValueError("Output numeric round-trip check failed")
    if len(splits) == 25 and not np.allclose(
        reopened["sd_coefficient_standardized"].to_numpy(),
        grouped["sd_coefficient_standardized"].to_numpy(),
        rtol=1e-10, atol=1e-10,
    ):
        raise ValueError("Output SD round-trip check failed")

    print("PASS | 3 methods x 2 predictors = 6 CSV rows")
    print("PASS | Original fitted models; no model refitting")
    print("PASS | Model-specific fitted numeric StandardScaler verified")
    print(f"PASS | n_splits_used = {len(splits)}")
    if len(splits) == 1:
        print("INFO | Split 001 only: coefficient SD left empty (undefined for n=1)")
    print("PASS | No participant-level output")
    print("STEP 3B: PASS")
    print("OUTPUT:", target.relative_to(root) if target.is_relative_to(root) else target)
    print("\n" + grouped.to_string(index=False, float_format=lambda x: f"{x:.7f}"))


if __name__ == "__main__":
    main()
