#!/usr/bin/env python3
"""Advisor follow-up: BMI-waist correlation and exact LR design-matrix VIF.

Operate in repository root, on the original Stage 3 examined cohort.
VIF is computed with an auxiliary intercept for *all* encoded LR columns.
Perfect aliasing due to full-level one-hot encoding is retained as +inf.
No participant-level output is saved.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from stage3_cv import PREDICTORS, build_preprocessor

EXPECTED_SHA = "a017e04e8c9cbbf5d1d5a85e7bbc241629cc8679bdd21b1a3f96cbf0a2338752"


def main() -> None:
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    assert branch == "stage3-followup", f"STOP: wrong Git branch: {branch}"

    source = Path("output/nhanes_examined_clean_with_missing.csv")
    out = Path("advisor_remaining_csv")
    corr_path = out / "collinearity_correlation.csv"
    vif_path = out / "collinearity_vif.csv"
    assert source.is_file(), f"STOP: missing input: {source}"
    assert not corr_path.exists() and not vif_path.exists(), "STOP: output exists"

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA, f"STOP: unexpected dataset SHA-256: {digest}"

    frame = pd.read_csv(source)
    assert len(frame) == 5098
    complete = frame[PREDICTORS].notna().all(axis=1)
    cc = frame.loc[complete, PREDICTORS].copy()
    assert len(cc) == 4031
    assert np.isfinite(cc.to_numpy(dtype=float)).all()

    a = cc["BMXWAIST"].to_numpy(dtype=float)
    b = cc["BMXBMI"].to_numpy(dtype=float)
    corr = pd.DataFrame([
        {"feature_1": "BMXWAIST", "feature_2": "BMXBMI", "method": "Pearson", "correlation": float(pearsonr(a,b).statistic), "n_complete_cases": 4031},
        {"feature_1": "BMXWAIST", "feature_2": "BMXBMI", "method": "Spearman", "correlation": float(spearmanr(a,b).statistic), "n_complete_cases": 4031},
    ])

    preprocessor = build_preprocessor("CCA", "LR")
    encoded = preprocessor.fit_transform(cc)
    if sparse.issparse(encoded):
        encoded = encoded.toarray()
    X = np.asarray(encoded, dtype=float)
    columns = list(preprocessor.get_feature_names_out())
    assert X.shape == (4031, len(columns))
    assert len(columns) == len(set(columns)) == 23
    assert np.isfinite(X).all()
    assert sum(name.startswith("categorical__") for name in columns) == 17

    records = []
    for j, column in enumerate(columns):
        y = X[:, j]
        design = np.column_stack((np.ones(len(y)), np.delete(X, j, axis=1)))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        sst = float(np.sum((y - y.mean()) ** 2))
        assert sst > 0, f"STOP: constant LR design column: {column}"
        r2 = float(np.clip(1.0 - float(resid @ resid) / sst, 0.0, 1.0))
        perfectly_aliased = (1.0 - r2) <= 1e-10
        vif = float("inf") if perfectly_aliased else 1.0 / (1.0 - r2)
        records.append({
            "design_column": column,
            "vif": vif,
            "n_complete_cases": 4031,
            "auxiliary_r_squared": r2,
            "status": "perfect_alias_full_onehot" if perfectly_aliased else "finite",
        })
    vif = pd.DataFrame(records)
    categorical = vif["design_column"].str.startswith("categorical__")
    assert np.isinf(vif.loc[categorical, "vif"].to_numpy()).all()
    assert np.isfinite(vif.loc[~categorical, "vif"].to_numpy()).all()
    assert len(corr) == 2 and len(vif) == 23
    assert not vif.duplicated("design_column").any()

    out.mkdir(exist_ok=True)
    corr.to_csv(corr_path, index=False, float_format="%.12g")
    vif.to_csv(vif_path, index=False, float_format="%.12g")

    reloaded_corr = pd.read_csv(corr_path)
    reloaded_vif = pd.read_csv(vif_path)
    assert len(reloaded_corr) == 2 and len(reloaded_vif) == 23
    assert reloaded_vif["vif"].isna().sum() == 0

    for method, coef in zip(corr["method"], corr["correlation"]):
        print(f"PASS | {method} correlation | n=4031 | r={coef:.9f}")
    print("PASS | Actual CCA/LR design: 23 columns (6 standardized numerical, 17 full one-hot)")
    print("PASS | VIF: 6 finite numeric + 17 infinite categorical (full one-hot aliasing)")
    for col in ["numeric__BMXBMI", "numeric__BMXWAIST"]:
        v = float(vif.loc[vif["design_column"].eq(col), "vif"].iloc[0])
        print(f"PASS | {col}: VIF={v:.6f}")
    print("STEP 3A: PASS")
    print("OUTPUT:", corr_path)
    print("OUTPUT:", vif_path)


if __name__ == "__main__":
    main()
