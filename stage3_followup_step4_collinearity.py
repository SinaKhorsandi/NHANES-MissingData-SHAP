#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4A: Pearson/Spearman correlation and 11-predictor VIF audit."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "output" / "nhanes_examined_clean_with_missing.csv"
OUT = ROOT / "output" / "stage3_followup" / "step4_collinearity"

FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "BP_SYS_MEAN",
    "BP_DIA_MEAN",
    "SMQ020",
    "PAQ605",
]

def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate conventional VIF for each of the 11 stored numeric columns.

    Each feature is regressed on the other ten features plus an intercept.
    VIF is invariant to linear rescaling. Categorical predictors retain their
    existing numeric coding to satisfy the requested 11-column VIF table.
    """
    x = frame.astype(float).to_numpy()
    # Standardize columns for numerical stability; VIF is scale-invariant.
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    if np.any(stds == 0):
        bad = [FEATURES[i] for i, value in enumerate(stds) if value == 0]
        raise ValueError(f"Zero-variance predictors found: {bad}")
    z = (x - means) / stds

    rows = []
    for j, feature in enumerate(FEATURES):
        y = z[:, j]
        others = np.delete(z, j, axis=1)
        design = np.column_stack([np.ones(len(z)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        r2 = min(max(r2, 0.0), 1.0)
        vif = float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({
            "feature": feature,
            "n_complete_cases": int(len(frame)),
            "r_squared_auxiliary_regression": r2,
            "VIF": vif,
        })
    return pd.DataFrame(rows)

def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    missing_cols = sorted(set(FEATURES) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing predictor columns: {missing_cols}")

    complete = df[FEATURES].notna().all(axis=1)
    cc = df.loc[complete, FEATURES].copy()
    for col in FEATURES:
        cc[col] = pd.to_numeric(cc[col], errors="raise")

    if len(cc) != 4031:
        raise ValueError(
            f"Expected 4031 complete cases across all 11 predictors; found {len(cc)}"
        )
    if not np.isfinite(cc.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found among complete cases")

    pair = cc[["BMXWAIST", "BMXBMI"]]
    pearson = pearsonr(pair["BMXWAIST"], pair["BMXBMI"])
    spearman = spearmanr(pair["BMXWAIST"], pair["BMXBMI"])

    corr = pd.DataFrame([
        {
            "feature_1": "BMXWAIST",
            "feature_2": "BMXBMI",
            "method": "Pearson",
            "correlation": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
            "n": int(len(cc)),
        },
        {
            "feature_1": "BMXWAIST",
            "feature_2": "BMXBMI",
            "method": "Spearman",
            "correlation": float(spearman.statistic),
            "p_value": float(spearman.pvalue),
            "n": int(len(cc)),
        },
    ])

    vif = calculate_vif(cc)

    corr_path = OUT / "step4_bmi_waist_correlations.csv"
    vif_path = OUT / "step4_vif_11_predictors.csv"
    corr.to_csv(corr_path, index=False)
    vif.to_csv(vif_path, index=False)

    metadata = {
        "step": "4A",
        "input_file": str(INPUT.relative_to(ROOT)),
        "complete_case_rule": "non-missing on all 11 predictors",
        "n_complete_cases": int(len(cc)),
        "predictors": FEATURES,
        "correlation_pair": ["BMXWAIST", "BMXBMI"],
        "vif_definition": (
            "Conventional VIF from each original numeric predictor regressed "
            "on the other ten original numeric columns plus intercept."
        ),
        "categorical_coding_note": (
            "VIF uses the 11 predictors in their stored numeric coding; "
            "categorical variables are not one-hot expanded in this requested "
            "11-row table."
        ),
        "outputs": [corr_path.name, vif_path.name],
    }
    (OUT / "step4_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("=== STEP 4A: BMI-WAIST CORRELATION ===")
    print(corr.to_string(index=False))
    print("\n=== STEP 4A: VIF FOR 11 PREDICTORS ===")
    print(vif.to_string(index=False))
    print("\n=== QUALITY CHECK ===")
    print("Complete cases:", len(cc))
    print("Predictors:", len(FEATURES))
    print("Correlation rows:", len(corr))
    print("VIF rows:", len(vif))
    checks = {
        "4031 complete cases": len(cc) == 4031,
        "11 VIF rows": len(vif) == 11,
        "2 correlation rows": len(corr) == 2,
        "No missing correlation values": not corr.isna().any().any(),
        "No missing VIF values": not vif.isna().any().any(),
        "All VIF values finite": np.isfinite(vif["VIF"].to_numpy()).all(),
    }
    for label, passed in checks.items():
        print(("PASS" if passed else "FAIL"), "|", label)
    if not all(checks.values()):
        raise SystemExit("STEP 4A AUDIT FAILED")
    print("\nSTEP 4A AUDIT: PASS")
    print("Saved to:", OUT)

if __name__ == "__main__":
    main()
