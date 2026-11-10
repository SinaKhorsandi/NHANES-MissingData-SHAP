#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 follow-up Step 5: frequency of top-five SHAP features.

For each model (LR, RF) and missing-data method (CCA, Median/Mode, KNN k=5),
counts in how many of the 25 outer splits each original feature has rank <= 5.
Outputs one 11 x 3 CSV table per model. No participant-level data are used.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT / "output" / "stage3_final" / "shap"
    / "shap_global_importance_final.csv"
)
OUT = ROOT / "output" / "stage3_followup" / "step5_top5_frequency"

MODELS = ["LR", "RF"]
METHODS = ["CCA", "Median/Mode", "KNN k=5"]
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
EXPECTED_SPLITS = 25


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"SHAP importance file not found: {INPUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    required = {
        "split_id", "model", "method", "feature", "rank", "mean_abs_shap"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 1650:
        raise ValueError(f"Expected 1650 rows; found {len(df)}")
    if df[["split_id", "model", "method", "feature", "rank"]].isna().any().any():
        raise ValueError("Missing values found in grouping/rank columns")
    if set(df["model"].unique()) != set(MODELS):
        raise ValueError(f"Unexpected model set: {sorted(df['model'].unique())}")
    if set(df["method"].unique()) != set(METHODS):
        raise ValueError(f"Unexpected method set: {sorted(df['method'].unique())}")
    if set(df["feature"].unique()) != set(FEATURES):
        raise ValueError("Feature set does not match the 11 expected predictors")
    if df["split_id"].nunique() != EXPECTED_SPLITS:
        raise ValueError(f"Expected 25 splits; found {df['split_id'].nunique()}")
    if df.duplicated(["split_id", "model", "method", "feature"]).any():
        raise ValueError("Duplicate split/model/method/feature rows found")

    df["rank"] = pd.to_numeric(df["rank"], errors="raise")
    if not np.isfinite(df["rank"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite rank values found")
    if not df["rank"].between(1, 11).all():
        raise ValueError("Ranks must be between 1 and 11")

    # Each model-method-split must rank all 11 features exactly once.
    per_split = df.groupby(["split_id", "model", "method"])
    if len(per_split) != EXPECTED_SPLITS * len(MODELS) * len(METHODS):
        raise ValueError("Unexpected number of split/model/method groups")
    for key, group in per_split:
        if len(group) != 11 or group["rank"].nunique() != 11:
            raise ValueError(f"Invalid 11-feature ranking for group {key}")

    df["is_top5"] = df["rank"] <= 5

    output_tables = {}
    for model in MODELS:
        subset = df[df["model"] == model]
        table = pd.DataFrame({"feature": FEATURES})

        for method in METHODS:
            method_subset = subset[subset["method"] == method]
            counts = (
                method_subset.groupby("feature")["is_top5"]
                .sum()
                .reindex(FEATURES)
            )
            n_per_feature = method_subset.groupby("feature")["split_id"].nunique().reindex(FEATURES)

            if counts.isna().any() or not n_per_feature.eq(EXPECTED_SPLITS).all():
                raise ValueError(
                    f"{model}/{method}: each feature must have 25 split records"
                )

            table[method] = counts.astype(int).to_numpy()

            # Exactly five features per split must be in the top five.
            total_top5 = int(method_subset["is_top5"].sum())
            expected_top5 = EXPECTED_SPLITS * 5
            if total_top5 != expected_top5:
                raise ValueError(
                    f"{model}/{method}: expected {expected_top5} top-five "
                    f"assignments, found {total_top5}"
                )

        if table[METHODS].isna().any().any():
            raise ValueError(f"{model}: missing frequency values")
        if not table[METHODS].apply(
            lambda col: col.between(0, EXPECTED_SPLITS).all()
        ).all():
            raise ValueError(f"{model}: frequency outside 0..25")

        path = OUT / f"{model}_top5_feature_frequency_11x3.csv"
        table.to_csv(path, index=False)
        output_tables[model] = table

        print(f"\n=== {model}: TOP-5 FREQUENCY (out of 25 splits) ===")
        print(table.to_string(index=False))
        print(f"Saved: {path}")

    metadata = {
        "step": 5,
        "input_file": str(INPUT.relative_to(ROOT)),
        "n_splits": EXPECTED_SPLITS,
        "models": MODELS,
        "methods": METHODS,
        "n_features": len(FEATURES),
        "top_k": 5,
        "count_definition": "Number of outer splits with feature rank <= 5",
        "participant_level_data_used": False,
        "outputs": [
            "LR_top5_feature_frequency_11x3.csv",
            "RF_top5_feature_frequency_11x3.csv",
        ],
    }
    (OUT / "step5_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== QUALITY CHECK ===")
    checks = {
        "1650 source rows": len(df) == 1650,
        "25 splits": df["split_id"].nunique() == 25,
        "2 models": df["model"].nunique() == 2,
        "3 methods": df["method"].nunique() == 3,
        "11 features": df["feature"].nunique() == 11,
        "all groups have 25 records per feature": all(
            df.groupby(["model", "method", "feature"])["split_id"]
            .nunique().eq(25)
        ),
        "exactly 125 top-five assignments per model-method": all(
            int(g["is_top5"].sum()) == 125
            for _, g in df.groupby(["model", "method"])
        ),
        "two output tables have 11 rows and 3 method columns": all(
            len(t) == 11 and list(t.columns) == ["feature"] + METHODS
            for t in output_tables.values()
        ),
    }
    for name, passed in checks.items():
        print(("PASS" if passed else "FAIL"), "|", name)

    if not all(checks.values()):
        raise SystemExit("STEP 5 AUDIT FAILED")

    print("\nSTEP 5 AUDIT: PASS")
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
