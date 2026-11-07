"""NHANES 2017-2018: stage-one, pre-imputation dataset preparation.

Run: python prepare_nhanes.py
Put the six official XPT files inside raw_xpt/ before running.
This script does NOT train models, impute features, or drop incomplete predictors.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_xpt"
OUTPUT_DIR = ROOT / "output"

COLUMNS = {
    "DEMO_J": ["SEQN", "RIDSTATR", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
    "DIQ_J": ["SEQN", "DIQ010"],
    "BMX_J": ["SEQN", "BMXBMI", "BMXWAIST"],
    "BPX_J": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "SMQ_J": ["SEQN", "SMQ020"],
    "PAQ_J": ["SEQN", "PAQ605"],
}

# Preliminary analysis feature list. Confirm against both manuscript drafts with supervisor.
PREDICTORS = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN", "SMQ020", "PAQ605",
]

# Only variable-specific codes are recoded. Do NOT replace all 7s/9s globally!
ALLOWED_CATEGORICAL_CODES = {
    "RIAGENDR": {1, 2},
    "RIDRETH3": {1, 2, 3, 4, 6, 7},
    "DMDEDUC2": {1, 2, 3, 4, 5},
    "SMQ020": {1, 2},
    "PAQ605": {1, 2},
}
SURVEY_DESIGN_COLUMNS = ["WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]


def read_raw_files(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name + ".XPT" for name in COLUMNS if not (raw_dir / (name + ".XPT")).exists()]
    if missing:
        raise FileNotFoundError(
            "Required official NHANES 2017-2018 files are missing from raw_xpt/: "
            + ", ".join(missing)
            + ". Download the six XPT files from CDC and retry."
        )

    frames = {}
    for name, columns in COLUMNS.items():
        path = raw_dir / (name + ".XPT")
        try:
            df = pd.read_sas(path, format="xport", encoding=None)
            # The pandas SAS XPORT reader may decode an actual zero as exactly
            # 5.397605346934028e-79 (IBM-float conversion artifact). Correct
            # that exact sentinel to zero WITHOUT altering missing values.
            df = df.replace(5.397605346934028e-79, 0.0)
        except Exception as exc:
            raise RuntimeError(f"Could not read {path.name}; verify this is a real .XPT file from CDC: {exc}") from exc
        df.columns = [col.decode() if isinstance(col, bytes) else str(col) for col in df.columns]
        not_found = [col for col in columns if col not in df.columns]
        if not_found:
            raise ValueError(f"{name} does not have expected columns: {not_found}. Check survey cycle and codebook.")
        df = df[columns].copy()
        if df["SEQN"].isna().any() or df["SEQN"].duplicated().any():
            raise ValueError(f"{name}: SEQN must be non-missing and unique for person-level merging.")
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="raise").astype("int64")
        frames[name] = df
    return frames


def prepare_from_frames(frames: dict[str, pd.DataFrame], out_dir: Path, raw_dir: Path | None = None) -> dict[str, int]:
    """Accepts in-memory source frames, allowing offline tests without real data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, columns in COLUMNS.items():
        if name not in frames:
            raise KeyError(f"Missing frame: {name}")
        missing_cols = set(columns) - set(frames[name].columns)
        if missing_cols:
            raise ValueError(f"{name}: missing columns {sorted(missing_cols)}")
        if frames[name]["SEQN"].isna().any() or frames[name]["SEQN"].duplicated().any():
            raise ValueError(f"Invalid person-level SEQN in {name}")

    base = frames["DEMO_J"].copy()
    adult = base.loc[pd.to_numeric(base["RIDAGEYR"], errors="coerce") >= 20].copy()
    joined = adult.copy()

    join_audit = []
    for name in ["DIQ_J", "BMX_J", "BPX_J", "SMQ_J", "PAQ_J"]:
        right = frames[name].copy()
        right["SEQN"] = pd.to_numeric(right["SEQN"], errors="raise").astype("int64")
        matched = adult["SEQN"].isin(right["SEQN"]).sum()
        join_audit.append({
            "source": name,
            "rows_in_raw_file": len(right),
            "adults_in_demo_matched_by_seqn": int(matched),
            "adults_in_demo_not_matched": int(len(adult) - matched),
        })
        joined = joined.merge(right, on="SEQN", how="left", validate="one_to_one")
    pd.DataFrame(join_audit).to_csv(out_dir / "join_audit.csv", index=False, encoding="utf-8-sig")

    target_numeric = pd.to_numeric(joined["DIQ010"], errors="coerce")
    code_counts = target_numeric.fillna(-1).value_counts(dropna=False).sort_index()
    pd.DataFrame({"DIQ010_raw_code": code_counts.index, "count_among_adults": code_counts.values}).to_csv(
        out_dir / "target_raw_code_counts.csv", index=False, encoding="utf-8-sig"
    )

    eligible = joined.loc[target_numeric.isin([1, 2])].copy()
    eligible["diabetes_reported"] = (pd.to_numeric(eligible["DIQ010"]) == 1).astype(int)
    code_audit = []
    for col, allowed in ALLOWED_CATEGORICAL_CODES.items():
        raw_values = pd.to_numeric(eligible[col], errors="coerce")
        invalid = raw_values.notna() & ~raw_values.isin(allowed)
        for value, count in raw_values[invalid].value_counts().items():
            code_audit.append({"variable": col, "code_replaced_with_missing": value, "count": int(count)})
        eligible[col] = raw_values.where(~invalid, np.nan)
    pd.DataFrame(code_audit, columns=["variable", "code_replaced_with_missing", "count"]).to_csv(
        out_dir / "recoded_invalid_values.csv", index=False, encoding="utf-8-sig"
    )

    # PROVISIONAL choice: mean of all available BP readings, using pandas skipna.
    # Diastolic zero is a documented possible value; do not automatically set 0 to NaN.
    systolic = [f"BPXSY{i}" for i in range(1, 5)]
    diastolic = [f"BPXDI{i}" for i in range(1, 5)]
    eligible["BP_SYS_MEAN"] = eligible[systolic].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    eligible["BP_DIA_MEAN"] = eligible[diastolic].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Keep a traceable version with selected raw readings for later supervisor review.
    eligible.to_csv(out_dir / "merged_selected_with_raw_bp.csv", index=False, encoding="utf-8-sig")

    output_columns = ["SEQN", "DIQ010", "diabetes_reported"] + PREDICTORS + SURVEY_DESIGN_COLUMNS
    cleaned = eligible[output_columns].copy()
    cleaned.to_csv(out_dir / "nhanes_stage1_clean_with_missing.csv", index=False, encoding="utf-8-sig")

    # Create the proposed MEC-examined analytic subset automatically, rather
    # than relying on a manual one-off terminal command. Keep both cohorts.
    examined = eligible.loc[eligible["RIDSTATR"] == 2, output_columns + ["RIDSTATR"]].copy()
    examined.to_csv(out_dir / "nhanes_examined_clean_with_missing.csv", index=False, encoding="utf-8-sig")
    examined_missing = pd.DataFrame({
        "variable": PREDICTORS,
        "missing_n": [int(examined[col].isna().sum()) for col in PREDICTORS],
        "missing_percent": [round(examined[col].isna().mean() * 100, 2) for col in PREDICTORS],
        "observed_n": [int(examined[col].notna().sum()) for col in PREDICTORS],
    }).sort_values("missing_percent", ascending=False)
    examined_missing.to_csv(out_dir / "examined_missingness_report.csv", index=False, encoding="utf-8-sig")
    group_report = []
    for status in (0, 1):
        group = examined.loc[examined["diabetes_reported"] == status]
        group_report.append({
            "diabetes_status": status,
            "participants": len(group),
            "with_missing_data": int(group[PREDICTORS].isna().any(axis=1).sum()),
            "missing_percent": round(group[PREDICTORS].isna().any(axis=1).mean() * 100, 2),
        })
    pd.DataFrame(group_report).to_csv(out_dir / "missingness_by_diabetes_status.csv", index=False, encoding="utf-8-sig")


    n_total = len(cleaned)
    missing_report = pd.DataFrame({
        "variable": PREDICTORS,
        "missing_n": [int(cleaned[col].isna().sum()) for col in PREDICTORS],
        "missing_percent": [round(float(cleaned[col].isna().mean() * 100), 2) if n_total else np.nan for col in PREDICTORS],
        "observed_n": [int(cleaned[col].notna().sum()) for col in PREDICTORS],
    })
    missing_report = missing_report.sort_values("missing_percent", ascending=False)
    missing_report.to_csv(out_dir / "missingness_report.csv", index=False, encoding="utf-8-sig")

    any_missing = cleaned[PREDICTORS].isna().any(axis=1)
    summary = {
        "original_demo_rows": int(len(base)),
        "adults_age_20_plus": int(len(adult)),
        "adults_with_valid_DIQ010_1_or_2": n_total,
        "diabetes_reported_yes": int(cleaned["diabetes_reported"].sum()),
        "diabetes_reported_no": int((cleaned["diabetes_reported"] == 0).sum()),
        "adults_with_at_least_one_missing_predictor": int(any_missing.sum()),
        "adults_with_all_predictors_observed": int((~any_missing).sum()),
        "number_of_initial_predictors": len(PREDICTORS),
        "examined_adults_with_valid_target": len(examined),
        "examined_with_missing_predictors": int(examined[PREDICTORS].isna().any(axis=1).sum()),
        "examined_complete_cases": int(examined[PREDICTORS].notna().all(axis=1).sum()),
    }
    flow = pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])
    flow.to_csv(out_dir / "cohort_flow.csv", index=False, encoding="utf-8-sig")

    # Useful QA flags, without deleting any observations from the main dataset.
    ranges = []
    for variable, flag in [
        ("RIDAGEYR", lambda x: x < 20),
        ("BMXBMI", lambda x: x <= 0),
        ("BMXWAIST", lambda x: x <= 0),
        ("BP_SYS_MEAN", lambda x: x <= 0),
        ("BP_DIA_MEAN", lambda x: x < 0),
    ]:
        observed = pd.to_numeric(cleaned[variable], errors="coerce")
        ranges.append({"variable": variable, "values_for_manual_review_n": int(flag(observed).fillna(False).sum())})
    pd.DataFrame(ranges).to_csv(out_dir / "range_review_flags.csv", index=False, encoding="utf-8-sig")

    sha_lines = []
    if raw_dir is not None:
        for name in COLUMNS:
            path = raw_dir / (name + ".XPT")
            if path.exists():
                hasher = hashlib.sha256()
                with path.open("rb") as infile:
                    for chunk in iter(lambda: infile.read(1024 * 1024), b""):
                        hasher.update(chunk)
                sha_lines.append(f"{path.name}\t{hasher.hexdigest()}")
    (out_dir / "raw_file_sha256.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    (out_dir / "preparation_notes.txt").write_text(
        "NHANES 2017-2018 preliminary data preparation (stage 1)\n"
        "No predictor imputation, no complete-case deletion, no modeling or SHAP done.\n"
        "DIQ010: 1 -> diabetes_reported=1, 2 -> 0; other responses excluded.\n"
        "Only adults age >= 20 with valid target included.\n"
        "Five categorical predictors are validated using documented variable-specific values.\n"
        "Blood pressure is provisionally the mean of available 1-4 readings; confirm with supervisor.\n"
        "Exact pandas XPORT zero artifact 5.397605346934028e-79 is recoded to numeric 0 before processing.\n"
        "RIDSTATR is retained in the examined subset; all adults are preserved in the broad subset.\n"
        "Diastolic 0 is not automatically treated as missing (allowed in CDC codebook).\n"
        "RIDRETH3 and DMDEDUC2 included provisionally; drafts differ in feature choices.\n"
        "WTINT2YR, WTMEC2YR, SDMVSTRA and SDMVPSU retained for later survey design review; NOT predictors.\n"
        "Missing due to questionnaire skips/ineligibility needs domain-specific review before imputation.\n"
        f"Pandas version: {pd.__version__}\n"
        f"Python version: {sys.version.split()[0]}\n"
        + "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    data = read_raw_files(RAW_DIR)
    summary = prepare_from_frames(data, OUTPUT_DIR, RAW_DIR)
    print("Stage one completed. Outputs saved to:", OUTPUT_DIR)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("IMPORTANT: The cleaned CSV intentionally still contains missing predictor values.")


if __name__ == "__main__":
    main()
