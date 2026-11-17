#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 4 / Step 8: MICE diagnostic tables from saved Stage 5 records.

READ-ONLY preflight and safe, short aggregate run. No MICE fitting, model
training, SHAP, or participant-level output. This is a descriptive diagnostic
report, NOT proof of MICE convergence or of missing-at-random assumptions.

The requested three tables are collected in ONE small CSV, distinguished by
`table_id`: (1) iterations/early stop, (2) observed/imputed distributions of
BMXWAIST and INDFMPIR, (3) post-MICE waist/BMI Pearson correlation versus
0.905 in the original complete cases. All summaries are arithmetic averages
of 25 splits x 5 draws, as specified in the advisor revision tasks.

Note: sklearn IterativeImputer(sample_posterior=True) does not use the usual
early stopping rule. All draws reaching max_iter=10 is EXPECTED and does not
by itself demonstrate convergence or non-convergence.

Only creates a NEW directory inside output/stage4_mice/ when --mode run.
PROVISIONAL; ADVISOR APPROVAL PENDING.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

SCHEMA = "sina-stage4-step8-v1"
STAGE5_SCHEMA = "sina-stage4-step5-v1"
ADVISOR_STATUS = "PENDING; PROVISIONAL"
N_SPLITS = 25
N_DRAWS = 5
MAX_ITER = 10
BASE_SEED = 2026
COMPLETE_CASE_WAIST_BMI_R = 0.905   # Advisor Step 8 specified reference
EXPECTED_DIAG_SHA256 = "122a9114148c0e1fb7f437cd9b26c173b0af24929217b9ff972474dce85957ca"
EXPECTED_STEP5_SUMMARY_SHA256 = "69937d483df14c837ea7b93e59e20151a494480450064e151e487e8243f1b27b"
EXPECTED_STAGE5_MANIFEST_SHA256 = "6e2daaa7cc25dafc139154094438b0b1a94733aec1dc61b715e09958dcd48e5f"
MANIFEST_FILES = (
    "mice_cv_metrics_25_splits.csv",
    "mice_imputer_diagnostics_25_splits.csv",
    "mice_inner_tuning_25_splits.csv",
    "step5_summary.json",
)
DIAGNOSTICS_FIELDS = (
    "split_id", "imputation_id", "seed", "n_iter_", "max_iter",
    "early_stop", "imputer_seconds", "n_train",
    "BMXWAIST_observed_n", "BMXWAIST_imputed_n",
    "BMXWAIST_observed_mean", "BMXWAIST_observed_sd",
    "BMXWAIST_imputed_mean", "BMXWAIST_imputed_sd",
    "INDFMPIR_observed_n", "INDFMPIR_imputed_n",
    "INDFMPIR_observed_mean", "INDFMPIR_observed_sd",
    "INDFMPIR_imputed_mean", "INDFMPIR_imputed_sd",
    "BMXWAIST_BMXBMI_pearson_after_imputation",
)
INT_FIELDS = (
    "split_id", "imputation_id", "seed", "n_iter_", "max_iter",
    "early_stop", "n_train", "BMXWAIST_observed_n",
    "BMXWAIST_imputed_n", "INDFMPIR_observed_n", "INDFMPIR_imputed_n",
)
AGGREGATE_CSV = "mice_diagnostics_step8_three_tables.csv"
SUMMARY_FILE = "step8_summary.json"
CONFIG_FILE = "step8_run_config.json"
HASH_FILE = "step8_aggregate_sha256.json"
REPORT_FIELDS = (
    "table_id", "table", "variable", "measure", "mean_over_125",
    "sd_across_125", "min_over_125", "max_over_125",
    "reference_value", "units", "description",
)


def stop(message: str) -> None:
    raise RuntimeError("STOP: " + message)


def log(message: str) -> None:
    print(message, flush=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        stop(f"missing or symlinked file: {path}")


def json_object(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        stop(f"JSON must be an object: {path}")
    return result


def rooted(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else root / expanded).resolve()


def within(parent: Path, child: Path) -> bool:
    return child != parent and child.is_relative_to(parent)


def safe_relative_file(parent: Path, rel: str) -> Path:
    rel_path = PurePosixPath(rel)
    if (rel_path.is_absolute() or ".." in rel_path.parts or "." in rel_path.parts
            or not rel_path.parts):
        stop(f"unsafe checkpoint path: {rel}")
    path = parent.joinpath(*rel_path.parts)
    if not within(parent.resolve(), path.resolve()):
        stop(f"checkpoint file escapes directory: {rel}")
    require_file(path)
    return path


def checked_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"STOP: invalid numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        stop(f"non-finite numeric value in {label}")
    return result


def checked_int(value: Any, label: str) -> int:
    x = checked_float(value, label)
    if not x.is_integer():
        stop(f"nonintegral integer value in {label}: {value!r}")
    return int(x)


def csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(DIAGNOSTICS_FIELDS):
            stop(f"unexpected Stage 5 diagnostic CSV schema: {path}\n"
                 f"actual={reader.fieldnames}")
        rows = list(reader)
    if any(None in row for row in rows):
        stop(f"malformed CSV row: {path}")
    return rows


def numeric_rows(raw: list[dict[str, str]]) -> list[dict[str, float | int]]:
    return [{col: checked_int(row[col], col) if col in INT_FIELDS
             else checked_float(row[col], col)
             for col in DIAGNOSTICS_FIELDS}
            for row in raw]


def canonical_split_files() -> set[str]:
    names = {"mice_performance.csv", "mice_imputer_diagnostics.csv", "mice_inner_tuning.csv"}
    for j in range(1, N_DRAWS + 1):
        names.add(f"models/MICE_draw{j:02d}_imputer_bundle.joblib")
        for model in ("LR", "RF"):
            names.add(f"models/{model}_draw{j:02d}_classifier.joblib")
    if len(names) != 18:
        stop("internal count error for expected split artifacts")
    return names


def validate_splits(source: Path, aggregate_raw: list[dict[str, str]]) -> dict[str, str]:
    by_split: dict[int, list[dict[str, str]]] = {
        s: [row for row in aggregate_raw if checked_int(row["split_id"], "split_id") == s]
        for s in range(1, N_SPLITS + 1)
    }
    expected_files = canonical_split_files()
    markers: dict[str, str] = {}
    for split in range(1, N_SPLITS + 1):
        folder = source / f"split_{split:03d}"
        if not folder.is_dir() or folder.is_symlink():
            stop(f"Stage 5 split checkpoint directory missing: {folder}")
        marker_file = folder / "split_complete.json"
        marker = json_object(marker_file)
        if (marker.get("schema") != STAGE5_SCHEMA or
            checked_int(marker.get("split_id"), "marker.split_id") != split or
            checked_int(marker.get("diagnostic_rows"), "marker.diagnostic_rows") != N_DRAWS or
            checked_int(marker.get("n_imputations"), "marker.n_imputations") != N_DRAWS):
            stop(f"wrong split checkpoint metadata: {folder}")
        hashes = marker.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != expected_files:
            stop(f"unexpected checkpoint file list: split {split}")
        for name in sorted(expected_files):
            actual_path = safe_relative_file(folder, name)
            if digest(actual_path) != hashes[name]:
                stop(f"checkpoint SHA256 mismatch split {split}: {name}")
        per_split = csv_rows(folder / "mice_imputer_diagnostics.csv")
        current = sorted(by_split[split], key=lambda r: int(r["imputation_id"]))
        per_split = sorted(per_split, key=lambda r: int(r["imputation_id"]))
        if len(per_split) != N_DRAWS or len(current) != N_DRAWS:
            stop(f"split {split} must have five diagnostic rows")
        for a, b in zip(per_split, current):
            for field in DIAGNOSTICS_FIELDS:
                if field in INT_FIELDS:
                    if checked_int(a[field], field) != checked_int(b[field], field):
                        stop(f"per-split/aggregate mismatch split {split}: {field}")
                else:
                    x = checked_float(a[field], field)
                    y = checked_float(b[field], field)
                    if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
                        stop(f"per-split/aggregate mismatch split {split}: {field}")
        markers[f"split_{split:03d}/split_complete.json"] = digest(marker_file)
    return markers


def validate_diagnostics(rows: list[dict[str, float | int]]) -> None:
    if len(rows) != N_SPLITS * N_DRAWS:
        stop(f"expected 125 diagnostics, got {len(rows)}")
    expected = {(s, j) for s in range(1, N_SPLITS + 1)
                for j in range(1, N_DRAWS + 1)}
    got = {(int(r["split_id"]), int(r["imputation_id"])) for r in rows}
    if got != expected or len(got) != len(rows):
        stop("missing or duplicated split/imputation keys")
    for row in rows:
        split, j = int(row["split_id"]), int(row["imputation_id"])
        expected_seed = BASE_SEED + 100 * split + j
        if int(row["seed"]) != expected_seed:
            stop(f"incorrect imputation seed for split {split}, draw {j}")
        count_iter, maximum = int(row["n_iter_"]), int(row["max_iter"])
        if maximum != MAX_ITER or not (1 <= count_iter <= MAX_ITER):
            stop(f"unexpected n_iter/max_iter for split {split}, draw {j}")
        if int(row["early_stop"]) != int(count_iter < maximum):
            stop(f"early_stop flag inconsistent for split {split}, draw {j}")
        if not 100 < int(row["n_train"]) or row["imputer_seconds"] < 0:
            stop(f"invalid training sample size / duration at split {split}")
        for variable in ("BMXWAIST", "INDFMPIR"):
            observed = int(row[f"{variable}_observed_n"])
            imputed = int(row[f"{variable}_imputed_n"])
            if observed < 2 or imputed < 2 or observed + imputed != int(row["n_train"]):
                stop(f"observed/imputed counts do not partition training data: {split}/{variable}")
            if row[f"{variable}_observed_sd"] <= 0 or row[f"{variable}_imputed_sd"] <= 0:
                stop(f"invalid within-split SD at {split}/{variable}")
        r = row["BMXWAIST_BMXBMI_pearson_after_imputation"]
        if not -1 <= r <= 1:
            stop(f"invalid Pearson correlation at split {split}")
    for split in range(1, N_SPLITS + 1):
        group = [r for r in rows if int(r["split_id"]) == split]
        # The OBSERVED training data (and missingness masks) must be the same
        # for all 5 posterior draws; only the imputed values may differ.
        invariant = ("n_train", "BMXWAIST_observed_n", "BMXWAIST_imputed_n",
                     "BMXWAIST_observed_mean", "BMXWAIST_observed_sd",
                     "INDFMPIR_observed_n", "INDFMPIR_imputed_n",
                     "INDFMPIR_observed_mean", "INDFMPIR_observed_sd")
        for field in invariant:
            reference = float(group[0][field])
            if any(not math.isclose(float(r[field]), reference, rel_tol=1e-12, abs_tol=1e-12)
                   for r in group[1:]):
                stop(f"observed training value changed across draws: split={split} field={field}")


def verify_input(root: Path, source_arg: Path) -> tuple[Path, list[dict[str, float | int]], dict[str, Any]]:
    stage4 = (root / "output" / "stage4_mice").resolve()
    source = rooted(root, source_arg)
    if not within(stage4, source) or not source.is_dir() or source.is_symlink():
        stop(f"Stage 5 source must be a real directory under {stage4}")
    manifest_file = source / "step5_aggregate_sha256.json"
    manifest = json_object(manifest_file)
    if digest(manifest_file) != EXPECTED_STAGE5_MANIFEST_SHA256:
        stop("Stage 5 aggregate manifest differs from independently audited original")
    if set(manifest) != set(MANIFEST_FILES):
        stop("unexpected Stage 5 aggregate manifest keys")
    for filename in MANIFEST_FILES:
        path = safe_relative_file(source, filename)
        if digest(path) != manifest[filename]:
            stop(f"Stage 5 aggregate SHA256 mismatch: {filename}")
    if manifest["mice_imputer_diagnostics_25_splits.csv"] != EXPECTED_DIAG_SHA256:
        stop("diagnostics digest differs from approved provisional Stage 5 audit")
    if manifest["step5_summary.json"] != EXPECTED_STEP5_SUMMARY_SHA256:
        stop("Stage 5 summary differs from original audited run")
    summary = json_object(source / "step5_summary.json")
    if (summary.get("schema") != STAGE5_SCHEMA or
        checked_int(summary.get("completed_splits"), "completed_splits") != N_SPLITS or
        checked_int(summary.get("mice_draws"), "mice_draws") != N_DRAWS or
        checked_int(summary.get("imputer_diagnostic_rows"), "imputer_diagnostic_rows") != 125 or
        summary.get("has_nonfinite_auc_or_brier") is not False):
        stop("Stage 5 summary is invalid")
    raw = csv_rows(source / "mice_imputer_diagnostics_25_splits.csv")
    records = numeric_rows(raw)
    validate_diagnostics(records)
    marker_hashes = validate_splits(source, raw)
    log("STEP 8 SOURCE PASS | original audited Stage 5 four aggregate SHA256 files; 25/25 split SHA256 checkpoints")
    log("STEP 8 DIAGNOSTICS PASS | 125/125 rows; 25 splits x 5 draws; seeds, iterations, observed/imputed counts, finite moments and correlations")
    provenance = {
        "stage5_dir": str(source),
        "stage5_aggregate_manifest_sha256": digest(manifest_file),
        "stage5_diagnostics_sha256": digest(source / "mice_imputer_diagnostics_25_splits.csv"),
        "stage5_summary_sha256": digest(source / "step5_summary.json"),
        "stage5_split_checkpoint_markers_sha256": marker_hashes,
    }
    return source, records, provenance


def desc(values: list[float]) -> dict[str, float]:
    if len(values) != 125:
        stop("expected 125 values per diagnostic")
    return {"mean": statistics.fmean(values),
            "sd_across_125": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def calculate(records: list[dict[str, float | int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def add(table_id: int, table: str, variable: str, measure: str,
            vals: list[float], units: str, description: str,
            reference: float | None = None) -> dict[str, float]:
        stats = desc(vals)
        result.append({
            "table_id": table_id, "table": table, "variable": variable,
            "measure": measure, "mean_over_125": stats["mean"],
            "sd_across_125": stats["sd_across_125"],
            "min_over_125": stats["min"], "max_over_125": stats["max"],
            "reference_value": "" if reference is None else reference,
            "units": units, "description": description,
        })
        return stats

    n_iter = add(1, "iteration_diagnostics", "MICE", "n_iter_",
                 [float(r["n_iter_"]) for r in records], "iterations",
                 "Posterior-sampling MICE: reaching max_iter is not a convergence test.")
    early_stop = add(1, "iteration_diagnostics", "MICE", "early_stop_percent",
                     [100.0 * float(r["early_stop"]) for r in records], "percent",
                     "Early stopping criterion is not active with sample_posterior=True.")

    distribution: dict[str, Any] = {}
    for variable in ("BMXWAIST", "INDFMPIR"):
        distribution[variable] = {}
        unit = "cm" if variable == "BMXWAIST" else "income_poverty_ratio"
        for observed_or_imputed in ("observed", "imputed"):
            prefix = f"{variable}_{observed_or_imputed}"
            count_stats = add(2, "observed_vs_imputed", variable,
                              f"{observed_or_imputed}_n",
                              [float(r[f"{prefix}_n"]) for r in records],
                              "participants_per_split_draw", "Mean subgroup size per split and draw.")
            mean_stats = add(2, "observed_vs_imputed", variable,
                             f"{observed_or_imputed}_mean",
                             [float(r[f"{prefix}_mean"]) for r in records],
                             unit, "Mean of 125 within-draw subgroup means.")
            sd_stats = add(2, "observed_vs_imputed", variable,
                           f"{observed_or_imputed}_sd",
                           [float(r[f"{prefix}_sd"]) for r in records],
                           unit, "Mean of 125 within-draw sample SDs (ddof=1).")
            distribution[variable][observed_or_imputed] = {
                "mean_n": count_stats["mean"],
                "mean_of_means": mean_stats["mean"],
                "mean_of_sample_sds": sd_stats["mean"],
                "sd_across_125_of_means": mean_stats["sd_across_125"],
                "sd_across_125_of_sds": sd_stats["sd_across_125"],
            }

    corr_values = [float(r["BMXWAIST_BMXBMI_pearson_after_imputation"])
                   for r in records]
    corr_stats = add(3, "waist_bmi_correlation", "BMXWAIST_vs_BMXBMI",
                     "pearson_after_imputation", corr_values, "Pearson_r",
                     "Mean of 125 within-draw training-set correlations.",
                     COMPLETE_CASE_WAIST_BMI_R)
    diff_stats = add(3, "waist_bmi_correlation", "BMXWAIST_vs_BMXBMI",
                     "pearson_minus_reference",
                     [r - COMPLETE_CASE_WAIST_BMI_R for r in corr_values],
                     "Pearson_r_difference", "Difference from advisor's 0.905 complete-case reference.",
                     COMPLETE_CASE_WAIST_BMI_R)

    if len(result) != 16:
        stop(f"internal error: expected 16 CSV rows, got {len(result)}")
    summary = {
        "schema": SCHEMA,
        "advisor_status": ADVISOR_STATUS,
        "source": "Stage 5 archived imputer diagnostics; no refit, no SHAP rerun",
        "table_count": 3,
        "table_rows": {"iteration_diagnostics": 2, "observed_vs_imputed": 12,
                       "waist_bmi_correlation": 2},
        "n_output_csv_rows": 16,
        "n_splits": N_SPLITS,
        "n_draws_per_split": N_DRAWS,
        "n_split_draw_records": N_SPLITS * N_DRAWS,
        "posterior_sampling": True,
        "max_iter": MAX_ITER,
        "iteration_diagnostics": {
            "mean_n_iter": n_iter["mean"],
            "sd_n_iter_across_125": n_iter["sd_across_125"],
            "n_early_stop": sum(int(r["early_stop"]) for r in records),
            "percent_early_stop": early_stop["mean"],
            "interpretation": "Early stopping is disabled for sklearn IterativeImputer(sample_posterior=True); this metric does not measure convergence.",
        },
        "distributions": distribution,
        "waist_bmi_correlation": {
            "mean_pearson_r": corr_stats["mean"],
            "sd_pearson_r_across_125": corr_stats["sd_across_125"],
            "min_pearson_r": corr_stats["min"],
            "max_pearson_r": corr_stats["max"],
            "complete_case_reference_r_from_advisor": COMPLETE_CASE_WAIST_BMI_R,
            "mean_difference_from_reference": diff_stats["mean"],
            "reference_is_external_fixed_value_not_reestimated": True,
        },
        "statistical_caveat": "125 split-draw records are NOT independent. Report descriptive means/SDs only; no independent-draw inference or convergence claim.",
        "participant_level_data_exported": False,
    }
    return result, summary


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"


def write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as fh:
        fh.write(content)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def emit_output(root: Path, stage5_dir: Path, out_arg: Path,
                rows: list[dict[str, Any]], summary: dict[str, Any],
                provenance: dict[str, Any]) -> Path:
    stage4 = (root / "output" / "stage4_mice").resolve()
    out = rooted(root, out_arg)
    if not within(stage4, out) or within(stage5_dir, out) or within(out, stage5_dir):
        stop("output must be separate from Stage 5 and inside output/stage4_mice/")
    if out.exists() or out.is_symlink():
        stop(f"refusing to overwrite existing output: {out}")
    if not out.parent.is_dir() or out.parent.is_symlink():
        stop(f"output parent directory must already exist: {out.parent}")
    temporary = out.parent / f".work_step8_{out.name}_{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        stop(f"temporary output path already exists: {temporary}")
    temporary.mkdir()
    try:
        write_report(temporary / AGGREGATE_CSV, rows)
        write_new(temporary / SUMMARY_FILE, json_text(summary))
        config = {
            "schema": SCHEMA,
            "advisor_status": ADVISOR_STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": digest(Path(__file__).resolve()),
            "source": provenance,
            "reference_waist_bmi_r": COMPLETE_CASE_WAIST_BMI_R,
            "sample_posterior": True,
            "per_split_and_per_draw_means_not_independent": True,
            "output_contains_no_participant_level_values": True,
        }
        write_new(temporary / CONFIG_FILE, json_text(config))
        hashes = {name: digest(temporary / name)
                  for name in (AGGREGATE_CSV, SUMMARY_FILE, CONFIG_FILE)}
        write_new(temporary / HASH_FILE, json_text(hashes))
        if any(digest(temporary / name) != expected for name, expected in hashes.items()):
            stop("post-write SHA256 validation failed")
        if out.exists():
            stop(f"output became occupied during run: {out}")
        temporary.rename(out)
    except Exception:
        # The newly created temporary output is the only path we may remove.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    log(f"STEP 8 PASS | 3 tables / 16 rows | 25 splits x 5 draws | "
        f"mean_n_iter={summary['iteration_diagnostics']['mean_n_iter']:.4f} | "
        f"early_stop_percent={summary['iteration_diagnostics']['percent_early_stop']:.2f} | "
        f"waist_bmi_r={summary['waist_bmi_correlation']['mean_pearson_r']:.6f} | SHA256 PASS")
    log(f"STEP 8 OUTPUT | {out}")
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    parser.add_argument("--stage5-dir", required=True, type=Path,
                        help="Original validated Stage 5 full 25-split run directory")
    parser.add_argument("--out", type=Path, help="NEW Stage 8 output directory (run only)")
    args = parser.parse_args()
    if args.mode == "preflight" and args.out is not None:
        parser.error("preflight is READ-ONLY: do not specify --out")
    if args.mode == "run" and args.out is None:
        parser.error("run requires a NEW --out directory")
    stage5_dir, records, provenance = verify_input(root, args.stage5_dir)
    rows, summary = calculate(records)
    if args.mode == "preflight":
        log("STEP 8 PREFLIGHT PASS | three-table descriptive report ready; READ-ONLY; no models/imputation/SHAP computed; NO files written")
        return
    emit_output(root, stage5_dir, args.out, rows, summary, provenance)


if __name__ == "__main__":
    main()
