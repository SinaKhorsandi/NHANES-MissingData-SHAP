#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Step 11A: read-only reproducibility and release preflight.

DOES NOT rerun models, SHAP, perform Git commits, or publish data.
Checks trusted baseline Stage 3 + Stage 5..10 outputs, complete Stage 5/6
checkpoint hashes, run-to-source lineage, code/env evidence, git hygiene and
capacity for a NEW full independent replay.

The advisor's Step 11 PASS additionally REQUIRES a new end-to-end rerun,
independent comparisons, README, a comprehensive sha256sum -c, final commit,
and a clean git status. This preflight does NOT satisfy that final gate.

All local/private participant-level artifacts remain local; do not upload them.
Advisor status: PENDING; PROVISIONAL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "stage4_mice"
HASH = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_MANIFESTS = {
    "stage5": ("step5_aggregate_sha256.json", "6e2daaa7cc25dafc139154094438b0b1a94733aec1dc61b715e09958dcd48e5f", 4),
    "stage6": ("step6_aggregate_sha256.json", "cb3c9077a5471e5caa4e9a51d79bdd583e0daf476da52c8a8d40794f407ca31e", 4),
    "stage7": ("step7_aggregate_sha256.json", "73061e16170d46f0ea2e5b2f74dde85db34fa19c2a778d7f8dc85a3e21cf8379", 3),
    "stage8": ("step8_aggregate_sha256.json", "61ae6209992af4384f89d06d1408d89287c2c3eb6c78a68166e759768a664822", 3),
    "stage9": ("step9_aggregate_sha256.json", "fea043e889721ed0d769f221a2735976f569ee16591ad59de700c35f0727b423", 3),
    "stage10": ("step10_aggregate_sha256.json", "bc6cc1273499cfc0bde1a7081c1021bf7a8f699a6aef0cf5edee32e361bae846", 19),
}
AGGREGATES = {
    "stage5": {"mice_cv_metrics_25_splits.csv", "mice_imputer_diagnostics_25_splits.csv", "mice_inner_tuning_25_splits.csv", "step5_summary.json"},
    "stage6": {"mice_shap_draw_additivity_25_splits.csv", "mice_shap_pooled_additivity_25_splits.csv", "mice_shap_global_importance_25_splits.csv", "step6_summary.json"},
    "stage7": {"shap_stability_metrics_4_methods_25_splits.csv", "step7_summary.json", "step7_run_config.json"},
    "stage8": {"mice_diagnostics_step8_three_tables.csv", "step8_summary.json", "step8_run_config.json"},
    "stage9": {"step9_corrected_intervals_4_methods.csv", "step9_summary.json", "step9_run_config.json"},
    "stage10": {
        "table1_predictive_performance_4_methods.csv", "table2_paired_NB_Holm_4_methods.csv",
        "table3_shap_stability_4_methods.csv", "table4_shap_mean_abs_by_feature_4_methods.csv",
        "table5_shap_top5_frequency_4_methods.csv", "table6_lr_waist_bmi_coefficients_4_methods.csv",
        "table7_waist_bmi_correlation_context.csv", "table8_collinearity_vif_complete_case_reference.csv",
        "table9_collinearity_correlation_complete_case_reference.csv",
        "figure1_mean_abs_shap_LR.png", "figure1_mean_abs_shap_RF.png",
        "figure2_feature_rank_bump_LR.png", "figure2_feature_rank_bump_RF.png",
        "figure1_mean_abs_shap_LR.pdf", "figure1_mean_abs_shap_RF.pdf",
        "figure2_feature_rank_bump_LR.pdf", "figure2_feature_rank_bump_RF.pdf",
        "step10_run_config.json", "step10_summary.json",
    },
}
SOURCE_FILES = {
    "stage3_cv": ("stage3", "cv_metrics_final.csv"),
    "stage3_importance": ("stage3", "shap/shap_global_importance_final.csv"),
    "stage3_stability": ("stage3", "shap/shap_stability_metrics_by_split_final.csv"),
    "stage5_cv": ("stage5", "mice_cv_metrics_25_splits.csv"),
    "stage6_importance": ("stage6", "mice_shap_global_importance_25_splits.csv"),
    "stage7_stability": ("stage7", "shap_stability_metrics_4_methods_25_splits.csv"),
    "step9_results": ("stage9", "step9_corrected_intervals_4_methods.csv"),
    "legacy_performance": ("legacy_report", "table1_predictive_performance.csv"),
    "legacy_stability": ("legacy_report", "table2_shap_stability.csv"),
    "legacy_importance_mean": ("legacy_numeric", "shap_mean_abs_by_feature.csv"),
    "legacy_top5": ("legacy_numeric", "shap_top5_frequency.csv"),
    "legacy_lr_coeff": ("legacy_numeric", "lr_waist_bmi_coefficients.csv"),
    "legacy_corr": ("legacy_numeric", "collinearity_correlation.csv"),
    "legacy_vif": ("legacy_numeric", "collinearity_vif.csv"),
    "stage5_manifest": ("stage5", "step5_aggregate_sha256.json"),
    "stage6_manifest": ("stage6", "step6_aggregate_sha256.json"),
    "stage7_manifest": ("stage7", "step7_aggregate_sha256.json"),
    "stage8_manifest": ("stage8", "step8_aggregate_sha256.json"),
    "stage9_manifest": ("stage9", "step9_aggregate_sha256.json"),
}
SCRIPT_NAMES = {
    "stage7": "stage4_mice_stability4.py",
    "stage8": "stage4_mice_diagnostics_step8.py",
    "stage9": "stage4_mice_inference_step9.py",
    "stage10": "stage4_mice_reporting_step10.py",
}


def log(message: str) -> None:
    print(message, flush=True)


def stop(message: str) -> None:
    raise RuntimeError("STOP: " + message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_user(arg: str) -> Path:
    v = Path(arg).expanduser()
    return (v if v.is_absolute() else ROOT / v).resolve()


def require_file(path: Path, expected: str | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        stop("missing or symlinked file " + str(path))
    got = digest(path)
    if expected is not None and got != expected:
        stop("SHA256 mismatch " + str(path) + " observed=" + got + " expected=" + expected)
    return got


def json_object(path: Path, expected: str | None = None) -> dict[str, Any]:
    require_file(path, expected)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        stop("unparseable JSON " + str(path) + ": " + str(exc))
    if not isinstance(result, dict):
        stop("expected JSON object " + str(path))
    return result


def confined_file(base: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        stop("invalid artifact relative name: " + repr(rel))
    relpath = Path(rel)
    if relpath.is_absolute() or any(t in ("..", ".") for t in rel.split("/")):
        stop("artifact manifest contains unsafe path: " + repr(rel))
    target = base / relpath
    if not target.resolve().is_relative_to(base.resolve()) or target.is_symlink():
        stop("artifact manifest escapes source directory: " + repr(rel))
    return target


def verify_manifest(root: Path, stage: str) -> dict[str, str]:
    name, pinned, expected_n = EXPECTED_MANIFESTS[stage]
    entries = json_object(root / name, pinned)
    if set(entries) != AGGREGATES[stage] or len(entries) != expected_n:
        stop("unexpected aggregate manifest content for " + stage)
    for rel, expected in entries.items():
        if not isinstance(expected, str) or HASH.fullmatch(expected) is None:
            stop("invalid SHA256 manifest entry: " + stage + "/" + rel)
        require_file(confined_file(root, rel), expected)
    return entries


def verify_checkpoints(root: Path, stage: str) -> tuple[int, int, int]:
    expected_marker_schema = "sina-stage4-step5-v1" if stage == "stage5" else "sina-stage4-step6-v1"
    nfiles = 0
    total_bytes = 0
    for split_id in range(1, 26):
        folder = root / f"split_{split_id:03d}"
        if folder.is_symlink() or not folder.is_dir():
            stop(f"{stage}: missing/symlinked split_{split_id:03d}")
        marker = json_object(folder / "split_complete.json")
        # The Stage 5 and Stage 6 schemas are not interchangeable. Match the
        # run config schema as the primary identity and enforce split_id.
        if marker.get("split_id") != split_id:
            stop(f"{stage}: invalid split marker identity {split_id}")
        if marker.get("schema") != expected_marker_schema:
            stop(f"{stage}: invalid split marker schema {split_id}: {marker.get('schema')!r}")
        sha = marker.get("sha256")
        target_count = 18 if stage == "stage5" else 24
        if not isinstance(sha, dict) or len(sha) != target_count:
            stop(f"{stage}: unexpected checkpoint file count for split {split_id}: {len(sha) if isinstance(sha,dict) else 'not-object'}")
        for rel, expected in sha.items():
            if not isinstance(expected, str) or HASH.fullmatch(expected) is None:
                stop(f"{stage}: invalid SHA256 for split {split_id}/{rel}")
            target = confined_file(folder, rel)
            if not target.is_file():
                stop(f"{stage}: missing checkpoint artifact {target}")
            if target.stat().st_size <= 0:
                stop(f"{stage}: empty checkpoint artifact {target}")
            require_file(target, expected)
            nfiles += 1
            total_bytes += target.stat().st_size
        if split_id % 5 == 0:
            log(f"STEP 11 HASH PROGRESS | {stage} {split_id}/25 split checkpoints verified")
    return (25, nfiles, total_bytes)


def check_schema(summary: dict[str, Any], key: str, target: Any, label: str) -> None:
    if summary.get(key) != target:
        stop(f"{label} {key} mismatch: observed {summary.get(key)!r} expected {target!r}")


def audit_sources(paths: dict[str, Path], benchmark_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for name, path in paths.items():
        if path.is_symlink() or not path.is_dir():
            stop("missing/symlinked source directory " + name + ": " + str(path))
    man = {}
    for stage in EXPECTED_MANIFESTS:
        man[stage] = verify_manifest(paths[stage], stage)
    log("STEP 11 SOURCE PASS | Stage 5/6-corrected/7/8/9/10 pinned aggregate manifests and all 36 aggregate files SHA256 verified")

    s10 = json_object(paths["stage10"] / "step10_summary.json")
    s10_cfg = json_object(paths["stage10"] / "step10_run_config.json")
    check_schema(s10, "schema", "sina-stage4-step10-v1", "stage10")
    check_schema(s10, "n_splits", 25, "stage10")
    check_schema(s10, "n_mice_lr_models_loaded", 125, "stage10")
    check_schema(s10, "advisor_status", "PENDING; PROVISIONAL", "stage10")
    expected_table_counts = [8, 24, 12, 88, 88, 8, 2, 23, 2]
    table_map = s10.get("output_tables")
    if (not isinstance(table_map, dict) or len(table_map) != 9 or
        sorted(table_map.values()) != sorted(expected_table_counts)):
        stop("Stage 10 missing/invalid 9 reporting tables")
    ref_sha = s10.get("source_sha256")
    if not isinstance(ref_sha, dict) or set(ref_sha) != set(SOURCE_FILES):
        stop("Stage 10 source SHA chain is incomplete")
    for name, (stage, relative) in SOURCE_FILES.items():
        require_file(paths[stage] / relative, ref_sha[name])
    if s10_cfg.get("schema") != "sina-stage4-step10-v1":
        stop("Stage 10 run_config schema invalid")
    for key, stage in (("stage3", "stage3"), ("stage5", "stage5"), ("stage6", "stage6"),
                       ("stage7", "stage7"), ("stage8", "stage8"), ("stage9", "stage9")):
        recorded = s10_cfg.get("source_dirs", {}).get(key)
        if not isinstance(recorded, str) or Path(recorded).resolve() != paths[stage]:
            stop("Stage 10 provenance points to unexpected " + key)
    for key, name in (("report", "legacy_report"), ("numeric", "legacy_numeric")):
        recorded = s10_cfg.get("legacy_reference_dirs", {}).get(key)
        if not isinstance(recorded, str) or Path(recorded).resolve() != paths[name]:
            stop("Stage 10 historical source path changed: " + key)

    for stage, script in SCRIPT_NAMES.items():
        cfgname = f"step{stage[5:]}_run_config.json"
        cfg = json_object(paths[stage] / cfgname)
        expected = cfg.get("script_sha256")
        if not isinstance(expected, str) or HASH.fullmatch(expected) is None:
            stop("missing recorded source script SHA: " + stage)
        require_file(ROOT / script, expected)
    log("STEP 11 LINEAGE PASS | original Stage 3 references, corrected Stage 6, Steps 7-10 and source-script SHA256 agree")

    stage3 = paths["stage3"]
    stage5 = paths["stage5"]
    s5summary = json_object(stage5 / "step5_summary.json")
    check_schema(s5summary, "completed_splits", 25, "stage5")
    check_schema(s5summary, "metric_rows", 50, "stage5")
    check_schema(s5summary, "imputer_diagnostic_rows", 125, "stage5")
    check_schema(s5summary, "tuning_rows", 325, "stage5")
    s5config = json_object(stage5 / "run_config.json")
    identity = s5config.get("identity", {})
    if not isinstance(identity, dict):
        stop("Stage 5 run config lacks identity")
    if identity.get("manifest_sha256") != require_file(stage3 / "split_manifest_final.csv"):
        stop("Stage 5 split manifest differs from original Stage 3")
    if identity.get("original_cv_metrics_sha256") != require_file(stage3 / "cv_metrics_final.csv"):
        stop("Stage 5 original AUC/Brier source differs")
    if identity.get("dataset_sha256") != require_file(ROOT / "output/nhanes_examined_clean_with_missing.csv"):
        stop("Original cleaned dataset SHA256 changed")
    for source, expected in identity.get("source_code_sha256", {}).items():
        if source not in ("stage3_cv.py", "stage4_mice_cv.py", "stage4_environment_gate.py", "stage4_mice_cv25.py"):
            stop("unexpected Stage 5 source script name " + source)
        require_file(ROOT / source, expected)
    if set(identity.get("source_code_sha256", {})) != {"stage3_cv.py", "stage4_mice_cv.py", "stage4_environment_gate.py", "stage4_mice_cv25.py"}:
        stop("incomplete Stage 5 source code fingerprints")
    benchmark_dir = benchmark_dir.resolve()
    bench_hashes = identity.get("benchmark_file_sha256")
    if (identity.get("benchmark_dir") != str(benchmark_dir) or
        not isinstance(bench_hashes, dict) or len(bench_hashes) != 3):
        stop("Stage 5 benchmark provenance incomplete or different")
    for name, expected in bench_hashes.items():
        require_file(confined_file(benchmark_dir, name), expected)
    bench = json_object(benchmark_dir / "mice_split1_benchmark_summary.json")
    if bench.get("time_gate") != "PASS" or not 0 < float(bench.get("estimated_25_split_hours_linear", 0)) < 12:
        stop("Stage 5 split-1 compute-time gate no longer passes")

    s6summary = json_object(paths["stage6"] / "step6_summary.json")
    for key, target in (("seed_correction_schema", "sina-stage4-step6-seedfix-v1"),
                        ("n_splits", 25), ("n_models", 2),
                        ("n_per_draw_audit_rows", 250), ("n_pooled_audit_rows", 50),
                        ("n_global_feature_rows", 550),
                        ("rf_draws_corrected_splits_2_to_25", 120),
                        ("lr_draws_reused_unchanged", 125),
                        ("all_local_additivity_pass", True),
                        ("all_pooled_additivity_pass", True)):
        check_schema(s6summary, key, target, "corrected stage6")
    config6 = json_object(paths["stage6"] / "step6_run_config.json")
    origin6 = config6.get("identity", {}).get("original_step6_path")
    origin6_hash = config6.get("identity", {}).get("original_step6_aggregate_manifest_sha256")
    if not isinstance(origin6, str) or not isinstance(origin6_hash, str):
        stop("corrected Stage 6 lacks original Stage 6 chain")
    original6 = Path(origin6).resolve()
    if original6 == paths["stage6"] or not original6.is_dir():
        stop("original Stage 6 source folder missing/overlaps corrected folder")
    original_manifest = json_object(original6 / "step6_aggregate_sha256.json", origin6_hash)
    if len(original_manifest) != 4:
        stop("original Stage 6 aggregate manifest invalid")
    for rel, expected in original_manifest.items():
        require_file(confined_file(original6, rel), expected)
    log("STEP 11 ORIGINAL SHAP PASS | superseded Stage 6 retained for audit; corrected seedfix is canonical")

    counts5 = verify_checkpoints(paths["stage5"], "stage5")
    counts6 = verify_checkpoints(paths["stage6"], "stage6")
    log(f"STEP 11 CHECKPOINT PASS | Stage 5: 25 splits, {counts5[1]} artifacts; corrected Stage 6: 25 splits, {counts6[1]} artifacts; FULL SHA256")
    log("STEP 11 BASELINE PASS | original Stage 3 / Stage 5-10 consistent; no SHAP/models rerun")
    return {"benchmark_hours": float(bench["estimated_25_split_hours_linear"]),
            "original6": original6, "checkpoint_bytes": counts5[2] + counts6[2]}, {"stage10_summary": s10, "stage5_identity": identity}


def available_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "shap", "matplotlib", "joblib", "Pillow")
    result = {"python": platform.python_version(), "python_executable": sys.executable}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def git_result(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *argv], cwd=ROOT, text=True, capture_output=True, check=False, timeout=20)


def inspect_git() -> tuple[int, bool]:
    if shutil.which("git") is None:
        log("STEP 11 GIT HOLD | git not installed; final commit/clean-status gate not met")
        return (-1, False)
    root = git_result(["rev-parse", "--show-toplevel"])
    if root.returncode != 0 or Path(root.stdout.strip()).resolve() != ROOT:
        log("STEP 11 GIT HOLD | project root not the expected Git worktree root")
        return (-1, False)
    br = git_result(["branch", "--show-current"]).stdout.strip()
    commit = git_result(["rev-parse", "HEAD"]).stdout.strip()
    status = git_result(["status", "--porcelain=v1", "--untracked-files=normal"])
    if status.returncode != 0:
        stop("Git status command failed: " + status.stderr.strip()[:500])
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    tracked = git_result(["ls-files", "-z", "--", "output/stage4_mice", "output/stage3_final"])
    if tracked.returncode != 0:
        stop("Git tracked-file check failed")
    names = [x for x in tracked.stdout.split("\0") if x]
    hazardous = [n for n in names if
                 re.search(r"(?:^|/)split_\d{3}/|/values_by_split/|(?:^|/)models/|(?:^|/)test_predictions|\.joblib$|\.npz$|\.pkl$|/split_manifest", n)]
    if hazardous:
        stop("PRIVATE FILES TRACKED IN GIT: count=" + str(len(hazardous)) + "; do not commit participant-level or model artifacts")
    private_examples = (
        "output/stage3_final/shap/values_by_split/shap_values_split_001.csv",
        "output/stage4_mice/local_rechecks/stage5_cv25_20261116T064635Z_2203846/split_001/models/LR_draw01_classifier.joblib",
    )
    unignored = []
    for rel in private_examples:
        result = git_result(["check-ignore", "-q", "--", rel])
        if result.returncode == 1:
            unignored.append(rel)
        elif result.returncode != 0:
            log("STEP 11 GIT IGNORE HOLD | cannot verify ignore rule for a private artifact")
    if unignored:
        log(f"STEP 11 GIT IGNORE HOLD | {len(unignored)} sensitive-file examples are not covered by ignore rules; fix before final commit")
    else:
        log("STEP 11 GIT IGNORE PASS | private Stage 3 SHAP and Stage 5 fitted-model examples ignored")
    clean = not lines
    log(f"STEP 11 GIT CHECK | branch={br or '(detached)'} | HEAD={commit[:12]} | {'CLEAN' if clean else 'DIRTY'} | status_entries={len(lines)} | tracked_private=0")
    if br != "stage3-revision":
        log("STEP 11 GIT HOLD | protocol calls for stage3-revision branch; do not change branches without checking the project history")
    if not clean:
        log("STEP 11 GIT HOLD | final acceptance requires a committed revision and clean Git status (not required for this preflight)")
    return (len(lines), clean and br == "stage3-revision")


def disk_size(root: Path) -> int:
    total = 0
    if root.is_symlink():
        stop("cannot measure symlinked rerun source directory: " + str(root))
    for directory, folders, files in os.walk(root, followlinks=False):
        for folder in folders:
            if (Path(directory) / folder).is_symlink():
                stop("symlinked output directory under " + str(root))
        for name in files:
            item = Path(directory) / name
            if item.is_symlink():
                stop("symlinked output file under " + str(root))
            total += item.stat().st_size
    return total


def recorded_hours(directory: Path, start_split: int = 1) -> float:
    seconds = 0.0
    for split in range(start_split, 26):
        marker = json_object(directory / f"split_{split:03d}" / "split_complete.json")
        elapsed = marker.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            stop(f"missing/invalid completed split runtime: {directory.name}/split_{split:03d}")
        seconds += float(elapsed)
    return seconds / 3600.0


def show_capacity(paths: dict[str, Path], original6: Path, estimate_hours: float) -> None:
    dirs = [paths["stage3"], paths["stage5"], original6, paths["stage6"],
            paths["stage7"], paths["stage8"], paths["stage9"], paths["stage10"]]
    sizes = [disk_size(x) for x in dirs]
    baseline_bytes = sum(sizes)
    usage = shutil.disk_usage(BASE)
    suggested = math.ceil(baseline_bytes * 1.3)
    ratio = usage.free / suggested if suggested else math.inf
    fmt = lambda n: f"{n/1024**3:.2f} GiB"
    log(f"STEP 11 CAPACITY CHECK | baseline_tree={fmt(baseline_bytes)} | free={fmt(usage.free)} | planning_reserve_1.3x={fmt(suggested)} | free/reserve={ratio:.2f}")
    h5 = recorded_hours(paths["stage5"])
    h6_original = recorded_hours(original6)
    h6_seedfix = recorded_hours(paths["stage6"], start_split=2)  # split 1 was copied unchanged
    measured = h5 + h6_original + h6_seedfix
    log(f"STEP 11 RUNTIME EVIDENCE | Stage5={h5:.2f}h; Stage6-original={h6_original:.2f}h; RF-seedfix-new={h6_seedfix:.2f}h; cumulative={measured:.2f}h (excludes Stage 3 and downstream reports)")
    log(f"STEP 11 RUNTIME GATE | original split-1 linear benchmark={estimate_hours:.3f}h (<12h); full replay must be independently planned")
    if measured >= 12:
        log("STEP 11 RUNTIME HOLD | recorded multi-stage workload >=12h; advisor/time-gate decision needed before full replay")
    if usage.free < suggested:
        log("STEP 11 CAPACITY HOLD | free disk space below 1.3x baseline; add storage/adjust plan before heavy replay")
    else:
        log("STEP 11 CAPACITY PASS | baseline-sized separate rerun appears to fit; this is an estimate, not a duration guarantee")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight"], required=True, help="Read-only; never reruns or writes anything")
    p.add_argument("--stage3-dir", default="output/stage3_final")
    p.add_argument("--legacy-report-dir", default="output/stage3_reporting_v1")
    p.add_argument("--legacy-numeric-dir", default="advisor_remaining_csv")
    p.add_argument("--stage5-dir", required=True)
    p.add_argument("--seedfix-dir", required=True)
    p.add_argument("--stage7-dir", required=True)
    p.add_argument("--stage8-dir", required=True)
    p.add_argument("--stage9-dir", required=True)
    p.add_argument("--stage10-dir", required=True)
    p.add_argument("--benchmark-dir", required=True)
    args = p.parse_args()
    if not BASE.is_dir():
        stop("missing Stage 4 base folder or script not placed in project root: " + str(BASE))
    paths = {
        "stage3": resolve_user(args.stage3_dir),
        "legacy_report": resolve_user(args.legacy_report_dir),
        "legacy_numeric": resolve_user(args.legacy_numeric_dir),
        "stage5": resolve_user(args.stage5_dir),
        "stage6": resolve_user(args.seedfix_dir),
        "stage7": resolve_user(args.stage7_dir),
        "stage8": resolve_user(args.stage8_dir),
        "stage9": resolve_user(args.stage9_dir),
        "stage10": resolve_user(args.stage10_dir),
    }
    for name in ("stage5", "stage6", "stage7", "stage8", "stage9", "stage10"):
        if not paths[name].is_relative_to(BASE.resolve()):
            stop(name + " must reside under output/stage4_mice")
    context, _ = audit_sources(paths, resolve_user(args.benchmark_dir))
    versions = available_versions()
    old_config = json_object(paths["stage5"] / "run_config.json")
    identity = old_config["identity"]
    if versions["python"] != str(identity.get("python")) or versions["scikit-learn"] != str(identity.get("sklearn")):
        stop("runtime Python / scikit-learn versions differ from original Stage 5 training environment")
    log("STEP 11 ENV PASS | " + " | ".join(f"{k}={v}" for k, v in versions.items() if k != "python_executable"))
    inspect_git()
    show_capacity(paths, context["original6"], context["benchmark_hours"])
    log("STEP 11 PREFLIGHT PASS | baseline and provenance audited; READ-ONLY; NO rerun, README update, checksum export, commit or files written")
    log("STEP 11 FINAL GATE PENDING | new full run + independent comparison + README/environment + sha256sum -c + commit ID + clean git status")
    log("STEP 11 ADVISOR STATUS | PENDING; PROVISIONAL")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, TypeError, KeyError, FileNotFoundError, OSError) as exc:
        print("STEP 11 PREFLIGHT FAIL | " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
