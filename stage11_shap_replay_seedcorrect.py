#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Step 11: independent *fresh* Stage 6 SHAP replay from Stage 5 replay models.

The source Stage 6 seedfix and its data are REFERENCE ONLY: not one old SHAP
matrix is imported into the fresh run. All 250 model/draw SHAP matrices are
recomputed, LR unchanged and RF with seed = 2026 + actual split ID.

All private per-person matrices remain on the user's computer. Preflight makes
no files. Run defaults to ONE split; a full run must first pass split 1 and the
advisor's estimated-total-runtime gate. Resume never overwrites checkpoints.

This uses the independently audited implementations in
stage4_mice_shap25_seedfix.py for correct RF seed, SHA-verified models,
checkpoint writing, feature aggregation, additivity and benchmark checks.
It intentionally DOES NOT use seedfix's 'copy original LR' entry point.

Usage:
 python -u stage11_shap_replay_seedcorrect.py --mode preflight \
   --original-stage5-dir ... --replay-stage5-dir ... \
   --canonical-stage6-dir ... --benchmark-dir ...
 python -u stage11_shap_replay_seedcorrect.py --mode run [same flags] \
   --out output/stage4_mice/local_rechecks/stage11_replay_shap_TIMESTAMP \
   --max-new-splits 1

Methodology: PROVISIONAL; advisor approval pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import stage4_mice_shap25_seedfix as seedfix

SCHEMA = "sina-stage11-direct-shap-replay-v1"
EXPECTED_CANONICAL_MANIFEST_SHA = "cb3c9077a5471e5caa4e9a51d79bdd583e0daf476da52c8a8d40794f407ca31e"
EXPECTED_ORIGINAL_STAGE5_MANIFEST_SHA = "6e2daaa7cc25dafc139154094438b0b1a94733aec1dc61b715e09958dcd48e5f"
ALL_SPLITS = 25
NUM_TOL = 1e-8
PERFORMANCE_TOL = 1e-10
SHAP_FIELDS = ("values", "base", "output", "probability")


def stop(message: str) -> None:
    raise RuntimeError("STOP: " + message)


def log(message: str) -> None:
    print(message, flush=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(test: bool, message: str) -> None:
    if not test:
        stop(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha_manifest_check(folder: Path, filename: str, required: set[str]) -> str:
    manifest_file = folder / filename
    manifest = read_json(manifest_file)
    require(set(manifest) == required, f"unexpected aggregate SHA entries: {folder}")
    for rel, sha in manifest.items():
        seedfix.checked_artifact(folder, rel, sha)
    return digest(manifest_file)


def source_stage5_check(original: Path, replay: Path) -> dict:
    require(original.is_dir() and replay.is_dir() and original != replay,
            "Stage 5 original and replay paths must be existing different folders")
    config_old = read_json(original / "run_config.json")
    config_new = read_json(replay / "run_config.json")
    require(config_old.get("schema") == config_new.get("schema") ==
            "sina-stage4-step5-v1", "Stage 5 run schema mismatch")
    require(config_old.get("identity") == config_new.get("identity"),
            "Stage 5 environment/dataset/algorithm identity mismatch")
    required = {"mice_cv_metrics_25_splits.csv", "mice_imputer_diagnostics_25_splits.csv",
                "mice_inner_tuning_25_splits.csv", "step5_summary.json"}
    original_manifest_sha = sha_manifest_check(original, "step5_aggregate_sha256.json", required)
    replay_manifest_sha = sha_manifest_check(replay, "step5_aggregate_sha256.json", required)
    require(original_manifest_sha == EXPECTED_ORIGINAL_STAGE5_MANIFEST_SHA,
            "original Stage 5 manifest differs from pinned reference")
    require(read_json(original / "step5_summary.json") == read_json(replay / "step5_summary.json"),
            "original/replay Stage 5 completion summaries differ")
    count = 0
    for split_id in range(1, 26):
        folder_old = original / f"split_{split_id:03d}"
        folder_new = replay / f"split_{split_id:03d}"
        old = read_json(folder_old / "split_complete.json")
        new = read_json(folder_new / "split_complete.json")
        expected_paths = {
            *{f"models/MICE_draw{i:02d}_imputer_bundle.joblib" for i in range(1, 6)},
            *{f"models/{model}_draw{i:02d}_classifier.joblib"
              for model in ("LR", "RF") for i in range(1, 6)},
            "mice_performance.csv", "mice_imputer_diagnostics.csv", "mice_inner_tuning.csv",
        }
        for marker in (old, new):
            require(marker.get("schema") == "sina-stage4-step5-v1"
                    and marker.get("split_id") == split_id
                    and set(marker.get("sha256", {})) == expected_paths,
                    f"Stage 5 checkpoint invalid in split {split_id}")
        for name in sorted(expected_paths):
            seedfix.checked_artifact(folder_old, name, old["sha256"][name])
            seedfix.checked_artifact(folder_new, name, new["sha256"][name])
            if name.startswith("models/"):
                require(old["sha256"][name] == new["sha256"][name],
                        f"Stage 5 model bytes differ: split {split_id}, {name}")
                count += 1
        if split_id % 5 == 0:
            log(f"STEP 11 MODEL REPLAY CHECK | {split_id}/25 checkpoint pairs; all referenced SHA256 verified")
    require(count == 375, "expected 375 independently reproduced Stage 5 model files")
    log("STEP 11 REPLAY MODELS PASS | 375/375 byte-identical models; 900 checkpoint artifacts verified")
    return {"original": str(original), "replay": str(replay),
            "original_manifest_sha256": original_manifest_sha,
            "replay_manifest_sha256": replay_manifest_sha,
            "original_config_sha256": digest(original / "run_config.json"),
            "replay_config_sha256": digest(replay / "run_config.json")}


def canonical_stage6_check(folder: Path) -> dict:
    required = {"mice_shap_draw_additivity_25_splits.csv",
                "mice_shap_pooled_additivity_25_splits.csv",
                "mice_shap_global_importance_25_splits.csv", "step6_summary.json"}
    manifest_sha = sha_manifest_check(folder, "step6_aggregate_sha256.json", required)
    require(manifest_sha == EXPECTED_CANONICAL_MANIFEST_SHA,
            "canonical corrected Stage 6 SHA manifest not pinned reference")
    summary = read_json(folder / "step6_summary.json")
    require(summary.get("seed_correction_schema") == seedfix.SEEDFIX_SCHEMA
            and summary.get("n_splits") == 25
            and summary.get("n_per_draw_audit_rows") == 250
            and summary.get("n_global_feature_rows") == 550
            and summary.get("rf_permutation_seed_rule") == "2026 + split_id",
            "canonical corrected Stage 6 summary invalid")
    for split_id in range(1, 26):
        seedfix.finished_split(folder / f"split_{split_id:03d}", split_id)
    log("STEP 11 CANONICAL SHAP PASS | corrected Stage 6 SHA256; 25/25 checkpoint markers; correct RF seeds")
    return {"path": str(folder), "manifest_sha256": manifest_sha,
            "config_sha256": digest(folder / "step6_run_config.json")}


def compare_vector(before: np.ndarray, after: np.ndarray, name: str, tol: float) -> float:
    require(before.shape == after.shape, f"{name}: shape mismatch")
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    require(np.isfinite(a).all() and np.isfinite(b).all(), f"{name}: nonfinite values")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(delta < tol, f"{name}: max abs difference {delta:.6g} >= {tol:.6g}")
    return delta


def compare_csv(original: Path, replay: Path, key: list[str], numerical: list[str],
                tolerance: float) -> float:
    ref = pd.read_csv(original)
    got = pd.read_csv(replay)
    require(len(ref) == len(got), f"CSV row count mismatch: {original.name}")
    require(set(key + numerical).issubset(ref.columns) and
            set(key + numerical).issubset(got.columns),
            f"missing CSV comparison columns in {original.name}")
    require(not ref.duplicated(key).any() and not got.duplicated(key).any(),
            f"duplicate CSV keys in {original.name}")
    a = ref[key + numerical].sort_values(key).reset_index(drop=True)
    b = got[key + numerical].sort_values(key).reset_index(drop=True)
    for k in key:
        require(a[k].astype(str).tolist() == b[k].astype(str).tolist(),
                f"CSV keys differ: {original.name} / {k}")
    max_delta = 0.0
    for n in numerical:
        max_delta = max(max_delta, compare_vector(a[n].to_numpy(), b[n].to_numpy(),
                                                 f"{original.name} / {n}", tolerance))
    return max_delta


def compare_split(reference: Path, replay: Path, split_id: int) -> dict:
    original = reference / f"split_{split_id:03d}"
    new = replay / f"split_{split_id:03d}"
    seedfix.finished_split(original, split_id)
    seedfix.finished_split(new, split_id)
    max_draw = 0.0
    for model in ("LR", "RF"):
        for draw in range(1, 6):
            old_npz, _ = seedfix.draw_paths(original, model, draw)
            new_npz, _ = seedfix.draw_paths(new, model, draw)
            with np.load(old_npz, allow_pickle=False) as a, np.load(new_npz, allow_pickle=False) as b:
                require(set(a.files) == set(b.files) == set(SHAP_FIELDS) | {"seqn"},
                        f"wrong SHAP .npz fields split={split_id} {model} draw={draw}")
                require(np.array_equal(a["seqn"], b["seqn"]),
                        f"test ID mismatch split={split_id} {model} draw={draw}")
                for name in SHAP_FIELDS:
                    max_draw = max(max_draw, compare_vector(a[name], b[name],
                              f"SHAP split {split_id}, {model}, draw {draw}, {name}", NUM_TOL))
            _, draw_marker = seedfix.draw_paths(new, model, draw)
            data = read_json(draw_marker)
            require(data.get("classifier_sha256") is not None,
                    "replay SHAP draw has no fitted classifier provenance")
            if model == "RF":
                require(data.get("permutation_seed") == 2026 + split_id,
                        f"invalid replay RF seed in split={split_id} draw={draw}")
    max_private = compare_csv(
        original / "shap_values_private.csv", new / "shap_values_private.csv",
        ["SEQN", "y_true", "split_id", "repeat", "fold"],
        [f"{model}__mice_m5__{feature}" for model in ("LR", "RF")
         for feature in list(seedfix.s3.PREDICTORS)] +
        [f"{model}__mice_m5__base_value" for model in ("LR", "RF")], NUM_TOL)
    max_importance = compare_csv(
        original / "mice_shap_global_importance.csv",
        new / "mice_shap_global_importance.csv",
        ["split_id", "model", "method", "feature", "rank"],
        ["mean_abs_shap", "mean_shap"], NUM_TOL)
    max_performance = compare_csv(
        original / "mice_shap_pooled_additivity.csv",
        new / "mice_shap_pooled_additivity.csv",
        ["split_id", "model", "n_imputations", "n_test_common", "background_n"],
        ["auc", "brier_score"], PERFORMANCE_TOL)
    return {"split_id": split_id, "max_abs_draw_shap_diff": max_draw,
            "max_abs_private_shap_diff": max_private,
            "max_abs_global_importance_diff": max_importance,
            "max_abs_pooled_performance_diff": max_performance,
            "status": "PASS"}


def check_out_path(root: Path, out_arg: Path, protected: list[Path]) -> Path:
    out = seedfix.within(root, out_arg)
    require(out.name.startswith("stage11_replay_shap_"),
            "SHAP replay output must use a stage11_replay_shap_ prefix")
    for p in protected:
        p = p.resolve()
        require(out != p and not out.is_relative_to(p) and not p.is_relative_to(out),
                f"replay SHAP destination overlaps protected directory: {p}")
    return out


def run_config_identity(root: Path, ctx: dict, provenance: dict, canon: dict) -> dict:
    code_hashes = {
        name: digest(root / name) for name in (
            Path(__file__).name,
            "stage4_mice_shap25_seedfix.py", "stage4_mice_shap25.py",
            "stage4_mice_cv25.py", "stage4_mice_benchmark_split1.py",
            "stage3_shap.py", "stage3_cv.py",
        )
    }
    return {"schema": SCHEMA, "stage5_replay": provenance, "canonical_stage6": canon,
            "seedfix_preflight_identity": ctx["identity"],
            "code_sha256": code_hashes,
            "python": platform.python_version(),
            "advisor_status": "PENDING; PROVISIONAL"}


def write_final(out: Path, per_split: list[dict], identity: dict) -> None:
    audits, pooled, globals_ = [], [], []
    for sid in range(1, ALL_SPLITS + 1):
        a, p, g, _ = seedfix.finished_split(out / f"split_{sid:03d}", sid)
        audits.extend(a)
        pooled.extend(p)
        globals_.extend(g)
    require((len(audits), len(pooled), len(globals_)) == (250, 50, 550),
            "incorrect aggregate row counts")
    for fname, data in (
        ("mice_shap_draw_additivity_25_splits.csv", audits),
        ("mice_shap_pooled_additivity_25_splits.csv", pooled),
        ("mice_shap_global_importance_25_splits.csv", globals_),
    ):
        seedfix.write_or_check_csv(out / fname, data)
    report = {
        "schema": seedfix.SCHEMA,
        "direct_replay_schema": SCHEMA,
        "advisor_status": "PENDING; PROVISIONAL",
        "n_splits": 25, "n_models": 2, "n_draws_per_model": 5,
        "n_per_draw_audit_rows": 250, "n_pooled_audit_rows": 50,
        "n_global_feature_rows": 550,
        "rf_permutation_seed_rule": "2026 + split_id",
        "lr_draws_recomputed": 125, "rf_draws_recomputed": 125,
        "all_250_shap_draws_fresh": True,
        "all_local_additivity_pass": all(x["status"] == "PASS" and
                                      x["max_abs_additivity_error"] < seedfix.TOL
                                      for x in audits),
        "all_pooled_additivity_pass": all(x["status"] == "PASS" and
                                       x["pooled_max_abs_additivity_error"] < seedfix.TOL
                                       for x in pooled),
        "private_participant_shap_not_for_distribution": True,
        "reference_stage6_read_only": True,
    }
    seedfix.write_or_check_json(out / "step6_summary.json", report)
    sha = {name: digest(out / name) for name in (
        "mice_shap_draw_additivity_25_splits.csv",
        "mice_shap_pooled_additivity_25_splits.csv",
        "mice_shap_global_importance_25_splits.csv", "step6_summary.json")}
    seedfix.write_or_check_json(out / "step6_aggregate_sha256.json", sha)
    seedfix.write_or_check_csv(out / "stage11_replay_shap_comparison_25_splits.csv", per_split)
    meta = {"schema": SCHEMA, "advisor_status": "PENDING; PROVISIONAL",
            "identity": identity, "n_splits": len(per_split), "n_draws_new": 250,
            "max_abs_shap_draw_diff": max(r["max_abs_draw_shap_diff"] for r in per_split),
            "max_abs_global_importance_diff": max(r["max_abs_global_importance_diff"] for r in per_split),
            "max_abs_pooled_performance_diff": max(r["max_abs_pooled_performance_diff"] for r in per_split),
            "reference_stage6_never_used_for_generation": True,
            "all_comparisons_pass": all(r["status"] == "PASS" for r in per_split)}
    seedfix.write_or_check_json(out / "stage11_replay_shap_summary.json", meta)
    sha_replay = {name: digest(out / name) for name in (
        "stage11_replay_shap_comparison_25_splits.csv",
        "stage11_replay_shap_summary.json", "step6_aggregate_sha256.json",
        "step6_run_config.json")}
    seedfix.write_or_check_json(out / "stage11_replay_shap_sha256.json", sha_replay)
    log("STEP 11 SHAP REPLAY PASS | 250/250 fresh SHAP draws; 25/25 canonical comparisons; SHA256 PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    parser.add_argument("--original-stage5-dir", required=True, type=Path)
    parser.add_argument("--replay-stage5-dir", required=True, type=Path)
    parser.add_argument("--canonical-stage6-dir", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-splits", type=int, default=1)
    args = parser.parse_args()
    require(1 <= args.max_new_splits <= 25, "--max-new-splits must be 1..25")
    if args.mode == "preflight":
        require(args.out is None and not args.resume,
                "preflight is read-only: do not pass --out or --resume")
    else:
        require(args.out is not None, "run requires a NEW --out directory")
    root = Path(__file__).resolve().parent
    original = seedfix.within(root, args.original_stage5_dir)
    replay = seedfix.within(root, args.replay_stage5_dir)
    canonical = seedfix.within(root, args.canonical_stage6_dir)
    benchmark = seedfix.within(root, args.benchmark_dir)
    require(len({str(original), str(replay), str(canonical), str(benchmark)}) == 4,
            "all four source paths must be different")
    provenance = source_stage5_check(original, replay)
    canon = canonical_stage6_check(canonical)
    ctx = seedfix.preflight(root, args.replay_stage5_dir, args.benchmark_dir)
    identity = run_config_identity(root, ctx, provenance, canon)
    log("STEP 11 SHAP DIRECT PREFLIGHT PASS | replay classifiers validated; "
        "correct RF seed=2026+split_id; 250 fresh SHAP draws ready")
    if args.mode == "preflight":
        log("STEP 11 SHAP PREFLIGHT PASS | READ-ONLY; no outputs, no model training or SHAP computation")
        return
    out = check_out_path(root, args.out, [original, replay, canonical, benchmark])
    cfgpath = out / "step6_run_config.json"
    if args.resume:
        require(out.is_dir() and cfgpath.is_file(),
                "no replay run_config at resume path")
        require(read_json(cfgpath).get("identity") == identity,
                "replay SHAP identity/algorithm/code/environment changed; refuse resume")
        log(f"STEP 11 SHAP RESUME | {out}")
    else:
        require(not out.exists(), "output directory already exists")
        out.mkdir(parents=True, exist_ok=False)
        seedfix.write_new_json(cfgpath, {
            "schema": SCHEMA, "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "advisor_status": "PENDING; PROVISIONAL"})
        log(f"STEP 11 SHAP NEW RUN | {out}")
    completed = []
    results = []
    incomplete_seen = False
    for split_id in range(1, ALL_SPLITS + 1):
        folder = out / f"split_{split_id:03d}"
        if (folder / "split_complete.json").exists():
            require(not incomplete_seen, "a complete split appears after incomplete split")
            seedfix.finished_split(folder, split_id)
            result = compare_split(canonical, out, split_id)
            completed.append(split_id)
            results.append(result)
        elif folder.exists():
            require(not incomplete_seen, "multiple incomplete splits exist")
            incomplete_seen = True
        elif incomplete_seen:
            continue
    if completed:
        require(completed == list(range(1, len(completed) + 1)),
                "completed splits are not a consecutive prefix")
    if incomplete_seen:
        next_id = len(completed) + 1
        require((out / f"split_{next_id:03d}").is_dir(),
                "an incomplete split is not next in expected order")
    if (out / "step6_summary.json").is_file():
        require(len(completed) == 25, "final summary exists but checkpoints are incomplete")
        if not (out / "stage11_replay_shap_sha256.json").exists():
            write_final(out, results, identity)
        else:
            manifest = read_json(out / "stage11_replay_shap_sha256.json")
            for name, sha in manifest.items():
                seedfix.checked_artifact(out, name, sha)
            log("STEP 11 SHAP REPLAY COMPLETE | existing 25/25 outputs verified")
        return
    if len(completed) >= 1 and args.max_new_splits > 1:
        anchor = read_json(out / "split_001" / "split_complete.json")
        hours = float(anchor["elapsed_seconds"]) * 25 / 3600
        log(f"STEP 11 SHAP TIME GATE | split1 linear projection={hours:.3f}h for 25 splits")
        require(math.isfinite(hours) and hours < 12.0,
                "projected replay SHAP exceeds advisor's 12h gate; get advisor decision")
    elif args.max_new_splits > 1:
        stop("pilot split 1 must pass first; run --max-new-splits 1, then resume")
    remaining = args.max_new_splits
    for split_id in range(1, ALL_SPLITS + 1):
        if split_id in completed:
            continue
        if remaining == 0:
            break
        if split_id != 1:
            require(1 in completed, "split 1 benchmark and canonical comparisons must pass first")
        log(f"STEP 11 SHAP REPLAY START | split={split_id}/25 | 10 fresh draws | RF seed={2026+split_id}")
        seedfix.run_one_split(ctx, out, split_id)
        seedfix.finished_split(out / f"split_{split_id:03d}", split_id)
        result = compare_split(canonical, out, split_id)
        results.append(result)
        completed.append(split_id)
        log(f"STEP 11 SHAP SPLIT PASS | {split_id}/25 | "
            f"max_draw_diff={result['max_abs_draw_shap_diff']:.3g} "
            f"max_global_diff={result['max_abs_global_importance_diff']:.3g}")
        remaining -= 1
    if len(completed) == ALL_SPLITS:
        write_final(out, results, identity)
    else:
        log(f"STEP 11 SHAP PARTIAL | completed={len(completed)}/25; "
            "preserve --out and resume after reviewing log")


if __name__ == "__main__":
    main()
