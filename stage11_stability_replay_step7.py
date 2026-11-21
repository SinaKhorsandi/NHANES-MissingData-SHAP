#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 11: independently replay Stage 7 stability from FRESH Stage 6 SHAP.

The 300 paired SHAP-stability records are recomputed from Stage 3 private SHAP
and the 250 freshly re-explained MICE draws of the Stage 11 direct SHAP replay.
No participant-level information is written. Canonical Stage 6 and original
Stage 7 are validated and used ONLY as read-only scientific references.

Run from the verified research root, alongside stage4_mice_stability4.py and
stage4_mice_shap25_seedfix.py. First --mode preflight (read-only); then run
--mode run with a NEW --out under output/stage4_mice/local_rechecks/.

Methodology remains PROVISIONAL; advisor approval pending.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import stage4_mice_stability4 as s7
import stage4_mice_shap25_seedfix as s6

SCHEMA = "sina-stage11-step7-stability-replay-v1"
ADVISOR = "PENDING; PROVISIONAL"
DIRECT_SHAP_SCHEMA = "sina-stage11-direct-shap-replay-v1"
CANONICAL_MANIFEST_SHA = "cb3c9077a5471e5caa4e9a51d79bdd583e0daf476da52c8a8d40794f407ca31e"
STAGE7_MANIFEST_SHA = "73061e16170d46f0ea2e5b2f74dde85db34fa19c2a778d7f8dc85a3e21cf8379"
STAGE7_CSV_SHA = "894df3061128440afac6f8a9659f45135f5a5a794bb78e58195c8f3672b4fa2f"
STAGE7_CODE_SHA = "3d99c308578d5eab6c5981f4d38d690f8d12d77221a2a44b73ab8bbe7d419d06"
DIRECT_SHAP_CODE_SHA = "5ada59053b768c4490580564a6bd2fcf74fa2180ef65ea4fa1254c68a0700a35"
TOLERANCE = 1e-12
ZERO_TOL = 1e-8


def stop(reason: str) -> None:
    raise RuntimeError("STOP: " + reason)


def need(test: bool, reason: str) -> None:
    if not test:
        stop(reason)


def announce(s: str) -> None:
    print(s, flush=True)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    s7.require_file(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(obj, dict), f"expected JSON object in {path}")
    return obj


def verified_manifest(folder: Path, name: str, expected_names: set[str]) -> str:
    manifest_path = folder / name
    manifest = load_json(manifest_path)
    need(set(manifest) == expected_names, f"wrong manifest entries: {manifest_path}")
    for rel, sha in manifest.items():
        need(isinstance(sha, str) and len(sha) == 64, f"invalid SHA256: {rel}")
        s7.verified_file(folder, rel, sha)
    return hash_file(manifest_path)


def read_csv_strict(path: Path, columns: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    s7.require_file(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        if columns is not None:
            need(tuple(rdr.fieldnames or []) == columns, f"unexpected columns in {path}")
        rows = list(rdr)
    need(all(None not in row for row in rows), f"malformed CSV row in {path}")
    return rows


def reference_stage7_check(ref: Path) -> dict[str, Any]:
    expected_names = {"shap_stability_metrics_4_methods_25_splits.csv",
                      "step7_run_config.json", "step7_summary.json"}
    manifest_sha = verified_manifest(ref, "step7_aggregate_sha256.json", expected_names)
    need(manifest_sha == STAGE7_MANIFEST_SHA,
         "original Stage 7 aggregate manifest differs from independently audited reference")
    ref_csv = ref / "shap_stability_metrics_4_methods_25_splits.csv"
    need(hash_file(ref_csv) == STAGE7_CSV_SHA,
         "original 300-row Stage 7 CSV differs from the reviewed reference")
    cfg = load_json(ref / "step7_run_config.json")
    summary = load_json(ref / "step7_summary.json")
    need(summary.get("schema") == s7.SCHEMA and summary.get("n_rows") == 300
         and summary.get("n_legacy_rows_exactly_preserved") == 150
         and summary.get("n_new_mice_rows") == 150
         and summary.get("advisor_status") == ADVISOR,
         "original Stage 7 reference summary metadata invalid")
    need(cfg.get("script_sha256") == STAGE7_CODE_SHA
         and cfg.get("stage3_reference_150_sha256") == s7.ORIGINAL_150_SHA256
         and cfg.get("corrected_step6_aggregate_manifest_sha256") == CANONICAL_MANIFEST_SHA,
         "original Stage 7 reference provenance invalid")
    rows = read_csv_strict(ref_csv, s7.COLS)
    expected = {(sid, model, a, b) for sid in range(1, 26)
                for model in s7.MODELS for a, b in s7.ALL_PAIRS}
    need(len(rows) == 300 and set(map(s7.pairkey, rows)) == expected,
         "original Stage 7 results are not a complete 300-row unique set")
    return {"rows": rows, "manifest_sha256": manifest_sha,
            "csv_sha256": hash_file(ref_csv), "summary_sha256": hash_file(ref / "step7_summary.json")}


def direct_replay_check(replay: Path, canonical: Path) -> tuple[dict, dict, dict]:
    """Validate two independent provenance chains and comparison of public aggregates.

    Participant-level checkpoint hashes are verified separately, during
    recomputation, using the ORIGINAL unchanged Stage 7 split validator.
    """
    canon = s7.audit_seedfix(canonical)
    need(canon["aggregate_hash"] == CANONICAL_MANIFEST_SHA,
         "Stage 6 seed-corrected reference is not the pinned canonical manifest")
    require6 = {"mice_shap_draw_additivity_25_splits.csv",
                "mice_shap_pooled_additivity_25_splits.csv",
                "mice_shap_global_importance_25_splits.csv", "step6_summary.json"}
    replay_manifest_sha = verified_manifest(replay, "step6_aggregate_sha256.json", require6)
    require11 = {"step6_aggregate_sha256.json", "step6_run_config.json",
                 "stage11_replay_shap_comparison_25_splits.csv",
                 "stage11_replay_shap_summary.json"}
    replay_sha = verified_manifest(replay, "stage11_replay_shap_sha256.json", require11)
    summary = load_json(replay / "step6_summary.json")
    replay_report = load_json(replay / "stage11_replay_shap_summary.json")
    cfg = load_json(replay / "step6_run_config.json")
    need(summary.get("schema") == s6.SCHEMA
         and summary.get("direct_replay_schema") == DIRECT_SHAP_SCHEMA
         and summary.get("n_splits") == 25
         and summary.get("n_per_draw_audit_rows") == 250
         and summary.get("n_pooled_audit_rows") == 50
         and summary.get("n_global_feature_rows") == 550
         and summary.get("lr_draws_recomputed") == 125
         and summary.get("rf_draws_recomputed") == 125
         and summary.get("all_250_shap_draws_fresh") is True
         and summary.get("all_local_additivity_pass") is True
         and summary.get("all_pooled_additivity_pass") is True
         and summary.get("rf_permutation_seed_rule") == "2026 + split_id"
         and summary.get("reference_stage6_read_only") is True
         and summary.get("private_participant_shap_not_for_distribution") is True
         and summary.get("advisor_status") == ADVISOR,
         "direct SHAP replay did not record complete, fresh, seed-corrected 250 draws")
    need(replay_report.get("schema") == DIRECT_SHAP_SCHEMA
         and replay_report.get("all_comparisons_pass") is True
         and replay_report.get("n_draws_new") == 250
         and replay_report.get("n_splits") == 25
         and replay_report.get("reference_stage6_never_used_for_generation") is True
         and replay_report.get("advisor_status") == ADVISOR,
         "Stage 11 SHAP replay comparisons not complete")
    identity = replay_report.get("identity", {})
    code_record = identity.get("code_sha256", {})
    need(code_record.get("stage11_shap_replay_seedcorrect.py") == DIRECT_SHAP_CODE_SHA
         and hash_file(Path(__file__).resolve().parent / "stage11_shap_replay_seedcorrect.py") == DIRECT_SHAP_CODE_SHA,
         "direct SHAP replay code differs from audited version")
    need(identity.get("canonical_stage6", {}).get("manifest_sha256") == CANONICAL_MANIFEST_SHA
         and cfg.get("identity") == identity and cfg.get("schema") == DIRECT_SHAP_SCHEMA,
         "SHAP replay identity does not resolve to canonical Stage 6")
    for col in ("max_abs_shap_draw_diff", "max_abs_global_importance_diff",
                "max_abs_pooled_performance_diff"):
        value = float(replay_report.get(col, float("nan")))
        need(math.isfinite(value) and 0 <= value < ZERO_TOL,
             f"nonzero or invalid SHAP replay global comparison: {col}")
    split_compare = pd.read_csv(replay / "stage11_replay_shap_comparison_25_splits.csv")
    checks = ("max_abs_draw_shap_diff", "max_abs_private_shap_diff",
              "max_abs_global_importance_diff", "max_abs_pooled_performance_diff")
    s7.require_columns(split_compare, {"split_id", "status", *checks},
                       "Stage 11 direct SHAP comparison")
    need(len(split_compare) == 25 and set(split_compare["split_id"]) == set(range(1, 26))
         and split_compare["split_id"].is_unique
         and split_compare["status"].eq("PASS").all(),
         "Stage 11 replay comparison does not cover 25 passing splits")
    for col in checks:
        values = pd.to_numeric(split_compare[col], errors="coerce").to_numpy(dtype=float)
        need(np.isfinite(values).all() and (np.abs(values) < ZERO_TOL).all(),
             f"Stage 11 split-level SHAP comparison violation: {col}")
    # The replayed global and pooled CSVs are expected to be byte-identical.
    for name in ("mice_shap_global_importance_25_splits.csv",
                 "mice_shap_pooled_additivity_25_splits.csv"):
        need(hash_file(replay / name) == hash_file(canonical / name),
             f"replayed {name} differs byte-for-byte from corrected canonical reference")
    # Draw-audit durations are allowed to differ; every scientific cell must agree.
    drawname = "mice_shap_draw_additivity_25_splits.csv"
    old_rows = read_csv_strict(canonical / drawname)
    new_rows = read_csv_strict(replay / drawname)
    def drawkey(r):
        return (int(r["split_id"]), r["model"], int(r["imputation_id"]))
    old_map, new_map = {drawkey(r): r for r in old_rows}, {drawkey(r): r for r in new_rows}
    expected_draws = {(sid, m, d) for sid in range(1, 26)
                      for m in s7.MODELS for d in range(1, 6)}
    need(len(old_rows) == len(new_rows) == len(old_map) == len(new_map) == 250
         and set(old_map) == set(new_map) == expected_draws,
         "draw-level audit does not contain exactly 250 unique keys")
    for key in expected_draws:
        a, b = old_map[key], new_map[key]
        need(set(a) == set(b), f"draw audit columns changed in {key}")
        for col in a:
            if col == "shap_seconds":
                sec = float(b[col])
                need(math.isfinite(sec) and sec >= 0, f"invalid replay duration {key}")
            else:
                need(a[col] == b[col], f"replay per-draw SHAP scientific value mismatch {key}/{col}")
    audit = {
        "draw": pd.read_csv(replay / drawname),
        "pooled": pd.read_csv(replay / "mice_shap_pooled_additivity_25_splits.csv"),
        "global": pd.read_csv(replay / "mice_shap_global_importance_25_splits.csv"),
    }
    need((len(audit["draw"]), len(audit["pooled"]), len(audit["global"])) == (250, 50, 550),
         "direct replay SHAP aggregate rows are incomplete")
    announce("STEP 11 STEP7 SOURCE PASS | original Stage 7 pinned, corrected Stage 6 pinned; 250 fresh SHAP audits scientifically identical")
    return audit, {"replay_manifest_sha256": replay_manifest_sha,
                   "replay_audit_manifest_sha256": replay_sha,
                   "replay_stage6_summary_sha256": hash_file(replay / "step6_summary.json"),
                   "replay_run_config_sha256": hash_file(replay / "step6_run_config.json")}, canon


def compare_reference(combined: list[dict], expected: list[dict[str, str]]) -> tuple[float, int]:
    """Compare all 300 pairs, including labels/top-5, within explicit tolerance."""
    got = {s7.pairkey(x): x for x in combined}
    ref = {s7.pairkey(x): x for x in expected}
    need(len(got) == len(ref) == len(combined) == len(expected) == 300
         and set(got) == set(ref), "Stage 7 replay/reference 300 keys differ")
    max_diff = 0.0
    nonzero_cells = 0
    for key, source in ref.items():
        row = got[key]
        for col in s7.COLS:
            if col in s7.FLOAT_COLS:
                before, after = float(source[col]), float(row[col])
                delta = abs(before - after)
                need(math.isfinite(delta) and delta <= TOLERANCE,
                     f"Stage 7 replay/reference metric mismatch {key}/{col}: {delta}")
                max_diff = max(max_diff, delta)
                nonzero_cells += delta > 0
            elif col in s7.INT_COLS:
                need(int(source[col]) == int(row[col]),
                     f"Stage 7 replay/reference integer mismatch {key}/{col}")
            else:
                need(str(source[col]) == str(row[col]),
                     f"Stage 7 replay/reference categorical mismatch {key}/{col}")
    return max_diff, nonzero_cells


def recompute_everything(stage3: Path, replay: Path, canonical: Path,
                         stage7_ref: Path) -> tuple[list[dict], dict]:
    root = Path(__file__).resolve().parent
    original_stage7_code = root / "stage4_mice_stability4.py"
    stage3_code = root / "stage3_stability.py"
    need(hash_file(original_stage7_code) == STAGE7_CODE_SHA
         and hash_file(stage3_code) == s7.STAGE3_STABILITY_SOURCE_SHA256,
         "Stage 3 / Stage 7 stability formula source SHA differs from audited version")
    original_ref = s7.read_original_ref(stage3 / "shap_stability_metrics_by_split_final.csv")
    old_audit = s7.audit_stage3(stage3)
    canonical_audit = s7.audit_seedfix(canonical)
    stage7_reference = reference_stage7_check(stage7_ref)
    replay_audit, fingerprints, _ = direct_replay_check(replay, canonical)
    s7.require_file(stage3 / "shap_stability_metrics_by_split_final.csv")
    new_rows: list[dict] = []
    old_calculated: list[dict] = []
    for sid in range(1, 26):
        # This validates all 24 artifacts per split (and RF seeds) in BOTH runs.
        s7.check_seedfix_split(canonical, sid, canonical_audit)
        private_replay = s7.check_seedfix_split(replay, sid, replay_audit)
        for d in range(1, 6):
            # New direct replay must explicitly record seed even for split 1.
            marker = load_json(replay / f"split_{sid:03d}" / "draws" / f"RF_draw{d:02d}.json")
            need(marker.get("permutation_seed") == 2026 + sid,
                 f"direct replay RF seed missing/incorrect: split={sid} draw={d}")
        old, mice = s7.check_alignment(
            stage3 / "values_by_split" / f"shap_values_split_{sid:03d}.csv",
            private_replay, sid,
            replay_audit["global"].loc[replay_audit["global"].split_id.eq(sid)],
            old_audit, replay_audit)
        repeat, fold = (sid - 1) // 5 + 1, (sid - 1) % 5 + 1
        for model in s7.MODELS:
            vals = {method: s7.matrix(old, model, method) for method in s7.METHODS_OLD}
            vals[s7.MICE] = s7.matrix(mice, model, s7.MICE)
            for a, b in s7.PAIRS_OLD:
                old_calculated.append(s7.comparison(sid, repeat, fold, model, a, b,
                                                     vals[a], vals[b]))
            for a, b in s7.PAIRS_NEW:
                new_rows.append(s7.comparison(sid, repeat, fold, model, a, b,
                                              vals[a], vals[b]))
        if sid % 5 == 0:
            announce(f"STEP 11 STEP7 CALC PROGRESS | {sid}/25 aligned cohorts and private SHAP matrices verified")
    legacy_diff = s7.verify_legacy(old_calculated, original_ref)
    need(len(new_rows) == 150 and len({s7.pairkey(r) for r in new_rows}) == 150,
         "MICE pairwise stability recomputation incomplete")
    # Preserve 150 original Stage 3 cells as text, as the audited Stage 7 did.
    combined: list[dict] = [*original_ref, *new_rows]
    combined.sort(key=s7.pairkey)
    max_ref_diff, n_nonzero = compare_reference(combined, stage7_reference["rows"])
    status = {
        "legacy_150_recalculation_max_abs_diff": legacy_diff,
        "reference_300_recalculation_max_abs_diff": max_ref_diff,
        "reference_float_cells_nonzero_difference": int(n_nonzero),
        "replay_fingerprints": fingerprints,
        "original_stage7_fingerprints": {k: v for k, v in stage7_reference.items() if k != "rows"},
        "stage3_stability_150_sha256": hash_file(stage3 / "shap_stability_metrics_by_split_final.csv"),
        "stage3_additivity_audit_sha256": hash_file(stage3 / "additivity_audit_final.csv"),
        "corrected_stage6_manifest_sha256": CANONICAL_MANIFEST_SHA,
    }
    announce(f"STEP 11 STEP7 HISTORICAL PASS | 150/150 historical rows recomputed; max_abs_diff={legacy_diff:.3g}")
    announce(f"STEP 11 STEP7 REPLAY MATCH PASS | 300/300 rows recomputed using fresh SHAP; max_abs_diff={max_ref_diff:.3g}")
    return combined, status


def write_json(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def run_output(out: Path, srcs: dict[str, Path], rows: list[dict], report: dict) -> None:
    root = Path(__file__).resolve().parent
    allowed_root = (root / "output" / "stage4_mice" / "local_rechecks").resolve()
    need(out.name.startswith("stage11_replay_step7_"),
         "output name must start with stage11_replay_step7_")
    need(s7.inside(allowed_root, out) and out.parent == allowed_root,
         "output must be a new immediate child of output/stage4_mice/local_rechecks")
    need(allowed_root.is_dir(), "output parent does not exist")
    need(not out.exists() and not out.is_symlink(), "output already exists; refuse overwrite")
    for src in srcs.values():
        need(out != src and not out.is_relative_to(src) and not src.is_relative_to(out),
             "output overlaps a protected reference directory")
    out.mkdir(exist_ok=False, parents=False)
    outcsv = out / "shap_stability_metrics_4_methods_25_splits.csv"
    with outcsv.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=s7.COLS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    readback = read_csv_strict(outcsv, s7.COLS)
    need(len(readback) == 300 and len({s7.pairkey(r) for r in readback}) == 300,
         "saved Stage 7 replay CSV failed readback")
    for key in ("split_id", "model", "method_a", "method_b"):
        need(all(key in row for row in readback), f"missing field {key} in output")
    stage7_result_sha = hash_file(outcsv)
    config = {
        "schema": SCHEMA, "advisor_status": ADVISOR,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hash_file(Path(__file__).resolve()),
        "stage7_formula_script_sha256": STAGE7_CODE_SHA,
        "stage3_stability_formula_script_sha256": s7.STAGE3_STABILITY_SOURCE_SHA256,
        "inputs": {k: str(v) for k, v in srcs.items()},
        "source_fingerprints": report,
        "private_participant_data_not_written": True,
        "canonical_and_original_stage7_read_only": True,
    }
    write_json(out / "step7_run_config.json", config)
    summary = {
        "schema": s7.SCHEMA, "replay_schema": SCHEMA,
        "advisor_status": ADVISOR,
        "n_splits": 25, "n_models": 2, "n_methods": 4, "n_pairs": 6,
        "n_rows": 300, "historical_rows_preserved": 150,
        "fresh_mice_pair_rows_recomputed": 150,
        "shap_values_from_direct_replay_not_canonical": True,
        "all_test_participants_and_y_labels_aligned": True,
        "rf_permutation_seed_rule": "2026 + split_id",
        "maximum_reference_metric_difference": report["reference_300_recalculation_max_abs_diff"],
        "reference_300_csv_byte_identical": stage7_result_sha == STAGE7_CSV_SHA,
        "source_direct_replay_sha256_manifest": report["replay_fingerprints"]["replay_audit_manifest_sha256"],
        "source_original_stage7_csv_sha256": STAGE7_CSV_SHA,
        "private_participant_data_not_in_outputs": True,
        "metric_scales": {"LR": "log_odds", "RF": "positive_class_probability"},
        "cross_model_shap_mse_not_comparable": True,
    }
    write_json(out / "step7_summary.json", summary)
    regular_names = ("shap_stability_metrics_4_methods_25_splits.csv",
                     "step7_run_config.json", "step7_summary.json")
    write_json(out / "step7_aggregate_sha256.json",
               {name: hash_file(out / name) for name in regular_names})
    compare = {
        "schema": SCHEMA, "advisor_status": ADVISOR,
        "reference_stage7_manifest_sha256": STAGE7_MANIFEST_SHA,
        "reference_stage7_results_sha256": STAGE7_CSV_SHA,
        "replay_results_sha256": stage7_result_sha,
        "byte_identical_300_row_csv": stage7_result_sha == STAGE7_CSV_SHA,
        "reference_max_abs_float_metric_difference": report["reference_300_recalculation_max_abs_diff"],
        "reference_float_cells_nonzero_difference": report["reference_float_cells_nonzero_difference"],
        "all_300_rows_pass": True,
        "legacy_150_reference_preserved": True,
        "new_150_mice_rows_recomputed_from_private_replay": True,
        "read_only_canonical_and_original_stage7": True,
    }
    write_json(out / "stage11_step7_replay_comparison.json", compare)
    replay_sha_names = ("step7_aggregate_sha256.json", "stage11_step7_replay_comparison.json")
    write_json(out / "stage11_step7_replay_sha256.json",
               {name: hash_file(out / name) for name in replay_sha_names})
    verified_manifest(out, "step7_aggregate_sha256.json", set(regular_names))
    verified_manifest(out, "stage11_step7_replay_sha256.json", set(replay_sha_names))
    announce("STEP 11 STEP7 REPLAY PASS | 300/300 SHAP stability rows recalculated; 150 historical rows exact; SHA256 PASS")
    announce(f"STEP 11 STEP7 REPLAY OUTPUT | {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    parser.add_argument("--stage3-shap-dir", type=Path, required=True)
    parser.add_argument("--replay-shap-dir", type=Path, required=True)
    parser.add_argument("--canonical-stage6-dir", type=Path, required=True)
    parser.add_argument("--stage7-ref-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    srcs = {
        "stage3_shap": s7.rooted(root, args.stage3_shap_dir),
        "direct_shap_replay": s7.rooted(root, args.replay_shap_dir),
        "canonical_corrected_stage6": s7.rooted(root, args.canonical_stage6_dir),
        "stage7_original": s7.rooted(root, args.stage7_ref_dir),
    }
    need(all(path.is_dir() and not path.is_symlink() for path in srcs.values()),
         "one or more required input directories are missing or symlinks")
    need(len(set(srcs.values())) == 4, "input directories must be distinct")
    if args.mode == "preflight":
        need(args.out is None, "preflight is read-only: do not supply --out")
    else:
        need(args.out is not None, "run requires --out")
    announce("STEP 11 STEP7 START | validate replay lineage, recompute 300 paired metrics from private SHAP, compare Stage 7")
    rows, report = recompute_everything(srcs["stage3_shap"], srcs["direct_shap_replay"],
                                        srcs["canonical_corrected_stage6"], srcs["stage7_original"])
    if args.mode == "preflight":
        announce("STEP 11 STEP7 PREFLIGHT PASS | READ-ONLY; 300/300 recomputed and matched; no model/SHAP runs or files written")
    else:
        run_output(s7.rooted(root, args.out), srcs, rows, report)


if __name__ == "__main__":
    main()
