#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 11: replay Stage 9 paired inference from INDEPENDENT Stage 5/7.

Scientific formulas are imported from the ORIGINAL SHA-pinned Stage 9 source.
The original Stage 9 output is a READ-ONLY comparator, never input to the
calculation. Input provenance and the 25 original/replay Stage 5 checkpoints
are verified. Only aggregate data are exported. No model/SHAP calculation.

Execute from the project root, next to stage4_mice_inference_step9.py:
 python -u stage11_inference_replay_step9.py --mode preflight \
   --stage3-cv output/stage3_final/cv_metrics_final.csv \
   --original-stage5-dir output/stage4_mice/local_rechecks/stage5_cv25_20261116T064635Z_2203846 \
   --replay-stage5-dir output/stage4_mice/local_rechecks/stage11_replay_stage5_20261117T151930Z_9784 \
   --original-stage7-dir output/stage4_mice/local_rechecks/stage7_stability_20261117T132020Z_9784 \
   --replay-stage7-dir output/stage4_mice/local_rechecks/stage11_replay_step7_20261121T051829Z_9784 \
   --stage9-ref-dir output/stage4_mice/local_rechecks/stage9_inference_20261117T140157Z_9784

Research remains PROVISIONAL; advisor approval pending.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
import stage4_mice_inference_step9 as s9

SCHEMA = "sina-stage11-step9-inference-replay-v1"
ADVISOR = "PENDING; PROVISIONAL"
STAGE9_SOURCE_SHA = "132863ca7458b1ae7b63451d9a152d63738fe94e03227619979024bc78ae46aa"
STAGE11_STEP7_SOURCE_SHA = "026f88ab8560d31d1613e94e93065fa49bf11cf8ceff43c2e42947098829bc01"
STAGE3_CV_SHA = s9.EXPECTED_STAGE3
ORIGINAL_STAGE5_MANIFEST_SHA = "6e2daaa7cc25dafc139154094438b0b1a94733aec1dc61b715e09958dcd48e5f"
REPLAY_STAGE5_MANIFEST_SHA = "389faf824d3338488e44a0c9ec817d8b042ed6dc62d5e112067a07ecf5f26336"
ORIGINAL_STAGE7_MANIFEST_SHA = "73061e16170d46f0ea2e5b2f74dde85db34fa19c2a778d7f8dc85a3e21cf8379"
ORIGINAL_STAGE7_RESULTS_SHA = s9.EXPECTED_STAGE7
ORIGINAL_STAGE9_MANIFEST_SHA = "fea043e889721ed0d769f221a2735976f569ee16591ad59de700c35f0727b423"
ORIGINAL_STAGE9_RESULTS_SHA = "13c758e266d019e8e2142c096d417acc50f37364205800134a952ddd57a2b3c2"
DIRECT_SHAP_AUDIT_MANIFEST_SHA = "4a692cfffd84c5d19bf02057470c67f01448b76e7fa4e57281158493c143d056"
TOL = 1e-12
SHA_FORMAT = re.compile(r"[0-9a-f]{64}\Z")
STAGE5_AGGREGATES = {"mice_cv_metrics_25_splits.csv", "mice_imputer_diagnostics_25_splits.csv",
                     "mice_inner_tuning_25_splits.csv", "step5_summary.json"}
STAGE7_AGGREGATES = {"shap_stability_metrics_4_methods_25_splits.csv", "step7_run_config.json", "step7_summary.json"}
STAGE9_AGGREGATES = {"step9_corrected_intervals_4_methods.csv", "step9_run_config.json", "step9_summary.json"}
STEP7_REPLAY_AUDIT = {"step7_aggregate_sha256.json", "stage11_step7_replay_comparison.json"}
STEP5_TIMING = {"tuning_seconds", "final_fit_seconds_total"}
STAGE5_MODEL_NAMES = {
    *(f"models/MICE_draw{i:02d}_imputer_bundle.joblib" for i in range(1, 6)),
    *(f"models/{model}_draw{i:02d}_classifier.joblib"
      for model in ("LR", "RF") for i in range(1, 6)),
}
STAGE5_SPLIT_NAMES = STAGE5_MODEL_NAMES | {"mice_performance.csv", "mice_imputer_diagnostics.csv", "mice_inner_tuning.csv"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("STOP: " + reason)


def log(s: str) -> None:
    print(s, flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def regular_file(p: Path) -> None:
    require(p.is_file() and not p.is_symlink(), f"missing or symlinked file: {p}")


def read_json(p: Path) -> dict[str, Any]:
    regular_file(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"expected JSON object: {p}")
    return obj


def manifest_check(folder: Path, filename: str, expected: set[str],
                   expected_manifest_sha: str | None = None) -> str:
    manifest_file = folder / filename
    entries = read_json(manifest_file)
    require(set(entries) == expected, f"unexpected manifest entries: {manifest_file}")
    manifest_sha = sha(manifest_file)
    if expected_manifest_sha is not None:
        require(manifest_sha == expected_manifest_sha, f"pinned manifest mismatch: {manifest_file}")
    for rel, digest in entries.items():
        require(isinstance(digest, str) and SHA_FORMAT.fullmatch(digest) is not None,
                f"invalid SHA256 for {rel}")
        require(Path(rel).as_posix() == rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
                f"unsafe manifest file path {rel}")
        p = folder / rel
        regular_file(p)
        require(sha(p) == digest, f"SHA256 mismatch: {p}")
    return manifest_sha


def csv_strict(p: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    regular_file(p)
    with p.open(encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        names = tuple(rdr.fieldnames or ())
        rows = list(rdr)
    require(bool(names) and len(rows) > 0 and all(None not in r for r in rows),
            f"empty/malformed CSV: {p}")
    return names, rows


def repo_dir(root: Path, arg: Path, label: str) -> Path:
    base = (root / "output" / "stage4_mice" / "local_rechecks").resolve()
    raw = arg.expanduser()
    abs_path = raw if raw.is_absolute() else root / raw
    require(not abs_path.is_symlink(), f"symlink source forbidden: {label}")
    path = abs_path.resolve()
    require(path.is_dir() and path.parent == base and not path.is_symlink(),
            f"{label}: expected an existing direct directory of local_rechecks: {path}")
    return path


def stage5_lineage(original: Path, replay: Path) -> dict[str, Any]:
    orig_msha = manifest_check(original, "step5_aggregate_sha256.json", STAGE5_AGGREGATES,
                               ORIGINAL_STAGE5_MANIFEST_SHA)
    rep_msha = manifest_check(replay, "step5_aggregate_sha256.json", STAGE5_AGGREGATES,
                              REPLAY_STAGE5_MANIFEST_SHA)
    a = read_json(original / "step5_summary.json")
    b = read_json(replay / "step5_summary.json")
    require(a == b and a.get("completed_splits") == 25 and a.get("metric_rows") == 50 and
            a.get("imputer_diagnostic_rows") == 125 and a.get("tuning_rows") == 325,
            "Stage 5 source/replay summaries not identical and complete")
    old_cfg, new_cfg = read_json(original / "run_config.json"), read_json(replay / "run_config.json")
    require(old_cfg.get("schema") == new_cfg.get("schema") == "sina-stage4-step5-v1" and
            isinstance(old_cfg.get("identity"), dict) and old_cfg["identity"] == new_cfg.get("identity"),
            "original/replay Stage 5 code, dataset, splits or environment provenance differs")
    old_headers, old_rows = csv_strict(original / "mice_cv_metrics_25_splits.csv")
    new_headers, new_rows = csv_strict(replay / "mice_cv_metrics_25_splits.csv")
    require(old_headers == new_headers and len(old_rows) == len(new_rows) == 50 and
            STEP5_TIMING.issubset(old_headers), "Stage 5 metrics schema is not as expected")
    id_columns = ("split_id", "model")
    old = {tuple(r[x] for x in id_columns): r for r in old_rows}
    new = {tuple(r[x] for x in id_columns): r for r in new_rows}
    require(len(old) == len(new) == 50 and set(old) == set(new),
            "Stage 5 split/model metric keys differ or duplicate")
    for key in old:
        for col in old_headers:
            if col not in STEP5_TIMING:
                require(old[key][col] == new[key][col],
                        f"Stage 5 replay scientific metric differs at {key}, {col}")
            else:
                val = float(new[key][col])
                require(math.isfinite(val) and val >= 0, f"invalid Stage 5 replay runtime: {key}/{col}")
    log("STEP 11 STEP9 PERFORMANCE PASS | 50/50 replay performance rows exact in all scientific fields; runtime excluded")
    models_identical = 0
    for sid in range(1, 26):
        pair = []
        for folder in (original, replay):
            split_folder = folder / f"split_{sid:03d}"
            require(split_folder.is_dir() and not split_folder.is_symlink(), f"missing split directory: {split_folder}")
            marker = read_json(split_folder / "split_complete.json")
            entries = marker.get("sha256")
            require(marker.get("schema") == "sina-stage4-step5-v1" and marker.get("split_id") == sid
                    and isinstance(entries, dict) and set(entries) == STAGE5_SPLIT_NAMES,
                    f"unexpected Stage 5 split checkpoint: {split_folder}")
            for rel, digest in entries.items():
                require(isinstance(digest, str) and SHA_FORMAT.fullmatch(digest) is not None,
                        f"invalid checkpoint hash for split {sid}/{rel}")
                artifact = split_folder / rel
                regular_file(artifact)
                require(sha(artifact) == digest, f"checkpoint corrupted: {artifact}")
            pair.append(entries)
        for name in STAGE5_MODEL_NAMES:
            require(pair[0][name] == pair[1][name],
                    f"Stage 5 saved model differs: split {sid}, {name}")
            models_identical += 1
        if sid % 5 == 0:
            log(f"STEP 11 STEP9 CHECKPOINT PASS | {sid}/25 original/replay pairs verified")
    require(models_identical == 375, "not all 375 Stage 5 models were checked")
    log("STEP 11 STEP9 MODEL PASS | 375/375 models byte-identical; 900/900 checkpoint artifacts verified")
    return {"original_manifest_sha256": orig_msha, "replay_manifest_sha256": rep_msha,
            "replay_metrics_sha256": sha(replay / "mice_cv_metrics_25_splits.csv"),
            "n_models_byte_identical": models_identical, "n_checkpoint_artifacts_verified": 900,
            "n_performance_rows_exact": 50}


def stage7_lineage(original: Path, replay: Path, root: Path) -> dict[str, Any]:
    original_sha = manifest_check(original, "step7_aggregate_sha256.json", STAGE7_AGGREGATES,
                                  ORIGINAL_STAGE7_MANIFEST_SHA)
    require(sha(original / "shap_stability_metrics_4_methods_25_splits.csv") == ORIGINAL_STAGE7_RESULTS_SHA,
            "Stage 7 canonical results SHA mismatch")
    replay_sha = manifest_check(replay, "step7_aggregate_sha256.json", STAGE7_AGGREGATES)
    replay_audit_sha = manifest_check(replay, "stage11_step7_replay_sha256.json", STEP7_REPLAY_AUDIT)
    original_summary = read_json(original / "step7_summary.json")
    replay_summary = read_json(replay / "step7_summary.json")
    compare = read_json(replay / "stage11_step7_replay_comparison.json")
    replay_cfg = read_json(replay / "step7_run_config.json")
    require(original_summary.get("n_rows") == replay_summary.get("n_rows") == 300 and
            original_summary.get("n_legacy_rows_exactly_preserved") == 150 and
            replay_summary.get("historical_rows_preserved") == 150 and
            replay_summary.get("fresh_mice_pair_rows_recomputed") == 150 and
            replay_summary.get("shap_values_from_direct_replay_not_canonical") is True and
            replay_summary.get("rf_permutation_seed_rule") == "2026 + split_id" and
            replay_summary.get("advisor_status") == ADVISOR,
            "Stage 7 replay source summary not complete")
    require(compare.get("all_300_rows_pass") is True and
            compare.get("legacy_150_reference_preserved") is True and
            compare.get("new_150_mice_rows_recomputed_from_private_replay") is True and
            compare.get("reference_stage7_results_sha256") == ORIGINAL_STAGE7_RESULTS_SHA and
            compare.get("reference_stage7_manifest_sha256") == ORIGINAL_STAGE7_MANIFEST_SHA,
            "Stage 7 replay comparison attestation invalid")
    require(float(compare.get("reference_max_abs_float_metric_difference", float("nan"))) <= TOL,
            "Stage 7 replay comparison exceeds numeric tolerance")
    require(replay_cfg.get("schema") == "sina-stage11-step7-stability-replay-v1" and
            replay_cfg.get("advisor_status") == ADVISOR and
            replay_cfg.get("script_sha256") == STAGE11_STEP7_SOURCE_SHA and
            sha(root / "stage11_stability_replay_step7.py") == STAGE11_STEP7_SOURCE_SHA and
            replay_summary.get("source_direct_replay_sha256_manifest") == DIRECT_SHAP_AUDIT_MANIFEST_SHA,
            "Stage 7 replay source code/direct-SHAP provenance differs")
    old_heads, old_rows = csv_strict(original / "shap_stability_metrics_4_methods_25_splits.csv")
    new_heads, new_rows = csv_strict(replay / "shap_stability_metrics_4_methods_25_splits.csv")
    require(old_heads == new_heads and len(old_rows) == len(new_rows) == 300,
            "replayed Stage 7 stability table schema/count differs")
    index_fields = ("split_id", "model", "method_a", "method_b")
    a = {tuple(r[k] for k in index_fields): r for r in old_rows}
    b = {tuple(r[k] for k in index_fields): r for r in new_rows}
    require(len(a) == len(b) == 300 and set(a) == set(b),
            "Stage 7 pair identity set differs or is duplicated")
    numeric_fields = {"spearman_rank_correlation", "top5_jaccard", "mse_shap", "mae_shap"}
    diff = 0.0
    for key in a:
        for field in old_heads:
            if field in numeric_fields:
                x, y = float(a[key][field]), float(b[key][field])
                require(math.isfinite(x) and math.isfinite(y) and abs(x - y) <= TOL,
                        f"Stage 7 stability numerical mismatch {key}/{field}")
                diff = max(diff, abs(x - y))
            else:
                require(a[key][field] == b[key][field],
                        f"Stage 7 stability categorical mismatch {key}/{field}")
    log(f"STEP 11 STEP9 STABILITY PASS | 300/300 replay stability rows; max_abs_diff={diff:.3g}")
    return {"original_manifest_sha256": original_sha, "replay_manifest_sha256": replay_sha,
            "replay_audit_manifest_sha256": replay_audit_sha,
            "replay_stability_csv_sha256": sha(replay / "shap_stability_metrics_4_methods_25_splits.csv"),
            "n_stability_rows_checked": 300, "max_stability_difference": diff}


def reference_step9(ref: Path, root: Path) -> list[dict[str, str]]:
    manifest_check(ref, "step9_aggregate_sha256.json", STAGE9_AGGREGATES, ORIGINAL_STAGE9_MANIFEST_SHA)
    require(sha(ref / "step9_corrected_intervals_4_methods.csv") == ORIGINAL_STAGE9_RESULTS_SHA,
            "original Step 9 scientific report SHA differs from pinned reference")
    require(sha(root / "stage4_mice_inference_step9.py") == STAGE9_SOURCE_SHA,
            "original Step 9 scientific formula code SHA differs")
    config = read_json(ref / "step9_run_config.json")
    summary = read_json(ref / "step9_summary.json")
    require(config.get("script_sha256") == STAGE9_SOURCE_SHA and
            config.get("input_sha256") == {
                "cv_metrics_final.csv": STAGE3_CV_SHA,
                "mice_cv_metrics_25_splits.csv": s9.EXPECTED_STAGE5,
                "shap_stability_metrics_4_methods_25_splits.csv": ORIGINAL_STAGE7_RESULTS_SHA,
            }, "Step 9 reference sources are not pinned inputs")
    require(summary.get("rows") == 48 and summary.get("performance_rows") == 24 and
            summary.get("descriptive_shap_rows") == 24 and
            summary.get("auc_holm_family_size") == 12 and
            summary.get("brier_holm_family_size") == 12 and
            summary.get("advisor_status") == ADVISOR,
            "original Stage 9 reference summary invalid")
    head, rows = csv_strict(ref / "step9_corrected_intervals_4_methods.csv")
    require(head == s9.FIELDS and len(rows) == 48, "original Stage 9 output schema invalid")
    log("STEP 11 STEP9 REFERENCE PASS | original audited 48-row Stage 9 report SHA256 pinned")
    return rows


def recompute(stage3: Path, replay5: Path, replay7: Path,
              ref_rows: list[dict[str, str]]) -> tuple[bytes, dict[str, Any], list[dict]]:
    f3 = stage3
    f5 = replay5 / "mice_cv_metrics_25_splits.csv"
    f7 = replay7 / "shap_stability_metrics_4_methods_25_splits.csv"
    # The original input validators are used verbatim on the NEW aggregate data.
    performance, stability = s9.validate_rows((s9.csv_rows(f3), s9.csv_rows(f5), s9.csv_rows(f7)))
    recalculated = []
    for metric in ("auc", "brier", "spearman", "jaccard"):
        for model in s9.MODELS:
            for a, b in s9.PAIRS:
                recalculated.append(s9.construct(metric, model, a, b, performance, stability))
    for family in ("auc", "brier"):
        s9.apply_holm(recalculated, family)
    require(len(recalculated) == len(ref_rows) == 48, "48 Stage 9 comparisons expected")
    for r in recalculated:
        require(r["n_paired_splits"] == 25 and r["df"] == 24 and
                math.isfinite(r["sigma_corr"]) and
                r["ci95_lower"] <= r["mean_difference"] <= r["ci95_upper"],
                "invalid NB variance, CI or repeated-CV sample size")
        if r["metric_family"] in ("spearman", "jaccard"):
            require(all(r[x] == "" for x in ("t_stat", "p_raw", "p_holm", "holm_reject_0_05")),
                    "descriptive SHAP similarity must not have hypothesis p-values")
        else:
            require(0 <= r["p_raw"] <= 1 and 0 <= r["p_holm"] <= 1 and
                    r["p_holm"] >= r["p_raw"], "invalid Holm correction")
    # Check all 48 rows AND each scientific scalar, not just the two rejections.
    pairs = ("metric_family", "model", "method_a", "method_b")
    old = {tuple(r[x] for x in pairs): r for r in ref_rows}
    new = {tuple(str(r[x]) for x in pairs): r for r in recalculated}
    require(len(old) == len(new) == 48 and set(old) == set(new),
            "Stage 9 reference/replay comparison identity mismatched or duplicated")
    max_delta, float_cells = 0.0, 0
    for key, row in new.items():
        ref = old[key]
        for col in s9.FIELDS:
            got, expected = row[col], ref[col]
            if col in ("n_paired_splits", "df"):
                require(int(got) == int(expected), f"integer mismatch {key}/{col}")
            elif (col in ("mean_difference", "sd_difference", "mean_n_train_smaller",
                           "mean_n_test", "correction_factor", "sigma_corr",
                           "ci95_lower", "ci95_upper", "t_stat", "p_raw", "p_holm")
                  and got != "" and expected != ""):
                delta = abs(float(got) - float(expected))
                require(math.isfinite(delta) and delta <= TOL, f"numeric mismatch {key}/{col} = {delta}")
                max_delta = max(max_delta, delta)
                float_cells += 1
            else:
                require(str(got) == str(expected), f"string mismatch {key}/{col}: {got!r}!={expected!r}")
    significant = [r for r in recalculated if r["metric_family"] in ("auc", "brier") and
                   r["holm_reject_0_05"]]
    expected_significant = {(family, "LR", "Median/Mode", "KNN k=5") for family in ("auc", "brier")}
    require({tuple(r[x] for x in pairs) for r in significant} == expected_significant and
            len(significant) == 2, "Step 9 Holm findings changed relative to audited reference")
    outside_bounds = int(sum(bool(r["ci95_upper"] > 1 or r["ci95_lower"] < -1)
                             for r in recalculated if r["metric_family"] in ("spearman", "jaccard")))
    require(outside_bounds == 6, "expected six unbounded descriptive t intervals to cross limits")
    # Exact serialization from canonical writer (CSV newline=CRLF) for audit.
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=s9.FIELDS)
    writer.writeheader()
    writer.writerows(recalculated)
    serialized = buf.getvalue().encode("utf-8")
    byte_exact = hashlib.sha256(serialized).hexdigest() == ORIGINAL_STAGE9_RESULTS_SHA
    log(f"STEP 11 STEP9 INFERENCE PASS | 48/48 rows; {float_cells} numeric cells; "
        f"max_abs_diff={max_delta:.3g}; Holm significant=2 (neither involves MICE)")
    log(f"STEP 11 STEP9 SCIENTIFIC HASH | byte-identical to reference={byte_exact}")
    return serialized, {"rows": 48, "numeric_cells_checked": float_cells,
                        "max_abs_reference_difference": max_delta,
                        "scientific_csv_byte_identical_to_reference": byte_exact,
                        "holm_significant_rejections": 2,
                        "holm_significant_involving_mice": 0,
                        "unbounded_descriptive_intervals": outside_bounds,
                        "scientific_csv_sha256": hashlib.sha256(serialized).hexdigest()}, recalculated


def json_write(path: Path, obj: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


def emit(root: Path, outarg: Path, input_files: dict[str, Path], provenance: dict[str, Any],
         result: bytes, comparison: dict[str, Any]) -> None:
    local_base = (root / "output" / "stage4_mice" / "local_rechecks").resolve()
    raw = outarg.expanduser()
    output = (raw if raw.is_absolute() else root / raw).resolve()
    require(output.parent == local_base and output.name.startswith("stage11_replay_step9_"),
            "output must be a new stage11_replay_step9_* child of local_rechecks")
    require(not output.exists() and not output.is_symlink(), "output exists; do not overwrite")
    require(not any(output == p or output.is_relative_to(p) for p in input_files.values() if p.is_dir()),
            "output overlaps protected source")
    tmp = local_base / f".work_{output.name}_{os.getpid()}"
    require(not tmp.exists(), "temporary output path already exists")
    tmp.mkdir(exist_ok=False)
    try:
        with (tmp / "step9_corrected_intervals_4_methods.csv").open("xb") as fh:
            fh.write(result)
        summary = {
            "schema": "sina-stage4-step9-v1", "replay_schema": SCHEMA,
            "advisor_status": ADVISOR, "n_splits": 25, "n_models": 2, "n_pairs": 6,
            "rows": 48, "performance_rows": 24, "descriptive_shap_rows": 24,
            "auc_holm_family_size": 12, "brier_holm_family_size": 12,
            "degrees_of_freedom": 24,
            "variance_formula": "(1/25 + mean_n_test/mean_n_train_smaller) * sample_variance_over_25_splits",
            "point_difference_rule": "method_a minus method_b; Brier negative favors method_a",
            "shap_CI_rule": "descriptive only; do not present hypothesis p-values",
            "cv_dependency_warning": "25 splits are repeated 5-fold CV; corrected t is approximate; no independent-sample interpretation",
            "interval_scope_warning": "per-contrast 95% CI is NOT simultaneous Holm-adjusted CI",
            "bounded_similarity_warning": "six approximate descriptive t intervals exceed natural similarity bounds; do not silently clip",
            "data_source": "independent Stage 5 and Stage 7 replay aggregates",
            "input_sha256": {k: sha(p) for k, p in input_files.items() if p.is_file()},
            "provisional_results_not_advisor_approved": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        json_write(tmp / "step9_summary.json", summary)
        config = {"schema": SCHEMA, "advisor_status": ADVISOR,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha(Path(__file__).resolve()),
                  "original_stage9_formula_script_sha256": STAGE9_SOURCE_SHA,
                  "inputs": {k: str(v) for k, v in input_files.items()},
                  "provenance": provenance,
                  "original_stage9_report_read_only": True,
                  "new_model_or_SHAP_computation": False,
                  "participant_level_data_exported": False}
        json_write(tmp / "step9_run_config.json", config)
        names = ("step9_corrected_intervals_4_methods.csv", "step9_summary.json", "step9_run_config.json")
        json_write(tmp / "step9_aggregate_sha256.json", {n: sha(tmp / n) for n in names})
        comparison_report = {"schema": SCHEMA, "advisor_status": ADVISOR,
                             "original_stage9_manifest_sha256": ORIGINAL_STAGE9_MANIFEST_SHA,
                             "original_stage9_csv_sha256": ORIGINAL_STAGE9_RESULTS_SHA,
                             **comparison,
                             "using_independent_stage5_stage7_replay": True,
                             "all_48_recomputed_rows_match_canonical": True,
                             "no_model_training_or_SHAP_computation": True}
        json_write(tmp / "stage11_step9_replay_comparison.json", comparison_report)
        json_write(tmp / "stage11_step9_replay_sha256.json", {
            n: sha(tmp / n) for n in ("step9_aggregate_sha256.json", "stage11_step9_replay_comparison.json")
        })
        manifest_check(tmp, "step9_aggregate_sha256.json", set(names))
        manifest_check(tmp, "stage11_step9_replay_sha256.json",
                       {"step9_aggregate_sha256.json", "stage11_step9_replay_comparison.json"})
        require(not output.exists(), "output path occupied during save")
        tmp.rename(output)
    except Exception:
        # Preserve temporary directory for investigation, never mutate input folders.
        raise
    log("STEP 11 STEP9 REPLAY PASS | 48/48 paired rows recalculated; Holm 12+12; descriptive 24; SHA256 PASS")
    log(f"STEP 11 STEP9 REPLAY OUTPUT | {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--stage3-cv", type=Path, required=True)
    parser.add_argument("--original-stage5-dir", type=Path, required=True)
    parser.add_argument("--replay-stage5-dir", type=Path, required=True)
    parser.add_argument("--original-stage7-dir", type=Path, required=True)
    parser.add_argument("--replay-stage7-dir", type=Path, required=True)
    parser.add_argument("--stage9-ref-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require((args.mode == "preflight" and args.out is None) or
            (args.mode == "run" and args.out is not None),
            "preflight forbids --out; run requires new --out")
    root = Path(__file__).resolve().parent
    require(sha(root / "stage4_mice_inference_step9.py") == STAGE9_SOURCE_SHA,
            "original Stage 9 scientific source changed")
    # Stage 3 is the historical unchanged reference; all new calculations use replay Stage 5/7.
    stage3_raw = args.stage3_cv.expanduser()
    stage3 = (stage3_raw if stage3_raw.is_absolute() else root / stage3_raw).resolve()
    regular_file(stage3)
    require(sha(stage3) == STAGE3_CV_SHA, "historical Stage 3 CV source SHA mismatch")
    o5 = repo_dir(root, args.original_stage5_dir, "original Stage 5")
    r5 = repo_dir(root, args.replay_stage5_dir, "replayed Stage 5")
    o7 = repo_dir(root, args.original_stage7_dir, "original Stage 7")
    r7 = repo_dir(root, args.replay_stage7_dir, "replayed Stage 7")
    o9 = repo_dir(root, args.stage9_ref_dir, "original Stage 9")
    require(len({o5, r5, o7, r7, o9}) == 5, "all source directories must be distinct")
    log("STEP 11 STEP9 START | validate Stage 3/5/7/9 lineage; recompute 48 Stage 9 rows from independent replay")
    stage5_provenance = stage5_lineage(o5, r5)
    stage7_provenance = stage7_lineage(o7, r7, root)
    ref_rows = reference_step9(o9, root)
    calculated, comparison, _ = recompute(stage3, r5, r7, ref_rows)
    if args.mode == "preflight":
        log("STEP 11 STEP9 PREFLIGHT PASS | READ-ONLY; 48/48 rows independently recalculated; no files written")
        return
    inputs = {"stage3_cv": stage3, "original_stage5": o5,
              "replay_stage5": r5, "original_stage7": o7,
              "replay_stage7": r7, "original_stage9": o9,
              "replay_stage5_metrics": r5 / "mice_cv_metrics_25_splits.csv",
              "replay_stage7_stability": r7 / "shap_stability_metrics_4_methods_25_splits.csv"}
    emit(root, args.out, inputs,
         {"stage5": stage5_provenance, "stage7": stage7_provenance}, calculated, comparison)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STEP 11 STEP9 FAIL | {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
