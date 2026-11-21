#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 11: reproduce all Stage 8 MICE diagnostic tables from Stage 5 REPLAY.

Read-only preflight checks original and replay Stage 5 aggregate SHA256 files,
all 25+25 split checkpoint manifests (900 files), and byte equality of all
375 fitted imputer/classifier artifacts. The 125 replay diagnostic records
are compared with the original, excluding only the timing column, then the
three original Stage 8 tables (16 rows) and summary are RECOMPUTED using the
UNCHANGED, hash-pinned Stage 8 scientific formulas.

On --mode run, write a new independent aggregate-only report directory.
NEVER load fitted models, export participant-level data, refit MICE, run SHAP,
or modify original/reference/replay inputs. Advisor approval pending.

Usage (from research root):
 python -u stage11_diagnostics_replay_step8.py --mode preflight \
   --original-stage5-dir output/stage4_mice/local_rechecks/stage5_cv25_20261116T064635Z_2203846 \
   --replay-stage5-dir output/stage4_mice/local_rechecks/stage11_replay_stage5_20261117T151930Z_9784 \
   --stage8-ref-dir output/stage4_mice/local_rechecks/stage8_diagnostics_20261117T134348Z_9784
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

sys.dont_write_bytecode = True

import stage4_mice_diagnostics_step8 as s8

SCHEMA = 'sina-stage11-step8-diagnostics-replay-v1'
ADVISOR = 'PENDING; PROVISIONAL'
STEP8_SCRIPT_SHA = 'b833ca462f0bde40449f351556a9bac4a41457ed82e1f74ecd096e12f7be211e'
STEP8_MANIFEST_SHA = '61ae6209992af4384f89d06d1408d89287c2c3eb6c78a68166e759768a664822'
STEP8_CSV_SHA = '2a088d5c5871ee0a6ff1c14418ba19adb749f95c600e1a271e513aae296830c7'
STEP8_SUMMARY_SHA = 'd4471117df4c7432c6333050b1fd4738e56c04a1532d47ddaf43e61a7933a952'
REPLAY_STEP5_MANIFEST_SHA = '389faf824d3338488e44a0c9ec817d8b042ed6dc62d5e112067a07ecf5f26336'
REPLAY_STEP5_DIAGNOSTICS_SHA = 'e7c9904a7758f7d84727ce1067b0245a53f01bb501332bab422632685c064e47'

# Cross-execution CSV fields allowed to change because they record elapsed time.
DIAGNOSTIC_RUNTIME_FIELDS = {'imputer_seconds'}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError('STOP: ' + reason)


def announce(text: str) -> None:
    print(text, flush=True)


def sha(path: Path) -> str:
    return s8.digest(path)


def load_json(path: Path) -> dict[str, Any]:
    return s8.json_object(path)


def source_directory(root: Path, arg: Path, name: str) -> Path:
    """Only real directories immediately under local_rechecks; reject symlinks."""
    base = (root / 'output' / 'stage4_mice' / 'local_rechecks').resolve()
    require(base.is_dir(), f'local_rechecks does not exist: {base}')
    unexpanded = arg.expanduser()
    absolute = unexpanded if unexpanded.is_absolute() else root / unexpanded
    require(not absolute.is_symlink(), f'{name}: symlink directory is not permitted')
    path = absolute.resolve()
    require(path.parent == base and path.is_dir() and not path.is_symlink(),
            f'{name} must be an existing direct directory under {base}: {path}')
    return path


def check_manifest(folder: Path, name: str, expected_names: set[str],
                   pinned_sha: str | None = None) -> tuple[dict[str, str], str]:
    path = folder / name
    manifest = load_json(path)
    require(set(manifest) == expected_names,
            f'unexpected entries in {path}: {sorted(manifest)}')
    manifest_sha = sha(path)
    if pinned_sha is not None:
        require(manifest_sha == pinned_sha,
                f'pinned SHA256 mismatch for manifest: {path}')
    for filename, expected_sha in manifest.items():
        require(isinstance(expected_sha, str) and len(expected_sha) == 64 and
                all(c in '0123456789abcdef' for c in expected_sha),
                f'invalid SHA256 for {filename}')
        target = s8.safe_relative_file(folder, filename)
        require(sha(target) == expected_sha,
                f'aggregate SHA256 mismatch in {folder.name}: {filename}')
    return manifest, manifest_sha


def validate_reference_stage8(ref: Path) -> dict[str, Any]:
    names = {s8.AGGREGATE_CSV, s8.SUMMARY_FILE, s8.CONFIG_FILE}
    manifest, manifest_sha = check_manifest(ref, s8.HASH_FILE, names, STEP8_MANIFEST_SHA)
    require(manifest[s8.AGGREGATE_CSV] == STEP8_CSV_SHA and
            manifest[s8.SUMMARY_FILE] == STEP8_SUMMARY_SHA,
            'original Stage 8 output does not match the independently reviewed aggregate files')
    config = load_json(ref / s8.CONFIG_FILE)
    summary = load_json(ref / s8.SUMMARY_FILE)
    require(config.get('schema') == s8.SCHEMA and
            config.get('advisor_status') == ADVISOR and
            config.get('script_sha256') == STEP8_SCRIPT_SHA and
            config.get('source', {}).get('stage5_aggregate_manifest_sha256') ==
                s8.EXPECTED_STAGE5_MANIFEST_SHA256,
            'original Stage 8 provenance/config invalid')
    require(summary.get('schema') == s8.SCHEMA and
            summary.get('advisor_status') == ADVISOR and
            summary.get('n_output_csv_rows') == 16 and
            summary.get('n_split_draw_records') == 125 and
            summary.get('posterior_sampling') is True and
            summary.get('participant_level_data_exported') is False,
            'original Stage 8 summary metadata invalid')
    with (ref / s8.AGGREGATE_CSV).open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        require(tuple(reader.fieldnames or []) == s8.REPORT_FIELDS,
                'original Stage 8 CSV report has unexpected columns')
        report_rows = list(reader)
    require(len(report_rows) == 16 and all(None not in row for row in report_rows),
            'original Stage 8 CSV does not have 16 valid rows')
    announce('STEP 11 STEP8 REFERENCE PASS | audited Stage 8 pinned; 3 tables / 16 rows')
    return {
        'manifest_sha256': manifest_sha,
        'report_csv_sha256': STEP8_CSV_SHA,
        'summary_sha256': STEP8_SUMMARY_SHA,
    }


def validate_replay_stage5(root: Path, original: Path, replay: Path, ref: Path) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    # The original Stage 8 input validator is intentionally left UNCHANGED.
    original_folder, original_records, original_prov = s8.verify_input(root, original)
    require(original_folder == original, 'original Stage 5 path changed unexpectedly')
    manifest, replay_manifest_sha = check_manifest(replay, 'step5_aggregate_sha256.json',
                                                  set(s8.MANIFEST_FILES), REPLAY_STEP5_MANIFEST_SHA)
    require(manifest['mice_imputer_diagnostics_25_splits.csv'] == REPLAY_STEP5_DIAGNOSTICS_SHA,
            'replay Stage 5 diagnostics CSV differs from audited independent replay')
    require(manifest['step5_summary.json'] == s8.EXPECTED_STEP5_SUMMARY_SHA256,
            'replay Stage 5 summary differs from original verified summary')
    summary = load_json(replay / 'step5_summary.json')
    require(summary.get('schema') == s8.STAGE5_SCHEMA and
            summary.get('completed_splits') == 25 and
            summary.get('mice_draws') == 5 and
            summary.get('imputer_diagnostic_rows') == 125 and
            summary.get('metric_rows') == 50 and
            summary.get('tuning_rows') == 325 and
            summary.get('has_nonfinite_auc_or_brier') is False,
            'replay Stage 5 summary is inconsistent')
    old_cfg = load_json(original / 'run_config.json')
    new_cfg = load_json(replay / 'run_config.json')
    require(old_cfg.get('schema') == new_cfg.get('schema') == s8.STAGE5_SCHEMA and
            isinstance(old_cfg.get('identity'), dict) and
            old_cfg['identity'] == new_cfg.get('identity'),
            'original and replay Stage 5 dataset/split/code/environment provenance differ')
    announce('STEP 11 STEP8 LINEAGE PASS | original/replay Stage 5 identity and four aggregate manifests verified')

    raw = s8.csv_rows(replay / 'mice_imputer_diagnostics_25_splits.csv')
    records = s8.numeric_rows(raw)
    s8.validate_diagnostics(records)
    replay_markers = s8.validate_splits(replay, raw)  # 25 x 18 SHA256, including saved models

    orig_markers = original_prov['stage5_split_checkpoint_markers_sha256']
    ref_config = load_json(ref / 'step8_run_config.json')
    require(ref_config.get('source', {}).get('stage5_split_checkpoint_markers_sha256') == orig_markers,
            'original Stage 5 checkpoint hashes do not agree with Stage 8 reference provenance')

    model_names = {name for name in s8.canonical_split_files() if name.startswith('models/')}
    require(len(model_names) == 15, 'internal error: 15 saved models per split expected')
    n_identical = 0
    for split in range(1, 26):
        old_marker = load_json(original / f'split_{split:03d}' / 'split_complete.json')
        new_marker = load_json(replay / f'split_{split:03d}' / 'split_complete.json')
        for name in model_names:
            require(old_marker['sha256'][name] == new_marker['sha256'][name],
                    f'original/replay fitted model not byte-identical: split={split} {name}')
            n_identical += 1
        if split % 5 == 0:
            announce(f'STEP 11 STEP8 CHECKPOINT PASS | {split}/25 original/replay pairs; 18 files each checked')
    require(n_identical == 375 and len(replay_markers) == 25 and len(orig_markers) == 25,
            'not all replay Stage 5 checkpoints/models validated')
    announce('STEP 11 STEP8 MODEL PASS | 375/375 saved model artifacts byte-identical; 900 checkpoint files SHA256 PASS')

    original_raw = s8.csv_rows(original / 'mice_imputer_diagnostics_25_splits.csv')
    original_by_key = {(int(row['split_id']), int(row['imputation_id'])): row for row in original_raw}
    replay_by_key = {(int(row['split_id']), int(row['imputation_id'])): row for row in raw}
    expected_keys = {(split, draw) for split in range(1, 26) for draw in range(1, 6)}
    require(len(original_by_key) == len(replay_by_key) == len(original_raw) == len(raw) == 125 and
            set(original_by_key) == set(replay_by_key) == expected_keys,
            'replay and original diagnostic row keys incomplete or duplicated')
    differences_in_runtime = 0
    for key in sorted(expected_keys):
        o, r = original_by_key[key], replay_by_key[key]
        for name in s8.DIAGNOSTICS_FIELDS:
            if name in DIAGNOSTIC_RUNTIME_FIELDS:
                t = float(r[name])
                require(math.isfinite(t) and t >= 0, f'nonfinite replay runtime at {key}')
                differences_in_runtime += (o[name] != r[name])
            else:
                require(o[name] == r[name],
                        f'non-runtime scientific diagnostic differs at split/draw {key}: {name}')
    announce('STEP 11 STEP8 DIAGNOSTICS PASS | 125/125 replay records identical to original in all 20 scientific fields; runtime excluded')
    return records, {
        'original_stage5_manifest_sha256': original_prov['stage5_aggregate_manifest_sha256'],
        'replay_stage5_manifest_sha256': replay_manifest_sha,
        'replay_stage5_diagnostics_sha256': sha(replay / 'mice_imputer_diagnostics_25_splits.csv'),
        'original_stage5_checkpoint_count': len(orig_markers),
        'replay_stage5_checkpoint_count': len(replay_markers),
        'byte_identical_fitted_models': n_identical,
        'scientific_cells_compared_exactly': len(expected_keys) * (len(s8.DIAGNOSTICS_FIELDS) - len(DIAGNOSTIC_RUNTIME_FIELDS)),
        'runtime_cells_differing': differences_in_runtime,
        'source_identity_match': True,
    }


def serialize_csv(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(s8.REPORT_FIELDS), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def compare_calculated(ref: Path, records: list[dict[str, float | int]]) -> tuple[bytes, bytes, dict[str, Any]]:
    rows, summary = s8.calculate(records)  # Same scientific formula, new replay records
    report_bytes = serialize_csv(rows)
    summary_bytes = s8.json_text(summary).encode('utf-8')
    expected_csv = (ref / s8.AGGREGATE_CSV).read_bytes()
    expected_summary = (ref / s8.SUMMARY_FILE).read_bytes()
    require(report_bytes == expected_csv and summary_bytes == expected_summary,
            'recomputed Stage 8 report or summary differs from original Stage 8 reference')
    require(hashlib.sha256(report_bytes).hexdigest() == STEP8_CSV_SHA and
            hashlib.sha256(summary_bytes).hexdigest() == STEP8_SUMMARY_SHA,
            'recomputed Stage 8 report hashes differ from independent audited fingerprints')
    announce('STEP 11 STEP8 REPLAY MATCH PASS | 3 tables / 16 rows and summary byte-identical to original; max_abs_diff=0')
    return report_bytes, summary_bytes, summary


def save_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as fh:
        fh.write(s8.json_text(obj))


def emit_output(root: Path, out_arg: Path, srcs: dict[str, Path],
                csv_bytes: bytes, summary_bytes: bytes,
                summary: dict[str, Any], provenance: dict[str, Any],
                reference: dict[str, Any]) -> None:
    parent = (root / 'output' / 'stage4_mice' / 'local_rechecks').resolve()
    arg = out_arg.expanduser()
    destination = (arg if arg.is_absolute() else root / arg).resolve()
    require(destination.parent == parent and destination.name.startswith('stage11_replay_step8_'),
            'output must be a new stage11_replay_step8_* child of local_rechecks')
    require(not destination.exists() and not destination.is_symlink(),
            'output directory already exists; do not overwrite')
    require(destination not in srcs.values(), 'output directory overlaps a source')
    temp = parent / f'.work_{destination.name}_{os.getpid()}'
    require(not temp.exists() and not temp.is_symlink(), 'temporary output path already exists')
    temp.mkdir(exist_ok=False)
    try:
        with (temp / s8.AGGREGATE_CSV).open('xb') as fh:
            fh.write(csv_bytes)
        with (temp / s8.SUMMARY_FILE).open('xb') as fh:
            fh.write(summary_bytes)
        config = {
            'schema': SCHEMA,
            'advisor_status': ADVISOR,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'script_sha256': sha(Path(__file__).resolve()),
            'stage8_original_formula_script_sha256': STEP8_SCRIPT_SHA,
            'original_reference_stage8': reference,
            'input_dirs': {name: str(path) for name, path in srcs.items()},
            'replay_stage5_provenance': provenance,
            'posterior_sampling': True,
            'complete_case_reference_r_is_fixed_advisor_value': s8.COMPLETE_CASE_WAIST_BMI_R,
            'no_convergence_claim': True,
            '125_split_draw_observations_are_not_independent': True,
            'participant_level_data_exported': False,
            'models_and_shap_not_loaded_or_recomputed': True,
        }
        save_json(temp / s8.CONFIG_FILE, config)
        names = (s8.AGGREGATE_CSV, s8.SUMMARY_FILE, s8.CONFIG_FILE)
        save_json(temp / s8.HASH_FILE, {name: sha(temp / name) for name in names})
        comparison = {
            'schema': SCHEMA,
            'advisor_status': ADVISOR,
            'report_csv_byte_identical_to_reference': True,
            'summary_json_byte_identical_to_reference': True,
            'max_abs_scientific_difference': 0,
            'n_reference_tables': 3,
            'n_reference_report_rows': 16,
            'n_replay_source_diagnostic_records': 125,
            'stage8_report_csv_sha256': STEP8_CSV_SHA,
            'stage8_summary_sha256': STEP8_SUMMARY_SHA,
            'all_replay_fitted_models_byte_identical': provenance['byte_identical_fitted_models'] == 375,
            'scientific_diagnostic_cells_compared_exactly': provenance['scientific_cells_compared_exactly'],
            'runtime_fields_excluded_from_scientific_match': ['imputer_seconds'],
            'reference_stage8_manifest_sha256': reference['manifest_sha256'],
            'no_new_imputations_or_models_or_shap': True,
        }
        save_json(temp / 'stage11_step8_replay_comparison.json', comparison)
        save_json(temp / 'stage11_step8_replay_sha256.json', {
            name: sha(temp / name) for name in
            (s8.HASH_FILE, 'stage11_step8_replay_comparison.json')
        })
        check_manifest(temp, s8.HASH_FILE, set(names))
        check_manifest(temp, 'stage11_step8_replay_sha256.json',
                       {s8.HASH_FILE, 'stage11_step8_replay_comparison.json'})
        require(sha(temp / s8.AGGREGATE_CSV) == STEP8_CSV_SHA and
                sha(temp / s8.SUMMARY_FILE) == STEP8_SUMMARY_SHA,
                'post-write scientific output SHA256 differs from original Stage 8')
        require(not destination.exists(), f'output directory occupied during run: {destination}')
        temp.rename(destination)
    except Exception:
        # On error, leave the new temporary directory for inspection; never touch inputs.
        raise
    announce('STEP 11 STEP8 REPLAY PASS | 125 MICE records; 3 tables / 16 rows; byte-identical scientific outputs; SHA256 PASS')
    announce(f'STEP 11 STEP8 REPLAY OUTPUT | {destination}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'run'), required=True)
    parser.add_argument('--original-stage5-dir', type=Path, required=True)
    parser.add_argument('--replay-stage5-dir', type=Path, required=True)
    parser.add_argument('--stage8-ref-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    require((args.mode == 'preflight' and args.out is None) or
            (args.mode == 'run' and args.out is not None),
            'preflight forbids --out; run requires a NEW --out directory')
    root = Path(__file__).resolve().parent
    original = source_directory(root, args.original_stage5_dir, 'original Stage 5')
    replay = source_directory(root, args.replay_stage5_dir, 'replayed Stage 5')
    reference = source_directory(root, args.stage8_ref_dir, 'original Stage 8')
    require(len({original, replay, reference}) == 3,
            'all three source/reference directories must be distinct')
    original_script = root / 'stage4_mice_diagnostics_step8.py'
    require(sha(original_script) == STEP8_SCRIPT_SHA,
            'original Stage 8 scientific script SHA256 changed; refuse unreviewed formula')
    announce('STEP 11 STEP8 START | verify both Stage 5 checkpoints; rebuild 3 diagnostic tables from independent replay')
    ref_fingerprints = validate_reference_stage8(reference)
    records, provenance = validate_replay_stage5(root, original, replay, reference)
    report_bytes, summary_bytes, computed_summary = compare_calculated(reference, records)
    if args.mode == 'preflight':
        announce('STEP 11 STEP8 PREFLIGHT PASS | READ-ONLY; 16/16 report rows and summary byte-identical; no files written')
    else:
        emit_output(root, args.out, {'original_stage5': original, 'replay_stage5': replay,
                                    'original_stage8': reference},
                    report_bytes, summary_bytes, computed_summary, provenance,
                    ref_fingerprints)


if __name__ == '__main__':
    main()
