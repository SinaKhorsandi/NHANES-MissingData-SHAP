#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 11: reproduce Stage 10 tables/figures from independent replay.

Preflight is read-only. Run loads ONLY 125 trusted SHA256-verified replay LR
classifiers; it uses the pinned original Stage 10 scientific tables/plotting
functions, with Stage 5/6/7/8/9 REPLAY inputs. The original Stage 10 is a
read-only comparator. No new model fitting, MICE, SHAP, or participant export.

Run at the project root. Results remain PROVISIONAL; advisor approval pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import stage4_mice_reporting_step10 as s10
import stage11_inference_replay_step9 as r9
import stage11_diagnostics_replay_step8 as r8
import stage11_stability_replay_step7 as r7

SCHEMA = 'sina-stage11-step10-reporting-replay-v1'
ADVISOR = 'PENDING; PROVISIONAL'
PIN_S10_SOURCE = '6864bcfc2d84c0c0c788c8613300e041597d77951b1dcee91f9799da3c28445f'
PIN_S10_MANIFEST = 'bc6cc1273499cfc0bde1a7081c1021bf7a8f699a6aef0cf5edee32e361bae846'
PIN_R9_SOURCE = 'a32c0ac997e4f3463affb218ba89a0e0fac612abdbabb62130f03007252f9172'
PIN_R8_SOURCE = 'eb2956d04512ca41dcbda8d17907120d574aeee9dd95e171c9fd44505a71910d'
PIN_R7_SOURCE = '026f88ab8560d31d1613e94e93065fa49bf11cf8ceff43c2e42947098829bc01'
TOL = 1e-10
TABLES = (
 'table1_predictive_performance_4_methods.csv',
 'table2_paired_NB_Holm_4_methods.csv',
 'table3_shap_stability_4_methods.csv',
 'table4_shap_mean_abs_by_feature_4_methods.csv',
 'table5_shap_top5_frequency_4_methods.csv',
 'table6_lr_waist_bmi_coefficients_4_methods.csv',
 'table7_waist_bmi_correlation_context.csv',
 'table8_collinearity_vif_complete_case_reference.csv',
 'table9_collinearity_correlation_complete_case_reference.csv',
)
FIGURES = tuple(f'{stem}_{model}.{ext}' for stem in ('figure1_mean_abs_shap', 'figure2_feature_rank_bump')
                for model in ('LR', 'RF') for ext in ('png', 'pdf'))
ORIGINAL_REPORT_NAMES = set(TABLES) | set(FIGURES) | {'step10_summary.json', 'step10_run_config.json'}


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError('STOP: ' + reason)


def announce(msg: str) -> None:
    print(msg, flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f'missing JSON {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    need(isinstance(obj, dict), f'not a JSON object: {path}')
    return obj


def write_json(path: Path, obj: dict) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def require_manifest(folder: Path, name: str, expected: set[str], pin: str | None = None) -> str:
    return r9.manifest_check(folder, name, expected, pin)


def local_dir(root: Path, value: str, label: str) -> Path:
    return r9.repo_dir(root, Path(value), label)


def reference_check(o10: Path) -> dict:
    manifest_sha = require_manifest(o10, 'step10_aggregate_sha256.json', ORIGINAL_REPORT_NAMES, PIN_S10_MANIFEST)
    config, summary = load_json(o10/'step10_run_config.json'), load_json(o10/'step10_summary.json')
    need(config.get('schema') == s10.SCHEMA and config.get('script_sha256') == PIN_S10_SOURCE
         and config.get('advisor_status') == ADVISOR,
         'original Step 10 run config identity mismatch')
    need(summary.get('schema') == s10.SCHEMA and summary.get('advisor_status') == ADVISOR
         and summary.get('n_mice_lr_models_loaded') == 125
         and summary.get('output_tables') == {n: k for n,k in zip(TABLES,(8,24,12,88,88,8,2,23,2))},
         'original Stage 10 reference summary incomplete')
    announce('STEP 11 STEP10 REFERENCE PASS | 19/19 Stage 10 reference files SHA256; 9 tables and 8 figures pinned')
    return {'manifest_sha256': manifest_sha, 'table_sha256': {n: sha(o10/n) for n in TABLES},
            'figure_sha256': {n: sha(o10/n) for n in FIGURES}}


def validate_all(root: Path, args) -> tuple[dict, dict[str, Path], dict]:
    for name, digest in [('stage4_mice_reporting_step10.py', PIN_S10_SOURCE),
                         ('stage11_inference_replay_step9.py', PIN_R9_SOURCE),
                         ('stage11_diagnostics_replay_step8.py', PIN_R8_SOURCE),
                         ('stage11_stability_replay_step7.py', PIN_R7_SOURCE)]:
        file = root/name
        need(file.is_file() and not file.is_symlink() and sha(file) == digest,
             f'pinned validation or scientific source script changed: {name}')
    need(Path.cwd().resolve() == root, 'run from project root beside all stage scripts')
    labels = ('original_stage5', 'replay_stage5', 'canonical_stage6', 'replay_stage6',
              'original_stage7', 'replay_stage7', 'original_stage8', 'replay_stage8',
              'original_stage9', 'replay_stage9', 'original_stage10')
    dirs = {key: local_dir(root, getattr(args, key), key) for key in labels}
    need(len(set(dirs.values())) == len(dirs), 'all 11 original/replay directories must be distinct')

    announce('STEP 11 STEP10 START | validate original/replay Stage 5-9 and Stage 10 reference; rebuild from replay')
    original_args = SimpleNamespace(
       stage3_dir=args.stage3_dir, legacy_report_dir=args.legacy_report_dir,
       legacy_numeric_dir=args.legacy_numeric_dir,
       stage5_dir=str(dirs['original_stage5']), seedfix_dir=str(dirs['canonical_stage6']),
       stage7_dir=str(dirs['original_stage7']), stage8_dir=str(dirs['original_stage8']),
       stage9_dir=str(dirs['original_stage9']))
    state = s10.validate_data(original_args)  # Original SHA-pinned references and 125 LR model hashes.
    ref = reference_check(dirs['original_stage10'])

    source5 = r9.stage5_lineage(dirs['original_stage5'], dirs['replay_stage5'])
    source7 = r9.stage7_lineage(dirs['original_stage7'], dirs['replay_stage7'], root)
    reference9_rows = r9.reference_step9(dirs['original_stage9'], root)
    result9, diff9, _ = r9.recompute((root/args.stage3_dir)/'cv_metrics_final.csv',
                                    dirs['replay_stage5'], dirs['replay_stage7'], reference9_rows)
    require_manifest(dirs['replay_stage9'], 'step9_aggregate_sha256.json', r9.STAGE9_AGGREGATES)
    require_manifest(dirs['replay_stage9'], 'stage11_step9_replay_sha256.json',
                     {'step9_aggregate_sha256.json', 'stage11_step9_replay_comparison.json'})
    need((dirs['replay_stage9']/'step9_corrected_intervals_4_methods.csv').read_bytes() == result9,
         'replayed Stage 9 report no longer equals independent calculation')
    cmp9 = load_json(dirs['replay_stage9']/'stage11_step9_replay_comparison.json')
    need(cmp9.get('all_48_recomputed_rows_match_canonical') is True and
         diff9.get('scientific_csv_byte_identical_to_reference') is True,
         'Stage 9 replay comparison/identity not PASS')

    # Verify fresh, seed-corrected direct SHAP replay, including the full comparison audit.
    _audit6, src6, _canonical = r7.direct_replay_check(dirs['replay_stage6'], dirs['canonical_stage6'])
    source8ref = r8.validate_reference_stage8(dirs['original_stage8'])
    diagnostics, source5diag = r8.validate_replay_stage5(
        root, dirs['original_stage5'], dirs['replay_stage5'], dirs['original_stage8'])
    csv8, summary8, _ = r8.compare_calculated(dirs['original_stage8'], diagnostics)
    require_manifest(dirs['replay_stage8'], 'step8_aggregate_sha256.json',
                     {r8.s8.AGGREGATE_CSV, r8.s8.CONFIG_FILE, r8.s8.SUMMARY_FILE})
    require_manifest(dirs['replay_stage8'], 'stage11_step8_replay_sha256.json',
                     {'step8_aggregate_sha256.json', 'stage11_step8_replay_comparison.json'})
    need((dirs['replay_stage8']/r8.s8.AGGREGATE_CSV).read_bytes() == csv8
         and (dirs['replay_stage8']/r8.s8.SUMMARY_FILE).read_bytes() == summary8,
         'Stage 8 replay report/summary differs from the recomputed source')
    cmp8 = load_json(dirs['replay_stage8']/'stage11_step8_replay_comparison.json')
    need(cmp8.get('report_csv_byte_identical_to_reference') is True
         and cmp8.get('summary_json_byte_identical_to_reference') is True,
         'Stage 8 replay comparison attestation invalid')

    replay5, replay6, replay7, replay8, replay9 = (dirs[f'replay_stage{i}'] for i in (5,6,7,8,9))
    state['s5cv'] = s10.read_csv(replay5/'mice_cv_metrics_25_splits.csv')
    st3 = s10.read_csv(state['paths']['stage3_importance'])
    mice = s10.read_csv(replay6/'mice_shap_global_importance_25_splits.csv')
    s10.check_importance(mice, ('MICE m=5',))
    col = ['split_id','model','method','feature','mean_abs_shap','rank']
    state['importance'] = pd.concat([st3[col], mice[col]], ignore_index=True)
    need(len(state['importance']) == 2200, 'expected 2200 SHAP feature rows')
    state['agg_importance'] = state['importance'].groupby(['model','method','feature'], as_index=False).agg(
        mean_abs_shap=('mean_abs_shap','mean'), sd_abs_shap=('mean_abs_shap','std'), n_splits=('split_id','nunique'))
    state['agg_top5'] = state['importance'].assign(in_top5=lambda x: x['rank']<=5).groupby(
        ['model','method','feature'], as_index=False).agg(n_splits_in_top5=('in_top5','sum'))
    need(len(state['agg_importance']) == len(state['agg_top5']) == 88, 'feature grouping incomplete')
    state['stability'] = s10.read_csv(replay7/'shap_stability_metrics_4_methods_25_splits.csv')
    state['inference'] = s10.read_csv(replay9/'step9_corrected_intervals_4_methods.csv')
    state['step8_summary'] = load_json(replay8/'step8_summary.json')
    state['paths'].update({
        'stage5_cv':replay5/'mice_cv_metrics_25_splits.csv',
        'stage6_importance':replay6/'mice_shap_global_importance_25_splits.csv',
        'stage7_stability':replay7/'shap_stability_metrics_4_methods_25_splits.csv',
        'step9_results':replay9/'step9_corrected_intervals_4_methods.csv',
    })
    state['model_paths'] = {
        (split, draw): replay5/f'split_{split:03d}'/'models'/f'LR_draw{draw:02d}_classifier.joblib'
        for split in range(1,26) for draw in range(1,6)}
    need(len(state['model_paths']) == 125, '125 replay LR paths required')
    for num in (5,6,7,8,9):
        state['dirs'][f'stage{num}'] = dirs[f'replay_stage{num}']
    announce('STEP 11 STEP10 REPLAY SOURCE PASS | 375 models / 900 checkpoints; '
             '250 fresh SHAP; 300 stability rows; 125 diagnostics; 48 inference rows; 125 replay LR models SHA256')
    return state, dirs, {'reference': ref, 'stage5': source5, 'stage6': src6,
                         'stage7': source7, 'stage8': source5diag,
                         'reference_stage8': source8ref, 'stage9': diff9}


def table_compare(new_file: Path, ref_file: Path) -> dict:
    new, old = pd.read_csv(new_file, dtype=str, keep_default_na=False), pd.read_csv(ref_file, dtype=str, keep_default_na=False)
    need(list(new.columns) == list(old.columns) and new.shape == old.shape,
         f'table schema/shape mismatch: {new_file.name}')
    maxdelta, differing, numerically_close = 0.0, 0, 0
    for i in range(len(new)):
        for col in new.columns:
            x,y = new.at[i,col], old.at[i,col]
            if x == y:
                continue
            differing += 1
            try:
                a,b = float(x), float(y)
            except (ValueError, TypeError):
                raise RuntimeError(f'STOP: text differs in {new_file.name} row {i} col {col}: {x!r} != {y!r}')
            delta = abs(a-b)
            need(math.isfinite(delta) and delta <= TOL,
                 f'scientific number differs in {new_file.name}, row {i}, col {col}: {delta}')
            maxdelta = max(maxdelta, delta)
            numerically_close += 1
    return {'rows': len(new), 'sha256_identical': sha(new_file) == sha(ref_file),
            'cells_text_not_identical': differing, 'scientific_numeric_tolerance': TOL,
            'numeric_cells_within_tolerance': numerically_close, 'max_numeric_difference': maxdelta}


def verify_figures(tmp: Path) -> dict:
    from PIL import Image
    results = {}
    for name in FIGURES:
        p = tmp/name
        need(p.is_file() and not p.is_symlink(), f'missing figure {name}')
        need(p.stat().st_size > 3000, f'figure unexpectedly small: {name}')
        if name.endswith('.png'):
            with Image.open(p) as image:
                image.verify()
            with Image.open(p) as image:
                dpi = image.info.get('dpi', (0,0))
                need(299 <= dpi[0] <= 301 and 299 <= dpi[1] <= 301
                     and image.width >= 2000 and image.height >= 1000,
                     f'bad figure dimensions/DPI: {name}')
                results[name] = {'size_bytes':p.stat().st_size, 'pixels':[image.width,image.height],
                                 'dpi':[float(dpi[0]),float(dpi[1])]}
        else:
            data = p.read_bytes()
            need(data.startswith(b'%PDF-') and b'%%EOF' in data[-2048:], f'invalid PDF header/trailer: {name}')
            results[name] = {'size_bytes':len(data), 'pdf_header_and_trailer':True}
    return results


def emit(root: Path, arg: str, state: dict, dirs: dict[str, Path], audit: dict) -> None:
    parent = (root/'output/stage4_mice/local_rechecks').resolve()
    raw = Path(arg).expanduser()
    output = (raw if raw.is_absolute() else root/raw).resolve()
    need(output.parent == parent and output.name.startswith('stage11_replay_step10_')
         and not output.exists() and not output.is_symlink(),
         'output must be a NEW direct stage11_replay_step10_* directory in local_rechecks')
    tmp = parent/f'.work_{output.name}_{os.getpid()}'
    need(not tmp.exists() and not tmp.is_symlink(), 'temporary directory already exists')
    tmp.mkdir(exist_ok=False)
    try:
        announce('STEP 11 STEP10 MODELS | loading 125 SHA256-verified INDEPENDENT replay LR models; never refit')
        coefs = s10.summarize_lr_mice_models(state)
        tables = s10.tables_from_data(state, coefs)
        need(set(tables) == set(TABLES), 'wrong 9 scientific report tables')
        for name, data in tables.items():
            file = tmp/name
            if name.startswith('table8_') or name.startswith('table9_'):
                source = state['paths']['legacy_vif' if name.startswith('table8_') else 'legacy_corr']
                shutil.copyfile(source, file)  # Historical complete-case references, by design.
            else:
                data.to_csv(file, index=False, float_format='%.17g')
        comparisons = {name: table_compare(tmp/name, dirs['original_stage10']/name) for name in TABLES}
        need(sum(v['rows'] for v in comparisons.values()) == 8+24+12+88+88+8+2+23+2,
             'unexpected table row counts')
        announce('STEP 11 STEP10 TABLES PASS | 9/9 tables scientifically match reference; replay 125 LR coefficient pooling')
        s10.build_figures(tmp, state['importance'])
        figs = verify_figures(tmp)
        figure_comparisons = {name:sha(tmp/name)==sha(dirs['original_stage10']/name) for name in FIGURES}
        report = {
            'schema':SCHEMA, 'advisor_status':ADVISOR,
            'tables_matched':9, 'table_rows_total':sum(v['rows'] for v in comparisons.values()),
            'original_table_byte_identical_count':sum(v['sha256_identical'] for v in comparisons.values()),
            'tables':comparisons, 'figures':figs,
            'figure_byte_identical_to_reference':figure_comparisons,
            'figure_hash_mismatch_not_necessarily_scientific_difference':True,
            'mice_lr_models_loaded_from_independent_stage5':125,
            'no_new_model_training_imputation_or_shap':True,
            'historical_collinearity_references_preserved':True,
            'shap_scale_warning':'LR log odds; RF positive-class probability; never compare magnitudes across model types',
            'uncertainty_warning':'Figure error bars = SD over 25 dependent splits, not CI; Holm for AUC/Brier only',
            'provisional_advisor_approval_pending':True,
        }
        summary = {
            'schema':s10.SCHEMA, 'replay_schema':SCHEMA, 'advisor_status':ADVISOR,
            'n_methods':4, 'n_models':2, 'n_splits':25, 'n_original_features':11,
            'n_mice_lr_models_loaded':125,
            'coefficient_rule':'average five fitted LR draw coefficients within split, mean and sample SD of 25 split averages; fitted StandardScaler units',
            'output_tables':{name: int(len(data)) for name,data in tables.items()},
            'figure_png_300dpi':{k:v for k,v in figs.items() if k.endswith('.png')},
            'data_privacy':'aggregate outputs only, no participant-level SHAP or model serialization',
            'original_stage10_is_read_only_comparator':True,
            'data_source':'independent Stage 5/6/7/8/9 replay with original Stage 3 historical references',
            'source_sha256':{key:sha(p) for key,p in state['paths'].items()},
            'created_utc':datetime.now(timezone.utc).isoformat(),
        }
        write_json(tmp/'step10_summary.json', summary)
        config = {
            'schema':SCHEMA, 'advisor_status':ADVISOR,
            'script_sha256':sha(Path(__file__).resolve()),
            'stage10_original_formula_script_sha256':PIN_S10_SOURCE,
            'original_stage10_manifest_sha256':PIN_S10_MANIFEST,
            'source_directories':{key:str(value) for key,value in dirs.items()},
            'stage11_replay_provenance':audit,
            'never_used_reference_stage10_for_table_or_figure_generation':True,
            'historical_complete_case_reference_tables_copied_intentionally':True,
            'no_training_imputation_or_shap':True,
        }
        write_json(tmp/'step10_run_config.json', config)
        expected = ORIGINAL_REPORT_NAMES
        need({p.name for p in tmp.iterdir()} == expected, 'expected 19 report files before manifest')
        write_json(tmp/'step10_aggregate_sha256.json', {n:sha(tmp/n) for n in sorted(expected)})
        comparison = {'schema':SCHEMA, 'advisor_status':ADVISOR, 'original_stage10_manifest_sha256':PIN_S10_MANIFEST,
                      'nine_scientific_tables_match':True, 'figure_inputs_replay_verified':True,
                      'table_report':comparisons, 'figure_hash_matches':figure_comparisons,
                      'replay_stage5_6_7_8_9_used':True,
                      'reference_outputs_not_used_for_calculation_except_historical_complete_case_tables':True}
        write_json(tmp/'stage11_step10_replay_comparison.json', comparison)
        write_json(tmp/'stage11_step10_replay_sha256.json', {
            n:sha(tmp/n) for n in ('step10_aggregate_sha256.json','stage11_step10_replay_comparison.json')})
        require_manifest(tmp, 'step10_aggregate_sha256.json', expected)
        require_manifest(tmp, 'stage11_step10_replay_sha256.json',
                         {'step10_aggregate_sha256.json','stage11_step10_replay_comparison.json'})
        need(not output.exists(), 'output directory occupied during generation')
        tmp.rename(output)
    except BaseException:
        announce(f'STEP 11 STEP10 ERROR | unpromoted temporary dir retained: {tmp}')
        raise
    announce('STEP 11 STEP10 REPLAY PASS | 9/9 tables + 4 PNG + 4 PDF from independent replay; 125 LR models; SHA256 PASS')
    announce(f'STEP 11 STEP10 REPLAY OUTPUT | {output}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight','run'), required=True)
    p.add_argument('--stage3-dir', default='output/stage3_final')
    p.add_argument('--legacy-report-dir', default='output/stage3_reporting_v1')
    p.add_argument('--legacy-numeric-dir', default='advisor_remaining_csv')
    for name in ('original_stage5','replay_stage5','canonical_stage6','replay_stage6',
                 'original_stage7','replay_stage7','original_stage8','replay_stage8',
                 'original_stage9','replay_stage9','original_stage10'):
        p.add_argument('--'+name.replace('_','-')+'-dir', dest=name, required=True)
    p.add_argument('--out', help='new stage11_replay_step10_* directory')
    args = p.parse_args()
    need((args.mode=='preflight' and args.out is None) or
         (args.mode=='run' and args.out is not None),
         '--mode preflight forbids --out; --mode run requires --out')
    root = Path(__file__).resolve().parent
    state, dirs, audit = validate_all(root, args)
    if args.mode == 'preflight':
        announce('STEP 11 STEP10 PREFLIGHT PASS | READ-ONLY; Stage 5-9 replay lineage, 9 tables/8 figures reference; no models loaded, no files written')
    else:
        emit(root, args.out, state, dirs, audit)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'STEP 11 STEP10 FAIL | {err}', file=sys.stderr, flush=True)
        sys.exit(1)
