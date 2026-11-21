#!/usr/bin/env python3
"""SINA step 11: read-only local comparison of Stage 5 private replay checkpoint files.

Never loads joblib files or exports participant-level data. Verifies all 25 original
and replay split checkpoint SHA256 manifests before comparing model file bytes.
Differences in joblib file bytes are reported, not silently accepted as scientific
or predictive differences. No output files are written.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError('STOP | ' + message)


def read_json(path):
    require(path.is_file(), f'missing {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def verify_split(run_dir, split):
    folder = (run_dir / f'split_{split:03d}').resolve()
    require(folder.is_dir() and folder.is_relative_to(run_dir), f'missing split {split}: {folder}')
    marker = read_json(folder / 'split_complete.json')
    require(marker.get('schema') == 'sina-stage4-step5-v1', f'wrong schema split {split}')
    require(int(marker.get('split_id', -1)) == split, f'wrong ID split {split}')
    entries = marker.get('sha256')
    require(isinstance(entries, dict) and len(entries) == 18, f'not 18 entries split {split}')
    required_tables = {'mice_performance.csv', 'mice_imputer_diagnostics.csv', 'mice_inner_tuning.csv'}
    model_names = {f'models/MICE_draw{i:02d}_imputer_bundle.joblib' for i in range(1, 6)}
    model_names.update(f'models/{model}_draw{i:02d}_classifier.joblib'
                       for model in ('LR', 'RF') for i in range(1, 6))
    require(set(entries) == required_tables | model_names,
            f'unexpected file names split {split}')
    for relative, checksum in entries.items():
        target = (folder / relative).resolve()
        require(target.is_relative_to(folder) and target.is_file(),
                f'missing/outside split {split}: {relative}')
        require(sha256_file(target) == checksum,
                f'split {split} corrupted file {relative}')
    return entries, model_names


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--original-dir', required=True, type=Path)
    cli.add_argument('--replay-dir', required=True, type=Path)
    args = cli.parse_args()
    original = args.original_dir.resolve()
    replay = args.replay_dir.resolve()
    require(original.is_dir() and replay.is_dir() and original != replay,
            'two distinct run directories required')
    old_config = read_json(original / 'run_config.json')
    new_config = read_json(replay / 'run_config.json')
    require(old_config.get('schema') == new_config.get('schema') == 'sina-stage4-step5-v1',
            'run schema differs')
    require(old_config.get('identity') == new_config.get('identity'),
            'input/dataset/code/benchmark/environment provenance differs')
    print('REPLAY IDENTITY PASS | recorded input/dataset/code/benchmark/environment identical', flush=True)
    total_models, identical_models, model_differences = 0, 0, []
    split_table_same = 0
    for split in range(1, 26):
        old_entries, old_models = verify_split(original, split)
        new_entries, new_models = verify_split(replay, split)
        require(old_models == new_models, f'model names split {split}')
        for relative in sorted(old_models):
            total_models += 1
            if old_entries[relative] == new_entries[relative]:
                identical_models += 1
            else:
                model_differences.append((split, relative))
        for table in ('mice_performance.csv', 'mice_imputer_diagnostics.csv', 'mice_inner_tuning.csv'):
            split_table_same += (old_entries[table] == new_entries[table])
        if split % 5 == 0:
            print(f'REPLAY HASH PROGRESS | {split}/25 pairs of 18-file checkpoint manifests verified', flush=True)
    require(total_models == 375, 'expected 375 private model artifacts')
    print('CHECKPOINT INTEGRITY PASS | 25+25 split manifests, 900 referenced files verified', flush=True)
    print(f'MODEL SHA256 COMPARISON | byte-identical {identical_models}/{total_models}; '
          f'different {len(model_differences)}/{total_models}', flush=True)
    print(f'SPLIT CSV SHA256 | byte-identical {split_table_same}/75 paired tables; '
          'runtime fields normally change', flush=True)
    if model_differences:
        print('MODEL BYTES DIFFER | names only; first 15 discrepancies:', flush=True)
        for split, name in model_differences[:15]:
            print(f'  split {split:02d} | {name}', flush=True)
        print('REPLAY ARTIFACT CHECK | INTEGRITY PASS, BYTE-IDENTITY NOT COMPLETE; '
              'do not infer a change in predictions without further checking', flush=True)
    else:
        print('REPLAY MODEL BYTE IDENTITY PASS | all 375 model artifacts exactly match', flush=True)
    print('READ-ONLY PASS | no files written or joblib models loaded', flush=True)


if __name__ == '__main__':
    main()
