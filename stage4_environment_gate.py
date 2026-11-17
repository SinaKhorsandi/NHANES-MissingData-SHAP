#!/usr/bin/env python3
"""Stage-4 pre-computation gate: reference versions and recorded Stage-2/3 PASS.

Safe to run on any environment. Reads Stage-3 and Stage-4 aggregate metadata,
never loads persisted models, and does not export participant-level rows.
The original stage3-revision protocol remains advisor-approval pending.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import shap


VERSIONS = {
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scikit_learn': sklearn.__version__,
    'scipy': scipy.__version__,
    'joblib': joblib.__version__,
    'shap': shap.__version__,
}


def inspect(root: Path) -> dict:
    root = root.resolve()
    s3 = root / 'output' / 'stage3_final'
    s4 = root / 'output' / 'stage4_mice'
    cv_meta = json.loads((s3 / 'stage3_cv_metadata.json').read_text(encoding='utf-8'))
    shap_meta = json.loads((s3 / 'shap' / 'stage3_shap_metadata.json').read_text(encoding='utf-8'))
    expected = dict(cv_meta['software_versions'])
    expected['shap'] = shap_meta['software_versions']['shap']
    expected_python = expected.pop('python')
    mismatch = {key: {'expected': value, 'actual': VERSIONS.get(key)}
                for key, value in expected.items() if VERSIONS.get(key) != value}
    actual_python = platform.python_version()
    if actual_python.split('.')[:2] != expected_python.split('.')[:2]:
        mismatch['python_major_minor'] = {'expected': expected_python, 'actual': actual_python}

    baseline = s4 / 'step2_baseline_split1' / 'original_six_split1_reproduction.csv'
    leak = s4 / 'step3_leakage_test' / 'stage4_mice_leakage_test_results.csv'
    checks = {}
    if baseline.is_file():
        frame = pd.read_csv(baseline)
        official = pd.read_csv(s3 / 'cv_metrics_final.csv')
        original = official[official['split_id'].eq(1)]
        merged = frame.merge(original[['split_id', 'model', 'method', 'roc_auc', 'brier_score']],
                             on=['split_id', 'model', 'method'], validate='one_to_one')
        same_scores = (merged['roc_auc_reproduced'].map(lambda x: f'{x:.6f}') ==
                       merged['roc_auc'].map(lambda x: f'{x:.6f}')).all() and (
                       merged['brier_reproduced'].map(lambda x: f'{x:.6f}') ==
                       merged['brier_score'].map(lambda x: f'{x:.6f}')).all()
        checks['recorded_baseline_6_of_6_pass'] = bool(
            len(frame) == len(original) == len(merged) == 6 and
            frame[['auc_match_6dp','brier_match_6dp']].to_numpy().all() and same_scores)
    else:
        checks['recorded_baseline_6_of_6_pass'] = False
    if leak.is_file():
        frame = pd.read_csv(leak)
        checks['recorded_leakage_3_of_3_pass'] = bool(len(frame) == 3 and
            frame['status'].eq('PASS').all() and
            set(frame['test']) == {'training_only_fit','complete_common_test',
                                   'test_perturbation_invariance'})
    else:
        checks['recorded_leakage_3_of_3_pass'] = False
    checks['version_match'] = not mismatch
    return {
        'status': 'PASS' if all(checks.values()) else 'HOLD',
        'advisor_approval_status': 'PENDING',
        'interpretation': 'Technical gate only; not an advisor approval',
        'python_version_actual': actual_python,
        'python_version_original': expected_python,
        'package_versions_actual': VERSIONS,
        'package_versions_original': expected,
        'version_mismatches': mismatch,
        'checks': checks,
        'source_dataset_sha256': cv_meta['input_dataset_sha256'],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--require-pass', action='store_true')
    p.add_argument('--write-aggregate', type=Path, default=None)
    args = p.parse_args()
    result = inspect(args.root)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    print(payload)
    if args.write_aggregate:
        target = args.write_aggregate
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f'Existing audit must not be overwritten: {target}')
        target.write_text(payload, encoding='utf-8')
    if args.require_pass and result['status'] != 'PASS':
        raise SystemExit('STOP: Stage-4 readiness gate HOLD (see details above)')


if __name__ == '__main__':
    main()
