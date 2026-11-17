#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 advisor step 4: *full* one-split MICE + SHAP timing gate.

No Stage 3 files are modified and no individual-level rows or SHAP values
are written to disk. Outputs are aggregate timing, validation, tuning,
additivity, and feature-importance summaries in a new Stage 4 subdirectory.

Use in the Stage 3 project root, branch stage3-revision:
    python -u stage4_mice_benchmark_split1.py --mode preflight
    python -u stage4_mice_benchmark_split1.py --mode benchmark-split1

IMPORTANT: The benchmark includes five imputation draws, two separately
trained classifiers for each draw, 3-fold inner tuning of each classifier,
and per-draw SHAP on ALL complete-case split-1 test participants. It is
not a reduced-sample shortcut. Never launch all 25 splits from this script.

Advisor protocol approval is still pending. Benchmark results are provisional.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import shap

import stage3_cv as s3
import stage3_shap as s3sh
import stage4_mice_cv as s4

MAX_HOURS = 12.0
SPLIT_ID = 1
N_TOTAL_SPLITS = 25


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'{stamp} | {message}', flush=True)


def preflight(inp: s4.Inputs) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Check the exact benchmark population, config and original SHAP background."""
    assert s4.M == 5 and s4.MICE_MAX_ITER == 10 and s4.SEED == 2026
    assert s3sh.BACKGROUND_N == 100
    assert s3sh.PERMUTATION_CYCLES == 5
    assert s3sh.ADD_TOLERANCE == 1e-8
    assert list(s3.PREDICTORS) == list(s3sh.PREDICTORS)
    assert list(s3.MODELS) == ['LR', 'RF']
    assert list(s3.NUMERIC) == list(s3sh.NUMERIC)
    assert list(s3.CATEGORICAL) == list(s3sh.CATEGORICAL)

    X_tr, y_tr, complete_tr, X_te, y_te = s4.get_split(inp, SPLIT_ID)
    assert len(X_tr) == 4078 and int(complete_tr.sum()) == 3222
    assert len(X_te) == 809 and not X_te.isna().any().any()

    split_manifest = inp.manifest.loc[inp.manifest['split_id'].eq(SPLIT_ID)]
    role = split_manifest.set_index('SEQN').reindex(inp.df['SEQN'])['role']
    training_complete = inp.df.loc[role.eq('train').to_numpy() & inp.complete].copy()
    background = training_complete.sample(
        n=s3sh.BACKGROUND_N,
        random_state=s3sh.BASE_BACKGROUND_SEED + SPLIT_ID,
    ).sort_values('SEQN').reset_index(drop=True)
    assert len(background) == 100
    assert background[s3.PREDICTORS].notna().all().all()
    assert not (set(background['SEQN']) & set(
        inp.df.loc[role.eq('test').to_numpy(), 'SEQN']
    ))

    log(f'PASS | branch=stage3-revision | split=1/{N_TOTAL_SPLITS}')
    log(f'PASS | train={len(X_tr)} complete_train={int(complete_tr.sum())} '
        f'common_test={len(X_te)} background={len(background)}')
    log(f'PASS | MICE draws={s4.M} max_iter={s4.MICE_MAX_ITER} '
        f'RF_SHAP_cycles={s3sh.PERMUTATION_CYCLES} additivity_tol={s3sh.ADD_TOLERANCE:g}')
    log(f'PASS | same original manifest and input SHA256; shap_version={shap.__version__}')
    return X_tr, y_tr, complete_tr, X_te, y_te, background[s3.PREDICTORS].copy()


def explain_one(pipeline: Any, model: str, bg_raw: pd.DataFrame, eval_raw: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Stage-3-equivalent SHAP, returning signed contributions on 11 raw features.

    All 809 split-1 common-test people are explained. No per-person output
    is saved to disk. For LR, additive outputs are *log-odds*; for RF,
    additive outputs are *positive-class probabilities*.
    """
    pp = pipeline.named_steps['preprocess']
    clf = pipeline.named_steps['classifier']
    X_bg = s3sh.dense_float(pp.transform(bg_raw))
    X_eval = s3sh.dense_float(pp.transform(eval_raw))
    names = list(pp.get_feature_names_out())
    groups = s3sh.map_original_features(names)
    assert X_eval.shape[1] == X_bg.shape[1] == len(names)
    masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))

    if model == 'LR':
        explainer = shap.LinearExplainer(clf, masker)
        expl = explainer(X_eval)
        values, base = s3sh.normalize_lr_shap(expl.values, expl.base_values, len(X_eval))
        output = np.asarray(clf.decision_function(X_eval), dtype=float).reshape(-1)
        max_evals = 0
    elif model == 'RF':
        if list(clf.classes_) != [0, 1]:
            raise ValueError('STOP: RF class order is not [0, 1]')
        # Same single-threaded inference as original stage3_shap.py.
        if 'n_jobs' in clf.get_params(deep=False):
            clf.set_params(n_jobs=1)

        def predict_positive(x: Any) -> np.ndarray:
            return clf.predict_proba(s3sh.dense_float(x))[:, 1]

        explainer = shap.PermutationExplainer(
            predict_positive,
            masker,
            feature_names=names,
            seed=s3sh.BASE_PERMUTATION_SEED + SPLIT_ID,
        )
        max_evals = 2 * X_eval.shape[1] * s3sh.PERMUTATION_CYCLES + 1
        # Keep the original 809 test rows, original 100 background rows,
        # original five cycles; progress display is useful for long RF runs.
        expl = explainer(X_eval, max_evals=max_evals, batch_size=256, silent=False)
        values, base = s3sh.normalize_scalar_shap(expl.values, expl.base_values, len(X_eval))
        output = clf.predict_proba(X_eval)[:, 1]
    else:
        raise ValueError(f'Unknown model: {model!r}')

    if values.shape != X_eval.shape:
        raise RuntimeError('STOP: SHAP matrix shape mismatch')
    if not np.isfinite(values).all() or not np.isfinite(base).all():
        raise RuntimeError('STOP: SHAP matrix/base contains a nonfinite value')
    errors = np.abs(base + values.sum(axis=1) - output)
    max_error = float(np.max(errors))
    if not np.isfinite(max_error) or max_error >= s3sh.ADD_TOLERANCE:
        raise RuntimeError(f'STOP: {model} SHAP additivity error {max_error} >= 1e-8')
    grouped = s3sh.aggregate_to_original(values, groups)
    assert grouped.shape == (len(eval_raw), 11)
    assert np.isfinite(grouped).all()
    return grouped, base, output, max_error, int(max_evals)


def run_one_split(inp: s4.Inputs, output_dir: Path) -> None:
    t0 = time.perf_counter()
    X_tr, y_tr, complete_tr, X_te, y_te, X_bg_raw = preflight(inp)
    assert np.array_equal(complete_tr, (~X_tr.isna().any(axis=1)).to_numpy())

    diag_imputation: list[dict[str, Any]] = []
    diag_shap: list[dict[str, Any]] = []
    diag_metrics: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []
    log('START | split1 FULL benchmark: 5 MICE draws, LR/RF and all SHAP evaluations')

    for j in range(1, s4.M + 1):
        seed = s4.SEED + 100 * SPLIT_ID + j
        t = time.perf_counter()
        bundle, filled = s4.fit_mice_draw(X_tr, seed=seed)
        assert not filled.isna().any().any()
        n_iter = int(bundle['mice'].n_iter_)
        elapsed = time.perf_counter() - t
        draws.append(filled)
        diag_imputation.append({
            'split_id': SPLIT_ID, 'imputation_id': j,
            'seed': seed, 'n_iter': n_iter, 'max_iter': s4.MICE_MAX_ITER,
            'seconds': elapsed,
        })
        log(f'MICE | draw={j}/5 | seed={seed} n_iter={n_iter} elapsed={elapsed:.2f}s')

    for model in ('LR', 'RF'):
        tune_start = time.perf_counter()
        params, candidates = s4.tune_mice_first_draw(
            X_tr, y_tr, complete_tr, model, SPLIT_ID,
        )
        tune_seconds = time.perf_counter() - tune_start
        candidates.to_csv(output_dir / f'split001_{model}_inner_tuning.csv', index=False)
        log(f'TUNING | {model} | completed in {tune_seconds:.1f}s | params={params}')

        probabilities: list[np.ndarray] = []
        margins: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        grouped_values: list[np.ndarray] = []
        for j, X_filled in enumerate(draws, start=1):
            tfit = time.perf_counter()
            fitted = s3.build_pipeline('CCA', model, params, s4.SEED + SPLIT_ID)
            fitted.fit(X_filled, y_tr)
            fit_seconds = time.perf_counter() - tfit
            prob = np.asarray(fitted.predict_proba(X_te)[:, 1], dtype=float)
            assert prob.shape == (len(X_te),) and np.isfinite(prob).all()
            probabilities.append(prob)
            log(f'SHAP START | model={model} | draw={j}/5 | test={len(X_te)} '
                f'background={len(X_bg_raw)}', )
            tshap = time.perf_counter()
            grouped, base, output, error, max_evals = explain_one(
                fitted, model, X_bg_raw, X_te,
            )
            shap_seconds = time.perf_counter() - tshap
            grouped_values.append(grouped)
            base_values.append(base)
            margins.append(output)
            diag_shap.append({
                'split_id': SPLIT_ID, 'model': model, 'imputation_id': j,
                'fit_seconds': fit_seconds, 'shap_seconds': shap_seconds,
                'n_train': len(X_tr), 'n_test_common': len(X_te),
                'background_n': len(X_bg_raw), 'n_imputations': s4.M,
                'encoded_feature_count': int(fitted.named_steps['preprocess'].transform(X_te.iloc[:1]).shape[1]),
                'permutation_max_evals': max_evals,
                'max_abs_additivity_error': error,
                'output_scale': ('log_odds' if model == 'LR' else 'positive_class_probability'),
                'status': 'PASS',
            })
            log(f'SHAP PASS | model={model} | draw={j}/5 | '
                f'fit={fit_seconds:.1f}s shap={shap_seconds:.1f}s '
                f'max_error={error:.3g}')
            del fitted

        p_bar = np.mean(np.stack(probabilities, axis=0), axis=0)
        grouped_bar = np.mean(np.stack(grouped_values, axis=0), axis=0)
        base_bar = np.mean(np.stack(base_values, axis=0), axis=0)
        output_bar = np.mean(np.stack(margins, axis=0), axis=0)
        pooled_err = float(np.max(np.abs(base_bar + grouped_bar.sum(axis=1) - output_bar)))
        if not np.isfinite(pooled_err) or pooled_err >= s3sh.ADD_TOLERANCE:
            raise ValueError(f'STOP: {model} pooled SHAP additivity error {pooled_err}')
        if model == 'RF':
            assert np.allclose(p_bar, output_bar, atol=1e-10, rtol=0)

        auc = float(s3.roc_auc_score(y_te, p_bar))
        brier = float(s3.brier_score_loss(y_te, p_bar))
        if not (np.isfinite(auc) and np.isfinite(brier)):
            raise ValueError('STOP: pooled predictive metrics nonfinite')
        diag_metrics.append({
            'split_id': SPLIT_ID, 'model': model, 'method': s4.MICE_NAME,
            'n_train': len(X_tr), 'n_test_common': len(X_te),
            'n_imputations': s4.M, 'roc_auc': auc, 'brier_score': brier,
            'pooled_max_abs_additivity_error': pooled_err,
            'output_scale': ('log_odds' if model == 'LR' else 'positive_class_probability'),
            'tuning_seconds': tune_seconds,
            'params_json': json.dumps(s3.json_safe(params), sort_keys=True),
        })
        for k, name in enumerate(s3.PREDICTORS):
            global_rows.append({
                'split_id': SPLIT_ID, 'model': model,
                'method': s4.MICE_NAME, 'feature': name,
                'mean_abs_shap': float(np.mean(np.abs(grouped_bar[:, k]))),
                'mean_shap': float(np.mean(grouped_bar[:, k])),
                'shap_scale': ('log_odds' if model == 'LR' else 'positive_class_probability'),
            })
        log(f'POOLED PASS | {model} | auc={auc:.6f} brier={brier:.6f} '
            f'additivity_error={pooled_err:.3g}')

    total = time.perf_counter() - t0
    estimated = N_TOTAL_SPLITS * total / 3600.0
    gate = 'PASS' if estimated < MAX_HOURS else 'HOLD'
    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
        'shap': shap.__version__,
    }
    pd.DataFrame(diag_imputation).to_csv(output_dir / 'mice_split1_imputer_iterations.csv', index=False)
    pd.DataFrame(diag_shap).to_csv(output_dir / 'mice_split1_shap_additivity_and_runtime.csv', index=False)
    pd.DataFrame(diag_metrics).to_csv(output_dir / 'mice_split1_pooled_performance.csv', index=False)
    pd.DataFrame(global_rows).to_csv(output_dir / 'mice_split1_global_shap.csv', index=False)
    summary = {
        'split_id': SPLIT_ID, 'n_total_splits': N_TOTAL_SPLITS,
        'n_train': len(X_tr), 'n_test_common': len(X_te),
        'background_n': len(X_bg_raw), 'n_imputations': s4.M,
        'n_models': 2, 'n_model_draw_shap_runs': len(diag_shap),
        'split1_elapsed_seconds': total,
        'estimated_25_split_hours_linear': estimated,
        'advisor_maximum_hours': MAX_HOURS,
        'time_gate': gate,
        'approval_status': 'PENDING',
        'versions': versions,
        'runtime_note': 'Linear estimate based on one exact full split; actual run times may vary.',
        'shap_note': 'LR SHAP is averaged in log-odds; mean LR probabilities are separately used for AUC/Brier.',
    }
    (output_dir / 'mice_split1_benchmark_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    assert len(diag_imputation) == 5
    assert len(diag_shap) == 10 and all(r['status'] == 'PASS' for r in diag_shap)
    assert len(diag_metrics) == 2 and len(global_rows) == 22
    log(f'STEP 4 BENCHMARK: {gate} | split1_seconds={total:.1f} '
        f'estimated_25_split_hours={estimated:.3f} | limit_hours={MAX_HOURS}')
    log(f'OUTPUT: {output_dir}')
    if gate != 'PASS':
        log('STOP | benchmark estimate >= 12 hours: do not run the 25 splits')


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'benchmark-split1'], required=True)
    parser.add_argument('--stage3-dir', type=Path, default=root / 'output/stage3_final')
    parser.add_argument('--stage4-dir', type=Path, default=root / 'output/stage4_mice')
    args = parser.parse_args()
    s4.ensure_revision_branch(root)
    inp = s4.read_inputs(root, args.stage3_dir)
    preflight(inp)
    if args.mode == 'preflight':
        log('STEP 4 PREFLIGHT: PASS | no model/SHAP computations performed')
        return
    directory = s4.choose_output(args.stage4_dir, 'step4_timing_split1')
    run_one_split(inp, directory)


if __name__ == '__main__':
    main()
