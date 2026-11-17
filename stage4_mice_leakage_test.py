#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4, advisor step 3: one-draw split-1 MICE leakage regression test.

This test is intentionally separate from the long-running MICE CV/SHAP pipeline.
It writes only three aggregate PASS rows; no participant-level rows or models.
Run from the project root, on branch stage3-revision:
    python -u stage4_mice_leakage_test.py
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import stage3_cv as stage3
import stage4_mice_cv as stage4


def check_same_fitted_imputer(before: IterativeImputer, after: IterativeImputer) -> int:
    """Check every fitted BayesianRidge's coefficients and intercept, exactly."""
    assert before.n_iter_ == after.n_iter_
    assert before.max_iter == after.max_iter == stage4.MICE_MAX_ITER
    assert before.sample_posterior is after.sample_posterior is True
    assert len(before.imputation_sequence_) == len(after.imputation_sequence_) > 0
    assert np.array_equal(before.initial_imputer_.statistics_,
                          after.initial_imputer_.statistics_)
    for a, b in zip(before.imputation_sequence_, after.imputation_sequence_):
        assert a.feat_idx == b.feat_idx
        assert np.array_equal(a.neighbor_feat_idx, b.neighbor_feat_idx)
        np.testing.assert_array_equal(a.estimator.coef_, b.estimator.coef_)
        np.testing.assert_array_equal(np.asarray(a.estimator.intercept_),
                                      np.asarray(b.estimator.intercept_))
    return len(before.imputation_sequence_)


def main() -> None:
    root = Path(__file__).resolve().parent
    stage4.ensure_revision_branch(root)
    inp = stage4.read_inputs(root, root / 'output' / 'stage3_final')
    X_tr, _y_tr, _complete_tr, X_te, _y_te = stage4.get_split(inp, 1)

    assert len(X_tr) == 4078 and len(X_te) == 809
    assert not X_te.isna().any().any(), 'FAIL: common complete-case test contains NaN'
    assert not np.shares_memory(X_tr.to_numpy(copy=False), X_te.to_numpy(copy=False))
    print(f'PASS | Common complete-case test contains no NaNs | n_test={len(X_te)}', flush=True)

    # Spy on every fit_transform in the original MICE implementation. An
    # augmented training matrix must have EXACTLY the training-set row count
    # and raw numeric contents, not just an approximately matching count.
    seen = {'numeric_mice': 0, 'categorical_mode': 0, 'aux_onehot': 0}
    real_mice = IterativeImputer.fit_transform
    real_mode = SimpleImputer.fit_transform
    real_ohe = OneHotEncoder.fit_transform
    raw_train_num = X_tr[stage3.NUMERIC].to_numpy(dtype=float)
    raw_train_cat = X_tr[stage3.CATEGORICAL].to_numpy()

    def guarded_mice(self, X, *args, **kwargs):
        array = np.asarray(X, dtype=float)
        assert array.ndim == 2 and array.shape[0] == len(X_tr), (
            'FAIL: MICE received a different row count from training set')
        np.testing.assert_allclose(array[:, :len(stage3.NUMERIC)],
                                   raw_train_num, atol=0, rtol=0, equal_nan=True)
        seen['numeric_mice'] += 1
        return real_mice(self, X, *args, **kwargs)

    def guarded_mode(self, X, *args, **kwargs):
        # IterativeImputer also calls SimpleImputer internally for its
        # initial fill; guard both call sites without conflating them.
        array = np.asarray(X)
        assert len(array) == len(X_tr)
        if self.strategy == 'most_frequent':
            assert array.shape == raw_train_cat.shape
            np.testing.assert_array_equal(pd.isna(array), pd.isna(raw_train_cat))
            seen['categorical_mode'] += 1
        else:
            # Internal initial fill of the MICE augmented feature matrix.
            assert array.shape[1] >= len(stage3.NUMERIC)
            np.testing.assert_allclose(array[:, :len(stage3.NUMERIC)],
                                       raw_train_num, atol=0, rtol=0, equal_nan=True)
        return real_mode(self, X, *args, **kwargs)

    def guarded_onehot(self, X, *args, **kwargs):
        assert len(X) == len(X_tr)
        seen['aux_onehot'] += 1
        return real_ohe(self, X, *args, **kwargs)

    seed = stage4.SEED + 100 * 1 + 1  # split 1, imputation 1: 2127
    start = time.perf_counter()
    with (patch.object(IterativeImputer, 'fit_transform', guarded_mice),
          patch.object(SimpleImputer, 'fit_transform', guarded_mode),
          patch.object(OneHotEncoder, 'fit_transform', guarded_onehot)):
        bundle_before, filled_before = stage4.fit_mice_draw(X_tr.copy(deep=True), seed)
    assert seen == {'numeric_mice': 1, 'categorical_mode': 1, 'aux_onehot': 1}, seen
    print(f'PASS | MICE, categorical imputer and one-hot fit only on training rows | n_train={len(X_tr)}', flush=True)

    # Adversarially change ALL common-test numerical values. Do not send any
    # test samples to the imputer's fit interface. With unchanged training
    # data and fixed random seed, coefficients and imputations must be exact.
    changed_test = X_te.copy(deep=True)
    for name in stage3.NUMERIC:
        changed_test[name] = changed_test[name].astype(float) + 10000.0
    assert not changed_test[stage3.NUMERIC].equals(X_te[stage3.NUMERIC])
    assert not changed_test.isna().any().any()

    # Crucially, the production function being tested accepts X_train alone.
    # A test-set change cannot influence its fit inputs.
    bundle_after, filled_after = stage4.fit_mice_draw(X_tr.copy(deep=True), seed)
    n_fit_steps = check_same_fitted_imputer(bundle_before['mice'], bundle_after['mice'])
    pd.testing.assert_frame_equal(filled_before, filled_after, check_exact=True)
    print(f'PASS | Test-set perturbation leaves all fitted MICE coefficients unchanged | '
          f'n_bayesian_ridge_steps={n_fit_steps}', flush=True)

    # Persist only an aggregate three-row audit, never training/testing rows.
    out_dir = stage4.choose_output(root / 'output' / 'stage4_mice', 'step3_leakage_test')
    audit = pd.DataFrame([
        {'test': 'training_only_fit', 'status': 'PASS',
         'numeric_result': int(seen['numeric_mice']), 'unit': 'verified MICE fit calls'},
        {'test': 'complete_common_test', 'status': 'PASS',
         'numeric_result': int(X_te.isna().sum().sum()), 'unit': 'missing test values'},
        {'test': 'test_perturbation_invariance', 'status': 'PASS',
         'numeric_result': int(n_fit_steps), 'unit': 'identical BayesianRidge fits'},
    ])
    output = out_dir / 'stage4_mice_leakage_test_results.csv'
    audit.to_csv(output, index=False)
    reread = pd.read_csv(output)
    assert len(reread) == 3 and reread['status'].eq('PASS').all()
    print(f'STEP 3 LEAKAGE: PASS | 3/3 checks | seconds={time.perf_counter() - start:.1f}', flush=True)
    print('OUTPUT:', output, flush=True)


if __name__ == '__main__':
    main()
