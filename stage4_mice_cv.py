#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 MICE development, gated by exact Stage 3 split-1 reproduction.

Advisor-approval status: PENDING. Running this script before approval is
provisional work at the researcher's direction, not advisor-approved results.

The source Stage 3 files remain READ ONLY. Outputs go only into new folders
under output/stage4_mice/. No participant-level rows are exported.

Modes (run in this order):
  python stage4_mice_cv.py --mode preflight
  python stage4_mice_cv.py --mode baseline-split1
  python stage4_mice_cv.py --mode mice-training-split1

The MICE training mode does NOT compute SHAP or authorize a 25-split run.
It is only a development step; the separate timed SHAP benchmark is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import stage3_cv as original

M = 5
MICE_MAX_ITER = 10
MICE_NAME = "MICE m=5"
SEED = 2026


@dataclass
class Inputs:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    complete: np.ndarray
    manifest: pd.DataFrame
    reference: pd.DataFrame


def ensure_revision_branch(root: Path) -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != "stage3-revision":
        raise RuntimeError(f"STOP: expected stage3-revision branch, got {branch!r}")


def read_inputs(root: Path, source_dir: Path) -> Inputs:
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    required = [
        data_path,
        source_dir / "split_manifest_final.csv",
        source_dir / "cv_metrics_final.csv",
        source_dir / "stage3_cv_metadata.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("STOP: Stage 3 inputs missing:\n" + "\n".join(missing))

    meta = json.loads((source_dir / "stage3_cv_metadata.json").read_text(encoding="utf-8"))
    expected_hash = meta["input_dataset_sha256"]
    observed_hash = original.sha256_file(data_path)
    if expected_hash != observed_hash:
        raise ValueError("STOP: Stage 1 input SHA256 differs from Stage 3 source")
    if observed_hash != "a017e04e8c9cbbf5d1d5a85e7bbc241629cc8679bdd21b1a3f96cbf0a2338752":
        raise ValueError("STOP: source dataset hash not equal to original verified study input")

    df = pd.read_csv(data_path)
    df, y_array, complete = original.validate_input(df, data_path, root)
    X = df[original.PREDICTORS].copy()
    y = pd.Series(y_array, name=original.TARGET)
    manifest = pd.read_csv(source_dir / "split_manifest_final.csv")
    ref = pd.read_csv(source_dir / "cv_metrics_final.csv")
    assert len(df) == 5098 and int(complete.sum()) == 4031
    assert len(manifest) == 25 * len(df), "STOP: manifest must have all 25 splits"
    assert len(ref) == 25 * 2 * 3, "STOP: Stage3 reference must contain 150 metrics"
    assert not ref.duplicated(["split_id", "model", "method"]).any()

    # Check the supplied manifest for all 25 splits; never silently recreate it.
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    for split_id, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        part = manifest.loc[manifest["split_id"].eq(split_id)]
        assert len(part) == len(df) and not part["SEQN"].duplicated().any(), (
            f"STOP: split {split_id} incomplete or duplicated"
        )
        aligned = part.set_index("SEQN").reindex(df["SEQN"])
        assert not aligned["role"].isna().any(), f"STOP: split {split_id} SEQN mismatch"
        is_test = aligned["role"].eq("test").to_numpy()
        expected_test = np.zeros(len(df), dtype=bool)
        expected_test[test_idx] = True
        assert np.array_equal(is_test, expected_test), (
            f"STOP: split {split_id} differs from original seed=2026 assignment"
        )
        assert np.array_equal(np.flatnonzero(~is_test), train_idx), (
            f"STOP: split {split_id} training order mismatch"
        )
        is_common = aligned["included_in_common_complete_case_test"].to_numpy(dtype=bool)
        assert np.array_equal(is_common, is_test & complete), (
            f"STOP: split {split_id} common-test mask mismatch"
        )
    return Inputs(df=df, X=X, y=y, complete=complete, manifest=manifest, reference=ref)


def get_split(inp: Inputs, split_id: int = 1) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series]:
    part = inp.manifest.loc[inp.manifest["split_id"].eq(split_id)]
    roles = part.set_index("SEQN").reindex(inp.df["SEQN"])["role"].to_numpy()
    train = roles == "train"
    test = (roles == "test") & inp.complete
    X_tr = inp.X.loc[train].reset_index(drop=True)
    y_tr = inp.y.loc[train].reset_index(drop=True)
    complete_tr = inp.complete[train].copy()
    X_te = inp.X.loc[test].reset_index(drop=True)
    y_te = inp.y.loc[test].reset_index(drop=True)
    assert len(X_tr) > 4000 and 780 <= len(X_te) <= 840
    assert not X_te.isna().any().any(), "STOP: test participants not complete"
    assert set(y_te.unique()) == {0, 1}
    return X_tr, y_tr, complete_tr, X_te, y_te


def choose_output(parent: Path, subdir: str) -> Path:
    output = (parent / subdir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"STOP: nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def silent_logger() -> logging.Logger:
    logger = logging.getLogger("stage4-stage3-reproduction")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def baseline_split_one(inp: Inputs, output: Path) -> None:
    X_tr, y_tr, complete_tr, X_te, y_te = get_split(inp, 1)
    records: list[dict[str, Any]] = []
    for model in original.MODELS:
        for method in original.METHODS:
            started = time.perf_counter()
            params, _, _, _ = original.tune_hyperparameters(
                X_outer_train=X_tr,
                y_outer_train=y_tr,
                complete_outer_train=complete_tr,
                method=method,
                model_name=model,
                split_id=1,
                random_state=SEED + 1,
                logger=silent_logger(),
            )
            fit_mask = complete_tr if method == "CCA" else np.ones(len(y_tr), dtype=bool)
            pipeline = original.build_pipeline(method, model, params, SEED + 1)
            pipeline.fit(X_tr.loc[fit_mask], y_tr.loc[fit_mask])
            predicted = pipeline.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, predicted))
            brier = float(brier_score_loss(y_te, predicted))
            ref = inp.reference.loc[
                inp.reference["split_id"].eq(1)
                & inp.reference["model"].eq(model)
                & inp.reference["method"].eq(method)
            ]
            if len(ref) != 1:
                raise RuntimeError(f"STOP: missing reference {model}/{method}")
            row = ref.iloc[0]
            auc_match = f"{auc:.6f}" == f"{float(row['roc_auc']):.6f}"
            brier_match = f"{brier:.6f}" == f"{float(row['brier_score']):.6f}"
            records.append({
                "split_id": 1,
                "model": model,
                "method": method,
                "roc_auc_reference": float(row["roc_auc"]),
                "roc_auc_reproduced": auc,
                "brier_reference": float(row["brier_score"]),
                "brier_reproduced": brier,
                "auc_match_6dp": bool(auc_match),
                "brier_match_6dp": bool(brier_match),
                "n_train": int(fit_mask.sum()),
                "n_test_common": int(len(X_te)),
                "seconds": float(time.perf_counter() - started),
            })
            print(f"CHECK | {model} | {method} | AUC {auc:.6f} | Brier {brier:.6f} | "
                  f"match={auc_match and brier_match}", flush=True)
    result = pd.DataFrame(records)
    target = output / "original_six_split1_reproduction.csv"
    result.to_csv(target, index=False, float_format="%.12g")
    if not result[["auc_match_6dp", "brier_match_6dp"]].to_numpy().all():
        raise RuntimeError(f"STOP: original six did NOT match to six decimals; inspect {target}")
    assert len(result) == 6
    print("STEP 2 BASELINE: PASS | 6/6 methods | AUC and Brier matched to 6 decimal places")
    print("OUTPUT:", target)


def fit_mice_draw(X_train: pd.DataFrame, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit ONLY on current training rows; use full one-hot as auxiliary predictors.

    No test data is passed here. Does not create participant-level output.
    """
    if not X_train.columns.tolist() == original.PREDICTORS:
        raise ValueError("MICE expects 11 original predictors in Stage3 order")
    categories = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    cat_filled = categories.fit_transform(X_train[original.CATEGORICAL])
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    aux = one_hot.fit_transform(cat_filled)
    numeric = X_train[original.NUMERIC].to_numpy(dtype=float)
    augmented = np.column_stack((numeric, aux))
    n_numeric = len(original.NUMERIC)
    assert augmented.shape[1] >= 11

    mice = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=True,
        max_iter=MICE_MAX_ITER,
        random_state=int(seed),
        keep_empty_features=True,
    )
    imputed = mice.fit_transform(augmented)
    if imputed.shape != augmented.shape or not np.isfinite(imputed).all():
        raise ValueError("STOP: MICE output not finite or changed feature count")
    if not np.array_equal(imputed[:, n_numeric:], aux):
        raise ValueError("STOP: MICE unexpectedly modified observed one-hot auxiliaries")
    missing_numeric = np.isnan(numeric)
    if not np.array_equal(imputed[:, :n_numeric][~missing_numeric], numeric[~missing_numeric]):
        raise ValueError("STOP: MICE modified observed numeric data")

    filled = X_train.copy()
    for j, name in enumerate(original.NUMERIC):
        filled[name] = imputed[:, j]
    for j, name in enumerate(original.CATEGORICAL):
        filled[name] = cat_filled[:, j]
    if filled.isna().any().any():
        raise ValueError("STOP: MICE result contains NaN")
    bundle = {
        "categories": categories,
        "one_hot": one_hot,
        "mice": mice,
        "seed": int(seed),
        "numeric_columns": original.NUMERIC,
        "categorical_columns": original.CATEGORICAL,
        "n_iter": int(mice.n_iter_),
    }
    return bundle, filled


def tune_mice_first_draw(
    X_tr: pd.DataFrame, y_tr: pd.Series, complete_tr: np.ndarray, model: str, split_id: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Stage3 inner 3-fold CV using first MICE draw fitted inside each inner train fold.

    The three inner validation cohorts are the SAME complete-case groups as
    the Stage3 hyperparameter search. Imputer is fit once per inner fold,
    separately from outer train and without any inner validation data.
    """
    split_seed = SEED + split_id
    inner_seed = split_seed + 1000 + split_id
    fold_generator = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed)
    folds = list(fold_generator.split(np.zeros(len(y_tr)), y_tr))
    prepared = []
    for inner_num, (train_idx, val_idx) in enumerate(folds, start=1):
        common_val = val_idx[complete_tr[val_idx]]
        train_x = X_tr.iloc[train_idx].copy()
        val_x = X_tr.iloc[common_val].copy()
        if len(common_val) == 0 or val_x.isna().any().any():
            raise ValueError("STOP: inner common-case validation invalid")
        _, first_draw = fit_mice_draw(train_x, seed=SEED + 100 * split_id + 1)
        prepared.append((first_draw, y_tr.iloc[train_idx], val_x, y_tr.iloc[common_val]))

    candidate_rows = []
    for candidate_id, params in enumerate(original.parameter_grid(model), start=1):
        scores = []
        for X_fit, y_fit, X_val, y_val in prepared:
            model_pipeline = original.build_pipeline("CCA", model, params, split_seed)
            model_pipeline.fit(X_fit, y_fit)
            probabilities = model_pipeline.predict_proba(X_val)[:, 1]
            scores.append(float(roc_auc_score(y_val, probabilities)))
        candidate_rows.append({
            "split_id": split_id,
            "model": model,
            "candidate_id": candidate_id,
            "params_json": json.dumps(original.json_safe(params), sort_keys=True),
            "inner_mean_roc_auc": float(np.mean(scores)),
            "inner_sd_roc_auc": float(np.std(scores, ddof=1)),
        })
    best_idx = int(np.argmax([item["inner_mean_roc_auc"] for item in candidate_rows]))
    return json.loads(candidate_rows[best_idx]["params_json"]), pd.DataFrame(candidate_rows)


def mice_training_split_one(inp: Inputs, output: Path) -> None:
    """Development fit only: no SHAP, no 25-fold full run, no participant-level CSV."""
    X_tr, y_tr, _, X_te, y_te = get_split(inp, 1)
    split_id = 1
    started = time.perf_counter()
    fit_rows = []
    bundle_and_draw = []
    for j in range(1, M + 1):
        seed = SEED + 100 * split_id + j
        fold_started = time.perf_counter()
        bundle, X_filled = fit_mice_draw(X_tr, seed=seed)
        fit_rows.append({
            "split_id": split_id, "imputation_id": j, "seed": seed,
            "n_iter": bundle["n_iter"],
            "seconds": float(time.perf_counter() - fold_started),
        })
        bundle_and_draw.append((bundle, X_filled))
        print(f"MICE | draw={j} | n_iter={bundle['n_iter']} | seed={seed}", flush=True)

    rows = []
    for model in original.MODELS:
        params, candidates = tune_mice_first_draw(
            X_tr, y_tr, (~X_tr.isna().any(axis=1)).to_numpy(), model, split_id,
        )
        candidates.to_csv(output / f"mice_tuning_split001_{model}.csv", index=False)
        preds = []
        for j, (bundle, X_filled) in enumerate(bundle_and_draw, start=1):
            fit_model = original.build_pipeline("CCA", model, params, SEED + split_id)
            fit_model.fit(X_filled, y_tr)
            p = fit_model.predict_proba(X_te)[:, 1]
            if p.shape != (len(y_te),) or not np.isfinite(p).all():
                raise ValueError("STOP: invalid probabilities")
            preds.append(p)
            # Files remain local inside ignored output/ and must NOT be committed.
            joblib.dump(
                {"mice_bundle": bundle, "classifier_pipeline": fit_model},
                output / f"split001_{model}_mice_draw{j}.joblib", compress=3,
            )
        p_bar = np.mean(preds, axis=0)
        rows.append({
            "split_id": split_id, "model": model, "method": MICE_NAME,
            "roc_auc": float(roc_auc_score(y_te, p_bar)),
            "brier_score": float(brier_score_loss(y_te, p_bar)),
            "n_imputations": M,
            "train_n": len(y_tr), "test_n": len(y_te),
            "params_json": json.dumps(original.json_safe(params), sort_keys=True),
        })
        print(f"MICE | {model} | AUC={rows[-1]['roc_auc']:.6f} | "
              f"Brier={rows[-1]['brier_score']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "mice_split1_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(fit_rows).to_csv(output / "mice_split1_imputer_iterations.csv", index=False)
    seconds = time.perf_counter() - started
    (output / "mice_split1_training_runtime.json").write_text(
        json.dumps({"seconds": seconds, "note": "EXCLUDES SHAP; cannot be used as the Stage4 12-hour gate"}, indent=2),
        encoding="utf-8",
    )
    print(f"MICE TRAINING (no SHAP): PASS | seconds={seconds:.1f}")
    print("OUTPUT:", output)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["preflight", "baseline-split1", "mice-training-split1"])
    parser.add_argument("--stage3-dir", type=Path, default=root / "output" / "stage3_final")
    parser.add_argument("--stage4-dir", type=Path, default=root / "output" / "stage4_mice")
    args = parser.parse_args()
    ensure_revision_branch(root)
    inputs = read_inputs(root, args.stage3_dir)
    X_tr, y_tr, complete_tr, X_te, _ = get_split(inputs, 1)
    if args.mode == "preflight":
        print("PASS | stage3-revision branch")
        print("PASS | Original input SHA-256 verified")
        print("PASS | All 25 manifest partitions match Stage3 seed 2026")
        print("PASS | 150 reference metrics")
        print(f"PASS | split1: train={len(X_tr)}, complete_train={int(complete_tr.sum())}, "
              f"common_test={len(X_te)}")
        print("STEP 2 PREFLIGHT: PASS")
    elif args.mode == "baseline-split1":
        output = choose_output(args.stage4_dir, "step2_baseline_split1")
        baseline_split_one(inputs, output)
    else:
        output = choose_output(args.stage4_dir, "step2_mice_training_split1")
        mice_training_split_one(inputs, output)


if __name__ == "__main__":
    main()
