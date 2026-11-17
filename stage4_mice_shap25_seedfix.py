#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 6 RF permutation-seed correction, safe continuation edition.

This script creates a completely separate, independently verified Stage 6
output using the original Stage 6 source as READ-ONLY data. It reuses all
125 LR SHAP draws and the five split-1 RF SHAP draws; re-evaluates only the
120 RF draws belonging to splits 2..25 with seed=2026+split_id.

PROVISIONAL research output; ADVISOR APPROVAL PENDING.

Run preflight first; then use --max-new-splits 1 for a split-2 pilot.
Original Stage 3, Stage 5 and Stage 6 outputs are NEVER modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid writing __pycache__ on the user's existing project modules.
sys.dont_write_bytecode = True

import stage4_mice_shap25 as legacy
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss, roc_auc_score

import stage3_cv as s3
import stage3_shap as s3sh
import stage4_mice_cv as s4
import stage4_mice_cv25 as s5
import stage4_mice_benchmark_split1 as s4bench

SCHEMA = "sina-stage4-step6-v1"  # Keep compatible original LR/split-1 checkpoints.
SEEDFIX_SCHEMA = "sina-stage4-step6-seedfix-v1"
LEGACY_STAGE6_SCRIPT_SHA256 = "76bff6d33b310a8dcb98c739a962fbf04cf5d6c10afcfbfe13685ee554696aa3"
MODELS = ("LR", "RF")
DRAWS = (1, 2, 3, 4, 5)
MAX_SPLITS = 25
TOL = 1e-8
PREFIX = "mice_m5"  # stage3_shap.method_slug(model, 'MICE m=5')
BENCH_SHAP_TOL = 1e-8


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | {message}", flush=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"STOP: refusing to replace existing file {path}")
    tmp = path.with_name(path.name + ".seedfix_writing")
    if tmp.exists():
        if not tmp.is_file():
            raise RuntimeError(f"STOP: unexpected JSON temp artifact {tmp}")
        tmp.unlink()  # Only this script's temporary, inside the NEW corrected run.
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(s3.json_safe(record), stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)  # Exclusive, atomic final name.
    tmp.unlink()


def write_new_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"STOP: refusing to replace existing file {path}")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"STOP: empty output {path}")
    tmp = path.with_name(path.name + ".seedfix_writing")
    if tmp.exists():
        if not tmp.is_file():
            raise RuntimeError(f"STOP: unexpected CSV temp artifact {tmp}")
        tmp.unlink()  # Only this script's temporary, inside the NEW corrected run.
    with tmp.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, float_format="%.17g")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()


def write_or_check_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    """For interruption after first of multiple CSVs, verify instead of replacing."""
    expected = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not path.exists():
        write_new_csv(path, expected)
        return
    if not path.is_file():
        raise RuntimeError(f"STOP: expected CSV path is not a file: {path}")
    actual = pd.read_csv(path)
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise RuntimeError(f"STOP: existing checkpoint CSV schema/shape changed: {path}")
    for name in expected.columns:
        a = actual[name]
        b = expected[name]
        if pd.api.types.is_numeric_dtype(b):
            if (not np.isfinite(a.to_numpy(dtype=float)).all()
                or not np.allclose(a.to_numpy(dtype=float),
                                   b.to_numpy(dtype=float), atol=1e-12, rtol=0)):
                raise RuntimeError(f"STOP: existing checkpoint CSV values changed in {path}, {name}")
        else:
            if not np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy()):
                raise RuntimeError(f"STOP: existing checkpoint CSV labels changed in {path}, {name}")


def write_or_check_json(path: Path, record: dict[str, Any]) -> None:
    if not path.exists():
        write_new_json(path, record)
    elif read_json(path) != s3.json_safe(record):
        raise RuntimeError(f"STOP: existing checkpoint JSON differs: {path}")


def new_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"STOP: refusing to replace existing SHAP draw {path}")
    tmp = path.with_name(path.name + ".seedfix_writing")
    if tmp.exists():
        if not tmp.is_file():
            raise RuntimeError(f"STOP: unexpected NPZ temp artifact {tmp}")
        tmp.unlink()  # Only this script's temporary, inside the NEW corrected run.
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()


def within(root: Path, path: Path) -> Path:
    parent = (root / "output" / "stage4_mice").resolve()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError("STOP: run path must be a child of output/stage4_mice/")
    return path


def checked_artifact(folder: Path, rel_name: str, expected_hash: str) -> Path:
    path = (folder / rel_name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise RuntimeError(f"STOP: missing/outside checkpoint artifact {rel_name}")
    if digest(path) != expected_hash:
        raise RuntimeError(f"STOP: sha256 mismatch: {path}")
    return path


def verify_stage5_aggregates(stage5_dir: Path, stage5_metrics: list[dict], diag: list[dict],
                             tuning: list[dict]) -> pd.DataFrame:
    expected_files = {
        "mice_cv_metrics_25_splits.csv", "mice_imputer_diagnostics_25_splits.csv",
        "mice_inner_tuning_25_splits.csv", "step5_summary.json",
    }
    sha_path = stage5_dir / "step5_aggregate_sha256.json"
    hashes = read_json(sha_path)
    if set(hashes) != expected_files:
        raise RuntimeError("STOP: Stage 5 aggregate SHA list differs from expected 4 files")
    for name, expected in hashes.items():
        checked_artifact(stage5_dir, name, expected)
    summary = read_json(stage5_dir / "step5_summary.json")
    if (summary.get("completed_splits"), summary.get("metric_rows"),
        summary.get("imputer_diagnostic_rows"), summary.get("tuning_rows"),
        summary.get("no_shap_done_in_step5")) != (25, 50, 125, 325, True):
        raise RuntimeError("STOP: Stage 5 completion summary does not pass")
    if summary.get("has_nonfinite_auc_or_brier"):
        raise RuntimeError("STOP: Stage 5 reported nonfinite AUC/Brier")
    if (len(stage5_metrics), len(diag), len(tuning)) != (50, 125, 325):
        raise RuntimeError("STOP: Stage 5 split checkpoints incomplete")
    frame = pd.read_csv(stage5_dir / "mice_cv_metrics_25_splits.csv")
    if (len(frame) != 50 or frame.duplicated(["split_id", "model"]).any()
        or set(frame["model"]) != set(MODELS)
        or set(frame["split_id"]) != set(range(1, MAX_SPLITS + 1))
        or not frame["n_imputations"].eq(5).all()
        or not np.isfinite(frame[["roc_auc", "brier_score"]].to_numpy(dtype=float)).all()):
        raise RuntimeError("STOP: invalid Stage 5 metrics CSV")
    by_run = {(int(r["split_id"]), str(r["model"])): r for r in stage5_metrics}
    for _, row in frame.iterrows():
        r = by_run[(int(row["split_id"]), row["model"])]
        if (abs(float(row["roc_auc"]) - float(r["roc_auc"])) > 1e-12
           or abs(float(row["brier_score"]) - float(r["brier_score"])) > 1e-12):
            raise RuntimeError("STOP: Stage 5 aggregate and checkpoints differ")
    df_diag = pd.read_csv(stage5_dir / "mice_imputer_diagnostics_25_splits.csv")
    if (len(df_diag) != 125
        or df_diag.duplicated(["split_id", "imputation_id"]).any()
        or not df_diag["n_iter_"].between(1, 10).all()
        or not np.isfinite(df_diag.select_dtypes(include="number").to_numpy(dtype=float)).all()):
        raise RuntimeError("STOP: invalid Stage 5 MICE diagnostics")
    for _, r in df_diag.iterrows():
        if int(r["seed"]) != 2026 + 100 * int(r["split_id"]) + int(r["imputation_id"]):
            raise RuntimeError("STOP: Stage 5 MICE seed violates advisor protocol")
    return frame


def original_test_and_background(inp: s4.Inputs, split_id: int,
                                  bg_manifest: pd.DataFrame,
                                  original_shap_root: Path
                                  ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    split_manifest = inp.manifest.loc[inp.manifest["split_id"].eq(split_id)]
    roles = split_manifest.set_index("SEQN").reindex(inp.df["SEQN"])["role"]
    if roles.isna().any():
        raise RuntimeError(f"STOP: original split {split_id} has missing roles")
    df = inp.df
    train_complete = df.loc[roles.eq("train").to_numpy() & inp.complete].copy()
    test = df.loc[roles.eq("test").to_numpy() & inp.complete].copy()
    bg = train_complete.sample(n=100, random_state=2026 + split_id)
    bg = bg.sort_values("SEQN").reset_index(drop=True)
    test = test.sort_values("SEQN").reset_index(drop=True)
    if not bg[s3.PREDICTORS].notna().all().all() or not test[s3.PREDICTORS].notna().all().all():
        raise RuntimeError(f"STOP: missing predictor in SHAP background/test split {split_id}")
    recorded_bg = bg_manifest.loc[bg_manifest["split_id"].eq(split_id), "SEQN"].to_numpy(dtype=np.int64)
    if len(recorded_bg) != 100 or not np.array_equal(bg["SEQN"].to_numpy(dtype=np.int64), recorded_bg):
        raise RuntimeError(f"STOP: SHAP background mismatch with original Stage 3 split {split_id}")
    # Compare participant alignment and original outcome, but never export them as aggregate metadata.
    original_values = pd.read_csv(
        original_shap_root / "values_by_split" / f"shap_values_split_{split_id:03d}.csv",
        usecols=["SEQN", "y_true"],
    )
    seqn = test["SEQN"].to_numpy(dtype=np.int64)
    y = test[s3.TARGET].to_numpy(dtype=np.int64)
    if (not np.array_equal(seqn, original_values["SEQN"].to_numpy(dtype=np.int64))
        or not np.array_equal(y, original_values["y_true"].to_numpy(dtype=np.int64))):
        raise RuntimeError(f"STOP: original Stage 3 test IDs/outcomes differ in split {split_id}")
    return seqn, y, bg[s3.PREDICTORS].copy(), test[s3.PREDICTORS].copy(), test.index.to_numpy()


def preflight(root: Path, stage5_arg: Path, bench_arg: Path) -> dict[str, Any]:
    # Stage5.preflight reads original inputs and enforces software, train/test,
    # dataset, six-way reproduction, leakage, and one-split SHAP benchmark gates.
    inp, bench_meta, bench_models, bench_path, identity5 = s5.preflight(root, bench_arg)
    if s3sh.ADD_TOLERANCE != TOL or s3sh.PERMUTATION_CYCLES != 5 or s3sh.BACKGROUND_N != 100:
        raise RuntimeError("STOP: SHAP protocol settings changed")
    if s3sh.PREDICTORS != s3.PREDICTORS or s3sh.BASE_PERMUTATION_SEED != 2026:
        raise RuntimeError("STOP: original SHAP feature order or seed changed")
    if s3sh.method_slug("LR", "MICE m=5") != f"LR__{PREFIX}":
        raise RuntimeError("STOP: MICE SHAP column naming convention changed")
    stage5_dir = within(root, stage5_arg)
    cfg = read_json(stage5_dir / "run_config.json")
    if cfg.get("schema") != s5.RUN_SCHEMA or cfg.get("identity") != identity5:
        raise RuntimeError("STOP: Stage 5 run does not match the current input/code/benchmark")
    completed, metrics, diag, tuning = s5.collect_progress(stage5_dir, bench_models)
    if completed != list(range(1, MAX_SPLITS + 1)):
        raise RuntimeError(f"STOP: Stage 5 completed splits are {completed}, expected 1..25")
    metric_frame = verify_stage5_aggregates(stage5_dir, metrics, diag, tuning)
    orig_shap = root / "output" / "stage3_final" / "shap"
    bg_file = orig_shap / "shap_background_manifest_final.csv"
    bg_manifest = pd.read_csv(bg_file)
    if len(bg_manifest) != MAX_SPLITS * 100:
        raise RuntimeError("STOP: original SHAP background must have 2500 participant rows")
    for split_id in range(1, MAX_SPLITS + 1):
        seqn, y, bg, test, _ = original_test_and_background(inp, split_id, bg_manifest, orig_shap)
        Xtr, ytr, complete, Xte, yte = s4.get_split(inp, split_id)
        if len(test) != len(Xte) or not np.array_equal(y, yte.to_numpy(dtype=int)):
            raise RuntimeError(f"STOP: SHAP and Stage 5 participant alignment differ, split {split_id}")
        if not test.equals(Xte):
            raise RuntimeError(f"STOP: SHAP and Stage 5 feature values/order differ, split {split_id}")
    global_bench = bench_path / "mice_split1_global_shap.csv"
    benchmark_global = pd.read_csv(global_bench)
    if (len(benchmark_global) != 22
        or benchmark_global.duplicated(["model", "feature"]).any()
        or set(benchmark_global["model"]) != set(MODELS)):
        raise RuntimeError("STOP: full split-1 benchmark lacks 22 global SHAP values")
    src = ("stage3_cv.py", "stage3_shap.py", "stage4_mice_cv.py",
           "stage4_mice_cv25.py", "stage4_mice_benchmark_split1.py",
           "stage4_environment_gate.py", Path(__file__).name)
    source_hashes = {name: digest(root / name) for name in src}
    run_identity = {
        "schema": SCHEMA, "stage5_path": str(stage5_dir),
        "stage5_run_config_sha256": digest(stage5_dir / "run_config.json"),
        "stage5_aggregate_sha256_manifest_sha256": digest(stage5_dir / "step5_aggregate_sha256.json"),
        "benchmark_path": str(bench_path),
        "benchmark_global_sha256": digest(global_bench),
        "original_stage3_background_manifest_sha256": digest(bg_file),
        "original_stage3_dataset_sha256": identity5["dataset_sha256"],
        "original_split_manifest_sha256": identity5["manifest_sha256"],
        "code_sha256": source_hashes, "python": platform.python_version(),
        "shap_version": shap.__version__,
        "advisor_status": "PENDING; PROVISIONAL",
    }
    log("STEP 6 PREFLIGHT: PASS | 25 Stage 5 splits, 250 saved classifiers, "
        "Stage 3 test IDs/backgrounds, benchmark SHAP, hashes")
    log("STEP 6 PREFLIGHT: PASS | check only; NO classifiers loaded, "
        "NO model training, NO SHAP computation, NO files written")
    return {"inp": inp, "bench_models": bench_models, "benchmark_global": benchmark_global,
            "stage5_dir": stage5_dir, "benchmark_dir": bench_path,
            "metrics": metric_frame, "bg_manifest": bg_manifest,
            "stage3_shap_dir": orig_shap, "identity": run_identity}


def draw_paths(split_dir: Path, model: str, draw: int) -> tuple[Path, Path]:
    base = split_dir / "draws" / f"{model}_draw{draw:02d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def validate_saved_draw(split_dir: Path, model: str, draw: int, seqn: np.ndarray,
                        expected_model_hash: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    npz_path, marker_path = draw_paths(split_dir, model, draw)
    if not marker_path.is_file() or not npz_path.is_file():
        raise RuntimeError(f"STOP: incomplete SHAP checkpoint {npz_path}; do not overwrite it")
    marker = read_json(marker_path)
    if (marker.get("schema") != SCHEMA or marker.get("model") != model
        or marker.get("imputation_id") != draw
        or marker.get("classifier_sha256") != expected_model_hash):
        raise RuntimeError(f"STOP: SHAP draw marker identity changed: {marker_path}")
    split_id = int(split_dir.name.removeprefix("split_"))
    if model == "RF":
        expected_seed = int(s3sh.BASE_PERMUTATION_SEED + split_id)
        if split_id == 1:
            if marker.get("permutation_seed", expected_seed) != expected_seed:
                raise RuntimeError("STOP: original split-1 RF seed marker invalid")
        elif marker.get("permutation_seed") != expected_seed:
            raise RuntimeError(f"STOP: RF SHAP seed marker mismatch in split {split_id}, draw {draw}")
    checked_artifact(split_dir, str(npz_path.relative_to(split_dir)), marker["shap_values_sha256"])
    with np.load(npz_path, allow_pickle=False) as npz:
        if set(npz.files) != {"seqn", "values", "base", "output", "probability"}:
            raise RuntimeError("STOP: private SHAP draw checkpoint has wrong fields")
        ids = npz["seqn"]
        values = npz["values"]
        base = npz["base"]
        output = npz["output"]
        prob = npz["probability"]
    if (not np.array_equal(ids, seqn) or values.shape != (len(seqn), 11)
        or any(x.shape != (len(seqn),) for x in (base, output, prob))):
        raise RuntimeError("STOP: private SHAP checkpoint shape/order mismatch")
    if (not np.isfinite(values).all() or not np.isfinite(base).all()
        or not np.isfinite(output).all() or not np.isfinite(prob).all()
        or not ((prob >= 0.0) & (prob <= 1.0)).all()):
        raise RuntimeError("STOP: private SHAP checkpoint has nonfinite/impossible values")
    err = float(np.max(np.abs(base + values.sum(axis=1) - output)))
    if not np.isfinite(err) or err >= TOL or abs(err - marker["max_abs_additivity_error"]) > 1e-12:
        raise RuntimeError("STOP: private SHAP checkpoint does not satisfy local additivity")
    if model == "RF" and not np.allclose(output, prob, atol=1e-10, rtol=0.0):
        raise RuntimeError("STOP: RF SHAP output is not positive class probability")
    return values, base, output, prob, marker


def explain_one_seedfixed(pipeline: Any, model: str, bg_raw: pd.DataFrame,
                          eval_raw: pd.DataFrame, split_id: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Stage-3-equivalent SHAP: FIXED RF seed=2026+actual split_id.

    Reuse the original benchmark's trusted LR implementation unchanged.
    This RF branch matches the benchmark's RF implementation except for
    explicitly accepting the real split ID instead of its global SPLIT_ID=1.
    """
    if model == "LR":
        return s4bench.explain_one(pipeline, model, bg_raw, eval_raw)
    if model != "RF" or not 1 <= split_id <= MAX_SPLITS:
        raise RuntimeError("STOP: invalid model or split ID for corrected RF SHAP")
    pp = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["classifier"]
    X_bg = s3sh.dense_float(pp.transform(bg_raw))
    X_eval = s3sh.dense_float(pp.transform(eval_raw))
    names = list(pp.get_feature_names_out())
    groups = s3sh.map_original_features(names)
    if X_eval.shape[1] != X_bg.shape[1] or X_eval.shape[1] != len(names):
        raise RuntimeError("STOP: encoded RF feature shape/name mismatch")
    masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
    if list(clf.classes_) != [0, 1]:
        raise RuntimeError("STOP: RF classes are not [0, 1]")
    if "n_jobs" in clf.get_params(deep=False):
        clf.set_params(n_jobs=1)

    def predict_positive(x: Any) -> np.ndarray:
        return clf.predict_proba(s3sh.dense_float(x))[:, 1]

    correct_seed = int(s3sh.BASE_PERMUTATION_SEED + split_id)
    explainer = shap.PermutationExplainer(
        predict_positive, masker, feature_names=names, seed=correct_seed)
    max_evals = 2 * X_eval.shape[1] * s3sh.PERMUTATION_CYCLES + 1
    expl = explainer(X_eval, max_evals=max_evals, batch_size=256, silent=False)
    values, base = s3sh.normalize_scalar_shap(
        expl.values, expl.base_values, len(X_eval))
    output = clf.predict_proba(X_eval)[:, 1]
    if values.shape != X_eval.shape or not np.isfinite(values).all() or not np.isfinite(base).all():
        raise RuntimeError("STOP: invalid RF SHAP matrix or base values")
    max_error = float(np.max(np.abs(base + values.sum(axis=1) - output)))
    if not np.isfinite(max_error) or max_error >= TOL:
        raise RuntimeError(f"STOP: corrected RF SHAP additivity failed: {max_error}")
    grouped = s3sh.aggregate_to_original(values, groups)
    if grouped.shape != (len(eval_raw), 11) or not np.isfinite(grouped).all():
        raise RuntimeError("STOP: corrected RF attribution aggregation failed")
    return grouped, base, output, max_error, int(max_evals)


def save_draw(split_dir: Path, model: str, draw: int, seqn: np.ndarray,
              model_path: Path, model_sha: str, Xbg: pd.DataFrame, Xte: pd.DataFrame,
              split_id: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    npz_path, marker_path = draw_paths(split_dir, model, draw)
    if npz_path.exists() or marker_path.exists():
        raise RuntimeError(f"STOP: refusing to overwrite partial SHAP draw {model} {draw}")
    pipeline = joblib.load(model_path)  # Only use the user's local, hash-verified, trusted models.
    t = time.perf_counter()
    values, base, output, err, max_evals = explain_one_seedfixed(pipeline, model, Xbg, Xte, split_id)
    seconds = time.perf_counter() - t
    probability = np.asarray(pipeline.predict_proba(Xte)[:, 1], dtype=float)
    if (probability.shape != (len(seqn),)
        or not np.isfinite(probability).all()
        or not ((probability >= 0.0) & (probability <= 1.0)).all()):
        raise RuntimeError("STOP: invalid per-draw predicted probabilities")
    if model == "RF" and not np.allclose(probability, output, atol=1e-10, rtol=0.0):
        raise RuntimeError("STOP: RF predicted probabilities do not match SHAP output")
    new_npz(npz_path, seqn=seqn, values=values, base=base,
            output=output, probability=probability)
    marker = {
        "schema": SCHEMA, "model": model, "imputation_id": draw,
        "shap_values_sha256": digest(npz_path), "classifier_sha256": model_sha,
        "n_test_common": len(seqn), "n_background": 100,
        "max_abs_additivity_error": float(err),
        "shap_seconds": float(seconds), "permutation_max_evals": int(max_evals),
        "shap_scale": "log_odds" if model == "LR" else "positive_class_probability",
        "permutation_seed": (int(s3sh.BASE_PERMUTATION_SEED + split_id)
                             if model == "RF" else None),
        "status": "PASS",
    }
    write_new_json(marker_path, marker)
    log(f"SHAP PASS | split={int(split_dir.name[-3:])}/25 | {model} "
        f"draw={draw}/5 | seconds={seconds:.1f} additivity_err={err:.3g}")
    return validate_saved_draw(split_dir, model, draw, seqn, model_sha)


def split5_model_path(stage5: Path, split_id: int, model: str, draw: int) -> tuple[Path, str]:
    folder = stage5 / f"split_{split_id:03d}"
    marker = read_json(folder / "split_complete.json")
    name = f"models/{model}_draw{draw:02d}_classifier.joblib"
    sha = marker.get("sha256", {}).get(name)
    if not sha:
        raise RuntimeError(f"STOP: no original Stage 5 model SHA for split {split_id}, {model}, {draw}")
    return checked_artifact(folder, name, sha), sha


def compare_benchmark_global(split_id: int, global_rows: list[dict], baseline: pd.DataFrame) -> None:
    if split_id != 1:
        return
    got = pd.DataFrame(global_rows)[["model", "feature", "mean_abs_shap", "mean_shap"]]
    ref = baseline[["model", "feature", "mean_abs_shap", "mean_shap"]]
    merged = got.merge(ref, on=["model", "feature"], suffixes=("_new", "_ref"), validate="one_to_one")
    if len(merged) != 22:
        raise RuntimeError("STOP: split 1 SHAP global feature list differs from full benchmark")
    for name in ("mean_abs_shap", "mean_shap"):
        delta = np.max(np.abs(merged[f"{name}_new"].to_numpy() - merged[f"{name}_ref"].to_numpy()))
        if not np.isfinite(delta) or delta > BENCH_SHAP_TOL:
            raise RuntimeError(f"STOP: split 1 pooled SHAP differs from validated SHAP benchmark in {name}: {delta}")
    log("SPLIT 1 SHAP BENCHMARK CONTROL: PASS | 22 global feature summaries match to 1e-8")


def finished_split(split_dir: Path, split_id: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    marker_path = split_dir / "split_complete.json"
    marker = read_json(marker_path)
    if marker.get("schema") != SCHEMA or marker.get("split_id") != split_id:
        raise RuntimeError("STOP: completed Stage 6 split marker invalid")
    expected = marker.get("sha256", {})
    if len(expected) != 10 * 2 + 4:  # 10 draw NPZs + 10 draw JSONs + 4 per-split outputs
        raise RuntimeError("STOP: Stage 6 complete marker has incorrect artifact count")
    for rel, expected_hash in expected.items():
        checked_artifact(split_dir, rel, expected_hash)
    audit = pd.read_csv(split_dir / "mice_shap_draw_additivity.csv").to_dict("records")
    pooled = pd.read_csv(split_dir / "mice_shap_pooled_additivity.csv").to_dict("records")
    global_imp = pd.read_csv(split_dir / "mice_shap_global_importance.csv").to_dict("records")
    if (len(audit), len(pooled), len(global_imp)) != (10, 2, 22):
        raise RuntimeError("STOP: Stage 6 completed split row counts incorrect")
    if any(x["status"] != "PASS" or float(x["max_abs_additivity_error"]) >= TOL for x in audit):
        raise RuntimeError("STOP: Stage 6 draw additivity violation in saved split")
    if any(x["status"] != "PASS" or float(x["pooled_max_abs_additivity_error"]) >= TOL for x in pooled):
        raise RuntimeError("STOP: Stage 6 pooled additivity violation in saved split")
    for draw in DRAWS:
        _, marker_path = draw_paths(split_dir, "RF", draw)
        data = read_json(marker_path)
        correct_seed = int(s3sh.BASE_PERMUTATION_SEED + split_id)
        if split_id == 1:
            if data.get("permutation_seed", correct_seed) != correct_seed:
                raise RuntimeError("STOP: split-1 RF seed control fails")
        elif data.get("permutation_seed") != correct_seed:
            raise RuntimeError(f"STOP: incorrect RF seed in completed corrected split {split_id}")
    return audit, pooled, global_imp, marker


def run_one_split(ctx: dict, run_dir: Path, split_id: int) -> None:
    start = time.perf_counter()
    inp: s4.Inputs = ctx["inp"]
    step5_dir: Path = ctx["stage5_dir"]
    split_dir = run_dir / f"split_{split_id:03d}"
    if not split_dir.exists():
        split_dir.mkdir(parents=False, exist_ok=False)
        (split_dir / "draws").mkdir(exist_ok=False)
    if (split_dir / "split_complete.json").exists():
        raise RuntimeError("STOP: a completed split must not be rerun")
    if not (split_dir / "draws").is_dir():
        raise RuntimeError("STOP: no private per-draw folder found in unfinished split")
    seqn, y, bg, test, _ = original_test_and_background(
        inp, split_id, ctx["bg_manifest"], ctx["stage3_shap_dir"])
    metrics_ref = ctx["metrics"]
    split5_metrics = metrics_ref.loc[metrics_ref["split_id"].eq(split_id)]
    if len(split5_metrics) != 2:
        raise RuntimeError("STOP: missing Stage 5 reference performance for current split")

    private_values = pd.DataFrame({
        "split_id": split_id,
        "repeat": (split_id - 1) // 5 + 1,
        "fold": (split_id - 1) % 5 + 1,
        "SEQN": seqn,
        "y_true": y,
    })
    draw_audit: list[dict] = []
    pooled_audit: list[dict] = []
    globals_: list[dict] = []
    for model in MODELS:
        model_values: list[np.ndarray] = []
        model_base: list[np.ndarray] = []
        model_output: list[np.ndarray] = []
        model_prob: list[np.ndarray] = []
        for draw in DRAWS:
            model_path, model_sha = split5_model_path(step5_dir, split_id, model, draw)
            npz_path, marker_path = draw_paths(split_dir, model, draw)
            if npz_path.exists() or marker_path.exists():
                values, base, output, prob, marker = validate_saved_draw(
                    split_dir, model, draw, seqn, model_sha)
                log(f"SHAP REUSED | split={split_id}/25 | {model} draw={draw}/5 | sha256 verified")
            else:
                log(f"SHAP START | split={split_id}/25 | {model} draw={draw}/5 | "
                    f"n_test={len(seqn)} background=100")
                values, base, output, prob, marker = save_draw(
                    split_dir, model, draw, seqn, model_path, model_sha, bg, test, split_id)
            model_values.append(values)
            model_base.append(base)
            model_output.append(output)
            model_prob.append(prob)
            draw_audit.append({
                "split_id": split_id, "model": model, "imputation_id": draw,
                "n_test_common": len(seqn), "background_n": 100,
                "max_abs_additivity_error": marker["max_abs_additivity_error"],
                "shap_seconds": marker["shap_seconds"],
                "shap_scale": marker["shap_scale"],
                "status": "PASS",
            })
        vbar = np.mean(np.stack(model_values, axis=0), axis=0)
        bbar = np.mean(np.stack(model_base, axis=0), axis=0)
        obar = np.mean(np.stack(model_output, axis=0), axis=0)
        pbar = np.mean(np.stack(model_prob, axis=0), axis=0)
        pooled_error = float(np.max(np.abs(bbar + vbar.sum(axis=1) - obar)))
        if not np.isfinite(pooled_error) or pooled_error >= TOL:
            raise RuntimeError(f"STOP: pooled {model} SHAP additivity failed split {split_id}")
        if model == "RF" and not np.allclose(obar, pbar, atol=1e-10, rtol=0):
            raise RuntimeError("STOP: pooled RF model output not equal to pooled probabilities")
        auc = float(roc_auc_score(y, pbar))
        brier = float(brier_score_loss(y, pbar))
        ref = split5_metrics.loc[split5_metrics["model"].eq(model)].iloc[0]
        if (abs(auc - float(ref["roc_auc"])) > 1e-10
            or abs(brier - float(ref["brier_score"])) > 1e-10):
            raise RuntimeError(f"STOP: {model} saved-model predictions do not match Stage5 performance "
                               f"split {split_id}: auc {auc} vs {ref['roc_auc']}, "
                               f"brier {brier} vs {ref['brier_score']}")
        prefix = s3sh.method_slug(model, "MICE m=5")
        private_values[f"{prefix}__base_value"] = bbar
        for idx, feature in enumerate(s3.PREDICTORS):
            private_values[f"{prefix}__{feature}"] = vbar[:, idx]
            globals_.append({
                "split_id": split_id,
                "repeat": (split_id - 1) // 5 + 1, "fold": (split_id - 1) % 5 + 1,
                "model": model, "method": s4.MICE_NAME, "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(vbar[:, idx]))),
                "mean_shap": float(np.mean(vbar[:, idx])),
                "shap_scale": "log_odds" if model == "LR" else "positive_class_probability",
            })
        pooled_audit.append({
            "split_id": split_id, "model": model,
            "n_imputations": 5, "n_test_common": len(seqn), "background_n": 100,
            "auc": auc, "brier_score": brier,
            "pooled_max_abs_additivity_error": pooled_error,
            "shap_scale": "log_odds" if model == "LR" else "positive_class_probability",
            "status": "PASS",
        })
        log(f"POOLED PASS | split={split_id}/25 | {model} | "
            f"AUC={auc:.6f} Brier={brier:.6f} pooled_additivity={pooled_error:.3g}")

    compare_benchmark_global(split_id, globals_, ctx["benchmark_global"])
    gdf = pd.DataFrame(globals_)
    gdf["rank"] = (gdf.groupby(["split_id", "model", "method"])["mean_abs_shap"]
                     .rank(method="min", ascending=False).astype(int))
    write_or_check_csv(split_dir / "mice_shap_draw_additivity.csv", draw_audit)
    write_or_check_csv(split_dir / "mice_shap_pooled_additivity.csv", pooled_audit)
    write_or_check_csv(split_dir / "mice_shap_global_importance.csv", gdf)
    # Private matched-participant data: necessary in Stage 7 for MSE-SHAP.
    write_or_check_csv(split_dir / "shap_values_private.csv", private_values)
    expected_paths = [p for p in sorted(split_dir.rglob("*")) if p.is_file()]
    if any(p.name.endswith((".seedfix_writing", ".seedfix_copying")) for p in expected_paths):
        raise RuntimeError("STOP: unfinished temporary artifact in corrected split")
    if len(expected_paths) != 10 * 2 + 4:
        raise RuntimeError(f"STOP: incomplete Stage 6 split outputs: {len(expected_paths)}")
    marker = {
        "schema": SCHEMA, "split_id": split_id,
        "n_test_common": len(seqn), "n_background": 100,
        "n_shap_draw_runs": 10, "elapsed_seconds": float(time.perf_counter() - start),
        "sha256": {str(p.relative_to(split_dir)): digest(p) for p in expected_paths},
    }
    write_new_json(split_dir / "split_complete.json", marker)
    log(f"CHECKPOINT PASS | split {split_id}/25 | all private and aggregate artifacts SHA256 verified")


def finish_run(out: Path) -> None:
    audits: list[dict] = []
    pooled: list[dict] = []
    globals_: list[dict] = []
    for split_id in range(1, MAX_SPLITS + 1):
        a, p, g, _ = finished_split(out / f"split_{split_id:03d}", split_id)
        audits.extend(a)
        pooled.extend(p)
        globals_.extend(g)
    if (len(audits), len(pooled), len(globals_)) != (250, 50, 550):
        raise RuntimeError("STOP: wrong Stage 6 aggregation size")
    write_or_check_csv(out / "mice_shap_draw_additivity_25_splits.csv", audits)
    write_or_check_csv(out / "mice_shap_pooled_additivity_25_splits.csv", pooled)
    write_or_check_csv(out / "mice_shap_global_importance_25_splits.csv", globals_)
    meta = {
        "schema": SCHEMA, "advisor_status": "PENDING; PROVISIONAL",
        "n_splits": 25, "n_models": 2, "n_draws_per_model": 5,
        "n_per_draw_audit_rows": len(audits),
        "n_pooled_audit_rows": len(pooled),
        "n_global_feature_rows": len(globals_),
        "all_local_additivity_pass": all(x["status"] == "PASS" and x["max_abs_additivity_error"] < TOL for x in audits),
        "all_pooled_additivity_pass": all(x["status"] == "PASS" and x["pooled_max_abs_additivity_error"] < TOL for x in pooled),
        "private_participant_shap_not_for_distribution": True,
        "seed_correction_schema": SEEDFIX_SCHEMA,
        "rf_permutation_seed_rule": "2026 + split_id",
        "rf_draws_corrected_splits_2_to_25": 120,
        "lr_draws_reused_unchanged": 125,
        "split1_rf_draws_reused_unchanged": 5,
        "original_stage6_unmodified": True,
    }
    write_or_check_json(out / "step6_summary.json", meta)
    sha = {p.name: digest(p) for p in out.glob("*.csv")}
    sha["step6_summary.json"] = digest(out / "step6_summary.json")
    write_or_check_json(out / "step6_aggregate_sha256.json", sha)
    log("STEP 6 SEEDFIX PASS | 25/25 splits | RF reseeded 120/120 | LR reused 125/125 | additivity PASS")


def verified_copy(source: Path, dest: Path, expected_sha256: str) -> None:
    """Copy exactly once to a separate private directory, with hash verification.

    Use an atomic rename of a script-owned temporary file, so interruption
    cannot create a false finalized checkpoint. Pre-existing files are never
    overwritten, and any old temporary file is checked rather than discarded.
    """
    if dest.exists():
        if not dest.is_file() or digest(dest) != expected_sha256:
            raise RuntimeError(f"STOP: destination differs from verified source: {dest}")
        return
    if not source.is_file() or digest(source) != expected_sha256:
        raise RuntimeError(f"STOP: source SHA256 mismatch before copy: {source}")
    if not dest.parent.is_dir():
        raise RuntimeError("STOP: destination parent not prepared")
    tmp = dest.with_name(dest.name + ".seedfix_copying")
    if tmp.exists() and digest(tmp) != expected_sha256:
        # Only a script-specific temporary path is removed; never touch the
        # original source, user checkpoints, or the finalized destination.
        if not tmp.is_file():
            raise RuntimeError(f"STOP: unexpected temporary artifact: {tmp}")
        tmp.unlink()
    if not tmp.exists():
        with source.open("rb") as src, tmp.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    if digest(tmp) != expected_sha256:
        raise RuntimeError(f"STOP: copy hash mismatch: {tmp}")
    if dest.exists():
        raise RuntimeError("STOP: destination unexpectedly appeared during copy")
    # Creating the final name never overwrites an existing file.
    os.link(tmp, dest)
    tmp.unlink()
    if digest(dest) != expected_sha256:
        raise RuntimeError("STOP: copied file does not match its original SHA")


def verify_source_stage6(root: Path, ctx: dict, original: Path, stage5_arg: Path,
                         bench_arg: Path) -> dict[str, Any]:
    if original == (root / "output" / "stage4_mice").resolve():
        raise RuntimeError("STOP: legacy directory must be a full Stage 6 run")
    original_script = root / "stage4_mice_shap25.py"
    if digest(original_script) != LEGACY_STAGE6_SCRIPT_SHA256:
        raise RuntimeError("STOP: original Stage 6 script differs from reviewed seed-bug version")
    if s4bench.SPLIT_ID != 1:
        raise RuntimeError("STOP: the split-one benchmark constant changed unexpectedly")
    old_ctx = legacy.preflight(root, stage5_arg, bench_arg)
    cfgpath = original / "step6_run_config.json"
    if not cfgpath.is_file():
        raise RuntimeError("STOP: original Stage 6 run config is missing")
    cfg = read_json(cfgpath)
    if cfg.get("identity") != old_ctx["identity"]:
        raise RuntimeError("STOP: original Stage 6 identity does not match validated inputs")
    summary_path = original / "step6_summary.json"
    sha_path = original / "step6_aggregate_sha256.json"
    if not summary_path.is_file() or not sha_path.is_file():
        raise RuntimeError("STOP: original Stage 6 has no complete aggregate summary/hashes")
    summary = read_json(summary_path)
    expected_names = {
        "mice_shap_draw_additivity_25_splits.csv",
        "mice_shap_pooled_additivity_25_splits.csv",
        "mice_shap_global_importance_25_splits.csv",
        "step6_summary.json",
    }
    manifest = read_json(sha_path)
    if set(manifest) != expected_names:
        raise RuntimeError("STOP: original Stage 6 aggregate manifest incomplete")
    for name, sha in manifest.items():
        checked_artifact(original, name, sha)
    if (summary.get("n_splits"), summary.get("n_per_draw_audit_rows"),
        summary.get("n_pooled_audit_rows"), summary.get("n_global_feature_rows"),
        summary.get("all_local_additivity_pass"), summary.get("all_pooled_additivity_pass")) != (
            25, 250, 50, 550, True, True):
        raise RuntimeError("STOP: original Stage 6 summary does not indicate complete PASS")
    old_markers: dict[int, dict] = {}
    for split_id in range(1, MAX_SPLITS + 1):
        _, _, _, marker = legacy.finished_split(original / f"split_{split_id:03d}", split_id)
        old_markers[split_id] = marker
    log("SEEDFIX SOURCE PASS | Stage 6 original 25/25 splits and all 4 aggregate files SHA256 verified")
    return {
        "original_path": str(original),
        "original_step6_config_sha256": digest(cfgpath),
        "original_step6_aggregate_manifest_sha256": digest(sha_path),
        "original_script_sha256": LEGACY_STAGE6_SCRIPT_SHA256,
        "markers": old_markers,
    }


def seedfix_prepare_split(out: Path, original: Path, split_id: int, marker: dict) -> None:
    """Import only the known-correct LR draws, plus the complete split-1 anchor."""
    source_split = original / f"split_{split_id:03d}"
    dest_split = out / f"split_{split_id:03d}"
    if not dest_split.exists():
        dest_split.mkdir(exist_ok=False)
    if not dest_split.is_dir():
        raise RuntimeError("STOP: corrected split folder is not a directory")
    drawdir = dest_split / "draws"
    if not drawdir.exists():
        drawdir.mkdir(exist_ok=False)
    if not drawdir.is_dir():
        raise RuntimeError("STOP: corrected draw folder is not a directory")
    expected = marker["sha256"]
    if split_id == 1:
        # The original split-one fixed seed 2027 is already exactly correct.
        # Make split_complete.json LAST to commit only after all 24 files.
        for rel_name, sha in sorted(expected.items()):
            verified_copy(source_split / rel_name, dest_split / rel_name, sha)
        verified_copy(source_split / "split_complete.json",
                      dest_split / "split_complete.json",
                      digest(source_split / "split_complete.json"))
        finished_split(dest_split, split_id)
        log("SEEDFIX ANCHOR PASS | split 1/25 entirely reused, original benchmark preserved")
    else:
        # Deliberately DO NOT copy old RF draws or old pooled/private files.
        # That data must be regenerated using actual split-specific seeds.
        for draw in DRAWS:
            for suffix in ("npz", "json"):
                rel = f"draws/LR_draw{draw:02d}.{suffix}"
                verified_copy(source_split / rel, dest_split / rel, expected[rel])
        log(f"SEEDFIX LR REUSED | split={split_id}/25 | 5 verified original LR draws")


def validate_unchanged_lr(out: Path, original: Path, split_id: int) -> None:
    """Prove LR SHAP summary and individual draw files did not change."""
    old = original / f"split_{split_id:03d}"
    new = out / f"split_{split_id:03d}"
    for draw in DRAWS:
        for suffix in ("json", "npz"):
            rel = f"draws/LR_draw{draw:02d}.{suffix}"
            if digest(old / rel) != digest(new / rel):
                raise RuntimeError("STOP: corrected output did not preserve LR draw exactly")
    for fname, sort_cols, values in (
        ("mice_shap_global_importance.csv", ["feature"], ["mean_abs_shap", "mean_shap", "rank"]),
        ("mice_shap_pooled_additivity.csv", ["model"], ["auc", "brier_score", "pooled_max_abs_additivity_error"]),
        ("mice_shap_draw_additivity.csv", ["imputation_id"], ["max_abs_additivity_error", "shap_seconds"]),
    ):
        a = pd.read_csv(old / fname).query("model == 'LR'").sort_values(sort_cols).reset_index(drop=True)
        b = pd.read_csv(new / fname).query("model == 'LR'").sort_values(sort_cols).reset_index(drop=True)
        if len(a) != len(b) or not np.allclose(a[values].to_numpy(), b[values].to_numpy(), atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP: recombined LR summaries changed: {fname}")
    a = pd.read_csv(old / "shap_values_private.csv")
    b = pd.read_csv(new / "shap_values_private.csv")
    lrcols = ["SEQN", "y_true"] + [x for x in a.columns if x.startswith("LR__mice_m5")]
    if (not set(lrcols).issubset(b.columns) or not np.array_equal(a["SEQN"], b["SEQN"])
        or not np.allclose(a[lrcols].to_numpy(dtype=float),
                           b[lrcols].to_numpy(dtype=float), atol=1e-12, rtol=0)):
        raise RuntimeError("STOP: private matched LR SHAP values were changed")
    log(f"SEEDFIX LR INVARIANCE PASS | split={split_id}/25")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    parser.add_argument("--stage5-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--source-stage6-dir", type=Path, required=True,
                        help="COMPLETED original Stage 6 results; READ-ONLY")
    parser.add_argument("--out", type=Path,
                        help="NEW separate private Stage 6 seedfix output directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-splits", type=int, default=1,
                        help="Number of NEW corrected RF splits 2..25 (default one)")
    args = parser.parse_args()
    if args.mode == "preflight" and (args.out is not None or args.resume):
        parser.error("Read-only preflight does not accept --out or --resume")
    if args.mode == "run" and args.out is None:
        parser.error("Run requires an explicit NEW --out directory")
    if not 1 <= args.max_new_splits <= 24:
        parser.error("--max-new-splits must be in 1..24")
    root = Path(__file__).resolve().parent
    original = within(root, args.source_stage6_dir)
    ctx = preflight(root, args.stage5_dir, args.benchmark_dir)
    src = verify_source_stage6(root, ctx, original, args.stage5_dir, args.benchmark_dir)
    if args.mode == "preflight":
        log("SEEDFIX PREFLIGHT PASS | read-only; 120 RF draws require seeds 2028..2051; no files written")
        return
    out = within(root, args.out)
    if out == original or out.is_relative_to(original) or original.is_relative_to(out):
        raise RuntimeError("STOP: corrected destination must be separate from original Stage 6")
    cfgpath = out / "step6_run_config.json"
    identity = {
        **ctx["identity"],
        "seed_correction_schema": SEEDFIX_SCHEMA,
        "rf_permutation_seed_rule": "2026 + split_id",
        "original_step6_path": src["original_path"],
        "original_step6_config_sha256": src["original_step6_config_sha256"],
        "original_step6_aggregate_manifest_sha256": src["original_step6_aggregate_manifest_sha256"],
        "original_stage6_script_sha256": src["original_script_sha256"],
    }
    if args.resume:
        if not out.is_dir() or not cfgpath.is_file():
            raise RuntimeError("STOP: cannot resume without corrected run config")
        if read_json(cfgpath).get("identity") != identity:
            raise RuntimeError("STOP: corrected run config/inputs changed; no unsafe resume")
        log(f"SEEDFIX RESUME | {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_new_json(cfgpath, {
            "schema": SEEDFIX_SCHEMA, "identity": identity,
            "advisor_status": "PENDING; PROVISIONAL",
            "created_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        })
        log(f"SEEDFIX NEW RUN | {out}")
    seedfix_prepare_split(out, original, 1, src["markers"][1])
    completed = [1]
    for split_id in range(2, MAX_SPLITS + 1):
        folder = out / f"split_{split_id:03d}"
        if (folder / "split_complete.json").is_file():
            finished_split(folder, split_id)
            validate_unchanged_lr(out, original, split_id)
            completed.append(split_id)
        elif folder.exists() and split_id != min(set(range(2, MAX_SPLITS + 1)) - set(completed)):
            raise RuntimeError("STOP: cannot skip an incompletely corrected split")
    if (out / "step6_summary.json").exists():
        if len(completed) != 25:
            raise RuntimeError("STOP: corrected summary exists but splits are incomplete")
        if not (out / "step6_aggregate_sha256.json").exists():
            log("SEEDFIX AGGREGATE RESUME | summary exists, final manifest incomplete")
            finish_run(out)
            return
        manifest = read_json(out / "step6_aggregate_sha256.json")
        if set(manifest) != {
            "mice_shap_draw_additivity_25_splits.csv",
            "mice_shap_pooled_additivity_25_splits.csv",
            "mice_shap_global_importance_25_splits.csv",
            "step6_summary.json",
        }:
            raise RuntimeError("STOP: corrected aggregate manifest has incorrect files")
        for name, sha in manifest.items():
            checked_artifact(out, name, sha)
        if read_json(out / "step6_summary.json").get("seed_correction_schema") != SEEDFIX_SCHEMA:
            raise RuntimeError("STOP: corrected summary missing required seedfix provenance")
        log("SEEDFIX COMPLETE | existing corrected 25/25 checkpoints and aggregate hashes verified")
        return
    left = args.max_new_splits
    for split_id in range(2, MAX_SPLITS + 1):
        if split_id in completed:
            continue
        if left == 0:
            break
        seedfix_prepare_split(out, original, split_id, src["markers"][split_id])
        log(f"SEEDFIX SHAP START | split={split_id}/25 | corrected RF seed={2026 + split_id}")
        run_one_split(ctx, out, split_id)
        finished_split(out / f"split_{split_id:03d}", split_id)
        validate_unchanged_lr(out, original, split_id)
        completed.append(split_id)
        left -= 1
    if len(completed) == MAX_SPLITS:
        finish_run(out)
    else:
        log(f"SEEDFIX PARTIAL | completed={len(completed)}/25 | re-run with --resume")


if __name__ == "__main__":
    main()
