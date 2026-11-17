#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 4, step 5: 25-split MICE performance CV (NO SHAP).

This is a separate, provisional runner. It imports the original Stage 3 and
Stage 4 implementation instead of editing either one. It NEVER writes to
stage3_final or stage3_followup and NEVER exports participant-level records.

Modes:
  --mode preflight : read-only validation; no training, no output directory
  --mode run       : train 25 splits, or at most --max-new-splits new splits

Safety:
 * Requires the original split manifest, exact input hash and original library
   versions, and the successful saved FULL split-1 MICE/SHAP benchmark.
 * Creates only a NEW output directory under output/stage4_mice/.
 * Supports explicit --resume of a matching run; validates saved file SHA-256.
 * Saves each split as an atomic checkpoint, then generates aggregate CSVs.
 * Rechecks split-1 MICE AUC/Brier AND chosen parameters against the full
   benchmark BEFORE running split 2.
 * Saves five trained classifiers for each model/split, plus five fitted MICE
   bundles, locally for use in step 6. Do not publish these model artifacts.

Protocol approval status: pending; outputs are PROVISIONAL.

Example (read-only):
  python -u stage4_mice_cv25.py --mode preflight --benchmark-dir \
    output/stage4_mice/local_rechecks/benchmark_.../step4_timing_split1

There is deliberately no default output path for --mode run: the user must
explicitly choose a NEW path within output/stage4_mice/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
# Avoid creating/updating __pycache__ files when importing original study code.
sys.dont_write_bytecode = True
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score

import stage3_cv as s3
import stage4_environment_gate as env_gate
import stage4_mice_cv as s4

N_SPLITS = 25
MODELS = ("LR", "RF")
MICE_DRAWS = 5
METHOD = "MICE m=5"
RUN_SCHEMA = "sina-stage4-step5-v1"
BENCHMARK_AUC_BRIER_DECIMALS = 6


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{stamp()} | {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(s3.json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_new_text(path: Path, value: str) -> None:
    # x prevents accidental replacement even within the new run directory.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def save_new_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"STOP: refusing to replace {path}")
    if not rows:
        raise ValueError(f"STOP: no data for {path.name}")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.16g")


def project_path(root: Path, value: Path) -> Path:
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def require_within_stage4(root: Path, value: Path) -> Path:
    stage4 = (root / "output" / "stage4_mice").resolve()
    resolved = project_path(root, value)
    if not resolved.is_relative_to(stage4) or resolved == stage4:
        raise RuntimeError("STOP: path must be a NEW child of output/stage4_mice/")
    return resolved


def read_benchmark(root: Path, path_arg: Path, inp: s4.Inputs) -> tuple[dict, dict[str, dict], Path]:
    bench = require_within_stage4(root, path_arg)
    js = bench / "mice_split1_benchmark_summary.json"
    csv = bench / "mice_split1_pooled_performance.csv"
    shap_csv = bench / "mice_split1_shap_additivity_and_runtime.csv"
    for p in (js, csv, shap_csv):
        if not p.is_file():
            raise FileNotFoundError(f"STOP: missing benchmark evidence {p}")
    summary = json.loads(js.read_text(encoding="utf-8"))
    necessary = {
        "split_id": 1,
        "n_total_splits": 25,
        "n_train": 4078,
        "n_test_common": 809,
        "background_n": 100,
        "n_imputations": 5,
        "n_models": 2,
        "n_model_draw_shap_runs": 10,
        "time_gate": "PASS",
    }
    for k, expected in necessary.items():
        if summary.get(k) != expected:
            raise RuntimeError(f"STOP: benchmark {k}={summary.get(k)!r} expected {expected!r}")
    if not 0 < float(summary["estimated_25_split_hours_linear"]) < 12.0:
        raise RuntimeError("STOP: benchmark did not meet the advisor's <12h threshold")
    if str(summary.get("approval_status")) not in ("PENDING", "APPROVED"):
        raise RuntimeError("STOP: benchmark approval metadata invalid")

    metrics = pd.read_csv(csv)
    required_metrics = {"split_id", "model", "method", "roc_auc", "brier_score", "params_json"}
    if not required_metrics.issubset(metrics.columns):
        raise RuntimeError("STOP: benchmark metrics columns missing")
    if len(metrics) != 2 or set(metrics["model"]) != set(MODELS):
        raise RuntimeError("STOP: benchmark must contain exactly LR and RF pooled results")
    if not metrics["split_id"].eq(1).all() or not metrics["method"].eq(METHOD).all():
        raise RuntimeError("STOP: benchmark is not original split 1 MICE m=5")
    if not np.isfinite(metrics[["roc_auc", "brier_score"]].to_numpy(dtype=float)).all():
        raise RuntimeError("STOP: benchmark metrics contain nonfinite values")
    shap = pd.read_csv(shap_csv)
    if len(shap) != 10 or not set(MODELS).issubset(set(shap["model"])):
        raise RuntimeError("STOP: benchmark SHAP audit must have 10 model x draw rows")
    if not shap["status"].eq("PASS").all():
        raise RuntimeError("STOP: benchmark SHAP audit contains failed rows")
    if not (shap["max_abs_additivity_error"].to_numpy(dtype=float) < 1e-8).all():
        raise RuntimeError("STOP: benchmark SHAP additivity gate failed")

    Xtr, _, _, Xte, _ = s4.get_split(inp, 1)
    if len(Xtr) != summary["n_train"] or len(Xte) != summary["n_test_common"]:
        raise RuntimeError("STOP: current split 1 differs from benchmark population")
    by_model = {str(r["model"]): r.to_dict() for _, r in metrics.iterrows()}
    hashes = {p.name: sha256_file(p) for p in (js, csv, shap_csv)}
    return {"summary": summary, "sha256": hashes}, by_model, bench


def preflight(root: Path, benchmark_arg: Path) -> tuple[s4.Inputs, dict, dict[str, dict], Path, dict]:
    s4.ensure_revision_branch(root)
    readiness = env_gate.inspect(root)
    if readiness["status"] != "PASS":
        raise RuntimeError("STOP: original-version/reproduction/leakage gate failed: "
                           + json.dumps(readiness, sort_keys=True))
    inp = s4.read_inputs(root, root / "output" / "stage3_final")
    if s4.SEED != 2026 or s4.M != MICE_DRAWS or s4.MICE_MAX_ITER != 10:
        raise RuntimeError("STOP: MICE protocol settings were changed")
    if set(s3.MODELS) != set(MODELS) or list(s3.PREDICTORS) != list(inp.X.columns):
        raise RuntimeError("STOP: Stage 3 model or predictor structure changed")
    benchmark, bench_models, bench_path = read_benchmark(root, benchmark_arg, inp)
    n_tests = []
    for split in range(1, N_SPLITS + 1):
        xtr, _, ctrain, xte, _ = s4.get_split(inp, split)
        if xte.isna().any().any() or not 0 < int(ctrain.sum()) < len(xtr):
            raise RuntimeError(f"STOP: invalid training/test completeness in split {split}")
        n_tests.append(len(xte))
    source_files = ("stage3_cv.py", "stage4_mice_cv.py",
                    "stage4_environment_gate.py", Path(__file__).name)
    source_hashes = {name: sha256_file(root / name) for name in source_files}
    identity = {
        "schema": RUN_SCHEMA,
        "dataset_sha256": sha256_file(root / "output/nhanes_examined_clean_with_missing.csv"),
        "manifest_sha256": sha256_file(root / "output/stage3_final/split_manifest_final.csv"),
        "original_cv_metrics_sha256": sha256_file(root / "output/stage3_final/cv_metrics_final.csv"),
        "source_code_sha256": source_hashes,
        "benchmark_dir": str(bench_path),
        "benchmark_file_sha256": benchmark["sha256"],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "advisor_approval_status": "PENDING",
    }
    log("PREFLIGHT PASS | original environment, dataset SHA256, original 25 splits, "
        "baseline 6/6, leakage 3/3")
    log("PREFLIGHT PASS | recorded FULL split-1 MICE+SHAP benchmark <12h "
        f"({float(benchmark['summary']['estimated_25_split_hours_linear']):.3f}h)")
    log("PREFLIGHT PASS | benchmark split 1: "
        + "; ".join(f"{m}: AUC={float(bench_models[m]['roc_auc']):.6f}, "
                    f"Brier={float(bench_models[m]['brier_score']):.6f}" for m in MODELS))
    log(f"PREFLIGHT PASS | 25 test sizes range {min(n_tests)}..{max(n_tests)}; "
        "NO models trained, NO files written")
    return inp, benchmark, bench_models, bench_path, identity


def moment(array: np.ndarray) -> tuple[float, float]:
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("STOP: diagnostic distribution unavailable/nonfinite")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def diagnostics_for_draw(split: int, j: int, bundle: dict, raw: pd.DataFrame,
                         imputed: pd.DataFrame, seconds: float) -> dict[str, Any]:
    n_iter = int(bundle["mice"].n_iter_)
    if not 1 <= n_iter <= s4.MICE_MAX_ITER:
        raise RuntimeError("STOP: missing/invalid n_iter_ for fitted imputer")
    row: dict[str, Any] = {
        "split_id": split, "imputation_id": j, "seed": int(bundle["seed"]),
        "n_iter_": n_iter, "max_iter": s4.MICE_MAX_ITER,
        "early_stop": int(n_iter < s4.MICE_MAX_ITER),
        "imputer_seconds": float(seconds), "n_train": len(raw),
    }
    for column in ("BMXWAIST", "INDFMPIR"):
        raw_col = raw[column].to_numpy(dtype=float)
        filled = imputed[column].to_numpy(dtype=float)
        observed_mask = np.isfinite(raw_col)
        missing_mask = ~observed_mask
        observed_mean, observed_sd = moment(raw_col[observed_mask])
        imputed_mean, imputed_sd = moment(filled[missing_mask])
        row.update({
            f"{column}_observed_n": int(observed_mask.sum()),
            f"{column}_imputed_n": int(missing_mask.sum()),
            f"{column}_observed_mean": observed_mean,
            f"{column}_observed_sd": observed_sd,
            f"{column}_imputed_mean": imputed_mean,
            f"{column}_imputed_sd": imputed_sd,
        })
    correlation = float(np.corrcoef(
        imputed["BMXWAIST"].to_numpy(dtype=float),
        imputed["BMXBMI"].to_numpy(dtype=float),
    )[0, 1])
    if not np.isfinite(correlation):
        raise RuntimeError("STOP: nonfinite BMI-waist post-MICE correlation")
    row["BMXWAIST_BMXBMI_pearson_after_imputation"] = correlation
    return row


def compare_split_one(rows: list[dict[str, Any]], refs: dict[str, dict]) -> None:
    for row in rows:
        model = row["model"]
        ref = refs[model]
        exact = all(
            f"{float(row[key]):.{BENCHMARK_AUC_BRIER_DECIMALS}f}"
            == f"{float(ref[key]):.{BENCHMARK_AUC_BRIER_DECIMALS}f}"
            for key in ("roc_auc", "brier_score")
        )
        selected = json.loads(row["best_params_json"]) == json.loads(ref["params_json"])
        log(f"SPLIT 1 CONTROL | {model} | AUC={row['roc_auc']:.6f} "
            f"Brier={row['brier_score']:.6f} | metrics_match={exact} params_match={selected}")
        if not exact or not selected:
            raise RuntimeError(f"STOP: split-1 {model} differs from FULL benchmark; "
                               "do not advance to split 2")


def run_split(inp: s4.Inputs, split_id: int, work: Path,
              bench_models: dict[str, dict]) -> None:
    started = time.perf_counter()
    Xtr, ytr, complete_tr, Xte, yte = s4.get_split(inp, split_id)
    raw_training = Xtr.copy()
    (work / "models").mkdir()
    imputed_draws: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for j in range(1, MICE_DRAWS + 1):
        t = time.perf_counter()
        seed = s4.SEED + split_id * 100 + j
        bundle, Xfilled = s4.fit_mice_draw(Xtr, seed)
        fit_seconds = time.perf_counter() - t
        if not np.isfinite(Xfilled[s3.NUMERIC].to_numpy(dtype=float)).all():
            raise RuntimeError(f"STOP: split {split_id}, MICE draw {j} produced NaN/Inf")
        diag = diagnostics_for_draw(split_id, j, bundle, raw_training, Xfilled, fit_seconds)
        diagnostics.append(diag)
        imputed_draws.append(Xfilled)
        model_file = work / "models" / f"MICE_draw{j:02d}_imputer_bundle.joblib"
        joblib.dump(bundle, model_file, compress=3)
        log(f"SPLIT {split_id:02d}/25 | MICE draw {j}/5 | seed={seed} "
            f"n_iter={diag['n_iter_']} | {fit_seconds:.1f}s")

    for model in MODELS:
        t_tune = time.perf_counter()
        params, candidates = s4.tune_mice_first_draw(
            Xtr, ytr, complete_tr, model, split_id,
        )
        tune_seconds = time.perf_counter() - t_tune
        candidate_rows = candidates.to_dict(orient="records")
        for r in candidate_rows:
            r["method"] = METHOD
        tuning_rows.extend(candidate_rows)
        predictions: list[np.ndarray] = []
        fit_total = 0.0
        for j, Xfilled in enumerate(imputed_draws, start=1):
            tfit = time.perf_counter()
            fitted = s3.build_pipeline("CCA", model, params, s4.SEED + split_id)
            fitted.fit(Xfilled, ytr)
            fit_total += time.perf_counter() - tfit
            p = np.asarray(fitted.predict_proba(Xte)[:, 1], dtype=float)
            if p.shape != (len(Xte),) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                raise ValueError(f"STOP: invalid prediction split {split_id}/{model}/draw {j}")
            predictions.append(p)
            model_path = work / "models" / f"{model}_draw{j:02d}_classifier.joblib"
            joblib.dump(fitted, model_path, compress=3)
        pooled = np.mean(np.stack(predictions, axis=0), axis=0)
        auc = float(roc_auc_score(yte, pooled))
        brier = float(brier_score_loss(yte, pooled))
        if not np.isfinite([auc, brier]).all():
            raise RuntimeError("STOP: nonfinite pooled performance")
        metric_rows.append({
            "split_id": split_id,
            "repeat": (split_id - 1) // 5 + 1,
            "fold": (split_id - 1) % 5 + 1,
            "model": model, "method": METHOD,
            "n_train": len(ytr), "n_test_common": len(yte),
            "n_imputations": MICE_DRAWS,
            "roc_auc": auc, "brier_score": brier,
            "best_params_json": json.dumps(s3.json_safe(params), sort_keys=True),
            "tuning_seconds": tune_seconds,
            "final_fit_seconds_total": fit_total,
        })
        log(f"SPLIT {split_id:02d}/25 | {model} | AUC={auc:.6f} "
            f"Brier={brier:.6f} | selected={params}")

    if split_id == 1:
        compare_split_one(metric_rows, bench_models)
    save_new_csv(work / "mice_performance.csv", metric_rows)
    save_new_csv(work / "mice_imputer_diagnostics.csv", diagnostics)
    save_new_csv(work / "mice_inner_tuning.csv", tuning_rows)
    expected_filenames = [p for p in sorted(work.rglob("*")) if p.is_file()]
    checksums = {str(p.relative_to(work)): sha256_file(p) for p in expected_filenames}
    if len(metric_rows) != 2 or len(diagnostics) != MICE_DRAWS or len(tuning_rows) != (4 + 9):
        raise RuntimeError("STOP: incorrect completed split row counts")
    if len(checksums) != 3 + 5 + 10:
        raise RuntimeError(f"STOP: expected 18 outputs per split, got {len(checksums)}")
    complete = {
        "schema": RUN_SCHEMA, "split_id": split_id,
        "elapsed_seconds": float(time.perf_counter() - started),
        "n_train": len(ytr), "n_test_common": len(yte),
        "n_imputations": MICE_DRAWS,
        "metrics_rows": len(metric_rows),
        "diagnostic_rows": len(diagnostics),
        "sha256": checksums,
    }
    write_new_text(work / "split_complete.json", json_text(complete))


def verify_completed_split(folder: Path, split_id: int,
                           benchmark_models: dict[str, dict]) -> tuple[list, list, list]:
    marker_path = folder / "split_complete.json"
    if not marker_path.is_file():
        raise RuntimeError(f"STOP: split {split_id} has no completion marker: {folder}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != RUN_SCHEMA or int(marker.get("split_id", -1)) != split_id:
        raise RuntimeError(f"STOP: invalid completion marker for split {split_id}")
    expected_hashes = marker.get("sha256")
    if not isinstance(expected_hashes, dict) or len(expected_hashes) != 18:
        raise RuntimeError(f"STOP: incomplete SHA256 manifest split {split_id}")
    for relative, expected in expected_hashes.items():
        item = (folder / relative).resolve()
        if not item.is_relative_to(folder.resolve()) or not item.is_file():
            raise RuntimeError(f"STOP: missing/outside split checkpoint: {relative}")
        if sha256_file(item) != expected:
            raise RuntimeError(f"STOP: changed output: {item}")
    result = pd.read_csv(folder / "mice_performance.csv")
    diag = pd.read_csv(folder / "mice_imputer_diagnostics.csv")
    tuning = pd.read_csv(folder / "mice_inner_tuning.csv")
    if len(result) != 2 or set(result["model"]) != set(MODELS) or not result["split_id"].eq(split_id).all():
        raise RuntimeError(f"STOP: invalid metric rows for split {split_id}")
    if len(diag) != 5 or not diag["split_id"].eq(split_id).all():
        raise RuntimeError(f"STOP: invalid imputer diagnostics split {split_id}")
    if not np.isfinite(result[["roc_auc", "brier_score"]].to_numpy(dtype=float)).all():
        raise RuntimeError(f"STOP: nonfinite metrics split {split_id}")
    if not (diag["n_iter_"].between(1, 10).all()):
        raise RuntimeError(f"STOP: invalid n_iter split {split_id}")
    if split_id == 1:
        compare_split_one(result.to_dict(orient="records"), benchmark_models)
    return result.to_dict("records"), diag.to_dict("records"), tuning.to_dict("records")


def collect_progress(run_dir: Path, bench_models: dict[str, dict]) -> tuple[list, list, list, list]:
    metrics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    completed: list[int] = []
    for split_id in range(1, N_SPLITS + 1):
        folder = run_dir / f"split_{split_id:03d}"
        if folder.exists():
            rows, diags, candidate_rows = verify_completed_split(folder, split_id, bench_models)
            completed.append(split_id)
            metrics.extend(rows)
            diagnostics.extend(diags)
            tuning.extend(candidate_rows)
    return completed, metrics, diagnostics, tuning


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    parser.add_argument("--benchmark-dir", type=Path, required=True,
                        help="Path to split-1 benchmark's step4_timing_split1 directory")
    parser.add_argument("--out", type=Path, help="NEW output directory below output/stage4_mice/ for --mode run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only an existing, validated run dir")
    parser.add_argument("--max-new-splits", type=int, default=1,
                        help="How many NEW splits to train in this invocation (1..25); DEFAULT=1 for safety")
    args = parser.parse_args()
    if not 1 <= args.max_new_splits <= N_SPLITS:
        parser.error("--max-new-splits must be 1..25")
    if args.mode == "preflight" and (args.out is not None or args.resume):
        parser.error("preflight is read-only: do not pass --out or --resume")
    if args.mode == "run" and args.out is None:
        parser.error("--mode run requires explicit --out inside output/stage4_mice/")

    inp, benchmark, bench_models, bench_path, identity = preflight(root, args.benchmark_dir)
    if args.mode == "preflight":
        log("STEP 5 PREFLIGHT: PASS | read-only | no model training / no output")
        return

    out = require_within_stage4(root, args.out)
    config = out / "run_config.json"
    if args.resume:
        if not out.is_dir() or not config.is_file():
            raise RuntimeError("STOP: --resume requires an existing Stage 5 run_config.json")
        recorded = json.loads(config.read_text(encoding="utf-8"))
        if recorded.get("identity") != identity:
            raise RuntimeError("STOP: resume identity differs (environment/data/code/benchmark)")
        log(f"RESUME | validated run directory {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_new_text(config, json_text({
            "schema": RUN_SCHEMA,
            "identity": identity,
            "created_utc": stamp(),
            "advisor_status": "PENDING; PROVISIONAL",
            "purpose": "Stage 4 step 5: MICE 25-split performance; SHAP excluded",
        }))
        log(f"NEW RUN | {out}")
    completed, metrics, diag, tuning = collect_progress(out, bench_models)
    # The first split is always the correctness gate. Do not proceed to split 2
    # until a locally trained split 1 matches the successful full benchmark.
    for split_id in range(1, N_SPLITS + 1):
        if split_id in completed:
            continue
        if completed and 1 not in completed:
            raise RuntimeError("STOP: cannot resume later splits without split 1 PASS")
        if len(completed) >= N_SPLITS:
            break
        if args.max_new_splits <= 0:
            break
        final_dir = out / f"split_{split_id:03d}"
        if final_dir.exists():
            raise RuntimeError(f"STOP: split directory already exists without valid checkpoint: {final_dir}")
        attempt = out / f".work_split_{split_id:03d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}"
        attempt.mkdir(exist_ok=False)
        log(f"START | split {split_id}/25 | independent provisional MICE performance run")
        run_split(inp, split_id, attempt, bench_models)
        # This rename is atomic on the same filesystem; never overwrite final_dir.
        if final_dir.exists():
            raise FileExistsError(f"STOP: refusing to overwrite existing {final_dir}")
        attempt.rename(final_dir)
        rows, diags, candidate_rows = verify_completed_split(final_dir, split_id, bench_models)
        completed.append(split_id)
        metrics.extend(rows)
        diag.extend(diags)
        tuning.extend(candidate_rows)
        args.max_new_splits -= 1
        log(f"CHECKPOINT PASS | split {split_id}/25 | completed={len(completed)} "
            f"| all artifacts SHA256 verified")

    if len(completed) < N_SPLITS:
        log(f"STEP 5 PARTIAL | completed={len(completed)}/{N_SPLITS} | "
            f"resume using --mode run --resume --out {out} --benchmark-dir {bench_path}")
        return
    if len(metrics) != 50 or len(diag) != 125 or len(tuning) != 25 * (4 + 9):
        raise RuntimeError(f"STOP: invalid full row counts: {len(metrics)}/{len(diag)}/{len(tuning)}")
    if (out / "mice_cv_metrics_25_splits.csv").exists():
        log("STEP 5 PASS | prior aggregate exists; all checkpoints SHA256 verified")
        return
    save_new_csv(out / "mice_cv_metrics_25_splits.csv", metrics)
    save_new_csv(out / "mice_imputer_diagnostics_25_splits.csv", diag)
    save_new_csv(out / "mice_inner_tuning_25_splits.csv", tuning)
    summary = {
        "schema": RUN_SCHEMA,
        "advisor_status": "PENDING; PROVISIONAL",
        "completed_splits": N_SPLITS,
        "models": list(MODELS),
        "mice_draws": MICE_DRAWS,
        "metric_rows": len(metrics),
        "imputer_diagnostic_rows": len(diag),
        "tuning_rows": len(tuning),
        "auc_brier_split_1_match_benchmark": True,
        "has_nonfinite_auc_or_brier": bool(not np.isfinite(pd.DataFrame(metrics)[["roc_auc", "brier_score"]].to_numpy(dtype=float)).all()),
        "no_shap_done_in_step5": True,
    }
    write_new_text(out / "step5_summary.json", json_text(summary))
    hashes = {
        p.name: sha256_file(p)
        for p in sorted(out.glob("mice_*.csv"))
    }
    hashes["step5_summary.json"] = sha256_file(out / "step5_summary.json")
    write_new_text(out / "step5_aggregate_sha256.json", json_text(hashes))
    log("STEP 5 PASS | completed=25/25 | metric_rows=50 | "
        "imputer_records=125 | no NaN/Inf in performance | models saved locally")
    log(f"OUTPUT: {out}")


if __name__ == "__main__":
    main()
