#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA: inspect the ORIGINAL Python environment and independently rerun Stage 3 split 1.

Place this file next to stage3_cv.py in the original NHANES_Stage1_Verified folder.
Run *with the same Python interpreter used in the prior research*:

  python sina_stage3_environment_recheck.py --mode versions
  python sina_stage3_environment_recheck.py --mode reproduce

The two modes are separate: the first never fits a model. The second only
trains six original combinations on Stage 3 split 1. No source or Stage 3
files are changed, and no participant-level outputs or fitted models are saved.

All reports are stored in a NEW timestamped subdirectory under:
  output/stage4_mice/local_rechecks/
This is a PROVISIONAL technical check; it does not count as advisor approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as pkgmeta
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

PROJECT_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn",
    "scipy": "scipy", "joblib": "joblib", "shap": "shap",
    "matplotlib": "matplotlib", "threadpoolctl": "threadpoolctl",
}
ENV_VARS = (
    "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
EXPECTED_DATA_SHA256 = "a017e04e8c9cbbf5d1d5a85e7bbc241629cc8679bdd21b1a3f96cbf0a2338752"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pkg_version(distribution: str) -> str | None:
    try:
        return pkgmeta.version(distribution)
    except pkgmeta.PackageNotFoundError:
        return None


def git_info(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=12,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "--short", "HEAD"),
        "working_tree_status": git("status", "--short"),
    }


def original_metadata(root: Path) -> tuple[dict, dict]:
    meta_path = root / "output" / "stage3_final" / "stage3_cv_metadata.json"
    shap_path = root / "output" / "stage3_final" / "shap" / "stage3_shap_metadata.json"
    base = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    sh = json.loads(shap_path.read_text(encoding="utf-8")) if shap_path.is_file() else {}
    return base, sh


def blas_info() -> list:
    """Useful when reproducing KNN/linear algebra across machines."""
    try:
        from threadpoolctl import threadpool_info
        return threadpool_info()
    except Exception:
        return []


def make_env_report(root: Path) -> dict:
    base, shap_meta = original_metadata(root)
    actual = {key: pkg_version(package) for key, package in PROJECT_PACKAGES.items()}
    expected = dict(base.get("software_versions", {}))
    shap_version = shap_meta.get("software_versions", {}).get("shap")
    if shap_version is not None:
        expected["shap"] = shap_version
    actual_python = platform.python_version()
    mismatches = {}
    if "python" in expected and expected["python"] != actual_python:
        mismatches["python"] = {"original": expected["python"], "current": actual_python}
    for key, reference_version in expected.items():
        if key == "python":
            continue
        if actual.get(key) != reference_version:
            mismatches[key] = {
                "original": reference_version, "current": actual.get(key)
            }
    data_path = root / "output" / "nhanes_examined_clean_with_missing.csv"
    data_hash = sha256(data_path) if data_path.is_file() else None
    expected_data_hash = base.get("input_dataset_sha256", EXPECTED_DATA_SHA256)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage 3 split 1 environmental reproducibility; not a MICE run",
        "advisor_approval": "PENDING",
        "python_executable": sys.executable,
        "python_version": actual_python,
        "python_full": sys.version,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "blas_threadpools": blas_info(),
        "environment_variables": {name: os.environ.get(name) for name in ENV_VARS},
        "original_versions": expected,
        "current_versions": actual,
        "exact_version_match": bool(expected) and not mismatches,
        "version_differences": mismatches,
        "git": git_info(root),
        "stage3_code_exists": (root / "stage3_cv.py").is_file(),
        "stage3_code_sha256": sha256(root / "stage3_cv.py") if (root / "stage3_cv.py").is_file() else None,
        "source_data_exists": data_path.is_file(),
        "source_data_sha256": data_hash,
        "reference_source_sha256": expected_data_hash,
        "source_data_matches_reference": data_hash is not None and (
            data_hash == expected_data_hash == EXPECTED_DATA_SHA256
        ),
        "manifest_exists": (root / "output" / "stage3_final" / "split_manifest_final.csv").is_file(),
        "reference_metrics_exists": (root / "output" / "stage3_final" / "cv_metrics_final.csv").is_file(),
        "preparation_notes_exist": (root / "output" / "preparation_notes.txt").is_file(),
    }


def make_new_report_dir(root: Path) -> Path:
    parent = root / "output" / "stage4_mice" / "local_rechecks"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = parent / f"environment_check_{stamp}_{os.getpid()}"
    dest.mkdir(parents=False, exist_ok=False)
    return dest


def print_version_report(info: dict) -> None:
    print("\n=== ORIGINAL vs CURRENT ENVIRONMENT ===", flush=True)
    print("Python executable:", info["python_executable"])
    print("Git branch:", info["git"]["branch"])
    print("Git HEAD:", info["git"]["head"])
    print("Package           Original          Current           Match")
    print("--------------------------------------------------------------")
    expected = info["original_versions"]
    actual = info["current_versions"]
    for key in ("python", "numpy", "pandas", "scikit_learn", "scipy", "joblib", "shap", "matplotlib", "threadpoolctl"):
        old = expected.get(key, "not recorded")
        new = info["python_version"] if key == "python" else actual.get(key)
        same = "YES" if old == new else ("UNKNOWN" if old == "not recorded" else "NO")
        print(f"{key:<17} {str(old):<17} {str(new):<17} {same}")
    print("Dataset SHA-256 match:", info["source_data_matches_reference"])
    print("25-split manifest file found:", info["manifest_exists"])
    print("Reference metrics file found:", info["reference_metrics_exists"])
    print("Preparation notes found:", info["preparation_notes_exist"])
    if info["version_differences"]:
        print("NOTE: Different versions detected. No package has been changed.")
    print("This check does not train models or change existing study outputs.\n")


def verify_manifest_all_splits(df, X, y, complete, manifest, stage3):
    import numpy as np
    from sklearn.model_selection import RepeatedStratifiedKFold

    expected = RepeatedStratifiedKFold(
        n_splits=stage3.OUTER_SPLITS,
        n_repeats=stage3.OUTER_REPEATS,
        random_state=stage3.BASE_SEED,
    )
    if len(manifest) != len(df) * stage3.OUTER_SPLITS * stage3.OUTER_REPEATS:
        raise RuntimeError("FAIL: 25-split manifest row count differs from reference")
    for split_id, (train_idx, test_idx) in enumerate(expected.split(X, y), start=1):
        part = manifest.loc[manifest["split_id"].eq(split_id)]
        if len(part) != len(df) or part["SEQN"].duplicated().any():
            raise RuntimeError(f"FAIL: manifest split {split_id} has duplicate/missing rows")
        aligned = part.set_index("SEQN").reindex(df["SEQN"])
        if aligned["role"].isna().any():
            raise RuntimeError(f"FAIL: manifest split {split_id} has unrecognized IDs")
        is_test = aligned["role"].eq("test").to_numpy()
        expected_test = np.zeros(len(df), dtype=bool)
        expected_test[test_idx] = True
        if not np.array_equal(is_test, expected_test):
            raise RuntimeError(f"FAIL: manifest split {split_id} has a different test cohort")
        if not np.array_equal(np.flatnonzero(~is_test), train_idx):
            raise RuntimeError(f"FAIL: manifest split {split_id} has a different train cohort")
        common = aligned["included_in_common_complete_case_test"].to_numpy(dtype=bool)
        if not np.array_equal(common, is_test & complete):
            raise RuntimeError(f"FAIL: manifest split {split_id} has a different common test cohort")
    print("PASS: all 25 manifest splits match the original base seed and test masks.", flush=True)


def reproduce(root: Path, out: Path, info: dict) -> bool:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss, roc_auc_score
    import stage3_cv as stage3

    if not info["source_data_matches_reference"]:
        raise RuntimeError("STOP: original input dataset is missing or SHA-256 differs")
    if not all(info[k] for k in ("manifest_exists", "reference_metrics_exists", "preparation_notes_exist")):
        raise RuntimeError("STOP: required Stage 3 files are missing")
    data = root / "output" / "nhanes_examined_clean_with_missing.csv"
    stage3_root = root / "output" / "stage3_final"
    df = pd.read_csv(data)
    df, y_values, complete = stage3.validate_input(df, data, root)
    X = df[stage3.PREDICTORS].copy()
    y = pd.Series(y_values, name=stage3.TARGET)
    manifest = pd.read_csv(stage3_root / "split_manifest_final.csv")
    verify_manifest_all_splits(df, X, y, complete, manifest, stage3)

    reference = pd.read_csv(stage3_root / "cv_metrics_final.csv")
    if len(reference) != 150:
        raise RuntimeError("STOP: Stage 3 reference must have 150 metric rows")
    expected_combos = {(model, method) for model in stage3.MODELS for method in stage3.METHODS}
    ref1 = reference.loc[reference["split_id"].eq(1)]
    if len(ref1) != 6 or set(zip(ref1["model"], ref1["method"])) != expected_combos:
        raise RuntimeError("STOP: Stage 3 split-1 reference combinations are invalid")

    split1 = manifest.loc[manifest["split_id"].eq(1)].set_index("SEQN").reindex(df["SEQN"])
    train_mask = split1["role"].eq("train").to_numpy()
    test_mask = split1["included_in_common_complete_case_test"].to_numpy(dtype=bool)
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    complete_train = complete[train_mask]
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    if X_test.isna().any().any():
        raise RuntimeError("STOP: common test has missing predictors")
    if len(X_train) != 4078 or len(X_test) != 809 or int(complete_train.sum()) != 3222:
        raise RuntimeError("STOP: unexpected first-split cohort sizes")
    print("PASS: split 1 has training=4078, complete train=3222, test=809.", flush=True)

    logger = logging.getLogger("sina-stage3-recheck")
    logger.addHandler(logging.NullHandler())
    summary = []
    csv_path = out / "split1_six_model_reproduction.csv"
    all_good = True
    run_started = time.perf_counter()
    for model in stage3.MODELS:
        for method in stage3.METHODS:
            started = time.perf_counter()
            print(f"RUNNING: split 1 | {model} | {method}", flush=True)
            best_params, inner_score, _rows, _secs = stage3.tune_hyperparameters(
                X_outer_train=X_train,
                y_outer_train=y_train,
                complete_outer_train=complete_train,
                method=method,
                model_name=model,
                split_id=1,
                random_state=stage3.BASE_SEED + 1,
                logger=logger,
            )
            mask = complete_train if method == "CCA" else np.ones(len(y_train), dtype=bool)
            pipeline = stage3.build_pipeline(method, model, best_params, stage3.BASE_SEED + 1)
            pipeline.fit(X_train.loc[mask], y_train.loc[mask])
            probability = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, probability))
            brier = float(brier_score_loss(y_test, probability))
            ref = ref1.loc[ref1["model"].eq(model) & ref1["method"].eq(method)].iloc[0]
            ref_params = json.loads(ref["best_params_json"])
            auc_ok = f"{auc:.6f}" == f"{float(ref['roc_auc']):.6f}"
            brier_ok = f"{brier:.6f}" == f"{float(ref['brier_score']):.6f}"
            params_ok = best_params == ref_params
            status = "PASS" if (auc_ok and brier_ok) else "FAIL"
            all_good = all_good and status == "PASS"
            summary.append({
                "split_id": 1, "model": model, "method": method,
                "status_6_decimals": status,
                "AUC_original": float(ref["roc_auc"]), "AUC_rerun": auc,
                "AUC_match_6_decimals": bool(auc_ok),
                "Brier_original": float(ref["brier_score"]), "Brier_rerun": brier,
                "Brier_match_6_decimals": bool(brier_ok),
                "hyperparameters_original": json.dumps(ref_params, sort_keys=True),
                "hyperparameters_rerun": json.dumps(best_params, sort_keys=True),
                "hyperparameters_equal": bool(params_ok),
                "inner_auc_rerun": float(inner_score),
                "n_train": int(mask.sum()), "n_test": int(len(X_test)),
                "seconds": float(time.perf_counter() - started),
            })
            pd.DataFrame(summary).to_csv(csv_path, index=False, float_format="%.15g")
            print(f"RESULT: {model} / {method}: {status} | AUC={auc:.6f} "
                  f"(reference {float(ref['roc_auc']):.6f}), "
                  f"Brier={brier:.6f} (reference {float(ref['brier_score']):.6f}), "
                  f"params_equal={params_ok}", flush=True)

    result = {
        "status": "PASS" if all_good and len(summary) == 6 else "FAIL",
        "checks_passed": sum(item["status_6_decimals"] == "PASS" for item in summary),
        "checks_total": 6,
        "elapsed_seconds": time.perf_counter() - run_started,
        "reference_path": "output/stage3_final/cv_metrics_final.csv",
        "result_csv": csv_path.name,
        "adviser_approval": "PENDING",
        "notes": "Provisional local reproducibility test. No MICE, SHAP or 25-fold execution.",
    }
    (out / "reproduction_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nFINAL:", result["status"], "|", result["checks_passed"], "/ 6 matched", flush=True)
    print("Aggregate output:", csv_path, flush=True)
    return all_good


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["versions", "reproduce"], default="versions",
                        help="versions: inspect only; reproduce: independently fit six models")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project root containing stage3_cv.py and output/")
    args = parser.parse_args()
    root = args.root.resolve()
    out = make_new_report_dir(root)
    info = make_env_report(root)
    (out / "environment_report.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_version_report(info)
    print("Report folder:", out, flush=True)
    if args.mode == "versions":
        print("DONE: version inspection only. NO models were trained.")
        return
    if not info["stage3_code_exists"]:
        raise SystemExit("STOP: stage3_cv.py not found in project root")
    sys.path.insert(0, str(root))
    try:
        success = reproduce(root, out, info)
    except Exception as exc:
        (out / "execution_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        print("ERROR:", type(exc).__name__, exc, flush=True)
        print("Error report:", out / "execution_error.txt", flush=True)
        raise SystemExit(2) from exc
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
