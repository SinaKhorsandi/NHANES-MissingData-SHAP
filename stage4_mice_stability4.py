#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Stage 7: four-strategy paired SHAP stability, read-only preflight + safe run.

This script DOES NOT train models, call SHAP, or publish participant-level data.
It reads the original Stage 3 participant-level SHAP output and the corrected
Stage 6 MICE participant-level SHAP output LOCALLY. They must never be shared.

The three historical method pairs are independently recomputed using the
exact Stage 3 formulas and checked to 1e-12. Historical CSV cells are then
preserved literally in the 300-row output; the new 150 MICE comparisons are
calculated on the same participants, 11 original features, and model scale.

Only write to a NEW, separate subdirectory of output/stage4_mice/.
Status: PROVISIONAL; ADVISOR APPROVAL PENDING.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCHEMA = "sina-stage4-step7-v1"
ADVISOR_STATUS = "PENDING; PROVISIONAL"
STAGE3_STABILITY_SOURCE_SHA256 = "2e4760572761e4ad438a26c7ac10bcd33e284934de6f292a433a93322255a35f"
ORIGINAL_150_SHA256 = "aba9994e692836dff87a8bbc2afc08c466eaade5f9b25f3396c0bd3a4d61bde6"
SEEDFIX_SCHEMA = "sina-stage4-step6-seedfix-v1"
METHODS_OLD = ("CCA", "Median/Mode", "KNN k=5")
MICE = "MICE m=5"
MODELS = ("LR", "RF")
PREDICTORS = (
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "BP_SYS_MEAN", "BP_DIA_MEAN", "SMQ020", "PAQ605",
)
PAIRS_OLD = (("CCA", "Median/Mode"), ("CCA", "KNN k=5"), ("Median/Mode", "KNN k=5"))
PAIRS_NEW = (("CCA", MICE), ("Median/Mode", MICE), ("KNN k=5", MICE))
ALL_PAIRS = PAIRS_OLD + PAIRS_NEW
TOL = 1e-8
LEGACY_TOL = 1e-12
GLOBAL_TOL = 1e-10
COLS = (
    "split_id", "repeat", "fold", "model", "method_a", "method_b", "test_n",
    "feature_n", "spearman_rank_correlation", "top5_jaccard", "top5_method_a",
    "top5_method_b", "top5_overlap_n", "mse_shap", "mae_shap", "shap_scale",
)
FLOAT_COLS = (
    "spearman_rank_correlation", "top5_jaccard", "mse_shap", "mae_shap"
)
INT_COLS = ("split_id", "repeat", "fold", "test_n", "feature_n", "top5_overlap_n")


def log(message: str) -> None:
    print(message, flush=True)


def stop(message: str) -> None:
    raise RuntimeError("STOP: " + message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        stop(f"expected a JSON object at {path}")
    return obj


def rooted(root: Path, arg: Path) -> Path:
    value = arg.expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def inside(parent: Path, child: Path) -> bool:
    return child != parent and child.is_relative_to(parent)


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        stop(f"missing or linked input file: {path}")


def verified_file(parent: Path, relative: str, expected: str) -> Path:
    path = parent / relative
    if not inside(parent.resolve(), path.resolve()):
        stop(f"artifact path escapes its source directory: {relative}")
    require_file(path)
    if digest(path) != expected:
        stop(f"SHA256 mismatch: {path}")
    return path


def require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        stop(f"{source} missing columns {sorted(missing)}")


def as_numerics(frame: pd.DataFrame, cols: list[str], source: str) -> None:
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            stop(f"{source}: nonfinite or missing in {col}")


def expected_scale(model: str) -> str:
    return "log_odds" if model == "LR" else "positive_class_probability"


def method_slug(model: str, method: str) -> str:
    # This is verbatim Stage 3 method_slug semantics.
    part = method.lower().replace("/", "_").replace(" ", "_").replace("=", "")
    return f"{model}__{part}"


def matrix(frame: pd.DataFrame, model: str, method: str) -> np.ndarray:
    prefix = method_slug(model, method)
    cols = [f"{prefix}__{feature}" for feature in PREDICTORS]
    require_columns(frame, set(cols), f"private SHAP matrix {model}/{method}")
    values = frame[cols].to_numpy(dtype=np.float64)
    if values.shape != (len(frame), len(PREDICTORS)) or not np.isfinite(values).all():
        stop(f"SHAP matrix has bad dimensions / finite values: {model}/{method}")
    return values


def feature_importance(values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(values), axis=0)


def stable_top5(importance: np.ndarray) -> list[str]:
    # Same stable tie-breaking as original stage3_stability.py.
    order = np.argsort(-importance, kind="stable")
    return [PREDICTORS[i] for i in order[:5]]


def comparison(split_id: int, repeat: int, fold: int, model: str,
               method_a: str, method_b: str, m_a: np.ndarray,
               m_b: np.ndarray) -> dict[str, Any]:
    if m_a.shape != m_b.shape or m_a.shape[1] != 11:
        stop(f"misaligned matrices for {split_id}/{model}/{method_a}/{method_b}")
    importance_a = feature_importance(m_a)
    importance_b = feature_importance(m_b)
    # Exact Stage 3 rank method and scipy call.
    rank_a = pd.Series(importance_a).rank(method="average").to_numpy()
    rank_b = pd.Series(importance_b).rank(method="average").to_numpy()
    rho = float(spearmanr(rank_a, rank_b).statistic)
    top_a = stable_top5(importance_a)
    top_b = stable_top5(importance_b)
    a, b = set(top_a), set(top_b)
    union = a | b
    jaccard = float(len(a & b) / len(union)) if union else float("nan")
    diff = m_a - m_b
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    if not np.isfinite([rho, jaccard, mse, mae]).all():
        stop(f"nonfinite stability metric in split {split_id}/{model}")
    return {
        "split_id": split_id, "repeat": repeat, "fold": fold, "model": model,
        "method_a": method_a, "method_b": method_b, "test_n": len(m_a),
        "feature_n": 11, "spearman_rank_correlation": rho,
        "top5_jaccard": jaccard, "top5_method_a": ";".join(top_a),
        "top5_method_b": ";".join(top_b), "top5_overlap_n": len(a & b),
        "mse_shap": mse, "mae_shap": mae, "shap_scale": expected_scale(model),
    }


def pairkey(row: dict | pd.Series) -> tuple[int, str, str, str]:
    return (int(row["split_id"]), str(row["model"]),
            str(row["method_a"]), str(row["method_b"]))


def verify_legacy(recomputed: list[dict[str, Any]], legacy_rows: list[dict[str, str]]) -> float:
    refs = {pairkey(row): row for row in legacy_rows}
    got = {pairkey(row): row for row in recomputed}
    if len(refs) != 150 or len(got) != 150 or set(refs) != set(got):
        stop("old three-pair comparison keys differ from the 150-row reference")
    worst = 0.0
    for key, reference in refs.items():
        row = got[key]
        for col in COLS:
            if col in FLOAT_COLS:
                diff = abs(float(reference[col]) - float(row[col]))
                worst = max(worst, diff)
                if not math.isfinite(diff) or diff > LEGACY_TOL:
                    stop(f"original 150-row numeric mismatch {key}/{col}: {diff}")
            elif col in INT_COLS:
                if int(reference[col]) != int(row[col]):
                    stop(f"original 150-row integer mismatch {key}/{col}")
            elif str(reference[col]) != str(row[col]):
                stop(f"original 150-row field mismatch {key}/{col}")
    return worst


def read_original_ref(path: Path) -> list[dict[str, str]]:
    require_file(path)
    if digest(path) != ORIGINAL_150_SHA256:
        stop("original Stage 3 150-row reference SHA256 is not the reviewed Stage 3 artifact")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLS:
            stop("original Stage 3 reference column order differs from Stage 3")
        result = list(reader)
    if len(result) != 150 or len({pairkey(row) for row in result}) != 150:
        stop("original Stage 3 reference does not contain 150 unique rows")
    if set((r["method_a"], r["method_b"]) for r in result) != set(PAIRS_OLD):
        stop("original Stage 3 reference method pairs changed")
    return result


def audit_stage3(shapdir: Path) -> pd.DataFrame:
    audit_path = shapdir / "additivity_audit_final.csv"
    require_file(audit_path)
    audit = pd.read_csv(audit_path)
    expected = {(s, m, method) for s in range(1, 26)
                for m in MODELS for method in METHODS_OLD}
    required = {"split_id", "model", "method", "audit_status", "output_scale",
                "n_explained", "background_n", "tolerance",
                "max_abs_additivity_error_log_odds", "max_abs_additivity_error_probability"}
    require_columns(audit, required, "Stage 3 additivity audit")
    keys = {(int(r.split_id), r.model, r.method) for r in audit.itertuples()}
    if len(audit) != 150 or keys != expected:
        stop("Stage 3 additivity audit must contain exactly 150 original combinations")
    if not audit["audit_status"].eq("PASS").all():
        stop("Stage 3 SHAP additivity audit contains non-PASS rows")
    for r in audit.itertuples():
        if (r.output_scale != expected_scale(r.model) or
            int(r.background_n) != 100 or int(r.n_explained) < 1 or
            not float(r.tolerance) == TOL):
            stop(f"Stage 3 audit metadata invalid: split={r.split_id}/{r.model}")
        errfield = ("max_abs_additivity_error_log_odds" if r.model == "LR" else
                    "max_abs_additivity_error_probability")
        err = float(getattr(r, errfield))
        if not (np.isfinite(err) and 0 <= err < TOL):
            stop(f"Stage 3 additivity error violated: split {r.split_id}/{r.model}")
        if r.model == "RF":
            if int(r.permutation_seed) != 2026 + int(r.split_id):
                stop("Stage 3 permutation seed does not follow the protocol")
    return audit


def audit_seedfix(seedfix: Path) -> dict[str, Any]:
    summary_path = seedfix / "step6_summary.json"
    sha_path = seedfix / "step6_aggregate_sha256.json"
    require_file(summary_path)
    require_file(sha_path)
    summary = read_json(summary_path)
    if (summary.get("seed_correction_schema") != SEEDFIX_SCHEMA or
        summary.get("schema") != "sina-stage4-step6-v1" or
        summary.get("advisor_status") != ADVISOR_STATUS or
        summary.get("rf_permutation_seed_rule") != "2026 + split_id" or
        summary.get("rf_draws_corrected_splits_2_to_25") != 120 or
        summary.get("lr_draws_reused_unchanged") != 125 or
        summary.get("split1_rf_draws_reused_unchanged") != 5 or
        summary.get("original_stage6_unmodified") is not True or
        summary.get("n_splits") != 25 or
        summary.get("n_models") != 2 or
        summary.get("n_draws_per_model") != 5 or
        summary.get("n_per_draw_audit_rows") != 250 or
        summary.get("n_pooled_audit_rows") != 50 or
        summary.get("n_global_feature_rows") != 550 or
        summary.get("all_local_additivity_pass") is not True or
        summary.get("all_pooled_additivity_pass") is not True):
        stop("corrected Stage 6 summary lacks completed seedfix provenance")
    manifests = read_json(sha_path)
    required_names = {
        "mice_shap_draw_additivity_25_splits.csv",
        "mice_shap_pooled_additivity_25_splits.csv",
        "mice_shap_global_importance_25_splits.csv",
        "step6_summary.json",
    }
    if set(manifests) != required_names:
        stop("corrected Stage 6 aggregate manifest is incomplete or contains unexpected entries")
    for name, expected_hash in manifests.items():
        verified_file(seedfix, name, expected_hash)
    draw = pd.read_csv(seedfix / "mice_shap_draw_additivity_25_splits.csv")
    pooled = pd.read_csv(seedfix / "mice_shap_pooled_additivity_25_splits.csv")
    global_imp = pd.read_csv(seedfix / "mice_shap_global_importance_25_splits.csv")
    if (len(draw), len(pooled), len(global_imp)) != (250, 50, 550):
        stop("corrected Stage 6 aggregate row counts incorrect")
    require_columns(draw, {"split_id", "model", "imputation_id", "n_test_common",
                           "background_n", "max_abs_additivity_error", "shap_scale", "status"},
                    "corrected Stage 6 per-draw audit")
    require_columns(pooled, {"split_id", "model", "n_imputations", "n_test_common",
                             "background_n", "auc", "brier_score",
                             "pooled_max_abs_additivity_error", "shap_scale", "status"},
                    "corrected Stage 6 pooled audit")
    require_columns(global_imp, {"split_id", "repeat", "fold", "model", "method",
                                 "feature", "mean_abs_shap", "mean_shap", "shap_scale", "rank"},
                    "corrected Stage 6 global importance")
    as_numerics(draw, ["split_id", "imputation_id", "n_test_common", "background_n",
                        "max_abs_additivity_error"], "corrected Stage 6 draw audit")
    as_numerics(pooled, ["split_id", "n_imputations", "n_test_common", "background_n",
                          "auc", "brier_score", "pooled_max_abs_additivity_error"],
                "corrected Stage 6 pooled audit")
    as_numerics(global_imp, ["split_id", "repeat", "fold", "mean_abs_shap", "mean_shap", "rank"],
                "corrected Stage 6 global importance")
    draw_expected = {(s, m, d) for s in range(1, 26) for m in MODELS for d in range(1, 6)}
    pooled_expected = {(s, m) for s in range(1, 26) for m in MODELS}
    glob_expected = {(s, m, f) for s in range(1, 26) for m in MODELS for f in PREDICTORS}
    if ({(int(r.split_id), r.model, int(r.imputation_id)) for r in draw.itertuples()} != draw_expected
        or {(int(r.split_id), r.model) for r in pooled.itertuples()} != pooled_expected
        or {(int(r.split_id), r.model, r.feature) for r in global_imp.itertuples()} != glob_expected):
        stop("corrected Stage 6 aggregate split/model/draw/feature coverage is incomplete")
    if (not draw["status"].eq("PASS").all() or
        not pooled["status"].eq("PASS").all()):
        stop("corrected Stage 6 audit contains non-PASS status")
    if ((draw["max_abs_additivity_error"] >= TOL).any() or
        (draw["max_abs_additivity_error"] < 0).any() or
        (pooled["pooled_max_abs_additivity_error"] >= TOL).any() or
        (pooled["pooled_max_abs_additivity_error"] < 0).any()):
        stop("corrected Stage 6 SHAP additivity failed")
    for m in MODELS:
        if (not draw.loc[draw.model.eq(m), "shap_scale"].eq(expected_scale(m)).all()
            or not pooled.loc[pooled.model.eq(m), "shap_scale"].eq(expected_scale(m)).all()
            or not global_imp.loc[global_imp.model.eq(m), "shap_scale"].eq(expected_scale(m)).all()):
            stop(f"corrected Stage 6 {m} SHAP scale changed")
    if (not global_imp["method"].eq(MICE).all()
        or not pooled["n_imputations"].eq(5).all()
        or not draw["background_n"].eq(100).all()
        or not pooled["background_n"].eq(100).all()
        or (pooled["auc"] < 0).any() or (pooled["auc"] > 1).any()
        or (pooled["brier_score"] < 0).any() or (pooled["brier_score"] > 1).any()):
        stop("corrected Stage 6 aggregate metadata or score constraints violated")
    return {"draw": draw, "pooled": pooled, "global": global_imp,
            "summary_hash": digest(summary_path), "aggregate_hash": digest(sha_path)}


def check_seedfix_split(seedfix: Path, split_id: int, audit: dict[str, Any]) -> Path:
    split = seedfix / f"split_{split_id:03d}"
    if not split.is_dir() or split.is_symlink():
        stop(f"missing or linked corrected Stage 6 split directory: {split}")
    marker_path = split / "split_complete.json"
    require_file(marker_path)
    marker = read_json(marker_path)
    if (marker.get("schema") != "sina-stage4-step6-v1" or
        marker.get("split_id") != split_id or
        marker.get("n_shap_draw_runs") != 10 or
        marker.get("n_background") != 100):
        stop(f"invalid corrected Stage 6 split marker {split_id}")
    expected = {f"draws/{m}_draw{d:02d}.{ext}"
                for m in MODELS for d in range(1, 6) for ext in ("npz", "json")}
    expected |= {"mice_shap_draw_additivity.csv", "mice_shap_pooled_additivity.csv",
                 "mice_shap_global_importance.csv", "shap_values_private.csv"}
    files = marker.get("sha256", {})
    if not isinstance(files, dict) or set(files) != expected:
        stop(f"corrected Stage 6 split {split_id}: 24 artifact manifest entries required")
    for rel, h in files.items():
        verified_file(split, rel, h)
    for d in range(1, 6):
        jpath = split / "draws" / f"RF_draw{d:02d}.json"
        mark = read_json(jpath)
        seed = mark.get("permutation_seed", 2027 if split_id == 1 else None)
        if seed != 2026 + split_id or mark.get("imputation_id") != d or mark.get("model") != "RF":
            stop(f"corrected RF draw seed invalid: split {split_id}, draw {d}")
    for name, aggregate, subset in (
        ("mice_shap_draw_additivity.csv", audit["draw"],
         audit["draw"].loc[audit["draw"].split_id.eq(split_id)]),
        ("mice_shap_pooled_additivity.csv", audit["pooled"],
         audit["pooled"].loc[audit["pooled"].split_id.eq(split_id)]),
        ("mice_shap_global_importance.csv", audit["global"],
         audit["global"].loc[audit["global"].split_id.eq(split_id)]),
    ):
        local = pd.read_csv(split / name)
        ref = subset.reset_index(drop=True)
        if not local.columns.equals(ref.columns) or len(local) != len(ref):
            stop(f"corrected Stage 6 split {split_id}: local {name} schema/rows differ from aggregate")
        for col in local.columns:
            if pd.api.types.is_numeric_dtype(local[col]) and pd.api.types.is_numeric_dtype(ref[col]):
                a, b = local[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float)
                if (not np.isfinite(a).all() or not np.isfinite(b).all()
                    or not np.allclose(a, b, atol=1e-12, rtol=0.0)):
                    stop(f"corrected Stage 6 split {split_id}: local {name}/{col} differs from aggregate")
            elif not local[col].astype(str).equals(ref[col].astype(str)):
                stop(f"corrected Stage 6 split {split_id}: local {name}/{col} differs from aggregate")
    return split / "shap_values_private.csv"


def check_alignment(stage3_file: Path, mice_file: Path, split_id: int,
                    global_rows: pd.DataFrame, audit_stage3: pd.DataFrame,
                    audit_stage6: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(stage3_file)
    require_file(mice_file)
    old = pd.read_csv(stage3_file)
    new = pd.read_csv(mice_file)
    meta = ["split_id", "repeat", "fold", "SEQN", "y_true"]
    require_columns(old, set(meta), f"Stage 3 split {split_id}")
    require_columns(new, set(meta), f"corrected MICE split {split_id}")
    if not len(old) or len(old) != len(new):
        stop(f"split {split_id}: Stage 3 / MICE test size mismatch")
    for col in meta:
        if (old[col].isna().any() or new[col].isna().any() or
            not np.array_equal(old[col].to_numpy(), new[col].to_numpy())):
            stop(f"split {split_id}: Stage 3 / MICE participant/label/order mismatch in {col}")
    if (not old["SEQN"].is_unique or not new["SEQN"].is_unique or
        int(old["split_id"].iloc[0]) != split_id or
        int(old["repeat"].iloc[0]) != (split_id - 1) // 5 + 1 or
        int(old["fold"].iloc[0]) != (split_id - 1) % 5 + 1 or
        set(old["y_true"].unique()) != {0, 1}):
        stop(f"split {split_id}: Stage 3 / corrected MICE test cohort invalid")
    if (not audit_stage3.loc[audit_stage3.split_id.eq(split_id), "n_explained"].eq(len(old)).all()
        or not audit_stage6["pooled"].loc[audit_stage6["pooled"].split_id.eq(split_id),
                                          "n_test_common"].eq(len(old)).all()):
        stop(f"split {split_id}: audit/test participant count disagrees")
    for m in MODELS:
        basecol = f"{method_slug(m, MICE)}__base_value"
        require_columns(new, {basecol}, f"corrected MICE {m} split {split_id}")
        if not np.isfinite(new[basecol].to_numpy(dtype=float)).all():
            stop(f"split {split_id}: nonfinite pooled MICE base for {m}")
        values = matrix(new, m, MICE)
        stats = global_rows.loc[global_rows.model.eq(m)].set_index("feature")
        if len(stats) != 11 or set(stats.index) != set(PREDICTORS):
            stop(f"split {split_id}: missing MICE global feature rows for {m}")
        observed_abs = feature_importance(values)
        observed_signed = np.mean(values, axis=0)
        indexed = stats.loc[list(PREDICTORS)]
        delta_a = np.max(np.abs(indexed["mean_abs_shap"].to_numpy() - observed_abs))
        delta_b = np.max(np.abs(indexed["mean_shap"].to_numpy() - observed_signed))
        if max(delta_a, delta_b) > GLOBAL_TOL:
            stop(f"split {split_id}/{m}: private MICE SHAP does not recreate global importances")
    return old, new


def compute_everything(stage3: Path, seedfix: Path, with_results: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    stability_script = root / "stage3_stability.py"
    require_file(stability_script)
    if digest(stability_script) != STAGE3_STABILITY_SOURCE_SHA256:
        stop("original Stage 3 stability source code differs from reviewed exact formulas")
    old_ref = read_original_ref(stage3 / "shap_stability_metrics_by_split_final.csv")
    old_audit = audit_stage3(stage3)
    new_audit = audit_seedfix(seedfix)
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    tested = 0
    for s in range(1, 26):
        mice_path = check_seedfix_split(seedfix, s, new_audit)
        global_rows = new_audit["global"].loc[new_audit["global"].split_id.eq(s)]
        old, new = check_alignment(
            stage3 / "values_by_split" / f"shap_values_split_{s:03d}.csv",
            mice_path, s, global_rows, old_audit, new_audit)
        repeat = (s - 1) // 5 + 1
        fold = (s - 1) % 5 + 1
        for model in MODELS:
            mats = {method: matrix(old, model, method) for method in METHODS_OLD}
            mats[MICE] = matrix(new, model, MICE)
            for a, b in PAIRS_OLD:
                old_rows.append(comparison(s, repeat, fold, model, a, b, mats[a], mats[b]))
            for a, b in PAIRS_NEW:
                new_rows.append(comparison(s, repeat, fold, model, a, b, mats[a], mats[b]))
        tested += 1
    difference = verify_legacy(old_rows, old_ref)
    if len(new_rows) != 150 or len({pairkey(row) for row in new_rows}) != 150:
        stop("new MICE stability comparisons must contain exactly 150 unique rows")
    if set(pairkey(row) for row in old_ref) & set(pairkey(row) for row in new_rows):
        stop("MICE comparisons overlap historical 150 keys")
    log(f"STEP 7 BASELINE PASS | 150/150 original pair rows independently recomputed; max_abs_diff={difference:.3g}")
    log(f"STEP 7 INPUT PASS | corrected Stage 6 25/25 split SHA256 markers, 125 RF seeds, 250 draw and 50 pooled audits")
    log(f"STEP 7 ALIGNMENT PASS | 25/25 identical Stage 3 and MICE test cohorts; 550 MICE global SHAP cross-checks")
    result: dict[str, Any] = {
        "old": old_ref if with_results else [],
        "new": new_rows if with_results else [],
        "old_diff": difference,
        "stage3_ref_sha256": digest(stage3 / "shap_stability_metrics_by_split_final.csv"),
        "stage3_audit_sha256": digest(stage3 / "additivity_audit_final.csv"),
        "seedfix_summary_sha256": new_audit["summary_hash"],
        "seedfix_manifest_sha256": new_audit["aggregate_hash"],
        "n_splits": tested,
    }
    return result


def write_json_new(path: Path, obj: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def run(stage3: Path, seedfix: Path, out: Path, result: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parent
    stage4_root = (root / "output" / "stage4_mice").resolve()
    if (not inside(stage4_root, out) or
        out.is_relative_to(seedfix) or seedfix.is_relative_to(out) or
        out.is_relative_to(stage3) or stage3.is_relative_to(out)):
        stop("output must be a fresh separate directory INSIDE output/stage4_mice")
    if out.exists():
        stop(f"output already exists; refusing overwrite: {out}")
    if not out.parent.is_dir() or not inside(stage4_root, out.parent):
        stop("output parent must already exist inside output/stage4_mice")
    out.mkdir(parents=False, exist_ok=False)
    cfg = {
        "schema": SCHEMA, "advisor_status": ADVISOR_STATUS,
        "stage3_shap_dir": str(stage3), "corrected_seedfix_dir": str(seedfix),
        "stage3_stability_source_sha256": STAGE3_STABILITY_SOURCE_SHA256,
        "script_sha256": digest(Path(__file__).resolve()),
        "stage3_reference_150_sha256": result["stage3_ref_sha256"],
        "stage3_additivity_audit_sha256": result["stage3_audit_sha256"],
        "corrected_step6_summary_sha256": result["seedfix_summary_sha256"],
        "corrected_step6_aggregate_manifest_sha256": result["seedfix_manifest_sha256"],
        "legacy_numeric_tolerance": LEGACY_TOL,
    }
    write_json_new(out / "step7_run_config.json", cfg)
    combined = [*result["old"], *result["new"]]
    combined.sort(key=pairkey)
    dest = out / "shap_stability_metrics_4_methods_25_splits.csv"
    with dest.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)
    # The historical fields are preserved as CSV text; verify the saved artifact.
    with dest.open("r", encoding="utf-8", newline="") as fh:
        readback = list(csv.DictReader(fh))
    if len(readback) != 300 or len({pairkey(row) for row in readback}) != 300:
        stop("saved Stage 7 CSV has incorrect 300-row key coverage")
    old_src = {pairkey(r): r for r in result["old"]}
    for row in readback:
        key = pairkey(row)
        if key in old_src and row != old_src[key]:
            stop(f"old Stage 3 CSV cells modified in combined output: {key}")
    if sum(pairkey(row) in old_src for row in readback) != 150:
        stop("saved Stage 7 CSV has wrong historical/new 150+150 split")
    meta = {
        "schema": SCHEMA, "advisor_status": ADVISOR_STATUS,
        "n_splits": 25, "n_models": 2, "n_methods": 4, "n_pairs": 6,
        "n_rows": 300, "n_legacy_rows_exactly_preserved": 150,
        "n_new_mice_rows": 150, "n_features": 11,
        "legacy_independent_recomputation_max_abs_diff": result["old_diff"],
        "legacy_comparison_tolerance": LEGACY_TOL,
        "all_shap_additivity_pass": True,
        "all_stage6_corrected_rf_seeds_verified": True,
        "all_test_participants_and_y_labels_aligned": True,
        "participant_level_data_not_in_stage7_outputs": True,
        "methods": [*METHODS_OLD, MICE],
        "pairs": [list(pair) for pair in ALL_PAIRS],
        "models": list(MODELS),
        "metrics": list(FLOAT_COLS),
        "mse_scale_rule": "LR=log_odds; RF=positive_class_probability; never compare across models",
        "jaccard_top_k": 5,
        "source_stage3_reference_sha256": result["stage3_ref_sha256"],
        "source_corrected_stage6_manifest_sha256": result["seedfix_manifest_sha256"],
    }
    write_json_new(out / "step7_summary.json", meta)
    manifests = {name: digest(out / name) for name in (
        "shap_stability_metrics_4_methods_25_splits.csv",
        "step7_run_config.json", "step7_summary.json")}
    write_json_new(out / "step7_aggregate_sha256.json", manifests)
    for name, expected in manifests.items():
        verified_file(out, name, expected)
    log("STEP 7 PASS | 25 splits x 2 models x 6 method pairs = 300 rows | historical 150 unchanged | SHA256 PASS")
    log(f"STEP 7 OUTPUT | {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--stage3-shap-dir", type=Path, required=True)
    parser.add_argument("--seedfix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    stage3 = rooted(root, args.stage3_shap_dir)
    seedfix = rooted(root, args.seedfix_dir)
    if not (stage3.is_dir() and seedfix.is_dir()):
        stop("Stage 3 SHAP or corrected Stage 6 directory is missing")
    if stage3 == seedfix or inside(stage3, seedfix) or inside(seedfix, stage3):
        stop("Stage 3 and corrected Stage 6 source directories must be different")
    if args.mode == "preflight" and args.out is not None:
        stop("preflight is read-only; do not provide --out")
    if args.mode == "run" and args.out is None:
        stop("run mode requires --out for a NEW Stage 7 directory")
    log("STEP 7 START | independently audit original 150 rows and corrected MICE participant alignment")
    result = compute_everything(stage3, seedfix, with_results=args.mode == "run")
    if args.mode == "preflight":
        log("STEP 7 PREFLIGHT: PASS | 300-row calculation ready; read-only, no models/SHAP rerun, no files written")
    else:
        run(stage3, seedfix, rooted(root, args.out), result)


if __name__ == "__main__":
    main()
