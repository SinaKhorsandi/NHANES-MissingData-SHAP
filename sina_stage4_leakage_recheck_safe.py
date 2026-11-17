#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely rerun the existing Stage 4 leakage test in the original project env.

This wrapper does NOT modify the original leakage test or any Stage 3 output.
It redirects only the final aggregate CSV to a brand-new unique folder under:
    output/stage4_mice/local_rechecks/

Run from the project root, after activating the original virtual environment:
    python -u sina_stage4_leakage_recheck_safe.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import stage4_mice_cv as stage4
import stage4_mice_leakage_test as leakage


ROOT = Path(__file__).resolve().parent
STAGE4_DIR = (ROOT / "output" / "stage4_mice").resolve()
NEW_REPORTS_DIR = STAGE4_DIR / "local_rechecks"


def choose_unique_recheck_dir(parent: Path, subdir: str) -> Path:
    """Intercept only the leakage-test output path; never reuse old folders."""
    if parent.resolve() != STAGE4_DIR or subdir != "step3_leakage_test":
        raise RuntimeError("STOP: unexpected output target; refusing to write")

    NEW_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = NEW_REPORTS_DIR / f"leakage_recheck_{stamp}_pid{os.getpid()}"
    destination.mkdir(parents=False, exist_ok=False)
    print(f"NEW, SEPARATE REPORT DIRECTORY: {destination}", flush=True)
    return destination


def main() -> None:
    if not (ROOT / "stage4_mice_leakage_test.py").is_file():
        raise FileNotFoundError("STOP: put this script beside stage4_mice_leakage_test.py")
    print("Stage 4 / step 3: leakage test only. No SHAP, no 25-split MICE run.", flush=True)
    print("Existing Stage 3 and Stage 4 results will NOT be overwritten.", flush=True)
    with patch.object(stage4, "choose_output", choose_unique_recheck_dir):
        leakage.main()


if __name__ == "__main__":
    main()
