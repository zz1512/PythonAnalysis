#!/usr/bin/env python3
"""Condition tests for learning-stage behavior proxies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [FINAL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lr_utils import fit_gee_by_roi, write_tsv  # noqa: E402


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping")
    args = parser.parse_args()
    path = args.out_dir / "learning_behavior_item.tsv"
    frame = pd.read_csv(path, sep="\t")
    for col in ["run3_understand_yes", "run4_like_yes"]:
        if col in frame.columns:
            frame[col] = frame[col].map(
                {
                    True: 1.0,
                    False: 0.0,
                    "True": 1.0,
                    "False": 0.0,
                    "1.0": 1.0,
                    "0.0": 0.0,
                    1: 1.0,
                    0: 0.0,
                    1.0: 1.0,
                    0.0: 0.0,
                }
            )
    # Reuse ROI model helper by adding a synthetic ROI. This keeps output schema consistent.
    frame["roi_set"] = "behavior"
    frame["roi"] = "learning_behavior"
    specs = [
        ("run3_rt_condition", "run3_rt_z_subject_condition ~ C(condition, Treatment(reference='kj'))", "run3_rt_z_subject_condition", "gaussian"),
        ("run4_rt_condition", "run4_rt_z_subject_condition ~ C(condition, Treatment(reference='kj'))", "run4_rt_z_subject_condition", "gaussian"),
        ("fluency_shift_condition", "learning_fluency_shift ~ C(condition, Treatment(reference='kj'))", "learning_fluency_shift", "gaussian"),
        ("run3_understand_condition", "run3_understand_yes ~ C(condition, Treatment(reference='kj'))", "run3_understand_yes", "binomial"),
        ("run4_like_condition", "run4_like_yes ~ C(condition, Treatment(reference='kj'))", "run4_like_yes", "binomial"),
    ]
    rows = []
    for model, formula, outcome, family in specs:
        rows.append(fit_gee_by_roi(frame, formula=formula, outcome=outcome, model_name=model, family=family, min_rows=80))
    out = pd.concat(rows, ignore_index=True)
    write_tsv(out, args.out_dir / "learning_behavior_models.tsv")
    summary = {"script": str(Path(__file__).resolve()), "n_rows": int(len(out))}
    (args.out_dir / "learning_behavior_models_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
