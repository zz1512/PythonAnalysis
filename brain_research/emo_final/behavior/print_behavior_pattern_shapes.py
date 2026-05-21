#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print behavior pattern shapes used for subject-by-subject behavior ISC")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--pattern-prefix", type=str, default="behavior_patterns_trial")
    p.add_argument("--conditions", nargs="*", default=None, help="Condition/stimulus_type directory names. Default: first 3.")
    p.add_argument("--n-conditions", type=int, default=3)
    p.add_argument("--n-subjects", type=int, default=5, help="Print this many subject examples per condition.")
    return p.parse_args()


def choose_conditions(root: Path, requested: List[str] | None, n_conditions: int) -> List[Path]:
    if requested:
        out = [root / str(x) for x in requested]
    else:
        out = sorted([p for p in root.iterdir() if p.is_dir()])[: int(n_conditions)]
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing condition directories: {[str(p) for p in missing]}")
    return out


def print_one_condition(stim_dir: Path, pattern_prefix: str, n_subjects: int) -> None:
    npz_path = stim_dir / f"{pattern_prefix}.npz"
    subjects_path = stim_dir / f"{pattern_prefix}_subjects.csv"
    features_path = stim_dir / f"{pattern_prefix}_features.csv"
    order_path = stim_dir / "stimulus_order.csv"

    for path in [npz_path, subjects_path, features_path, order_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    npz = np.load(npz_path)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    features = pd.read_csv(features_path)["feature"].astype(str).tolist()
    stim_order = pd.read_csv(order_path)["stimulus_order"].astype(str).tolist()

    shapes = {}
    for sub in subjects:
        arr = np.asarray(npz[str(sub)], dtype=np.float32)
        shapes[tuple(arr.shape)] = shapes.get(tuple(arr.shape), 0) + 1

    print("=" * 88)
    print(f"condition: {stim_dir.name}")
    print(f"n_subjects: {len(subjects)}")
    print(f"n_trials_in_stimulus_order: {len(stim_order)}")
    print(f"features ({len(features)}): {features}")
    print(f"unique subject pattern shapes: {shapes}")
    print("")
    print("examples:")
    for sub in subjects[: int(n_subjects)]:
        arr = np.asarray(npz[str(sub)], dtype=np.float32)
        vec = arr.reshape(-1)
        n_missing = int(np.sum(~np.isfinite(vec)))
        print(
            f"  {sub}: pattern_shape={tuple(arr.shape)}, "
            f"flatten_shape={tuple(vec.shape)}, missing={n_missing}/{vec.size}"
        )


def main() -> None:
    args = parse_args()
    root = Path(args.matrix_dir) / str(args.stimulus_dir_name)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find stimulus root: {root}")

    for stim_dir in choose_conditions(root, args.conditions, int(args.n_conditions)):
        print_one_condition(stim_dir, pattern_prefix=str(args.pattern_prefix), n_subjects=int(args.n_subjects))


if __name__ == "__main__":
    main()
