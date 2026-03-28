#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from ..calc_roi_isc_by_age import compute_isc, flatten_upper, load_subject_ages_map, sort_subjects_by_age  # noqa: E402

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute subject-by-subject behavior similarity matrices sorted by age")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--subject-info", type=Path, required=True)
    p.add_argument("--repr-prefix", type=str, default="behavior_repr_matrix_trial")
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--isc-prefix", type=str, default=None)
    return p.parse_args()


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
    )
    return all(p.exists() for p in required)


def run_one_stimulus(stim_dir: Path, subject_info: Path, repr_prefix: str, isc_method: str, isc_prefix: str) -> Dict[str, object]:
    npz = np.load(stim_dir / f"{repr_prefix}.npz")
    subjects = pd.read_csv(stim_dir / f"{repr_prefix}_subjects.csv")["subject"].astype(str).tolist()
    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)

    vecs = np.stack([flatten_upper(np.asarray(npz[s], dtype=np.float32)) for s in subjects_sorted], axis=0)
    isc = compute_isc(vecs, method=str(isc_method))
    np.save(stim_dir / f"{isc_prefix}.npy", isc.astype(np.float32))
    pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted}).to_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"isc_method": [str(isc_method)], "repr_prefix": [str(repr_prefix)], "isc_prefix": [str(isc_prefix)]}).to_csv(
        stim_dir / f"{isc_prefix}_meta.csv",
        index=False,
    )
    return {"stimulus_type": stim_dir.name, "n_subjects": int(len(subjects_sorted)), "isc_method": str(isc_method), "isc_prefix": str(isc_prefix)}


def run(matrix_dir: Path, stimulus_dir_name: str, subject_info: Path, repr_prefix: str, isc_method: str, isc_prefix: str) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_repr_files(stim_dir, repr_prefix=str(repr_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing {repr_prefix} input files")
            continue
        rows.append(run_one_stimulus(stim_dir, subject_info, repr_prefix, isc_method=str(isc_method), isc_prefix=str(isc_prefix)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(Path(matrix_dir) / f"{isc_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    isc_prefix = str(args.isc_prefix) if args.isc_prefix is not None else f"behavior_isc_{str(args.isc_method)}_by_age"
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        subject_info=Path(args.subject_info),
        repr_prefix=str(args.repr_prefix),
        isc_method=str(args.isc_method),
        isc_prefix=str(isc_prefix),
    )


if __name__ == "__main__":
    main()
