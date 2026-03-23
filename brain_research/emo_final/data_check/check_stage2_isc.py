#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step2 ISC 输出")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--trial-isc-prefix", type=str, default="roi_isc_mahalanobis_by_age")
    p.add_argument("--emotion-isc-prefix", type=str, default="roi_isc_mahalanobis_by_age")
    return p.parse_args()


def _age_sorted_ok(ages: np.ndarray) -> bool:
    a = np.asarray(ages, dtype=float).reshape(-1)
    if a.size <= 1:
        return True
    finite = np.isfinite(a)
    if not finite.any():
        return True
    af = a[finite]
    if af.size <= 1:
        return True
    return bool(np.all(np.diff(af) >= -1e-8))


def main() -> None:
    args = parse_args()
    rows = []

    def scan_branch(stimulus_dir_name: str, isc_prefix: str) -> None:
        by_stim = args.matrix_dir / str(stimulus_dir_name)
        if not by_stim.exists():
            return
        for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
            row = {"branch": str(stimulus_dir_name), "stimulus_type": d.name, "isc_prefix": str(isc_prefix)}
            try:
                npy_path = d / f"{isc_prefix}.npy"
                subs_path = d / f"{isc_prefix}_subjects_sorted.csv"
                rois_path = d / f"{isc_prefix}_rois.csv"
                ok_files = all(p.exists() for p in (npy_path, subs_path, rois_path))
                row["files_ok"] = bool(ok_files)
                if not ok_files:
                    rows.append(row)
                    continue
                isc = np.load(npy_path)
                subs = pd.read_csv(subs_path)
                rois = pd.read_csv(rois_path)
                row["shape"] = str(tuple(isc.shape))
                row["n_subjects"] = int(subs.shape[0])
                row["n_rois"] = int(rois.shape[0])
                diag = np.diagonal(isc, axis1=1, axis2=2)
                row["diag_mean"] = float(np.nanmean(diag))
                sym_err = float(np.nanmean(np.abs(isc - np.swapaxes(isc, 1, 2))))
                row["sym_err"] = sym_err
                row["nan_ratio"] = float(np.mean(~np.isfinite(isc)))
                if "age" in subs.columns:
                    row["age_sorted_ok"] = bool(_age_sorted_ok(subs["age"].to_numpy(dtype=float)))
                    row["age_min"] = float(np.nanmin(subs["age"].to_numpy(dtype=float)))
                    row["age_max"] = float(np.nanmax(subs["age"].to_numpy(dtype=float)))
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    scan_branch(args.trial_dir_name, args.trial_isc_prefix)
    scan_branch(args.emotion_dir_name, args.emotion_isc_prefix)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["branch", "stimulus_type"])
    out.to_csv(args.matrix_dir / "data_check_stage2.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
