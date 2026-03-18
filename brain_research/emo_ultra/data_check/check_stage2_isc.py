#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step2 ISC 输出")
    p.add_argument("--matrix-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for d in sorted([p for p in (args.matrix_dir / "by_stimulus").iterdir() if p.is_dir()]):
        isc = np.load(d / "roi_isc_spearman_by_age.npy")
        subs = pd.read_csv(d / "roi_isc_spearman_by_age_subjects_sorted.csv")
        rois = pd.read_csv(d / "roi_isc_spearman_by_age_rois.csv")
        diag_mean = float(np.mean(np.diagonal(isc, axis1=1, axis2=2)))
        rows.append({"stimulus_type": d.name, "shape": str(tuple(isc.shape)), "n_subjects": int(subs.shape[0]), "n_rois": int(rois.shape[0]), "diag_mean": diag_mean})
    out = pd.DataFrame(rows).sort_values("stimulus_type")
    out.to_csv(args.matrix_dir / "data_check_stage2.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
