#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step1 输出完整性")
    p.add_argument("--matrix-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    by_stim = args.matrix_dir / "by_stimulus"
    rows = []
    for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        npz = np.load(d / "roi_repr_matrix_232.npz")
        rois = pd.read_csv(d / "roi_repr_matrix_232_rois.csv")["roi"].astype(str).tolist()
        subs = pd.read_csv(d / "roi_repr_matrix_232_subjects.csv")["subject"].astype(str).tolist()
        stim_order = pd.read_csv(d / "stimulus_order.csv")["stimulus_order"].astype(str).tolist()
        first = np.asarray(npz[rois[0]])
        ok_shape = first.shape == (len(subs), len(stim_order), len(stim_order))
        rows.append({"stimulus_type": d.name, "n_rois": len(rois), "n_subjects": len(subs), "n_stimuli": len(stim_order), "shape_ok": bool(ok_shape)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(args.matrix_dir / "data_check_stage1.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
