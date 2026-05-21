#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step1 输出完整性")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--trial-repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument("--emotion-repr-prefix", type=str, default="roi_repr_matrix_232_emotion4")
    p.add_argument("--pattern-prefix", type=str, default="roi_beta_patterns_232")
    return p.parse_args()


def _safe_read_csv(path: Path, col: str) -> list:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"{path} 缺少列: {col}")
    return df[col].astype(str).tolist()


def _check_rsm(npz_path: Path, rois_path: Path, subs_path: Path, order_path: Path) -> dict:
    rois = _safe_read_csv(rois_path, "roi")
    subs = _safe_read_csv(subs_path, "subject")
    stim_order = _safe_read_csv(order_path, "stimulus_order")
    npz = np.load(npz_path)
    if len(rois) == 0:
        raise ValueError(f"{rois_path} roi 为空")
    first = np.asarray(npz[rois[0]])
    ok_shape = first.shape == (len(subs), len(stim_order), len(stim_order))
    diag = np.diagonal(first, axis1=1, axis2=2) if first.ndim == 3 else np.asarray([])
    diag_mean = float(np.nanmean(diag)) if diag.size else float("nan")
    sym_err = float(np.nanmean(np.abs(first - np.swapaxes(first, 1, 2)))) if first.ndim == 3 else float("nan")
    nan_ratio = float(np.mean(~np.isfinite(first))) if first.size else float("nan")
    return {
        "n_rois": int(len(rois)),
        "n_subjects": int(len(subs)),
        "n_stimuli": int(len(stim_order)),
        "shape_ok": bool(ok_shape),
        "diag_mean_first_roi": diag_mean,
        "sym_err_first_roi": sym_err,
        "nan_ratio_first_roi": nan_ratio,
    }


def _check_patterns(stim_dir: Path, pattern_prefix: str, n_subjects: int, n_stimuli: int, n_rois: int) -> dict:
    npz_path = stim_dir / f"{pattern_prefix}.npz"
    subs_path = stim_dir / f"{pattern_prefix}_subjects.csv"
    rois_path = stim_dir / f"{pattern_prefix}_rois.csv"
    sizes_path = stim_dir / f"{pattern_prefix}_roi_feat_sizes.csv"
    meta_path = stim_dir / f"{pattern_prefix}_meta.csv"
    ok_files = all(p.exists() for p in (npz_path, subs_path, rois_path, sizes_path, meta_path))
    if not ok_files:
        return {"pattern_files_ok": False}
    subs = _safe_read_csv(subs_path, "subject")
    rois = _safe_read_csv(rois_path, "roi")
    sizes_df = pd.read_csv(sizes_path)
    ok_sizes = ("n_feat" in sizes_df.columns) and (int(sizes_df.shape[0]) == int(len(rois)))
    npz = np.load(npz_path)
    missing_sub_keys = [s for s in subs if s not in npz]
    pat_ok = True
    max_feat = None
    if subs:
        arr = np.asarray(npz[subs[0]])
        pat_ok = arr.ndim == 3 and arr.shape[0] == int(len(rois)) and arr.shape[1] == int(n_stimuli)
        max_feat = int(arr.shape[2]) if arr.ndim == 3 else None
    return {
        "pattern_files_ok": True,
        "pattern_subjects_ok": bool(int(len(subs)) == int(n_subjects)),
        "pattern_rois_ok": bool(int(len(rois)) == int(n_rois)),
        "pattern_sizes_ok": bool(ok_sizes),
        "pattern_missing_subject_keys": int(len(missing_sub_keys)),
        "pattern_shape_ok_first_subject": bool(pat_ok),
        "pattern_max_feat_first_subject": max_feat if max_feat is not None else "",
    }


def main() -> None:
    args = parse_args()
    rows = []

    def scan_branch(stimulus_dir_name: str, repr_prefix: str, expect_patterns: bool) -> None:
        by_stim = args.matrix_dir / str(stimulus_dir_name)
        if not by_stim.exists():
            return
        for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
            row = {"branch": str(stimulus_dir_name), "stimulus_type": d.name, "repr_prefix": str(repr_prefix)}
            try:
                npz_path = d / f"{repr_prefix}.npz"
                rois_path = d / f"{repr_prefix}_rois.csv"
                subs_path = d / f"{repr_prefix}_subjects.csv"
                order_path = d / "stimulus_order.csv"
                ok_files = all(p.exists() for p in (npz_path, rois_path, subs_path, order_path))
                row["files_ok"] = bool(ok_files)
                if not ok_files:
                    rows.append(row)
                    continue
                row.update(_check_rsm(npz_path, rois_path, subs_path, order_path))
                if bool(expect_patterns):
                    row.update(_check_patterns(d, pattern_prefix=str(args.pattern_prefix), n_subjects=row["n_subjects"], n_stimuli=row["n_stimuli"], n_rois=row["n_rois"]))
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    scan_branch(args.trial_dir_name, args.trial_repr_prefix, expect_patterns=True)
    scan_branch(args.emotion_dir_name, args.emotion_repr_prefix, expect_patterns=False)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["branch", "stimulus_type"])
    out.to_csv(args.matrix_dir / "data_check_stage1.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
