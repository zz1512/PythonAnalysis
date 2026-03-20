#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_roi_isc_by_age.py

Step 2:
基于 Step 1 的 ROI 表征矩阵（subject x stim x stim），
计算每个 ROI 的被试×被试相似性矩阵，并按年龄排序被试。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 ROI 表征矩阵计算按龄 ROI ISC")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--subject-info", type=Path, default=Path(os.environ.get("SUBJECT_AGE_TABLE", "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")))
    p.add_argument("--repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument("--isc-method", type=str, default="spearman", choices=("spearman", "pearson"))
    p.add_argument("--isc-prefix", type=str, default=None)
    return p.parse_args()


def parse_chinese_age(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return 999.0
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d_match = re.search(r"(\d+)\s*天", s)
    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    return float(round(years + months / 12.0 + days / 365.0, 4))


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    path_str = str(subject_info_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的年龄表格式")

    if "sub_id" in info_df.columns and "age" in info_df.columns:
        sub_col, age_col = "sub_id", "age"
    elif "被试编号" in info_df.columns and "采集年龄" in info_df.columns:
        sub_col, age_col = "被试编号", "采集年龄"
    else:
        raise ValueError(f"年龄表缺少必要列: {list(info_df.columns)}")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw = str(row[sub_col]).strip()
        if raw.endswith(".0"):
            raw = raw[:-2]
        sid = raw if raw.startswith("sub-") else f"sub-{raw}"
        age_map[sid] = parse_chinese_age(row[age_col])
    return age_map


def sort_subjects_by_age(subjects: List[str], age_map: Dict[str, float]) -> Tuple[List[str], List[float]]:
    def key(s: str) -> Tuple[float, str]:
        a = age_map.get(s, np.nan)
        if not np.isfinite(a):
            return 999.0, s
        return float(a), s

    ss = sorted(subjects, key=key)
    aa = [float(age_map.get(s, 999.0)) for s in ss]
    return ss, aa


def spearman_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def pearson_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (X - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def compute_isc(X: np.ndarray, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    if m == "spearman":
        return spearman_corr_matrix_rows(X)
    if m == "pearson":
        return pearson_corr_matrix_rows(X)
    raise ValueError(f"未知 isc-method: {method}")


def flatten_upper(rsm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rsm.shape[0], k=1)
    return rsm[iu].astype(np.float32)


def run_one_stimulus(stim_dir: Path, subject_info: Path, repr_prefix: str, isc_method: str, isc_prefix: str) -> Dict[str, object]:
    npz_path = stim_dir / f"{repr_prefix}.npz"
    subjects_path = stim_dir / f"{repr_prefix}_subjects.csv"
    rois_path = stim_dir / f"{repr_prefix}_rois.csv"

    npz = np.load(npz_path)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    idx_map = {s: i for i, s in enumerate(subjects)}
    order = [idx_map[s] for s in subjects_sorted]

    n_sub = len(subjects_sorted)
    isc = np.zeros((len(rois), n_sub, n_sub), dtype=np.float32)
    for ri, roi in enumerate(rois):
        arr = np.asarray(npz[roi], dtype=np.float32)
        arr = arr[order, :, :]
        vecs = np.stack([flatten_upper(arr[i]) for i in range(n_sub)], axis=0)
        isc[ri] = compute_isc(vecs, method=str(isc_method))

    out_prefix = str(isc_prefix)
    np.save(stim_dir / f"{out_prefix}.npy", isc)
    pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted}).to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    pd.DataFrame({"isc_method": [str(isc_method)], "repr_prefix": [str(repr_prefix)], "isc_prefix": [str(out_prefix)]}).to_csv(
        stim_dir / f"{out_prefix}_meta.csv",
        index=False,
    )
    return {"stimulus_type": stim_dir.name, "n_subjects": n_sub, "n_rois": len(rois), "isc_method": str(isc_method), "isc_prefix": str(out_prefix)}


def run(matrix_dir: Path, stimulus_dir_name: str, subject_info: Path, repr_prefix: str, isc_method: str, isc_prefix: str) -> None:
    by_stim = matrix_dir / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"未找到目录: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        rows.append(run_one_stimulus(stim_dir, subject_info, repr_prefix, isc_method=str(isc_method), isc_prefix=str(isc_prefix)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(matrix_dir / f"{isc_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    isc_prefix = str(args.isc_prefix) if args.isc_prefix is not None else f"roi_isc_{str(args.isc_method)}_by_age"
    run(args.matrix_dir, args.stimulus_dir_name, args.subject_info, args.repr_prefix, isc_method=str(args.isc_method), isc_prefix=str(isc_prefix))


if __name__ == "__main__":
    main()
