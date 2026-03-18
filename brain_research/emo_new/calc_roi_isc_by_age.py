#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_roi_isc_by_age.py

执行顺序（新链路 Step 2/3）：
1) build_roi_beta_matrix_200.py
2) 本脚本：calc_roi_isc_by_age.py
3) joint_analysis_roi_isc_dev_models_perm_fwer.py

读取 build_roi_beta_matrix_200.py 输出的 roi_beta_matrix_200.npz，
按年龄排序被试后，对每个 ROI 计算 Spearman 被试×被试相似性矩阵，输出 roi_isc_spearman_by_age.*。
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

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 roi_beta_matrix_200.npz 计算 ROI 级 Spearman ISC（按龄排序）")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--subject-info", type=Path, default=Path(os.environ.get("SUBJECT_AGE_TABLE", "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")))
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
    exact_age = years + (months / 12.0) + (days / 365.0)
    return float(round(exact_age))


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    path_str = str(subject_info_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

    col_sub_id = "sub_id"
    col_age = "age"
    legacy_col_sub_id = "被试编号"
    legacy_col_age = "采集年龄"
    if col_sub_id in info_df.columns and col_age in info_df.columns:
        sub_col, age_col = col_sub_id, col_age
    elif legacy_col_sub_id in info_df.columns and legacy_col_age in info_df.columns:
        sub_col, age_col = legacy_col_sub_id, legacy_col_age
    else:
        raise ValueError(f"年龄表缺少必要列: {info_df.columns}")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = f"sub-{raw_id}" if not raw_id.startswith("sub-") else raw_id
        age_map[sub_id] = parse_chinese_age(row[age_col])
    return age_map


def load_subject_list(subjects_csv: Path) -> List[str]:
    df = pd.read_csv(subjects_csv)
    if "subject" in df.columns:
        return df["subject"].astype(str).tolist()
    if "sub_id" in df.columns:
        return df["sub_id"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def load_rois(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "roi" in df.columns:
        return df["roi"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def sort_subjects_by_age(subjects: List[str], age_map: Dict[str, float]) -> Tuple[List[str], List[float]]:
    def sort_key(sub_id: str) -> Tuple[float, str]:
        age = age_map.get(sub_id, np.nan)
        if not np.isfinite(age):
            return 999.0, sub_id
        return float(age), sub_id

    subjects_sorted = sorted(subjects, key=sort_key)
    ages_sorted = [float(age_map.get(s, 999.0)) for s in subjects_sorted]
    return subjects_sorted, ages_sorted


def spearman_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / float(p - 1)
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(np.float32, copy=False)


def run_one_stimulus(stim_dir: Path, subject_info: Path) -> None:
    beta_prefix = "roi_beta_matrix_200"
    beta_path = stim_dir / f"{beta_prefix}.npz"
    subjects_path = stim_dir / f"{beta_prefix}_subjects.csv"
    rois_path = stim_dir / f"{beta_prefix}_rois.csv"

    if not beta_path.exists():
        raise FileNotFoundError(f"缺少文件: {beta_path}")
    subjects = load_subject_list(subjects_path)
    rois = load_rois(rois_path)

    npz = np.load(beta_path)
    for r in rois:
        if r not in npz.files:
            raise ValueError(f"npz 缺少 ROI 键: {r}")

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    index_map = {s: i for i, s in enumerate(subjects)}
    sorted_indices = [index_map[s] for s in subjects_sorted]

    n_roi = len(rois)
    n_sub = len(subjects_sorted)
    isc = np.zeros((n_roi, n_sub, n_sub), dtype=np.float32)

    for ri, roi in enumerate(rois):
        X = np.asarray(npz[roi], dtype=np.float32)
        if X.shape[0] != len(subjects):
            raise ValueError(f"ROI 矩阵被试维度不一致: {roi} {X.shape[0]} vs {len(subjects)}")
        Xs = X[sorted_indices, :]
        isc[ri] = spearman_corr_matrix_rows(Xs)

    out_prefix = "roi_isc_spearman_by_age"
    stim_dir.mkdir(parents=True, exist_ok=True)
    np.save(stim_dir / f"{out_prefix}.npy", isc)
    pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted}).to_csv(
        stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False
    )
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)


def run(matrix_dir: Path, subject_info: Path) -> None:
    by_stim_dir = matrix_dir / "by_stimulus"
    if not by_stim_dir.exists():
        raise FileNotFoundError(f"缺少目录: {by_stim_dir}")

    stim_dirs = sorted([p for p in by_stim_dir.iterdir() if p.is_dir()])
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim_dir} 下未找到条件子目录")

    records = []
    for stim_dir in stim_dirs:
        run_one_stimulus(stim_dir, subject_info)
        sub_path = stim_dir / "roi_isc_spearman_by_age_subjects_sorted.csv"
        n_subjects = int(pd.read_csv(sub_path).shape[0])
        records.append({"stimulus_type": stim_dir.name, "n_subjects": n_subjects})

    pd.DataFrame(records).sort_values("stimulus_type").to_csv(
        matrix_dir / "roi_isc_spearman_by_age_by_stimulus_summary.csv", index=False
    )


def main() -> None:
    args = parse_args()
    run(args.matrix_dir, args.subject_info)


if __name__ == "__main__":
    main()
