#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_roi_isc_by_age.py

2. 第二步：计算 ROI * 被试 * 被试 的相关矩阵
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def parse_chinese_age(age_str: object) -> float:
    """将年龄字段解析为数值，缺失或异常放到最后。"""
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
    return round(exact_age)


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    """读取被试年龄表，返回 {sub-xxx: age} 映射，兼容多种表格格式与列名。"""
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
    """从 subjects CSV 读取被试列表，自动识别常见列名。"""
    df = pd.read_csv(subjects_csv)
    if "subject" in df.columns:
        return df["subject"].astype(str).tolist()
    if "sub_id" in df.columns:
        return df["sub_id"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def sort_subjects_by_age(subjects: List[str], age_map: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """按年龄升序排序被试，年龄缺失放到末尾。"""
    def sort_key(sub_id: str) -> Tuple[float, str]:
        age = age_map.get(sub_id, np.nan)
        if not np.isfinite(age):
            return (999.0, sub_id)
        return (float(age), sub_id)
    subjects_sorted = sorted(subjects, key=sort_key)
    ages_sorted = [float(age_map.get(s, 999.0)) for s in subjects_sorted]
    return subjects_sorted, ages_sorted


def spearman_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    """对行向量做 Spearman 相关，输出被试×被试矩阵。"""
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / (p - 1)
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def main() -> None:
    """主流程：读取 ROI×被试×刺激矩阵，按年龄排序被试并计算 ROI 级相关矩阵。"""
    MATRIX_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_beta_matrix_232.npy")
    SUBJECTS_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_beta_matrix_232_subjects.csv")
    ROI_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_beta_matrix_232_rois.csv")
    SUBJECT_INFO_PATH = Path(
        os.environ.get(
            "SUBJECT_AGE_TABLE",
            "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv",
        )
    )
    OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
    OUT_PREFIX = "roi_isc_spearman_by_age"

    matrix = np.load(MATRIX_PATH)
    subjects = load_subject_list(SUBJECTS_PATH)
    rois = pd.read_csv(ROI_PATH)
    if matrix.shape[1] != len(subjects):
        raise ValueError(f"被试数量不一致: matrix={matrix.shape[1]} list={len(subjects)}")
    if matrix.shape[0] != len(rois):
        raise ValueError(f"ROI 数量不一致: matrix={matrix.shape[0]} list={len(rois)}")

    age_map = load_subject_ages_map(SUBJECT_INFO_PATH)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    index_map = {s: i for i, s in enumerate(subjects)}
    sorted_indices = [index_map[s] for s in subjects_sorted]
    matrix_sorted = matrix[:, sorted_indices, :]

    n_roi = matrix_sorted.shape[0]
    n_sub = matrix_sorted.shape[1]
    corr_all = np.zeros((n_roi, n_sub, n_sub), dtype=np.float32)
    for i in range(n_roi):
        corr_all[i] = spearman_corr_matrix_rows(matrix_sorted[i])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{OUT_PREFIX}.npy", corr_all)
    pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted}).to_csv(
        OUT_DIR / f"{OUT_PREFIX}_subjects_sorted.csv", index=False
    )
    rois.to_csv(OUT_DIR / f"{OUT_PREFIX}_rois.csv", index=False)


if __name__ == "__main__":
    main()
