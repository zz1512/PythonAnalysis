#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合分析：
1) 读取 calc_isc_combined_strict 保存的被试*被试相似性矩阵。
2) 参考 calc_isc_combined_strict 的被试排序方式（subject_order_*.csv）。
3) 根据 modeling_individual_diff_beh.py 的 Spearman 距离逻辑，生成实际年龄距离矩阵。
4) 对相似性矩阵与年龄距离矩阵做联合分析（相关性）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ===== 默认路径（可通过命令行覆盖） =====
DEFAULT_OUTPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
DEFAULT_SUBJECT_INFO_PATH = Path("/public/home/dingrui/BIDS_DATA/emo_20250623/横断队列被试信息表.tsv")

# 被试信息表中的列名
COL_SUB_ID = "被试编号"
COL_AGE = "采集年龄"


def parse_chinese_age_exact(age_str: str) -> float:
    """
    解析 '11岁11个月10天' 格式的年龄字符串，返回精确年龄（年，float）。
    """
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan

    age_str = str(age_str)

    y_match = re.search(r"(\d+)\s*岁", age_str)
    m_match = re.search(r"(\d+)\s*个月", age_str)
    if not m_match:
        m_match = re.search(r"(\d+)\s*月", age_str)
    d_match = re.search(r"(\d+)\s*天", age_str)

    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0

    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map(subject_info_path: Path) -> dict:
    """
    智能读取表格 (支持 xlsx/tsv/csv)，建立 {sub-ID: age} 映射。
    """
    path_str = str(subject_info_path)

    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

    if COL_SUB_ID not in info_df.columns or COL_AGE not in info_df.columns:
        raise ValueError(f"表格中找不到列名: '{COL_SUB_ID}' 或 '{COL_AGE}'")

    age_map = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[COL_SUB_ID]).strip()
        age_str = row[COL_AGE]

        if not raw_id.startswith("sub-"):
            sub_id = f"sub-{raw_id}"
        else:
            sub_id = raw_id

        age_map[sub_id] = parse_chinese_age_exact(age_str)

    return age_map


def spearman_distance_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    Compute NxN distance matrix using Spearman correlation between row-vectors (across features):
    distance = 1 - Spearman rho.

    If only one feature is available, fall back to rank-distance normalized to [0, 1].
    """
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        X_num = X.apply(lambda s: pd.Categorical(s).codes if s.dtype == "O" else s).astype(float)

    A = X_num.to_numpy(dtype=float)
    n, p = A.shape

    if p == 1:
        ranks = stats.rankdata(A[:, 0], method="average")
        denom = (n - 1) if n > 1 else 1.0
        D = np.abs(ranks[:, None] - ranks[None, :]) / denom
        np.fill_diagonal(D, 0.0)
        return D

    ranks = np.apply_along_axis(stats.rankdata, 1, A)
    ranks = (ranks - ranks.mean(axis=1, keepdims=True)) / ranks.std(axis=1, ddof=0, keepdims=True)
    corr = (ranks @ ranks.T) / (p - 1)
    corr = np.clip(corr, -1.0, 1.0)
    D = 1.0 - corr
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def load_subject_order(order_path: Optional[Path], similarity_df: pd.DataFrame) -> List[str]:
    if order_path and order_path.exists():
        order_df = pd.read_csv(order_path)
        if "subject" not in order_df.columns:
            raise ValueError("subject_order 文件缺少 'subject' 列")
        return order_df["subject"].astype(str).tolist()

    return similarity_df.index.astype(str).tolist()


def infer_context_name(similarity_path: Path) -> str:
    name = similarity_path.stem
    match = re.match(r"similarity_(.*)_AgeSorted", name)
    if match:
        return match.group(1)
    return name


def vectorize_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu]


def run_analysis(
    similarity_path: Path,
    subject_info_path: Path,
    output_dir: Path,
    subject_order_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    similarity_df = pd.read_csv(similarity_path, index_col=0)
    subjects = load_subject_order(subject_order_path, similarity_df)

    similarity_df = similarity_df.loc[subjects, subjects]

    age_map = load_subject_ages_map(subject_info_path)
    ages = [age_map.get(sub, np.nan) for sub in subjects]
    age_df = pd.DataFrame({"age": ages}, index=subjects)

    age_distance = spearman_distance_matrix(age_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    context_name = infer_context_name(similarity_path)

    age_distance_path = output_dir / f"age_distance_{context_name}.csv"
    pd.DataFrame(age_distance, index=subjects, columns=subjects).to_csv(age_distance_path)

    sim_vec = vectorize_upper_triangle(similarity_df.to_numpy())
    age_vec = vectorize_upper_triangle(age_distance)

    mask = np.isfinite(sim_vec) & np.isfinite(age_vec)
    if mask.sum() == 0:
        raise ValueError("可用数据为空，无法进行联合分析。")

    pearson_r, pearson_p = stats.pearsonr(sim_vec[mask], age_vec[mask])
    spearman_r, spearman_p = stats.spearmanr(sim_vec[mask], age_vec[mask])

    result_df = pd.DataFrame(
        {
            "metric": ["pearson", "spearman"],
            "r": [pearson_r, spearman_r],
            "p": [pearson_p, spearman_p],
            "n": [int(mask.sum())] * 2,
        }
    )

    result_path = output_dir / f"joint_analysis_{context_name}.csv"
    result_df.to_csv(result_path, index=False)

    pair_df = pd.DataFrame(
        {
            "subject_i": np.repeat(subjects, len(subjects)),
            "subject_j": np.tile(subjects, len(subjects)),
            "similarity": similarity_df.to_numpy().ravel(),
            "age_distance": age_distance.ravel(),
        }
    )
    pair_path = output_dir / f"pairwise_similarity_age_{context_name}.csv"
    pair_df.to_csv(pair_path, index=False)

    return age_distance_path, result_path, pair_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="联合分析相似性矩阵与年龄距离矩阵")
    parser.add_argument(
        "--similarity",
        type=Path,
        required=True,
        help="calc_isc_combined_strict 输出的相似性矩阵 CSV",
    )
    parser.add_argument(
        "--subject-info",
        type=Path,
        default=DEFAULT_SUBJECT_INFO_PATH,
        help="被试信息表路径 (tsv/xlsx/csv)",
    )
    parser.add_argument(
        "--subject-order",
        type=Path,
        default=None,
        help="calc_isc_combined_strict 输出的 subject_order_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    age_distance_path, result_path, pair_path = run_analysis(
        similarity_path=args.similarity,
        subject_info_path=args.subject_info,
        output_dir=args.output_dir,
        subject_order_path=args.subject_order,
    )

    print(f"年龄距离矩阵已保存: {age_distance_path}")
    print(f"联合分析结果已保存: {result_path}")
    print(f"配对矩阵已保存: {pair_path}")


if __name__ == "__main__":
    main()
