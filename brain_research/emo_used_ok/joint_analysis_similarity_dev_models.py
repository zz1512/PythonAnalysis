#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合分析：
1) 读取 calc_isc_combined_strict 保存的被试*被试相似性矩阵。
2) 参考 calc_isc_combined_strict 的被试排序方式（subject_order_*.csv）。
3) 根据 modeling_individual_diff_beh.py 的三种发育模型矩阵构造方式生成矩阵：
   - M_nn（最近邻模型）
   - M_conv（收敛性模型）
   - M_div（发散性模型）
4) 对相似性矩阵与三个模型矩阵进行联合分析（皮尔逊相关）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# 全脑
DEFAULT_OUTPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
# 具体脑区
# DEFAULT_OUTPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi_age_sorted")
DEFAULT_SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"


@dataclass
class DevModels:
    nearest_neighbor: np.ndarray
    convergence: np.ndarray
    divergence: np.ndarray


def parse_chinese_age_exact(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan

    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    age_str = str(age_str).strip()
    try:
        return float(age_str)
    except Exception:
        pass

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
    path_str = str(subject_info_path)

    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(
            "年龄表缺少必要列。需要以下任意一组列名：\n"
            f"- {COL_SUB_ID}, {COL_AGE}\n"
            f"- {LEGACY_COL_SUB_ID}, {LEGACY_COL_AGE}\n"
            f"实际列名: {list(info_df.columns)}"
        )

    age_map = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        age_str = row[age_col]

        if not raw_id.startswith("sub-"):
            sub_id = f"sub-{raw_id}"
        else:
            sub_id = raw_id

        age_map[sub_id] = parse_chinese_age_exact(age_str)

    return age_map


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


def triu_vectorize(matrix: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices_from(matrix, k=k)
    return matrix[iu]


def build_dev_models(age: np.ndarray, normalize: bool = True) -> DevModels:
    a = np.asarray(age, float).reshape(-1)
    n = a.size
    A_i = np.repeat(a[:, None], n, axis=1)
    A_j = A_i.T
    amax = np.nanmax(a)

    M_nn = amax - np.abs(A_i - A_j)
    M_conv = np.minimum(A_i, A_j)
    M_div = amax - 0.5 * (A_i + A_j)

    if normalize:
        def _norm(M: np.ndarray) -> np.ndarray:
            vec = triu_vectorize(M, 1)
            mu = np.nanmean(vec)
            sd = np.nanstd(vec)
            return (M - mu) / (sd if sd > 0 else 1.0)

        M_nn = _norm(M_nn)
        M_conv = _norm(M_conv)
        M_div = _norm(M_div)

    np.fill_diagonal(M_nn, np.nan)
    np.fill_diagonal(M_conv, np.nan)
    np.fill_diagonal(M_div, np.nan)

    return DevModels(M_nn, M_conv, M_div)


def vectorize_valid_pairs(similarity: np.ndarray, model: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_vec = triu_vectorize(similarity, 1)
    m_vec = triu_vectorize(model, 1)
    mask = np.isfinite(s_vec) & np.isfinite(m_vec)
    return s_vec[mask], m_vec[mask]

# 全脑可选文件：similarity_surface_L_AgeSorted.csv、similarity_surface_R_AgeSorted.csv、similarity_volume_AgeSorted.csv
# roi可选文件：similarity_surface_L_Control_PFC_AgeSorted.csv
# similarity_surface_L_Salience_Insula_AgeSorted.csv
# similarity_surface_L_Visual_AgeSorted.csv
# similarity_surface_R_Control_PFC_AgeSorted.csv
# similarity_surface_R_Salience_Insula_AgeSorted.csv
# similarity_surface_R_Visual_AgeSorted.csv
def resolve_similarity_path(similarity_path: Optional[Path], output_dir: Path) -> Path:
    if similarity_path is not None:
        return similarity_path

    candidates = sorted(output_dir.glob("similarity_surface_L_AgeSorted.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"未在 {output_dir} 找到 similarity_*_AgeSorted.csv，请指定 --similarity。")
    return candidates[-1]


def resolve_subject_order_path(subject_order_path: Optional[Path], output_dir: Path, context_name: str) -> Optional[Path]:
    if subject_order_path is not None:
        return subject_order_path

    candidate = output_dir / f"subject_order_{context_name}.csv"
    if candidate.exists():
        return candidate
    return None


def run_analysis(
    similarity_path: Optional[Path],
    subject_info_path: Path,
    output_dir: Path,
    subject_order_path: Optional[Path] = None,
    normalize_models: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    similarity_path = resolve_similarity_path(similarity_path, output_dir)
    context_name = infer_context_name(similarity_path)
    subject_order_path = resolve_subject_order_path(subject_order_path, output_dir, context_name)

    similarity_df = pd.read_csv(similarity_path, index_col=0)
    subjects = load_subject_order(subject_order_path, similarity_df)

    similarity_df = similarity_df.loc[subjects, subjects]

    age_map = load_subject_ages_map(subject_info_path)
    ages = np.array([age_map.get(sub, np.nan) for sub in subjects], dtype=float)

    if np.all(np.isnan(ages)):
        raise ValueError("所有被试年龄均缺失，无法构建发育模型矩阵。")

    dev_models = build_dev_models(ages, normalize=normalize_models)

    output_dir.mkdir(parents=True, exist_ok=True)
    m_nn_path = output_dir / f"dev_model_nn_{context_name}.csv"
    m_conv_path = output_dir / f"dev_model_conv_{context_name}.csv"
    m_div_path = output_dir / f"dev_model_div_{context_name}.csv"

    pd.DataFrame(dev_models.nearest_neighbor, index=subjects, columns=subjects).to_csv(m_nn_path)
    pd.DataFrame(dev_models.convergence, index=subjects, columns=subjects).to_csv(m_conv_path)
    pd.DataFrame(dev_models.divergence, index=subjects, columns=subjects).to_csv(m_div_path)

    sim_matrix = similarity_df.to_numpy()

    results = []
    for name, model in (
        ("M_nn", dev_models.nearest_neighbor),
        ("M_conv", dev_models.convergence),
        ("M_div", dev_models.divergence),
    ):
        s_vec, m_vec = vectorize_valid_pairs(sim_matrix, model)
        if s_vec.size == 0:
            raise ValueError(f"模型 {name} 的可用配对为空，无法计算相关。")
        r_val, p_val = stats.pearsonr(s_vec, m_vec)
        results.append({"model": name, "r": r_val, "p": p_val, "n": int(s_vec.size)})

    result_df = pd.DataFrame(results)
    result_path = output_dir / f"joint_analysis_dev_models_{context_name}.csv"
    result_df.to_csv(result_path, index=False)

    return m_nn_path, m_conv_path, m_div_path, result_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="联合分析相似性矩阵与发育模型矩阵")
    parser.add_argument(
        "--similarity",
        type=Path,
        default=None,
        help="calc_isc_combined_strict 输出的相似性矩阵 CSV（不提供则在输出目录中自动寻找最新文件）",
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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="不对模型矩阵进行标准化",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    m_nn_path, m_conv_path, m_div_path, result_path = run_analysis(
        similarity_path=args.similarity,
        subject_info_path=args.subject_info,
        output_dir=args.output_dir,
        subject_order_path=args.subject_order,
        normalize_models=not args.no_normalize,
    )

    print(f"发育模型矩阵已保存: {m_nn_path}")
    print(f"发育模型矩阵已保存: {m_conv_path}")
    print(f"发育模型矩阵已保存: {m_div_path}")
    print(f"联合分析结果已保存: {result_path}")


if __name__ == "__main__":
    main()