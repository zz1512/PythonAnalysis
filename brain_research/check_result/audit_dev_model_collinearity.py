#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_dev_model_collinearity.py

论文级审计（发育模型共线性）：
从年龄表构造 3 个发育模型的上三角向量（M_nn / M_conv / M_div），并输出它们之间的相关矩阵。
该结果用于回答审稿人常问的问题：这些模型是否高度相关（共线性），是否需要“模型比较/控制共同成分”。

输入：
- 年龄表：优先列名 sub_id/age；兼容旧列名 被试编号/采集年龄（以及中文年龄字符串）
- 可选 subject_order：用于固定 subjects 顺序（确保与相似性矩阵一致）

输出：
- audit_dev_model_collinearity.csv：3×3 相关矩阵
- audit_age_distribution.csv：年龄分布摘要（describe）

扩展建议（未来）：
- 输出 VIF / 条件数（更直观量化共线性）
- 在置换框架中加入“控制变量矩阵”，做 partial correlation / matrix regression 的敏感性分析
"""

import argparse
import os
from pathlib import Path
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"


def parse_age(age_str: object) -> float:
    """将年龄字段解析为 float 年龄（年）。支持数值、纯数字字符串、中文“岁/月/天”字符串。"""
    if pd.isna(age_str) or str(age_str).strip() == "":
        return float("nan")
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y = re.search(r"(\d+)\s*岁", s)
    m = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d = re.search(r"(\d+)\s*天", s)
    years = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    days = int(d.group(1)) if d else 0
    return years + (months / 12.0) + (days / 365.0)


def load_age_table(path: Path) -> pd.DataFrame:
    """读取年龄表并规范化为两列：subject（sub-xxx）与 age（float）。"""
    p = str(path)
    if p.endswith(".tsv") or p.endswith(".txt"):
        df = pd.read_csv(p, sep="\t")
    elif p.endswith(".csv"):
        df = pd.read_csv(p)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        df = pd.read_excel(p)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")
    if COL_SUB_ID in df.columns and COL_AGE in df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in df.columns and LEGACY_COL_AGE in df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(f"年龄表缺少列: {COL_SUB_ID}/{COL_AGE} 或 {LEGACY_COL_SUB_ID}/{LEGACY_COL_AGE}")

    out = pd.DataFrame(
        {
            "subject": df[sub_col].astype(str).map(lambda x: x if x.startswith("sub-") else f"sub-{x}"),
            "age": df[age_col].map(parse_age).astype(float),
        }
    )
    out = out.dropna(subset=["age"])
    return out


def dev_model_vectors(ages: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于年龄数组构造 3 个模型的上三角向量。"""
    a = np.asarray(ages, dtype=float).reshape(-1)
    n = a.size
    iu = np.triu_indices(n, k=1)
    ai = a[iu[0]]
    aj = a[iu[1]]
    amax = float(np.max(a))
    m_nn = amax - np.abs(ai - aj)
    m_conv = np.minimum(ai, aj)
    m_div = amax - 0.5 * (ai + aj)
    return m_nn, m_conv, m_div


def corr_matrix(vecs: Dict[str, np.ndarray]) -> pd.DataFrame:
    """计算多个向量之间的相关矩阵（忽略 NaN/Inf）。"""
    keys = list(vecs.keys())
    mat = np.full((len(keys), len(keys)), np.nan, dtype=float)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                mat[i, j] = 1.0
            else:
                x = vecs[ki]
                y = vecs[kj]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 10:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = float(np.corrcoef(x[m], y[m])[0, 1])
    return pd.DataFrame(mat, index=keys, columns=keys)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计发育模型（M_nn/M_conv/M_div）的共线性")
    p.add_argument(
        "--age-table",
        type=Path,
        default=Path(
            os.environ.get(
                "SUBJECT_AGE_TABLE",
                "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv",
            )
        ),
        help="年龄表路径（sub_id/age）。可用环境变量 SUBJECT_AGE_TABLE 覆盖。",
    )
    p.add_argument("--subject-order", type=Path, default=None, help="subject_order_*.csv（可选，用于固定 subjects 顺序；未指定将尝试自动发现）")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    p.add_argument("--fail-abs-corr", type=float, default=0.95, help="模型间最大绝对相关超过该阈值则判为未通过")
    p.add_argument("--log-matrix", action="store_true", help="在日志中打印 3×3 相关矩阵")
    p.add_argument("--lss-root", type=Path, default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"), help="自动发现 subject_order 时搜索的根目录")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    age_table = Path(args.age_table)
    if not age_table.exists():
        print("[QC] audit_dev_model_collinearity: FAIL")
        print(f"[QC] Age table not found: {age_table}")
        print("[QC] Provide --age-table or set SUBJECT_AGE_TABLE.")
        return

    ages_df = load_age_table(age_table)

    subject_order_path = Path(args.subject_order) if args.subject_order is not None else None
    if subject_order_path is None:
        candidates = sorted(Path(args.lss_root).rglob("subject_order_*.csv"), key=lambda p: p.stat().st_mtime)
        subject_order_path = candidates[-1] if candidates else None

    if subject_order_path is not None:
        if subject_order_path.exists():
            order_df = pd.read_csv(subject_order_path)
            if "subject" in order_df.columns:
                order = order_df["subject"].astype(str).tolist()
                ages_df = ages_df.set_index("subject").reindex(order).reset_index()
                if "age" in ages_df.columns:
                    before = int(ages_df.shape[0])
                    ages_df = ages_df.dropna(subset=["age"])
                    dropped = before - int(ages_df.shape[0])
                    if dropped > 0:
                        print(f"[QC] Warning: subject_order 中有 {dropped} 个被试缺少年龄，已剔除。")
            else:
                print(f"[QC] Warning: subject_order 缺少 subject 列，已忽略: {subject_order_path}")
        else:
            print(f"[QC] Warning: subject_order 不存在，已忽略: {subject_order_path}")

    ages = ages_df["age"].to_numpy(dtype=float)
    if np.isfinite(ages).sum() < 10:
        print("[QC] audit_dev_model_collinearity: FAIL")
        print("[QC] Too few valid ages (<10).")
        return

    m_nn, m_conv, m_div = dev_model_vectors(ages)
    vecs = {"M_nn": m_nn, "M_conv": m_conv, "M_div": m_div}
    cmat = corr_matrix(vecs)
    out_path = out_dir / "audit_dev_model_collinearity.csv"
    cmat.to_csv(out_path)

    desc = ages_df["age"].describe()
    desc_path = out_dir / "audit_age_distribution.csv"
    desc.to_csv(desc_path, header=True)

    print(f"Saved: {out_path}")
    print(f"Saved: {desc_path}")

    off = cmat.copy()
    for k in off.index:
        off.loc[k, k] = np.nan
    max_abs = float(np.nanmax(np.abs(off.to_numpy(dtype=float)))) if off.size else float("nan")
    ok = bool(np.isfinite(max_abs) and (max_abs <= float(args.fail_abs_corr)))
    status = "PASS" if ok else "FAIL"
    n_sub = int(np.isfinite(ages).sum())
    a_min = float(np.nanmin(ages))
    a_max = float(np.nanmax(ages))
    print(f"[QC] audit_dev_model_collinearity: {status}")
    print(
        f"[QC] N={n_sub} | AgeRange=[{a_min:.3f}, {a_max:.3f}] | "
        f"MaxAbsOffDiagCorr={max_abs:.4f} | Threshold={float(args.fail_abs_corr):.3f}"
    )
    if bool(args.log_matrix) or status == "FAIL":
        print("[QC] Model correlation matrix:")
        print(cmat.round(4).to_string())


if __name__ == "__main__":
    main()
