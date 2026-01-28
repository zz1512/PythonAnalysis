#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_dev_model_relationships.py

论文图（F5）：展示三个发育模型（M_nn/M_conv/M_div）之间的相关结构（共线性）。
默认零参数运行：
- 年龄表优先读取环境变量 SUBJECT_AGE_TABLE，否则使用项目默认路径
- 若存在 subject_order_*.csv（通常由 calc_isc_* 生成），自动使用最新文件固定 subjects 顺序
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

DEFAULT_AGE_TABLE = Path(
    os.environ.get(
        "SUBJECT_AGE_TABLE",
        "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv",
    )
)
DEFAULT_LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
DEFAULT_OUT_DIR = Path("./figures_dev_models")


def parse_age(age_str: object) -> float:
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
    p = str(path)
    if p.endswith(".tsv") or p.endswith(".txt"):
        df = pd.read_csv(p, sep="\t")
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        df = pd.read_excel(p)
    elif p.endswith(".csv"):
        df = pd.read_csv(p)
    else:
        raise ValueError("不支持的年龄表格式（tsv/xlsx/csv）")

    if COL_SUB_ID in df.columns and COL_AGE in df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in df.columns and LEGACY_COL_AGE in df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError("年龄表缺少必要列 sub_id/age 或 被试编号/采集年龄")

    out = pd.DataFrame(
        {
            "subject": df[sub_col].astype(str).map(lambda x: x if x.startswith("sub-") else f"sub-{x}"),
            "age": df[age_col].map(parse_age).astype(float),
        }
    )
    out = out.dropna(subset=["age"])
    return out


def _find_latest(paths: list) -> Optional[Path]:
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _auto_subject_order(lss_root: Path) -> Optional[Path]:
    if not lss_root.exists():
        return None
    return _find_latest(list(lss_root.rglob("subject_order_*.csv")))


def dev_model_vectors(ages: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def main() -> None:
    age_table = DEFAULT_AGE_TABLE
    if not age_table.exists():
        print("[QC] plot_dev_model_relationships: FAIL")
        print(f"[QC] Age table not found: {age_table}")
        return

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ages_df = load_age_table(age_table)
    order_path = _auto_subject_order(DEFAULT_LSS_ROOT)
    if order_path is not None:
        order_df = pd.read_csv(order_path)
        if "subject" in order_df.columns:
            order = order_df["subject"].astype(str).tolist()
            ages_df = ages_df.set_index("subject").reindex(order).reset_index()

    ages = ages_df["age"].to_numpy(dtype=float)
    ages = ages[np.isfinite(ages)]
    if ages.size < 10:
        print("[QC] plot_dev_model_relationships: FAIL")
        print("[QC] Too few valid ages (<10).")
        return

    m_nn, m_conv, m_div = dev_model_vectors(ages)
    cmat = corr_matrix({"M_nn": m_nn, "M_conv": m_conv, "M_div": m_div})

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(
        cmat,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
        linewidths=0.6,
        linecolor="white",
    )
    ax.set_title("Collinearity Among Development Models")
    plt.tight_layout()
    out_path = out_dir / "F5_dev_model_collinearity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

