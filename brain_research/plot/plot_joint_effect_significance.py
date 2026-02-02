#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_joint_effect_significance.py

论文/补充材料诊断图：
- 效应大小（r_obs） vs 显著性（-log10 p_fwer*）散点图
- 效应符号/分布图（用于检查 one-sided 假设是否合理、是否存在系统性负效应）

默认文件发现路径与 emo 分析输出保持一致：
- LSS_OUTPUT_ROOT（默认 /public/home/dingrui/fmri_analysis/zz_analysis/lss_results）
- FIG_JOINT_DIR（默认 <LSS_OUTPUT_ROOT 的上一级>/figures_joint）
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _default_lss_output_root() -> Path:
    return Path(os.environ.get("LSS_OUTPUT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"))


def _default_fig_out_dir() -> Path:
    p = os.environ.get("FIG_JOINT_DIR", "")
    if p:
        return Path(p)
    lss = _default_lss_output_root()
    return lss.parent / "figures_joint"


def _find_latest(pattern: str, roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for r in roots:
        if r.exists():
            candidates.extend(list(r.rglob(pattern)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _default_roots() -> List[Path]:
    lss = _default_lss_output_root()
    return [
        Path.cwd(),
        lss.parent,
        lss,
    ]


def _choose_p_fwer_column(df: pd.DataFrame) -> str:
    for c in ["p_fwer", "p_fwer_one_tailed", "p_fwer_two_tailed"]:
        if c in df.columns:
            return c
    raise ValueError("结果CSV缺少 p_fwer 列（支持：p_fwer / p_fwer_one_tailed / p_fwer_two_tailed）")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="论文图：joint 结果的效应-显著性诊断（volcano/符号分布）")
    p.add_argument("--joint-csv", type=Path, default=None, help="joint_analysis_dev_models_perm_fwer*.csv（不填则自动找最新）")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性阈值（用于参考线）")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认使用 FIG_JOINT_DIR 或 zz_analysis/figures_joint）")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="输出格式")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = _default_roots()
    joint_csv = Path(args.joint_csv) if args.joint_csv is not None else _find_latest("joint_analysis_dev_models_perm_fwer*.csv", roots)
    if joint_csv is None or not joint_csv.exists():
        raise FileNotFoundError("未找到 joint_analysis_dev_models_perm_fwer*.csv，请先运行联合分析脚本")

    df = pd.read_csv(joint_csv)
    pcol = _choose_p_fwer_column(df)
    required = {"matrix", "model", "r_obs", pcol}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"结果CSV缺少列: {sorted(miss)}")

    d = df.copy()
    d["matrix"] = d["matrix"].astype(str)
    d["model"] = d["model"].astype(str)
    d["r_obs"] = pd.to_numeric(d["r_obs"], errors="coerce")
    d[pcol] = pd.to_numeric(d[pcol], errors="coerce")
    d = d[np.isfinite(d["r_obs"].to_numpy(dtype=float)) & np.isfinite(d[pcol].to_numpy(dtype=float))].copy()
    d["neglog10p"] = -np.log10(d[pcol].clip(lower=np.finfo(float).tiny))
    d["sig"] = d[pcol] <= float(args.alpha)

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.suptitle(f"Joint Analysis Diagnostics ({joint_csv.name}; {pcol})", y=1.02)

    ax0 = axes[0]
    sns.scatterplot(
        data=d,
        x="r_obs",
        y="neglog10p",
        hue="model",
        style="matrix",
        s=70,
        alpha=0.85,
        ax=ax0,
    )
    ax0.axhline(-np.log10(float(args.alpha)), color="gray", linestyle="--", linewidth=1.2, alpha=0.75)
    ax0.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax0.set_xlabel("Effect (r_obs)")
    ax0.set_ylabel(f"-log10({pcol})")
    ax0.set_title("Effect vs Significance")
    ax0.legend(loc="best", fontsize=8, frameon=True)

    ax1 = axes[1]
    sns.histplot(data=d, x="r_obs", hue="model", element="step", stat="density", common_norm=False, kde=True, ax=ax1, alpha=0.35)
    ax1.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.75)
    ax1.set_title("Effect Sign Distribution")
    ax1.set_xlabel("r_obs")
    ax1.set_ylabel("density")
    sig_n = int(d["sig"].sum())
    ax1.text(0.98, 0.98, f"Total={d.shape[0]}\nSig(p≤{float(args.alpha):g})={sig_n}", transform=ax1.transAxes, ha="right", va="top", fontsize=10)

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_fig_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"joint_effect_significance_{pcol}.{str(args.format)}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
