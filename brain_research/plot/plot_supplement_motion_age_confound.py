#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_supplement_motion_age_confound.py

补充材料推荐图：Motion–Age 混杂可视化（发育队列常见审稿问题）。

默认文件发现路径与 emo 分析输出保持一致：
- LSS_OUTPUT_ROOT（默认 /public/home/dingrui/fmri_analysis/zz_analysis/lss_results）
- QC_AUDIT_OUT_DIR（默认 <LSS_OUTPUT_ROOT 的上一级>/qc_audit_out）
- FIG_SUPP_DIR（默认 <LSS_OUTPUT_ROOT 的上一级>/figures_supplement）
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


def _default_qc_out_dir() -> Path:
    p = os.environ.get("QC_AUDIT_OUT_DIR", "")
    if p:
        return Path(p)
    lss = _default_lss_output_root()
    return lss.parent / "qc_audit_out"


def _default_fig_out_dir() -> Path:
    p = os.environ.get("FIG_SUPP_DIR", "")
    if p:
        return Path(p)
    lss = _default_lss_output_root()
    return lss.parent / "figures_supplement"


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
    qc = _default_qc_out_dir()
    return [
        Path.cwd(),
        qc,
        lss.parent,
        lss,
    ]


def _read_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="补充图：motion-age 混杂可视化（基于 audit_motion_age_confound 输出）")
    p.add_argument("--by-subject", type=Path, default=None, help="audit_motion_age_confound_by_subject.csv（不填则自动找最新）")
    p.add_argument("--summary", type=Path, default=None, help="audit_motion_age_confound_summary.csv（不填则自动找最新）")
    p.add_argument("--task", type=str, default="EMO", help="绘制哪个 task（EMO/SOC）")
    p.add_argument("--metric", type=str, default="outlier_frac", help="散点图使用的 motion 指标列名（默认 outlier_frac）")
    p.add_argument("--high-motion-threshold", type=float, default=0.2, help="高 motion 阈值（用于箱线图分组）")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认使用 FIG_SUPP_DIR 或 zz_analysis/figures_supplement）")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="输出格式")
    return p


def _pick_stat(summary: Optional[pd.DataFrame], task: str, metric: str) -> str:
    if summary is None or summary.empty:
        return ""
    s = summary.copy()
    s["task"] = s.get("task", "").astype(str)
    s["metric"] = s.get("metric", "").astype(str)
    row = s[(s["task"] == task) & (s["metric"] == metric)]
    if row.empty:
        return ""
    r = pd.to_numeric(row["spearman_r"].iloc[0], errors="coerce")
    p = pd.to_numeric(row["spearman_p_perm_greater"].iloc[0], errors="coerce")
    if not (np.isfinite(r) and np.isfinite(p)):
        return ""
    return f"Spearman r={float(r):.3f}, p_perm(greater)={float(p):.4g}"


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = _default_roots()

    by_sub_path = Path(args.by_subject) if args.by_subject is not None else _find_latest("audit_motion_age_confound_by_subject.csv", roots)
    summary_path = Path(args.summary) if args.summary is not None else _find_latest("audit_motion_age_confound_summary.csv", roots)

    df = _read_optional(by_sub_path)
    summary = _read_optional(summary_path)

    task = str(args.task).strip()
    metric = str(args.metric).strip()

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))
    fig.suptitle(f"Supplementary: Motion–Age Confound ({task})", y=1.02)

    if df is None or df.empty:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing audit_motion_age_confound_by_subject.csv", ha="center", va="center")
    else:
        d = df.copy()
        d["task"] = d.get("task", "").astype(str)
        d = d[d["task"] == task].copy()
        d["age"] = pd.to_numeric(d.get("age", np.nan), errors="coerce")
        d[metric] = pd.to_numeric(d.get(metric, np.nan), errors="coerce")
        d = d[np.isfinite(d["age"].to_numpy(dtype=float)) & np.isfinite(d[metric].to_numpy(dtype=float))].copy()

        if d.empty:
            for ax in axes:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No valid rows for task={task}", ha="center", va="center")
        else:
            ax0 = axes[0]
            sns.regplot(
                data=d,
                x="age",
                y=metric,
                ax=ax0,
                scatter_kws={"s": 22, "alpha": 0.7, "edgecolor": "none"},
                line_kws={"linewidth": 2.0},
                color="#4c72b0",
            )
            ax0.set_title(f"Age vs {metric}")
            ax0.set_xlabel("Age")
            ax0.set_ylabel(metric)
            stat = _pick_stat(summary, task=task, metric=metric)
            if stat:
                ax0.text(0.02, 0.98, stat, transform=ax0.transAxes, ha="left", va="top", fontsize=10)
            ax0.text(0.98, 0.02, f"N={d.shape[0]}", transform=ax0.transAxes, ha="right", va="bottom", fontsize=10)

            ax1 = axes[1]
            thr = float(args.high_motion_threshold)
            d["high_motion"] = d[metric] > thr
            sns.boxplot(data=d, x="high_motion", y="age", ax=ax1, color="#55a868")
            sns.stripplot(data=d, x="high_motion", y="age", ax=ax1, color="black", alpha=0.5, size=2.5, jitter=0.25)
            ax1.set_title(f"Age by High Motion (>{thr:g} {metric})")
            ax1.set_xlabel("high_motion")
            ax1.set_ylabel("Age")

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_fig_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"S_motion_age_confound_{task}_{metric}.{str(args.format)}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
