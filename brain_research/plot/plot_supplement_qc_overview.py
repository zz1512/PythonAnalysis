#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_supplement_qc_overview.py

补充材料推荐图：QC 一页总览
面板（尽量自动发现输入文件；缺失则跳过并在图中标注）：
- confounds scrubbing 强度（outlier_frac）
- LSS trial 成功率（trial_success_rate，来自 lss_audit 汇总）
- split-half reliability（若存在 reliability_check_EMO.csv）
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _find_latest(pattern: str, roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for r in roots:
        if r.exists():
            candidates.extend(list(r.rglob(pattern)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def main() -> None:
    roots = [
        Path.cwd(),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis"),
    ]

    conf_csv = _find_latest("audit_confounds_quality.csv", roots)
    lss_by_sub = _find_latest("audit_lss_failures_by_subject.csv", roots)
    rel_csv = _find_latest("reliability_check_EMO.csv", roots) or _find_latest("reliability_results.csv", roots)

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    def _panel_missing(ax, title: str, hint: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, hint, ha="center", va="center", fontsize=10)

    if conf_csv is None:
        _panel_missing(axes[0], "Confounds Scrubbing", "Missing audit_confounds_quality.csv")
    else:
        df = pd.read_csv(conf_csv)
        x = pd.to_numeric(df.get("outlier_frac", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        axes[0].set_title("Confounds Scrubbing (outlier_frac)")
        if x.size == 0:
            _panel_missing(axes[0], "Confounds Scrubbing", "No valid outlier_frac values")
        else:
            sns.histplot(x, bins=30, kde=True, ax=axes[0], color="#4c72b0", alpha=0.65)
            axes[0].set_xlabel("outlier_frac")
            axes[0].set_ylabel("Subjects")
            axes[0].axvline(0.2, color="#c44e52", linewidth=2, linestyle="--", alpha=0.9)

    if lss_by_sub is None:
        _panel_missing(axes[1], "LSS Trial Success Rate", "Missing audit_lss_failures_by_subject.csv")
    else:
        df = pd.read_csv(lss_by_sub)
        x = pd.to_numeric(df.get("trial_success_rate", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        axes[1].set_title("LSS Trial Success Rate")
        if x.size == 0:
            _panel_missing(axes[1], "LSS Trial Success Rate", "No valid trial_success_rate values")
        else:
            sns.histplot(x, bins=30, kde=True, ax=axes[1], color="#55a868", alpha=0.65)
            axes[1].set_xlabel("trial_success_rate")
            axes[1].set_ylabel("Subjects")
            axes[1].axvline(0.9, color="#c44e52", linewidth=2, linestyle="--", alpha=0.9)

    if rel_csv is None:
        _panel_missing(axes[2], "Split-Half Reliability", "Missing reliability_check_EMO.csv")
    else:
        df = pd.read_csv(rel_csv)
        x = pd.to_numeric(df.get("reliability", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        axes[2].set_title("Split-Half Reliability")
        if x.size == 0:
            _panel_missing(axes[2], "Split-Half Reliability", "No valid reliability values")
        else:
            sns.histplot(x, bins=30, kde=True, ax=axes[2], color="#8172b2", alpha=0.65)
            axes[2].set_xlabel("reliability (r)")
            axes[2].set_ylabel("Subjects")
            axes[2].axvline(0.1, color="#c44e52", linewidth=2, linestyle="--", alpha=0.9)

    fig.suptitle("Supplementary QC Overview", y=1.02)
    plt.tight_layout()

    out_dir = Path("./figures_supplement")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "S_QC_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

