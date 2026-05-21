#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_supplement_missingness.py

补充材料推荐图：Missingness / Stimulus Coverage 总览。

默认文件发现路径与 emo 分析输出保持一致：
- LSS_OUTPUT_ROOT（默认 /public/home/dingrui/fmri_analysis/zz_analysis/lss_results）
- QC_AUDIT_OUT_DIR（默认 <LSS_OUTPUT_ROOT 的上一级>/qc_audit_out）
- FIG_SUPP_DIR（默认 <LSS_OUTPUT_ROOT 的上一级>/figures_supplement）

你可以通过命令行参数显式指定输入 CSV；否则脚本会在默认根目录中自动搜索最新文件。
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
    p = argparse.ArgumentParser(description="补充图：missingness / stimulus coverage 可视化（基于 audit_lss_index_coverage 输出）")
    p.add_argument("--coverage-csv", type=Path, default=None, help="audit_*_coverage_by_stimulus.csv（不填则自动找最新）")
    p.add_argument("--missing-csv", type=Path, default=None, help="audit_*_missing_by_subject.csv（不填则自动找最新）")
    p.add_argument("--missing-matrix", type=Path, default=None, help="audit_*_missing_matrix__TASK.csv（可选，不填则尝试自动找）")
    p.add_argument("--task", type=str, default="", help="只绘制某个 task（如 EMO/SOC），为空表示不筛选")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认使用 FIG_SUPP_DIR 或 zz_analysis/figures_supplement）")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="输出格式")
    p.add_argument("--max-matrix-cols", type=int, default=200, help="missing_matrix 时最多绘制的刺激列数（按覆盖率排序截断）")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = _default_roots()

    coverage_csv = Path(args.coverage_csv) if args.coverage_csv is not None else _find_latest("audit_*_coverage_by_stimulus.csv", roots)
    missing_csv = Path(args.missing_csv) if args.missing_csv is not None else _find_latest("audit_*_missing_by_subject.csv", roots)
    matrix_csv = Path(args.missing_matrix) if args.missing_matrix is not None else _find_latest("audit_*_missing_matrix__*.csv", roots)

    cov = _read_optional(coverage_csv)
    miss = _read_optional(missing_csv)

    task = str(args.task).strip()
    if cov is not None and task:
        if "task" in cov.columns:
            cov = cov[cov["task"].astype(str) == task].copy()
    if miss is not None and task:
        if "task" in miss.columns:
            miss = miss[miss["task"].astype(str) == task].copy()

    sns.set_context("paper", font_scale=1.15)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.25, hspace=0.32)
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_miss = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    title_task = f" ({task})" if task else ""
    fig.suptitle(f"Supplementary: Missingness & Stimulus Coverage{title_task}", y=1.01)

    if cov is None or cov.empty:
        ax_cov.axis("off")
        ax_cov.text(0.5, 0.5, "Missing coverage_by_stimulus CSV", ha="center", va="center")
    else:
        n_subjects = int(pd.to_numeric(cov.get("n_subjects", np.nan), errors="coerce").max()) if "n_subjects" in cov.columns else None
        x = pd.to_numeric(cov.get("n_subjects", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        if x.size == 0:
            ax_cov.axis("off")
            ax_cov.text(0.5, 0.5, "No valid n_subjects in coverage CSV", ha="center", va="center")
        else:
            sns.histplot(x, bins=min(40, max(10, int(np.sqrt(x.size)))), ax=ax_cov, color="#4c72b0", alpha=0.7)
            ax_cov.set_title("Stimulus Coverage (n_subjects per stimulus)")
            ax_cov.set_xlabel("n_subjects")
            ax_cov.set_ylabel("stimuli")
            if n_subjects is not None and np.isfinite(n_subjects):
                ax_cov.axvline(float(n_subjects), color="gray", linestyle="--", linewidth=1.2, alpha=0.75)
            ax_cov.text(0.98, 0.93, f"N_stimuli={x.size}", transform=ax_cov.transAxes, ha="right", va="top", fontsize=9)

    if miss is None or miss.empty:
        ax_miss.axis("off")
        ax_miss.text(0.5, 0.5, "Missing missing_by_subject CSV", ha="center", va="center")
    else:
        frac = pd.to_numeric(miss.get("frac_missing", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        if frac.size == 0:
            ax_miss.axis("off")
            ax_miss.text(0.5, 0.5, "No valid frac_missing in missing CSV", ha="center", va="center")
        else:
            sns.histplot(frac, bins=30, ax=ax_miss, color="#55a868", alpha=0.7)
            ax_miss.set_title("Subject Missingness (frac_missing)")
            ax_miss.set_xlabel("frac_missing")
            ax_miss.set_ylabel("subjects")
            ax_miss.set_xlim(0, max(0.02, float(np.nanmax(frac)) * 1.02))
            ax_miss.text(0.98, 0.93, f"N_subjects={frac.size}", transform=ax_miss.transAxes, ha="right", va="top", fontsize=9)

    if matrix_csv is None or not matrix_csv.exists():
        ax_heat.axis("off")
        ax_heat.text(0.5, 0.5, "Missing subject×stimulus matrix CSV (optional)", ha="center", va="center")
    else:
        try:
            mat = pd.read_csv(matrix_csv, index_col=0)
            if mat.empty:
                raise ValueError("empty matrix")
            mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            col_means = mat.mean(axis=0).sort_values(ascending=False)
            keep_cols = col_means.index.astype(str).tolist()[: int(args.max_matrix_cols)]
            mat = mat[keep_cols]
            if mat.shape[0] > 160:
                step = int(np.ceil(mat.shape[0] / 160.0))
                mat = mat.iloc[::step, :]
            sns.heatmap(mat, ax=ax_heat, cmap="Greys", cbar_kws={"label": "beta_exists (0/1)"}, xticklabels=False, yticklabels=False)
            ax_heat.set_title(f"Subject × Stimulus Existence Matrix ({matrix_csv.name})")
            ax_heat.set_xlabel("stimulus_content (subset)")
            ax_heat.set_ylabel("subjects (subset)")
        except Exception:
            ax_heat.axis("off")
            ax_heat.text(0.5, 0.5, f"Failed to read matrix: {matrix_csv}", ha="center", va="center")

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_fig_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"S_missingness{title_task.replace(' ', '_')}.{str(args.format)}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
