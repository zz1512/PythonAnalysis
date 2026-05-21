#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roi_isc_age_trajectory.py

基于 Step2 的 ROI-ISC（被试×被试），为指定 stimulus_type 与模型筛选 Top ROI，
绘制被试对平均年龄 vs ISC 的散点图，并拟合回归/lowess 曲线。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="emo_ultra: Pair mean age vs ROI ISC trajectory plot")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--repr-prefix", type=str, default=None)
    p.add_argument("--stimulus-type", type=str, required=True)
    p.add_argument("--rois", nargs="+", default=None, help="Optional ROI list. If provided, skip automatic top-ROI selection.")
    p.add_argument("--model", type=str, default="M_conv", choices=["M_nn", "M_conv", "M_div"])
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=["spearman", "pearson", "euclidean", "mahalanobis"])
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--method", type=str, default="fdr_model_wise", choices=["fwer", "fdr_model_wise", "fdr_global", "raw_p"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--fit", type=str, default="lowess", choices=["linear", "poly2", "poly3", "lowess"])
    p.add_argument("--lowess", action="store_true")
    p.add_argument("--plot-mode", type=str, default="hexbin", choices=["hexbin", "scatter"])
    p.add_argument("--max-points", type=int, default=40000)
    p.add_argument("--fit-max-points", type=int, default=5000)
    p.add_argument("--hexbin-gridsize", type=int, default=55)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
        stim_dir / f"{repr_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)


def detect_isc_prefix(stim_dir: Path) -> List[str]:
    prefixes = []
    for npy in sorted(stim_dir.glob("roi_isc_*_by_age.npy")):
        prefix = npy.stem
        if (stim_dir / f"{prefix}_subjects_sorted.csv").exists() and (stim_dir / f"{prefix}_rois.csv").exists():
            prefixes.append(prefix)
    return prefixes


def resolve_isc_prefix(stim_dir: Path, isc_prefix: Optional[str], isc_method: Optional[str]) -> str:
    if isc_prefix is not None:
        return str(isc_prefix)
    found = detect_isc_prefix(stim_dir)
    if isc_method is not None:
        expected = f"roi_isc_{str(isc_method).strip().lower()}_by_age"
        if expected in found:
            return expected
        raise FileNotFoundError(f"未找到指定 isc-method 对应产物: {expected}（可用: {found}）")
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        raise FileNotFoundError(f"{stim_dir} 下未找到 roi_isc_*_by_age.npy")
    raise ValueError(f"{stim_dir} 下存在多个 ISC 产物前缀: {found}，请指定 --isc-prefix 或 --isc-method")


def select_top_rois(result_csv: Path, model: str, method: str, alpha: float, top_k: int, positive_only: bool) -> pd.DataFrame:
    df = pd.read_csv(result_csv)
    need = {"roi", "model", "r_obs", "p_perm_one_tailed", "p_fwer_model_wise", "p_fdr_bh_model_wise", "p_fdr_bh_global"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{result_csv} 缺少列: {sorted(miss)}")
    df = df[np.isfinite(df["r_obs"].to_numpy(dtype=float)) & np.isfinite(df["p_perm_one_tailed"].to_numpy(dtype=float))].copy()
    if df.empty:
        raise ValueError("无可用 ROI（r_obs 或 p_perm_one_tailed 全部无效）")
    sub = df[df["model"].astype(str) == str(model)].copy()
    if sub.empty:
        raise ValueError("无可用 ROI（指定 model 过滤后为空）")
    if bool(positive_only):
        sub = sub[sub["r_obs"] > 0].copy()
        if sub.empty:
            raise ValueError("无可用 ROI（positive_only 过滤后为空）")

    if method == "fwer":
        sub = sub[np.isfinite(sub["p_fwer_model_wise"].to_numpy(dtype=float))].copy()
        sub = sub[sub["p_fwer_model_wise"] <= float(alpha)]
        sub = sub.sort_values(["p_fwer_model_wise", "r_obs"], ascending=[True, False])
    elif method == "raw_p":
        sub = sub[sub["p_perm_one_tailed"] <= float(alpha)]
        sub = sub.sort_values(["p_perm_one_tailed", "r_obs"], ascending=[True, False])
    elif method == "fdr_model_wise":
        sub = sub[np.isfinite(sub["p_fdr_bh_model_wise"].to_numpy(dtype=float))].copy()
        sub = sub[sub["p_fdr_bh_model_wise"] <= float(alpha)]
        sub = sub.sort_values(["p_fdr_bh_model_wise", "r_obs"], ascending=[True, False])
    elif method == "fdr_global":
        sub = sub[np.isfinite(sub["p_fdr_bh_global"].to_numpy(dtype=float))].copy()
        sub = sub[sub["p_fdr_bh_global"] <= float(alpha)]
        sub = sub.sort_values(["p_fdr_bh_global", "r_obs"], ascending=[True, False])
    else:
        raise ValueError(f"未知 method: {method}")

    if sub.empty:
        raise ValueError("在当前阈值下无显著 ROI")
    return sub.head(int(top_k)).reset_index(drop=True)


def load_pair_data(stim_dir: Path, roi: str, isc_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    isc = np.load(stim_dir / f"{isc_prefix}.npy")
    subs = pd.read_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv")
    rois = pd.read_csv(stim_dir / f"{isc_prefix}_rois.csv")["roi"].astype(str).tolist()
    if roi not in rois:
        raise ValueError(f"ROI 不存在于该 stimulus_type: {roi}")
    ri = int(rois.index(roi))
    ages = subs["age"].astype(float).to_numpy()
    n_sub = int(ages.size)
    iu, ju = np.triu_indices(n_sub, k=1)
    # Each point is a subject-pair: x = mean age of the pair, y = ISC value for that pair.
    x = 0.5 * (ages[iu] + ages[ju])
    y = isc[ri][iu, ju].astype(np.float32)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def validate_rois(stim_dir: Path, rois: List[str], isc_prefix: str) -> List[str]:
    roi_list = pd.read_csv(stim_dir / f"{isc_prefix}_rois.csv")["roi"].astype(str).tolist()
    keep: List[str] = []
    missing: List[str] = []
    roi_set = set(roi_list)
    for roi in [str(x) for x in rois]:
        if roi in roi_set:
            keep.append(roi)
        else:
            missing.append(roi)
    if missing:
        raise ValueError(f"ROIs not found in {stim_dir.name}: {missing}")
    return keep


def plot_one(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    y_label: str,
    fit: str,
    plot_mode: str,
    max_points: int,
    fit_max_points: int,
    hexbin_gridsize: int,
    seed: int,
    dpi: int,
) -> None:
    df = pd.DataFrame({"pair_mean_age": x, "isc": y})
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    rng = np.random.default_rng(int(seed))
    if int(df.shape[0]) > int(max_points):
        # Downsample for plotting to avoid overdraw and huge figure files.
        idx = rng.choice(df.shape[0], size=int(max_points), replace=False)
        df_plot = df.iloc[idx].copy()
    else:
        df_plot = df

    df_fit = df_plot
    if str(fit) == "lowess" and int(df_fit.shape[0]) > int(fit_max_points):
        # LOWESS is expensive; use an additional downsample for the fit curve only.
        idx2 = rng.choice(df_fit.shape[0], size=int(fit_max_points), replace=False)
        df_fit = df_fit.iloc[idx2].copy()

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.5))
    if str(plot_mode) == "hexbin":
        ax.hexbin(
            df_plot["pair_mean_age"].to_numpy(dtype=float),
            df_plot["isc"].to_numpy(dtype=float),
            gridsize=int(hexbin_gridsize),
            cmap="viridis",
            mincnt=1,
            linewidths=0,
        )
    else:
        ax.scatter(
            df_plot["pair_mean_age"].to_numpy(dtype=float),
            df_plot["isc"].to_numpy(dtype=float),
            s=8,
            alpha=0.12,
            linewidths=0,
        )

    fit_mode = str(fit)
    if fit_mode == "lowess":
        sns.regplot(
            data=df_fit,
            x="pair_mean_age",
            y="isc",
            ax=ax,
            scatter=False,
            line_kws={"color": "#d62728", "lw": 2.0},
            lowess=True,
            ci=None,
            truncate=False,
        )
    else:
        order = 1
        if fit_mode == "poly2":
            order = 2
        elif fit_mode == "poly3":
            order = 3
        sns.regplot(
            data=df_plot,
            x="pair_mean_age",
            y="isc",
            ax=ax,
            scatter=False,
            line_kws={"color": "#d62728", "lw": 2.0},
            order=int(order),
            ci=95,
            truncate=False,
        )
    ax.set_title(title)
    ax.set_xlabel("Pair mean age")
    ax.set_ylabel(str(y_label))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stim_dir = Path(args.matrix_dir) / str(args.stimulus_dir_name) / str(args.stimulus_type)
    if not stim_dir.exists():
        raise FileNotFoundError(f"未找到条件目录: {stim_dir}")
    if args.repr_prefix is not None and not has_repr_files(stim_dir, repr_prefix=str(args.repr_prefix)):
        print(f"[SKIP] {stim_dir.name}: 缺少 {args.repr_prefix} 对应输入文件，跳过。")
        return

    result_csv = stim_dir / "roi_isc_dev_models_perm_fwer.csv"
    isc_prefix = resolve_isc_prefix(stim_dir, isc_prefix=args.isc_prefix, isc_method=args.isc_method)
    if args.rois:
        roi_names = validate_rois(stim_dir, rois=[str(x) for x in args.rois], isc_prefix=str(isc_prefix))
        top = pd.DataFrame({"roi": roi_names})
    else:
        top = select_top_rois(
            result_csv=result_csv,
            model=str(args.model),
            method=str(args.method),
            alpha=float(args.alpha),
            top_k=int(args.top_k),
            positive_only=bool(args.positive_only),
        )

    fig_dir = Path(args.matrix_dir) / "figures" / f"{str(args.stimulus_dir_name)}_pair_age_traj"
    fit = "lowess" if bool(args.lowess) else str(args.fit)
    y_label = "ISC"
    if "pearson" in str(isc_prefix).lower():
        y_label = "ISC (Pearson)"
    elif "spearman" in str(isc_prefix).lower():
        y_label = "ISC (Spearman)"
    for _, row in top.iterrows():
        roi = str(row["roi"])
        x, y = load_pair_data(stim_dir, roi, isc_prefix=str(isc_prefix))
        if x.size <= 10:
            continue
        method_tag = str(args.method)
        if bool(args.positive_only):
            method_tag += "_pos"
        out_path = fig_dir / (
            f"emo_ultra_pair_age_traj__{args.stimulus_dir_name}__{args.stimulus_type}__"
            f"{args.model}__{roi}__{method_tag}_a{args.alpha:g}.png"
        )
        title = f"{args.stimulus_type} | {args.model} | {roi}"
        plot_one(
            out_path,
            x,
            y,
            title=title,
            y_label=str(y_label),
            fit=fit,
            plot_mode=str(args.plot_mode),
            max_points=int(args.max_points),
            fit_max_points=int(args.fit_max_points),
            hexbin_gridsize=int(args.hexbin_gridsize),
            seed=int(args.seed),
            dpi=int(args.dpi),
        )
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
