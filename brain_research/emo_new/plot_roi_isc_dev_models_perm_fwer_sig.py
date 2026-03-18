#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emo_new：按 3 个 stimulus_type 条件绘制 ROI ISC 显著热图（FWER 版本）。

输入默认来自：
  <roi_results>/by_stimulus/<stimulus_type>/roi_isc_dev_models_perm_fwer.csv

输出：
1) 三条件并排热图（每个条件一个子图）
2) 显著结果汇总表（包含 stimulus_type 列）
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_L_ATLAS = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
DEFAULT_R_ATLAS = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")


def _default_roi_root() -> Path:
    return Path(os.environ.get("ROI_RESULT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"))


def _default_fig_out_dir(roi_root: Path) -> Path:
    p = os.environ.get("FIG_ROI_DIR", "")
    if p:
        return Path(p)
    return roi_root / "figures"


def _model_sort_key(name: str) -> int:
    order = {"M_nn": 0, "M_conv": 1, "M_div": 2}
    return order.get(name, 9)


def _parse_roi_token(roi: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^([LRV])_(\d+)$", str(roi).strip())
    if not m:
        return str(roi), None
    return m.group(1), int(m.group(2))


def _load_surface_label_map(label_gii: Path) -> Dict[int, str]:
    if not label_gii.exists():
        return {}
    try:
        gii = nib.load(str(label_gii))
        table = gii.labeltable
        if table is None:
            return {}
        out: Dict[int, str] = {}
        for lb in table.labels:
            key = int(lb.key)
            if key <= 0:
                continue
            name = str(lb.label).strip() if lb.label is not None else ""
            out[key] = name if name else f"ROI_{key}"
        return out
    except Exception:
        return {}


def _build_roi_display(roi: str, map_l: Dict[int, str], map_r: Dict[int, str]) -> str:
    prefix, rid = _parse_roi_token(roi)
    if rid is None:
        return str(roi)
    label = map_l.get(rid, "") if prefix == "L" else map_r.get(rid, "") if prefix == "R" else ""
    if label:
        return f"{roi} | id={rid} | {label}"
    return f"{roi} | id={rid}"


def _collect_condition_csvs(roi_root: Path) -> Dict[str, Path]:
    by_stim = roi_root / "by_stimulus"
    if not by_stim.exists():
        raise FileNotFoundError(f"未找到目录: {by_stim}")
    out: Dict[str, Path] = {}
    for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        csv_path = d / "roi_isc_dev_models_perm_fwer.csv"
        if csv_path.exists():
            out[d.name] = csv_path
    if not out:
        raise FileNotFoundError(f"{by_stim} 下未找到 roi_isc_dev_models_perm_fwer.csv")
    return out


def _pick_three_conditions(all_conds: List[str], arg_stimuli: Optional[str]) -> List[str]:
    if arg_stimuli:
        picked = [x.strip() for x in arg_stimuli.split(",") if x.strip()]
    else:
        picked = sorted(all_conds)[:3]
    if len(picked) != 3:
        raise ValueError(f"需要恰好 3 个条件，当前={picked}")
    return picked


def _pick_conditions(all_conds: List[str], arg_stimuli: Optional[str]) -> List[str]:
    if arg_stimuli:
        picked = [x.strip() for x in arg_stimuli.split(",") if x.strip()]
    else:
        picked = sorted(all_conds)
    if not picked:
        raise ValueError("未选择任何条件")
    return picked


def _prepare_sig_df(result_csv: Path, alpha: float, positive_only: bool) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(result_csv)
    p_col = None
    for c in ["p_fwer_model_wise", "p_fwer", "p_fwer_one_tailed"]:
        if c in df.columns:
            p_col = c
            break
    if p_col is None:
        raise ValueError(f"{result_csv} 缺少 FWER p 列")

    req = {"roi", "model", "r_obs", p_col}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{result_csv} 缺少列: {sorted(miss)}")

    sub = df.copy()
    sub = sub[np.isfinite(sub["r_obs"]) & np.isfinite(sub[p_col])]
    if positive_only:
        sub = sub[sub["r_obs"] > 0]
    sub = sub[sub[p_col] <= float(alpha)]
    return sub, p_col


def plot_three_conditions(
    roi_root: Path,
    out_dir: Path,
    stimuli_arg: Optional[str],
    alpha: float,
    positive_only: bool,
    atlas_l: Path,
    atlas_r: Path,
    dpi: int,
    fmt: str,
) -> List[Path]:
    cond2csv = _collect_condition_csvs(roi_root)
    stimuli = _pick_three_conditions(list(cond2csv.keys()), stimuli_arg)

    sig_by_cond: Dict[str, pd.DataFrame] = {}
    p_col_ref: Optional[str] = None
    for stim in stimuli:
        if stim not in cond2csv:
            raise FileNotFoundError(f"条件 {stim} 缺少结果文件")
        sub, p_col = _prepare_sig_df(cond2csv[stim], alpha=float(alpha), positive_only=bool(positive_only))
        sig_by_cond[stim] = sub
        if p_col_ref is None:
            p_col_ref = p_col

    if p_col_ref is None:
        raise ValueError("未找到 p 列")

    union = []
    for stim in stimuli:
        sub = sig_by_cond[stim]
        if not sub.empty:
            union.append(sub[["roi", p_col_ref]].groupby("roi", as_index=False)[p_col_ref].min())
    if not union:
        raise ValueError("3 个条件下都没有显著 ROI，无法绘图")

    merged = pd.concat(union, axis=0, ignore_index=True)
    roi_order = merged.groupby("roi", as_index=False)[p_col_ref].min().sort_values(p_col_ref)["roi"].astype(str).tolist()
    models = ["M_nn", "M_conv", "M_div"]

    map_l = _load_surface_label_map(atlas_l)
    map_r = _load_surface_label_map(atlas_r)
    ylabels = [_build_roi_display(r, map_l, map_r) for r in roi_order]

    mats_r: Dict[str, pd.DataFrame] = {}
    mats_p: Dict[str, pd.DataFrame] = {}
    vmax = 0.1
    for stim in stimuli:
        sub = sig_by_cond[stim]
        if sub.empty:
            mat_r = pd.DataFrame(np.nan, index=roi_order, columns=models)
            mat_p = pd.DataFrame(np.nan, index=roi_order, columns=models)
        else:
            mat_r = sub.pivot(index="roi", columns="model", values="r_obs").reindex(index=roi_order, columns=models)
            mat_p = sub.pivot(index="roi", columns="model", values=p_col_ref).reindex(index=roi_order, columns=models)
        mats_r[stim] = mat_r
        mats_p[stim] = mat_p
        v = mat_r.to_numpy(dtype=float)
        if np.isfinite(v).any():
            vmax = max(vmax, float(np.nanmax(np.abs(v))))

    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig_h = max(4.5, 0.26 * len(roi_order) + 2.2)
    fig_w = 5.2 * 3
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharey=True)

    for i, stim in enumerate(stimuli):
        ax = axes[i]
        mat_r = mats_r[stim]
        mat_p = mats_p[stim]
        annot = mat_r.copy().astype(object)
        for r in range(annot.shape[0]):
            for c in range(annot.shape[1]):
                rv = mat_r.iat[r, c]
                pv = mat_p.iat[r, c]
                annot.iat[r, c] = f"{rv:.2f}\n(p={pv:.3g})" if np.isfinite(rv) and np.isfinite(pv) else ""

        hm = sns.heatmap(
            mat_r,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            annot=annot,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=(i == 2),
            cbar_kws={"label": "r (ROI ISC vs. Development Model)"} if i == 2 else None,
        )
        ax.set_title(stim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        if i == 0:
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            ax.set_yticklabels(ylabels, rotation=0)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if hm.collections and i != 2:
            hm.collections[0].colorbar = None

    title = f"emo_new Significant ROI ISC Effects (FWER; {p_col_ref}≤{alpha:g})"
    if positive_only:
        title += "\n(positive r only)"
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    stim_tag = "__".join(stimuli)
    fig_out = out_dir / f"emo_new_roi_isc_dev_models_sig_heatmap_3conds__{stim_tag}__{p_col_ref}__a{alpha:g}.{fmt}"
    fig.savefig(fig_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for stim in stimuli:
        sub = sig_by_cond[stim].copy()
        sub.insert(0, "stimulus_type", stim)
        rows.append(sub)
    sig_table = pd.concat(rows, axis=0, ignore_index=True)
    sig_table = sig_table.sort_values(["stimulus_type", "model", p_col_ref, "r_obs"], ascending=[True, True, True, False])
    csv_out = out_dir / f"emo_new_roi_isc_dev_models_sig_table_3conds__{stim_tag}__{p_col_ref}__a{alpha:g}.csv"
    sig_table.to_csv(csv_out, index=False)

    return [fig_out, csv_out]


def plot_each_condition(
    roi_root: Path,
    out_dir: Path,
    stimuli_arg: Optional[str],
    alpha: float,
    positive_only: bool,
    atlas_l: Path,
    atlas_r: Path,
    dpi: int,
    fmt: str,
) -> List[Path]:
    cond2csv = _collect_condition_csvs(roi_root)
    stimuli = _pick_conditions(list(cond2csv.keys()), stimuli_arg)

    map_l = _load_surface_label_map(atlas_l)
    map_r = _load_surface_label_map(atlas_r)
    models = ["M_nn", "M_conv", "M_div"]
    outs: List[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for stim in stimuli:
        if stim not in cond2csv:
            raise FileNotFoundError(f"条件 {stim} 缺少结果文件")
        sub, p_col = _prepare_sig_df(cond2csv[stim], alpha=float(alpha), positive_only=bool(positive_only))
        if sub.empty:
            continue

        roi_order = (
            sub[["roi", p_col]]
            .groupby("roi", as_index=False)[p_col]
            .min()
            .sort_values(p_col)["roi"]
            .astype(str)
            .tolist()
        )
        ylabels = [_build_roi_display(r, map_l, map_r) for r in roi_order]
        mat_r = sub.pivot(index="roi", columns="model", values="r_obs").reindex(index=roi_order, columns=models)
        mat_p = sub.pivot(index="roi", columns="model", values=p_col).reindex(index=roi_order, columns=models)

        vmax = 0.1
        v = mat_r.to_numpy(dtype=float)
        if np.isfinite(v).any():
            vmax = max(vmax, float(np.nanmax(np.abs(v))))

        fig_h = max(4.5, 0.26 * len(roi_order) + 2.2)
        fig, ax = plt.subplots(1, 1, figsize=(5.8, fig_h))
        annot = mat_r.copy().astype(object)
        for r in range(annot.shape[0]):
            for c in range(annot.shape[1]):
                rv = mat_r.iat[r, c]
                pv = mat_p.iat[r, c]
                annot.iat[r, c] = f"{rv:.2f}\n(p={pv:.3g})" if np.isfinite(rv) and np.isfinite(pv) else ""

        sns.heatmap(
            mat_r,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            annot=annot,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            cbar_kws={"label": "r (ROI ISC vs. Development Model)"},
        )
        ax.set_title(stim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        title = f"emo_new Significant ROI ISC Effects (FWER; {p_col}≤{alpha:g})"
        if positive_only:
            title += "\n(positive r only)"
        fig.suptitle(title, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        fig_out = out_dir / f"emo_new_roi_isc_dev_models_sig_heatmap_single__{stim}__{p_col}__a{alpha:g}.{fmt}"
        fig.savefig(fig_out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outs.append(fig_out)

        sub_out = sub.copy().sort_values(["model", p_col, "r_obs"], ascending=[True, True, False])
        sub_out.insert(0, "stimulus_type", stim)
        csv_out = out_dir / f"emo_new_roi_isc_dev_models_sig_table_single__{stim}__{p_col}__a{alpha:g}.csv"
        sub_out.to_csv(csv_out, index=False)
        outs.append(csv_out)

    if not outs:
        raise ValueError("指定条件下均无显著 ROI，未生成单条件图")
    return outs


def build_arg_parser() -> argparse.ArgumentParser:
    roi_root = _default_roi_root()
    p = argparse.ArgumentParser(description="emo_new: 3 条件 FWER 显著 ROI 热图")
    p.add_argument("--roi-root", type=Path, default=roi_root, help="roi_results 根目录（含 by_stimulus）")
    p.add_argument("--stimuli", type=str, default=None, help="3个条件名，逗号分隔；不填则默认按字母序取前3个")
    p.add_argument("--out-dir", type=Path, default=_default_fig_out_dir(roi_root), help="输出目录")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--atlas-l", type=Path, default=DEFAULT_L_ATLAS)
    p.add_argument("--atlas-r", type=Path, default=DEFAULT_R_ATLAS)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    p.add_argument("--per-condition", action="store_true", help="额外输出每个条件的单独热图")
    p.add_argument("--per-condition-only", action="store_true", help="仅输出每个条件的单独热图，不生成3条件并排图")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    outs: List[Path] = []
    if not bool(args.per_condition_only):
        outs.extend(
            plot_three_conditions(
                roi_root=Path(args.roi_root),
                out_dir=Path(args.out_dir),
                stimuli_arg=args.stimuli,
                alpha=float(args.alpha),
                positive_only=bool(args.positive_only),
                atlas_l=Path(args.atlas_l),
                atlas_r=Path(args.atlas_r),
                dpi=int(args.dpi),
                fmt=str(args.format),
            )
        )
    if bool(args.per_condition) or bool(args.per_condition_only):
        outs.extend(
            plot_each_condition(
                roi_root=Path(args.roi_root),
                out_dir=Path(args.out_dir),
                stimuli_arg=args.stimuli,
                alpha=float(args.alpha),
                positive_only=bool(args.positive_only),
                atlas_l=Path(args.atlas_l),
                atlas_r=Path(args.atlas_r),
                dpi=int(args.dpi),
                fmt=str(args.format),
            )
        )
    for p in outs:
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
