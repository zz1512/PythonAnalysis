#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制 calc_roi_isc_pearson_direct 后续联合分析（roi_isc_dev_models_perm_fwer*.csv）的显著 ROI 结果。

设计目标：
1) 参考 plot_joint_analysis_dev_models_perm_fwer.py 的热图风格；
2) 结合 ROI 编号与脑区标签（如可从 atlas/标签表中读取）；
3) 只显示多重比较校正后显著（默认 p_fwer_model_wise <= 0.05）的 ROI，避免 232 ROI 全量图过于拥挤。
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


def _find_latest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _auto_find_result_csv() -> Optional[Path]:
    roi_root = Path(os.environ.get("ROI_RESULT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"))
    roots = [Path.cwd(), roi_root, roi_root.parent]
    cands: List[Path] = []
    for r in roots:
        if r.exists():
            cands.extend(r.rglob("roi_isc_dev_models_perm_fwer*.csv"))
    return _find_latest(cands)


def _default_fig_out_dir() -> Path:
    p = os.environ.get("FIG_ROI_DIR", "")
    if p:
        return Path(p)
    roi_root = Path(os.environ.get("ROI_RESULT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"))
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
            if not name:
                name = f"ROI_{key}"
            out[key] = name
        return out
    except Exception:
        return {}


def _load_volume_label_map(label_csv: Optional[Path]) -> Dict[int, str]:
    if label_csv is None or not label_csv.exists():
        return {}
    df = pd.read_csv(label_csv)
    id_col = "id" if "id" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            rid = int(row[id_col])
        except Exception:
            continue
        if rid <= 0:
            continue
        out[rid] = str(row[label_col])
    return out


def _build_roi_display(roi: str, map_l: Dict[int, str], map_r: Dict[int, str], map_v: Dict[int, str]) -> str:
    prefix, rid = _parse_roi_token(roi)
    if rid is None:
        return str(roi)

    if prefix == "L":
        label = map_l.get(rid, "")
    elif prefix == "R":
        label = map_r.get(rid, "")
    elif prefix == "V":
        label = map_v.get(rid, "")
    else:
        label = ""

    if label:
        return f"{roi} | id={rid} | {label}"
    return f"{roi} | id={rid}"


def plot_sig_heatmap(
    result_csv: Path,
    out_dir: Path,
    alpha: float,
    positive_only: bool,
    atlas_l: Path,
    atlas_r: Path,
    volume_label_csv: Optional[Path],
    dpi: int,
    fmt: str,
) -> List[Path]:
    df = pd.read_csv(result_csv)
    p_col = None
    for c in ["p_fwer_model_wise", "p_fwer", "p_fwer_one_tailed"]:
        if c in df.columns:
            p_col = c
            break
    if p_col is None:
        raise ValueError(f"结果文件缺少校正 p 值列，当前列={list(df.columns)}")

    required = {"roi", "model", "r_obs", p_col}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"结果文件缺少列: {sorted(miss)}")

    sub = df.copy()
    sub = sub[np.isfinite(sub["r_obs"]) & np.isfinite(sub[p_col])]
    if positive_only:
        sub = sub[sub["r_obs"] > 0]
    sub = sub[sub[p_col] <= float(alpha)]
    if sub.empty:
        raise ValueError(f"没有满足条件的显著 ROI（alpha={alpha}, positive_only={positive_only}）。")

    models = sorted(sub["model"].astype(str).unique().tolist(), key=_model_sort_key)

    agg = sub.groupby("roi", as_index=False)[p_col].min().sort_values(p_col, ascending=True)
    roi_order = agg["roi"].astype(str).tolist()

    map_l = _load_surface_label_map(atlas_l)
    map_r = _load_surface_label_map(atlas_r)
    map_v = _load_volume_label_map(volume_label_csv)

    display_map = {roi: _build_roi_display(str(roi), map_l, map_r, map_v) for roi in roi_order}

    mat_r = sub.pivot(index="roi", columns="model", values="r_obs").reindex(index=roi_order, columns=models)
    mat_p = sub.pivot(index="roi", columns="model", values=p_col).reindex(index=roi_order, columns=models)

    annot = mat_r.copy().astype(object)
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            rv = mat_r.iat[i, j]
            pv = mat_p.iat[i, j]
            if np.isfinite(rv) and np.isfinite(pv):
                annot.iat[i, j] = f"{rv:.2f}\n(p={pv:.3g})"
            else:
                annot.iat[i, j] = ""

    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    v = mat_r.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.2
    vmax = max(vmax, 0.1)

    fig_h = max(4.2, 0.28 * len(roi_order) + 1.6)
    fig_w = 6.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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
        cbar_kws={"label": "r (ROI ISC vs. Development Model)"},
    )

    ylabels = [display_map.get(str(x), str(x)) for x in mat_r.index.tolist()]
    ax.set_yticklabels(ylabels, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    title = f"Significant ROI ISC Effects (FWER-corrected; {p_col}≤{alpha:g})"
    if positive_only:
        title += "\n(positive r only)"
    ax.set_title(title, pad=12)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"roi_isc_dev_models_sig_heatmap__{p_col}__a{alpha:g}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 同时导出显著 ROI 表（附显示标签）
    sig_table = sub.copy()
    sig_table["roi_display"] = sig_table["roi"].astype(str).map(display_map)
    sig_table = sig_table.sort_values(["model", p_col, "r_obs"], ascending=[True, True, False])
    csv_out = out_dir / f"roi_isc_dev_models_sig_table__{p_col}__a{alpha:g}.csv"
    sig_table.to_csv(csv_out, index=False)

    return [out_path, csv_out]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="绘制 ROI ISC 联合分析中校正后显著 ROI 的热图")
    p.add_argument("--input-csv", type=Path, default=None, help="roi_isc_dev_models_perm_fwer*.csv 路径（不填则自动搜索）")
    p.add_argument("--out-dir", type=Path, default=_default_fig_out_dir(), help="输出目录")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性阈值（校正后 p 值）")
    p.add_argument("--positive-only", action="store_true", help="只保留 r_obs>0 的显著结果")
    p.add_argument("--atlas-l", type=Path, default=DEFAULT_L_ATLAS, help="左半球 label.gii（用于 ROI 标签）")
    p.add_argument("--atlas-r", type=Path, default=DEFAULT_R_ATLAS, help="右半球 label.gii（用于 ROI 标签）")
    p.add_argument("--volume-label-csv", type=Path, default=None, help="体积 ROI 标签表（可选，列示例: id,label）")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    in_csv = Path(args.input_csv) if args.input_csv is not None else _auto_find_result_csv()
    if in_csv is None or not in_csv.exists():
        raise FileNotFoundError("未找到 roi_isc_dev_models_perm_fwer*.csv，请使用 --input-csv 指定。")

    outs = plot_sig_heatmap(
        result_csv=in_csv,
        out_dir=Path(args.out_dir),
        alpha=float(args.alpha),
        positive_only=bool(args.positive_only),
        atlas_l=Path(args.atlas_l),
        atlas_r=Path(args.atlas_r),
        volume_label_csv=Path(args.volume_label_csv) if args.volume_label_csv is not None else None,
        dpi=int(args.dpi),
        fmt=str(args.format),
    )
    for p in outs:
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
