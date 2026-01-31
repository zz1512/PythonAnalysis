#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_parcel_sig_maps.py

目的
-
把 parcel 级显著性结果（来自 parcel_joint_analysis_dev_models_perm_fwer.py）
转换为“顶点级” func.gii，便于用 Connectome Workbench（wb_view）直接上脑图查看；
同时导出每个模型的显著 parcel 列表，并画一个 Top-K 条形图。

输入
-
- --result-csv: parcel_joint_analysis_dev_models_perm_fwer__*.csv
  必需列：parcel_id, model, r_obs, p_fwer
- --atlas-label:
  - surface：Schaefer label.gii（必须与分析的半球一致）
  - volume：Schaefer 体素图谱 label.nii / label.nii.gz（与分析时使用的 atlas_label_used 一致更好）

输出（每个 model 一套）
-
- sig_parcels__{model}.csv：显著 parcel 列表（已按 p_fwer、r_obs 排序）
- sig_r__{model}.func.gii：显著 parcel 的 r 值映射到顶点（非显著为 0）
- sig_mask__{model}.func.gii：显著 parcel mask（显著为 1，非显著为 0）
- top_sig_parcels__{model}.png：Top-K 显著 parcel 条形图
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting

@dataclass(frozen=True)
class ParcelPlotPreset:
    name: str
    result_csv: Path
    atlas_label: Path
    out_dir: Path
    alpha: float = 0.05
    positive_only: bool = True
    top_k: int = 20


ACTIVE_PLOT_PRESET = "EXAMPLE_surface_L_EMO_all"


PRESETS_PLOT: Dict[str, ParcelPlotPreset] = {
    "EXAMPLE_surface_L_EMO_all": ParcelPlotPreset(
        name="EXAMPLE_surface_L_EMO_all",
        result_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/parcel_joint_analysis_dev_models_perm_fwer__surface_L__EMO__all.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/figures_surface_L_EMO_all"),
        alpha=0.05,
        positive_only=True,
        top_k=20,
    ),
    "EXAMPLE_surface_R_EMO_all": ParcelPlotPreset(
        name="EXAMPLE_surface_R_EMO_all",
        result_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/parcel_joint_analysis_dev_models_perm_fwer__surface_R__EMO__all.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/figures_surface_R_EMO_all"),
        alpha=0.05,
        positive_only=True,
        top_k=20,
    ),
    "EXAMPLE_volume_EMO_all": ParcelPlotPreset(
        name="EXAMPLE_volume_EMO_all",
        result_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/parcel_joint_analysis_dev_models_perm_fwer__volume__EMO__all.csv"),
        atlas_label=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/atlas_resampled__volume__EMO__all.nii.gz"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/figures_volume_EMO_all"),
        alpha=0.05,
        positive_only=True,
        top_k=20,
    ),
}


def choose_preset(name: str, presets: Dict[str, ParcelPlotPreset]) -> ParcelPlotPreset:
    if name in presets:
        return presets[name]
    keys = sorted(presets.keys())
    if not keys:
        raise ValueError("未配置任何 preset。")
    print("可用 presets：")
    for i, k in enumerate(keys, start=1):
        print(f"  [{i}] {k}")
    s = input(f"请输入编号 (1-{len(keys)}): ").strip()
    idx = int(s) - 1
    if idx < 0 or idx >= len(keys):
        raise ValueError("编号超出范围。")
    return presets[keys[idx]]

def load_label_gii(path: Path) -> np.ndarray:
    # label.gii: 每个顶点一个整数标签（parcel 编号）
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.int32)


def _is_nifti(path: Path) -> bool:
    suf = "".join(path.suffixes).lower()
    return suf.endswith(".nii") or suf.endswith(".nii.gz")


def load_label_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def save_nifti_like(atlas_img: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=atlas_img.affine, header=atlas_img.header)
    nib.save(img, str(out_path))


def save_func_gii(values: np.ndarray, out_path: Path) -> None:
    # 将顶点级数值写为 Gifti func.gii（Workbench 友好）
    da = nib.gifti.GiftiDataArray(values.astype(np.float32), intent="NIFTI_INTENT_ESTIMATE")
    img = nib.gifti.GiftiImage(darrays=[da])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def main() -> None:
    preset = choose_preset(ACTIVE_PLOT_PRESET, PRESETS_PLOT)

    df = pd.read_csv(preset.result_csv)
    required = {"parcel_id", "model", "r_obs", "p_fwer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"结果文件缺少列: {sorted(missing)}")

    atlas_path = preset.atlas_label
    if (not atlas_path.exists()) and ("atlas_label_used" in df.columns) and df["atlas_label_used"].notna().any():
        cand = str(df["atlas_label_used"].dropna().iloc[0])
        atlas_path = Path(cand)

    is_volume = _is_nifti(atlas_path)
    if is_volume:
        atlas_img = load_label_nifti(atlas_path)
        labels_vol = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
        parcel_ids = sorted(set(int(x) for x in np.unique(labels_vol) if int(x) > 0))
    else:
        labels = load_label_gii(atlas_path)
        parcel_ids = sorted(set(int(x) for x in np.unique(labels) if int(x) > 0))

    models = sorted(df["model"].astype(str).unique().tolist())
    out_dir = preset.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=1.3)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    sig_tables: Dict[str, pd.DataFrame] = {}
    for model in models:
        # 1) 取当前模型的结果，并按显著性/效应量排序
        sub = df[df["model"] == model].copy()
        sub["parcel_id"] = sub["parcel_id"].astype(int)
        sub = sub[sub["parcel_id"].isin(parcel_ids)].copy()
        sub = sub.sort_values(["p_fwer", "r_obs"], ascending=[True, False])

        # 2) 显著性筛选（默认用 p_fwer；可选只保留正相关）
        sig = sub[sub["p_fwer"] < float(preset.alpha)].copy()
        if bool(preset.positive_only):
            sig = sig[sig["r_obs"] > 0].copy()

        sig_tables[model] = sig
        sig.to_csv(out_dir / f"sig_parcels__{model}.csv", index=False)

        # 3) 把显著 parcel 的数值“扩展回顶点”
        #    - r 图：显著 parcel 顶点赋值为对应 r_obs，其他为 0
        #    - mask 图：显著 parcel 顶点赋值为 1，其他为 0
        if is_volume:
            r_map = {int(pid): float(r) for pid, r in zip(sig["parcel_id"].tolist(), sig["r_obs"].tolist())} if not sig.empty else {}
            v_r = np.zeros(labels_vol.shape, dtype=np.float32)
            v_mask = np.zeros(labels_vol.shape, dtype=np.float32)
            for pid, r in r_map.items():
                m = labels_vol == int(pid)
                v_r[m] = np.float32(r)
                v_mask[m] = np.float32(1.0)
            save_nifti_like(atlas_img, v_r, out_dir / f"sig_r__{model}.nii.gz")
            save_nifti_like(atlas_img, v_mask, out_dir / f"sig_mask__{model}.nii.gz")
            try:
                fig = plotting.plot_stat_map(out_dir / f"sig_r__{model}.nii.gz", title=f"sig_r__{model}", colorbar=True)
                fig.savefig(out_dir / f"sig_r__{model}.png", dpi=300, bbox_inches="tight")
                fig.close()
            except Exception:
                pass
        else:
            v_r = np.zeros(labels.shape[0], dtype=np.float32)
            v_mask = np.zeros(labels.shape[0], dtype=np.float32)
            if not sig.empty:
                r_map = {int(pid): float(r) for pid, r in zip(sig["parcel_id"].tolist(), sig["r_obs"].tolist())}
                for pid, r in r_map.items():
                    m = labels == int(pid)
                    v_r[m] = np.float32(r)
                    v_mask[m] = np.float32(1.0)
            save_func_gii(v_r, out_dir / f"sig_r__{model}.func.gii")
            save_func_gii(v_mask, out_dir / f"sig_mask__{model}.func.gii")

        # 4) Top-K 条形图（快速报告“哪些编号最显著/效应最大”）
        top_k = int(preset.top_k)
        sig_top = sig.head(top_k).copy()
        if not sig_top.empty:
            fig_h = max(3.8, 0.22 * len(sig_top) + 1.6)
            fig, ax = plt.subplots(figsize=(7.2, fig_h))
            sig_top = sig_top.sort_values("r_obs", ascending=True)
            sns.barplot(data=sig_top, y="parcel_id", x="r_obs", ax=ax, color="#4c72b0")
            ax.set_title(f"Top-{min(top_k, len(sig_top))} Significant Parcels ({model}; p_fwer<{preset.alpha:g})")
            ax.set_xlabel("r_obs")
            ax.set_ylabel("parcel_id")
            plt.tight_layout()
            fig.savefig(out_dir / f"top_sig_parcels__{model}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
