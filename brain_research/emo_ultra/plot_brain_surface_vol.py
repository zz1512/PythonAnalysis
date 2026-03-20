#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_brain_surface_vol.py

Step 5:
将显著的脑区按照 r 值映射回 Schaefer Surface (L/R) 和 Tian Volume，
导出标准 .func.gii 与 .nii.gz 统计图并渲染可视化。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")
ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
ATLAS_VOL = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将显著 ROI r 值映射回大脑")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--stimulus-type", type=str, required=True)
    p.add_argument("--model", type=str, default="M_conv", choices=["M_nn", "M_conv", "M_div"])
    p.add_argument("--sig-method", type=str, default="fwer", choices=["fwer", "fdr_model_wise", "fdr_global", "raw_p"])
    p.add_argument("--sig-col", type=str, default="p_perm_one_tailed", choices=["p_perm_one_tailed"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def load_atlas() -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image, np.ndarray]:
    """加载皮层与体积图谱数据"""
    surf_l = nib.load(str(ATLAS_SURF_L)).darrays[0].data.astype(int)
    surf_r = nib.load(str(ATLAS_SURF_R)).darrays[0].data.astype(int)
    
    vol_img = image.load_img(str(ATLAS_VOL))
    vol_data = np.rint(vol_img.get_fdata()).astype(int)
    return surf_l, surf_r, vol_img, vol_data


def map_values_to_brain(df: pd.DataFrame, surf_l_data: np.ndarray, surf_r_data: np.ndarray, vol_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 Dataframe 将 r_obs 映射回 3 个空间的 numpy array"""
    out_surf_l = np.zeros_like(surf_l_data, dtype=np.float32)
    out_surf_r = np.zeros_like(surf_r_data, dtype=np.float32)
    out_vol = np.zeros_like(vol_data, dtype=np.float32)

    for _, row in df.iterrows():
        roi = str(row["roi"])
        r_val = float(row["r_obs"])
        
        if roi.startswith("L_"):
            roi_id = int(roi.split("_")[1])
            out_surf_l[surf_l_data == roi_id] = r_val
        elif roi.startswith("R_"):
            roi_id = int(roi.split("_")[1])
            out_surf_r[surf_r_data == roi_id] = r_val
        elif roi.startswith("V_"):
            roi_id = int(roi.split("_")[1])
            out_vol[vol_data == roi_id] = r_val

    return out_surf_l, out_surf_r, out_vol


def apply_significance(df: pd.DataFrame, model: str, method: str, sig_col: str, alpha: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        raise ValueError("结果为空，无法进行显著性映射")

    if str(method) == "fwer":
        out = out[out["model"].astype(str) == str(model)].copy()
        if out.empty:
            raise ValueError(f"结果中未找到 model={model} 的记录")
        if "p_fwer_model_wise" not in out.columns:
            raise ValueError("结果文件缺少 p_fwer_model_wise，无法按 FWER 映射")
        sig = out["p_fwer_model_wise"].astype(float) <= float(alpha)
        out["sig_method"] = "fwer"
        out["p_sig"] = out["p_fwer_model_wise"].astype(float)
    elif str(method) == "raw_p":
        out = out[out["model"].astype(str) == str(model)].copy()
        if out.empty:
            raise ValueError(f"结果中未找到 model={model} 的记录")
        if str(sig_col) not in out.columns:
            raise ValueError(f"结果文件缺少列: {sig_col}")
        sig = out[str(sig_col)].astype(float) <= float(alpha)
        out["sig_method"] = f"raw_{sig_col}"
        out["p_sig"] = out[str(sig_col)].astype(float)
    elif str(method) in ("fdr_model_wise", "fdr_global"):
        if str(method) == "fdr_model_wise":
            out = out[out["model"].astype(str) == str(model)].copy()
            if out.empty:
                raise ValueError(f"结果中未找到 model={model} 的记录")
            if "p_fdr_bh_model_wise" not in out.columns:
                raise ValueError("结果文件缺少 p_fdr_bh_model_wise，无法按 FDR 映射")
            sig = out["p_fdr_bh_model_wise"].astype(float) <= float(alpha)
            out["sig_method"] = "fdr_model_wise"
            out["p_sig"] = out["p_fdr_bh_model_wise"].astype(float)
        else:
            out = out[out["model"].astype(str) == str(model)].copy()
            if out.empty:
                raise ValueError(f"结果中未找到 model={model} 的记录")
            if "p_fdr_bh_global" not in out.columns:
                raise ValueError("结果文件缺少 p_fdr_bh_global，无法按 FDR 映射")
            sig = out["p_fdr_bh_global"].astype(float) <= float(alpha)
            out["sig_method"] = "fdr_global"
            out["p_sig"] = out["p_fdr_bh_global"].astype(float)
    else:
        raise ValueError(f"未知 sig-method: {method}")

    out = out.copy()
    out.loc[~sig, "r_obs"] = 0.0
    return out


def save_and_plot_maps(
    out_dir: Path, prefix: str, l_array: np.ndarray, r_array: np.ndarray, 
    vol_img: nib.Nifti1Image, vol_array: np.ndarray, vmax: float, preview: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存为 NIfTI 和 GIfTI 文件（用于专业的 Connectome Workbench 查看）
    # 皮层下体积
    out_vol_img = image.new_img_like(vol_img, vol_array)
    vol_path = out_dir / f"{prefix}_volume.nii.gz"
    out_vol_img.to_filename(str(vol_path))
    
    # 左脑皮层
    gi_l = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(np.asarray(l_array, dtype=np.float32), intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"])]
    )
    path_l = out_dir / f"{prefix}_surf_L.func.gii"
    nib.save(gi_l, str(path_l))
    
    # 右脑皮层
    gi_r = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(np.asarray(r_array, dtype=np.float32), intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"])]
    )
    path_r = out_dir / f"{prefix}_surf_R.func.gii"
    nib.save(gi_r, str(path_r))
    print(f"成功导出神经影像文件至: {out_dir}")

    if not bool(preview):
        return

    # 2. 尝试使用 Nilearn 直接出图预览
    try:
        # 获取 fsaverage 表面模板 (若离线环境可能报错)
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        
        # 绘图容器
        fig = plt.figure(figsize=(12, 8))
        
        # 画左脑外侧面
        ax1 = fig.add_subplot(221, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, l_array, hemi='left', view='lateral',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_left,
            axes=ax1, title="Left Lateral", colorbar=False
        )
        # 画左脑内侧面
        ax2 = fig.add_subplot(222, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, l_array, hemi='left', view='medial',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_left,
            axes=ax2, title="Left Medial", colorbar=False
        )
        # 画右脑外侧面
        ax3 = fig.add_subplot(223, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, r_array, hemi='right', view='lateral',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_right,
            axes=ax3, title="Right Lateral", colorbar=False
        )
        # 画皮层下切片图
        ax4 = fig.add_subplot(224)
        if np.max(np.abs(vol_array)) > 0:
            plotting.plot_stat_map(
                out_vol_img, display_mode='z', cut_coords=3, cmap='RdBu_r',
                vmax=vmax, axes=ax4, colorbar=True, title="Subcortex Volume"
            )
        else:
            ax4.text(0.5, 0.5, 'No Significant Subcortical ROIs', ha='center', va='center')
            ax4.axis("off")
            
        fig.suptitle(f"Brain Projection: {prefix}", fontsize=16)
        png_path = out_dir / f"{prefix}_brain_map.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"成功导出预览图至: {png_path}")
        
    except Exception as e:
        print(f"Nilearn 绘图预览失败 (通常是因为离线环境无法下载 fsaverage)。\n报错信息: {e}\n无需担心，你可以使用 Connectome Workbench 打开生成的 .func.gii 和 .nii.gz 文件进行完美渲染。")


def main():
    args = parse_args()
    stim_dir = Path(args.matrix_dir) / str(args.stimulus_dir_name) / args.stimulus_type
    result_csv = stim_dir / "roi_isc_dev_models_perm_fwer.csv"
    
    if not result_csv.exists():
        raise FileNotFoundError(f"未找到结果文件: {result_csv}")

    df = pd.read_csv(result_csv)

    df_sig = apply_significance(
        df=df,
        model=str(args.model),
        method=str(args.sig_method),
        sig_col=str(args.sig_col),
        alpha=float(args.alpha),
    )
    
    if df_sig["r_obs"].abs().max() == 0:
        print("警告：在当前阈值下没有任何显著的 ROI，将生成空白的大脑图。")

    # 执行映射
    surf_l_data, surf_r_data, vol_img, vol_data = load_atlas()
    l_arr, r_arr, vol_arr = map_values_to_brain(df_sig, surf_l_data, surf_r_data, vol_data)
    
    # 取一个合理的色彩标尺极限，防止被极端离群点拉伸
    vmax = max(0.05, float(df_sig["r_obs"].abs().quantile(0.99)))

    # 保存与绘图
    prefix = f"brain_map_{args.stimulus_type}_{args.model}_{args.sig_method}_a{args.alpha:g}"
    out_dir = Path(args.matrix_dir) / "figures" / "brain_maps"
    
    save_and_plot_maps(out_dir, prefix, l_arr, r_arr, vol_img, vol_arr, vmax, preview=not bool(args.no_preview))

if __name__ == "__main__":
    main()
