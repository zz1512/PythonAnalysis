#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all_parcels_joint_analysis.py

功能：
1. 不再进行“网络 -> 脑区”的分层分析。
2. 直接提取所有配置的空间（左脑、右脑、Volume）中的所有脑区（Parcels）。
3. 将所有脑区放入同一个池子中进行联合分析，并做全局 FWER 校正。
4. 适用于 Schaefer 200 Parcels 或其他 Atlas。

使用方法：
   直接运行此脚本。请确保底部的 MY_PRESET 和 ATLAS_MAPPING 配置正确。
"""

import argparse
import re
import gc
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# 复用之前的逻辑
from combined_roi_joint_analysis_dev_models_perm_fwer import (
    JointPreset, SimilaritySpec, load_subject_ages_map, determine_subject_list,
    filter_dataframe, resolve_beta_path, zscore_1d, build_dev_models,
    COL_SUB_ID, COL_AGE
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 1. Schaefer Atlas 标签解析 (保持一致) =================

SCHAEFER_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

def parse_schaefer_label(label_name: str) -> Tuple[str, str]:
    """
    解析 Schaefer 标签，返回 (Network, ParcelName)
    虽然本脚本不按网络聚合，但保留此函数以备查看或过滤
    """
    clean_name = label_name.replace("7Networks_LH_", "").replace("7Networks_RH_", "")
    
    found_net = "Other"
    for net in SCHAEFER_NETWORKS:
        if clean_name.startswith(net):
            found_net = net
            break
            
    return found_net, label_name

def parse_schaefer_txt(txt_path: Path) -> Dict[int, str]:
    """
    解析 Schaefer 标准提供的 .txt 文件
    """
    id_to_name = {}
    if not txt_path.exists():
        raise FileNotFoundError(f"指定的标签文件不存在: {txt_path}")
        
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    id_to_name[idx] = name
                except ValueError:
                    continue
    
    logger.info(f"  [Atlas] 已从 {txt_path.name} 加载 {len(id_to_name)} 个脑区标签")
    return id_to_name

def load_atlas_labels(atlas_img_path: Path, atlas_txt_path: Path, space: str) -> Tuple[object, Dict[int, str]]:
    """
    加载 Atlas 图像和标签映射表
    """
    if not atlas_img_path.exists():
        raise FileNotFoundError(f"Atlas 图像文件不存在: {atlas_img_path}")

    # 1. 优先解析 TXT 获取 ID->Name 映射
    id_to_name = parse_schaefer_txt(atlas_txt_path)

    # 2. 加载图像数据
    if space == "volume":
        img = image.load_img(str(atlas_img_path))
        return img, id_to_name

    else:
        path_str = str(atlas_img_path)
        if path_str.endswith('.annot'):
            labels, ctab, names = nib.freesurfer.read_annot(path_str)
            return labels, id_to_name
        elif path_str.endswith('.gii'):
            gii = nib.load(path_str)
            data = gii.darrays[0].data.astype(int)
            return data, id_to_name
        else:
            raise ValueError(f"不支持的 Surface Atlas 格式: {path_str}")

# ================= 2. ROI 数据提取核心 (简化版) =================

def extract_all_parcels(
    spec: SimilaritySpec,
    atlas_config: Dict[str, Path], # 包含 'img' and 'txt'
    df: pd.DataFrame,
    subjects: List[str],
    lss_root: Path,
    preset: JointPreset
) -> Dict[str, np.ndarray]:
    """
    提取指定 Space 下 Atlas 中定义的所有 Parcel 的数据。
    """
    img_path = atlas_config['img']
    txt_path = atlas_config['txt']
    
    logger.info(f"  [Extract] Image: {img_path.name}")
    logger.info(f"  [Extract] Text : {txt_path.name}")
    
    # 1. 加载 Atlas 和 Labels
    atlas_obj, id_to_name = load_atlas_labels(img_path, txt_path, spec.space)
    
    # 建立 ROI 掩膜索引
    roi_indices = {}
    
    # 获取 Atlas Data 数组 (1D)
    if spec.space == "volume":
        atlas_data = atlas_obj.get_fdata().reshape(-1)
    else:
        if isinstance(atlas_obj, np.ndarray):
            atlas_data = atlas_obj.reshape(-1)
        else:
            atlas_data = atlas_obj.reshape(-1)

    unique_ids_in_data = np.unique(atlas_data)
    unique_ids_in_data = unique_ids_in_data[unique_ids_in_data > 0]
    
    valid_roi_count = 0
    
    for uid in unique_ids_in_data:
        if uid not in id_to_name: 
            continue
            
        label_name = id_to_name[uid]
        # 直接使用脑区名称作为 Key
        key = label_name 
            
        idx = np.where(atlas_data == uid)[0]
        
        # 确保每个 key 唯一 (如果 txt 有重名，这里会覆盖或追加，Schaefer 一般无重名)
        if key not in roi_indices:
            roi_indices[key] = []
        roi_indices[key].append(idx)
        valid_roi_count += 1
        
    # 合并索引
    final_masks = {}
    for key, idx_list in roi_indices.items():
        if len(idx_list) > 0:
            final_masks[key] = np.concatenate(idx_list)
            
    if not final_masks:
        logger.warning(f"  [Extract] 在 {spec.space} 中未找到符合条件的 ROI。")
        return {}
        
    logger.info(f"  [Extract] 提取到 {len(final_masks)} 个 Parcels")
    
    # 2. 确定公共刺激
    df_sub = df[df["subject"].isin(subjects)].copy()
    counts = df_sub["stimulus_content"].value_counts()
    threshold = int(len(subjects) * preset.min_coverage)
    keys = counts[counts >= threshold].index.tolist()
    keys.sort()
    
    if not keys: return {}
    
    lookup = {}
    for _, row in df_sub.iterrows():
        lookup[(str(row["subject"]), str(row["stimulus_content"]))] = resolve_beta_path(lss_root, row)

    # 3. 加载 Beta 并提取特征
    roi_data_containers = {k: [] for k in final_masks.keys()} 
    
    for s in tqdm(subjects, desc=f"  Reading {spec.space}", leave=False):
        subj_betas = []
        for k in keys:
            p = lookup.get((s, k))
            if p and p.exists():
                try:
                    if spec.space == "volume":
                        img = image.load_img(str(p))
                        if img.shape != atlas_obj.shape or not np.allclose(img.affine, atlas_obj.affine):
                             img = image.resample_to_img(img, atlas_obj, interpolation="nearest")
                        data = img.get_fdata().reshape(-1)
                    else:
                        g = nib.load(str(p))
                        data = g.darrays[0].data.astype(np.float32).reshape(-1)
                    subj_betas.append(data)
                except Exception:
                    subj_betas.append(None)
            else:
                subj_betas.append(None)

        if any(b is None for b in subj_betas):
            for k in roi_data_containers:
                roi_data_containers[k].append(None)
            continue
            
        subj_stack = np.stack(subj_betas) # (n_stim, n_voxels_total)
        
        for roi_name, voxel_indices in final_masks.items():
            if voxel_indices.max() >= subj_stack.shape[1]: continue
            roi_vals = subj_stack[:, voxel_indices]
            # 空间 Flatten
            roi_feat = roi_vals.flatten()
            roi_data_containers[roi_name].append(roi_feat)
            
    final_matrices = {}
    for roi_name, sub_list in roi_data_containers.items():
        valid_subs = [x for x in sub_list if x is not None]
        if len(valid_subs) != len(subjects): continue
        final_matrices[roi_name] = np.stack(valid_subs)
        
    return final_matrices

# ================= 3. 主分析流程 =================

def run_all_parcels_analysis(
    preset: JointPreset, 
    atlas_mapping: Dict[str, Dict[str, Path]]
):
    print(f"\n{'='*60}")
    print(f"开始全脑区联合分析 (All Parcels Joint Analysis)")
    print(f"包含空间: {[s.name for s in preset.specs]}")
    print(f"{'='*60}")
    
    # 路径准备
    folder_name = f"roi_analysis_all_parcels"
    out_dir = preset.out_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Basic Info
    age_map = load_subject_ages_map(preset.subject_info)
    dfs = []
    for spec in preset.specs:
        df = pd.read_csv(spec.aligned_csv)
        df = filter_dataframe(df, preset.task, preset.condition_group)
        dfs.append(df)
        
    subjects = determine_subject_list(preset, age_map, dfs)
    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    n_subs = len(subjects)
    n_pairs = int(n_subs * (n_subs - 1) / 2)
    
    print(f"被试数量: {n_subs} (Pairs={n_pairs})")
    
    dev_models = build_dev_models(ages, n_subs, preset.normalize_models)
    model_keys = ["M_nn", "M_conv", "M_div"]
    
    roi_isc_vectors = [] 
    roi_meta_info = [] 
    
    # 遍历每个 Space
    for spec, df in zip(preset.specs, dfs):
        atlas_config = atlas_mapping.get(spec.space)
        
        if not atlas_config:
            logger.warning(f"⚠️  跳过 {spec.name}: 未配置 Atlas。")
            continue
            
        try:
            print(f"\n>>> 处理空间: {spec.space}")
            roi_matrices = extract_all_parcels(
                spec, atlas_config, df, subjects, preset.lss_root, preset
            )
        except Exception as e:
            logger.error(f"提取失败 {spec.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

        # 计算 ISC
        iu, ju = np.triu_indices(n_subs, k=1)
        
        for roi_name, data_mat in roi_matrices.items():
            corr_mat = np.corrcoef(data_mat)
            vec_z = zscore_1d(corr_mat[iu, ju])
            
            if np.isnan(vec_z).any(): continue
            
            roi_isc_vectors.append(vec_z)
            roi_meta_info.append({
                "roi": roi_name, 
                "space": spec.space,
                "n_voxels": data_mat.shape[1] // 24 # 粗略估计
            })
            
    if not roi_isc_vectors:
        print("❌ 未提取到数据，终止。")
        return

    # FWER 校正
    S_stack = np.stack(roi_isc_vectors)
    n_rois_total = S_stack.shape[0]
    
    print(f"\n>>> 统计检验: Pool={n_rois_total} ROIs")
    print(f"    - FWER Correction: Across all {n_rois_total} parcels (Spatial)")
    print(f"    - Model Independence: Each model (M_nn, M_conv, M_div) is corrected SEPARATELY")
    
    r_obs = {m: (S_stack @ dev_models[m]) / n_pairs for m in model_keys}
    count_fwer = {m: np.zeros(n_rois_total) for m in model_keys}
    count_raw  = {m: np.zeros(n_rois_total) for m in model_keys}
    
    rng = np.random.default_rng(preset.seed)
    
    for i in tqdm(range(preset.n_perm), desc="Permutation"):
        perm_idx = rng.permutation(n_subs)
        ages_perm = ages[perm_idx]
        models_perm = build_dev_models(ages_perm, n_subs, preset.normalize_models)
        
        for m in model_keys:
            r_perm = (S_stack @ models_perm[m]) / n_pairs
            count_raw[m] += (r_perm >= r_obs[m])
            # Max across ALL parcels (Spatial FWER), but specific to this model
            max_r = r_perm.max()
            count_fwer[m] += (max_r >= r_obs[m])
            
    # 输出
    results = []
    for i, meta in enumerate(roi_meta_info):
        for m in model_keys:
            results.append({
                "roi": meta["roi"],
                "space": meta["space"],
                "model": m,
                "r_obs": float(r_obs[m][i]),
                "p_raw": (count_raw[m][i] + 1) / (preset.n_perm + 1),
                "p_fwer": (count_fwer[m][i] + 1) / (preset.n_perm + 1)
            })
            
    res_df = pd.DataFrame(results)
    out_csv = out_dir / f"roi_stats_all_parcels_perm{preset.n_perm}.csv"
    res_df.to_csv(out_csv, index=False)
    
    sig_df = res_df[res_df["p_fwer"] < 0.05]
    print(f"\n✅ 完成。结果: {out_csv}")
    if not sig_df.empty:
        print(f"🎉 显著结果 (FWER<0.05): {len(sig_df)} 个")
        print(sig_df[["roi", "space", "model", "r_obs", "p_fwer"]].head(20).to_string(index=False))

# ================= 4. 配置入口 =================

if __name__ == "__main__":
    
    # -----------------------------------------------------------
    # 【Atlas 配置区】 
    # -----------------------------------------------------------
    
    TXT_SURFACE = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order_info.txt")
    TXT_VOLUME  = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_MNI152.txt")
    
    ATLAS_MAPPING = {
        "surface_L": {
            "img": Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
            "txt": TXT_SURFACE
        },
        "surface_R": {
            "img": Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
            "txt": TXT_SURFACE
        },
        "volume": {
            "img": Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_MNI152_label.nii.gz"),
            "txt": TXT_VOLUME
        }
    }
    
    # -----------------------------------------------------------
    # 【分析预设】
    # -----------------------------------------------------------
    MY_PRESET = JointPreset(
        name="EMO_Schaefer_All_Parcels",
        specs=(
            # 1. 左脑
            SimilaritySpec(
                name="surface_L",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="combined"
            ),
            # 2. 右脑 (按需取消注释)
            SimilaritySpec(
                name="surface_R",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="combined"
            ),
            # 3. Volume (按需取消注释)
            SimilaritySpec(
                name="volume",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
                space="volume",
                kind="combined",
                # 注意：如果使用 Schaefer Volume Atlas，通常不需要额外的 volume_mask，因为 Atlas 本身就是 Mask
                # 但如果你的 volume_aligned.csv 是全脑的，这里加载 Atlas NIfTI 会自动处理
            ),
        ),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="",
        condition_group="all", 
        n_perm=5000,
        min_coverage=0.9
    )
    
    # 执行分析
    run_all_parcels_analysis(MY_PRESET, ATLAS_MAPPING)
