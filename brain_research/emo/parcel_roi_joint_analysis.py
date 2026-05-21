#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parcel_roi_joint_analysis.py [Fix: Robust Text Parsing]

功能：
1. 修复：智能解析各种格式的 Atlas 标签文件（兼容 Schaefer info 格式和 Tian 纯文本格式）。
2. 修复：正确识别 Tian Subcortex 脑区，不再将其归类为 "Other"。
3. 保持：严格被试对齐 (Strict Alignment) 和 Model-wise FWER 校正。
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
    JointPreset, SimilaritySpec, load_subject_ages_map,
    determine_subject_list_strict,
    filter_dataframe, resolve_beta_path, zscore_1d, build_dev_models,
    COL_SUB_ID, COL_AGE
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 1. 智能标签解析 (核心修复) =================

SCHAEFER_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
# Tian 模版常见的缩写，用于识别皮下组织
TIAN_SUB_KEYWORDS = ["HIP", "AMY", "THA", "NAc", "GP", "PUT", "CAU"]


def parse_label_category(label_name: str) -> Tuple[str, str]:
    """
    解析标签所属的大类 (Network)。
    1. 优先匹配 Schaefer 7 Networks。
    2. 其次匹配 Tian Subcortex 关键词。
    3. 否则归为 Other。
    """
    # 1. 清洗 Schaefer 前缀
    clean_name = label_name.replace("7Networks_LH_", "").replace("7Networks_RH_", "")

    # 2. 匹配 Schaefer 网络
    for net in SCHAEFER_NETWORKS:
        if clean_name.startswith(net):
            return net, label_name

    # 3. 匹配皮下组织 (Subcortex) - 针对 Tian Atlas
    # 检查是否包含 HIP, AMY 等关键词，且包含方位词 (lh/rh)
    upper_name = label_name.upper()
    if any(k in upper_name for k in TIAN_SUB_KEYWORDS):
        return "Subcortex", label_name

    return "Other", label_name


def parse_robust_txt(txt_path: Path) -> Dict[int, str]:
    """
    【核心修复】智能解析标签文件，兼容多种格式：
    1. 标准 LUT: "1 LabelName 255 0 0"
    2. Tian 纯文本: "LabelName" (行号作为ID)
    3. Schaefer Info: 奇数行Name，偶数行ColorInfo
    """
    id_to_name = {}
    if not txt_path.exists():
        raise FileNotFoundError(f"指定的标签文件不存在: {txt_path}")

    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 策略：遍历每一行，判断是否以数字开头
    # 针对 Schaefer Info 文件的特殊处理：如果检测到交替结构，需要跳过颜色行

    current_auto_id = 1

    for line in lines:
        parts = line.split()
        first_token = parts[0]

        # 情况 A: 以数字开头 (可能是 ID，也可能是 Schaefer 的颜色行)
        if first_token.isdigit():
            possible_id = int(first_token)

            # 检查第二个 token
            if len(parts) > 1:
                second_token = parts[1]
                # 特殊情况：Schaefer Info 文件的偶数行通常是 "1 120 18..."
                # 如果第二个 token 也是数字，这很可能是一行颜色定义，而不是 "ID Name"
                if second_token.isdigit():
                    # 这是一个颜色/几何信息行，跳过
                    continue
                else:
                    # 这是一个标准的 "ID Name" 行
                    name = second_token
                    id_to_name[possible_id] = name
            else:
                # 只有一个数字？罕见，忽略
                continue

        # 情况 B: 以文字开头 (Tian 格式，或 Schaefer Info 的奇数行)
        else:
            # 这一行就是名字
            name = line
            # 使用自动递增的 ID
            id_to_name[current_auto_id] = name
            current_auto_id += 1

    logger.info(f"  [Atlas Parser] 从 {txt_path.name} 解析出 {len(id_to_name)} 个标签")
    if len(id_to_name) > 0:
        # 打印前3个看看是否解析正确
        sample = list(id_to_name.items())[:3]
        logger.info(f"  [Parser Check] Sample: {sample}")

    return id_to_name


def load_atlas_labels(atlas_img_path: Path, atlas_txt_path: Path, space: str) -> Tuple[object, Dict[int, str]]:
    """
    加载 Atlas 图像和标签映射表
    """
    if not atlas_img_path.exists():
        raise FileNotFoundError(f"Atlas 图像文件不存在: {atlas_img_path}")

    # 1. 使用增强的解析器
    id_to_name = parse_robust_txt(atlas_txt_path)

    # 2. 加载图像数据
    if space == "volume":
        img = image.load_img(str(atlas_img_path))
        return img, id_to_name
    else:
        path_str = str(atlas_img_path)
        if path_str.endswith('.annot'):
            labels, ctab, names = nib.freesurfer.read_annot(path_str)
            # 即使是 annot，也优先使用我们解析的 txt 映射，保证名称一致性
            return labels, id_to_name
        elif path_str.endswith('.gii'):
            gii = nib.load(path_str)
            data = gii.darrays[0].data.astype(int)
            return data, id_to_name
        else:
            raise ValueError(f"不支持的 Surface Atlas 格式: {path_str}")


# ================= 2. ROI 数据提取核心 =================

def extract_roi_data(
        spec: SimilaritySpec,
        atlas_config: Dict[str, Path],
        df: pd.DataFrame,
        subjects: List[str],
        lss_root: Path,
        preset: JointPreset,
        target_level: str,
        target_network: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    根据 Atlas 提取每个 ROI 的数据矩阵。
    """
    img_path = atlas_config['img']
    txt_path = atlas_config['txt']

    logger.info(f"  [Extract] Image: {img_path.name}")
    logger.info(f"  [Extract] Text : {txt_path.name}")

    # 1. 加载 Atlas 和 Labels
    atlas_obj, id_to_name = load_atlas_labels(img_path, txt_path, spec.space)

    # 建立 ROI 掩膜索引
    roi_indices = {}

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
    mapped_networks = set()

    for uid in unique_ids_in_data:
        # 如果 Atlas 图片里的 ID 在 Text 里找不到，就跳过 (e.g. 背景 0)
        if uid not in id_to_name:
            continue

        label_name = id_to_name[uid]
        # 解析类别 (Network 或 Subcortex)
        network_name, parcel_name = parse_label_category(label_name)

        # 过滤器
        if target_network and network_name != target_network:
            continue

        # 决定聚合 Key
        if target_level == "network":
            key = network_name
        else:
            key = parcel_name

        idx = np.where(atlas_data == uid)[0]
        if key not in roi_indices:
            roi_indices[key] = []
        roi_indices[key].append(idx)

        valid_roi_count += 1
        mapped_networks.add(network_name)

    final_masks = {}
    for key, idx_list in roi_indices.items():
        if len(idx_list) > 0:
            final_masks[key] = np.concatenate(idx_list)

    if not final_masks:
        logger.warning(f"  [Extract] 在 {spec.space} 中未找到符合条件的 ROI。")
        logger.warning(
            f"  [Debug] 图片中的ID: {unique_ids_in_data[:10]}... 文本中的ID: {list(id_to_name.keys())[:10]}...")
        return {}

    logger.info(f"  [Extract] 成功匹配 {valid_roi_count} 个 ROI 单元 -> 聚合为 {len(final_masks)} 个 {target_level}")
    logger.info(f"  [Extract] 包含的 Networks: {mapped_networks}")

    # 2. 确定公共刺激
    df_sub = df[df["subject"].isin(subjects)].copy()
    if df_sub.empty: return {}

    counts = df_sub["stimulus_content"].value_counts()
    threshold = int(len(subjects) * preset.min_coverage)
    keys = counts[counts >= threshold].index.tolist()
    keys.sort()

    if not keys: return {}

    lookup = {}
    for _, row in df_sub.iterrows():
        lookup[(str(row["subject"]), str(row["stimulus_content"]))] = resolve_beta_path(lss_root, row)

    if preset.ensure_complete_keys:
        valid_keys = []
        for k in keys:
            is_complete = True
            for s in subjects:
                if not (lookup.get((s, k)) and lookup.get((s, k)).exists()):
                    is_complete = False;
                    break
            if is_complete: valid_keys.append(k)
        keys = valid_keys

    if not keys: return {}

    # 3. 加载 Beta
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
            for k in roi_data_containers: roi_data_containers[k].append(None)
            continue

        subj_stack = np.stack(subj_betas)

        for roi_name, voxel_indices in final_masks.items():
            if voxel_indices.max() >= subj_stack.shape[1]: continue
            roi_vals = subj_stack[:, voxel_indices]
            roi_feat = roi_vals.flatten()
            roi_data_containers[roi_name].append(roi_feat)

    final_matrices = {}
    for roi_name, sub_list in roi_data_containers.items():
        if any(x is None for x in sub_list) or len(sub_list) != len(subjects): continue
        final_matrices[roi_name] = np.stack(sub_list)

    return final_matrices


# ================= 3. 主分析流程 =================

def run_roi_analysis(
        preset: JointPreset,
        atlas_mapping: Dict[str, Dict[str, Path]],
        analysis_level: str,
        specific_network: str = None
):
    print(f"\n{'=' * 60}")
    print(f"开始 ROI 联合分析 | Level: {analysis_level} | Filter: {specific_network}")
    print(f"Atlas: Robust Parsing Enabled")
    print(f"{'=' * 60}")

    folder_name = f"roi_analysis_{analysis_level}"
    if specific_network:
        folder_name += f"_{specific_network}"
    out_dir = preset.out_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Info
    age_map = load_subject_ages_map(preset.subject_info)
    subjects = determine_subject_list_strict(preset, age_map)
    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    n_subs = len(subjects)
    n_pairs = int(n_subs * (n_subs - 1) / 2)

    # Save Subjects
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_dir / "subjects_roi_analysis.csv", index=False)

    dfs = []
    for spec in preset.specs:
        df = pd.read_csv(spec.aligned_csv)
        df = filter_dataframe(df, preset.task, preset.condition_group)
        dfs.append(df)

    dev_models = build_dev_models(ages, n_subs, preset.normalize_models)
    model_keys = ["M_nn", "M_conv", "M_div"]

    roi_isc_vectors = []
    roi_meta_info = []

    # Process
    for spec, df in zip(preset.specs, dfs):
        atlas_config = atlas_mapping.get(spec.space)
        if not atlas_config: continue

        try:
            print(f"\n>>> 处理空间: {spec.space}")
            roi_matrices = extract_roi_data(
                spec, atlas_config, df, subjects, preset.lss_root, preset,
                analysis_level, specific_network
            )
        except Exception as e:
            logger.error(f"提取失败 {spec.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

        iu, ju = np.triu_indices(n_subs, k=1)

        for roi_name, data_mat in roi_matrices.items():
            corr_mat = np.corrcoef(data_mat)
            vec_z = zscore_1d(corr_mat[iu, ju])

            if np.isnan(vec_z).any(): continue

            roi_isc_vectors.append(vec_z)
            roi_meta_info.append({
                "roi": roi_name,
                "space": spec.space,
                "n_voxels": data_mat.shape[1] // 24
            })

    if not roi_isc_vectors:
        print("❌ 未提取到数据，终止。")
        return

    # Statistics (Model-wise FWER)
    S_stack = np.stack(roi_isc_vectors)
    n_rois_total = S_stack.shape[0]

    print(f"\n>>> 统计检验: Pool={n_rois_total} ROIs (Model-wise FWER)")

    r_obs = {m: (S_stack @ dev_models[m]) / n_pairs for m in model_keys}
    count_fwer = {m: np.zeros(n_rois_total) for m in model_keys}
    count_raw = {m: np.zeros(n_rois_total) for m in model_keys}

    rng = np.random.default_rng(preset.seed)

    for i in tqdm(range(preset.n_perm), desc="Permutation"):
        perm_idx = rng.permutation(n_subs)
        ages_perm = ages[perm_idx]
        models_perm = build_dev_models(ages_perm, n_subs, preset.normalize_models)

        for m in model_keys:
            r_perm = (S_stack @ models_perm[m]) / n_pairs
            count_raw[m] += (r_perm >= r_obs[m])
            max_r = r_perm.max()
            count_fwer[m] += (max_r >= r_obs[m])

    results = []
    for i, meta in enumerate(roi_meta_info):
        for m in model_keys:
            results.append({
                "roi": meta["roi"],
                "space": meta["space"],
                "model": m,
                "r_obs": float(r_obs[m][i]),
                "p_raw": (count_raw[m][i] + 1) / (preset.n_perm + 1),
                "p_fwer_model_wise": (count_fwer[m][i] + 1) / (preset.n_perm + 1),
                "level": analysis_level
            })

    res_df = pd.DataFrame(results)
    out_csv = out_dir / f"roi_stats_{analysis_level}_perm{preset.n_perm}.csv"
    res_df.to_csv(out_csv, index=False)

    sig_df = res_df[res_df["p_fwer_model_wise"] < 0.05]
    print(f"\n✅ 完成。结果: {out_csv}")
    if not sig_df.empty:
        print(f"🎉 显著结果: {len(sig_df)} 个")
        print(sig_df[["roi", "space", "model", "r_obs", "p_fwer_model_wise"]].head(10).to_string(index=False))


# ================= 4. 配置入口 =================

if __name__ == "__main__":
    # 请确保路径正确
    TXT_SURFACE = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order_info.txt")
    TXT_VOLUME = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_label.txt")

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
            "img": Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz"),
            "txt": TXT_VOLUME
        }
    }

    MY_PRESET = JointPreset(
        name="EMO_Schaefer_Combined",
        specs=(
            SimilaritySpec(
                name="surface_L",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="combined"
            ),
            SimilaritySpec(
                name="surface_R",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="combined"
            ),
            SimilaritySpec(
                name="volume",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
                space="volume",
                kind="combined",
                volume_mask=Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")
            ),
        ),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        condition_group="all",
        n_perm=5000,
        min_coverage=0.9
    )

    # 执行分析
    run_roi_analysis(MY_PRESET, ATLAS_MAPPING, analysis_level="network")

    # 2. 探索特定网络的细分脑区 (如发现 Visual 显著)
    # run_roi_analysis(MY_PRESET, ATLAS_MAPPING, analysis_level="parcel", specific_network="Vis")