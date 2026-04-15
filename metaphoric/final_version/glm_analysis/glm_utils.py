#!/usr/bin/env python3
"""
glm_utils.py
通用工具函数库 (Refactored to remove duplication)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import image, datasets
from nilearn.image import new_img_like, math_img
from nilearn.reporting import get_clusters_table
from scipy.ndimage import binary_dilation
import logging

logger = logging.getLogger(__name__)


def robust_load_confounds(confounds_path, strategy_config):
    """
    手动读取 confounds 并计算 Friston-24 和 Scrubbing Mask
    """
    try:
        df = pd.read_csv(confounds_path, sep='\t')

        # 1. 基础 6 参数
        motion_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        # 简单容错：如果列名不匹配，尝试查找类似列 (此处省略复杂逻辑，假设标准输出)
        if not all(col in df.columns for col in motion_axes):
            logger.warning(f"Motion columns missing in {Path(confounds_path).name}")

        motion_df = df[motion_axes].copy()

        # 2. Friston-24
        if strategy_config['motion'] == 'friston24':
            motion_sq = motion_df ** 2
            motion_deriv = motion_df.diff().fillna(0)
            motion_deriv_sq = motion_deriv ** 2

            motion_sq.columns = [c + '_sq' for c in motion_df.columns]
            motion_deriv.columns = [c + '_dt' for c in motion_df.columns]
            motion_deriv_sq.columns = [c + '_dt_sq' for c in motion_df.columns]

            final_confounds = pd.concat([motion_df, motion_sq, motion_deriv, motion_deriv_sq], axis=1)
        else:
            final_confounds = motion_df

        # 3. Scrubbing
        sample_mask = None
        if strategy_config['denoise_strategy'] == 'scrubbing':
            outlier_cols = [c for c in df.columns if 'motion_outlier' in c]
            if outlier_cols:
                is_outlier = df[outlier_cols].sum(axis=1) > 0
                if strategy_config['scrub'] > 0:
                    iterations = int(strategy_config['scrub'] / 2)
                    is_outlier = binary_dilation(is_outlier.values, iterations=iterations)

                # 生成保留索引
                n_vols = len(df)
                sample_mask = np.arange(n_vols)[~is_outlier]

        return final_confounds.fillna(0), sample_mask

    except Exception as e:
        logger.error(f"Error loading confounds {confounds_path}: {e}")
        raise


def compute_group_mask(contrast_imgs, threshold_ratio=0.7):
    """
    计算组掩膜：只有当 X% 的被试在某体素有值时保留
    """
    if not contrast_imgs:
        raise ValueError("No input images for mask computation.")

    first_img = image.load_img(contrast_imgs[0])
    valid_count = np.zeros(first_img.shape, dtype=np.int32)

    for fname in contrast_imgs:
        data = image.load_img(fname).get_fdata()
        # 检查非 NaN 且非 0
        mask = (np.abs(data) > 1e-6) & (~np.isnan(data))
        valid_count += mask.astype(np.int32)

    min_subjects = int(np.floor(len(contrast_imgs) * threshold_ratio))
    group_mask_data = valid_count >= min_subjects

    return new_img_like(first_img, group_mask_data.astype(np.uint8))


def generate_cluster_report(stat_img, threshold, output_path, min_voxels=20, atlas_name='cort-maxprob-thr25-2mm'):
    """
    生成聚类表格并自动标注脑区
    """
    try:
        table = get_clusters_table(
            stat_img, stat_threshold=threshold, cluster_threshold=min_voxels, two_sided=False
        )
    except ValueError:
        logger.warning("No suprathreshold clusters found.")
        return

    if table is None or table.empty:
        return

    # 加载 Atlas
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
    atlas_data = atlas.maps.get_fdata()
    labels = atlas.labels
    affine_inv = np.linalg.inv(atlas.maps.affine)

    regions = []
    from nilearn.image import coord_transform

    for _, row in table.iterrows():
        vox = coord_transform(row['X'], row['Y'], row['Z'], affine_inv)
        vox = [int(round(v)) for v in vox]

        region_name = "Unknown"
        # 边界检查
        if (0 <= vox[0] < atlas_data.shape[0] and
                0 <= vox[1] < atlas_data.shape[1] and
                0 <= vox[2] < atlas_data.shape[2]):
            idx = int(atlas_data[vox[0], vox[1], vox[2]])
            if 0 <= idx < len(labels):
                region_name = labels[idx]

        regions.append(region_name)

    table.insert(0, 'Region', regions)
    table.to_csv(output_path, index=False)
    logger.info(f"Report saved: {output_path}")


def save_binary_mask(stat_img, threshold, output_path):
    mask = math_img(f'img > {threshold}', img=stat_img)
    mask.to_filename(output_path)