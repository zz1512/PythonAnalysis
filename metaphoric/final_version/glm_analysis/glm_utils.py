#!/usr/bin/env python3
"""
glm_utils.py

用途
- GLM/LSS 相关的通用工具函数：confounds 读取与 scrubbing、组水平 mask 生成、簇报告与 ROI mask 输出。

论文意义（这一层回答什么问题）
- 这些工具本身不产生“论文结论”，但它们决定：
  - 你的 GLM 是否在对齐/去噪层面足够稳健（直接影响假阳性/假阴性）。
  - 组水平推断是否在同一体素集合上比较（避免因缺失数据导致的偏差）。

方法
- `robust_load_confounds()`：从 fMRIPrep confounds 里构造 Friston-24，并生成 scrubbing 的 `sample_mask`。
- `compute_group_mask()`：按“至少 X% 被试在该体素有有效值”生成组 mask（减少边缘缺失体素污染）。
- `generate_cluster_report()`：基于阈值化统计图生成簇表，并用 Harvard-Oxford atlas 给峰值坐标粗标注。

输入
- confounds TSV（包含 motion 参数与可选 `motion_outlier*` 列）。
- 对比图列表（NIfTI path 列表）。
- 二阶统计图（NIfTI img）与阈值。

输出
- `sample_mask`：用于 nilearn GLM 的“保留时间点索引”，防止坏点驱动假阳性。
- 组 mask NIfTI、簇报告 CSV、二值 ROI mask NIfTI。

关键参数原因
- scrubbing 扩窗：坏点前后若干 TR 常伴随瞬时伪影，扩窗可更保守地控制头动污染。
- `threshold_ratio`：组 mask 越严格，越减少缺失体素带来的偏差，但也可能删掉真实效应的边缘区域。

结果解读
- 如果“没有任何簇”或组 mask 过小，优先检查：scrubbing 是否过强、被试缺失是否集中在某些脑区、
  以及 confounds/events 是否对齐。

常见坑
- confounds 列名不一致（不同 fMRIPrep 版本/自定义 pipeline），导致 motion 列读取失败或静默丢列。
- scrubbing 后有效时间点过少，模型拟合不稳定（尤其是事件密度高或回归量多时）。
- 组 mask 按“非 0 且非 NaN”判断：若上游对比图写入方式不同（例如背景不是 0），需要同步调整阈值逻辑。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import image, datasets
from nilearn.image import new_img_like, math_img
from nilearn.reporting import get_clusters_table
from scipy.ndimage import binary_dilation, label as ndi_label
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
                    # 扩窗 scrubbing：把坏点相邻 TR 也删掉，降低头动伪影“溢出”到邻近时间点的风险。
                    is_outlier = binary_dilation(is_outlier.values, iterations=iterations)

                # 生成保留索引
                n_vols = len(df)
                # sample_mask 是“保留的时间点索引”，与 nilearn FirstLevelModel.fit(sample_mask=...) 语义一致。
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
    # 组水平推断只在“足够多被试都有值”的体素上进行，减少因缺失/裁剪差异造成的偏差。
    group_mask_data = valid_count >= min_subjects

    return new_img_like(first_img, group_mask_data.astype(np.uint8))


def _lookup_atlas_label(atlas_data, affine_inv, labels, xyz):
    """在给定 atlas 上查峰值坐标的标签；命中背景/越界返回 None。"""
    from nilearn.image import coord_transform
    vox = coord_transform(xyz[0], xyz[1], xyz[2], affine_inv)
    vox = [int(round(v)) for v in vox]
    if not (0 <= vox[0] < atlas_data.shape[0]
            and 0 <= vox[1] < atlas_data.shape[1]
            and 0 <= vox[2] < atlas_data.shape[2]):
        return None
    idx = int(atlas_data[vox[0], vox[1], vox[2]])
    if idx <= 0:
        return None
    if 0 <= idx < len(labels):
        name = str(labels[idx]).strip()
        if name and name.lower() != "background":
            return name
    return None


def _lookup_aal_label(atlas_data, affine_inv, label_to_name, xyz):
    """AAL atlas 的标签是稀疏整数 id（非连续），需要用 dict 查。"""
    from nilearn.image import coord_transform
    vox = coord_transform(xyz[0], xyz[1], xyz[2], affine_inv)
    vox = [int(round(v)) for v in vox]
    if not (0 <= vox[0] < atlas_data.shape[0]
            and 0 <= vox[1] < atlas_data.shape[1]
            and 0 <= vox[2] < atlas_data.shape[2]):
        return None
    idx = int(atlas_data[vox[0], vox[1], vox[2]])
    if idx <= 0:
        return None
    name = label_to_name.get(idx)
    if name:
        return str(name).strip()
    return None


def generate_cluster_report(stat_img, threshold, output_path, min_voxels=20, atlas_name='cort-maxprob-thr25-2mm'):
    """
    生成聚类表格并自动标注脑区。

    标注策略（四级 fallback）
    1. Harvard-Oxford cortical (`atlas_name`)
    2. Harvard-Oxford subcortical (`sub-maxprob-thr25-2mm`)
    3. AAL
    4. fallback 为 `Unknown_<hemi>_<near_label>`（仅给半球 + 最近非零标签，避免下游文件名 `cluster_XX`）
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

    # 一级：H-O cort
    cort = datasets.fetch_atlas_harvard_oxford(atlas_name)
    cort_data = cort.maps.get_fdata()
    cort_labels = cort.labels
    cort_affine_inv = np.linalg.inv(cort.maps.affine)

    # 二级：H-O subcortical
    try:
        sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        sub_data = sub.maps.get_fdata()
        sub_labels = sub.labels
        sub_affine_inv = np.linalg.inv(sub.maps.affine)
    except Exception as e:
        logger.warning(f"Could not load Harvard-Oxford subcortical atlas: {e}")
        sub_data = sub_labels = sub_affine_inv = None

    # 三级：AAL
    try:
        aal = datasets.fetch_atlas_aal()
        aal_data = np.asarray(aal.maps.get_fdata()).astype(int) \
            if hasattr(aal, 'maps') else np.asarray(image.load_img(aal['maps']).get_fdata()).astype(int)
        aal_affine_inv = np.linalg.inv(
            aal.maps.affine if hasattr(aal, 'maps') else image.load_img(aal['maps']).affine
        )
        aal_label_to_name = {int(idx): str(lab) for lab, idx in zip(aal['labels'], aal['indices'])}
    except Exception as e:
        logger.warning(f"Could not load AAL atlas: {e}")
        aal_data = aal_label_to_name = aal_affine_inv = None

    regions = []
    for _, row in table.iterrows():
        xyz = (row['X'], row['Y'], row['Z'])

        name = _lookup_atlas_label(cort_data, cort_affine_inv, cort_labels, xyz)
        source = "HO_cort"

        if name is None and sub_data is not None:
            name = _lookup_atlas_label(sub_data, sub_affine_inv, sub_labels, xyz)
            source = "HO_sub"

        if name is None and aal_data is not None:
            name = _lookup_aal_label(aal_data, aal_affine_inv, aal_label_to_name, xyz)
            source = "AAL"

        if name is None:
            hemi = "L" if float(row['X']) < 0 else ("R" if float(row['X']) > 0 else "M")
            name = f"Unknown_{hemi}_x{int(round(row['X']))}_y{int(round(row['Y']))}_z{int(round(row['Z']))}"
            source = "fallback"

        regions.append(f"{name}|{source}")

    table.insert(0, 'Region', regions)
    table.to_csv(output_path, index=False)
    logger.info(f"Report saved: {output_path}")


def save_binary_mask(stat_img, threshold, output_path, min_cluster_voxels: int = 0):
    """
    保存阈值化的二值 ROI mask。

    关键改动
    - 增加 `min_cluster_voxels` 参数，与 `generate_cluster_report` 的 `cluster_threshold`
      口径对齐，避免 mask 里保留了小于最小 cluster 体素数的零散显著碎片，
      导致下游 ROI 切分看到的体素集合与 cluster_report.csv 不一致。

    参数
    - stat_img: 统计图（例如 -log10(p) map）。
    - threshold: 阈值（体素 > threshold 视为显著）。
    - output_path: 输出 NIfTI 路径。
    - min_cluster_voxels: 最小 cluster 体素数（<= 0 表示不做 cluster 过滤，向后兼容）。
    """
    mask_img = math_img(f'img > {threshold}', img=stat_img)
    if min_cluster_voxels and min_cluster_voxels > 0:
        mask_data = np.asarray(mask_img.get_fdata()) > 0
        labeled, n_components = ndi_label(mask_data)
        if n_components > 0:
            sizes = np.bincount(labeled.ravel())
            # sizes[0] 是背景；从 id=1 开始比较
            keep_ids = np.where(sizes[1:] >= int(min_cluster_voxels))[0] + 1
            filtered = np.isin(labeled, keep_ids)
            mask_img = new_img_like(mask_img, filtered.astype(np.uint8))
    mask_img.to_filename(output_path)
