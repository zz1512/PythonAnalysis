#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_secondlevel_exp2_core_contrasts.py

实验2核心差值 contrasts 的二阶组分析脚本

用途：
1. 自动读取 first_level_exp2_core_contrasts 中全部被试的一阶 contrast 结果
2. 对每个 contrast 做 one-sample second-level GLM（组均值是否显著偏离 0）
3. 默认快速输出 FDR 校正结果；可选 cluster-FWE permutation（含 GPU）
4. 输出适合向老师汇报的结果图、阈值图、cluster 表和汇总表

推荐：
- 先用 METHOD='fdr' 快速出整套组结果给老师看
- 如果某几个 contrast 很关键，再切 METHOD='cluster_fwe' 做更严格验证
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from nilearn import image, plotting, datasets
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import new_img_like, math_img
from nilearn.reporting import get_clusters_table

# =========================
# 0. 全局配置
# =========================
BASE_DIR = Path(r"C:/python_fertility")
FIRST_LEVEL_ROOT = BASE_DIR / "first_level_exp2_core_contrasts"
OUTPUT_ROOT = BASE_DIR / "second_level_exp2_core_contrasts"

# 二阶分析方式：'fdr'（快，推荐先汇报） | 'cluster_fwe'（严格，慢）
METHOD = "fdr"

# 至少多少个被试有该 contrast 才跑二阶
MIN_SUBJECTS = 10

# 组掩膜：至少多少比例被试在某体素有非零值
GROUP_MASK_THRESHOLD_RATIO = 0.7

# FDR 参数
FDR_ALPHA = 0.05
FDR_CLUSTER_MIN_SIZE = 20

# Cluster-FWE 参数
PERMUTATION_N = 5000
CLUSTER_FORMING_P = 0.001
TARGET_ALPHA = 0.05
USE_GPU = True
N_JOBS_PERM = -1

# 可视化参数
REPORT_MIN_CLUSTER_VOXELS = 20
GLASS_THRESHOLD_Z = 3.1
ATLAS_NAME = "cort-maxprob-thr25-2mm"

# 中文显示修复
try:
    import matplotlib
    from matplotlib import font_manager

    _CANDIDATES = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_ROOT / "second_level_exp2.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("second_level_exp2")


# =========================
# 1. 工具函数
# =========================
def save_binary_mask(stat_img, threshold, output_path: Path):
    mask = math_img(f"img > {threshold}", img=stat_img)
    mask.to_filename(output_path)


def compute_group_mask(contrast_imgs: List[str], threshold_ratio: float = 0.7):
    """仅保留足够多被试有有效值的体素。"""
    if not contrast_imgs:
        raise ValueError("No contrast images provided for mask computation.")

    first_img = image.load_img(contrast_imgs[0])
    valid_count = np.zeros(first_img.shape, dtype=np.int32)

    for fname in contrast_imgs:
        data = image.load_img(fname).get_fdata()
        mask = (np.abs(data) > 1e-6) & (~np.isnan(data))
        valid_count += mask.astype(np.int32)

    min_subjects = max(1, int(np.floor(len(contrast_imgs) * threshold_ratio)))
    group_mask_data = valid_count >= min_subjects
    return new_img_like(first_img, group_mask_data.astype(np.uint8))


def generate_cluster_report(stat_img, threshold: float, output_path: Path,
                            min_voxels: int = 20, atlas_name: str = ATLAS_NAME):
    try:
        table = get_clusters_table(
            stat_img,
            stat_threshold=threshold,
            cluster_threshold=min_voxels,
            two_sided=False,
        )
    except ValueError:
        logger.warning(f"No suprathreshold clusters found for {output_path.name}")
        return

    if table is None or table.empty:
        logger.warning(f"Cluster table empty: {output_path.name}")
        return

    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
    atlas_data = atlas.maps.get_fdata()
    affine_inv = np.linalg.inv(atlas.maps.affine)
    labels = atlas.labels

    from nilearn.image import coord_transform

    regions = []
    for _, row in table.iterrows():
        vox = coord_transform(row["X"], row["Y"], row["Z"], affine_inv)
        vox = [int(round(v)) for v in vox]
        region_name = "Unknown"
        if (0 <= vox[0] < atlas_data.shape[0] and
            0 <= vox[1] < atlas_data.shape[1] and
            0 <= vox[2] < atlas_data.shape[2]):
            idx = int(atlas_data[vox[0], vox[1], vox[2]])
            if 0 <= idx < len(labels):
                region_name = labels[idx]
        regions.append(region_name)

    table.insert(0, "Region", regions)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")


def load_contrast_label_map(first_level_root: Path) -> Dict[str, str]:
    p = first_level_root / "core_contrast_definitions.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if "contrast" in df.columns and "label_cn" in df.columns:
            return dict(zip(df["contrast"], df["label_cn"]))
    except Exception:
        pass
    return {}


def discover_first_level_maps(first_level_root: Path) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    从各被试 manifest 自动收集每个 contrast 的 effect_size map。
    返回：
        contrast_to_files: {contrast_name: [effect_size_paths...]}
        long_df: 明细表（subject, contrast, label_cn, effect_size, z_score, stat）
    """
    manifests = sorted(first_level_root.glob("sub-*/reports/*_core_contrast_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"未找到一阶 manifest：{first_level_root}")

    rows = []
    for mf in manifests:
        try:
            df = pd.read_csv(mf)
            if len(df) == 0:
                continue
            rows.append(df)
        except Exception as e:
            logger.warning(f"读取 manifest 失败 {mf}: {e}")

    if not rows:
        raise RuntimeError("没有可用的一阶 manifest 内容。")

    long_df = pd.concat(rows, axis=0, ignore_index=True)
    if "effect_size" not in long_df.columns:
        raise ValueError("manifest 中缺少 effect_size 列。")

    # 只保留文件存在的行
    long_df = long_df[long_df["effect_size"].map(lambda x: Path(str(x)).exists())].copy()
    if len(long_df) == 0:
        raise RuntimeError("没有找到实际存在的一阶 effect_size 图。")

    contrast_to_files: Dict[str, List[str]] = {}
    for contrast, subdf in long_df.groupby("contrast"):
        contrast_to_files[contrast] = subdf["effect_size"].astype(str).tolist()

    return contrast_to_files, long_df


def try_gpu_permutation(files: List[str], group_mask, out_dir: Path) -> Optional[Tuple[object, object]]:
    if not USE_GPU:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA 不可用，跳过 GPU permutation。")
            return None
    except Exception:
        logger.info("未检测到 torch.cuda，跳过 GPU permutation。")
        return None

    try:
        import gpu_permutation  # 复用你之前的模块
        logger.info("尝试 GPU permutation...")
        log_p_map, raw_t_img = gpu_permutation.run_permutation(
            files,
            group_mask,
            n_perms=PERMUTATION_N,
            cluster_p_thresh=CLUSTER_FORMING_P,
            n_jobs=N_JOBS_PERM,
        )
        raw_t_img.to_filename(out_dir / "raw_t_stat.nii.gz")
        return log_p_map, raw_t_img
    except Exception as e:
        logger.warning(f"GPU permutation 失败，将回退 CPU：{e}")
        return None


# =========================
# 2. 单个 contrast 的二阶分析
# =========================
def run_second_level_for_one_contrast(contrast: str, files: List[str], label_cn: str) -> Optional[Dict]:
    if len(files) < MIN_SUBJECTS:
        logger.warning(f"跳过 {contrast}: 被试数不足 ({len(files)} < {MIN_SUBJECTS})")
        return None

    logger.info(f"开始二阶分析: {contrast} | N={len(files)} | {label_cn}")
    out_dir = OUTPUT_ROOT / contrast
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存输入清单
    pd.DataFrame({"effect_size": files}).to_csv(out_dir / "input_files.csv", index=False, encoding="utf-8-sig")

    # 组掩膜
    group_mask = compute_group_mask(files, threshold_ratio=GROUP_MASK_THRESHOLD_RATIO)
    group_mask.to_filename(out_dir / "group_mask.nii.gz")

    design_matrix = pd.DataFrame({"intercept": np.ones(len(files), dtype=float)})

    result_row = {
        "contrast": contrast,
        "label_cn": label_cn,
        "n_subjects": len(files),
        "method": METHOD,
        "status": "started",
    }

    try:
        if METHOD == "fdr":
            model = SecondLevelModel(mask_img=group_mask)
            model.fit(files, design_matrix=design_matrix)

            z_map = model.compute_contrast("intercept", output_type="z_score")
            effect_map = model.compute_contrast("intercept", output_type="effect_size")
            stat_map = model.compute_contrast("intercept", output_type="stat")

            z_map.to_filename(out_dir / "group_z_map.nii.gz")
            effect_map.to_filename(out_dir / "group_effect_size_map.nii.gz")
            stat_map.to_filename(out_dir / "group_stat_map.nii.gz")

            # FDR threshold
            try:
                thresh_map, thresh_val = threshold_stats_img(
                    z_map,
                    alpha=FDR_ALPHA,
                    height_control="fdr",
                    cluster_threshold=FDR_CLUSTER_MIN_SIZE,
                )
                thresh_map.to_filename(out_dir / "group_fdr_corrected_z_map.nii.gz")

                generate_cluster_report(
                    z_map,
                    thresh_val,
                    out_dir / "cluster_report_fdr.csv",
                    min_voxels=FDR_CLUSTER_MIN_SIZE,
                )

                save_binary_mask(z_map, thresh_val, out_dir / f"{contrast}_fdr_mask.nii.gz")

                plotting.plot_glass_brain(
                    thresh_map,
                    colorbar=True,
                    plot_abs=False,
                    title=f"{label_cn} | 二阶 FDR q<{FDR_ALPHA}",
                    output_file=str(out_dir / "glass_brain_fdr.png"),
                )
                plotting.plot_stat_map(
                    thresh_map,
                    display_mode="ortho",
                    colorbar=True,
                    threshold=thresh_val,
                    title=f"{label_cn} | 二阶 FDR q<{FDR_ALPHA}",
                    output_file=str(out_dir / "ortho_fdr.png"),
                )

                result_row.update({
                    "threshold": float(thresh_val),
                    "status": "ok",
                    "significant_map": str(out_dir / "group_fdr_corrected_z_map.nii.gz"),
                })
            except Exception as e:
                logger.warning(f"{contrast} FDR 下未找到显著团块：{e}")
                # 仍保存未校正可视化，方便给老师看整体模式
                unc_thresh = GLASS_THRESHOLD_Z
                plotting.plot_glass_brain(
                    z_map,
                    threshold=unc_thresh,
                    colorbar=True,
                    plot_abs=False,
                    title=f"{label_cn} | 二阶未校正 Z>{unc_thresh}",
                    output_file=str(out_dir / "glass_brain_uncorrected.png"),
                )
                plotting.plot_stat_map(
                    z_map,
                    threshold=unc_thresh,
                    display_mode="ortho",
                    colorbar=True,
                    title=f"{label_cn} | 二阶未校正 Z>{unc_thresh}",
                    output_file=str(out_dir / "ortho_uncorrected.png"),
                )
                generate_cluster_report(
                    z_map,
                    unc_thresh,
                    out_dir / "cluster_report_uncorrected.csv",
                    min_voxels=REPORT_MIN_CLUSTER_VOXELS,
                )
                result_row.update({
                    "threshold": float(unc_thresh),
                    "status": "ok_no_fdr_clusters",
                    "significant_map": str(out_dir / "group_z_map.nii.gz"),
                })

        elif METHOD == "cluster_fwe":
            log_p_map = None
            gpu_res = try_gpu_permutation(files, group_mask, out_dir)
            if gpu_res is not None:
                log_p_map, raw_t = gpu_res
            else:
                threshold_z = norm.isf(CLUSTER_FORMING_P)
                res = non_parametric_inference(
                    files,
                    design_matrix=design_matrix,
                    second_level_contrast="intercept",
                    mask=group_mask,
                    n_perm=PERMUTATION_N,
                    two_sided_test=False,
                    n_jobs=N_JOBS_PERM,
                    threshold=threshold_z,
                )
                log_p_map = res["logp_max_size"]

            log_p_map.to_filename(out_dir / "log_p_map.nii.gz")
            viz_thresh = -np.log10(TARGET_ALPHA)

            generate_cluster_report(
                log_p_map,
                viz_thresh,
                out_dir / "cluster_report_cluster_fwe.csv",
                min_voxels=REPORT_MIN_CLUSTER_VOXELS,
            )
            save_binary_mask(log_p_map, viz_thresh, out_dir / f"{contrast}_cluster_fwe_mask.nii.gz")

            plotting.plot_glass_brain(
                log_p_map,
                threshold=viz_thresh,
                colorbar=True,
                title=f"{label_cn} | Cluster-FWE<{TARGET_ALPHA}",
                output_file=str(out_dir / "glass_brain_cluster_fwe.png"),
            )
            plotting.plot_stat_map(
                log_p_map,
                threshold=viz_thresh,
                display_mode="ortho",
                colorbar=True,
                title=f"{label_cn} | Cluster-FWE<{TARGET_ALPHA}",
                output_file=str(out_dir / "ortho_cluster_fwe.png"),
            )

            result_row.update({
                "threshold": float(viz_thresh),
                "status": "ok",
                "significant_map": str(out_dir / "log_p_map.nii.gz"),
            })
        else:
            raise ValueError(f"未知 METHOD: {METHOD}")

        return result_row

    except Exception as e:
        logger.error(f"{contrast} 二阶分析失败: {e}")
        logger.error(traceback.format_exc())
        result_row.update({"status": f"failed: {e}"})
        return result_row


# =========================
# 3. 主程序
# =========================
def main():
    logger.info("=" * 80)
    logger.info("实验2核心差值 contrasts 二阶组分析开始")
    logger.info(f"FIRST_LEVEL_ROOT = {FIRST_LEVEL_ROOT}")
    logger.info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
    logger.info(f"METHOD = {METHOD}")
    logger.info("=" * 80)

    contrast_label_map = load_contrast_label_map(FIRST_LEVEL_ROOT)
    contrast_to_files, long_df = discover_first_level_maps(FIRST_LEVEL_ROOT)
    long_df.to_csv(OUTPUT_ROOT / "all_first_level_inputs_long.csv", index=False, encoding="utf-8-sig")

    results = []
    for contrast in sorted(contrast_to_files.keys()):
        files = contrast_to_files[contrast]
        label_cn = contrast_label_map.get(contrast, contrast)
        res = run_second_level_for_one_contrast(contrast, files, label_cn)
        if res is not None:
            results.append(res)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_ROOT / "second_level_summary.csv", index=False, encoding="utf-8-sig")

    logger.info("=" * 80)
    logger.info(f"完成。结果汇总：{OUTPUT_ROOT / 'second_level_summary.csv'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
