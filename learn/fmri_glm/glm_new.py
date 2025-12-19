#!/usr/bin/env python3
"""
fMRI GLM Analysis Pipeline (Optimized)
Features:
- Robust First-level GLM
- Cluster-level FWE via Non-parametric Permutation
- Automatic Group Mask Intersection
- Quality Control (QC)
"""

import os
import re  # 需要用到正则
from scipy.ndimage import binary_dilation # 用于扩展坏点
# 限制底层库的线程数，避免与多进程冲突
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import logging
try:
    import gpu_permutation
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("gpu_permutation module not found. GPU acceleration disabled.")
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from nilearn.masking import compute_multi_epi_mask

from nilearn import image, plotting, datasets, masking
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.reporting import get_clusters_table
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.image import mean_img, math_img, resample_to_img

# 导入配置
try:
    from glm_config_new import config
except ImportError:
    print("Error: config_enhance.py not found.")
    sys.exit(1)

# =========================
# 0. 初始化与日志
# =========================

config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_ROOT / 'analysis_pipeline.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# 1. 辅助函数 (Utils)
# =========================

def infer_tr(img_path):
    """从Nifti头文件安全读取TR"""
    try:
        img = nib.load(str(img_path))
        tr = img.header.get_zooms()[3]
        if tr <= 0: raise ValueError("Invalid TR in header")
        return float(tr)
    except Exception:
        logger.warning(f"Could not infer TR from {img_path}, using fallback 2.0s")
        return 2.0


def load_confounds(motion_path):
    """加载头动参数，处理格式差异"""
    try:
        # 尝试读取，兼容空格或制表符分隔
        df = pd.read_csv(motion_path, sep=r'\s+', header=None)
        # 命名列，防止GLM报错
        df.columns = [f'motion_{i}' for i in range(df.shape[1])]
        return df
    except Exception as e:
        logger.error(f"Failed to load confounds: {motion_path} - {e}")
        return None


def check_mask_coverage(group_mask_img):
    """QC: 检查组掩膜的覆盖率 (修复了网格不匹配的Bug)"""
    try:
        # 1. 加载标准 MNI 模板
        mni = datasets.load_mni152_template(resolution=2)

        # 2. [关键修复] 将 MNI 模板重采样到 group_mask 的网格上
        # 使用 interpolation='nearest' 保持二值属性
        mni_resampled = image.resample_to_img(
            mni,
            group_mask_img,
            interpolation='nearest',
            copy_header=True,  # 消除 header 警告
            force_resample=True  # 消除 resample 警告
        )

        # 3. 获取数据矩阵
        group_data = group_mask_img.get_fdata() > 0
        mni_data = mni_resampled.get_fdata() > 0

        # 4. 计算覆盖率
        # 交集体积 / 标准脑体积
        overlap = np.sum(group_data & mni_data)
        expected = np.sum(mni_data)

        if expected == 0:
            logger.warning("MNI template mask is empty after resampling!")
            return 0.0

        coverage = overlap / expected

        logger.info(f"Group Mask Coverage (vs MNI): {coverage:.1%}")

        if coverage < 0.8:
            logger.warning("WARNING: Group mask coverage is low (<80%). Check for subject dropout or artifacts.")

        return coverage

    except Exception as e:
        logger.warning(f"QC check failed: {e}. Skipping coverage check.")
        return 0.0

def make_contrast_vector(contrast_dict, design_matrix_columns):
    """
    核心修复：将对比字典转换为 Numpy 向量
    输入: {"yy": 1, "kj": -1}, columns=['yy', 'kj', 'drift_1', ...]
    输出: [1, -1, 0, ...]
    """
    con_vector = np.zeros(len(design_matrix_columns))
    for condition, weight in contrast_dict.items():
        if condition in design_matrix_columns:
            idx = design_matrix_columns.index(condition)
            con_vector[idx] = weight
        else:
            logger.warning(f"  Warning: Condition '{condition}' not found in design matrix columns: {design_matrix_columns}")
    return con_vector

# =========================
# 2. 一阶分析 (First Level)
# =========================

def run_single_subject_glm(sub, run, contrasts_def):
    """单个被试单个Run的一阶分析核心逻辑"""
    try:
        # 1. 路径构建
        fmri_file = Path(config.FMRI_TPL.format(sub=sub, run=run))
        event_file = Path(config.EVENT_TPL.format(sub=sub, run=run))
        confounds_file = Path(config.CONFOUNDS_TPL.format(sub=sub, run=run))

        out_dir = config.OUTPUT_ROOT / "1st_level" / sub / f"run-{run}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. 数据检查
        if not fmri_file.exists():
            return {"success": False, "msg": f"Missing fMRI: {fmri_file}"}
        if not confounds_file.exists():
             return {"success": False, "msg": f"Missing Confounds TSV: {confounds_file}"}

        # 3. 加载数据
        tr = infer_tr(fmri_file)
        events = pd.read_csv(event_file, sep='\t')
        # 彻底绕过 Nilearn 的 BIDS 检测
        confounds, sample_mask = custom_load_confounds(
            confounds_file,
            config.CONFOUNDS_STRATEGY
        )

        mni_mask = datasets.load_mni152_brain_mask()

        # 重要：标准 mask 的分辨率可能与你的数据(2mm/3mm)不同，需要重采样
        # 这一步非常快
        # target_img 只需要读取头文件，不需要读取整个数据，所以加载 fmri_file 很快
        target_affine_img = image.load_img(str(fmri_file))

        # 使用 nearest 插值保持 mask 的二值属性 (0/1)
        used_mask_img = image.resample_to_img(
            mni_mask,
            target_affine_img,
            interpolation='nearest'
        )

        # 4. 建模 (应用标准化与平滑)
        # 注意: mask_img=None 让 Nilearn 自动计算个体的 mask
        glm = FirstLevelModel(
            t_r=tr,
            slice_time_ref=0.5,
            mask_img=used_mask_img,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            **config.GLM_PARAMS
        )

        # 传入 sample_mask 以剔除坏点
        glm.fit(fmri_file, events=events, confounds=confounds, sample_masks=sample_mask)

        design_columns = glm.design_matrices_[0].columns.tolist()

        # [优化] 更安全的对比向量构建
        results = []
        for con_name, con_dict in contrasts_def.items():
            # 检查关键条件是否存在
            missing_conds = [k for k in con_dict.keys() if k not in design_columns]
            if missing_conds:
                logger.warning(
                    f"  [SKIP] Contrast {con_name} skipped for {sub} run {run}. Missing conditions: {missing_conds}")
                continue  # 跳过该对比，而不是生成错误的 0 向量

            con_vector = make_contrast_vector(con_dict, design_columns)

            # 检查向量是否全0 (防止无效对比)
            if np.all(con_vector == 0):
                logger.warning(f"  Contrast '{con_name}' is all zeros! Check event names vs design matrix.")
                continue

            # 计算
            con_map = glm.compute_contrast(con_vector, output_type='effect_size')
            stat_map = glm.compute_contrast(con_vector, output_type='stat')

            con_out = out_dir / f"{con_name}_con.nii.gz"
            stat_out = out_dir / f"{con_name}_tmap.nii.gz"

            con_map.to_filename(con_out)
            stat_map.to_filename(stat_out)

            results.append({
                "contrast": con_name,
                "con_file": str(con_out),
                "t_file": str(stat_out)
            })

        return {"success": True, "sub": sub, "run": run, "results": results}

    except Exception as e:
        # 打印详细堆栈以便调试
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "sub": sub, "run": run, "msg": str(e)}


def custom_load_confounds(confounds_file, strategy_config):
    """
    手动加载并清洗混淆因子，替代 nilearn 的自动匹配，解决 BIDS 解析失败问题。
    实现：Friston-24 头动模型 + Scrubbing (坏点剔除)
    """
    try:
        # 1. 读取 TSV 文件
        df = pd.read_csv(confounds_file, sep='\t')

        # 2. 提取 Friston-24 头动参数
        # 包括：6个基础参数 + 6个导数 + 6个平方 + 6个导数的平方
        # fMRIPrep 的命名通常是: trans_x, trans_x_derivative1, trans_x_power2, trans_x_derivative1_power2
        # 我们用正则表达式一次性匹配所有

        # 基础6参数名 (Nilearn/fMRIPrep 标准)
        motion_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

        # 构建 Friston-24 的正则模式
        # 匹配: 轴名 + (可选: _derivative1) + (可选: _power2)
        # 例如: trans_x, trans_x_derivative1, trans_x_power2, trans_x_derivative1_power2
        pattern = r'^(' + '|'.join(motion_axes) + r')(_derivative1)?(_power2)?$'

        # 筛选列
        motion_cols = [c for c in df.columns if re.match(pattern, c)]

        # 检查是否找齐了 (通常应该是24个，如果是power2模型则是12个)
        if len(motion_cols) < 6:
            logger.warning(f"Found only {len(motion_cols)} motion columns. Check naming in TSV.")

        final_confounds = df[motion_cols].copy()

        # 3. 处理 Scrubbing (坏点)
        # 找到所有名为 motion_outlierXX 的列
        outlier_cols = [c for c in df.columns if 'motion_outlier' in c]

        sample_mask = None

        if outlier_cols:
            # 只要某一个时间点在任一 outlier 列中为 1，即视为坏点
            outliers = df[outlier_cols].sum(axis=1) > 0  # Boolean Series

            # 扩展坏点 (Scrubbing Window)
            # strategy_config['scrub'] 是一个整数，比如 5
            # 意味着如果第 t 帧坏了，我们不仅剔除 t，还要剔除周围的帧
            # 简单实现：使用 binary_dilation 向前后扩展
            scrub_window = strategy_config.get('scrub', 0)

            if scrub_window > 0:
                # 扩展坏点掩膜 (Dilate)
                # structure 定义了扩展的形状。这里简单地向前后扩展
                # 这里的逻辑是：如果 scrub=5，我们假设它是指"最小连续好帧长度"或"切除窗口"
                # 为了简化且稳健，我们将坏点前后各扩展 (scrub // 2) 帧
                import scipy.ndimage
                # iterations = 扩展次数
                dilated_outliers = scipy.ndimage.binary_dilation(
                    outliers.values,
                    iterations=max(1, scrub_window // 2)
                )
            else:
                dilated_outliers = outliers.values

            # sample_mask 需要的是“保留”的帧的索引
            # 即：dilated_outliers 为 False 的位置
            # Nilearn 要求 sample_mask 是一个 numpy array 包含要保留的 index
            all_indices = np.arange(len(df))
            sample_mask = all_indices[~dilated_outliers]

            logger.info(
                f"    Manual Scrubbing: Removed {len(all_indices) - len(sample_mask)} volumes (Outliers: {len(outlier_cols)} columns)")
        else:
            logger.info("    No motion outliers found in TSV.")

        # 4. 填充缺失值 (以防万一第一帧导数为NaN)
        final_confounds = final_confounds.fillna(0)

        return final_confounds, sample_mask

    except Exception as e:
        logger.error(f"Error in custom_load_confounds: {e}")
        raise e

# =========================
# 3. 二阶分析 (Second Level - Permutation)
# =========================

def run_group_permutation(contrast_name, input_files, output_dir):
    """
    运行置换检验 (修复 Numpy 报错和 Nilearn 警告)
    """
    logger.info(f"Running Permutation Test for: {contrast_name}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. 稳健生成组掩膜 (Robust Group Masking)
    # ==========================================
    logger.info(f"Computing robust group mask from {len(input_files)} input maps...")

    ref_img = nib.load(input_files[0])
    valid_masks = []

    for i, fpath in enumerate(input_files):
        img = nib.load(fpath)
        # [优化] 消除重采样警告
        if i > 0:
            img = resample_to_img(
                img,
                ref_img,
                interpolation='nearest',
                copy_header=True,  # 新版 Nilearn 要求
                force_resample=True  # 新版 Nilearn 要求
            )

        subj_mask = math_img('np.isfinite(img) & (np.abs(img) > 1e-6)', img=img)
        valid_masks.append(subj_mask)

    # [优化] 消除 mean_img 警告 (如果 nilearn 版本较新)
    try:
        overlap_map = mean_img(valid_masks, copy_header=True)
    except TypeError:
        # 旧版 nilearn 可能不支持 copy_header
        overlap_map = mean_img(valid_masks)

    final_group_mask = math_img('img >= 0.5', img=overlap_map)

    final_group_mask.to_filename(out_dir / "group_mask_robust.nii.gz")

    # --- [关键修复] 使用 np.prod 代替 np.product ---
    n_voxels = np.sum(final_group_mask.get_fdata() > 0)
    total_voxels = np.prod(final_group_mask.shape)  # <--- 这里改了

    logger.info(f"Robust Group Mask Size: {n_voxels} voxels (out of {total_voxels})")

    # 检查覆盖率
    check_mask_coverage(final_group_mask)

    if n_voxels < 5000:
        logger.error("CRITICAL: Group mask is still too small! Check inputs for massive NaN values.")
        return None

    # ==========================================
    # 2. 选择计算引擎 (GPU/CPU)
    # ==========================================
    use_gpu = config.PERM_CONFIG.get("use_gpu", False) and GPU_AVAILABLE and gpu_permutation.HAS_TORCH

    if use_gpu:
        try:
            logger.info("Using GPU Acceleration...")
            log_p_img = gpu_permutation.run_gpu_permutation_test(
                second_level_input=input_files,
                mask_img=final_group_mask,
                n_permutations=config.PERM_CONFIG['n_permutations'],
                cluster_forming_p=config.PERM_CONFIG['cluster_forming_p'],
                output_dir=out_dir
            )
        except Exception as e:
            logger.error(f"GPU failed: {e}. Fallback to CPU.")
            use_gpu = False

    if not use_gpu:
        logger.info("Using CPU (Nilearn)...")
        design_matrix = pd.DataFrame({'intercept': np.ones(len(input_files))})
        cluster_forming_z = norm.isf(config.PERM_CONFIG["cluster_forming_p"])

        log_p_img = non_parametric_inference(
            second_level_input=input_files,
            design_matrix=design_matrix,
            second_level_contrast='intercept',
            model_intercept=True,
            n_perm=config.PERM_CONFIG['n_permutations'],
            two_sided_test=False,
            mask=final_group_mask,
            n_jobs=config.PERM_CONFIG['n_jobs'],
            threshold=cluster_forming_z
        )

    res_path = out_dir / f"{contrast_name}_logp_fwe.nii.gz"
    nib.save(log_p_img, res_path)
    return res_path


# =========================
# 4. 可视化 (Visualization)
# =========================

def visualize_results(log_p_img_path, contrast_name, output_dir):
    """可视化 FWE 校正后的结果"""
    img = nib.load(log_p_img_path)
    output_dir = Path(output_dir)

    # 计算显示阈值: -log10(0.05) ≈ 1.301
    alpha = config.PERM_CONFIG['target_alpha']
    threshold = -np.log10(alpha)

    logger.info(f"Visualizing {contrast_name}: Threshold > {threshold:.2f} (p < {alpha} FWE)")

    # 检查是否有显著体素
    data = img.get_fdata()
    if np.max(data) < threshold:
        logger.warning(f"No significant clusters found for {contrast_name} at FWE < {alpha}")

        # [核心修复] 不要传 None！传 img 本身。
        # 因为所有体素都小于 threshold，Nilearn 会自动画出一个空的玻璃脑（只有轮廓），
        # 这正是我们想要展示的 "无显著结果" 图。
        plotting.plot_glass_brain(
            img,  # <--- 改为 img
            threshold=threshold,  # <--- 保持阈值
            colorbar=True,
            display_mode='lyrz',
            plot_abs=False,
            cmap=config.VIZ_CONFIG['cmap'],
            title=f"{contrast_name} (NS - No Significance)",
            output_file=str(output_dir / f"{contrast_name}_glass_ns.png")
        )
        return

    # 1. 玻璃脑图 (MIP)
    plotting.plot_glass_brain(
        img,
        threshold=threshold,
        colorbar=True,
        display_mode='lyrz',
        plot_abs=False,
        cmap=config.VIZ_CONFIG['cmap'],
        title=f"{contrast_name} (Cluster FWE p<{alpha})",
        output_file=str(output_dir / f"{contrast_name}_glass_fwe.png")
    )

    # 2. 切片图 (Ortho)
    plotting.plot_stat_map(
        img,
        threshold=threshold,
        display_mode='ortho',
        cut_coords=plotting.find_xyz_cut_coords(img),
        title=f"{contrast_name} (-log10 p-corr)",
        cmap=config.VIZ_CONFIG['cmap'],
        output_file=str(output_dir / f"{contrast_name}_ortho_fwe.png")
    )

    # 3. 聚类表格 (CSV)
    # 注意: get_clusters_table 通常针对 Z/T map。
    # 对于 log_p map，stat_threshold 即为显著性阈值。
    try:
        table = get_clusters_table(
            img,
            stat_threshold=threshold,
            cluster_threshold=0,  # 已经在置换检验中做过cluster控制，这里只要显示显著的体素
            two_sided=False
        )
        if table is not None:
            table.to_csv(output_dir / f"{contrast_name}_clusters.csv")
            logger.info(f"Cluster table saved with {len(table)} clusters.")
    except Exception as e:
        logger.warning(f"Could not generate cluster table: {e}")


# =========================
# 5. 主流程控制器
# =========================

def main():
    logger.info("=== Starting Enhanced fMRI GLM Pipeline ===")
    logger.info(
        f"Config: Smoothing={config.SMOOTHING_FWHM}mm, Standardize=True, Permutations={config.PERM_CONFIG['n_permutations']}")

    # --- Step 1: First Level Analysis ---
    logger.info("--- Step 1: First Level Analysis ---")

    tasks = []
    # 收集任务
    for sub in config.SUBJECTS:
        for run in config.RUNS:
            tasks.append((sub, run))

    successful_files = {con: [] for con in config.CONTRASTS.keys()}

    # 并行执行
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_subject_glm, sub, run, config.CONTRASTS): (sub, run) for sub, run in
                   tasks}

        for future in as_completed(futures):
            res = future.result()
            sub, run = futures[future]

            if res["success"]:
                logger.info(f"  [OK] {sub} Run {run}")
                # 收集成功的结果文件路径用于二阶分析
                for item in res["results"]:
                    successful_files[item["contrast"]].append(item["con_file"])
            else:
                logger.error(f"  [FAIL] {sub} Run {run}: {res['msg']}")

    # --- Step 1.5: Subject-Level Averaging (Fixed Effects) ---
    logger.info("--- Step 1.5: Combining Runs per Subject ---")

    # 结构: final_subjects_inputs = {'Contrast_Name': ['path/to/sub-01_mean.nii', ...]}
    final_subjects_inputs = {con: [] for con in config.CONTRASTS.keys()}

    # 按照被试分组
    from nilearn.image import mean_img

    # 获取所有成功处理的被试ID
    processed_subs = sorted(list(set([s for s, r in tasks])))

    for con_name in config.CONTRASTS.keys():
        for sub in processed_subs:
            # 找到该被试该对比度的所有 Run 文件
            sub_files = [f for f in successful_files[con_name] if f'{sub}' in str(f) or f'{sub}/' in str(f)]

            if len(sub_files) == 0:
                continue

            # 如果只有一个 Run，直接用
            if len(sub_files) == 1:
                final_subjects_inputs[con_name].append(sub_files[0])
            else:
                # 如果有多个 Run (Run3, Run4)，计算平均图 (Fixed Effects 近似)
                # 这样每个被试只贡献一个输入
                out_mean_dir = config.OUTPUT_ROOT / "1st_level" / sub
                out_mean_file = out_mean_dir / f"{sub}_{con_name}_mean.nii.gz"

                # 仅当文件不存在时计算，避免重复
                if not out_mean_file.exists():
                    logger.info(f"  Averaging {len(sub_files)} runs for {sub} - {con_name}")
                    avg_img = mean_img(sub_files)
                    avg_img.to_filename(out_mean_file)

                final_subjects_inputs[con_name].append(str(out_mean_file))

    # --- Step 2: Second Level Analysis (Group) ---
    logger.info("--- Step 2: Second Level Analysis (Cluster FWE) ---")

    grp_out_root = config.OUTPUT_ROOT / "2nd_level_FWE"

    for con_name, files in final_subjects_inputs.items():
        n_subs = len(files)
        # 简单检查：文件数量是否足够 (例如 >= 15)
        if n_subs < 5:
            logger.warning(f"Skipping {con_name}: Not enough subjects ({n_subs})")
            continue

        logger.info(f"Processing Contrast: {con_name} (Subjects: {n_subs})")

        try:
            # 运行置换检验
            log_p_map = run_group_permutation(con_name, files, grp_out_root)

            # 可视化结果
            visualize_results(log_p_map, con_name, grp_out_root)

        except Exception as e:
            logger.error(f"Group analysis failed for {con_name}: {e}", exc_info=True)

    logger.info("=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()