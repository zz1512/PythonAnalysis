#!/usr/bin/env python3
"""
main_analysis.py
统一执行入口：支持单独或联合运行一二阶分析 (FDR/FWE 修复版)

用途
- 学习阶段（默认 run3-4）的一阶 GLM + 二阶组水平分析（FDR 或 cluster-FWE）。
- 输出显著簇、阈值化统计图、以及用于下游 ROI 定义的 mask（如有配置）。

输入（来自 glm_config.py）
- BOLD：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/run{run}/*.nii(.gz)`
- events：`${PYTHON_METAPHOR_ROOT}/data_events/{sub}/{sub}_run-{run}_events.tsv`
- confounds：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/multi_reg/*confounds_timeseries.tsv`

输出（`${PYTHON_METAPHOR_ROOT}/glm_analysis_fwe_final`）
- 一阶：每被试对比图（effect/z/t 等，视实现）
- 二阶：阈值化 group map、cluster 报告、日志 `analysis.log`

注意
- 该脚本是“定位哪里”的证据（Layer 1），不直接做表征几何/RSA。
- 如果你想把前后测也纳入 GLM，需要在 glm_config.py 扩展 RUNS 与对比定义。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import norm  # <--- 新增引入，用于动态计算Z值
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn import plotting
import logging
import traceback

# 导入自定义模块
try:
    from glm_config import config
    import glm_utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(
                            config.OUTPUT_ROOT / "analysis.log" if config.OUTPUT_ROOT.exists() else "analysis.log",
                            mode='a'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


# ==========================================
# 1. 一阶分析逻辑
# ==========================================
def process_single_subject(sub):
    """单个被试的一阶分析处理函数"""
    try:
        imgs, events, confounds_list, sample_masks = [], [], [], []
        valid_run = False

        for run in config.RUNS:
            # 路径处理 (支持 .nii 和 .nii.gz)
            f_path_obj = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
            if not f_path_obj.exists():
                if f_path_obj.with_suffix('.nii.gz').exists():
                    f_path_obj = f_path_obj.with_suffix('.nii.gz')
                else:
                    logger.warning(f"Image not found: {f_path_obj}")
                    continue

            c_path = str(config.CONFOUNDS_TPL).format(sub=sub, run=run)
            e_path = str(config.EVENT_TPL).format(sub=sub, run=run)

            if not Path(c_path).exists() or not Path(e_path).exists():
                logger.warning(f"Missing confounds/events for {sub} run {run}")
                continue

            # 加载 Confounds
            conf, mask = glm_utils.robust_load_confounds(c_path, config.DENOISE_STRATEGY)

            imgs.append(str(f_path_obj))
            events.append(pd.read_csv(e_path, sep='\t'))
            confounds_list.append(conf)
            sample_masks.append(mask)
            valid_run = True

        if not valid_run:
            return None

        # 建模
        glm = FirstLevelModel(
            t_r=config.TR,
            slice_time_ref=0.5,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            noise_model='ar1',
            standardize=True,
            signal_scaling=False,
            hrf_model='spm + derivative',
            drift_model='cosine',
            high_pass=1 / 128.0,
            minimize_memory=False,
            verbose=0
        )

        glm.fit(imgs, events, confounds_list, sample_masks=sample_masks)

        # 输出结果
        outputs = {}
        for name, formula in config.CONTRASTS.items():
            z_map = glm.compute_contrast(formula, output_type='effect_size')
            out_path = config.OUTPUT_ROOT / "1st_level" / sub / f"{sub}_{name}_con.nii.gz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            z_map.to_filename(out_path)
            outputs[name] = str(out_path)

        return outputs

    except Exception as e:
        logger.error(f"Failed processing {sub}: {e}")
        logger.error(traceback.format_exc())
        return None


def run_first_level_stage():
    logger.info(">>> Stage 1: First Level Analysis")
    results = Parallel(n_jobs=config.N_JOBS_1ST_LEVEL)(
        delayed(process_single_subject)(sub) for sub in config.SUBJECTS
    )
    logger.info("Stage 1 Completed.")


# ==========================================
# 2. 二阶分析逻辑
# ==========================================
def collect_contrast_files():
    """收集所有已经生成的一阶文件"""
    inputs = {con: [] for con in config.CONTRASTS}
    for sub in config.SUBJECTS:
        for con in config.CONTRASTS:
            p = config.OUTPUT_ROOT / "1st_level" / sub / f"{sub}_{con}_con.nii.gz"
            if p.exists():
                inputs[con].append(str(p))
    return inputs


def run_second_level_stage():
    method = config.SECOND_LEVEL_METHOD
    logger.info(f">>> Stage 2: Second Level Analysis using [{method.upper()}]")

    # 检测 GPU
    gpu_available = False
    if config.USE_GPU:
        try:
            import gpu_permutation
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                logger.info("GPU acceleration enabled.")
        except ImportError:
            logger.warning("GPU module import failed. Using CPU.")

    # 收集文件
    contrast_groups = collect_contrast_files()

    for con_name, files in contrast_groups.items():
        if len(files) < 10:
            logger.warning(f"Skipping {con_name}: Not enough subjects ({len(files)})")
            continue

        logger.info(f"Processing {con_name} (N={len(files)})...")

        # [修复 1] 动态输出目录：根据方法名区分文件夹 (FDR vs FWE)
        out_dir = config.OUTPUT_ROOT / f"2nd_level_{method.upper()}" / con_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # A. 计算组掩膜
        group_mask = glm_utils.compute_group_mask(files, threshold_ratio=0.7)
        group_mask.to_filename(out_dir / "group_mask.nii.gz")

        # ==========================================
        # 分支 B1: FDR 校正 (Parametric)
        # ==========================================
        if method == 'fdr':
            logger.info("Running Parametric GLM with FDR correction...")

            # 1. 构建设计矩阵
            design_matrix = pd.DataFrame([1] * len(files), columns=['intercept'])

            # 2. 拟合二阶模型
            model = SecondLevelModel(mask_img=group_mask)
            model.fit(files, design_matrix=design_matrix)

            # 3. 计算 Z 统计图
            z_map = model.compute_contrast('intercept', output_type='z_score')
            z_map.to_filename(out_dir / "z_score_map.nii.gz")

            # 4. 应用 FDR 阈值
            try:
                thresh_z_map, thresh_val = threshold_stats_img(
                    z_map,
                    alpha=config.FDR_PARAMS['alpha'],
                    height_control='fdr',
                    cluster_threshold=config.FDR_PARAMS['cluster_min_size']
                )
                logger.info(f"FDR threshold calculated: Z > {thresh_val:.3f}")

                # 保存结果
                thresh_z_map.to_filename(out_dir / "fdr_corrected_z_map.nii.gz")

                # 报告生成
                viz_title = f"{con_name} (FDR q<{config.FDR_PARAMS['alpha']})"

                glm_utils.generate_cluster_report(
                    z_map,  # 传入原始 Z 图
                    thresh_val,  # 传入计算出的 FDR 阈值
                    out_dir / "cluster_report_fdr.csv",
                    min_voxels=config.FDR_PARAMS['cluster_min_size'],
                    atlas_name=config.REPORT_PARAMS['atlas_name']
                )

                # 绘图
                plotting.plot_glass_brain(
                    thresh_z_map, colorbar=True, plot_abs=False,
                    title=viz_title, output_file=out_dir / "glass_brain_fdr.png"
                )

            except Exception as e:
                logger.warning(f"FDR correction found no significant clusters: {e}")
                # 备选：生成未校正图
                logger.info("Generating uncorrected p<0.001 map for inspection...")
                unc_thresh = norm.isf(0.001)
                glm_utils.generate_cluster_report(
                    z_map, unc_thresh, out_dir / "cluster_report_uncorrected.csv",
                    min_voxels=20
                )

        # ==========================================
        # 分支 B2: Cluster-FWE (Non-Parametric)
        # ==========================================
        elif method == 'cluster_fwe':
            log_p_map = None

            # GPU 路径
            if gpu_available:
                try:
                    logger.info("Running GPU Permutation...")
                    log_p_map, raw_t = gpu_permutation.run_permutation(
                        files, group_mask,
                        n_perms=config.PERM_PARAMS['n_permutations'],
                        cluster_p_thresh=config.PERM_PARAMS['cluster_forming_p'],
                        n_jobs=config.N_JOBS_PERM
                    )
                    raw_t.to_filename(out_dir / "raw_t_stat.nii.gz")
                except Exception as e:
                    logger.error(f"GPU execution failed: {e}. Switching to CPU.")

            # CPU 路径 (Nilearn)
            if log_p_map is None:
                logger.info("Running CPU Permutation (Nilearn)...")
                design_matrix = pd.DataFrame([1] * len(files), columns=['intercept'])

                # [修复 2] 动态计算 cluster forming threshold
                # 这样你在 config 里改了 0.005，这里也会自动变
                threshold_z = norm.isf(config.PERM_PARAMS['cluster_forming_p'])
                logger.info(
                    f"Using cluster forming threshold: Z > {threshold_z:.3f} (p < {config.PERM_PARAMS['cluster_forming_p']})")

                res = non_parametric_inference(
                    files, design_matrix=design_matrix, second_level_contrast='intercept',
                    mask=group_mask, n_perm=config.PERM_PARAMS['n_permutations'],
                    two_sided_test=False, n_jobs=config.N_JOBS_PERM,
                    threshold=threshold_z
                )
                log_p_map = res['logp_max_size']

            log_p_map.to_filename(out_dir / "log_p_map.nii.gz")

            # C. 报告
            viz_thresh = -np.log10(config.PERM_PARAMS['target_alpha'])

            glm_utils.generate_cluster_report(
                log_p_map, viz_thresh,
                out_dir / "cluster_report.csv",
                min_voxels=config.REPORT_PARAMS['min_cluster_voxels'],
                atlas_name=config.REPORT_PARAMS['atlas_name']
            )

            glm_utils.save_binary_mask(
                log_p_map, viz_thresh, out_dir / f"{con_name}_roi_mask.nii.gz",
                min_cluster_voxels=config.REPORT_PARAMS['min_cluster_voxels']
            )

            try:
                plotting.plot_glass_brain(
                    log_p_map, threshold=viz_thresh, colorbar=True,
                    title=f"{con_name} (FWE < {config.PERM_PARAMS['target_alpha']})",
                    output_file=out_dir / "glass_brain.png"
                )
            except Exception:
                pass

    logger.info("Stage 2 Completed.")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if config.RUN_FIRST_LEVEL:
        run_first_level_stage()

    if config.RUN_SECOND_LEVEL:
        run_second_level_stage()

    logger.info("All requested analyses finished.")
