#!/usr/bin/env python3
"""
fMRI GLM 分析优化版本 - 集成化流程

主要优化：
1. 移除单独的平滑步骤，在GLM分析中直接使用smoothing_fwhm参数
2. 集成一阶、二阶分析和可视化为一个完整流程
3. 支持灵活的平滑参数配置（可选择是否平滑）
4. 优化内存使用和并行处理
5. 增强错误处理和日志记录

理论依据：
- fMRIPrep输出的数据已经过标准化预处理，包括头动校正、时间层校正等
- nilearn.glm.FirstLevelModel的smoothing_fwhm参数可以在GLM分析时直接应用平滑
- 这样避免了额外的I/O操作和存储空间占用
- 平滑操作在GLM建模时进行更符合统计学原理
"""

import os
# 多进程环境优化
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

import gc
import json
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from nilearn import image, datasets, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.image import new_img_like

# =========================
# 配置参数
# =========================

# 导入配置
from config_enhance import default_config
config = default_config
config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def update_config(new_config):
    """更新全局配置"""
    global config
    config = new_config

# 设置日志
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_ROOT / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# 工具函数
# =========================

def load_motion_confounds(sub: str, run: int) -> Optional[pd.DataFrame]:
    """加载头动参数"""
    motion_path = Path(config.MOTION_TPL.format(sub=sub, run=run))
    if not motion_path.exists():
        logger.warning(f"Motion file not found: {motion_path}")
        return None
    
    try:
        df = pd.read_csv(motion_path, sep='\t', header=None)
        df.columns = [f'motion_{i}' for i in range(df.shape[1])]
        logger.debug(f"Loaded motion confounds: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading motion confounds: {e}")
        return None

def load_events_from_table(path: Path) -> Optional[pd.DataFrame]:
    """加载事件文件"""
    if not path.exists():
        logger.warning(f"Events file not found: {path}")
        return None
    
    try:
        df = pd.read_csv(path, sep='\t')
        required_cols = ['onset', 'duration', 'trial_type']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in events file: {path}")
            return None
        
        # 过滤有效的试验类型
        valid_trials = df[df['trial_type'].isin(['yy', 'kj'])].copy()
        if valid_trials.empty:
            logger.warning(f"No valid trials (yy/kj) found in: {path}")
            return None
        
        logger.debug(f"Loaded events: {valid_trials.shape[0]} trials")
        return valid_trials
    except Exception as e:
        logger.error(f"Error loading events: {e}")
        return None

def infer_tr_from_img(img) -> float:
    """从图像头文件推断TR"""
    try:
        tr = float(img.header.get_zooms()[3])
        return tr if tr > 0 else config.TR_FALLBACK
    except:
        return config.TR_FALLBACK

def contrast_vector(name2weight: Dict[str, float], dm_cols: List[str]) -> np.ndarray:
    """构建对比向量"""
    vec = np.zeros(len(dm_cols))
    for name, weight in name2weight.items():
        if name in dm_cols:
            vec[dm_cols.index(name)] = weight
    return vec

def build_design_matrix(fmri_img, events_df: pd.DataFrame) -> pd.DataFrame:
    """构建设计矩阵（仅事件相关）"""
    tr = infer_tr_from_img(fmri_img)
    n_scans = fmri_img.shape[3]
    frame_times = np.arange(n_scans) * tr
    
    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model=config.HRF_MODEL,
        drift_model=None,  # 在FirstLevelModel中处理
        high_pass=None     # 在FirstLevelModel中处理
    )
    return dm

# =========================
# 一阶分析
# =========================

def run_first_level_analysis(sub: str, run: int, contrasts: Dict[str, Dict[str, float]]) -> Dict:
    """运行一阶GLM分析"""
    logger.info(f"Starting first-level analysis: {sub}, run {run}")
    
    # 构建文件路径
    fmri_path = Path(config.FMRI_TPL.format(sub=sub, run=run))
    events_path = Path(config.EVENT_TPL.format(sub=sub, run=run))
    
    # 检查文件存在性
    if not fmri_path.exists():
        logger.error(f"fMRI file not found: {fmri_path}")
        return {"success": False, "error": "fMRI file missing"}
    
    # 加载数据
    events = load_events_from_table(events_path)
    if events is None or events.empty:
        logger.error(f"Invalid events file: {events_path}")
        return {"success": False, "error": "Events file invalid"}
    
    try:
        # 加载fMRI数据
        logger.debug(f"Loading fMRI data: {fmri_path}")
        fmri_img = image.load_img(fmri_path)
        
        # 加载头动参数
        motion_df = load_motion_confounds(sub, run)
        
        # 推断TR
        tr = infer_tr_from_img(fmri_img)
        logger.debug(f"TR: {tr}s")
        
        # 创建FirstLevelModel - 关键优化：直接在这里应用平滑
        model = FirstLevelModel(
            t_r=tr,  # 使用推断的TR值而不是配置中的固定值
            slice_time_ref=config.SLICE_TIME_REF,
            noise_model=config.NOISE_MODEL,
            hrf_model=config.HRF_MODEL,
            drift_model=config.DRIFT_MODEL,
            high_pass=config.HIGH_PASS,
            standardize=config.STANDARDIZE,
            signal_scaling=config.SIGNAL_SCALING,
            smoothing_fwhm=config.SMOOTHING_FWHM,  # 关键：在GLM中直接平滑
            minimize_memory=config.MINIMIZE_MEMORY,
            n_jobs=config.N_JOBS_GLM,
            verbose=config.VERBOSE
        )
        
        # 拟合模型
        logger.debug("Fitting GLM model...")
        model = model.fit(fmri_img, events=events, confounds=motion_df)
        
        # 获取完整设计矩阵
        dm_full = model.design_matrices_[0]
        dm_cols = list(dm_full.columns)
        logger.debug(f"Design matrix columns: {dm_cols}")
        
        # 创建输出目录
        run_tag = f"run-{run}"
        sub_out = config.OUTPUT_ROOT / sub
        sub_out.mkdir(exist_ok=True)
        spm_out_dir = sub_out / f"{run_tag}_spm_style"
        spm_out_dir.mkdir(exist_ok=True)
        
        # 保存模型信息
        model_info = {
            "subject": sub,
            "run": run,
            "tr": tr,
            "n_scans": fmri_img.shape[3],
            "smoothing_fwhm": config.SMOOTHING_FWHM,
            "design_matrix_columns": dm_cols,
            "n_events": len(events),
            "contrasts": list(contrasts.keys())
        }
        
        with open(spm_out_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # 计算并保存对比
        contrast_results = []
        for idx, (cname, spec) in enumerate(contrasts.items(), start=1):
            logger.debug(f"Computing contrast: {cname}")
            
            # 构建对比向量
            contrast_vec = contrast_vector(spec, dm_cols)
            
            # 计算对比
            con_img = model.compute_contrast(contrast_vec, output_type="effect_size")
            t_img = model.compute_contrast(contrast_vec, output_type="stat")
            
            # 保存结果
            con_path = spm_out_dir / f"con_{idx:04d}.nii.gz"
            t_path = spm_out_dir / f"spmT_{idx:04d}.nii.gz"
            
            nib.save(con_img, str(con_path))
            nib.save(t_img, str(t_path))
            
            contrast_results.append({
                "name": cname,
                "index": idx,
                "con_file": str(con_path),
                "t_file": str(t_path)
            })
        
        # 清理内存
        del model, fmri_img, dm_full
        gc.collect()
        
        logger.info(f"First-level analysis completed: {sub}, run {run}")
        return {
            "success": True,
            "subject": sub,
            "run": run,
            "output_dir": str(spm_out_dir),
            "contrasts": contrast_results
        }
        
    except Exception as e:
        logger.error(f"Error in first-level analysis {sub}, run {run}: {e}")
        return {"success": False, "error": str(e)}

# =========================
# 二阶分析
# =========================

def run_second_level_analysis(contrast_name: str, contrast_index: int, run: int) -> Dict:
    """运行二阶组分析"""
    logger.info(f"Starting second-level analysis: {contrast_name}, run {run}")
    
    try:
        # 收集所有被试的对比文件
        pattern = config.OUTPUT_ROOT / f"sub-*" / f"run-{run}_spm_style" / f"con_{contrast_index:04d}.nii.gz"
        con_paths = list(config.OUTPUT_ROOT.glob(f"sub-*/run-{run}_spm_style/con_{contrast_index:04d}.nii.gz"))
        con_paths = sorted([str(p) for p in con_paths])
        
        if len(con_paths) < 3:
            logger.error(f"Insufficient subjects for group analysis: {len(con_paths)}")
            return {"success": False, "error": "Insufficient subjects"}
        
        logger.info(f"Found {len(con_paths)} subjects for group analysis")
        
        # 创建输出目录
        second_out_dir = config.OUTPUT_ROOT / "2nd_level" / f"run{run}" / contrast_name.replace(" ", "_").replace(">", "vs")
        second_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成组掩膜
        logger.debug("Creating group mask...")
        group_mask = None
        ref_img = None
        
        for p in con_paths:
            img = nib.load(p)
            dat = img.get_fdata()
            mask = np.isfinite(dat) & (dat != 0)
            
            if group_mask is None:
                group_mask = mask
                ref_img = img
            else:
                group_mask = group_mask & mask
        
        mask_img = new_img_like(ref_img, group_mask.astype(np.uint8))
        nib.save(mask_img, str(second_out_dir / "mask.nii"))
        
        # 二阶模型
        logger.debug("Fitting second-level model...")
        n_subjects = len(con_paths)
        design_matrix = pd.DataFrame({"intercept": np.ones(n_subjects)})
        
        slm = SecondLevelModel(
            mask_img=mask_img,
            smoothing_fwhm=None,  # 已在一阶分析中平滑
            minimize_memory=config.MINIMIZE_MEMORY
        )
        slm = slm.fit(con_paths, design_matrix=design_matrix)
        
        # 计算组统计
        con_img = slm.compute_contrast("intercept", output_type="effect_size")
        t_img = slm.compute_contrast("intercept", output_type="stat")
        
        # 保存结果
        con_path = second_out_dir / f"con_{contrast_index:04d}.nii"
        t_path = second_out_dir / f"spmT_{contrast_index:04d}.nii"
        
        nib.save(con_img, str(con_path))
        nib.save(t_img, str(t_path))
        
        # 计算ResMS
        logger.debug("Computing ResMS...")
        df = n_subjects - 1
        stack = []
        for p in con_paths:
            img = nib.load(p)
            if img.shape != mask_img.shape:
                img = image.resample_to_img(img, mask_img, interpolation="continuous")
            stack.append(img.get_fdata())
        
        stack = np.stack(stack, axis=-1)
        mean_ = np.nanmean(stack, axis=-1, keepdims=True)
        rss = np.nansum((stack - mean_)**2, axis=-1)
        rss[~group_mask] = np.nan
        resms = rss / df
        
        resms_img = new_img_like(mask_img, resms)
        nib.save(resms_img, str(second_out_dir / "ResMS.nii"))
        
        # 保存元信息
        meta = {
            "contrast_name": contrast_name,
            "contrast_index": contrast_index,
            "run": run,
            "n_subjects": n_subjects,
            "df": int(df),
            "smoothing_applied_in_first_level": config.SMOOTHING_FWHM,
            "outputs": {
                "mask": "mask.nii",
                "con": f"con_{contrast_index:04d}.nii",
                "t": f"spmT_{contrast_index:04d}.nii",
                "ResMS": "ResMS.nii"
            },
            "subject_files": con_paths
        }
        
        with open(second_out_dir / "model_info.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Second-level analysis completed: {contrast_name}")
        return {
            "success": True,
            "contrast_name": contrast_name,
            "output_dir": str(second_out_dir),
            "n_subjects": n_subjects,
            "stat_file": str(t_path)
        }
        
    except Exception as e:
        logger.error(f"Error in second-level analysis {contrast_name}: {e}")
        return {"success": False, "error": str(e)}

# =========================
# 可视化
# =========================

def create_visualization(second_level_result: Dict) -> Dict:
    """创建统计结果可视化"""
    if not second_level_result["success"]:
        return {"success": False, "error": "Second-level analysis failed"}
    
    try:
        output_dir = Path(second_level_result["output_dir"])
        stat_file = second_level_result["stat_file"]
        contrast_name = second_level_result["contrast_name"]
        
        logger.info(f"Creating visualization for: {contrast_name}")
        
        # 加载统计图像和掩膜
        stat_img = image.load_img(stat_file)
        mask_file = output_dir / "mask.nii"
        mask_img = nib.load(str(mask_file)) if mask_file.exists() else None
        
        # 重采样（如果需要）
        if mask_img is not None and stat_img.shape != mask_img.shape:
            stat_img = image.resample_to_img(stat_img, mask_img, interpolation="continuous")
        
        # 阈值处理
        cfg = config.SECOND_LEVEL_CONFIG
        if cfg["use_fdr"]:
            thr_img, thr_val = threshold_stats_img(
                stat_img, alpha=cfg["alpha"], height_control="fdr",
                cluster_threshold=0, two_sided=cfg["two_sided"]
            )
            thr_label = f"FDR q<{cfg['alpha']}"
        else:
            thr_img, thr_val = threshold_stats_img(
                stat_img, alpha=cfg["p_unc"], height_control="fpr",
                cluster_threshold=cfg["cluster_k"], two_sided=cfg["two_sided"]
            )
            thr_label = f"p<{cfg['p_unc']} unc., k≥{cfg['cluster_k']}"
        
        logger.debug(f"Threshold: {thr_val:.3f} ({thr_label})")
        
        # 生成可视化
        has_data = np.nanmax(np.abs(thr_img.get_fdata())) > 0
        coords = plotting.find_xyz_cut_coords(thr_img if has_data else stat_img)
        
        # 正交切片图
        disp = plotting.plot_stat_map(
            thr_img,
            threshold=thr_val,
            display_mode="ortho",
            cut_coords=coords,
            annotate=True,
            title=f"Second-level: {contrast_name} ({thr_label})"
        )
        ortho_path = output_dir / "stat_map_ortho.png"
        disp.savefig(str(ortho_path), dpi=220)
        disp.close()
        
        # 玻璃脑图
        disp2 = plotting.plot_glass_brain(
            thr_img, threshold=thr_val, display_mode="lyrz",
            colorbar=True, plot_abs=False, 
            title=f"Glass brain: {contrast_name} ({thr_label})"
        )
        glass_path = output_dir / "glass_brain.png"
        disp2.savefig(str(glass_path), dpi=220)
        disp2.close()
        
        # 聚类表
        try:
            clusters_df = get_clusters_table(
                stat_img=stat_img, stat_threshold=thr_val,
                cluster_threshold=(0 if cfg["use_fdr"] else cfg["cluster_k"]),
                two_sided=cfg["two_sided"]
            )
        except TypeError:
            clusters_df = get_clusters_table(thr_img, min_distance=8)
        
        # 脑区标注
        if clusters_df is not None and not clusters_df.empty:
            clusters_df = label_with_harvard_oxford(clusters_df)
        
        # 保存聚类表
        csv_path = output_dir / "clusters_table.csv"
        if clusters_df is None or clusters_df.empty:
            logger.info("No suprathreshold clusters found")
            pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            clusters_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"Found {len(clusters_df)} clusters")
        
        logger.info(f"Visualization completed: {contrast_name}")
        return {
            "success": True,
            "contrast_name": contrast_name,
            "ortho_plot": str(ortho_path),
            "glass_brain": str(glass_path),
            "clusters_table": str(csv_path),
            "threshold_info": {"value": thr_val, "label": thr_label}
        }
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        return {"success": False, "error": str(e)}

def label_with_harvard_oxford(df: pd.DataFrame) -> pd.DataFrame:
    """使用Harvard-Oxford图谱标注脑区"""
    if df is None or df.empty:
        return df
    
    try:
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        ho_subc = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        atlas_imgs = [image.load_img(ho_cort.maps), image.load_img(ho_subc.maps)]
        labels = [ho_cort.labels, ho_subc.labels]
        
        # 检测坐标列名
        xyz_cols = None
        for cols in [('X','Y','Z'), ('x','y','z')]:
            if set(cols).issubset(df.columns):
                xyz_cols = cols
                break
        
        if xyz_cols is None:
            logger.warning("No XYZ coordinates found in clusters table")
            return df
        
        # 标注脑区
        regions = []
        for _, row in df.iterrows():
            coord = [row[xyz_cols[0]], row[xyz_cols[1]], row[xyz_cols[2]]]
            region_names = []
            
            for atlas_img, atlas_labels in zip(atlas_imgs, labels):
                try:
                    from nilearn.image import coord_transform
                    voxel_coord = coord_transform(coord[0], coord[1], coord[2], 
                                                np.linalg.inv(atlas_img.affine))
                    voxel_coord = [int(round(x)) for x in voxel_coord]
                    
                    if all(0 <= voxel_coord[i] < atlas_img.shape[i] for i in range(3)):
                        label_idx = atlas_img.get_fdata()[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                        if 0 < label_idx < len(atlas_labels):
                            region_names.append(atlas_labels[int(label_idx)])
                except:
                    continue
            
            regions.append('; '.join(region_names) if region_names else 'Unknown')
        
        df['Region'] = regions
        return df
        
    except Exception as e:
        logger.warning(f"Error in brain region labeling: {e}")
        return df

# =========================
# 主流程
# =========================

def run_complete_analysis():
    """运行完整的fMRI GLM分析流程"""
    logger.info("Starting complete fMRI GLM analysis pipeline")
    logger.info(f"Smoothing FWHM: {config.SMOOTHING_FWHM}mm" if config.SMOOTHING_FWHM else "No smoothing applied")
    
    # 第一步：一阶分析
    logger.info("=== STEP 1: First-level analysis ===")
    first_level_results = []
    
    if config.USE_MULTIPROCESS:
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = []
            for sub in config.SUBJECTS:
                for run in config.RUNS:
                    future = executor.submit(run_first_level_analysis, sub, run, config.CONTRASTS)
                    futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                first_level_results.append(result)
                if result["success"]:
                    logger.info(f"Completed: {result['subject']}, run {result['run']}")
                else:
                    logger.error(f"Failed: {result.get('subject', 'unknown')}, error: {result['error']}")
    else:
        for sub in config.SUBJECTS:
            for run in config.RUNS:
                result = run_first_level_analysis(sub, run, config.CONTRASTS)
                first_level_results.append(result)
    
    # 统计一阶分析结果
    successful_first = [r for r in first_level_results if r["success"]]
    logger.info(f"First-level analysis: {len(successful_first)}/{len(first_level_results)} successful")
    
    if len(successful_first) < 3:
        logger.error("Insufficient successful first-level analyses for group analysis")
        return
    
    # 第二步：二阶分析
    logger.info("=== STEP 2: Second-level analysis ===")
    second_level_results = []
    
    for run in config.RUNS:
        for idx, (contrast_name, _) in enumerate(config.CONTRASTS.items(), start=1):
            result = run_second_level_analysis(contrast_name, idx, run)
            second_level_results.append(result)
            if result["success"]:
                logger.info(f"Completed second-level: {contrast_name}, run {run}")
            else:
                logger.error(f"Failed second-level: {contrast_name}, error: {result['error']}")
    
    # 第三步：可视化
    logger.info("=== STEP 3: Visualization ===")
    visualization_results = []
    
    for result in second_level_results:
        if result["success"]:
            viz_result = create_visualization(result)
            visualization_results.append(viz_result)
            if viz_result["success"]:
                logger.info(f"Completed visualization: {viz_result['contrast_name']}")
            else:
                logger.error(f"Failed visualization: {result['contrast_name']}")
    
    # 生成总结报告
    logger.info("=== STEP 4: Summary report ===")
    summary = {
        "analysis_config": {
            "smoothing_fwhm": config.SMOOTHING_FWHM,
            "subjects": config.SUBJECTS,
            "runs": config.RUNS,
            "contrasts": list(config.CONTRASTS.keys()),
            "use_multiprocess": config.USE_MULTIPROCESS
        },
        "first_level_results": {
            "total": len(first_level_results),
            "successful": len(successful_first),
            "failed": len(first_level_results) - len(successful_first)
        },
        "second_level_results": {
            "total": len(second_level_results),
            "successful": len([r for r in second_level_results if r["success"]]),
            "failed": len([r for r in second_level_results if not r["success"]])
        },
        "visualization_results": {
            "total": len(visualization_results),
            "successful": len([r for r in visualization_results if r["success"]]),
            "failed": len([r for r in visualization_results if not r["success"]])
        }
    }
    
    summary_path = config.OUTPUT_ROOT / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Analysis pipeline completed. Summary saved to: {summary_path}")
    logger.info(f"Results directory: {config.OUTPUT_ROOT}")

if __name__ == "__main__":
    run_complete_analysis()