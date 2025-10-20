# data_loader.py
# 数据加载模块

import pandas as pd
from pathlib import Path
from utils import log

def load_roi_masks(config):
    """加载ROI mask文件"""
    log("加载ROI masks", config)
    
    roi_masks = {}
    roi_files = list(config.roi_dir.glob("*.nii.gz"))
    
    if not roi_files:
        raise FileNotFoundError(f"在{config.roi_dir}中未找到ROI mask文件")
    
    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        roi_masks[roi_name] = str(roi_file)
        log(f"  加载ROI: {roi_name}", config)
    
    log(f"总共加载了{len(roi_masks)}个ROI masks", config)
    return roi_masks

def load_lss_trial_data(subject, run, config):
    """加载LSS分析的trial数据和beta图像"""
    lss_dir = config.lss_root / subject / f"run-{run}_LSS"
    
    if not lss_dir.exists():
        log(f"LSS目录不存在: {lss_dir}", config)
        return None, None
    
    # 加载trial信息
    trial_map_path = lss_dir / "trial_info.csv"
    if not trial_map_path.exists():
        log(f"trial_info.csv不存在: {trial_map_path}", config)
        return None, None
    
    trial_info = pd.read_csv(trial_map_path)
    
    # 加载所有beta图像
    beta_images = []
    valid_trials = []
    
    for _, trial in trial_info.iterrows():
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        if beta_path.exists():
            beta_images.append(str(beta_path))
            # 添加run信息到trial数据中
            trial_with_run = trial.copy()
            trial_with_run['run'] = run
            trial_with_run['global_trial_index'] = f"run{run}_trial{trial['trial_index']}"
            valid_trials.append(trial_with_run)
        else:
            log(f"Beta图像缺失: {beta_path}", config)
    
    if len(beta_images) == 0:
        log(f"没有找到有效的beta图像: {subject} run-{run}", config)
        return None, None
    
    valid_trials_df = pd.DataFrame(valid_trials)
    log(f"加载 {subject} run-{run}: {len(beta_images)}个trial", config)
    
    return beta_images, valid_trials_df

def load_multi_run_lss_data(subject, runs, config):
    """加载并合并多个run的LSS trial数据"""
    log(f"开始加载 {subject} 的多run数据: runs {runs}", config)
    
    all_beta_images = []
    all_trial_info = []
    
    for run in runs:
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        if beta_images is not None and trial_info is not None:
            all_beta_images.extend(beta_images)
            all_trial_info.append(trial_info)
            log(f"  成功加载 run-{run}: {len(beta_images)}个trial", config)
        else:
            log(f"  跳过 run-{run}: 数据加载失败", config)
    
    if not all_beta_images:
        log(f"错误: {subject} 没有加载到任何有效的trial数据", config)
        return None, None
    
    # 合并所有run的trial信息
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    
    # 重新索引trial，确保每个trial有唯一标识
    combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
    
    log(f"合并完成 {subject}: 总共{len(all_beta_images)}个trial (来自{len(all_trial_info)}个run)", config)
    
    # 打印每个条件的trial数量统计
    if 'original_condition' in combined_trial_info.columns:
        condition_counts = combined_trial_info['original_condition'].value_counts()
        log(f"  条件统计: {dict(condition_counts)}", config)
    
    return all_beta_images, combined_trial_info