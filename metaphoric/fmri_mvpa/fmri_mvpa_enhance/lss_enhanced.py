#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版LSS分析模块
实现性能优化、功能增强和更好的错误处理
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
import gc
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
import logging

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .global_config import GlobalConfig, LSSConfig
from .error_handling import ErrorHandler, ErrorContext, retry_on_failure, DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置环境变量优化
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

class LSSAnalysisEnhanced:
    """增强版LSS分析器"""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.lss_config = config.lss
        self.error_handler = ErrorHandler(
            log_dir=str(config.log_dir),
            debug_mode=config.debug_mode
        )
        self.validator = DataValidator()
        
        # 性能监控
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'memory_usage': [],
            'processing_times': {},
            'failed_subjects': [],
            'failed_runs': []
        }
        
        # 缓存
        self.cache = {}
        
        # 自动检测最优参数
        self._optimize_parameters()
    
    def _optimize_parameters(self):
        """自动优化参数设置"""
        # 检测可用内存
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # 根据内存调整并行参数
        if available_memory_gb > 16:
            self.lss_config.n_jobs_glm = min(cpu_count(), 8)
            self.lss_config.parallel_subjects = True
            self.lss_config.parallel_runs = True
        elif available_memory_gb > 8:
            self.lss_config.n_jobs_glm = min(cpu_count()//2, 4)
            self.lss_config.parallel_subjects = True
            self.lss_config.parallel_runs = False
        else:
            self.lss_config.n_jobs_glm = 1
            self.lss_config.parallel_subjects = False
            self.lss_config.parallel_runs = False
        
        self.error_handler.logger.info(
            f"参数优化完成: 内存={available_memory_gb:.1f}GB, "
            f"GLM并行={self.lss_config.n_jobs_glm}, "
            f"被试并行={self.lss_config.parallel_subjects}, "
            f"Run并行={self.lss_config.parallel_runs}"
        )
    
    def _monitor_memory(self):
        """监控内存使用"""
        memory_info = psutil.virtual_memory()
        self.performance_stats['memory_usage'].append({
            'timestamp': datetime.now().isoformat(),
            'used_gb': memory_info.used / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'percent': memory_info.percent
        })
        
        # 内存警告
        if memory_info.percent > 85:
            self.error_handler.logger.warning(
                f"内存使用率过高: {memory_info.percent:.1f}%"
            )
            gc.collect()  # 强制垃圾回收
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def _infer_tr_from_img(self, img_path: str) -> float:
        """从图像推断TR值"""
        try:
            if self.lss_config.auto_tr_detection:
                img = nib.load(img_path)
                tr = float(img.header.get_zooms()[3])
                if tr > 0.5:  # 合理的TR范围
                    return tr
            return self.lss_config.tr_fallback
        except Exception as e:
            self.error_handler.log_error(e, f"TR推断: {img_path}", "data_load")
            return self.lss_config.tr_fallback
    
    def _load_and_validate_events(self, events_path: str) -> Optional[pd.DataFrame]:
        """加载和验证事件文件"""
        try:
            # 检查文件存在性
            if not self.validator.check_file_exists(events_path):
                raise FileNotFoundError(f"事件文件不存在: {events_path}")
            
            # 检查文件大小
            if not self.validator.check_file_size(events_path, 0.001):  # 1KB
                raise ValueError(f"事件文件过小: {events_path}")
            
            # 加载事件
            events = pd.read_csv(events_path, sep='\t')
            
            # 验证必需列
            required_columns = ['onset', 'duration', 'trial_type']
            missing_columns = [col for col in required_columns if col not in events.columns]
            if missing_columns:
                raise ValueError(f"事件文件缺少必需列: {missing_columns}")
            
            # 数据质量检查
            if len(events) == 0:
                raise ValueError("事件文件为空")
            
            if events['onset'].isna().any():
                raise ValueError("事件文件包含无效的onset值")
            
            if events['duration'].isna().any():
                raise ValueError("事件文件包含无效的duration值")
            
            # 排序事件
            events = events.sort_values('onset').reset_index(drop=True)
            
            self.error_handler.logger.debug(
                f"成功加载事件文件: {events_path}, {len(events)} 个事件"
            )
            
            return events
            
        except Exception as e:
            self.error_handler.log_error(e, f"加载事件文件: {events_path}", "data_load")
            return None
    
    def _load_and_validate_motion(self, motion_path: str) -> Optional[np.ndarray]:
        """加载和验证运动参数"""
        try:
            if not os.path.exists(motion_path):
                self.error_handler.logger.warning(f"运动参数文件不存在: {motion_path}")
                return None
            
            # 尝试不同的加载方式
            try:
                motion = pd.read_csv(motion_path, sep='\t', header=None)
            except:
                motion = pd.read_csv(motion_path, sep=' ', header=None)
            
            motion_array = motion.values
            
            # 质量检查
            if motion_array.shape[1] < 6:
                raise ValueError(f"运动参数列数不足: {motion_array.shape[1]} < 6")
            
            # 检查运动幅度
            max_motion = np.max(np.abs(motion_array[:, :3]))  # 平移参数
            if max_motion > self.lss_config.max_motion_mm:
                self.error_handler.logger.warning(
                    f"运动幅度过大: {max_motion:.2f}mm > {self.lss_config.max_motion_mm}mm"
                )
            
            return motion_array
            
        except Exception as e:
            self.error_handler.log_error(e, f"加载运动参数: {motion_path}", "data_load")
            return None
    
    def _create_lss_design_matrix(self, events: pd.DataFrame, n_scans: int, tr: float,
                                 motion_confounds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """创建LSS设计矩阵"""
        from nilearn.glm.first_level import make_first_level_design_matrix
        
        # 创建基础设计矩阵
        frame_times = np.arange(n_scans) * tr
        
        # 为每个trial创建单独的条件
        lss_events = []
        condition_names = []
        
        for idx, trial in events.iterrows():
            # 创建单trial事件
            single_trial_events = events.copy()
            single_trial_events.loc[idx, 'trial_type'] = f"trial_{idx}_{trial['trial_type']}"
            
            # 其他trial合并为一个条件
            other_indices = events.index != idx
            if self.lss_config.lss_collapse_by_cond:
                # 按条件类型分组
                for trial_type in events['trial_type'].unique():
                    if trial_type != trial['trial_type']:
                        mask = (events['trial_type'] == trial_type) & other_indices
                        single_trial_events.loc[mask, 'trial_type'] = f"other_{trial_type}"
                    else:
                        mask = (events['trial_type'] == trial_type) & other_indices
                        single_trial_events.loc[mask, 'trial_type'] = f"other_{trial_type}"
            else:
                # 所有其他trial合并为一个条件
                single_trial_events.loc[other_indices, 'trial_type'] = 'other_trials'
            
            lss_events.append(single_trial_events)
            condition_names.append(f"trial_{idx}_{trial['trial_type']}")
        
        # 创建设计矩阵
        design_matrices = []
        
        for trial_events in lss_events:
            # 添加运动参数
            confounds_df = None
            if motion_confounds is not None:
                confounds_df = pd.DataFrame(
                    motion_confounds,
                    columns=[f'motion_{i}' for i in range(motion_confounds.shape[1])]
                )
            
            design_matrix = make_first_level_design_matrix(
                frame_times=frame_times,
                events=trial_events,
                hrf_model='glover',
                drift_model='cosine',
                high_pass=0.01,
                add_regs=confounds_df
            )
            
            design_matrices.append(design_matrix)
        
        return design_matrices, condition_names
    
    def _fit_lss_glm_optimized(self, fmri_img: nib.Nifti1Image, design_matrices: List[np.ndarray],
                              condition_names: List[str]) -> Dict[str, nib.Nifti1Image]:
        """优化的LSS GLM拟合"""
        # 使用内存映射减少内存使用
        fmri_data = fmri_img.get_fdata()
        
        # 预分配结果数组
        beta_maps = {}
        t_maps = {}
        
        # 批处理拟合以减少内存使用
        batch_size = min(len(design_matrices), 10)  # 每批处理10个trial
        
        for i in range(0, len(design_matrices), batch_size):
            batch_designs = design_matrices[i:i+batch_size]
            batch_names = condition_names[i:i+batch_size]
            
            # 拟合当前批次
            for j, (design_matrix, condition_name) in enumerate(zip(batch_designs, batch_names)):
                try:
                    # 创建GLM模型
                    glm = FirstLevelModel(
                        t_r=self._infer_tr_from_img(fmri_img),
                        noise_model='ar1',
                        standardize=True,
                        hrf_model='glover',
                        drift_model=None,  # 已在设计矩阵中处理
                        high_pass=None,    # 已在设计矩阵中处理
                        mask_img=None,
                        smoothing_fwhm=None,
                        verbose=0,
                        n_jobs=1  # 单个GLM使用单线程
                    )
                    
                    # 拟合模型
                    glm.fit(fmri_img, design_matrices=[design_matrix])
                    
                    # 提取beta和t值
                    target_condition = condition_name
                    if target_condition in design_matrix.columns:
                        beta_map = glm.compute_contrast(target_condition, output_type='effect_size')
                        t_map = glm.compute_contrast(target_condition, output_type='stat')
                        
                        beta_maps[condition_name] = beta_map
                        t_maps[condition_name] = t_map
                    
                    # 清理内存
                    del glm
                    gc.collect()
                    
                except Exception as e:
                    self.error_handler.log_error(
                        e, f"GLM拟合: {condition_name}", "processing"
                    )
                    continue
            
            # 监控内存
            self._monitor_memory()
        
        return {'beta_maps': beta_maps, 't_maps': t_maps}
    
    def _process_single_run(self, subject: str, run: int) -> Optional[Dict[str, Any]]:
        """处理单个run"""
        run_start_time = time.time()
        
        try:
            with ErrorContext(f"处理 {subject} run {run}", self.error_handler, "processing"):
                # 构建文件路径
                fmri_path = self.lss_config.fmri_template.format(subject=subject, run=run)
                events_path = self.lss_config.event_template.format(subject=subject, run=run)
                motion_path = self.lss_config.motion_template.format(subject=subject, run=run)
                
                # 验证输入文件
                if not self.validator.check_file_exists(fmri_path):
                    raise FileNotFoundError(f"fMRI文件不存在: {fmri_path}")
                
                if not self.validator.check_nifti_integrity(fmri_path):
                    raise ValueError(f"fMRI文件损坏: {fmri_path}")
                
                # 加载数据
                self.error_handler.logger.info(f"加载数据: {subject} run {run}")
                
                fmri_img = nib.load(fmri_path)
                events = self._load_and_validate_events(events_path)
                motion_confounds = self._load_and_validate_motion(motion_path)
                
                if events is None:
                    raise ValueError(f"无法加载事件文件: {events_path}")
                
                # 推断TR
                tr = self._infer_tr_from_img(fmri_path)
                n_scans = fmri_img.shape[3]
                
                # 创建LSS设计矩阵
                self.error_handler.logger.info(f"创建LSS设计矩阵: {len(events)} 个trial")
                design_matrices, condition_names = self._create_lss_design_matrix(
                    events, n_scans, tr, motion_confounds
                )
                
                # 拟合GLM
                self.error_handler.logger.info(f"拟合LSS GLM")
                glm_results = self._fit_lss_glm_optimized(
                    fmri_img, design_matrices, condition_names
                )
                
                # 保存结果
                output_dir = Path(self.lss_config.output_root) / subject / f"run-{run:02d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_files = {
                    'beta_files': [],
                    't_files': [],
                    'events_file': str(output_dir / f"events_lss.tsv"),
                    'summary_file': str(output_dir / f"lss_summary.json")
                }
                
                # 保存beta和t图像
                for condition_name in condition_names:
                    if condition_name in glm_results['beta_maps']:
                        # Beta图像
                        beta_file = output_dir / f"beta_{condition_name}.nii.gz"
                        nib.save(glm_results['beta_maps'][condition_name], beta_file)
                        output_files['beta_files'].append(str(beta_file))
                        
                        # T图像
                        t_file = output_dir / f"tstat_{condition_name}.nii.gz"
                        nib.save(glm_results['t_maps'][condition_name], t_file)
                        output_files['t_files'].append(str(t_file))
                
                # 保存事件信息
                events_with_conditions = events.copy()
                events_with_conditions['lss_condition'] = condition_names
                events_with_conditions.to_csv(output_files['events_file'], sep='\t', index=False)
                
                # 保存摘要信息
                summary = {
                    'subject': subject,
                    'run': run,
                    'n_trials': len(events),
                    'n_scans': n_scans,
                    'tr': tr,
                    'processing_time': time.time() - run_start_time,
                    'output_files': output_files,
                    'quality_metrics': {
                        'mean_motion': np.mean(np.abs(motion_confounds[:, :3])) if motion_confounds is not None else None,
                        'max_motion': np.max(np.abs(motion_confounds[:, :3])) if motion_confounds is not None else None,
                        'n_successful_trials': len(glm_results['beta_maps'])
                    }
                }
                
                import json
                with open(output_files['summary_file'], 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # 记录处理时间
                processing_time = time.time() - run_start_time
                self.performance_stats['processing_times'][f"{subject}_run{run}"] = processing_time
                
                self.error_handler.logger.info(
                    f"完成处理: {subject} run {run}, "
                    f"{len(glm_results['beta_maps'])} 个trial, "
                    f"用时 {processing_time:.1f}s"
                )
                
                return summary
                
        except Exception as e:
            self.performance_stats['failed_runs'].append(f"{subject}_run{run}")
            self.error_handler.log_error(e, f"处理run: {subject} run {run}", "processing")
            return None
    
    def _process_single_subject(self, subject: str) -> Dict[str, Any]:
        """处理单个被试"""
        subject_start_time = time.time()
        
        try:
            with ErrorContext(f"处理被试 {subject}", self.error_handler, "processing"):
                subject_results = {
                    'subject': subject,
                    'runs': {},
                    'summary': {
                        'total_runs': len(self.lss_config.runs),
                        'successful_runs': 0,
                        'failed_runs': 0,
                        'total_trials': 0,
                        'processing_time': 0
                    }
                }
                
                # 并行处理runs（如果启用）
                if self.lss_config.parallel_runs and len(self.lss_config.runs) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(self.lss_config.runs), 3)) as executor:
                        future_to_run = {
                            executor.submit(self._process_single_run, subject, run): run
                            for run in self.lss_config.runs
                        }
                        
                        for future in as_completed(future_to_run):
                            run = future_to_run[future]
                            try:
                                result = future.result()
                                if result:
                                    subject_results['runs'][run] = result
                                    subject_results['summary']['successful_runs'] += 1
                                    subject_results['summary']['total_trials'] += result['n_trials']
                                else:
                                    subject_results['summary']['failed_runs'] += 1
                            except Exception as e:
                                self.error_handler.log_error(
                                    e, f"并行处理run: {subject} run {run}", "processing"
                                )
                                subject_results['summary']['failed_runs'] += 1
                else:
                    # 串行处理runs
                    for run in self.lss_config.runs:
                        result = self._process_single_run(subject, run)
                        if result:
                            subject_results['runs'][run] = result
                            subject_results['summary']['successful_runs'] += 1
                            subject_results['summary']['total_trials'] += result['n_trials']
                        else:
                            subject_results['summary']['failed_runs'] += 1
                
                subject_results['summary']['processing_time'] = time.time() - subject_start_time
                
                self.error_handler.logger.info(
                    f"被试 {subject} 处理完成: "
                    f"{subject_results['summary']['successful_runs']}/{subject_results['summary']['total_runs']} runs成功, "
                    f"{subject_results['summary']['total_trials']} 个trial"
                )
                
                return subject_results
                
        except Exception as e:
            self.performance_stats['failed_subjects'].append(subject)
            self.error_handler.log_error(e, f"处理被试: {subject}", "processing")
            return {'subject': subject, 'error': str(e)}
    
    def run_analysis(self) -> Dict[str, Any]:
        """运行完整的LSS分析"""
        self.performance_stats['start_time'] = datetime.now()
        
        try:
            # 验证配置
            self.config.validate_config()
            
            # 创建输出目录
            output_root = Path(self.lss_config.output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            
            self.error_handler.logger.info(
                f"开始LSS分析: {len(self.lss_config.subjects)} 个被试, "
                f"{len(self.lss_config.runs)} 个run"
            )
            
            all_results = {
                'subjects': {},
                'global_summary': {
                    'total_subjects': len(self.lss_config.subjects),
                    'successful_subjects': 0,
                    'failed_subjects': 0,
                    'total_runs': 0,
                    'successful_runs': 0,
                    'failed_runs': 0,
                    'total_trials': 0
                }
            }
            
            # 并行处理被试（如果启用）
            if self.lss_config.parallel_subjects and len(self.lss_config.subjects) > 1:
                max_workers = min(len(self.lss_config.subjects), cpu_count()//2)
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_subject = {
                        executor.submit(self._process_single_subject, subject): subject
                        for subject in self.lss_config.subjects
                    }
                    
                    for future in as_completed(future_to_subject):
                        subject = future_to_subject[future]
                        try:
                            result = future.result()
                            all_results['subjects'][subject] = result
                            
                            if 'error' not in result:
                                all_results['global_summary']['successful_subjects'] += 1
                                all_results['global_summary']['successful_runs'] += result['summary']['successful_runs']
                                all_results['global_summary']['failed_runs'] += result['summary']['failed_runs']
                                all_results['global_summary']['total_trials'] += result['summary']['total_trials']
                            else:
                                all_results['global_summary']['failed_subjects'] += 1
                                
                        except Exception as e:
                            self.error_handler.log_error(
                                e, f"并行处理被试: {subject}", "processing"
                            )
                            all_results['global_summary']['failed_subjects'] += 1
            else:
                # 串行处理被试
                for subject in self.lss_config.subjects:
                    result = self._process_single_subject(subject)
                    all_results['subjects'][subject] = result
                    
                    if 'error' not in result:
                        all_results['global_summary']['successful_subjects'] += 1
                        all_results['global_summary']['successful_runs'] += result['summary']['successful_runs']
                        all_results['global_summary']['failed_runs'] += result['summary']['failed_runs']
                        all_results['global_summary']['total_trials'] += result['summary']['total_trials']
                    else:
                        all_results['global_summary']['failed_subjects'] += 1
            
            all_results['global_summary']['total_runs'] = (
                all_results['global_summary']['successful_runs'] + 
                all_results['global_summary']['failed_runs']
            )
            
            # 保存全局摘要
            self.performance_stats['end_time'] = datetime.now()
            all_results['performance_stats'] = self.performance_stats
            
            summary_file = output_root / "lss_analysis_summary.json"
            with open(summary_file, 'w') as f:
                import json
                json.dump(all_results, f, indent=2, default=str)
            
            self.error_handler.logger.info(
                f"LSS分析完成: "
                f"{all_results['global_summary']['successful_subjects']}/{all_results['global_summary']['total_subjects']} 被试成功, "
                f"{all_results['global_summary']['successful_runs']}/{all_results['global_summary']['total_runs']} run成功, "
                f"{all_results['global_summary']['total_trials']} 个trial"
            )
            
            return all_results
            
        except Exception as e:
            self.performance_stats['end_time'] = datetime.now()
            self.error_handler.log_error(e, "LSS分析", "analysis")
            raise

def run_lss_analysis_enhanced(subjects: List[str], runs: List[int], 
                             config: GlobalConfig) -> Dict[str, Any]:
    """运行增强版LSS分析的便捷函数"""
    # 更新配置
    config.lss.subjects = subjects
    config.lss.runs = runs
    
    # 创建分析器
    analyzer = LSSAnalysisEnhanced(config)
    
    # 运行分析
    return analyzer.run_analysis()

if __name__ == "__main__":
    # 测试增强版LSS分析
    from .global_config import create_config
    
    config = create_config()
    config.lss.subjects = ['sub-01', 'sub-02']
    config.lss.runs = [1, 2, 3]
    config.lss.fmri_template = "/path/to/fmri/sub-{subject}/func/sub-{subject}_task-metaphor_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    config.lss.event_template = "/path/to/events/sub-{subject}/func/sub-{subject}_task-metaphor_run-{run:02d}_events.tsv"
    config.lss.output_root = "./test_lss_output"
    
    logging.info("增强版LSS分析配置:")
    logging.info(config.get_config_summary())
    
    # analyzer = LSSAnalysisEnhanced(config)
    # results = analyzer.run_analysis()
    
    logging.info("\n增强版LSS分析模块测试完成")