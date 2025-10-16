#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版LSS分析模块
基于最新的fmri_lss_enhance.py进行优化
集成高效算法、并行处理和内存优化
"""

import os, math, gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_brain_mask
import warnings
import logging
from datetime import datetime

# 设置环境变量优化
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedLSSAnalyzer:
    """优化版LSS分析器"""
    
    def __init__(self, config):
        self.config = config
        self.lss_config = config.lss
        
        # 性能统计
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'memory_usage': [],
            'processing_times': {},
            'failed_subjects': [],
            'failed_runs': []
        }
        
        # 缓存
        self.mask_cache = {}
        
    def log(self, message: str):
        """日志记录"""
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}", flush=True)
        logging.info(message)
    
    def infer_tr_from_img(self, img) -> float:
        """快速TR推断"""
        try:
            return float(img.header.get_zooms()[-1])
        except Exception:
            return self.lss_config.tr_fallback
    
    def load_events_fast(self, path: Path) -> Optional[pd.DataFrame]:
        """快速加载事件文件"""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, sep="\t")
            # 快速列名检测
            col_map = {}
            for col in df.columns:
                lower_col = col.lower()
                if "onset" in lower_col: col_map["onset"] = col
                elif "duration" in lower_col: col_map["duration"] = col  
                elif "trial" in lower_col or "condition" in lower_col: col_map["trial_type"] = col
            
            if len(col_map) != 3:
                return None
                
            ev = df[list(col_map.values())].copy()
            ev.columns = ["onset", "duration", "trial_type"]
            
            # 快速数据清洗
            ev = ev.dropna(subset=["onset", "duration", "trial_type"])
            ev["onset"] = pd.to_numeric(ev["onset"], errors="coerce")
            ev["duration"] = pd.to_numeric(ev["duration"], errors="coerce")
            ev = ev.dropna(subset=["onset", "duration"])
            
            return ev.sort_values("onset").reset_index(drop=True) if not ev.empty else None
            
        except Exception as e:
            self.log(f"加载事件文件失败 {path}: {e}")
            return None
    
    def load_motion_confounds_fast(self, sub: str, run: int, n_volumes: int) -> Optional[pd.DataFrame]:
        """快速加载头动参数"""
        path = Path(self.lss_config.motion_template.format(sub=sub, run=run))
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, delimiter='\t')
            # 快速列选择
            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            available_cols = [col for col in motion_cols if col in df.columns]
            
            if len(available_cols) >= 3:  # 至少需要3个运动参数
                confounds = df[available_cols].fillna(0).iloc[:n_volumes]
                return confounds if len(confounds) == n_volumes else None
        except Exception:
            pass
        return None
    
    def create_lss_events(self, trial_idx: int, base_events: pd.DataFrame) -> pd.DataFrame:
        """
        为LSS模型创建事件DataFrame，此版本针对多条件进行了优化。
        
        该函数为指定的单个trial创建事件列表，采用的策略如下：
        1. **目标Trial**: 您感兴趣的特定trial被单独标记为 'target'。
        2. **同条件下的其他Trial**: 与目标trial属于相同实验条件的所有其他trial
           被合并为一个回归器，标记为 'other_same_condition'。
        3. **不同条件的Trial**: 所有属于不同实验条件的trial被合并为另一个
           回归器，标记为 'other_different_condition'。
        """
        target_trial = base_events.iloc[trial_idx]
        target_condition = target_trial['trial_type']
        
        lss_events = []
        
        for idx, trial in base_events.iterrows():
            if idx == trial_idx:
                # 目标trial
                lss_events.append({
                    'onset': trial['onset'],
                    'duration': trial['duration'],
                    'trial_type': 'target'
                })
            elif trial['trial_type'] == target_condition:
                # 同条件的其他trial
                lss_events.append({
                    'onset': trial['onset'],
                    'duration': trial['duration'],
                    'trial_type': 'other_same_condition'
                })
            else:
                # 不同条件的trial
                lss_events.append({
                    'onset': trial['onset'],
                    'duration': trial['duration'],
                    'trial_type': 'other_different_condition'
                })
        
        return pd.DataFrame(lss_events)
    
    def precompute_shared_components(self, fmri_img, events, confounds, base_kwargs):
        """预计算共享组件以提高效率"""
        if self.lss_config.use_precomputed_mask and 'mask' not in base_kwargs:
            mask_key = str(fmri_img.shape)
            if mask_key not in self.mask_cache:
                self.mask_cache[mask_key] = compute_brain_mask(fmri_img)
            base_kwargs['mask_img'] = self.mask_cache[mask_key]
        else:
            base_kwargs['mask_img'] = None
        
        return base_kwargs
    
    def process_single_trial(self, args):
        """处理单个trial的LSS分析"""
        trial_idx, fmri_img, base_events, tr, confounds, base_kwargs, target_condition = args
        
        try:
            # 创建LSS事件
            lss_events = self.create_lss_events(trial_idx, base_events)
            
            # 创建GLM模型
            glm = FirstLevelModel(
                t_r=tr,
                standardize=self.lss_config.standardize,
                **base_kwargs
            )
            
            # 拟合模型
            glm.fit(fmri_img, lss_events, confounds=confounds)
            
            # 计算对比
            contrast_map = glm.compute_contrast('target', output_type='effect_size')
            
            return {
                'trial_idx': trial_idx,
                'condition': target_condition,
                'beta_map': contrast_map,
                'success': True
            }
            
        except Exception as e:
            return {
                'trial_idx': trial_idx,
                'condition': target_condition,
                'beta_map': None,
                'success': False,
                'error': str(e)
            }
    
    def process_one_run_optimized(self, sub: str, run: int):
        """优化版单个run处理"""
        self.log(f"处理 {sub} run-{run}")
        
        # 构建文件路径
        fmri_path = Path(self.lss_config.fmri_template.format(sub=sub, run=run))
        events_path = Path(self.lss_config.event_template.format(sub=sub, run=run))
        
        if not fmri_path.exists():
            self.log(f"fMRI文件不存在: {fmri_path}")
            return None
            
        if not events_path.exists():
            self.log(f"事件文件不存在: {events_path}")
            return None
        
        try:
            # 加载数据
            fmri_img = image.load_img(str(fmri_path))
            events = self.load_events_fast(events_path)
            
            if events is None or events.empty:
                self.log(f"事件数据无效: {events_path}")
                return None
            
            # 推断TR
            tr = self.infer_tr_from_img(fmri_img)
            n_volumes = fmri_img.shape[-1]
            
            # 加载头动参数
            confounds = self.load_motion_confounds_fast(sub, run, n_volumes)
            
            # 基础GLM参数
            base_kwargs = {
                'hrf_model': 'spm',
                'drift_model': 'cosine',
                'high_pass': 0.01,
                'smoothing_fwhm': None,
                'n_jobs': 1,  # 单个GLM使用单线程
                'verbose': 0,
                'minimize_memory': True
            }
            
            # 预计算共享组件
            base_kwargs = self.precompute_shared_components(fmri_img, events, confounds, base_kwargs)
            
            # 准备并行处理参数
            trial_args = []
            for trial_idx in range(len(events)):
                target_condition = events.iloc[trial_idx]['trial_type']
                trial_args.append((
                    trial_idx, fmri_img, events, tr, confounds, 
                    base_kwargs, target_condition
                ))
            
            # 并行处理trials
            results = []
            if self.lss_config.n_jobs_glm > 1:
                with ProcessPoolExecutor(max_workers=self.lss_config.n_jobs_glm) as executor:
                    future_to_trial = {executor.submit(self.process_single_trial, args): args[0] 
                                     for args in trial_args}
                    
                    for future in as_completed(future_to_trial):
                        result = future.result()
                        results.append(result)
            else:
                # 串行处理
                for args in trial_args:
                    result = self.process_single_trial(args)
                    results.append(result)
            
            # 保存结果
            output_dir = Path(self.lss_config.output_root) / sub / f"run{run}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            if failed_results:
                self.log(f"失败的trials: {len(failed_results)}/{len(results)}")
            
            # 保存beta maps
            for result in successful_results:
                trial_idx = result['trial_idx']
                condition = result['condition']
                beta_map = result['beta_map']
                
                output_file = output_dir / f"trial_{trial_idx:03d}_{condition}_beta.nii.gz"
                beta_map.to_filename(str(output_file))
            
            # 保存试验信息
            trial_info = []
            for i, (_, trial) in enumerate(events.iterrows()):
                trial_info.append({
                    'trial_idx': i,
                    'condition': trial['trial_type'],
                    'onset': trial['onset'],
                    'duration': trial['duration'],
                    'success': any(r['trial_idx'] == i and r['success'] for r in results)
                })
            
            trial_info_df = pd.DataFrame(trial_info)
            trial_info_file = output_dir / "trial_info.csv"
            trial_info_df.to_csv(trial_info_file, index=False)
            
            self.log(f"完成 {sub} run-{run}: {len(successful_results)}/{len(results)} trials成功")
            
            # 清理内存
            del fmri_img, results
            gc.collect()
            
            return {
                'subject': sub,
                'run': run,
                'total_trials': len(results),
                'successful_trials': len(successful_results),
                'failed_trials': len(failed_results),
                'output_dir': str(output_dir)
            }
            
        except Exception as e:
            self.log(f"处理 {sub} run-{run} 时出错: {e}")
            return None
    
    def process_subject_run(self, args):
        """处理单个被试的单个run（用于并行处理）"""
        sub, run = args
        return self.process_one_run_optimized(sub, run)
    
    def run_analysis(self) -> Dict[str, Any]:
        """运行完整的LSS分析"""
        self.performance_stats['start_time'] = datetime.now()
        self.log("开始优化版LSS分析")
        
        # 创建输出目录
        Path(self.lss_config.output_root).mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        if self.lss_config.parallel_subjects and self.lss_config.n_jobs_parallel > 1:
            # 并行处理被试和runs
            self.log(f"使用并行模式: {self.lss_config.n_jobs_parallel} 进程")
            
            tasks = [(sub, run) for sub in self.lss_config.subjects for run in self.lss_config.runs]
            
            with ProcessPoolExecutor(max_workers=self.lss_config.n_jobs_parallel) as executor:
                future_to_task = {executor.submit(self.process_subject_run, task): task for task in tasks}
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                        else:
                            self.performance_stats['failed_runs'].append(task)
                    except Exception as e:
                        self.log(f"任务 {task} 失败: {e}")
                        self.performance_stats['failed_runs'].append(task)
        else:
            # 串行处理
            self.log("使用串行模式")
            for sub in self.lss_config.subjects:
                self.log(f"\n{'='*50}")
                self.log(f"处理被试: {sub}")
                self.log(f"{'='*50}")
                for run in self.lss_config.runs:
                    try:
                        result = self.process_one_run_optimized(sub, run)
                        if result:
                            all_results.append(result)
                        else:
                            self.performance_stats['failed_runs'].append((sub, run))
                    except Exception as e:
                        self.log(f"[ERROR] {sub}|run-{run} 失败: {e}")
                        self.performance_stats['failed_runs'].append((sub, run))
        
        self.performance_stats['end_time'] = datetime.now()
        
        # 生成总结报告
        total_tasks = len(self.lss_config.subjects) * len(self.lss_config.runs)
        successful_tasks = len(all_results)
        failed_tasks = len(self.performance_stats['failed_runs'])
        
        self.log(f"\n{'='*60}")
        self.log(f"LSS分析完成")
        self.log(f"总任务数: {total_tasks}")
        self.log(f"成功: {successful_tasks}")
        self.log(f"失败: {failed_tasks}")
        self.log(f"成功率: {successful_tasks/total_tasks*100:.1f}%")
        self.log(f"总耗时: {self.performance_stats['end_time'] - self.performance_stats['start_time']}")
        self.log(f"{'='*60}")
        
        return {
            'results': all_results,
            'performance_stats': self.performance_stats,
            'summary': {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': successful_tasks/total_tasks if total_tasks > 0 else 0,
                'duration': str(self.performance_stats['end_time'] - self.performance_stats['start_time'])
            }
        }

def run_optimized_lss_analysis(config) -> Dict[str, Any]:
    """运行优化版LSS分析的便捷函数"""
    analyzer = OptimizedLSSAnalyzer(config)
    return analyzer.run_analysis()

if __name__ == "__main__":
    from .global_config import GlobalConfig
    
    # 创建配置
    config = GlobalConfig.create_default_config()
    
    # 运行分析
    results = run_optimized_lss_analysis(config)
    print("优化版LSS分析完成")
    print(f"成功率: {results['summary']['success_rate']*100:.1f}%")