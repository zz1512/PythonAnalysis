#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量控制和性能监控模块
提供数据质量评估、性能监控和分析质量控制功能
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import image, plotting
from nilearn.image import load_img, math_img
from nilearn.masking import compute_brain_mask
from nilearn.signal import clean
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import psutil
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .error_handling import ErrorHandler, ErrorContext

warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'disk_io': [],
            'processing_times': {},
            'error_counts': {},
            'quality_metrics': {}
        }
        
        self.monitoring_active = False
        self.monitoring_interval = 5  # 秒
    
    def start_monitoring(self):
        """开始性能监控"""
        self.start_time = datetime.now()
        self.monitoring_active = True
        
        # 记录初始状态
        self._record_system_metrics()
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        
        # 记录最终状态
        self._record_system_metrics()
        
        # 生成监控报告
        return self.generate_performance_report()
    
    def _record_system_metrics(self):
        """记录系统指标"""
        timestamp = datetime.now()
        
        # 内存使用
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'].append({
            'timestamp': timestamp.isoformat(),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        })
        
        # CPU使用
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics['cpu_usage'].append({
            'timestamp': timestamp.isoformat(),
            'percent': cpu_percent,
            'cores': psutil.cpu_count()
        })
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.metrics['disk_io'].append({
                'timestamp': timestamp.isoformat(),
                'read_mb': disk_io.read_bytes / (1024**2),
                'write_mb': disk_io.write_bytes / (1024**2)
            })
    
    def record_processing_time(self, task_name: str, duration: float):
        """记录处理时间"""
        if task_name not in self.metrics['processing_times']:
            self.metrics['processing_times'][task_name] = []
        
        self.metrics['processing_times'][task_name].append({
            'timestamp': datetime.now().isoformat(),
            'duration': duration
        })
    
    def record_error(self, error_type: str):
        """记录错误"""
        if error_type not in self.metrics['error_counts']:
            self.metrics['error_counts'][error_type] = 0
        
        self.metrics['error_counts'][error_type] += 1
    
    def record_quality_metric(self, metric_name: str, value: float, context: str = ""):
        """记录质量指标"""
        if metric_name not in self.metrics['quality_metrics']:
            self.metrics['quality_metrics'][metric_name] = []
        
        self.metrics['quality_metrics'][metric_name].append({
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'context': context
        })
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.start_time:
            return {}
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'total_duration_formatted': str(timedelta(seconds=int(total_duration)))
            },
            'memory': self._analyze_memory_usage(),
            'cpu': self._analyze_cpu_usage(),
            'processing_times': self._analyze_processing_times(),
            'errors': self.metrics['error_counts'],
            'quality_metrics': self._analyze_quality_metrics()
        }
        
        # 保存报告
        report_file = self.log_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用"""
        if not self.metrics['memory_usage']:
            return {}
        
        memory_data = [m['percent'] for m in self.metrics['memory_usage']]
        
        return {
            'peak_usage_percent': max(memory_data),
            'average_usage_percent': np.mean(memory_data),
            'min_usage_percent': min(memory_data),
            'peak_used_gb': max([m['used_gb'] for m in self.metrics['memory_usage']]),
            'memory_warnings': len([m for m in memory_data if m > 85])
        }
    
    def _analyze_cpu_usage(self) -> Dict[str, Any]:
        """分析CPU使用"""
        if not self.metrics['cpu_usage']:
            return {}
        
        cpu_data = [c['percent'] for c in self.metrics['cpu_usage']]
        
        return {
            'peak_usage_percent': max(cpu_data),
            'average_usage_percent': np.mean(cpu_data),
            'min_usage_percent': min(cpu_data),
            'high_usage_periods': len([c for c in cpu_data if c > 80])
        }
    
    def _analyze_processing_times(self) -> Dict[str, Any]:
        """分析处理时间"""
        analysis = {}
        
        for task_name, times in self.metrics['processing_times'].items():
            durations = [t['duration'] for t in times]
            
            analysis[task_name] = {
                'count': len(durations),
                'total_time': sum(durations),
                'average_time': np.mean(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'std_time': np.std(durations)
            }
        
        return analysis
    
    def _analyze_quality_metrics(self) -> Dict[str, Any]:
        """分析质量指标"""
        analysis = {}
        
        for metric_name, values in self.metrics['quality_metrics'].items():
            metric_values = [v['value'] for v in values]
            
            analysis[metric_name] = {
                'count': len(metric_values),
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'min': min(metric_values),
                'max': max(metric_values),
                'median': np.median(metric_values)
            }
        
        return analysis

class DataQualityAssessor:
    """数据质量评估器"""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.quality_thresholds = {
            'motion_threshold_mm': 3.0,
            'signal_to_noise_threshold': 100,
            'temporal_snr_threshold': 50,
            'mean_fd_threshold': 0.5,
            'outlier_threshold_percent': 5,
            'brain_coverage_threshold': 0.8,
            'classification_accuracy_threshold': 0.6
        }
    
    def assess_fmri_quality(self, fmri_path: str, motion_path: Optional[str] = None) -> Dict[str, Any]:
        """评估fMRI数据质量"""
        try:
            with ErrorContext(f"评估fMRI质量: {fmri_path}", self.error_handler, "quality_check"):
                # 加载数据
                fmri_img = nib.load(fmri_path)
                fmri_data = fmri_img.get_fdata()
                
                quality_metrics = {
                    'file_path': fmri_path,
                    'dimensions': fmri_data.shape,
                    'voxel_size': fmri_img.header.get_zooms()[:3],
                    'tr': float(fmri_img.header.get_zooms()[3]) if len(fmri_img.header.get_zooms()) > 3 else None
                }
                
                # 基本统计
                quality_metrics.update(self._compute_basic_stats(fmri_data))
                
                # 信噪比
                quality_metrics.update(self._compute_snr_metrics(fmri_data))
                
                # 时间序列质量
                quality_metrics.update(self._compute_temporal_metrics(fmri_data))
                
                # 脑覆盖度
                quality_metrics.update(self._compute_brain_coverage(fmri_img))
                
                # 运动参数质量（如果提供）
                if motion_path and os.path.exists(motion_path):
                    quality_metrics.update(self._assess_motion_quality(motion_path))
                
                # 异常值检测
                quality_metrics.update(self._detect_outliers(fmri_data))
                
                # 质量评级
                quality_metrics['quality_rating'] = self._compute_quality_rating(quality_metrics)
                
                return quality_metrics
                
        except Exception as e:
            self.error_handler.log_error(e, f"fMRI质量评估: {fmri_path}", "quality_check")
            return {'error': str(e), 'file_path': fmri_path}
    
    def _compute_basic_stats(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """计算基本统计量"""
        # 排除零值体素
        brain_mask = fmri_data.mean(axis=3) > 0
        brain_data = fmri_data[brain_mask]
        
        return {
            'mean_intensity': float(np.mean(brain_data)),
            'std_intensity': float(np.std(brain_data)),
            'min_intensity': float(np.min(brain_data)),
            'max_intensity': float(np.max(brain_data)),
            'brain_voxels': int(np.sum(brain_mask)),
            'total_voxels': int(np.prod(fmri_data.shape[:3]))
        }
    
    def _compute_snr_metrics(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """计算信噪比指标"""
        # 时间平均
        mean_img = np.mean(fmri_data, axis=3)
        std_img = np.std(fmri_data, axis=3)
        
        # 避免除零
        std_img[std_img == 0] = np.finfo(float).eps
        
        # 信噪比
        snr_img = mean_img / std_img
        
        # 只考虑脑区域
        brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)
        brain_snr = snr_img[brain_mask]
        
        return {
            'mean_snr': float(np.mean(brain_snr)),
            'median_snr': float(np.median(brain_snr)),
            'std_snr': float(np.std(brain_snr)),
            'min_snr': float(np.min(brain_snr)),
            'max_snr': float(np.max(brain_snr))
        }
    
    def _compute_temporal_metrics(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """计算时间序列指标"""
        # 时间信噪比 (tSNR)
        mean_ts = np.mean(fmri_data, axis=3)
        std_ts = np.std(fmri_data, axis=3)
        
        # 避免除零
        std_ts[std_ts == 0] = np.finfo(float).eps
        tsnr = mean_ts / std_ts
        
        # 只考虑脑区域
        brain_mask = mean_ts > np.percentile(mean_ts[mean_ts > 0], 25)
        brain_tsnr = tsnr[brain_mask]
        
        # 时间序列稳定性
        n_timepoints = fmri_data.shape[3]
        temporal_stability = []
        
        for i in range(0, n_timepoints-10, 10):  # 每10个时间点计算一次
            chunk1 = fmri_data[:, :, :, i:i+5]
            chunk2 = fmri_data[:, :, :, i+5:i+10]
            
            corr = np.corrcoef(
                chunk1[brain_mask].mean(axis=0),
                chunk2[brain_mask].mean(axis=0)
            )[0, 1]
            
            if not np.isnan(corr):
                temporal_stability.append(corr)
        
        return {
            'mean_tsnr': float(np.mean(brain_tsnr)),
            'median_tsnr': float(np.median(brain_tsnr)),
            'temporal_stability': float(np.mean(temporal_stability)) if temporal_stability else 0.0,
            'temporal_stability_std': float(np.std(temporal_stability)) if temporal_stability else 0.0
        }
    
    def _compute_brain_coverage(self, fmri_img: nib.Nifti1Image) -> Dict[str, float]:
        """计算脑覆盖度"""
        try:
            # 计算脑掩模
            brain_mask = compute_brain_mask(fmri_img)
            brain_mask_data = brain_mask.get_fdata()
            
            # 数据掩模（非零体素）
            data_mask = fmri_img.get_fdata().mean(axis=3) > 0
            
            # 覆盖度计算
            brain_voxels = np.sum(brain_mask_data)
            covered_brain_voxels = np.sum(brain_mask_data & data_mask)
            
            coverage = covered_brain_voxels / brain_voxels if brain_voxels > 0 else 0
            
            return {
                'brain_coverage': float(coverage),
                'brain_voxels': int(brain_voxels),
                'covered_brain_voxels': int(covered_brain_voxels)
            }
            
        except Exception as e:
            self.error_handler.log_error(e, "脑覆盖度计算", "quality_check")
            return {
                'brain_coverage': 0.0,
                'brain_voxels': 0,
                'covered_brain_voxels': 0
            }
    
    def _assess_motion_quality(self, motion_path: str) -> Dict[str, float]:
        """评估运动参数质量"""
        try:
            # 加载运动参数
            motion_data = pd.read_csv(motion_path, sep='\t', header=None)
            if motion_data.shape[1] < 6:
                motion_data = pd.read_csv(motion_path, sep=' ', header=None)
            
            motion_array = motion_data.values
            
            # 平移参数 (前3列，单位：mm)
            translation = motion_array[:, :3]
            # 旋转参数 (后3列，单位：弧度转度)
            rotation = motion_array[:, 3:6] * 180 / np.pi
            
            # 计算帧间位移 (Framewise Displacement)
            trans_diff = np.diff(translation, axis=0)
            rot_diff = np.diff(rotation, axis=0)
            
            # 旋转转换为mm (假设头部半径50mm)
            rot_diff_mm = rot_diff * np.pi / 180 * 50
            
            fd = np.sum(np.abs(trans_diff), axis=1) + np.sum(np.abs(rot_diff_mm), axis=1)
            
            return {
                'max_translation': float(np.max(np.abs(translation))),
                'max_rotation': float(np.max(np.abs(rotation))),
                'mean_fd': float(np.mean(fd)),
                'max_fd': float(np.max(fd)),
                'high_motion_frames': int(np.sum(fd > self.quality_thresholds['mean_fd_threshold'])),
                'motion_outliers_percent': float(np.sum(fd > self.quality_thresholds['mean_fd_threshold']) / len(fd) * 100)
            }
            
        except Exception as e:
            self.error_handler.log_error(e, f"运动质量评估: {motion_path}", "quality_check")
            return {
                'max_translation': 999.0,
                'max_rotation': 999.0,
                'mean_fd': 999.0,
                'max_fd': 999.0,
                'high_motion_frames': 999,
                'motion_outliers_percent': 100.0
            }
    
    def _detect_outliers(self, fmri_data: np.ndarray) -> Dict[str, Any]:
        """检测异常值"""
        # 全脑平均信号
        brain_mask = fmri_data.mean(axis=3) > 0
        global_signal = np.mean(fmri_data[brain_mask], axis=0)
        
        # 检测异常时间点
        z_scores = np.abs(stats.zscore(global_signal))
        outlier_timepoints = np.where(z_scores > 3)[0]
        
        # 检测异常体素
        voxel_means = np.mean(fmri_data, axis=3)
        voxel_z_scores = np.abs(stats.zscore(voxel_means[brain_mask]))
        outlier_voxels = np.sum(voxel_z_scores > 3)
        
        return {
            'outlier_timepoints': len(outlier_timepoints),
            'outlier_timepoints_percent': float(len(outlier_timepoints) / fmri_data.shape[3] * 100),
            'outlier_voxels': int(outlier_voxels),
            'outlier_voxels_percent': float(outlier_voxels / np.sum(brain_mask) * 100),
            'global_signal_std': float(np.std(global_signal))
        }
    
    def _compute_quality_rating(self, metrics: Dict[str, Any]) -> str:
        """计算质量评级"""
        score = 0
        max_score = 0
        
        # SNR评分
        if 'mean_snr' in metrics:
            max_score += 20
            if metrics['mean_snr'] > self.quality_thresholds['signal_to_noise_threshold']:
                score += 20
            elif metrics['mean_snr'] > self.quality_thresholds['signal_to_noise_threshold'] * 0.7:
                score += 15
            elif metrics['mean_snr'] > self.quality_thresholds['signal_to_noise_threshold'] * 0.5:
                score += 10
        
        # tSNR评分
        if 'mean_tsnr' in metrics:
            max_score += 20
            if metrics['mean_tsnr'] > self.quality_thresholds['temporal_snr_threshold']:
                score += 20
            elif metrics['mean_tsnr'] > self.quality_thresholds['temporal_snr_threshold'] * 0.7:
                score += 15
            elif metrics['mean_tsnr'] > self.quality_thresholds['temporal_snr_threshold'] * 0.5:
                score += 10
        
        # 运动评分
        if 'mean_fd' in metrics:
            max_score += 20
            if metrics['mean_fd'] < self.quality_thresholds['mean_fd_threshold']:
                score += 20
            elif metrics['mean_fd'] < self.quality_thresholds['mean_fd_threshold'] * 2:
                score += 15
            elif metrics['mean_fd'] < self.quality_thresholds['mean_fd_threshold'] * 3:
                score += 10
        
        # 脑覆盖度评分
        if 'brain_coverage' in metrics:
            max_score += 20
            if metrics['brain_coverage'] > self.quality_thresholds['brain_coverage_threshold']:
                score += 20
            elif metrics['brain_coverage'] > self.quality_thresholds['brain_coverage_threshold'] * 0.8:
                score += 15
            elif metrics['brain_coverage'] > self.quality_thresholds['brain_coverage_threshold'] * 0.6:
                score += 10
        
        # 异常值评分
        if 'outlier_timepoints_percent' in metrics:
            max_score += 20
            if metrics['outlier_timepoints_percent'] < self.quality_thresholds['outlier_threshold_percent']:
                score += 20
            elif metrics['outlier_timepoints_percent'] < self.quality_thresholds['outlier_threshold_percent'] * 2:
                score += 15
            elif metrics['outlier_timepoints_percent'] < self.quality_thresholds['outlier_threshold_percent'] * 3:
                score += 10
        
        # 计算百分比
        if max_score > 0:
            percentage = (score / max_score) * 100
            
            if percentage >= 80:
                return "Excellent"
            elif percentage >= 60:
                return "Good"
            elif percentage >= 40:
                return "Fair"
            else:
                return "Poor"
        
        return "Unknown"
    
    def assess_mvpa_results_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """评估MVPA结果质量"""
        quality_metrics = {
            'assessment_time': datetime.now().isoformat()
        }
        
        # 分类准确率评估
        if 'accuracy' in results:
            accuracy = results['accuracy']
            quality_metrics['accuracy_rating'] = self._rate_accuracy(accuracy)
        
        # 置换检验评估
        if 'permutation_pvalue' in results:
            p_value = results['permutation_pvalue']
            quality_metrics['significance_rating'] = self._rate_significance(p_value)
        
        # 特征重要性评估
        if 'feature_importance' in results:
            importance = results['feature_importance']
            quality_metrics['feature_consistency'] = self._assess_feature_consistency(importance)
        
        # 交叉验证稳定性
        if 'cv_scores' in results:
            cv_scores = results['cv_scores']
            quality_metrics['cv_stability'] = self._assess_cv_stability(cv_scores)
        
        # 整体质量评级
        quality_metrics['overall_quality'] = self._compute_mvpa_quality_rating(quality_metrics)
        
        return quality_metrics
    
    def _rate_accuracy(self, accuracy: float) -> str:
        """评估准确率"""
        if accuracy >= 0.8:
            return "Excellent"
        elif accuracy >= 0.7:
            return "Good"
        elif accuracy >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def _rate_significance(self, p_value: float) -> str:
        """评估显著性"""
        if p_value < 0.001:
            return "Highly Significant"
        elif p_value < 0.01:
            return "Significant"
        elif p_value < 0.05:
            return "Marginally Significant"
        else:
            return "Not Significant"
    
    def _assess_feature_consistency(self, importance: np.ndarray) -> Dict[str, float]:
        """评估特征一致性"""
        return {
            'mean_importance': float(np.mean(importance)),
            'std_importance': float(np.std(importance)),
            'max_importance': float(np.max(importance)),
            'sparsity': float(np.sum(importance > np.mean(importance)) / len(importance))
        }
    
    def _assess_cv_stability(self, cv_scores: List[float]) -> Dict[str, float]:
        """评估交叉验证稳定性"""
        return {
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'min_cv_score': float(np.min(cv_scores)),
            'max_cv_score': float(np.max(cv_scores)),
            'cv_stability_ratio': float(np.std(cv_scores) / np.mean(cv_scores)) if np.mean(cv_scores) > 0 else float('inf')
        }
    
    def _compute_mvpa_quality_rating(self, metrics: Dict[str, Any]) -> str:
        """计算MVPA质量评级"""
        score = 0
        max_score = 0
        
        # 准确率评分
        if 'accuracy_rating' in metrics:
            max_score += 40
            rating = metrics['accuracy_rating']
            if rating == "Excellent":
                score += 40
            elif rating == "Good":
                score += 30
            elif rating == "Fair":
                score += 20
        
        # 显著性评分
        if 'significance_rating' in metrics:
            max_score += 30
            rating = metrics['significance_rating']
            if rating == "Highly Significant":
                score += 30
            elif rating == "Significant":
                score += 25
            elif rating == "Marginally Significant":
                score += 15
        
        # 稳定性评分
        if 'cv_stability' in metrics and 'cv_stability_ratio' in metrics['cv_stability']:
            max_score += 30
            stability_ratio = metrics['cv_stability']['cv_stability_ratio']
            if stability_ratio < 0.1:
                score += 30
            elif stability_ratio < 0.2:
                score += 25
            elif stability_ratio < 0.3:
                score += 15
        
        # 计算百分比
        if max_score > 0:
            percentage = (score / max_score) * 100
            
            if percentage >= 80:
                return "Excellent"
            elif percentage >= 60:
                return "Good"
            elif percentage >= 40:
                return "Fair"
            else:
                return "Poor"
        
        return "Unknown"

class QualityReporter:
    """质量报告生成器"""
    
    def __init__(self, output_dir: str = "./quality_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_quality_report(self, quality_data: Dict[str, Any], 
                              report_type: str = "comprehensive") -> str:
        """生成质量报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"quality_report_{report_type}_{timestamp}.html"
        
        html_content = self._create_html_report(quality_data, report_type)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def _create_html_report(self, quality_data: Dict[str, Any], report_type: str) -> str:
        """创建HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MVPA质量控制报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .error {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MVPA质量控制报告</h1>
                <p>报告类型: {report_type}</p>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # 添加具体内容
        if 'performance' in quality_data:
            html += self._add_performance_section(quality_data['performance'])
        
        if 'data_quality' in quality_data:
            html += self._add_data_quality_section(quality_data['data_quality'])
        
        if 'mvpa_quality' in quality_data:
            html += self._add_mvpa_quality_section(quality_data['mvpa_quality'])
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _add_performance_section(self, performance_data: Dict[str, Any]) -> str:
        """添加性能部分"""
        html = '<div class="section"><h2>性能监控</h2>'
        
        if 'summary' in performance_data:
            summary = performance_data['summary']
            html += f"""
            <div class="metric">总处理时间: {summary.get('total_duration_formatted', 'N/A')}</div>
            """
        
        if 'memory' in performance_data:
            memory = performance_data['memory']
            html += f"""
            <div class="metric">峰值内存使用: {memory.get('peak_usage_percent', 0):.1f}%</div>
            <div class="metric">平均内存使用: {memory.get('average_usage_percent', 0):.1f}%</div>
            """
        
        html += '</div>'
        return html
    
    def _add_data_quality_section(self, data_quality: Dict[str, Any]) -> str:
        """添加数据质量部分"""
        html = '<div class="section"><h2>数据质量</h2>'
        
        # 创建质量摘要表
        html += '<table><tr><th>指标</th><th>值</th><th>状态</th></tr>'
        
        for subject, metrics in data_quality.items():
            if isinstance(metrics, dict) and 'quality_rating' in metrics:
                rating = metrics['quality_rating']
                status_class = 'good' if rating in ['Excellent', 'Good'] else 'warning' if rating == 'Fair' else 'error'
                html += f'<tr><td>{subject}</td><td>{rating}</td><td class="{status_class}">{rating}</td></tr>'
        
        html += '</table></div>'
        return html
    
    def _add_mvpa_quality_section(self, mvpa_quality: Dict[str, Any]) -> str:
        """添加MVPA质量部分"""
        html = '<div class="section"><h2>MVPA分析质量</h2>'
        
        # 添加MVPA质量指标
        for roi, metrics in mvpa_quality.items():
            if isinstance(metrics, dict):
                html += f'<h3>{roi}</h3>'
                
                if 'overall_quality' in metrics:
                    quality = metrics['overall_quality']
                    status_class = 'good' if quality in ['Excellent', 'Good'] else 'warning' if quality == 'Fair' else 'error'
                    html += f'<div class="metric">整体质量: <span class="{status_class}">{quality}</span></div>'
                
                if 'accuracy_rating' in metrics:
                    html += f'<div class="metric">准确率评级: {metrics["accuracy_rating"]}</div>'
                
                if 'significance_rating' in metrics:
                    html += f'<div class="metric">显著性评级: {metrics["significance_rating"]}</div>'
        
        html += '</div>'
        return html

def run_comprehensive_quality_check(config, subjects: List[str], runs: List[int]) -> Dict[str, Any]:
    """运行综合质量检查"""
    # 初始化组件
    monitor = PerformanceMonitor()
    assessor = DataQualityAssessor()
    reporter = QualityReporter()
    
    monitor.start_monitoring()
    
    quality_results = {
        'data_quality': {},
        'performance': {},
        'mvpa_quality': {},
        'summary': {
            'total_subjects': len(subjects),
            'total_runs': len(runs),
            'assessment_time': datetime.now().isoformat()
        }
    }
    
    try:
        # 评估数据质量
        for subject in subjects:
            subject_quality = {}
            
            for run in runs:
                # 构建文件路径
                fmri_path = config.lss.fmri_template.format(subject=subject, run=run)
                motion_path = config.lss.motion_template.format(subject=subject, run=run)
                
                # 评估质量
                run_quality = assessor.assess_fmri_quality(fmri_path, motion_path)
                subject_quality[f'run_{run}'] = run_quality
                
                # 记录质量指标
                if 'mean_snr' in run_quality:
                    monitor.record_quality_metric('snr', run_quality['mean_snr'], f'{subject}_run_{run}')
                if 'mean_tsnr' in run_quality:
                    monitor.record_quality_metric('tsnr', run_quality['mean_tsnr'], f'{subject}_run_{run}')
            
            quality_results['data_quality'][subject] = subject_quality
        
        # 停止监控并获取性能报告
        performance_report = monitor.stop_monitoring()
        quality_results['performance'] = performance_report
        
        # 生成报告
        report_file = reporter.generate_quality_report(quality_results)
        quality_results['report_file'] = report_file
        
        return quality_results
        
    except Exception as e:
        monitor.stop_monitoring()
        raise e

if __name__ == "__main__":
    # 测试质量控制模块
    print("质量控制模块测试")
    
    # 创建测试监控器
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 模拟一些处理
    time.sleep(2)
    monitor.record_processing_time("test_task", 1.5)
    monitor.record_quality_metric("test_metric", 0.85)
    
    # 生成报告
    report = monitor.stop_monitoring()
    print("性能监控测试完成")
    print(f"报告: {report}")
    
    print("\n质量控制模块测试完成")