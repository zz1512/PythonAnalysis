#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版MVPA分析流水线
整合所有优化模块，提供统一的分析接口
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import asdict

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 导入增强模块
from .global_config import GlobalConfig, LSSConfig, ROIConfig, MVPAConfig
from .error_handling import ErrorHandler, ErrorContext, MVPAError
from .data_flow import MVPAPipeline, DataCache, create_lss_analysis_step, create_roi_analysis_step, create_mvpa_analysis_step
from .lss_enhanced import LSSAnalysisEnhanced
from .quality_control import PerformanceMonitor, DataQualityAssessor, QualityReporter, run_comprehensive_quality_check
from .visualization import MVPAVisualizer, ReportGenerator, create_comprehensive_visualization_report

class EnhancedMVPAPipeline:
    """增强版MVPA分析流水线"""
    
    def __init__(self, config_path: Optional[str] = None, 
                 output_dir: str = "./enhanced_mvpa_results",
                 enable_cache: bool = True,
                 enable_quality_control: bool = True,
                 enable_visualization: bool = True,
                 n_jobs: int = -1):
        """
        初始化增强版MVPA流水线
        
        Parameters:
        -----------
        config_path : str, optional
            配置文件路径
        output_dir : str
            输出目录
        enable_cache : bool
            是否启用缓存
        enable_quality_control : bool
            是否启用质量控制
        enable_visualization : bool
            是否启用可视化
        n_jobs : int
            并行作业数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化配置
        if config_path and os.path.exists(config_path):
            self.config = GlobalConfig.load_config(config_path)
        else:
            self.config = GlobalConfig.create_default_config()
            # 保存默认配置
            default_config_path = self.output_dir / "default_config.json"
            self.config.save_config(str(default_config_path))
            print(f"已创建默认配置文件: {default_config_path}")
        
        # 更新输出目录
        self.config.lss_config.output_dir = str(self.output_dir / "lss_results")
        self.config.roi_config.output_dir = str(self.output_dir / "roi_masks")
        self.config.mvpa_config.results_dir = str(self.output_dir / "mvpa_results")
        
        # 创建输出目录
        self.config.create_output_directories()
        
        # 初始化组件
        self.error_handler = ErrorHandler(
            log_file=str(self.output_dir / "pipeline.log"),
            enable_checkpoints=True
        )
        
        self.enable_cache = enable_cache
        self.enable_quality_control = enable_quality_control
        self.enable_visualization = enable_visualization
        self.n_jobs = n_jobs
        
        # 初始化缓存
        if self.enable_cache:
            self.cache = DataCache(cache_dir=str(self.output_dir / "cache"))
        else:
            self.cache = None
        
        # 初始化性能监控
        if self.enable_quality_control:
            self.performance_monitor = PerformanceMonitor(
                output_dir=str(self.output_dir / "performance")
            )
            self.quality_assessor = DataQualityAssessor()
            self.quality_reporter = QualityReporter(
                output_dir=str(self.output_dir / "quality_reports")
            )
        
        # 初始化可视化
        if self.enable_visualization:
            self.visualizer = MVPAVisualizer(
                output_dir=str(self.output_dir / "visualizations"),
                error_handler=self.error_handler
            )
            self.report_generator = ReportGenerator(
                output_dir=str(self.output_dir / "reports"),
                error_handler=self.error_handler
            )
        
        # 初始化流水线
        self.pipeline = MVPAPipeline(
            cache=self.cache,
            error_handler=self.error_handler
        )
        
        # 分析结果存储
        self.results = {
            'lss_results': {},
            'roi_results': {},
            'mvpa_results': {},
            'quality_results': {},
            'performance_results': {},
            'visualization_results': {}
        }
        
        print(f"增强版MVPA流水线已初始化")
        print(f"输出目录: {self.output_dir}")
        print(f"缓存: {'启用' if self.enable_cache else '禁用'}")
        print(f"质量控制: {'启用' if self.enable_quality_control else '禁用'}")
        print(f"可视化: {'启用' if self.enable_visualization else '禁用'}")
    
    def run_complete_analysis(self, 
                            subjects: Optional[List[str]] = None,
                            runs: Optional[List[str]] = None,
                            skip_lss: bool = False,
                            skip_roi: bool = False,
                            skip_mvpa: bool = False) -> Dict[str, Any]:
        """运行完整的MVPA分析流程"""
        try:
            with ErrorContext("完整MVPA分析", self.error_handler, "pipeline"):
                start_time = time.time()
                
                # 开始性能监控
                if self.enable_quality_control:
                    self.performance_monitor.start_monitoring()
                
                print("\n=== 开始增强版MVPA分析 ===")
                print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 使用配置中的被试和run列表（如果未指定）
                if subjects is None:
                    subjects = self.config.lss_config.subjects
                if runs is None:
                    runs = self.config.lss_config.runs
                
                print(f"被试数量: {len(subjects)}")
                print(f"Run数量: {len(runs)}")
                
                # 1. LSS分析
                if not skip_lss:
                    print("\n--- 步骤1: LSS分析 ---")
                    lss_results = self.run_lss_analysis(subjects, runs)
                    self.results['lss_results'] = lss_results
                    print(f"LSS分析完成，处理了 {len(lss_results)} 个被试")
                
                # 2. ROI分析（如果需要）
                if not skip_roi:
                    print("\n--- 步骤2: ROI掩模创建 ---")
                    roi_results = self.run_roi_analysis()
                    self.results['roi_results'] = roi_results
                    print(f"ROI分析完成，创建了 {len(roi_results)} 个ROI")
                
                # 3. MVPA分析
                if not skip_mvpa:
                    print("\n--- 步骤3: MVPA分析 ---")
                    mvpa_results = self.run_mvpa_analysis(subjects)
                    self.results['mvpa_results'] = mvpa_results
                    print(f"MVPA分析完成")
                
                # 4. 质量控制
                if self.enable_quality_control:
                    print("\n--- 步骤4: 质量控制 ---")
                    quality_results = self.run_quality_control(subjects, runs)
                    self.results['quality_results'] = quality_results
                    print(f"质量控制完成")
                
                # 5. 可视化和报告
                if self.enable_visualization:
                    print("\n--- 步骤5: 可视化和报告生成 ---")
                    viz_results = self.generate_visualizations_and_reports()
                    self.results['visualization_results'] = viz_results
                    print(f"可视化和报告生成完成")
                
                # 停止性能监控
                if self.enable_quality_control:
                    performance_results = self.performance_monitor.stop_monitoring()
                    self.results['performance_results'] = performance_results
                
                # 保存最终结果
                self.save_final_results()
                
                total_time = time.time() - start_time
                print(f"\n=== 分析完成 ===")
                print(f"总耗时: {total_time:.2f} 秒")
                print(f"结果保存在: {self.output_dir}")
                
                return self.results
                
        except Exception as e:
            self.error_handler.log_error(e, "完整MVPA分析", "pipeline")
            raise MVPAError(f"MVPA分析失败: {str(e)}") from e
    
    def run_lss_analysis(self, subjects: List[str], runs: List[str]) -> Dict[str, Any]:
        """运行LSS分析"""
        try:
            with ErrorContext("LSS分析", self.error_handler, "lss"):
                # 创建LSS分析器
                lss_analyzer = LSSAnalysisEnhanced(
                    config=self.config.lss_config,
                    error_handler=self.error_handler,
                    n_jobs=self.n_jobs
                )
                
                # 运行分析
                results = lss_analyzer.run_group_lss_analysis(
                    subjects=subjects,
                    runs=runs
                )
                
                return results
                
        except Exception as e:
            self.error_handler.log_error(e, "LSS分析", "lss")
            return {}
    
    def run_roi_analysis(self) -> Dict[str, Any]:
        """运行ROI分析"""
        try:
            with ErrorContext("ROI分析", self.error_handler, "roi"):
                # 这里可以集成ROI创建逻辑
                # 目前返回空结果，实际使用时需要集成create_optimized_roi_masks.py
                print("ROI分析功能待集成")
                return {}
                
        except Exception as e:
            self.error_handler.log_error(e, "ROI分析", "roi")
            return {}
    
    def run_mvpa_analysis(self, subjects: List[str]) -> Dict[str, Any]:
        """运行MVPA分析"""
        try:
            with ErrorContext("MVPA分析", self.error_handler, "mvpa"):
                # 这里可以集成MVPA分析逻辑
                # 目前返回空结果，实际使用时需要集成fmri_mvpa_roi_pipeline_enhanced.py
                print("MVPA分析功能待集成")
                return {}
                
        except Exception as e:
            self.error_handler.log_error(e, "MVPA分析", "mvpa")
            return {}
    
    def run_quality_control(self, subjects: List[str], runs: List[str]) -> Dict[str, Any]:
        """运行质量控制"""
        try:
            with ErrorContext("质量控制", self.error_handler, "quality"):
                quality_results = {}
                
                # 对每个被试进行质量评估
                for subject in subjects:
                    subject_quality = {}
                    
                    for run in runs:
                        # 构建数据路径
                        func_file = os.path.join(
                            self.config.lss_config.data_dir,
                            subject,
                            f"func/{subject}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                        )
                        
                        if os.path.exists(func_file):
                            # 运行质量评估
                            run_quality = run_comprehensive_quality_check(
                                func_file=func_file,
                                output_dir=str(self.output_dir / "quality_reports" / subject),
                                subject_id=subject,
                                run_id=run
                            )
                            subject_quality[run] = run_quality
                    
                    if subject_quality:
                        quality_results[subject] = subject_quality
                
                return quality_results
                
        except Exception as e:
            self.error_handler.log_error(e, "质量控制", "quality")
            return {}
    
    def generate_visualizations_and_reports(self) -> Dict[str, Any]:
        """生成可视化和报告"""
        try:
            with ErrorContext("可视化和报告生成", self.error_handler, "visualization"):
                viz_results = {}
                
                # 生成综合可视化报告
                if self.results['mvpa_results']:
                    comprehensive_viz = create_comprehensive_visualization_report(
                        analysis_results=self.results['mvpa_results'],
                        quality_data=self.results.get('quality_results'),
                        config=asdict(self.config),
                        output_dir=str(self.output_dir / "visualizations")
                    )
                    viz_results['comprehensive'] = comprehensive_viz
                
                # 生成质量报告
                if self.results['quality_results'] and self.enable_quality_control:
                    quality_report = self.quality_reporter.generate_comprehensive_report(
                        self.results['quality_results']
                    )
                    viz_results['quality_report'] = quality_report
                
                # 生成性能报告
                if self.results['performance_results'] and self.enable_quality_control:
                    performance_report = self.performance_monitor.generate_report()
                    viz_results['performance_report'] = performance_report
                
                return viz_results
                
        except Exception as e:
            self.error_handler.log_error(e, "可视化和报告生成", "visualization")
            return {}
    
    def save_final_results(self):
        """保存最终结果"""
        try:
            with ErrorContext("保存最终结果", self.error_handler, "save"):
                # 保存结果摘要
                results_summary = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'config': asdict(self.config),
                    'results_summary': {
                        'lss_subjects_processed': len(self.results.get('lss_results', {})),
                        'roi_created': len(self.results.get('roi_results', {})),
                        'mvpa_completed': bool(self.results.get('mvpa_results')),
                        'quality_assessed': len(self.results.get('quality_results', {})),
                        'visualizations_created': len(self.results.get('visualization_results', {}))
                    },
                    'output_directories': {
                        'main': str(self.output_dir),
                        'lss': self.config.lss_config.output_dir,
                        'roi': self.config.roi_config.output_dir,
                        'mvpa': self.config.mvpa_config.results_dir,
                        'visualizations': str(self.output_dir / "visualizations"),
                        'reports': str(self.output_dir / "reports")
                    }
                }
                
                # 保存摘要文件
                summary_file = self.output_dir / "analysis_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(results_summary, f, indent=2, ensure_ascii=False)
                
                # 保存详细结果
                detailed_results_file = self.output_dir / "detailed_results.json"
                with open(detailed_results_file, 'w', encoding='utf-8') as f:
                    # 转换numpy数组为列表以便JSON序列化
                    serializable_results = self._make_json_serializable(self.results)
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
                print(f"结果已保存:")
                print(f"  摘要: {summary_file}")
                print(f"  详细结果: {detailed_results_file}")
                
        except Exception as e:
            self.error_handler.log_error(e, "保存最终结果", "save")
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """获取分析状态"""
        status = {
            'pipeline_initialized': True,
            'config_loaded': bool(self.config),
            'cache_enabled': self.enable_cache,
            'quality_control_enabled': self.enable_quality_control,
            'visualization_enabled': self.enable_visualization,
            'output_directory': str(self.output_dir),
            'analysis_completed': {
                'lss': bool(self.results.get('lss_results')),
                'roi': bool(self.results.get('roi_results')),
                'mvpa': bool(self.results.get('mvpa_results')),
                'quality': bool(self.results.get('quality_results')),
                'visualization': bool(self.results.get('visualization_results'))
            },
            'error_count': len(self.error_handler.errors) if self.error_handler else 0
        }
        
        return status
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.cache:
                self.cache.cleanup()
            
            if self.enable_quality_control and hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            
            print("流水线资源清理完成")
            
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

def create_enhanced_pipeline(config_path: Optional[str] = None,
                           output_dir: str = "./enhanced_mvpa_results",
                           **kwargs) -> EnhancedMVPAPipeline:
    """创建增强版MVPA流水线的便捷函数"""
    return EnhancedMVPAPipeline(
        config_path=config_path,
        output_dir=output_dir,
        **kwargs
    )

def run_quick_analysis(data_dir: str,
                      subjects: List[str],
                      runs: List[str],
                      output_dir: str = "./quick_mvpa_results",
                      **kwargs) -> Dict[str, Any]:
    """快速分析函数"""
    # 创建快速配置
    config = GlobalConfig.create_default_config()
    config.lss_config.data_dir = data_dir
    config.lss_config.subjects = subjects
    config.lss_config.runs = runs
    
    # 创建流水线
    pipeline = EnhancedMVPAPipeline(
        output_dir=output_dir,
        **kwargs
    )
    pipeline.config = config
    
    try:
        # 运行分析
        results = pipeline.run_complete_analysis(subjects=subjects, runs=runs)
        return results
    finally:
        # 清理资源
        pipeline.cleanup()

if __name__ == "__main__":
    # 示例用法
    print("增强版MVPA分析流水线")
    print("\n主要功能:")
    print("- EnhancedMVPAPipeline: 完整的增强版分析流水线")
    print("- create_enhanced_pipeline: 便捷的流水线创建函数")
    print("- run_quick_analysis: 快速分析函数")
    
    print("\n使用示例:")
    print("# 创建流水线")
    print('pipeline = create_enhanced_pipeline(output_dir="./my_results")')
    print("\n# 运行完整分析")
    print('results = pipeline.run_complete_analysis(subjects=["sub-01", "sub-02"], runs=["run-1", "run-2"])')
    print("\n# 快速分析")
    print('results = run_quick_analysis(data_dir="/path/to/data", subjects=["sub-01"], runs=["run-1"])')