#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced MVPA Pipeline - 增强版 v2.0
增强版MVPA分析流水线

集成最新的LSS分析、ROI mask创建、MVPA分析、质量控制和可视化
基于最新优化算法，提供完整的fMRI MVPA分析解决方案
"""

import os
import sys
import json
import time
import warnings
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
from .lss_optimized import OptimizedLSSAnalyzer
from .roi_mask_creator import OptimizedROIMaskCreator
from .mvpa_enhanced import EnhancedMVPAAnalyzer
from .quality_control import PerformanceMonitor, DataQualityAssessor, QualityReporter, run_comprehensive_quality_check
from .visualization import MVPAVisualizer, ReportGenerator, create_comprehensive_visualization_report

warnings.filterwarnings('ignore')

class EnhancedMVPAPipeline:
    """增强版MVPA分析流水线 v2.0
    
    集成最新的LSS分析、ROI mask创建、MVPA分析功能
    提供完整的端到端fMRI MVPA分析解决方案
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[GlobalConfig] = None,
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
        config : GlobalConfig, optional
            配置对象
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
        
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            self.config = GlobalConfig.load_config(config_path)
        else:
            self.config = GlobalConfig.create_default_config()
            # 保存默认配置
            default_config_path = self.output_dir / "default_config.json"
            self.config.save_config(str(default_config_path))
            print(f"已创建默认配置文件: {default_config_path}")
        
        # 验证配置
        self.config.validate_paths()
        
        # 更新输出目录
        self.config.lss_config.output_dir = str(self.output_dir / "lss_results")
        self.config.roi_config.output_dir = str(self.output_dir / "roi_masks")
        self.config.mvpa_config.results_dir = str(self.output_dir / "mvpa_results")
        
        # 创建输出目录
        self.config.create_output_directories()
        
        # 初始化组件
        self._initialize_components()
        
        self.enable_cache = enable_cache
        self.enable_quality_control = enable_quality_control
        self.enable_visualization = enable_visualization
        self.n_jobs = n_jobs
        
        # 初始化流水线
        self.pipeline = MVPAPipeline(
            cache=self.cache,
            error_handler=self.error_handler
        )
    
    def _initialize_components(self):
        """初始化增强版组件"""
        # 错误处理器
        self.error_handler = ErrorHandler(
            log_file=str(self.output_dir / "pipeline.log"),
            enable_checkpoints=True
        )
        
        # 数据缓存
        if self.enable_cache:
            self.cache = DataCache(cache_dir=str(self.output_dir / "cache"))
        else:
            self.cache = None
        
        # 优化版LSS分析器
        self.lss_analyzer = OptimizedLSSAnalyzer(
            config=self.config.lss_config,
            error_handler=self.error_handler,
            n_jobs=self.n_jobs
        )
        
        # ROI mask创建器
        self.roi_creator = OptimizedROIMaskCreator(
            config=self.config.roi_config,
            error_handler=self.error_handler
        )
        
        # 增强版MVPA分析器
        self.mvpa_analyzer = EnhancedMVPAAnalyzer(
            config=self.config.mvpa_config,
            error_handler=self.error_handler
        )
        
        # 性能监控器
        if self.enable_quality_control:
            self.performance_monitor = PerformanceMonitor(
                output_dir=str(self.output_dir / "performance")
            )
            self.quality_assessor = DataQualityAssessor()
            self.quality_reporter = QualityReporter(
                output_dir=str(self.output_dir / "quality_reports")
            )
        
        # 可视化组件
        if self.enable_visualization:
            self.visualizer = MVPAVisualizer(
                output_dir=str(self.output_dir / "visualizations"),
                error_handler=self.error_handler
            )
            self.report_generator = ReportGenerator(
                output_dir=str(self.output_dir / "reports"),
                error_handler=self.error_handler
            )
    
    def log(self, message: str):
        """增强的日志函数"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [PIPELINE] {message}")
        
        # 保存到日志文件
        log_file = Path(self.config.lss_config.output_dir).parent / "pipeline.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [PIPELINE] {message}\n")
    
    def run_comprehensive_quality_control(self, lss_results: Dict, mvpa_results: Dict) -> Dict[str, Any]:
        """运行综合质量控制"""
        try:
            quality_results = {
                'data_quality': self.quality_assessor.assess_data_quality(lss_results),
                'performance_metrics': self.performance_monitor.get_performance_summary(),
                'validation_results': self.validate_analysis_results(mvpa_results)
            }
            return quality_results
        except Exception as e:
            self.log(f"质量控制失败: {e}")
            return {'error': str(e)}
    
    def validate_analysis_results(self, mvpa_results: Dict) -> Dict[str, Any]:
        """验证分析结果"""
        validation = {
            'has_results': len(mvpa_results.get('group_results', [])) > 0,
            'has_statistics': mvpa_results.get('statistical_results') is not None,
            'significant_findings': 0
        }
        
        if 'statistical_results' in mvpa_results:
            stats_df = mvpa_results['statistical_results']
            if hasattr(stats_df, 'shape'):
                validation['significant_findings'] = len(stats_df[stats_df.get('significant', False) == True])
        
        return validation
    
    def create_comprehensive_report(self, lss_results: Dict, roi_results: Dict, 
                                  mvpa_results: Dict, quality_results: Dict) -> Dict[str, Any]:
        """创建综合报告"""
        try:
            report_results = {
                'summary_report': self.generate_summary_report(lss_results, roi_results, mvpa_results),
                'visualizations': self.create_analysis_visualizations(mvpa_results),
                'html_report': mvpa_results.get('html_report', ''),
                'quality_report': quality_results
            }
            return report_results
        except Exception as e:
            self.log(f"报告生成失败: {e}")
            return {'error': str(e)}
    
    def generate_summary_report(self, lss_results: Dict, roi_results: Dict, mvpa_results: Dict) -> str:
        """生成摘要报告"""
        summary = f"""
增强版MVPA分析摘要报告
{'='*50}

LSS分析结果:
- 成功被试: {lss_results.get('successful_subjects', 0)}
- 总试验数: {lss_results.get('total_trials', 0)}

ROI创建结果:
- 创建的ROI数量: {len(roi_results.get('created_rois', []))}

MVPA分析结果:
- 分析的ROI数量: {len(mvpa_results.get('group_results', []))}
- 显著结果数量: {mvpa_results.get('summary', {}).get('significant_results', 0)}

分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return summary
    
    def create_analysis_visualizations(self, mvpa_results: Dict) -> Dict[str, str]:
        """创建分析可视化"""
        try:
            return mvpa_results.get('visualizations', {})
        except Exception as e:
            self.log(f"可视化创建失败: {e}")
            return {}
        
        # 分析结果存储
        self.results = {
            'lss_results': {},
            'roi_results': {},
            'mvpa_results': {},
            'quality_results': {},
            'performance_results': {},
            'visualization_results': {}
        }
        
        # 性能监控
        self.performance_stats = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'stage_times': {},
            'memory_usage': [],
            'errors': []
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
    
    def run_complete_analysis_enhanced(self, subjects: Optional[List[str]] = None, 
                                      runs: Optional[List[int]] = None,
                                      contrasts: Optional[List[Tuple[str, str, str]]] = None,
                                      create_rois: bool = True,
                                      enable_quality_control: bool = True,
                                      enable_visualization: bool = True) -> Dict[str, Any]:
        """运行完整的增强版MVPA分析流水线
        
        Parameters:
        -----------
        subjects : List[str], optional
            被试列表
        runs : List[int], optional
            run列表
        contrasts : List[Tuple[str, str, str]], optional
            对比列表 [(name, cond1, cond2), ...]
        create_rois : bool
            是否创建ROI masks
        enable_quality_control : bool
            是否启用质量控制
        enable_visualization : bool
            是否启用可视化
            
        Returns:
        --------
        Dict[str, Any]
            完整分析结果
        """
        self.performance_stats['pipeline_start_time'] = datetime.now()
        
        try:
            # 使用默认参数
            if subjects is None:
                subjects = self.config.lss_config.subjects
            if runs is None:
                runs = self.config.lss_config.runs
            if contrasts is None:
                contrasts = self.config.mvpa_config.contrasts
            
            print(f"\n{'='*60}")
            print(f"开始增强版MVPA分析流水线 v2.0")
            print(f"被试数量: {len(subjects)}")
            print(f"Run数量: {len(runs)}")
            print(f"对比数量: {len(contrasts)}")
            print(f"{'='*60}\n")
            
            # 阶段1: LSS分析
            stage_start = datetime.now()
            print("阶段 1/5: 优化版LSS分析")
            lss_results = self.lss_analyzer.run_group_lss_analysis(subjects, runs)
            self.performance_stats['stage_times']['lss'] = (datetime.now() - stage_start).total_seconds()
            
            # 阶段2: ROI mask创建
            if create_rois:
                stage_start = datetime.now()
                print("\n阶段 2/5: ROI mask创建")
                roi_results = self.roi_creator.create_all_roi_masks()
                self.performance_stats['stage_times']['roi_creation'] = (datetime.now() - stage_start).total_seconds()
            else:
                roi_results = {'message': 'ROI创建已跳过'}
                self.performance_stats['stage_times']['roi_creation'] = 0
            
            # 阶段3: MVPA分析
            stage_start = datetime.now()
            print("\n阶段 3/5: 增强版MVPA分析")
            mvpa_results = self.mvpa_analyzer.run_group_analysis(subjects=subjects)
            self.performance_stats['stage_times']['mvpa'] = (datetime.now() - stage_start).total_seconds()
            
            # 阶段4: 质量控制
            if enable_quality_control:
                stage_start = datetime.now()
                print("\n阶段 4/5: 质量控制")
                quality_results = self.run_quality_control(subjects, runs)
                self.performance_stats['stage_times']['quality_control'] = (datetime.now() - stage_start).total_seconds()
            else:
                quality_results = {'message': '质量控制已跳过'}
                self.performance_stats['stage_times']['quality_control'] = 0
            
            # 阶段5: 可视化和报告
            if enable_visualization:
                stage_start = datetime.now()
                print("\n阶段 5/5: 可视化和报告生成")
                visualization_results = self.generate_visualizations_and_reports()
                self.performance_stats['stage_times']['visualization'] = (datetime.now() - stage_start).total_seconds()
            else:
                visualization_results = {'message': '可视化已跳过'}
                self.performance_stats['stage_times']['visualization'] = 0
            
            self.performance_stats['pipeline_end_time'] = datetime.now()
            
            # 整合结果
            complete_results = {
                'lss_results': lss_results,
                'roi_results': roi_results,
                'mvpa_results': mvpa_results,
                'quality_results': quality_results,
                'visualization_results': visualization_results,
                'performance_stats': self.performance_stats,
                'analysis_info': {
                    'subjects': subjects,
                    'runs': runs,
                    'contrasts': contrasts,
                    'pipeline_version': '2.0',
                    'config_version': getattr(self.config, 'version', '1.0'),
                    'start_time': self.performance_stats['pipeline_start_time'].isoformat(),
                    'end_time': self.performance_stats['pipeline_end_time'].isoformat(),
                    'total_duration': str(self.performance_stats['pipeline_end_time'] - self.performance_stats['pipeline_start_time'])
                }
            }
            
            # 保存完整结果
            results_file = self.output_dir / "complete_analysis_results_v2.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存配置快照
            config_file = self.output_dir / "analysis_config_v2.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False, default=str)
            
            total_time = (self.performance_stats['pipeline_end_time'] - self.performance_stats['pipeline_start_time']).total_seconds()
            
            print(f"\n{'='*60}")
            print(f"增强版MVPA分析流水线完成")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"结果已保存至: {results_file}")
            print(f"配置已保存至: {config_file}")
            print(f"{'='*60}")
            
            return complete_results
            
        except Exception as e:
            self.performance_stats['errors'].append({
                'stage': 'complete_analysis',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print(f"完整分析流水线失败: {e}")
            raise
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
                # 使用优化版LSS分析器
                results = self.lss_analyzer.run_group_lss_analysis(
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
                # 使用优化版ROI创建器
                results = self.roi_creator.create_all_roi_masks()
                return results
                
        except Exception as e:
            self.error_handler.log_error(e, "ROI分析", "roi")
            return {}
    
    def run_mvpa_analysis(self, subjects: List[str]) -> Dict[str, Any]:
        """运行MVPA分析"""
        try:
            with ErrorContext("MVPA分析", self.error_handler, "mvpa"):
                # 使用增强版MVPA分析器
                results = self.mvpa_analyzer.run_group_analysis(subjects=subjects)
                return results
                
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
            if hasattr(self, 'cache') and self.cache:
                self.cache.cleanup()
            
            if self.enable_quality_control and hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            
            if hasattr(self, 'lss_analyzer'):
                self.lss_analyzer.cleanup()
            
            if hasattr(self, 'roi_creator'):
                self.roi_creator.cleanup()
            
            if hasattr(self, 'mvpa_analyzer'):
                self.mvpa_analyzer.cleanup()
            
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