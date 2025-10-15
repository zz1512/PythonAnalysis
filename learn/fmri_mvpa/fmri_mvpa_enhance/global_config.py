#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置管理模块
统一管理MVPA分析流程中的所有配置参数
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union
import warnings

@dataclass
class LSSConfig:
    """LSS分析配置"""
    # 基本参数
    subjects: List[str] = None
    runs: List[int] = None
    tr_fallback: float = 2.0
    n_jobs_glm: int = 4
    lss_collapse_by_cond: bool = True
    
    # 路径配置
    fmri_template: str = ""
    event_template: str = ""
    motion_template: str = ""
    output_root: str = ""
    
    # 质量控制
    max_motion_mm: float = 3.0
    min_signal_quality: float = 0.8
    auto_tr_detection: bool = True
    
    # 性能优化
    parallel_subjects: bool = True
    parallel_runs: bool = True
    memory_limit_gb: float = 8.0
    cache_intermediate: bool = True
    
    def __post_init__(self):
        if self.subjects is None:
            self.subjects = []
        if self.runs is None:
            self.runs = [1, 2, 3]

@dataclass
class ROIConfig:
    """ROI生成配置"""
    # 基本参数
    clusters_file: Optional[str] = None
    output_dir: Optional[str] = None
    
    # 球形ROI配置
    sphere_radius: float = 8.0
    min_voxels_sphere: int = 50
    
    # 解剖ROI配置
    anatomical_threshold: float = 25.0
    min_voxels_anatomical: int = 100
    
    # 激活ROI配置
    activation_threshold: float = 3.1
    cluster_size_threshold: int = 10
    min_voxels_activation: int = 20
    
    # 功能网络配置
    network_threshold: float = 0.2
    min_voxels_network: int = 200
    
    # 质量控制
    max_roi_volume: float = 10000.0
    min_roi_volume: float = 100.0
    overlap_threshold: float = 0.8
    apply_hemisphere_separation: bool = True
    
    # 可视化配置
    plot_dpi: int = 150
    plot_figsize: tuple = (15, 10)
    colormap: str = 'hot'

@dataclass
class MVPAConfig:
    """MVPA分析配置"""
    # 基本参数
    subjects: List[str] = None
    runs: List[int] = None
    contrasts: Dict[str, List[str]] = None
    
    # 路径配置
    lss_dir: str = ""
    roi_dir: str = ""
    results_dir: str = ""
    
    # 分类器参数
    svm_c: float = 1.0
    svm_kernel: str = 'linear'
    svm_gamma: str = 'scale'
    
    # 交叉验证参数
    cv_folds: int = 5
    cv_strategy: str = 'stratified'
    cv_random_state: int = 42
    
    # 置换检验参数
    n_permutations: int = 1000
    permutation_random_state: int = 42
    
    # 多重比较校正
    alpha_level: float = 0.05
    correction_method: str = 'fdr_bh'
    
    # 并行处理
    n_jobs: int = -1
    parallel_backend: str = 'loky'
    
    # 内存管理
    memory_limit_gb: float = 16.0
    batch_size: int = 10
    
    def __post_init__(self):
        if self.subjects is None:
            self.subjects = []
        if self.runs is None:
            self.runs = [1, 2, 3]
        if self.contrasts is None:
            self.contrasts = {}

class GlobalConfig:
    """全局配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.lss = LSSConfig()
        self.roi = ROIConfig()
        self.mvpa = MVPAConfig()
        
        # 全局设置
        self.project_name = "MVPA_Enhanced"
        self.version = "1.0.0"
        self.debug_mode = False
        self.log_level = "INFO"
        
        # 路径设置
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "results"
        self.log_dir = self.base_dir / "logs"
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.json'):
                    config_data = json.load(f)
                elif config_file.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file}")
            
            # 更新配置
            if 'lss' in config_data:
                self.lss = LSSConfig(**config_data['lss'])
            if 'roi' in config_data:
                self.roi = ROIConfig(**config_data['roi'])
            if 'mvpa' in config_data:
                self.mvpa = MVPAConfig(**config_data['mvpa'])
            
            # 更新全局设置
            for key, value in config_data.get('global', {}).items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            print(f"配置已从 {config_file} 加载")
            
        except Exception as e:
            warnings.warn(f"加载配置文件失败: {e}，使用默认配置")
    
    def save_config(self, config_file: str, format: str = 'json'):
        """保存配置到文件"""
        try:
            config_data = {
                'global': {
                    'project_name': self.project_name,
                    'version': self.version,
                    'debug_mode': self.debug_mode,
                    'log_level': self.log_level
                },
                'lss': asdict(self.lss),
                'roi': asdict(self.roi),
                'mvpa': asdict(self.mvpa)
            }
            
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                elif format in ['yml', 'yaml']:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的格式: {format}")
            
            print(f"配置已保存到 {config_file}")
            
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")
    
    def validate_config(self):
        """验证配置参数"""
        errors = []
        
        # 验证LSS配置
        if not self.lss.subjects:
            errors.append("LSS配置: subjects不能为空")
        if self.lss.tr_fallback <= 0:
            errors.append("LSS配置: tr_fallback必须大于0")
        if self.lss.n_jobs_glm <= 0:
            errors.append("LSS配置: n_jobs_glm必须大于0")
        
        # 验证ROI配置
        if self.roi.sphere_radius <= 0:
            errors.append("ROI配置: sphere_radius必须大于0")
        if self.roi.min_voxels_sphere <= 0:
            errors.append("ROI配置: min_voxels_sphere必须大于0")
        if not (0 < self.roi.anatomical_threshold <= 100):
            errors.append("ROI配置: anatomical_threshold必须在0-100之间")
        
        # 验证MVPA配置
        if not self.mvpa.subjects:
            errors.append("MVPA配置: subjects不能为空")
        if self.mvpa.cv_folds <= 1:
            errors.append("MVPA配置: cv_folds必须大于1")
        if self.mvpa.n_permutations <= 0:
            errors.append("MVPA配置: n_permutations必须大于0")
        if not (0 < self.mvpa.alpha_level < 1):
            errors.append("MVPA配置: alpha_level必须在0-1之间")
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(errors))
        
        print("配置验证通过")
        return True
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.output_dir,
            self.log_dir,
            Path(self.lss.output_root) if self.lss.output_root else None,
            Path(self.roi.output_dir) if self.roi.output_dir else None,
            Path(self.mvpa.results_dir) if self.mvpa.results_dir else None
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"目录已创建: {directory}")
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        summary = f"""
=== {self.project_name} v{self.version} 配置摘要 ===

LSS配置:
  - 被试数量: {len(self.lss.subjects)}
  - Run数量: {len(self.lss.runs)}
  - TR: {self.lss.tr_fallback}s
  - 并行作业数: {self.lss.n_jobs_glm}
  - 并行处理: 被试={self.lss.parallel_subjects}, Run={self.lss.parallel_runs}

ROI配置:
  - 球形ROI半径: {self.roi.sphere_radius}mm
  - 最小体素数: {self.roi.min_voxels_sphere}
  - 解剖ROI阈值: {self.roi.anatomical_threshold}%
  - 半球分离: {self.roi.apply_hemisphere_separation}

MVPA配置:
  - 交叉验证折数: {self.mvpa.cv_folds}
  - 置换检验次数: {self.mvpa.n_permutations}
  - 显著性水平: {self.mvpa.alpha_level}
  - 多重比较校正: {self.mvpa.correction_method}
  - 并行作业数: {self.mvpa.n_jobs}

全局设置:
  - 调试模式: {self.debug_mode}
  - 日志级别: {self.log_level}
  - 基础目录: {self.base_dir}
        """
        return summary

# 创建默认配置实例
default_config = GlobalConfig()

# 配置工厂函数
def create_config(config_file: Optional[str] = None, **kwargs) -> GlobalConfig:
    """创建配置实例"""
    config = GlobalConfig(config_file)
    
    # 应用关键字参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.lss, key):
            setattr(config.lss, key, value)
        elif hasattr(config.roi, key):
            setattr(config.roi, key, value)
        elif hasattr(config.mvpa, key):
            setattr(config.mvpa, key, value)
    
    return config

# 示例配置文件生成
def generate_example_config(output_file: str = "example_config.json"):
    """生成示例配置文件"""
    config = GlobalConfig()
    
    # 设置示例值
    config.lss.subjects = ['sub-01', 'sub-02', 'sub-03']
    config.lss.runs = [1, 2, 3]
    config.lss.fmri_template = "/path/to/fmri/sub-{subject}/func/sub-{subject}_task-metaphor_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    config.lss.event_template = "/path/to/events/sub-{subject}/func/sub-{subject}_task-metaphor_run-{run:02d}_events.tsv"
    config.lss.output_root = "/path/to/output/lss_results"
    
    config.roi.output_dir = "/path/to/output/roi_masks"
    config.roi.clusters_file = "/path/to/activation/clusters_table.csv"
    
    config.mvpa.lss_dir = "/path/to/output/lss_results"
    config.mvpa.roi_dir = "/path/to/output/roi_masks"
    config.mvpa.results_dir = "/path/to/output/mvpa_results"
    config.mvpa.contrasts = {
        "metaphor_vs_literal": ["metaphor", "literal"]
    }
    
    config.save_config(output_file)
    print(f"示例配置文件已生成: {output_file}")

if __name__ == "__main__":
    # 生成示例配置
    generate_example_config()
    
    # 测试配置
    config = create_config()
    print(config.get_config_summary())
    config.validate_config()