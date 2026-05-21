#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的MVPA Searchlight配置
专注于隐喻vs空间分类的灰质脑区分析

主要优化:
1. 简化配置，专注核心目标
2. 修正路径问题
3. 优化内存使用
4. 提高分析效率
5. 确保只分析灰质区域
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

@dataclass
class OptimizedMVPAConfig:
    """优化的MVPA Searchlight配置类"""
    
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        overrides = overrides or {}
        
        # === 核心分析参数 ===
        # 被试设置 - 根据实际数据调整
        self.subjects = overrides.get('subjects', [f"sub-{i:02d}" for i in range(1, 4)])
        self.runs = overrides.get('runs', [3, 4])  # 实际可用的run
        
        # === 路径配置 ===
        # 修正LSS数据路径 - 使用相对路径避免路径错误
        self.lss_root = Path(overrides.get('lss_root', "../../learn_LSS"))
        
        # 灰质mask路径 - 确保使用正确的灰质mask
        self.mask_dir = Path(overrides.get('mask_dir', "../../../data/masks"))
        self.gray_matter_mask = overrides.get('gray_matter_mask', "gray_matter_mask.nii.gz")
        
        # 结果输出路径
        self.results_dir = Path(overrides.get('results_dir', "../../../results/searchlight_mvpa_optimized"))
        
        # === Searchlight核心参数 ===
        self.searchlight_radius = overrides.get('searchlight_radius', 3)  # 3mm半径
        self.min_region_size = overrides.get('min_region_size', 20)  # 最小簇大小
        
        # === 分类器设置 ===
        self.svm_params = overrides.get('svm_params', {
            "C": 1.0,
            "kernel": "linear",
            "class_weight": "balanced",
            "random_state": 42
        })
        
        # === 交叉验证设置 ===
        self.cv_folds = overrides.get('cv_folds', 5)
        self.cv_strategy = overrides.get('cv_strategy', 'stratified')  # 简化CV策略
        self.cv_random_state = overrides.get('cv_random_state', 42)
        
        # === 统计设置 ===
        self.alpha_level = overrides.get('alpha_level', 0.05)
        self.correction_method = overrides.get('correction_method', 'fdr')  # FDR校正
        self.chance_level = overrides.get('chance_level', 0.5)
        
        # === 对比条件 - 专注于隐喻vs空间 ===
        self.contrasts = overrides.get('contrasts', [
            ('metaphor_vs_space', 'yy', 'kj')  # 隐喻(yy) vs 空间(kj)
        ])
        
        # === 性能优化设置 ===
        self.n_jobs = overrides.get('n_jobs', 2)  # 保守的并行设置
        self.use_parallel = overrides.get('use_parallel', True)
        self.memory_limit_gb = overrides.get('memory_limit_gb', 8.0)
        
        # === 简化的置换测试 ===
        self.n_permutations = overrides.get('n_permutations', 100)  # 减少置换次数
        self.within_subject_permutations = overrides.get('within_subject_permutations', 50)
        self.permutation_random_state = overrides.get('permutation_random_state', 42)
        
        # === 缓存设置 ===
        self.enable_cache = overrides.get('enable_cache', True)
        self.cache_max_items = overrides.get('cache_max_items', 4)
        
        # === 调试和日志 ===
        self.debug_mode = overrides.get('debug_mode', False)
        self.log_level = overrides.get('log_level', logging.INFO)
        
        # === 质量控制 ===
        self.min_trials_per_condition = overrides.get('min_trials_per_condition', 10)
        self.balance_trials = overrides.get('balance_trials', True)
        
        # === 输出控制 ===
        self.save_individual_maps = overrides.get('save_individual_maps', True)
        self.save_group_maps = overrides.get('save_group_maps', True)
        self.generate_report = overrides.get('generate_report', True)
        
        # 创建必要的目录
        self._create_directories()
        
        # 设置日志
        self._setup_logging()
    
    def _create_directories(self):
        """创建必要的输出目录"""
        directories = [
            self.results_dir,
            self.results_dir / "accuracy_maps",
            self.results_dir / "statistical_maps", 
            self.results_dir / "subject_results",
            self.results_dir / "logs",
            self.results_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.results_dir / "logs" / "analysis.log"
        
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_gray_matter_mask_path(self) -> Path:
        """获取灰质mask的完整路径"""
        mask_path = self.mask_dir / self.gray_matter_mask
        if not mask_path.exists():
            # 尝试其他可能的灰质mask文件
            alternatives = [
                "cortical_gray_matter_mask.nii.gz",
                "mni152_gm_mask.nii.gz",
                "gray_matter_mask_91x109x91.nii.gz"
            ]
            
            for alt in alternatives:
                alt_path = self.mask_dir / alt
                if alt_path.exists():
                    self.logger.info(f"使用替代灰质mask: {alt}")
                    return alt_path
            
            raise FileNotFoundError(f"未找到灰质mask文件: {mask_path}")
        
        return mask_path
    
    def validate_paths(self) -> bool:
        """验证关键路径是否存在"""
        errors = []
        
        # 检查LSS数据目录
        if not self.lss_root.exists():
            errors.append(f"LSS数据目录不存在: {self.lss_root}")
        
        # 检查mask目录
        if not self.mask_dir.exists():
            errors.append(f"Mask目录不存在: {self.mask_dir}")
        
        # 检查灰质mask
        try:
            self.get_gray_matter_mask_path()
        except FileNotFoundError as e:
            errors.append(str(e))
        
        # 检查被试数据
        missing_subjects = []
        for subject in self.subjects:
            subject_dir = self.lss_root / subject
            if not subject_dir.exists():
                missing_subjects.append(subject)
        
        if missing_subjects:
            errors.append(f"缺失被试数据: {missing_subjects}")
        
        if errors:
            for error in errors:
                self.logger.error(error)
            return False
        
        self.logger.info("路径验证通过")
        return True
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        return f"""
=== 优化MVPA Searchlight配置摘要 ===

数据设置:
  - 被试数量: {len(self.subjects)}
  - Run数量: {len(self.runs)}
  - 对比条件: {[c[0] for c in self.contrasts]}

Searchlight参数:
  - 半径: {self.searchlight_radius}mm
  - 最小簇大小: {self.min_region_size}体素
  - 灰质mask: {self.gray_matter_mask}

分类设置:
  - 分类器: 线性SVM (C={self.svm_params['C']})
  - 交叉验证: {self.cv_folds}折
  - 置换测试: {self.n_permutations}次

输出设置:
  - 结果目录: {self.results_dir}
  - 校正方法: {self.correction_method}
  - 显著性水平: {self.alpha_level}

性能设置:
  - 并行作业: {self.n_jobs}
  - 内存限制: {self.memory_limit_gb}GB
  - 缓存: {'启用' if self.enable_cache else '禁用'}
        """

def create_optimized_config(overrides: Optional[Dict[str, Any]] = None) -> OptimizedMVPAConfig:
    """创建优化配置的便捷函数"""
    return OptimizedMVPAConfig(overrides)

# 预定义的配置模板
QUICK_TEST_CONFIG = {
    'subjects': ['sub-01', 'sub-02'],
    'runs': [3],
    'n_permutations': 50,
    'within_subject_permutations': 25,
    'debug_mode': True
}

FULL_ANALYSIS_CONFIG = {
    'subjects': [f"sub-{i:02d}" for i in range(1, 29)],
    'runs': [3, 4],
    'n_permutations': 500,
    'within_subject_permutations': 100,
    'debug_mode': False
}

if __name__ == "__main__":
    # 测试配置
    config = create_optimized_config(QUICK_TEST_CONFIG)
    print(config.get_config_summary())
    
    # 验证路径
    if config.validate_paths():
        print("\n✅ 配置验证通过，可以开始分析")
    else:
        print("\n❌ 配置验证失败，请检查路径设置")