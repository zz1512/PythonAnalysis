#!/usr/bin/env python3
"""
fMRI GLM 分析配置文件

这个文件包含了所有可配置的参数，用户可以根据需要修改这些参数
而不需要修改主分析脚本。
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

class AnalysisConfig:
    """fMRI GLM分析配置类"""
    
    def __init__(self):
        # =========================
        # 基本数据配置
        # =========================
        
        # 被试列表 - 可以是具体列表或范围
        self.SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]  # sub-01 到 sub-28
        # 或者指定特定被试：
        # self.SUBJECTS = ["sub-01", "sub-02", "sub-03"]
        
        # 要分析的run列表
        self.RUNS = [4]  # 可以是多个run: [1, 2, 3, 4]
        
        # TR回退值（当无法从图像头文件读取时使用）
        self.TR_FALLBACK = 2.0
        
        # =========================
        # 文件路径模板
        # =========================
        
        # fMRI数据路径模板
        # 注意：这里使用原始fMRIPrep输出，不使用预平滑的文件
        self.FMRI_TPL = r"../../Pro_proc_data/{sub}/run{run}/{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
        
        # 如果你的数据已经是.nii.gz格式，使用：
        # self.FMRI_TPL = r"../../Pro_proc_data/{sub}/run{run}/{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        
        # 事件文件路径模板
        self.EVENT_TPL = r"../../data_events/{sub}/{sub}_run-{run}_events.tsv"
        
        # 头动参数文件路径模板
        self.MOTION_TPL = r"../../Pro_proc_data/{sub}/multi_reg/multi_reg_run{run}.txt"
        
        # 输出根目录
        self.OUTPUT_ROOT = Path(r"../../learn_batch_enhanced")
        
        # =========================
        # GLM分析参数
        # =========================
        
        # 对比定义
        self.CONTRASTS: Dict[str, Dict[str, float]] = {
            "yy > kj": {"yy": +1.0, "kj": -1.0},
            # 可以添加更多对比：
            # "kj > yy": {"yy": -1.0, "kj": +1.0},
            # "yy": {"yy": +1.0},  # 单条件vs基线
            # "kj": {"kj": +1.0},
        }
        
        # =========================
        # 平滑参数 - 关键配置
        # =========================
        
        # 平滑核大小（FWHM，单位：毫米）
        # 设置为None则不进行平滑
        self.SMOOTHING_FWHM = 6.0  # 标准值：6mm
        
        # 其他常用值：
        # self.SMOOTHING_FWHM = 4.0   # 较小的平滑
        # self.SMOOTHING_FWHM = 8.0   # 较大的平滑
        # self.SMOOTHING_FWHM = None  # 不平滑
        
        # =========================
        # FirstLevelModel参数
        # =========================
        
        # 噪声模型
        self.NOISE_MODEL = "ar1"  # 或 "ols"
        
        # HRF模型
        self.HRF_MODEL = "spm"  # 或 "glover", "fir"
        
        # 漂移模型
        self.DRIFT_MODEL = "cosine"  # 或 "polynomial", None
        
        # 高通滤波截止频率（Hz）
        self.HIGH_PASS = 1/128.0  # SPM默认值
        
        # 切片时间参考点
        self.SLICE_TIME_REF = 0.5  # TR的中点
        
        # 是否标准化
        self.STANDARDIZE = False
        
        # 是否进行信号缩放
        self.SIGNAL_SCALING = False
        
        # =========================
        # 并行处理配置
        # =========================
        
        # 是否使用多进程
        self.USE_MULTIPROCESS = True
        
        # 最大工作进程数
        self.MAX_WORKERS = 2
        
        # GLM内部并行作业数
        # 当使用多进程时，GLM内部应该使用单线程避免过度并行
        # 当不使用多进程时，GLM可以使用所有可用核心
        self.N_JOBS_GLM = 1 if self.USE_MULTIPROCESS else -1
        
        # =========================
        # 二阶分析配置
        # =========================
        
        self.SECOND_LEVEL_CONFIG = {
            # 统计阈值
            "alpha": 0.05,          # FDR或未校正p值阈值
            "use_fdr": True,        # True: 使用FDR校正; False: 使用未校正p值
            "p_unc": 0.001,         # 未校正p值阈值（当use_fdr=False时使用）
            "cluster_k": 10,        # 聚类大小阈值（体素数）
            "two_sided": True,      # 是否进行双侧检验
            
            # 可视化参数
            "plot_dpi": 220,        # 图像分辨率
            "colormap": "cold_hot", # 颜色映射
        }
        
        # =========================
        # 日志配置
        # =========================
        
        # 日志级别
        self.LOG_LEVEL = logging.INFO  # 或 logging.DEBUG, logging.WARNING
        
        # 是否保存日志到文件
        self.SAVE_LOG_TO_FILE = True
        
        # 日志格式
        self.LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
        
        # =========================
        # 高级配置
        # =========================
        
        # 内存优化
        self.MINIMIZE_MEMORY = False  # 是否最小化内存使用
        
        # 详细输出
        self.VERBOSE = 0  # 0: 静默, 1: 基本信息, 2: 详细信息
        
        # 是否保存中间结果
        self.SAVE_INTERMEDIATE = True
        
        # 是否覆盖已存在的结果
        self.OVERWRITE_EXISTING = False
        
        # 脑区标注图谱
        self.ATLAS_CONFIG = {
            "use_harvard_oxford": True,
            "cortical_threshold": 25,  # Harvard-Oxford皮层图谱阈值
            "subcortical_threshold": 25,  # Harvard-Oxford皮下图谱阈值
        }
    
    def validate_config(self) -> bool:
        """验证配置参数的有效性"""
        errors = []
        
        # 检查必要的路径
        if not self.OUTPUT_ROOT:
            errors.append("OUTPUT_ROOT must be specified")
        
        # 检查被试列表
        if not self.SUBJECTS:
            errors.append("SUBJECTS list cannot be empty")
        
        # 检查run列表
        if not self.RUNS:
            errors.append("RUNS list cannot be empty")
        
        # 检查对比定义
        if not self.CONTRASTS:
            errors.append("CONTRASTS dictionary cannot be empty")
        
        # 检查平滑参数
        if self.SMOOTHING_FWHM is not None and self.SMOOTHING_FWHM <= 0:
            errors.append("SMOOTHING_FWHM must be positive or None")
        
        # 检查并行参数
        if self.MAX_WORKERS <= 0:
            errors.append("MAX_WORKERS must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_summary(self):
        """打印配置摘要"""
        print("=== fMRI GLM Analysis Configuration ===")
        print(f"Subjects: {len(self.SUBJECTS)} ({self.SUBJECTS[0]} to {self.SUBJECTS[-1]})")
        print(f"Runs: {self.RUNS}")
        print(f"Contrasts: {list(self.CONTRASTS.keys())}")
        print(f"Smoothing FWHM: {self.SMOOTHING_FWHM}mm" if self.SMOOTHING_FWHM else "No smoothing")
        print(f"Multiprocessing: {self.USE_MULTIPROCESS} (workers: {self.MAX_WORKERS})")
        print(f"Output directory: {self.OUTPUT_ROOT}")
        print(f"Second-level threshold: {'FDR' if self.SECOND_LEVEL_CONFIG['use_fdr'] else 'Uncorrected'} p<{self.SECOND_LEVEL_CONFIG['alpha']}")
        print("="*50)

# 创建默认配置实例
default_config = AnalysisConfig()

# 用户可以创建自定义配置
def create_custom_config():
    """创建自定义配置的示例"""
    config = AnalysisConfig()
    
    # 修改特定参数
    config.SUBJECTS = ["sub-01", "sub-02", "sub-03"]  # 只分析前3个被试
    config.SMOOTHING_FWHM = None  # 不进行平滑
    config.USE_MULTIPROCESS = False  # 关闭多进程
    config.SECOND_LEVEL_CONFIG["use_fdr"] = False  # 使用未校正阈值
    config.SECOND_LEVEL_CONFIG["p_unc"] = 0.001
    
    return config

if __name__ == "__main__":
    # 验证和显示默认配置
    if default_config.validate_config():
        default_config.print_summary()
    else:
        print("Default configuration has errors!")