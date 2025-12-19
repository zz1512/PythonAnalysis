#!/usr/bin/env python3
"""
fMRI GLM 分析配置文件 (Pro Version)
集成 Cluster-level FWE (Permutation) 标准
"""

from pathlib import Path
from typing import Dict, List
import logging


class AnalysisConfig:
    def __init__(self):
        # =========================
        # 1. 基础路径与被试设置
        # =========================

        # 被试列表 (sub-01 ... sub-29)
        self.SUBJECTS = [f"sub-{i:02d}" for i in range(1, 30)]

        # 分析的 Run (Run 3 & 4 为定位任务)
        self.RUNS = [3, 4]

        # 路径模板 (请根据实际路径修改)
        # 原始预处理数据 (未平滑)
        self.FMRI_TPL = r"../../Pro_proc_data/{sub}/run{run}/{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
        # 事件文件
        self.EVENT_TPL = r"../../data_events/{sub}/{sub}_run-{run}_events.tsv"
        # 头动参数文件
        self.MOTION_TPL = r"../../Pro_proc_data/{sub}/multi_reg/multi_reg_run{run}.txt"
        self.CONFOUNDS_TPL = r"../../Pro_proc_data/{sub}/multi_reg/{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv"

        # 输出目录
        self.OUTPUT_ROOT = Path(r"../../glm_analysis_fwe")

        # =========================
        # 2. 一阶分析参数 (First Level)
        # =========================

        # 对比定义 (Metaphor vs Spatial)
        # 建议分别定义正反向对比，以便做单侧检验
        self.CONTRASTS: Dict[str, Dict[str, float]] = {
            "Metaphor_gt_Spatial": {"yy": 1.0, "kj": -1.0},  # yy=隐喻, kj=空间
            "Spatial_gt_Metaphor": {"kj": 1.0, "yy": -1.0},
        }

        # 平滑核 (用于GLM定位，推荐6mm)
        self.SMOOTHING_FWHM = 6.0

        # GLM 模型参数 (标准化配置)
        self.GLM_PARAMS = {
            "noise_model": "ar1",  # 处理自相关
            "hrf_model": "spm + derivative",  # 推荐：加入时间导数吸收延迟
            "drift_model": "cosine",  # 高通滤波
            "high_pass": 1 / 128.0,
            "standardize": True,  # 必须开启：z-score标准化
            "signal_scaling": False,
            "minimize_memory": False,
        }

        # 噪音回归策略 (Crucial Update)
        # =========================
        self.CONFOUNDS_STRATEGY = {
            # 1. 策略列表：
            # 'motion': 提取头动
            # 'scrub': 提取 motion_outlierXX 列 (处理你截图最后的那些 outlier)
            # 注意：这里去掉了 'high_pass'，因为我们在 GLM_PARAMS 里已经设了 drift_model='cosine'，
            # 避免重复滤波导致统计偏差。
            "strategy": "scrubbing",

            # 2. 头动模型选择：
            # 'basic': 6参数 (仅 trans, rot) -> 较弱，不推荐发顶刊
            # 'power2': 12参数 (6参数 + 导数)
            # 'friston24': 24参数 (6参数 + 导数 + 平方 + 导数的平方) -> 顶刊标准，适合N=29
            "motion": "friston24",

            # 3. Scrubbing (坏点剔除) 设置：
            # 即使 fMRIPrep 已经算好了 outlier 列，这里可以再次收紧阈值
            # 这里的 5 表示：如果第 T 帧坏了，把 T-1, T, T+1, T+2... 共5帧都作为坏点剔除
            "scrub": 0,
            "fd_threshold": 0.5,  # 只有 FD > 0.5mm 的才算大头动
            "std_dvars_threshold": 1.5
        }

        # 随机种子
        self.RANDOM_SEED = 42

        # =========================
        # 3. 二阶分析参数 (Cluster-level FWE)
        # =========================

        # 仅使用置换检验 (Permutation Test)
        self.PERM_CONFIG = {
            # 置换次数：测试用1000，发表用5000-10000
            "n_permutations": 1000,

            # 簇形成阈值 (Cluster-forming threshold)
            # p < 0.001 (uncorrected) 对应的 Z 值约为 3.1
            "cluster_forming_p": 0.01,

            # 显著性水平 (FWE Corrected)
            "target_alpha": 0.05,
            # 是否使用 GPU 加速
            "use_gpu": True,
            # 并行计算的核心数 (-1 使用所有核心)
            "n_jobs": -1
        }

        # 可视化设置
        self.VIZ_CONFIG = {
            "dpi": 300,
            "cmap": "cold_hot",  # 或者是 'RdBu_r'
            "bg_img": "mni",  # 使用标准MNI模板作为背景
        }

        # =========================
        # 4. 系统设置
        # =========================

        # 多进程设置 (用于一阶分析)
        self.USE_MULTIPROCESS = True
        self.MAX_WORKERS = 4  # 根据你的内存调整 (建议：总内存GB / 4)

        # 日志级别
        self.LOG_LEVEL = logging.INFO


# 实例化配置
config = AnalysisConfig()