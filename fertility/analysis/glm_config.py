#!/usr/bin/env python3
"""
glm_config.py
fMRI GLM 分析全局配置文件
"""
from pathlib import Path


class AnalysisConfig:
    def __init__(self):
        # =========================
        # 1. 任务控制开关 (Action Flags)
        # =========================
        self.RUN_FIRST_LEVEL = True  # 是否执行一阶分析
        self.RUN_SECOND_LEVEL = False  # 是否执行二阶分析

        # --- 二阶分析方法选择 ---
        # 选项: 'cluster_fwe' (严谨, 慢, 适合大团块) | 'fdr' (灵敏, 快, 适合中小团块)
        self.SECOND_LEVEL_METHOD = 'cluster_fwe'

        # =========================
        # 2. 基础路径与被试设置
        # =========================
        self.SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
        self.RUNS = [4,5,6]

        # 基础目录
        self.BASE_DIR = Path("C:/python_metaphor")
        self.OUTPUT_ROOT = self.BASE_DIR / "glm_analysis_fwe_final"

        # 路径模板
        self.FMRI_TPL = self.BASE_DIR / "Pro_proc_data/{sub}/run{run}/{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
        self.EVENT_TPL = self.BASE_DIR / "data_events/{sub}/{sub}_run-{run}_events.tsv"
        self.CONFOUNDS_TPL = self.BASE_DIR / "Pro_proc_data/{sub}/multi_reg/{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv"

        # =========================
        # 3. 硬件与性能
        # =========================
        self.N_JOBS_1ST_LEVEL = 4  # 一阶分析并行进程数 (根据内存调整)
        self.N_JOBS_PERM = -1  # 二阶分析并行进程数 (-1 = All CPUs)
        self.USE_GPU = True  # 是否尝试使用 GPU 加速

        # =========================
        # 4. 一阶分析参数
        # =========================
        self.TR = 2.0
        self.SMOOTHING_FWHM = 6.0

        self.CONTRASTS = {
            "Metaphor_gt_Spatial": "yy - kj",
            "Spatial_gt_Metaphor": "kj - yy",
        }

        self.DENOISE_STRATEGY = {
            "denoise_strategy": "scrubbing",
            "motion": "friston24",
            "scrub": 0,  # 窗口大小 (0=仅坏点, 5=前后扩展2帧)
            "fd_threshold": 0.5,
            "std_dvars_threshold": 1.5
        }

        # =========================
        # 5. 二阶分析参数
        # =========================
        # 方案 A: Cluster-FWE (置换检验)
        self.PERM_PARAMS = {
            "n_permutations": 5000,  # 建议 5000-10000
            "cluster_forming_p": 0.001,  # Z > 3.1
            "target_alpha": 0.05,  # FWE 显著性水平
        }

        # 方案 B: FDR (参数化检验)
        self.FDR_PARAMS = {
            "alpha": 0.05,        # FDR q < 0.05
            "cluster_min_size": 20 # FDR校正后，依然过滤掉小于20体素的碎片，保持干净
        }

        # 报告参数
        self.REPORT_PARAMS = {
            "min_cluster_voxels": 20,  # 最小汇报团块大小
            "atlas_name": 'cort-maxprob-thr25-2mm'  # Harvard-Oxford Atlas
        }


config = AnalysisConfig()