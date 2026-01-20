#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_config.py
全局配置文件 - 适配分离路径 (影像在 midprep, 头动在 miniprep)
"""
from pathlib import Path


class Config:
    def __init__(self):
        # =========================
        # 1. 核心控制开关
        # =========================
        self.PARALLEL = True  # True=并行跑满CPU
        self.N_JOBS = 10  # 并行进程数

        # 数据空间模式: 'volume' (MNI) 或 'surface' (fsLR)
        self.DATA_SPACE = 'surface'
        self.HEMI = 'L'  # 左脑

        # =========================
        # 2. 路径映射定义
        # =========================
        self.BASE_DATA_DIR = Path("/public/home/dingrui/BIDS_DATA")
        self.OUTPUT_ROOT = self.BASE_DATA_DIR / "lss_results_combined_final"

        # 定义映射逻辑
        self.RUN_MAP = {
            1: {
                "root": self.BASE_DATA_DIR / "emo_20250623/v1",
                "task": "EMO",
                "folder_task": "emo",
                "fmri_dir": "emo_midprep",  # [关键修改] 脑影像在 midprep
                "conf_dir": "emo_miniprep"  # [关键修改] 头动在 miniprep
            },
            2: {
                "root": self.BASE_DATA_DIR / "soc_20250623/v1",
                "task": "SOC",
                "folder_task": "soc",
                "fmri_dir": "soc_midprep",  # [关键修改] 脑影像在 midprep
                "conf_dir": "soc_miniprep"  # [关键修改] 头动在 miniprep
            }
        }

        self.RUNS = [1, 2]

        # 自动获取被试列表 (基于 EMO 文件夹)
        run1_root = self.RUN_MAP[1]["root"]
        if run1_root.exists():
            self.SUBJECTS = sorted([p.name for p in run1_root.glob("sub-*") if p.is_dir()])
        else:
            print(f"Error: 找不到数据根目录 {run1_root}")
            self.SUBJECTS = []

        # =========================
        # 3. 辅助配置
        # =========================
        if self.DATA_SPACE == 'volume':
            self.MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")
        else:
            self.MNI_MASK_PATH = None

        self.TR = 2.0
        self.SMOOTHING_FWHM = None

        # --- 核心路径生成函数 ---

    def get_paths(self, sub, run_idx):
        info = self.RUN_MAP.get(run_idx)
        if not info: raise ValueError(f"Run {run_idx} 配置不存在")

        root = info["root"]
        task = info["task"]
        f_task = info["folder_task"]

        # 分别获取两个目录
        dir_fmri = info["fmri_dir"]  # midprep
        dir_conf = info["conf_dir"]  # miniprep

        # 1. Event 路径
        event_path = (
                root /
                f"{sub}/beh/func_task_{f_task}/run-1/{sub}_task-{task}_events.tsv"
        )

        # 2. fMRI 数据路径 (去 midprep 找)
        if self.DATA_SPACE == 'volume':
            fmri_path = (
                    root /
                    f"{sub}/func/{dir_fmri}/{sub}_task-{task}_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
        else:
            fmri_path = (
                    root /
                    f"{sub}/func/{dir_fmri}/"
                    f"{sub}_hemi-{self.HEMI}_space-fsLR_desc-taskHM6_custom-smooth_bold.shape.gii"
            )

        # 3. Confounds 路径 (去 miniprep 找)
        confounds_path = (
                root /
                f"{sub}/func/{dir_conf}/{sub}_task-{task}_desc-confounds_timeseries.tsv"
        )

        return str(fmri_path), str(event_path), str(confounds_path)


config = Config()