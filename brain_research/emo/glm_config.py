#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_config.py
全局配置文件 - 适配分离路径 (影像在 midprep, 头动在 miniprep)
"""
import os
from pathlib import Path


class Config:
    def __init__(self, data_space='surface', hemi='L'):
        # =========================
        # 1. 核心控制开关
        # =========================
        self.PARALLEL = True  # True=并行跑满CPU
        self.DEBUG = True  # 调试模式
        self.DEBUG_N_SUBJECTS = 30
        # 数据空间模式: 'volume' (MNI) 或 'surface' (fsLR)
        self.DATA_SPACE = data_space
        self.HEMI = hemi  # 左脑，可选: 'L' (左脑), 'R' (右脑)
        if self.DATA_SPACE == 'volume':
            self.N_JOBS = 4
        else:
            # Surface: 原来是12，建议降到 8 或 6
            # 8 是最稳妥的甜蜜点
            self.N_JOBS = 6
        # =========================
        # 2. 路径映射定义
        # =========================
        # 使用环境变量管理路径
        base_data_dir = os.environ.get("BIDS_DATA_DIR", "/public/home/dingrui/BIDS_DATA")
        self.BASE_DATA_DIR = Path(base_data_dir)

        # 验证基础数据目录
        if not self.BASE_DATA_DIR.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.BASE_DATA_DIR}\n请设置环境变量 BIDS_DATA_DIR 指向正确的路径")

        # 统一输出根目录
        self.OUTPUT_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
        # 确保输出目录存在
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        # 场景标识符
        self.SCENARIO_ID = f"{self.DATA_SPACE}{'_' + self.HEMI if self.DATA_SPACE == 'surface' else ''}"

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
            if not self.SUBJECTS:
                raise ValueError(f"数据根目录存在但未找到被试文件夹: {run1_root}")
        else:
            raise FileNotFoundError(f"找不到数据根目录 {run1_root}")

        # =========================
        # 3. 辅助配置
        # =========================
        if self.DATA_SPACE == 'volume':
            self.MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")
        else:
            self.MNI_MASK_PATH = None

        self.TR = 0.75
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
                    f"{sub}/func/{dir_fmri}/{sub}_task-{task}_space-MNI152NLin2009cAsym_desc-taskFriston_custom-smooth_bold.nii.gz"
            )
        else:
            fmri_path = (
                    root /
                    f"{sub}/func/{dir_fmri}/"
                    f"{sub}_hemi-{self.HEMI}_space-fsLR_desc-taskFriston_custom-smooth_bold.shape.gii"
            )

        # 3. Confounds 路径 (去 miniprep 找)
        confounds_path = (
                root /
                f"{sub}/func/{dir_conf}/{sub}_task-{task}_desc-confounds_timeseries.tsv"
        )

        return str(fmri_path), str(event_path), str(confounds_path)


# 默认配置实例
config = Config()

# 场景配置列表
SCENARIOS = [
    # (data_space, hemi)
    ('surface', 'L'),
    ('surface', 'R'),
    ('volume', ''),  # volume 不需要半球
]

# 获取所有场景的配置实例
def get_scenario_configs():
    configs = []
    for data_space, hemi in SCENARIOS:
        # volume 场景不需要半球参数
        if data_space == 'volume':
            configs.append(Config(data_space='volume', hemi=''))
        else:
            configs.append(Config(data_space=data_space, hemi=hemi))
    return configs