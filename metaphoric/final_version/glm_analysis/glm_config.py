#!/usr/bin/env python3
"""
glm_config.py

用途
- 统一管理 fMRI GLM（一阶 + 二阶）分析的路径、被试/Run 列表、对比、去噪与多重比较控制参数。

论文意义（这一层回答什么问题）
- 这是证据链中“定位哪里发生了条件差异”的一层：在学习阶段（默认 run3-4）比较 `yy` vs `kj`，
  得到组水平显著簇，并为下游 ROI 级表征分析（MVPA/RSA/gPPI 等）提供 ROI mask 的来源与命名约定。

方法（做了什么）
- 一阶：Nilearn `FirstLevelModel` 的 GLM（含平滑、confounds、可选 scrubbing）。
- 二阶：参数化（FDR）或非参数置换（cluster-FWE）的组水平推断。

输入
- `PYTHON_METAPHOR_ROOT` 指向数据根目录（默认 `E:/python_metaphor`）。
- BOLD / events / confounds 路径模板由 `FMRI_TPL` / `EVENT_TPL` / `CONFOUNDS_TPL` 定义。

输出
- `OUTPUT_ROOT`：二阶结果目录（阈值化统计图、簇报告、ROI mask 等；具体文件名由运行脚本决定）。
- `LSS_OUTPUT_ROOT`：LSS beta 输出目录（供 RSA/MVPA 堆叠与对齐）。

关键参数为何这样设（常见理由）
- `RUNS=[3,4]`：默认聚焦学习阶段；若要做前后测/其它阶段，需同步调整事件与对比解释。
- `SMOOTHING_FWHM=6.0`：适合 GLM 定位层面的 SNR 与空间不确定性折中；但 RSA/LSS 通常不应平滑。
- `DENOISE_STRATEGY`：
  - `friston24` 以较稳健方式吸收头动及其导数/二次项；
  - `scrubbing`（motion_outlier 扩窗）用于降低尖峰头动导致的假阳性。
- `PERM_PARAMS.cluster_forming_p=0.001`：更严格的 cluster-forming 门槛，降低“大片漂移噪声”形成假簇的风险。

结果解读（你应该看什么）
- 二阶输出的阈值化 group map 与簇报告：回答“在哪些脑区，yy 与 kj 在学习阶段呈现可重复差异”。
- ROI mask：用于下游 ROI-level 分析时，确保每一步都在同一套空间定义下对齐。

常见坑
- 路径模板与实际文件名不一致（尤其是 `.nii` vs `.nii.gz`、run 编号、space/res 字段）。
- 把 GLM 的平滑参数沿用到 LSS/RSA（会损害体素模式信息，甚至改变效应方向）。
- scrubbing 过强导致有效时间点不足（模型奇异/拟合失败），或不同被试删点比例差异过大造成组水平偏差。
- 用同一份数据同时“定义 ROI”又“在 ROI 内检验同一对比”的循环分析风险；若要推断严格结论，
  建议使用独立数据、留一法或预注册 ROI。
"""
import os
from pathlib import Path


class AnalysisConfig:
    def __init__(self):
        # =========================
        # 1. 任务控制开关 (Action Flags)
        # =========================
        self.RUN_FIRST_LEVEL = False  # 是否执行一阶分析
        self.RUN_SECOND_LEVEL = True  # 是否执行二阶分析

        # --- 二阶分析方法选择 ---
        # 选项: 'cluster_fwe' (严谨, 慢, 适合大团块) | 'fdr' (灵敏, 快, 适合中小团块)
        self.SECOND_LEVEL_METHOD = 'cluster_fwe'

        # =========================
        # 2. 基础路径与被试设置
        # =========================
        self.SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
        self.RUNS = [3, 4]

        # 基础目录
        # Allow overriding data root for portability/reproducibility.
        # Example (macOS/Linux): export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor
        # Example (Windows PowerShell): $env:PYTHON_METAPHOR_ROOT="E:/python_metaphor"
        self.BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
        self.OUTPUT_ROOT = self.BASE_DIR / "glm_analysis_fwe_final"
        self.LSS_OUTPUT_ROOT = self.BASE_DIR / "lss_betas_final"

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
