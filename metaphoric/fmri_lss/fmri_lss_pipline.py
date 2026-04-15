#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI LSS（Least Squares Separate）分析流水线

功能描述:
    实现LSS分析方法，为每个试次单独建立GLM模型，获取试次级的激活估计。
    LSS方法通过将目标试次与其他试次分离建模，减少试次间的相关性影响。

分析原理:
    LSS（Least Squares Separate）是一种单试次分析方法：
    1. 对每个目标试次，建立包含该试次和所有其他试次的GLM模型
    2. 目标试次单独作为一个回归子，其他试次合并为另一个回归子
    3. 提取目标试次的beta系数作为该试次的激活强度估计
    4. 重复此过程直到所有试次都被分析

主要功能:
    1. 加载fMRI数据、事件文件和头动参数
    2. 为每个试次构建LSS设计矩阵
    3. 拟合GLM模型并提取试次级beta图像
    4. 保存试次级激活图像供后续分析使用

输入数据:
    - fMRI预处理数据（已平滑）
    - 事件时间文件（onset, duration, trial_type）
    - 头动参数文件（用作协变量）

输出结果:
    - 每个试次的beta系数图像
    - 试次信息汇总表格

配置参数:
    - 被试范围：sub-01 到 sub-28
    - 分析run：run-3
    - GLM并行度：可调整以适应计算资源

注意事项:
    - LSS分析计算量较大，建议使用并行处理
    - 需要足够的内存来处理多个GLM模型
    - 输出文件较多，注意存储空间

作者: 研究团队
版本: 1.0
日期: 2024

参考文献:
    Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012).
    Deconvolving BOLD activation in event-related designs for multivoxel pattern classification analyses.
    NeuroImage, 59(3), 2636-2643.
"""

# 多线程环境优化：限制各数值库的线程数以避免过度并行
import os, math, gc
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel

# === LSS分析配置参数 ===

# 被试和run设置
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]  # 被试列表：sub-01 到 sub-28
RUNS = [3]                                          # 要分析的run列表
TR_FALLBACK = 2.0                                   # TR默认值（当无法从图像头文件读取时）

# GLM分析参数
N_JOBS_GLM = 2                                      # GLM内部并行线程数（CPU资源紧张时可设为1）
LSS_COLLAPSE_BY_COND = False                        # LSS其他试次处理方式：
                                                    # True: 按条件类型分别建模
                                                    # False: 所有其他试次合并为一个回归子

# === 文件路径模板 ===
# fMRI数据路径（使用平滑后的数据）
FMRI_TPL = r"../../Pro_proc_data/{sub}/run{run}/smooth{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"

# 事件文件路径
EVENT_TPL = r"../../data_events/{sub}/{sub}_run-{run}_events.tsv"

# 头动参数文件路径
MOTION_TPL = r"../../Pro_proc_data/{sub}/multi_reg/{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv"

# 输出目录
OUTPUT_ROOT = Path(r"../../learn_LSS")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# === 工具函数 ===

def log(s: str):
    """
    日志输出函数
    
    Args:
        s (str): 要输出的日志信息
    """
    print(s, flush=True)

def infer_tr_from_img(img) -> float:
    """
    从fMRI图像头文件推断TR值
    
    Args:
        img: nibabel图像对象
    
    Returns:
        float: TR值（秒），如果无法读取则返回默认值
    
    Note:
        TR通常存储在图像头文件的第4维（时间维）的体素尺寸中
    """
    try:
        tr = float(img.header.get_zooms()[-1])
        return tr if math.isfinite(tr) and tr > 0 else TR_FALLBACK
    except Exception:
        return TR_FALLBACK

def load_events(path: Path) -> Optional[pd.DataFrame]:
    """
    加载事件文件
    
    Args:
        path (Path): 事件文件路径
    
    Returns:
        Optional[pd.DataFrame]: 事件数据框，包含onset、duration、trial_type列
                               如果文件不存在或格式错误则返回None
    
    Note:
        事件文件必须包含以下列（不区分大小写）：
        - onset: 事件开始时间（秒）
        - duration: 事件持续时间（秒）
        - trial_type: 试次类型标识
    """
    if not path.exists():
        log(f"[WARN] 事件文件缺失: {path}")
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        # 不区分大小写的列名映射
        lower = {c.lower(): c for c in df.columns}
        onset = lower.get("onset")
        dur = lower.get("duration")
        ttype = lower.get("trial_type")
        
        if onset is None or dur is None or ttype is None:
            log(f"[WARN] 事件文件缺少必要列: {path}")
            return None
        ev = df[[onset, dur, ttype]].copy()
        ev.columns = ["onset", "duration", "trial_type"]
        ev = ev.dropna(subset=["onset", "duration", "trial_type"])
        ev["onset"] = ev["onset"].astype(float)
        ev["duration"] = ev["duration"].astype(float)
        ev["trial_type"] = ev["trial_type"].astype(str)
        # 保留原condition便于LSS按条件折叠
        ev["condition"] = ev["trial_type"]
        ev = ev.sort_values("onset").reset_index(drop=True)
        return ev
    except Exception as e:
        log(f"[WARN] read events failed: {path} | {e}")
        return None

def load_motion_confounds(sub: str, run: int) -> Optional[pd.DataFrame]:
    """
    加载头动参数和生理噪声协变量
    
    Args:
        sub (str): 被试ID（如'sub-01'）
        run (int): run编号
    
    Returns:
        Optional[pd.DataFrame]: 协变量数据框，包含头动参数和生理信号
                               如果文件不存在或读取失败则返回None
    
    Note:
        选择的协变量包括：
        - 生理信号：白质信号、全脑信号
        - 头动指标：帧间位移（FD）
        - 头动参数：平移（x,y,z）和旋转（x,y,z）
        缺失值用0填充
    """
    path = Path(MOTION_TPL.format(sub=sub, run=run))
    if not path.exists():
        log(f"[INFO] 协变量文件缺失: {path}")
        return None
    try:
        df = pd.read_csv(path, delimiter='\t')
        # 选择关键的协变量：生理信号 + 头动参数
        confound_cols = [
            'white_matter',           # 白质信号
            'global_signal',          # 全脑信号
            'framewise_displacement', # 帧间位移
            'trans_x', 'trans_y', 'trans_z',  # 平移参数
            'rot_x', 'rot_y', 'rot_z'         # 旋转参数
        ]
        confound_glm = df[confound_cols].replace(np.nan, 0)
        return confound_glm
    except Exception as e:
        log(f"[WARN] 协变量文件读取失败: {path} | {e}")
        return None

# === LSS分析核心函数 ===

def iter_events_LSS(events: pd.DataFrame):
    """
    LSS迭代器：为每个试次生成LSS设计矩阵
    
    LSS（Least Squares Separate）的核心思想是为每个目标试次单独建立GLM模型。
    对于每个目标试次，生成一个包含该试次和所有其他试次的设计矩阵，
    其中目标试次单独作为一个回归子，其他试次根据配置合并或分组。
    
    Args:
        events (pd.DataFrame): 事件数据框，必须包含以下列：
                              - onset: 事件开始时间（秒）
                              - duration: 事件持续时间（秒）
                              - trial_type: 试次类型标识
                              - condition: 条件标识（用于分组）
    
    Yields:
        tuple: (trial_idx, events_i, trial_col_name)
            - trial_idx (int): 目标试次的索引（从1开始）
            - events_i (pd.DataFrame): LSS设计矩阵用的事件列表
            - trial_col_name (str): 目标试次在设计矩阵中的列名
    
    Note:
        根据LSS_COLLAPSE_BY_COND配置：
        - True: 其他试次按条件类型分别建模（如other_yy, other_kj）
        - False: 所有其他试次合并为一个回归子（other）
    
    Example:
        假设有3个试次：yy1, kj1, yy2
        当处理yy1时，生成的LSS事件包含：
        - trial_tr001 (yy1)
        - other (kj1 + yy2) 或 other_yy (yy2) + other_kj (kj1)
    """
    base = events.copy()
    n = len(base)
    for i in range(n):
        this = base.iloc[[i]][["onset", "duration"]].copy()
        this["trial_type"] = f"trial_tr{str(i+1).zfill(3)}"

        others = base.drop(base.index[i])[["onset", "duration", "condition"]].copy()
        if LSS_COLLAPSE_BY_COND:
            others["trial_type"] = "other_" + others["condition"].astype(str)
        else:
            others["trial_type"] = "other"

        ev_i = pd.concat([this, others[["onset","duration","trial_type"]]], ignore_index=True)
        yield (i+1, ev_i, this["trial_type"].iloc[0])  # trial_idx, events_i, trial_col_name

# ========== 主流程 ==========
def process_one_run(sub: str, run: int):
    """
    处理单个被试的单个run的LSS分析
    
    对指定被试的指定run执行完整的LSS（Least Squares Separate）分析流程：
    1. 加载fMRI数据、事件文件和运动协变量
    2. 为每个试次构建独立的GLM模型
    3. 提取每个试次的beta值和t值图像
    4. 保存分析结果到指定目录
    
    Args:
        sub (str): 被试ID，格式如'sub-01'
        run (int): run编号，如3
    
    Returns:
        None
    
    Output Files:
        - {OUTPUT_ROOT}/{sub}/run-{run}_LSS/mask.nii.gz: 分析掩膜
        - {OUTPUT_ROOT}/{sub}/run-{run}_LSS/trial_{tidx:03d}_beta.nii.gz: 试次beta图像
        - {OUTPUT_ROOT}/{sub}/run-{run}_LSS/trial_{tidx:03d}_tstat.nii.gz: 试次t统计图像
        - {OUTPUT_ROOT}/{sub}/run-{run}_LSS/trials_summary.csv: 试次摘要信息
    
    Note:
        - 使用与SPM对齐的GLM参数设置
        - 自动推断TR值，失败时使用TR_FALLBACK
        - 支持运动协变量回归
        - 内存优化：及时清理中间变量
    
    Example:
        >>> process_one_run('sub-01', 3)
        # 处理sub-01的run-3数据，输出到../../learn_LSS/sub-01/run-3_LSS/
    """
    # 构建文件路径
    fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
    events_path = Path(EVENT_TPL.format(sub=sub, run=run))
    
    # 检查fMRI文件是否存在
    if not fmri_path.exists():
        log(f"[WARN] fMRI文件缺失: {fmri_path}"); return

    # 加载事件文件
    ev = load_events(events_path)
    if ev is None or ev.empty:
        log(f"[WARN] 事件文件为空或损坏: {events_path}"); return

    # 加载fMRI数据和相关参数
    log(f"[{sub}|run-{run}] 加载fMRI数据...")
    fmri_img = image.load_img(str(fmri_path))
    tr = infer_tr_from_img(fmri_img)  # 自动推断TR值
    conf = load_motion_confounds(sub, run)  # 加载运动协变量

    # 创建输出目录
    out_dir = OUTPUT_ROOT / sub / f"run-{run}_LSS"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一的GLM参数设置（与SPM尽量对齐）
    base_kwargs = dict(
        t_r=tr,                    # 重复时间
        slice_time_ref=0.5,        # 切片时间参考点（中间切片）
        noise_model="ar1",         # AR(1)噪声模型
        hrf_model="spm",           # SPM标准HRF模型
        drift_model="cosine",      # 余弦漂移模型
        high_pass=1/128.0,         # 高通滤波截止频率
        standardize=False,         # 不标准化信号
        signal_scaling=False,      # 不缩放信号
        minimize_memory=True,      # 内存优化
        n_jobs=N_JOBS_GLM,         # 并行线程数
        verbose=0                  # 静默模式
    )

    # 初始化LSS分析变量
    trial_rows = []            # 存储每个试次的摘要信息
    saved_mask = False         # 掩膜保存标志
    total = len(ev)            # 总试次数
    log(f"[{sub}|run-{run}] 开始LSS分析，试次数={total}...")

    for tidx, ev_i, trial_col in iter_events_LSS(ev):
        # 为当前试次拟合独立的GLM模型
        model_i = FirstLevelModel(**base_kwargs).fit(
            fmri_img, events=ev_i, confounds=conf
        )
        dm_i = model_i.design_matrices_[0]  # 获取设计矩阵
        cols = list(dm_i.columns)           # 设计矩阵列名

        # 构建对比向量：只对目标试次的回归子设为1，其他为0
        w = np.zeros(len(cols))
        w[cols.index(trial_col)] = 1.0
        
        # 计算目标试次的beta值和t统计值
        beta_i = model_i.compute_contrast(w, output_type="effect_size")  # beta系数图像
        t_i = model_i.compute_contrast(w, output_type="stat")            # t统计图像

        # 保存beta和t统计图像
        beta_i.to_filename(out_dir / f"beta_trial_{tidx:03d}.nii.gz")
        t_i.to_filename(out_dir / f"spmT_trial_{tidx:03d}.nii.gz")

        # 保存分析掩膜（只需保存一次）
        if (not saved_mask) and hasattr(model_i, "masker_") and getattr(model_i.masker_, "mask_img_", None) is not None:
            model_i.masker_.mask_img_.to_filename(out_dir / "mask.nii.gz")
            saved_mask = True

        trial_rows.append({
            "subject": sub, "run": run,
            "trial_index": tidx,
            "trial_col": trial_col,
            "trial_condition": ev.loc[tidx-1, "condition"]
        })

        # 释放
        del model_i, dm_i, beta_i, t_i
        gc.collect()

        if tidx % 10 == 0 or tidx == total:
            log(f"[{sub}|run-{run}] done {tidx}/{total}")

    pd.DataFrame(trial_rows).to_csv(out_dir / "trial_map.csv", index=False, encoding="utf-8-sig")
    del fmri_img
    gc.collect()
    log(f"[{sub}|run-{run}] LSS finished → {out_dir}")

def main():
    """
    LSS分析主程序入口
    
    批量处理所有配置的被试和run，执行LSS（Least Squares Separate）分析。
    对每个被试的每个run，调用process_one_run函数进行独立的LSS分析。
    
    处理流程：
    1. 遍历SUBJECTS列表中的所有被试
    2. 对每个被试遍历RUNS列表中的所有run
    3. 调用process_one_run执行LSS分析
    4. 异常处理：记录错误但继续处理其他数据
    
    配置参数：
        - SUBJECTS: 被试列表，默认sub-01到sub-28
        - RUNS: run列表，默认[3]
        - 其他GLM和路径参数见文件顶部配置区域
    
    输出：
        每个被试每个run的LSS结果保存在：
        {OUTPUT_ROOT}/{sub}/run-{run}_LSS/
    
    Note:
        使用异常处理确保单个被试/run的错误不会中断整个批处理流程
    """
    for sub in SUBJECTS:
        log(f"\n===== {sub} =====")
        for run in RUNS:
            try:
                process_one_run(sub, run)
            except Exception as e:
                log(f"[ERROR] {sub}|run-{run} crashed: {e}")

if __name__ == "__main__":
    main()
