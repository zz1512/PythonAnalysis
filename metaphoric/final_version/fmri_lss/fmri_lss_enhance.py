#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI LSS (Least Squares Separate) 分析优化版本

本脚本实现了高效的fMRI LSS分析流程，包含以下主要特性：
- 内存优化的批处理策略
- 智能缓存管理系统
- 并行和串行处理模式
- 完善的错误处理和资源清理
- 实时性能监控

LSS方法为每个trial单独建模，相比传统GLM能够提供更精确的单trial beta估计。

输入
- BOLD：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/run{run}/*.nii(.gz)`（默认 run3-4）
- events：`${PYTHON_METAPHOR_ROOT}/data_events/{sub}/{sub}_run-{run}_events.tsv`
- confounds：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/multi_reg/*confounds_timeseries.tsv`

输出（`${PYTHON_METAPHOR_ROOT}/lss_betas_final`）
- `lss_betas_final/sub-xx/run-3/`、`run-4/`：单 trial beta maps
-（脚本内部如有）trial-level metadata（供后续 MVPA/RSA/stack_patterns 对齐与追溯）

注意
- 该脚本默认用于学习阶段（run3-4）。前后测（run1-2/5-6）的 LSS 使用 `glm_analysis/run_lss.py`。
- MVPA/RSA 通常不使用平滑数据，因此这里使用的是未平滑 BOLD 路径模板。

Author: Enhanced fMRI Analysis Pipeline
Version: 2.0 (Optimized)
Date: 2024
"""

import os, math, gc, shutil, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置线程数限制以避免过度并行化导致的性能问题
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_brain_mask

# ========== 分析配置参数 ==========

# 被试和实验设置
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]  # 被试编号：sub-01 到 sub-28
RUNS = [3, 4]  # 分析的run编号
TR_FALLBACK = 2.0  # 默认TR值(秒)，当无法从图像头文件推断时使用

# 并行处理配置
N_JOBS_GLM = 2  # GLM模型内部并行线程数
N_JOBS_PARALLEL = 4  # 跨被试并行处理的进程数

# 性能优化开关
USE_PRECOMPUTED_MASK = True  # 是否预计算脑mask以避免重复计算
CLEAN_CACHE_AFTER_RUN = True  # 是否在每个run完成后自动清理缓存
MAX_CACHE_SIZE_GB = 100.0  # 缓存目录最大允许大小(GB)，超过将触发自动清理

# ========== 数据路径模板 ==========
# 预处理后的fMRI数据路径,MVPA不能使用平滑数据
BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
FMRI_TPL = str(BASE_DIR / "Pro_proc_data/{sub}/run{run}/{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii")
# 实验事件文件路径
EVENT_TPL = str(BASE_DIR / "data_events/{sub}/{sub}_run-{run}_events.tsv")
# 头动参数文件路径
MOTION_TPL = str(BASE_DIR / "Pro_proc_data/{sub}/multi_reg/{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv")
# 输出结果根目录
OUTPUT_ROOT = BASE_DIR / "lss_betas_final"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ========== 核心工具函数 ==========

def log(s: str):
    """带时间戳的日志输出函数
    
    Args:
        s (str): 要输出的日志信息
    """
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {s}", flush=True)

def get_dir_size_gb(path: Path) -> float:
    """计算目录总大小
    
    Args:
        path (Path): 目录路径
        
    Returns:
        float: 目录大小(GB)，如果目录不存在返回0.0
    """
    if not path.exists():
        return 0.0
    total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total_size / (1024**3)

def clean_cache_directory(cache_dir: Path, force: bool = False):
    """智能缓存目录清理
    
    根据缓存大小和force参数决定是否清理缓存目录。
    当缓存超过MAX_CACHE_SIZE_GB或force=True时执行清理。
    
    Args:
        cache_dir (Path): 缓存目录路径
        force (bool): 是否强制清理，默认False
    """
    if not cache_dir.exists():
        return
    
    try:
        cache_size = get_dir_size_gb(cache_dir)
        if force or cache_size > MAX_CACHE_SIZE_GB:
            log(f"清理缓存目录: {cache_dir} (大小: {cache_size:.2f}GB)")
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(exist_ok=True)
        else:
            log(f"缓存目录大小正常: {cache_size:.2f}GB")
    except Exception as e:
        log(f"[WARN] 清理缓存失败: {e}")

def create_temp_cache_dir() -> Path:
    """创建临时缓存目录
    
    在系统临时目录中创建以'lss_cache_'为前缀的临时目录，
    用于存储单次分析的缓存文件，避免缓存累积。
    
    Returns:
        Path: 创建的临时目录路径
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="lss_cache_"))
    return temp_dir

def cleanup_temp_cache(cache_dir: Path):
    """安全清理临时缓存目录
    
    仅清理名称包含'lss_cache_'的目录，确保不会误删其他重要目录。
    
    Args:
        cache_dir (Path): 要清理的缓存目录路径
    """
    try:
        if cache_dir.exists() and "lss_cache_" in cache_dir.name:
            shutil.rmtree(cache_dir, ignore_errors=True)
            log(f"已清理临时缓存: {cache_dir.name}")
    except Exception as e:
        log(f"[WARN] 清理临时缓存失败: {e}")

def infer_tr_from_img(img) -> float:
    """从fMRI图像头文件推断TR值
    
    Args:
        img: Nilearn图像对象
        
    Returns:
        float: TR值(秒)，推断失败时返回TR_FALLBACK
    """
    try:
        return float(img.header.get_zooms()[-1])
    except Exception:
        return TR_FALLBACK

def load_events_fast(path: Path) -> Optional[pd.DataFrame]:
    """高效加载和验证事件文件
    
    自动检测列名并进行数据清洗，确保返回标准格式的事件DataFrame。
    
    Args:
        path (Path): 事件文件路径(.tsv格式)
        
    Returns:
        Optional[pd.DataFrame]: 标准化的事件DataFrame，包含onset、duration、trial_type列
                               加载失败时返回None
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        
        # 智能列名映射：自动识别onset、duration、trial_type列
        col_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if "onset" in lower_col: 
                col_map["onset"] = col
            elif "duration" in lower_col: 
                col_map["duration"] = col  
            elif "trial" in lower_col or "condition" in lower_col: 
                col_map["trial_type"] = col
        
        # 验证必需列是否存在
        if len(col_map) != 3:
            return None
            
        ev = df[list(col_map.values())].copy()
        ev.columns = ["onset", "duration", "trial_type"]
        
        # 数据清洗和验证
        ev = ev.dropna(subset=["onset", "duration", "trial_type"])
        ev["onset"] = pd.to_numeric(ev["onset"], errors="coerce")
        ev["duration"] = pd.to_numeric(ev["duration"], errors="coerce")
        ev = ev.dropna(subset=["onset", "duration"])
        
        # 按时间排序并重置索引
        return ev.sort_values("onset").reset_index(drop=True) if not ev.empty else None
        
    except Exception:
        return None

def load_motion_confounds_fast(sub: str, run: int, n_volumes: int) -> Optional[pd.DataFrame]:
    """高效加载头动参数作为混淆变量
    
    从预处理输出的混淆变量文件中提取头动参数，用于GLM分析中的噪声回归。
    
    Args:
        sub (str): 被试编号 (如 'sub-01')
        run (int): run编号
        n_volumes (int): fMRI数据的时间点数，用于验证数据长度匹配
        
    Returns:
        Optional[pd.DataFrame]: 头动参数DataFrame，包含平移和旋转参数
                               加载失败或数据不匹配时返回None
    """
    path = Path(MOTION_TPL.format(sub=sub, run=run))
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, delimiter='\t')
        
        # 选择标准头动参数列（6个自由度）
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        available_cols = [col for col in motion_cols if col in df.columns]
        
        if len(available_cols) >= 3:  # 至少需要3个运动参数才有意义
            confounds = df[available_cols].fillna(0).iloc[:n_volumes]
            # 确保数据长度与fMRI时间点数匹配
            return confounds if len(confounds) == n_volumes else None
    except Exception:
        pass
    return None

def create_lss_events(trial_idx: int, base_events: pd.DataFrame) -> pd.DataFrame:
    """
    为LSS模型创建事件DataFrame，此版本针对多条件进行了优化。

    该函数为指定的单个trial创建事件列表，采用的策略如下：
    1.  **目标Trial**: 您感兴趣的特定trial被单独标记为 'target'。
        这是您希望为其计算beta值的trial。
    2.  **同条件下的其他Trial**: 与目标trial属于相同实验条件的所有其他trial
        被归为一组，标记为 '{condition_name}_other'。
        这有助于将来自同一条件但非目标的trial的方差分离开。
    3.  **其他条件的Trial**: 所有不属于目标trial实验条件的trial将保留其原始的条件标签。
        例如，如果您的实验有'ConditionA'和'ConditionB'两个条件，当目标trial
        来自'ConditionA'时，所有'ConditionB'的trial将继续被标记为'ConditionB'。

    这种方法（Least Squares-Separate, LSS）比将所有其他trial都视为一个'other'
    组（Least Squares-All, LSA）更精确，因为它能更好地对不同来源的神经活动
    进行建模，从而为您关心的'target' trial提供更准确的beta估计。

    Args:
        trial_idx (int): 在原始事件DataFrame中，要作为'target'的trial的索引。
        base_events (pd.DataFrame): 包含一整个run的所有事件的原始DataFrame。

    Returns:
        pd.DataFrame: 一个为LSS分析准备好的、经过特殊格式化的新事件DataFrame。
    """
    events_lss = base_events.copy()
    target_condition = events_lss.loc[trial_idx, 'trial_type']

    # 使用.values可以更快地进行数组操作，并确保类型为object以容纳字符串
    trial_types = events_lss['trial_type'].values.astype(object)

    # 步骤1: 将与目标trial相同条件的其他trial标记为一个组
    same_condition_mask = (trial_types == target_condition)
    trial_types[same_condition_mask] = f'{target_condition}_other'

    # 步骤2: 单独标记出我们的目标trial
    trial_types[trial_idx] = 'target'

    # 将更新后的一系列trial标签应用回DataFrame
    events_lss['trial_type'] = trial_types

    # 按时间排序，这对于fMRI模型很重要
    return events_lss.sort_values("onset").reset_index(drop=True)

def precompute_shared_components(fmri_img, events, confounds, base_kwargs):
    """预计算共享组件以提高处理效率
    
    预先计算脑mask和基础GLM模型，避免在每个trial中重复计算相同的组件。
    
    Args:
        fmri_img: fMRI图像对象
        events: 事件DataFrame
        confounds: 混淆变量DataFrame
        base_kwargs: GLM模型基础参数
        
    Returns:
        tuple: (基础GLM模型, 脑mask图像)
    """
    # 根据配置决定是否预计算脑mask
    if USE_PRECOMPUTED_MASK:
        mask_img = compute_brain_mask(fmri_img)
    else:
        mask_img = None
        
    # 预拟合基础模型以缓存HRF卷积等计算密集型组件
    base_model = FirstLevelModel(mask_img=mask_img, **base_kwargs)
    base_model.fit(fmri_img, events=events, confounds=confounds)
    
    return base_model, mask_img

def process_single_trial(args):
    """处理单个trial的LSS分析
    
    为指定trial执行LSS建模，计算该trial的beta系数并保存结果。
    该函数设计为独立运行，便于并行化处理。
    
    Args:
        args (tuple): 包含以下参数的元组
            - trial_idx (int): trial索引
            - fmri_data: fMRI图像数据
            - base_events (pd.DataFrame): 原始事件DataFrame
            - confounds: 混淆变量
            - base_kwargs (dict): GLM参数
            - out_dir (Path): 输出目录
            - original_condition (str): 原始条件标签
            
    Returns:
        dict or None: 包含trial信息和结果文件路径的字典，失败时返回None
    """
    trial_idx, fmri_data, base_events, confounds, base_kwargs, out_dir, original_condition = args
    
    # 初始化变量以确保异常处理时能正确清理
    model = None
    beta_map = None
    
    try:
        # 步骤1: 为当前trial创建LSS事件设计
        lss_events = create_lss_events(trial_idx, base_events)
        
        # 步骤2: 拟合GLM模型
        model = FirstLevelModel(**base_kwargs)
        model.fit(fmri_data, events=lss_events, confounds=confounds)
        
        # 步骤3: 构建对比向量并计算beta系数
        design_matrix = model.design_matrices_[0]
        if "target" not in design_matrix.columns:
            return None  # 目标trial未在设计矩阵中找到
            
        # 创建对比向量：仅对'target'条件赋值1，其他为0
        contrast_vec = np.zeros(len(design_matrix.columns))
        contrast_vec[list(design_matrix.columns).index("target")] = 1.0
        
        # 计算效应量（beta系数）
        beta_map = model.compute_contrast(contrast_vec, output_type="effect_size")
        
        # 步骤4: 保存beta图像
        pic_num = base_events.iloc[trial_idx].get("pic_num", trial_idx)
        unique_label = f"{original_condition}_{int(pic_num)}" if pd.notna(pic_num) else f"{original_condition}_{trial_idx}"
        beta_filename = out_dir / f"beta_trial-{trial_idx:03d}_{unique_label}.nii.gz"
        beta_map.to_filename(beta_filename)
        
        # ??5: ??????
        result = {
            "trial_index": trial_idx,
            "condition": original_condition,
            "original_condition": original_condition,
            "unique_label": unique_label,
            "pic_num": int(pic_num) if pd.notna(pic_num) else None,
            "beta_file": beta_filename.name,
            "onset": base_events.iloc[trial_idx]["onset"],
            "duration": base_events.iloc[trial_idx]["duration"]
        }
        
        del model, beta_map
        gc.collect()
        
        return result
        
    except Exception as e:
        # 异常处理：确保内存得到清理
        if model is not None:
            del model
        if beta_map is not None:
            del beta_map
        gc.collect()
        return None

# ========== 主要分析流程 ==========

def process_one_run_optimized(sub: str, run: int):
    """执行单个被试单个run的LSS分析
    
    这是主要的分析函数，负责协调整个LSS分析流程，包括数据加载、
    参数设置、批处理执行和结果保存。
    
    Args:
        sub (str): 被试编号 (如 'sub-01')
        run (int): run编号
    """
    # 构建数据文件路径
    fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
    events_path = Path(EVENT_TPL.format(sub=sub, run=run))
    
    # 检查必需文件是否存在
    if not fmri_path.exists():
        log(f"[SKIP] fMRI文件缺失: {sub}|run-{run}")
        return

    # === 数据加载阶段 ===
    log(f"[{sub}|run-{run}] 开始加载数据...")
    try:
        # 加载fMRI图像并提取基本信息
        fmri_img = image.load_img(str(fmri_path))
        tr = infer_tr_from_img(fmri_img)
        n_volumes = fmri_img.shape[-1]
        
        # 预加载图像数据到内存以减少后续IO开销
        fmri_data = fmri_img.get_fdata()
        fmri_img = image.new_img_like(fmri_img, fmri_data)
        
    except Exception as e:
        log(f"[ERROR] 加载fMRI数据失败: {e}")
        return

    # 加载实验事件信息
    events = load_events_fast(events_path)
    if events is None or events.empty:
        log(f"[SKIP] 事件文件无效或为空: {sub}|run-{run}")
        return

    # 加载头动参数作为混淆变量
    confounds = load_motion_confounds_fast(sub, run, n_volumes)
    
    # === 环境准备阶段 ===
    # 创建输出目录
    out_dir = OUTPUT_ROOT / sub / f"run-{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时缓存目录以避免缓存累积
    cache_dir = create_temp_cache_dir()
    log(f"[{sub}|run-{run}] 使用临时缓存: {cache_dir.name}")

    # === GLM参数配置 ===
    # 优化的GLM参数：平衡计算速度和内存使用
    base_kwargs = dict(
        t_r=tr,                    # 重复时间
        slice_time_ref=0.5,        # 切片时间参考点
        noise_model="ar1",         # AR(1)噪声模型
        hrf_model="spm",           # SPM标准HRF模型
        drift_model="cosine",      # 余弦漂移模型
        high_pass=0.01,            # 高通滤波截止频率
        standardize=True,          # 标准化设计矩阵
        signal_scaling=False,      # 不进行信号缩放
        minimize_memory=True,      # 启用内存优化
        memory=str(cache_dir),     # 缓存目录
        memory_level=1,            # 较低的缓存级别以减少磁盘使用
        n_jobs=1,                  # 单线程避免嵌套并行
        verbose=0                  # 静默模式
    )

    total_trials = len(events)
    log(f"[{sub}|run-{run}] 开始LSS分析，共{total_trials}个trial...")

    # === 批处理执行阶段 ===
    # 采用小批量处理策略以平衡内存使用和处理效率
    trial_info = []
    batch_size = min(10, total_trials)  # 限制批大小以控制内存峰值
    
    for batch_start in range(0, total_trials, batch_size):
        batch_end = min(batch_start + batch_size, total_trials)
        batch_trials = list(range(batch_start, batch_end))
        
        # 实时监控缓存使用情况
        cache_size = get_dir_size_gb(cache_dir)
        if cache_size > MAX_CACHE_SIZE_GB:
            log(f"[{sub}|run-{run}] 缓存超限({cache_size:.2f}GB)，执行清理...")
            clean_cache_directory(cache_dir, force=True)
        
        # 为当前批次准备参数
        batch_args = []
        for trial_idx in batch_trials:
            original_condition = events.iloc[trial_idx]["trial_type"]
            batch_args.append((
                trial_idx, fmri_img, events, confounds, 
                base_kwargs, out_dir, original_condition
            ))
        
        # 串行处理批内trial（避免并行导致的内存爆炸）
        for args in batch_args:
            result = process_single_trial(args)
            if result is not None:
                result.update({"subject": sub, "run": run})
                trial_info.append(result)
        
        # 批次进度报告
        completed = len(trial_info)
        log(f"[{sub}|run-{run}] 批次完成: {completed}/{total_trials} (缓存: {get_dir_size_gb(cache_dir):.2f}GB)")
        
        # 批次间内存清理
        gc.collect()
        
        # 定期缓存清理：每5个批次清理一次
        if (batch_start // batch_size + 1) % 5 == 0:
            clean_cache_directory(cache_dir, force=False)

    # === 结果保存阶段 ===
    if trial_info:
        # 保存trial级别的分析结果信息
        trial_df = pd.DataFrame(trial_info)
        trial_df.to_csv(out_dir / "trial_info.csv", index=False, encoding="utf-8-sig")
        
        # 保存脑mask（仅在不存在时创建）
        mask_path = out_dir / "mask.nii.gz"
        if not mask_path.exists():
            try:
                mask_img = compute_brain_mask(fmri_img)
                mask_img.to_filename(mask_path)
                del mask_img  # 立即清理
            except Exception as e:
                log(f"[WARN] 无法保存脑mask: {e}")

    # === 最终清理阶段 ===
    # 记录最终缓存使用情况
    final_cache_size = get_dir_size_gb(cache_dir)
    log(f"[{sub}|run-{run}] 最终缓存大小: {final_cache_size:.2f}GB")
    
    # 根据配置清理临时缓存目录
    if CLEAN_CACHE_AFTER_RUN:
        cleanup_temp_cache(cache_dir)
    
    # 清理主要内存对象
    del fmri_img, fmri_data
    if 'trial_df' in locals():
        del trial_df
    gc.collect()  # 强制垃圾回收
    
    # 输出最终统计信息
    success_count = len(trial_info)
    log(f"[{sub}|run-{run}] LSS分析完成: {success_count}/{total_trials} 成功，资源已清理")

def process_subject_run(args):
    """
    单个被试和run的处理包装函数
    
    提供异常处理和资源清理的安全包装，确保即使处理失败也能正确清理资源。
    
    Args:
        args (tuple): 包含被试编号和run编号的元组 (sub, run)
    
    Returns:
        str: 处理结果字符串，包含成功或失败信息
    """
    sub, run = args
    try:
        process_one_run_optimized(sub, run)
        return f"{sub}|run-{run}: 成功"
    except Exception as e:
        # 异常时也要清理可能的临时缓存
        try:
            temp_dirs = [d for d in Path("/tmp").glob("lss_cache_*") if d.is_dir()]
            for temp_dir in temp_dirs:
                cleanup_temp_cache(temp_dir)
        except:
            pass
        gc.collect()
        return f"{sub}|run-{run}: 失败 - {e}"

def cleanup_all_temp_caches():
    """
    清理所有LSS相关的临时缓存目录
    
    扫描/tmp目录下所有以'lss_cache_'开头的目录并进行清理。
    通常在程序结束时调用以确保不留下临时文件。
    
    Note:
        该函数会静默处理清理过程中的异常，不会影响主程序流程。
    """
    try:
        temp_dirs = [d for d in Path("/tmp").glob("lss_cache_*") if d.is_dir()]
        cleaned_count = 0
        for temp_dir in temp_dirs:
            cleanup_temp_cache(temp_dir)
            cleaned_count += 1
        if cleaned_count > 0:
            log(f"清理了 {cleaned_count} 个临时缓存目录")
    except Exception as e:
        log(f"[WARN] 清理临时缓存时出错: {e}")

def main_optimized():
    """
    主函数：基于joblib的并行LSS分析处理
    
    实现多被试、多run的LSS分析，支持并行和串行两种处理模式。
    包含完整的资源管理和异常处理机制。
    
    Features:
        - 自动任务分配和负载均衡
        - 智能并行/串行模式切换
        - 完整的进度监控和日志记录
        - 异常安全的资源清理
        - 详细的处理结果保存
    """
    log("开始高效LSS分析流程...")
    
    try:
        # === 任务准备阶段 ===
        # 生成所有被试和run的组合任务
        tasks = [(sub, run) for sub in SUBJECTS for run in RUNS]
        total_tasks = len(tasks)
        log(f"总任务数: {total_tasks} (被试: {len(SUBJECTS)}, runs: {len(RUNS)})")
        
        # === 执行阶段 ===
        # 使用joblib进行并行处理
        from joblib import Parallel, delayed
        log(f"启用并行处理模式，进程数: {N_JOBS_PARALLEL}")
        results = Parallel(n_jobs=N_JOBS_PARALLEL)(delayed(process_subject_run)(task) for task in tasks)

        # === 结果统计阶段 ===
        success_count = sum(1 for r in results if "成功" in r)
        log(f"\n{'='*60}")
        log(f"LSS分析完成: {success_count}/{total_tasks} 任务成功")
        log(f"{'='*60}")
        
        # 保存详细处理结果到CSV文件
        result_df = pd.DataFrame({"result": results})
        result_df.to_csv(OUTPUT_ROOT / "processing_results.csv", index=False, encoding="utf-8-sig")
        
    finally:
        # 确保清理所有临时缓存（异常安全）
        cleanup_all_temp_caches()
        gc.collect()
        log("所有临时缓存已清理完毕")

# === 程序入口点 ===
if __name__ == "__main__":
    # 根据需求选择运行模式
    USE_PARALLEL = True  # 设置为False使用串行模式
    
    try:
        if USE_PARALLEL and N_JOBS_PARALLEL > 1:
            # 使用优化的并行处理模式
            main_optimized()
        else:
            # === 串行处理模式（备用方案） ===
            # 适用于调试或资源受限的环境
            log("使用串行模式...")
            
            success_count = 0
            total_tasks = len(SUBJECTS) * len(RUNS)
            
            # 逐个处理每个被试的每个run
            for sub in SUBJECTS:
                log(f"\n{'='*50}")
                log(f"处理被试: {sub}")
                log(f"{'='*50}")
                for run in RUNS:
                    try:
                        process_one_run_optimized(sub, run)
                        success_count += 1
                        log(f"进度: {success_count}/{total_tasks}")
                    except Exception as e:
                        log(f"[ERROR] {sub}|run-{run} 处理失败: {e}")
                        # 清理可能的临时缓存
                        cleanup_all_temp_caches()
            
            # 串行模式结果统计
            log(f"=== LSS分析完成 ===")
            log(f"成功: {success_count}/{total_tasks}")
            
    finally:
        # === 程序结束清理 ===
        # 确保所有临时资源都被正确清理
        cleanup_all_temp_caches()
        gc.collect()
        log("程序结束，所有资源已清理")
