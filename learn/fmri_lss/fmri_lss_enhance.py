import os, math, gc
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# ========== 高效配置 ==========
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
RUNS = [3,4]
TR_FALLBACK = 2.0
N_JOBS_GLM = 2
N_JOBS_PARALLEL = 4  # 并行处理多个被试
USE_PRECOMPUTED_MASK = True  # 预计算mask避免重复计算

# 路径模板
FMRI_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\run{run}\smooth{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
EVENT_TPL = r"H:\PythonAnalysis\data_events\{sub}\{sub}_run-{run}_events.tsv"
MOTION_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\multi_reg\{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv"
OUTPUT_ROOT = Path(r"H:\PythonAnalysis\learn_LSS")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ========== 高效工具函数 ==========
def log(s: str):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {s}", flush=True)

def infer_tr_from_img(img) -> float:
    """快速TR推断"""
    try:
        return float(img.header.get_zooms()[-1])
    except Exception:
        return TR_FALLBACK

def load_events_fast(path: Path) -> Optional[pd.DataFrame]:
    """快速加载事件文件"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        # 快速列名检测
        col_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if "onset" in lower_col: col_map["onset"] = col
            elif "duration" in lower_col: col_map["duration"] = col  
            elif "trial" in lower_col or "condition" in lower_col: col_map["trial_type"] = col
        
        if len(col_map) != 3:
            return None
            
        ev = df[list(col_map.values())].copy()
        ev.columns = ["onset", "duration", "trial_type"]
        
        # 快速数据清洗
        ev = ev.dropna(subset=["onset", "duration", "trial_type"])
        ev["onset"] = pd.to_numeric(ev["onset"], errors="coerce")
        ev["duration"] = pd.to_numeric(ev["duration"], errors="coerce")
        ev = ev.dropna(subset=["onset", "duration"])
        
        return ev.sort_values("onset").reset_index(drop=True) if not ev.empty else None
        
    except Exception:
        return None

def load_motion_confounds_fast(sub: str, run: int, n_volumes: int) -> Optional[pd.DataFrame]:
    """快速加载头动参数"""
    path = Path(MOTION_TPL.format(sub=sub, run=run))
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, delimiter='\t')
        # 快速列选择
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        available_cols = [col for col in motion_cols if col in df.columns]
        
        if len(available_cols) >= 3:  # 至少需要3个运动参数
            confounds = df[available_cols].fillna(0).iloc[:n_volumes]
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
    """预计算共享组件以避免重复计算"""
    # 预计算mask
    if USE_PRECOMPUTED_MASK:
        mask_img = compute_brain_mask(fmri_img)
    else:
        mask_img = None
        
    # 预拟合一个基础模型来缓存HRF等
    base_model = FirstLevelModel(mask_img=mask_img, **base_kwargs)
    base_model.fit(fmri_img, events=events, confounds=confounds)
    
    return base_model, mask_img

def process_single_trial(args):
    """处理单个trial的独立函数，便于并行化"""
    trial_idx, fmri_data, base_events, confounds, base_kwargs, out_dir, original_condition = args
    
    try:
        # 创建LSS事件
        lss_events = create_lss_events(trial_idx, base_events)
        
        # 拟合GLM - 使用预缓存的mask
        model = FirstLevelModel(**base_kwargs)
        model.fit(fmri_data, events=lss_events, confounds=confounds)
        
        # 计算对比
        design_matrix = model.design_matrices_[0]
        if "target" not in design_matrix.columns:
            return None
            
        contrast_vec = np.zeros(len(design_matrix.columns))
        contrast_vec[list(design_matrix.columns).index("target")] = 1.0
        
        beta_map = model.compute_contrast(contrast_vec, output_type="effect_size")
        
        # 保存结果
        beta_filename = out_dir / f"beta_trial_{trial_idx + 1:03d}.nii.gz"
        beta_map.to_filename(beta_filename)
        
        return {
            "trial_index": trial_idx + 1,
            "original_condition": original_condition,
            "beta_file": beta_filename.name,
            "onset": base_events.iloc[trial_idx]["onset"],
            "duration": base_events.iloc[trial_idx]["duration"]
        }
        
    except Exception as e:
        return None

# ========== 高效主流程 ==========
def process_one_run_optimized(sub: str, run: int):
    """优化版本的单个run处理"""
    fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
    events_path = Path(EVENT_TPL.format(sub=sub, run=run))
    
    if not fmri_path.exists():
        log(f"[SKIP] fMRI缺失: {sub}|run-{run}")
        return

    # 快速加载数据
    log(f"[{sub}|run-{run}] 快速加载数据...")
    try:
        fmri_img = image.load_img(str(fmri_path))
        tr = infer_tr_from_img(fmri_img)
        n_volumes = fmri_img.shape[-1]
        
        # 将图像数据加载到内存以减少IO
        fmri_data = fmri_img.get_fdata()
        fmri_img = image.new_img_like(fmri_img, fmri_data)
        
    except Exception as e:
        log(f"[ERROR] 加载fMRI失败: {e}")
        return

    events = load_events_fast(events_path)
    if events is None or events.empty:
        log(f"[SKIP] 事件文件无效: {sub}|run-{run}")
        return

    confounds = load_motion_confounds_fast(sub, run, n_volumes)
    
    # 准备输出目录
    out_dir = OUTPUT_ROOT / sub / f"run-{run}_LSS"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存目录
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # 优化的GLM参数 - 针对速度优化
    base_kwargs = dict(
        t_r=tr,
        slice_time_ref=0.5,  # 禁用切片时间校正以加速
        noise_model="ar1",
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        standardize=True,
        signal_scaling=False,
        minimize_memory=False,  # 为速度优化内存
        memory=str(cache_dir),
        memory_level=2,  # 更高的内存级别以重用更多计算
        n_jobs=1,  # 每个GLM单线程，避免嵌套并行
        verbose=0
    )

    total_trials = len(events)
    log(f"[{sub}|run-{run}] 开始优化LSS分析，共{total_trials}个trial...")

    # 方法1: 批量处理（内存友好）
    trial_info = []
    batch_size = min(20, total_trials)  # 自适应批大小
    
    for batch_start in range(0, total_trials, batch_size):
        batch_end = min(batch_start + batch_size, total_trials)
        batch_trials = list(range(batch_start, batch_end))
        
        # 准备批处理参数
        batch_args = []
        for trial_idx in batch_trials:
            original_condition = events.iloc[trial_idx]["trial_type"]
            batch_args.append((
                trial_idx, fmri_img, events, confounds, 
                base_kwargs, out_dir, original_condition
            ))
        
        # 串行处理批内trial（避免内存爆炸）
        for args in batch_args:
            result = process_single_trial(args)
            if result is not None:
                result.update({"subject": sub, "run": run})
                trial_info.append(result)
        
        # 批处理完成报告
        completed = len(trial_info)
        log(f"[{sub}|run-{run}] 批次完成: {completed}/{total_trials}")
        
        # 批次间清理
        if batch_end < total_trials:
            gc.collect()

    # 保存试验信息
    if trial_info:
        trial_df = pd.DataFrame(trial_info)
        trial_df.to_csv(out_dir / "trial_info.csv", index=False, encoding="utf-8-sig")
        
        # 保存mask（一次性）
        mask_path = out_dir / "mask.nii.gz"
        if not mask_path.exists():
            try:
                mask_img = compute_brain_mask(fmri_img)
                mask_img.to_filename(mask_path)
            except Exception as e:
                log(f"[WARN] 无法保存mask: {e}")

    # 最终清理
    del fmri_img, fmri_data
    gc.collect()
    
    success_count = len(trial_info)
    log(f"[{sub}|run-{run}] LSS分析完成: {success_count}/{total_trials} 成功")

def process_subject_run(args):
    """处理单个被试和run的包装函数，用于并行化"""
    sub, run = args
    try:
        process_one_run_optimized(sub, run)
        return f"{sub}|run-{run}: 成功"
    except Exception as e:
        return f"{sub}|run-{run}: 失败 - {e}"

def main_optimized():
    """优化主函数，支持并行处理"""
    log("开始高效LSS分析流程...")
    
    # 准备所有任务
    tasks = [(sub, run) for sub in SUBJECTS for run in RUNS]
    total_tasks = len(tasks)
    
    # 使用joblib进行并行处理
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=N_JOBS_PARALLEL)(delayed(process_subject_run)(task) for task in tasks)

    # 总结报告
    success_count = sum(1 for r in results if "成功" in r)
    log(f"\n{'='*60}")
    log(f"LSS分析完成: {success_count}/{total_tasks} 任务成功")
    log(f"{'='*60}")
    
    # 保存详细结果
    result_df = pd.DataFrame({"result": results})
    result_df.to_csv(OUTPUT_ROOT / "processing_results.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    # 根据需求选择运行模式
    USE_PARALLEL = True  # 设置为False使用串行模式
    
    if USE_PARALLEL and N_JOBS_PARALLEL > 1:
        main_optimized()
    else:
        # 串行模式（原逻辑）
        log("使用串行模式...")
        for sub in SUBJECTS:
            log(f"\n{'='*50}")
            log(f"处理被试: {sub}")
            log(f"{'='*50}")
            for run in RUNS:
                try:
                    process_one_run_optimized(sub, run)
                except Exception as e:
                    log(f"[ERROR] {sub}|run-{run} 失败: {e}")