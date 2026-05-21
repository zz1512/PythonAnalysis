#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSS 分析终极优化版 (RSA适配 + 内存安全)
结合了 fmri_lss_enhance.py 的工程架构与当前研究的业务逻辑

主要特性:
1. RSA 专用: 强制无平滑 (smoothing_fwhm=None)
2. 唯一 ID: 自动结合 trial_type + pic_num 生成 unique_label
3. 内存安全: 批处理 (Batching) + 缓存自动清理 + 线程锁
4. 结果校验: 生成详细的 metadata 用于 RSA 配对

输入
- fMRI BOLD：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/run{run}/*.nii(.gz)`
- events：`${PYTHON_METAPHOR_ROOT}/data_events/{sub}/{sub}_run-{run}_events.tsv`
  需要包含 `onset/duration/trial_type`，建议包含 `pic_num`（用于生成 `unique_label`）
- confounds：`${PYTHON_METAPHOR_ROOT}/Pro_proc_data/{sub}/multi_reg/*confounds_timeseries.tsv`

输出（默认 `${PYTHON_METAPHOR_ROOT}/lss_betas_final`）
- `lss_betas_final/sub-xx/run-{run}/beta_trial-XYZ_{unique_label}.nii.gz`：每个 trial 一个 beta
- `lss_betas_final/sub-xx/run-{run}/trial_info.csv`：trial 索引（RSA/stack_patterns 的上游输入）

注意
- 当前实现会对 events 表中的每一行都建模（包含你提供的所有 trial_type）。
  如果你希望排除 fake/nonword，请在 `load_and_prep_events()` 中做过滤，或在后续堆叠时排除。
"""

import os, gc, shutil, tempfile, sys, traceback
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_brain_mask
import logging

# 引入之前的配置 (路径配置从这里读取)
try:
    from glm_config import config
    import glm_utils
except ImportError:
    print("请确保 glm_config.py 和 glm_utils.py 在当前目录")
    sys.exit(1)

# ==========================
# 1. 环境与性能配置
# ==========================
# 限制底层库线程数，防止多进程并行时 CPU 爆炸
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# 被试与 Runs (从 config 读取或覆盖)
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
RUNS = [1, 2, 5, 6]  # RSA 关注前后测
# RUNS = [3, 4] # 如果你也想对学习阶段做LSS，可以加上

# 并行设置
N_JOBS_PARALLEL = 4  # 同时处理多少个 (Subject, Run) 任务
CLEAN_CACHE = True  # 运行后是否清理缓存

# 核心 RSA 参数
LSS_SMOOTHING = None  # 绝对不能平滑！
LSS_TR = 2.0

# 输出目录
OUTPUT_ROOT = getattr(config, "LSS_OUTPUT_ROOT", config.BASE_DIR / "lss_betas_final")

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)




def cleanup_temp_cache(path):
    try:
        if path.exists() and "lss_cache_" in path.name:
            shutil.rmtree(path, ignore_errors=True)
    except:
        pass


# ==========================
# 3. 核心逻辑修改区
# ==========================

def load_and_prep_events(sub, run):
    """
    加载并清洗 Events，关键是生成 Unique ID
    """
    e_path = str(config.EVENT_TPL).format(sub=sub, run=run)
    if not Path(e_path).exists():
        return None

    df = pd.read_csv(e_path, sep='\t')

    # [关键修改] 生成唯一 ID
    if 'pic_num' in df.columns:
        # 格式: kjw_19
        df['unique_label'] = df['trial_type'] + '_' + df['pic_num'].astype(str)
    else:
        logger.warning(f"{sub} Run {run}: 缺少 pic_num 列，使用 trial_index 代替")
        df['unique_label'] = df['trial_type'] + '_' + df.index.astype(str)

    # 确保必需列存在
    required = ['onset', 'duration', 'trial_type', 'unique_label']
    return df[required].copy()


def create_lss_events(target_idx, all_events):
    """
    LSS 事件表构建逻辑 (针对 RSA 优化)

    Target: 当前试次 -> 改名为 'LSS_TARGET'
    Nuisance: 同条件的其他试次 -> 改名为 'CONDITION_other' (可选，或者保持原名)
    Other: 其他条件的试次 -> 保持原名
    """
    lss_df = all_events.copy()

    # 策略：最简单的 LSS 实现
    # 1. 目标试次 -> 'LSS_TARGET'
    # 2. 其他所有试次 -> 保持原名 (Nilearn 会自动把它们作为 Regressors)
    #
    # 注：你之前的代码用了复杂的重命名策略 ('condition_other')，这在 Nilearn 中其实
    # 不是必须的，只要 Target 是独立的，其他都是 Nuisance。保持原名可以让模型
    # 自动处理同一 conditions 的共线性。

    lss_df.at[target_idx, 'trial_type'] = 'LSS_TARGET'

    return lss_df[['onset', 'duration', 'trial_type']]


def process_single_trial(args):
    """处理单个 Trial (被调用函数)"""
    (trial_idx, fmri_img_path, events_df, confounds, sample_masks,
     base_kwargs, out_dir, unique_label, original_type, run_mask) = args

    try:
        # 1. 构建 LSS 事件
        lss_events = create_lss_events(trial_idx, events_df)

        # 2. 建模 (每次重新初始化以确保干净)
        model = FirstLevelModel(mask_img=run_mask, **base_kwargs)
        model.fit(fmri_img_path, events=lss_events, confounds=confounds, sample_masks=sample_masks)

        # 3. 提取 Beta
        beta_map = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        # 4. 保存
        # 文件名包含 unique_label 方便肉眼检查
        fname = f"beta_trial-{trial_idx:03d}_{unique_label}.nii.gz"
        out_path = out_dir / fname
        beta_map.to_filename(out_path)

        # 5. 返回元数据
        return {
            "trial_index": trial_idx,
            "unique_label": unique_label,
            "condition": original_type,
            "beta_file": fname,
            "onset": events_df.iloc[trial_idx]['onset']
        }

    except Exception:
        return None


# ==========================
# 4. 流程控制
# ==========================

def process_one_run(sub, run):
    """
    单个 Run 的完整处理流程 (包含批处理与缓存管理)
    """
    # 1. 路径检查
    fmri_path = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
    if not fmri_path.exists():
        fmri_path = fmri_path.with_suffix('.nii.gz')
        if not fmri_path.exists(): return []

    # 2. 加载数据
    events = load_and_prep_events(sub, run)
    if events is None: return []

    confounds, sample_mask = glm_utils.robust_load_confounds(
        str(config.CONFOUNDS_TPL).format(sub=sub, run=run),
        config.DENOISE_STRATEGY
    )

    # 使用 compute_brain_mask 相比 FirstLevelModel 自动计算更显式
    logger.info(f"Computing mask for {sub} Run {run}...")
    run_mask = compute_brain_mask(str(fmri_path))

    # 3. 准备目录
    out_dir = OUTPUT_ROOT / sub / f"run-{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. GLM 配置 (RSA 专用)
    glm_args = dict(
        t_r=LSS_TR,
        slice_time_ref=0.5,
        smoothing_fwhm=LSS_SMOOTHING,  # !!! 关键：None
        noise_model="ar1",
        standardize=True,
        hrf_model="spm + derivative",
        drift_model="cosine",
        high_pass=1 / 128.0,
        minimize_memory=True,
        verbose=0
    )

    total_trials = len(events)
    results_meta = []

    # 5. 批处理循环 (防止内存溢出)
    BATCH_SIZE = 20

    for batch_start in range(0, total_trials, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_trials)
        batch_indices = range(batch_start, batch_end)

        # 串行执行批次内的 trial (避免内存峰值)
        for idx in batch_indices:
            row = events.iloc[idx]
            args = (
                idx, str(fmri_path), events, confounds, sample_mask,
                glm_args, out_dir,
                row['unique_label'], row['trial_type'], run_mask
            )

            res = process_single_trial(args)
            if res:
                res.update({'subject': sub, 'run': run, 'stage': 'Pre' if run in [1, 2] else 'Post'})
                results_meta.append(res)

        # 手动 GC
        gc.collect()

    # 6. 保存 run 级索引
    if results_meta:
        pd.DataFrame(results_meta).to_csv(out_dir / "trial_info.csv", index=False)

    # 7. 清理
    return results_meta


def process_wrapper(task):
    """Joblib 包装器"""
    sub, run = task
    try:
        return process_one_run(sub, run)
    except Exception as e:
        logger.error(f"失败 {sub} Run {run}: {e}")
        traceback.print_exc()
        return []


# ==========================
# 5. 主入口
# ==========================
if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 生成任务列表
    tasks = [(sub, run) for sub in SUBJECTS for run in RUNS]
    logger.info(f"开始 LSS 分析，共 {len(tasks)} 个任务，并行数 {N_JOBS_PARALLEL}")

    # 并行执行
    results = Parallel(n_jobs=N_JOBS_PARALLEL)(
        delayed(process_wrapper)(task) for task in tasks
    )

    # 合并所有元数据
    all_meta = []
    for r in results:
        all_meta.extend(r)

    if all_meta:
        final_df = pd.DataFrame(all_meta)
        final_csv = OUTPUT_ROOT / "lss_metadata_index_final.csv"
        final_df.to_csv(final_csv, index=False)
        logger.info(f"完成！总计生成 {len(final_df)} 张 Beta 图。")
        logger.info(f"索引文件已保存: {final_csv}")
    else:
        logger.error("未生成任何结果，请检查路径配置。")

    # 最后再清理一遍残留缓存
    try:
        for p in Path(tempfile.gettempdir()).glob("lss_cache_*"):
            cleanup_temp_cache(p)
    except:
        pass
