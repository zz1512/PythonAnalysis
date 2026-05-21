#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2（威胁情境任务）LSS 一阶分析脚本 - 总表版

适配你的当前目录结构/适配一个总 events 表（Excel/CSV/TSV），例如：
C:/python_fertility/exp2_behavior_onset.xlsx

总表关键列（可大小写/格式轻微变动，脚本会自动兼容）：
- Subject / subject
- run
- onset
- duration
- type / trial_type
- stim（可选，但强烈建议保留）

你的表格截图对应列示例：
DataFile.Basename | Subject | run | ... | stim | duration | ... | onset | type

核心特性：
1. 仅处理实验2：run-4, run-5, run-6
2. LSS（least-squares separate）：每个 trial 单独建模，目标 trial -> LSS_TARGET
3. 专门适配 RSA：不做空间平滑（smoothing_fwhm=None）
4. 直接从总表中按被试和 run 筛选事件
5. 自动识别 onset/duration 单位是否为毫秒；若为毫秒自动转换为秒
6. 每个 run 输出：
   - beta_trial-XXX_<unique_label>.nii.gz
   - trial_info.csv
7. 总索引输出：lss_metadata_index_exp2.csv
"""

import gc
import logging
import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_brain_mask
from scipy.ndimage import binary_dilation

# =========================
# 0. 线程限制（避免多进程时底层 BLAS 抢线程）
# =========================
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# =========================
# 1. 路径与任务配置
# =========================
BASE_DIR = Path(r"C:/python_fertility")
FMRI_ROOT = BASE_DIR / "fmriprep_core_data"

# 直接使用总事件表
MASTER_EVENT_FILE = BASE_DIR / "exp2_behavior_onset.xlsx"
# 如果你的文件不在这里，改这一行即可

OUTPUT_ROOT = BASE_DIR / "lss_betas_exp2"

# 被试编号格式：sub-001 / sub-004 / ...
SUBJECTS = [
    "sub-031", "sub-033", "sub-035", "sub-038", "sub-039", "sub-040",
    "sub-042", "sub-045", "sub-046", "sub-049", "sub-050", "sub-051", "sub-053",
    "sub-056", "sub-058", "sub-061", "sub-064", "sub-065", "sub-066", "sub-071",
    "sub-075", "sub-076", "sub-080", "sub-081", "sub-084", "sub-087", "sub-088",
    "sub-090", "sub-093", "sub-094", "sub-097", "sub-098", "sub-099", "sub-101",
    "sub-102", "sub-103", "sub-108", "sub-110", "sub-111", "sub-114", "sub-115",
    "sub-116", "sub-118", "sub-120", "sub-121", "sub-124", "sub-125", "sub-126",
    "sub-127", "sub-129"
]
RUNS = [4, 5, 6]

# 实验2每个 run 理论上 40 trials；不是硬性限制，只做检查提醒
EXPECTED_TRIALS_PER_RUN = 40

# 并行：建议 2~4，Windows 下太大容易内存吃满
N_JOBS_PARALLEL = 4
BATCH_SIZE = 20

# GLM 参数（RSA 专用）
TR = 2.0
SMOOTHING_FWHM = None  # RSA 必须保留体素模式
HRF_MODEL = "spm + derivative"
NOISE_MODEL = "ar1"
HIGH_PASS = 1 / 128.0

# Confounds 去噪参数
DENOISE_STRATEGY = {
    "denoise_strategy": "scrubbing",
    "motion": "friston24",
    "scrub": 0,
    "fd_threshold": 0.5,
    "std_dvars_threshold": 1.5,
}

# =========================
# 2. 日志
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(message)s'
)
logger = logging.getLogger("LSS_EXP2_MASTER")


# =========================
# 3. 工具函数
# =========================
def robust_load_confounds(confounds_path: Path, strategy_config: dict):
    """
    从 fMRIPrep confounds tsv 读取混杂变量，并构建：
    - Friston-24 motion regressors
    - sample_mask（scrubbing 后保留的体积索引）
    """
    df = pd.read_csv(confounds_path, sep='\t')

    motion_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    missing = [c for c in motion_axes if c not in df.columns]
    if missing:
        raise ValueError(f"{confounds_path.name} 缺少运动参数列: {missing}")

    motion_df = df[motion_axes].copy()

    # Friston-24
    if strategy_config.get('motion', '') == 'friston24':
        motion_sq = motion_df ** 2
        motion_dt = motion_df.diff().fillna(0)
        motion_dt_sq = motion_dt ** 2

        motion_sq.columns = [f"{c}_sq" for c in motion_df.columns]
        motion_dt.columns = [f"{c}_dt" for c in motion_df.columns]
        motion_dt_sq.columns = [f"{c}_dt_sq" for c in motion_df.columns]

        final_confounds = pd.concat([motion_df, motion_sq, motion_dt, motion_dt_sq], axis=1)
    else:
        final_confounds = motion_df

    # 再尽量补上一些常见的生理噪声列（如果存在）
    extra_candidates = [
        'csf', 'white_matter', 'global_signal',
        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05'
    ]
    extra_cols = [c for c in extra_candidates if c in df.columns]
    if extra_cols:
        final_confounds = pd.concat([final_confounds, df[extra_cols]], axis=1)

    final_confounds = final_confounds.fillna(0)

    # sample_mask（剔除 motion outlier TR）
    sample_mask = None
    if strategy_config.get('denoise_strategy', '') == 'scrubbing':
        outlier_cols = [c for c in df.columns if 'motion_outlier' in c]
        if outlier_cols:
            is_outlier = df[outlier_cols].sum(axis=1) > 0
            scrub = int(strategy_config.get('scrub', 0))
            if scrub > 0:
                iterations = int(scrub / 2)
                is_outlier = binary_dilation(is_outlier.values, iterations=iterations)
            sample_mask = np.where(~is_outlier)[0]

    return final_confounds, sample_mask


def subject_to_numeric(sub: str):
    """sub-004 -> 4"""
    try:
        return int(str(sub).split('-')[-1])
    except Exception:
        return None


def resolve_fmri_path(sub: str, run: int) -> Path:
    """
    适配当前文件名：
    C:/python_fertility/fmriprep_core_data/sub-004/run-4/sub-004_run-4_MNI_preproc_bold.nii.gz
    """
    candidates = [
        FMRI_ROOT / sub / f"run-{run}" / f"{sub}_run-{run}_MNI_preproc_bold.nii.gz",
        FMRI_ROOT / sub / f"run-{run}" / f"{sub}_run-{run}_MNI_preproc_bold.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    print(f"未找到 fMRI 文件: {sub} run-{run}")



def resolve_confounds_path(sub: str, run: int) -> Path:
    """
    适配当前文件名：
    C:/python_fertility/fmriprep_core_data/sub-004/run-4/sub-004_run-4_confounds.tsv
    """
    candidates = [
        FMRI_ROOT / sub / f"run-{run}" / f"{sub}_run-{run}_confounds.tsv",
        FMRI_ROOT / sub / f"run-{run}" / f"{sub}_run-{run}_desc-confounds_timeseries.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 confounds 文件: {sub} run-{run}")



def _read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    raise ValueError(f"不支持的事件文件格式: {path}")



def _normalize_colname(name: str) -> str:
    """
    列名标准化：保留字母数字，转小写。
    例如：DataFile.Basename -> datafilebasename
          stim.OnsetTime -> stimonsettime
    """
    return re.sub(r'[^a-zA-Z0-9]+', '', str(name)).lower()



def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把总表里的各种列名映射到统一命名。"""
    mapping = {}
    for c in df.columns:
        nc = _normalize_colname(c)
        if nc == 'subject':
            mapping[c] = 'subject'
        elif nc == 'run':
            mapping[c] = 'run'
        elif nc == 'onset':
            mapping[c] = 'onset'
        elif nc == 'duration':
            mapping[c] = 'duration'
        elif nc in {'type', 'trialtype'}:
            mapping[c] = 'trial_type'
        elif nc == 'stim':
            mapping[c] = 'stim'
        elif nc == 'picnum':
            mapping[c] = 'pic_num'
        elif nc == 'datafilebasename':
            mapping[c] = 'datafile_basename'

    return df.rename(columns=mapping)



def _deduplicate_labels(labels):
    """保证 unique_label 唯一"""
    counter = {}
    out = []
    for lab in labels:
        if lab not in counter:
            counter[lab] = 0
            out.append(lab)
        else:
            counter[lab] += 1
            out.append(f"{lab}_{counter[lab]}")
    return out



def _subject_mask(series: pd.Series, subj_num: int, sub_label: str):
    s = series.astype(str).str.strip()
    return (
        (s == str(subj_num)) |
        (s == str(sub_label)) |
        (s.str.zfill(3) == f"{subj_num:03d}") |
        (s.str.extract(r'(\d+)', expand=False).fillna('').astype(str) == str(subj_num))
    )



def load_and_prepare_events(sub: str, run: int) -> pd.DataFrame:
    """
    直接从总表读取并标准化事件表。

    兼容列名：
    - Subject / subject
    - run
    - onset
    - duration
    - type / trial_type
    - stim（可选）

    重要：
    你的截图显示 onset=13581, duration=4000，这更像毫秒。
    这里会自动判断：如果 duration 中位数 > 50，则按毫秒转秒。
    """
    if MASTER_EVENT_FILE is None or not Path(MASTER_EVENT_FILE).exists():
        raise FileNotFoundError(f"总 events 表不存在: {MASTER_EVENT_FILE}")

    df = _read_any_table(Path(MASTER_EVENT_FILE))
    df.columns = [str(c).strip() for c in df.columns]
    df = _canonicalize_columns(df)

    # 必须先有 subject/run
    required_base = ['subject', 'run']
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"{Path(MASTER_EVENT_FILE).name} 缺少列: {missing_base}")

    subj_num = subject_to_numeric(sub)
    if subj_num is None:
        raise ValueError(f"无法解析被试编号: {sub}")

    df = df.loc[_subject_mask(df['subject'], subj_num, sub)].copy()
    df = df.loc[pd.to_numeric(df['run'], errors='coerce') == int(run)].copy()

    if df.empty:
        raise ValueError(f"总表中没有找到 {sub} run-{run} 的事件")

    # 必需列
    required = ['onset', 'duration', 'trial_type']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(MASTER_EVENT_FILE).name} 缺少必需列: {missing}")

    # 清理空行 / 非刺激行
    df = df.dropna(subset=['onset', 'duration', 'trial_type']).copy()
    df['trial_type'] = df['trial_type'].astype(str).str.strip()
    df = df.loc[df['trial_type'] != ''].copy()

    # onset / duration 转数值
    onset = pd.to_numeric(df['onset'], errors='coerce')
    duration = pd.to_numeric(df['duration'], errors='coerce')
    if onset.isna().any() or duration.isna().any():
        bad_n = int(onset.isna().sum() + duration.isna().sum())
        raise ValueError(f"{Path(MASTER_EVENT_FILE).name} 的 onset/duration 存在非数值，问题单元数={bad_n}")

    # 自动单位检测：毫秒 -> 秒
    if float(duration.median()) > 50:
        logger.info(f"{sub} run-{run}: 检测到 onset/duration 很可能是毫秒，自动除以1000转换为秒")
        df['onset'] = onset / 1000.0
        df['duration'] = duration / 1000.0
    else:
        df['onset'] = onset.astype(float)
        df['duration'] = duration.astype(float)

    # stim
    if 'stim' not in df.columns:
        df['stim'] = ''
    df['stim'] = df['stim'].fillna('').astype(str).str.strip()

    # unique label
    if df['stim'].astype(str).str.len().gt(0).any():
        stim_base = df['stim'].apply(lambda x: Path(x).stem if str(x).strip() else 'nostim')
        labels = [f"{tt}_{sb}" for tt, sb in zip(df['trial_type'].astype(str), stim_base)]
    elif 'pic_num' in df.columns:
        labels = [f"{tt}_{pn}" for tt, pn in zip(df['trial_type'].astype(str), df['pic_num'].astype(str))]
    else:
        labels = [f"{tt}_{i:03d}" for i, tt in enumerate(df['trial_type'].astype(str), start=1)]

    df['unique_label'] = _deduplicate_labels(labels)

    # 排序：按 onset
    df = df.sort_values('onset').reset_index(drop=True)

    # 提醒 trial 数是否符合实验2设计
    if len(df) != EXPECTED_TRIALS_PER_RUN:
        logger.warning(
            f"{sub} run-{run}: 当前 events 共 {len(df)} 行，和实验2每run预期 {EXPECTED_TRIALS_PER_RUN} trials 不一致。"
        )

    return df[['onset', 'duration', 'trial_type', 'unique_label', 'stim']].copy()



def create_lss_events(target_idx: int, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    LSS 设计：
    - 当前目标 trial -> LSS_TARGET
    - 其他 trial 保持原始 trial_type
    """
    lss_df = events_df[['onset', 'duration', 'trial_type']].copy()
    lss_df.loc[target_idx, 'trial_type'] = 'LSS_TARGET'
    return lss_df



def process_single_trial(
    trial_idx: int,
    fmri_path: Path,
    events_df: pd.DataFrame,
    confounds: pd.DataFrame,
    sample_mask,
    glm_kwargs: dict,
    out_dir: Path,
    run_mask,
):
    row = events_df.iloc[trial_idx]
    unique_label = row['unique_label']
    original_type = row['trial_type']

    try:
        lss_events = create_lss_events(trial_idx, events_df)

        model = FirstLevelModel(mask_img=run_mask, **glm_kwargs)
        model.fit(
            run_imgs=str(fmri_path),
            events=lss_events,
            confounds=confounds,
            sample_masks=sample_mask,
        )

        beta_map = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        fname = f"beta_trial-{trial_idx:03d}_{unique_label}.nii.gz"
        out_path = out_dir / fname
        beta_map.to_filename(out_path)

        return {
            'trial_index': int(trial_idx),
            'unique_label': unique_label,
            'condition': original_type,
            'stim': row['stim'],
            'beta_file': fname,
            'onset': float(row['onset']),
            'duration': float(row['duration']),
        }
    except Exception as e:
        logger.error(f"{fmri_path.name} | trial {trial_idx} | {unique_label} 失败: {e}")
        logger.error(traceback.format_exc())
        return None



def process_one_run(sub: str, run: int):
    fmri_path = resolve_fmri_path(sub, run)
    confounds_path = resolve_confounds_path(sub, run)

    events_df = load_and_prepare_events(sub, run)
    confounds, sample_mask = robust_load_confounds(confounds_path, DENOISE_STRATEGY)

    logger.info(f"{sub} run-{run}: 计算脑掩膜")
    run_mask = compute_brain_mask(str(fmri_path))

    out_dir = OUTPUT_ROOT / sub / f"run-{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    glm_kwargs = dict(
        t_r=TR,
        slice_time_ref=0.5,
        smoothing_fwhm=SMOOTHING_FWHM,
        noise_model=NOISE_MODEL,
        standardize=True,
        signal_scaling=False,
        hrf_model=HRF_MODEL,
        drift_model='cosine',
        high_pass=HIGH_PASS,
        minimize_memory=True,
        verbose=0,
    )

    total_trials = len(events_df)
    logger.info(f"{sub} run-{run}: 开始 LSS，共 {total_trials} trials")
    results_meta = []

    for batch_start in range(0, total_trials, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_trials)
        logger.info(f"{sub} run-{run}: 处理 trials {batch_start:03d}-{batch_end - 1:03d}")

        for idx in range(batch_start, batch_end):
            res = process_single_trial(
                trial_idx=idx,
                fmri_path=fmri_path,
                events_df=events_df,
                confounds=confounds,
                sample_mask=sample_mask,
                glm_kwargs=glm_kwargs,
                out_dir=out_dir,
                run_mask=run_mask,
            )
            if res is not None:
                res.update({
                    'subject': sub,
                    'run': run,
                    'task': 'exp2_threat',
                })
                results_meta.append(res)

        gc.collect()

    if results_meta:
        pd.DataFrame(results_meta).to_csv(out_dir / 'trial_info.csv', index=False, encoding='utf-8-sig')
        logger.info(f"{sub} run-{run}: 已保存 trial_info.csv，成功 beta 数={len(results_meta)}")
    else:
        logger.warning(f"{sub} run-{run}: 没有生成任何 beta")

    return results_meta



def process_wrapper(task):
    sub, run = task
    try:
        return process_one_run(sub, run)
    except Exception as e:
        logger.error(f"{sub} run-{run} 总任务失败: {e}")
        logger.error(traceback.format_exc())
        return []


# =========================
# 4. 主入口
# =========================
if __name__ == '__main__':
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not Path(MASTER_EVENT_FILE).exists():
        raise FileNotFoundError(f"请先确认总事件表路径是否正确: {MASTER_EVENT_FILE}")

    tasks = [(sub, run) for sub in SUBJECTS for run in RUNS]
    logger.info(f"开始实验2 LSS（总表版）：{len(SUBJECTS)} 个被试 × {len(RUNS)} 个 runs = {len(tasks)} 个任务")
    logger.info(f"fMRI 根目录: {FMRI_ROOT}")
    logger.info(f"总 events 表: {MASTER_EVENT_FILE}")
    logger.info(f"输出目录: {OUTPUT_ROOT}")

    results = Parallel(n_jobs=N_JOBS_PARALLEL)(
        delayed(process_wrapper)(task) for task in tasks
    )

    all_meta = []
    for r in results:
        all_meta.extend(r)

    if all_meta:
        final_df = pd.DataFrame(all_meta)
        final_csv = OUTPUT_ROOT / 'lss_metadata_index_exp2.csv'
        final_df.to_csv(final_csv, index=False, encoding='utf-8-sig')
        logger.info(f"全部完成：共生成 {len(final_df)} 张 beta 图")
        logger.info(f"总索引已保存: {final_csv}")
    else:
        logger.error("没有生成任何 beta 图。请先检查：")
        logger.error("1) FMRI_ROOT 是否正确")
        logger.error("2) MASTER_EVENT_FILE 是否正确")
        logger.error("3) 总表是否包含 Subject/run/onset/duration/type(stim) 等列")
