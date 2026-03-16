#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_firstlevel_exp2_conditions.py

实验2（6类威胁）一阶 GLM 分析脚本

目标：
1. 对每个被试整合 run-4/5/6，拟合一阶模型
2. 输出每个条件（fer/soc/val/hed/exi/atf）相对隐式基线的脑激活图
3. 便于逐被试检查每个条件的激活情况，为后续修正模型和 RSA 提供依据

数据结构（按你当前目录）：
C:/python_fertility/
    fmriprep_core_data/
        sub-004/
            run-4/
                sub-004_run-4_MNI_preproc_bold.nii.gz
                sub-004_run-4_confounds.tsv
    exp2_behavior_onset.xlsx   # 总事件表

说明：
- 自动从总事件表中过滤 Subject 和 run
- 自动将 onset / duration 从毫秒转为秒（若检测到数值过大）
- 默认只建模刺激呈现阶段（onset + duration），不把评分阶段单独建模
- 输出每个条件的：beta/effect_size, z, t, variance
- 同时保存设计矩阵、玻璃脑图和切片图，方便逐被试质检
"""

from __future__ import annotations

import logging
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel

# =========================
# 0. 全局配置
# =========================
BASE_DIR = Path(r"C:/python_fertility")
FMRI_ROOT = BASE_DIR / "fmriprep_core_data"
MASTER_EVENT_FILE = BASE_DIR / "exp2_behavior_onset.xlsx"   # 改成你的真实总表路径
OUTPUT_ROOT = BASE_DIR / "first_level_exp2_conditions"

# 被试列表
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 130)]
RUNS = [4, 5, 6]

# 条件编码
CONDITIONS = ["fer", "soc", "val", "hed", "exi", "atf"]
COND_LABELS = {
    "fer": "生育威胁",
    "soc": "人际威胁",
    "val": "价值威胁",
    "hed": "享乐威胁",
    "exi": "生存威胁",
    "atf": "不生育威胁",
}

# 是否额外输出条件两两对比（会增加输出量）
SAVE_PAIRWISE_CONTRASTS = False

# 并行
N_JOBS = 4

# 一阶模型参数（沿用你之前的 GLM 逻辑）
TR = 2.0
SMOOTHING_FWHM = 6.0
SLICE_TIME_REF = 0.5
HRF_MODEL = "spm + derivative"
DRIFT_MODEL = "cosine"
HIGH_PASS = 1 / 128.0
NOISE_MODEL = "ar1"
STANDARDIZE = True
SIGNAL_SCALING = False

# 质控 / 绘图
PLOT_THRESHOLD_Z = 3.1
SAVE_DESIGN_MATRIX = True
SAVE_STAT_FIGURES = True
SAVE_HTML_REPORT = True  # 若你装了 matplotlib/html 相关依赖，可以改 True

# 去噪策略（借鉴你旧代码）
DENOISE_STRATEGY = {
    "denoise_strategy": "scrubbing",
    "motion": "friston24",
    "scrub": 0,
    "fd_threshold": 0.5,
    "std_dvars_threshold": 1.5,
}

# 中文绘图修复
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _CANDIDATES = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    plt = None

# =========================
# 1. 日志
# =========================
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_ROOT / "first_level_exp2.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("exp2_first_level")


# =========================
# 2. 工具函数
# =========================
def parse_subject_num(sub: str) -> int:
    """sub-004 -> 4"""
    return int(sub.split("-")[-1])



def robust_load_master_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"总事件表不存在: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df



def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None



def normalize_condition_code(x: str) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in CONDITIONS:
        return s
    return None



def build_run_events(master_df: pd.DataFrame, sub: str, run: int) -> pd.DataFrame:
    """
    从总表中提取某被试某 run 的事件表，输出 Nilearn 需要的三列：onset, duration, trial_type
    """
    subject_col = _pick_first_existing(master_df, ["Subject", "subject", "sub", "participant"])
    run_col = _pick_first_existing(master_df, ["run", "Run"])
    onset_col = _pick_first_existing(master_df, ["onset", "stim.OnsetTime", "stim_OnsetTime", "stim.OnsetDelay"])
    duration_col = _pick_first_existing(master_df, ["duration", "stim.Duration", "stim_Duration"])
    type_col = _pick_first_existing(master_df, ["type", "trial_type", "condition"])
    stim_col = _pick_first_existing(master_df, ["stim", "pic_num", "image", "item"])

    required = {
        "subject": subject_col,
        "run": run_col,
        "onset": onset_col,
        "duration": duration_col,
        "type": type_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"总事件表缺少必要列: {missing}; 实际列名={list(master_df.columns)}")

    sub_num = parse_subject_num(sub)
    df = master_df.copy()
    df = df[(pd.to_numeric(df[subject_col], errors="coerce") == sub_num) &
            (pd.to_numeric(df[run_col], errors="coerce") == run)]
    df = df.copy()
    if len(df) == 0:
        raise ValueError(f"总事件表中未找到 {sub} run-{run} 的记录")

    # 条件编码
    df["trial_type"] = df[type_col].map(normalize_condition_code)
    df = df[df["trial_type"].notna()].copy()
    if len(df) == 0:
        raise ValueError(f"{sub} run-{run} 的记录中没有有效条件码 ({CONDITIONS})")

    # onset / duration -> 秒
    df["onset"] = pd.to_numeric(df[onset_col], errors="coerce")
    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")
    if df[["onset", "duration"]].isna().any().any():
        bad_n = int(df[["onset", "duration"]].isna().any(axis=1).sum())
        raise ValueError(f"{sub} run-{run} 有 {bad_n} 行 onset/duration 无法转为数值")

    # 自动判断是否为毫秒
    if df["duration"].median() > 50 or df["onset"].max() > 1000:
        df["onset"] = df["onset"] / 1000.0
        df["duration"] = df["duration"] / 1000.0

    # 可选保留刺激编号，便于 QC
    if stim_col is not None:
        df["stim"] = df[stim_col].astype(str)
    else:
        df["stim"] = np.arange(len(df)).astype(str)

    out = df[["onset", "duration", "trial_type", "stim"]].sort_values("onset").reset_index(drop=True)
    return out



def robust_load_confounds(confounds_path: Path, strategy_config: Dict) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """延续你旧代码的思路：Friston-24 + scrubbing"""
    if not confounds_path.exists():
        raise FileNotFoundError(f"confounds 文件不存在: {confounds_path}")

    df = pd.read_csv(confounds_path, sep="\t")

    motion_axes = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    missing_motion = [c for c in motion_axes if c not in df.columns]
    if missing_motion:
        raise ValueError(f"{confounds_path.name} 缺少运动参数列: {missing_motion}")

    motion_df = df[motion_axes].copy()

    if strategy_config.get("motion", "friston24") == "friston24":
        motion_sq = motion_df ** 2
        motion_deriv = motion_df.diff().fillna(0)
        motion_deriv_sq = motion_deriv ** 2
        motion_sq.columns = [c + "_sq" for c in motion_df.columns]
        motion_deriv.columns = [c + "_dt" for c in motion_df.columns]
        motion_deriv_sq.columns = [c + "_dt_sq" for c in motion_df.columns]
        confounds = pd.concat([motion_df, motion_sq, motion_deriv, motion_deriv_sq], axis=1)
    else:
        confounds = motion_df

    # 可选加入 WM/CSF 等常见噪声回归项（如果有）
    extra_candidates = [
        "white_matter", "csf", "global_signal",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    ]
    extra_cols = [c for c in extra_candidates if c in df.columns and c not in confounds.columns]
    if extra_cols:
        confounds = pd.concat([confounds, df[extra_cols]], axis=1)

    sample_mask = None
    if strategy_config.get("denoise_strategy") == "scrubbing":
        outlier_cols = [c for c in df.columns if "motion_outlier" in c.lower()]
        if outlier_cols:
            is_outlier = df[outlier_cols].sum(axis=1) > 0
            n_vols = len(df)
            sample_mask = np.arange(n_vols)[~is_outlier.values]

    return confounds.fillna(0), sample_mask



def find_fmri_and_confounds(sub: str, run: int) -> Tuple[Path, Path]:
    run_dir = FMRI_ROOT / sub / f"run-{run}"
    fmri_path = run_dir / f"{sub}_run-{run}_MNI_preproc_bold.nii.gz"
    conf_path = run_dir / f"{sub}_run-{run}_confounds.tsv"

    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI 文件不存在: {fmri_path}")
    if not conf_path.exists():
        raise FileNotFoundError(f"confounds 文件不存在: {conf_path}")
    return fmri_path, conf_path



def make_subject_output_dirs(sub: str) -> Dict[str, Path]:
    base = OUTPUT_ROOT / sub
    paths = {
        "base": base,
        "maps": base / "maps",
        "figures": base / "figures",
        "design": base / "design_matrices",
        "qc": base / "qc",
        "reports": base / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths



def make_condition_contrasts(design_columns: List[str]) -> Dict[str, np.ndarray]:
    """
    为 6 个条件构造 contrast vector。由于 design matrix 可能包含导数列和 confounds，
    我们只给主条件列赋值 1，其他列全 0。
    """
    contrasts = {}
    for cond in CONDITIONS:
        vec = np.zeros(len(design_columns), dtype=float)
        for i, col in enumerate(design_columns):
            if col == cond:
                vec[i] = 1.0
        if vec.sum() == 0:
            logger.warning(f"设计矩阵中未找到主条件列: {cond}")
        contrasts[cond] = vec

    if SAVE_PAIRWISE_CONTRASTS:
        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1:]:
                vec = np.zeros(len(design_columns), dtype=float)
                for j, col in enumerate(design_columns):
                    if col == a:
                        vec[j] = 1.0
                    elif col == b:
                        vec[j] = -1.0
                contrasts[f"{a}_gt_{b}"] = vec
                contrasts[f"{b}_gt_{a}"] = -vec
    return contrasts



def save_design_matrices(glm: FirstLevelModel, out_dir: Path, sub: str):
    if not SAVE_DESIGN_MATRIX or plt is None:
        return
    for idx, dm in enumerate(glm.design_matrices_):
        try:
            fig, ax = plt.subplots(figsize=(18, 8))
            plotting.plot_design_matrix(dm, ax=ax)
            ax.set_title(f"{sub} run-{RUNS[idx]} 设计矩阵")
            fig.tight_layout()
            fig.savefig(out_dir / f"{sub}_run-{RUNS[idx]}_design_matrix.png", dpi=180)
            plt.close(fig)
            dm.to_csv(out_dir / f"{sub}_run-{RUNS[idx]}_design_matrix.csv", index=False)
        except Exception as e:
            logger.warning(f"保存 {sub} 设计矩阵失败: {e}")



def save_stat_figures(stat_img, out_dir: Path, sub: str, contrast_name: str, title: str):
    if not SAVE_STAT_FIGURES:
        return
    try:
        glass_path = out_dir / f"{sub}_{contrast_name}_glass.png"
        ortho_path = out_dir / f"{sub}_{contrast_name}_ortho.png"

        plotting.plot_glass_brain(
            stat_img,
            threshold=PLOT_THRESHOLD_Z,
            colorbar=True,
            plot_abs=False,
            title=title,
            output_file=str(glass_path),
        )
        plotting.plot_stat_map(
            stat_img,
            threshold=PLOT_THRESHOLD_Z,
            display_mode="ortho",
            colorbar=True,
            black_bg=False,
            title=title,
            output_file=str(ortho_path),
        )
    except Exception as e:
        logger.warning(f"保存 {sub} {contrast_name} 统计图失败: {e}")



def summarize_subject_events(event_tables: Dict[int, pd.DataFrame], out_path: Path):
    rows = []
    for run, df in event_tables.items():
        counts = df["trial_type"].value_counts().to_dict()
        row = {"run": run, "n_trials": len(df)}
        for cond in CONDITIONS:
            row[f"n_{cond}"] = counts.get(cond, 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# =========================
# 3. 核心：单被试一阶分析
# =========================
def process_single_subject(sub: str, master_events: pd.DataFrame) -> Optional[Dict[str, str]]:
    try:
        out_dirs = make_subject_output_dirs(sub)
        imgs: List[str] = []
        events_list: List[pd.DataFrame] = []
        confounds_list: List[pd.DataFrame] = []
        sample_masks: List[Optional[np.ndarray]] = []
        event_tables_for_qc: Dict[int, pd.DataFrame] = {}

        for run in RUNS:
            fmri_path, conf_path = find_fmri_and_confounds(sub, run)
            events_df = build_run_events(master_events, sub, run)
            conf_df, sample_mask = robust_load_confounds(conf_path, DENOISE_STRATEGY)

            imgs.append(str(fmri_path))
            events_list.append(events_df[["onset", "duration", "trial_type"]].copy())
            confounds_list.append(conf_df)
            sample_masks.append(sample_mask)
            event_tables_for_qc[run] = events_df.copy()

        summarize_subject_events(event_tables_for_qc, out_dirs["qc"] / f"{sub}_event_counts.csv")

        glm = FirstLevelModel(
            t_r=TR,
            slice_time_ref=SLICE_TIME_REF,
            smoothing_fwhm=SMOOTHING_FWHM,
            noise_model=NOISE_MODEL,
            standardize=STANDARDIZE,
            signal_scaling=SIGNAL_SCALING,
            hrf_model=HRF_MODEL,
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
            minimize_memory=False,
            verbose=0,
        )

        glm.fit(imgs, events=events_list, confounds=confounds_list, sample_masks=sample_masks)
        save_design_matrices(glm, out_dirs["design"], sub)

        design_columns = list(glm.design_matrices_[0].columns)
        contrasts = make_condition_contrasts(design_columns)

        manifest_rows = []
        for cname, cvec in contrasts.items():
            if np.allclose(cvec, 0):
                logger.warning(f"{sub} contrast {cname} 向量全 0，跳过")
                continue

            # 保存 4 类图
            outputs = {
                "effect_size": glm.compute_contrast(cvec, output_type="effect_size"),
                "z_score": glm.compute_contrast(cvec, output_type="z_score"),
                "stat": glm.compute_contrast(cvec, output_type="stat"),
                "effect_variance": glm.compute_contrast(cvec, output_type="effect_variance"),
            }

            file_map = {}
            for out_type, img in outputs.items():
                out_file = out_dirs["maps"] / f"{sub}_{cname}_{out_type}.nii.gz"
                img.to_filename(out_file)
                file_map[out_type] = str(out_file)

            title = f"{sub} | {COND_LABELS.get(cname, cname)}"
            save_stat_figures(outputs["z_score"], out_dirs["figures"], sub, cname, title)

            manifest_rows.append({
                "subject": sub,
                "contrast": cname,
                "label_cn": COND_LABELS.get(cname, cname),
                **file_map,
            })

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(out_dirs["reports"] / f"{sub}_contrast_manifest.csv", index=False)

        return {
            "subject": sub,
            "n_runs": len(RUNS),
            "manifest": str(out_dirs["reports"] / f"{sub}_contrast_manifest.csv"),
        }

    except Exception as e:
        logger.error(f"{sub} 一阶分析失败: {e}")
        logger.error(traceback.format_exc())
        return None


# =========================
# 4. 主程序
# =========================
def main():
    logger.info("=" * 70)
    logger.info("开始实验2（6类威胁）一阶 GLM 分析")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"MASTER_EVENT_FILE: {MASTER_EVENT_FILE}")
    logger.info(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    logger.info(f"SUBJECTS: {len(SUBJECTS)} | RUNS: {RUNS}")
    logger.info("=" * 70)

    master_events = robust_load_master_events(MASTER_EVENT_FILE)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_subject)(sub, master_events) for sub in SUBJECTS
    )

    results = [r for r in results if r is not None]
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / "first_level_subject_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info(f"完成。一阶分析成功被试数: {len(results)}/{len(SUBJECTS)}")
    logger.info(f"汇总文件: {OUTPUT_ROOT / 'first_level_subject_summary.csv'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
