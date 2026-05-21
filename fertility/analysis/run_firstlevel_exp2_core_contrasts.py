#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_firstlevel_exp2_core_contrasts.py

实验2（6类威胁）核心差值 contrasts 一阶 GLM 脚本

用途：
1. 重新拟合每个被试的一阶模型（run-4/5/6）
2. 只输出理论上最关键的条件差值图，不再输出单条件图
3. 方便逐被试检查：
   - 生存 vs 享乐（总体威胁-舒适维度）
   - 生育 vs 不生育（核心研究问题）
   - 生育 vs 生存（检验“生育≈生存”）
   - 不生育 vs 享乐（检验“不生育≈享乐/安享”）
   - 交叉特异性对比（生育 vs 享乐；不生育 vs 生存）

说明：
- 虽然单条件已经跑完，但若要得到严格正确的 pairwise contrast z/t 图，
  最稳妥的方式仍然是重新基于完整设计矩阵拟合 FirstLevelModel 后再算 contrast。
- 本脚本因此会重新 fit 一阶模型，但只导出差值 contrasts。
"""

from __future__ import annotations

import logging
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
MASTER_EVENT_FILE = BASE_DIR / "exp2_behavior_onset.xlsx"
OUTPUT_ROOT = BASE_DIR / "first_level_exp2_core_contrasts"

SUBJECTS = [
    "sub-004", "sub-005", "sub-006", "sub-007", "sub-010", "sub-013", "sub-016",
    "sub-020", "sub-021", "sub-022", "sub-023", "sub-024", "sub-027", "sub-029",
    "sub-030", "sub-031", "sub-033", "sub-035", "sub-038", "sub-039", "sub-040",
    "sub-042", "sub-045", "sub-046", "sub-049", "sub-050", "sub-051", "sub-053",
    "sub-056", "sub-058", "sub-061", "sub-064", "sub-065", "sub-066", "sub-071",
    "sub-075", "sub-076", "sub-080", "sub-081", "sub-084", "sub-087", "sub-088",
    "sub-090", "sub-093", "sub-094", "sub-097", "sub-098", "sub-099", "sub-101",
    "sub-102", "sub-103", "sub-108", "sub-110", "sub-111", "sub-114", "sub-115",
    "sub-116", "sub-118", "sub-120", "sub-121", "sub-124", "sub-125", "sub-126",
    "sub-127", "sub-129"
]
RUNS = [4, 5, 6]
N_JOBS = 4

CONDITIONS = ["fer", "soc", "val", "hed", "exi", "atf"]
COND_LABELS = {
    "fer": "生育威胁",
    "soc": "人际威胁",
    "val": "价值威胁",
    "hed": "享乐威胁",
    "exi": "生存威胁",
    "atf": "不生育威胁",
}

# =========================
# 核心 contrasts 设计
# =========================
# 第一层：理论主 contrasts（最优先）
CORE_CONTRASTS = [
    # 1) 总体威胁-安享参照轴：帮助看 exi/hed 的典型激活模式差异
    ("exi_gt_hed", "生存威胁 > 享乐威胁", {"exi": 1, "hed": -1}),
    ("hed_gt_exi", "享乐威胁 > 生存威胁", {"hed": 1, "exi": -1}),

    # 2) 研究核心轴：生育 vs 不生育
    ("fer_gt_atf", "生育威胁 > 不生育威胁", {"fer": 1, "atf": -1}),
    ("atf_gt_fer", "不生育威胁 > 生育威胁", {"atf": 1, "fer": -1}),

    # 3) 理论锚定比较：生育是否更靠近生存
    ("fer_gt_exi", "生育威胁 > 生存威胁", {"fer": 1, "exi": -1}),
    ("exi_gt_fer", "生存威胁 > 生育威胁", {"exi": 1, "fer": -1}),

    # 4) 理论锚定比较：不生育是否更靠近享乐/安享
    ("atf_gt_hed", "不生育威胁 > 享乐威胁", {"atf": 1, "hed": -1}),
    ("hed_gt_atf", "享乐威胁 > 不生育威胁", {"hed": 1, "atf": -1}),

    # 5) 交叉特异性对照：帮助判断是不是“锚定特异”，而不是普遍更强
    ("fer_gt_hed", "生育威胁 > 享乐威胁", {"fer": 1, "hed": -1}),
    ("hed_gt_fer", "享乐威胁 > 生育威胁", {"hed": 1, "fer": -1}),
    ("atf_gt_exi", "不生育威胁 > 生存威胁", {"atf": 1, "exi": -1}),
    ("exi_gt_atf", "生存威胁 > 不生育威胁", {"exi": 1, "atf": -1}),
]

# 模型参数（沿用你之前脚本）
TR = 2.0
SMOOTHING_FWHM = 6.0
SLICE_TIME_REF = 0.5
HRF_MODEL = "spm + derivative"
DRIFT_MODEL = "cosine"
HIGH_PASS = 1 / 128.0
NOISE_MODEL = "ar1"
STANDARDIZE = True
SIGNAL_SCALING = False

PLOT_THRESHOLD_Z = 3.1
SAVE_DESIGN_MATRIX = False  # 单条件已经看过，这里默认不重复输出
SAVE_STAT_FIGURES = True
SAVE_EFFECT_SIZE = True
SAVE_STAT = True
SAVE_Z = True
SAVE_EFFECT_VARIANCE = False

DENOISE_STRATEGY = {
    "denoise_strategy": "scrubbing",
    "motion": "friston24",
    "scrub": 0,
    "fd_threshold": 0.5,
    "std_dvars_threshold": 1.5,
}

# 中文字体修复
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

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_ROOT / "first_level_exp2_core_contrasts.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("exp2_core_contrasts")


# =========================
# 工具函数
# =========================
def parse_subject_num(sub: str) -> int:
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
    return s if s in CONDITIONS else None


def build_run_events(master_df: pd.DataFrame, sub: str, run: int) -> pd.DataFrame:
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
    if len(df) == 0:
        raise ValueError(f"总事件表中未找到 {sub} run-{run} 的记录")

    df = df.copy()
    df["trial_type"] = df[type_col].map(normalize_condition_code)
    df = df[df["trial_type"].notna()].copy()
    if len(df) == 0:
        raise ValueError(f"{sub} run-{run} 的记录中没有有效条件码 ({CONDITIONS})")

    df["onset"] = pd.to_numeric(df[onset_col], errors="coerce")
    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")
    if df[["onset", "duration"]].isna().any().any():
        bad_n = int(df[["onset", "duration"]].isna().any(axis=1).sum())
        raise ValueError(f"{sub} run-{run} 有 {bad_n} 行 onset/duration 无法转为数值")

    # 自动从毫秒转秒
    if df["duration"].median() > 50 or df["onset"].max() > 1000:
        df["onset"] = df["onset"] / 1000.0
        df["duration"] = df["duration"] / 1000.0

    if stim_col is not None:
        df["stim"] = df[stim_col].astype(str)
    else:
        df["stim"] = np.arange(len(df)).astype(str)

    return df[["onset", "duration", "trial_type", "stim"]].sort_values("onset").reset_index(drop=True)


def robust_load_confounds(confounds_path: Path, strategy_config: Dict) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
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
        "reports": base / "reports",
        "qc": base / "qc",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def summarize_subject_events(event_tables: Dict[int, pd.DataFrame], out_path: Path):
    rows = []
    for run, df in event_tables.items():
        counts = df["trial_type"].value_counts().to_dict()
        row = {"run": run, "n_trials": len(df)}
        for cond in CONDITIONS:
            row[f"n_{cond}"] = counts.get(cond, 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def make_core_contrast_vectors(design_columns: List[str]) -> Dict[str, Tuple[np.ndarray, str]]:
    """
    返回: {contrast_name: (vector, label_cn)}
    只对 design matrix 中的主条件列赋权重，导数列和 confounds 自动为 0。
    """
    out = {}
    for name, label_cn, weight_map in CORE_CONTRASTS:
        vec = np.zeros(len(design_columns), dtype=float)
        for i, col in enumerate(design_columns):
            if col in weight_map:
                vec[i] = weight_map[col]
        if np.allclose(vec, 0):
            logger.warning(f"contrast {name} 在设计矩阵中找不到任何主条件列，跳过")
            continue
        out[name] = (vec, label_cn)
    return out


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
        logger.warning(f"保存 {sub} {contrast_name} 图失败: {e}")


# =========================
# 核心：单被试一阶分析
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

        design_columns = list(glm.design_matrices_[0].columns)
        contrasts = make_core_contrast_vectors(design_columns)

        manifest_rows = []
        for cname, (cvec, label_cn) in contrasts.items():
            outputs = {}
            if SAVE_EFFECT_SIZE:
                outputs["effect_size"] = glm.compute_contrast(cvec, output_type="effect_size")
            if SAVE_Z:
                outputs["z_score"] = glm.compute_contrast(cvec, output_type="z_score")
            if SAVE_STAT:
                outputs["stat"] = glm.compute_contrast(cvec, output_type="stat")
            if SAVE_EFFECT_VARIANCE:
                outputs["effect_variance"] = glm.compute_contrast(cvec, output_type="effect_variance")

            file_map = {}
            for out_type, img in outputs.items():
                out_file = out_dirs["maps"] / f"{sub}_{cname}_{out_type}.nii.gz"
                img.to_filename(out_file)
                file_map[out_type] = str(out_file)

            if "z_score" in outputs:
                save_stat_figures(outputs["z_score"], out_dirs["figures"], sub, cname, f"{sub} | {label_cn}")

            manifest_rows.append({
                "subject": sub,
                "contrast": cname,
                "label_cn": label_cn,
                **file_map,
            })

        manifest_path = out_dirs["reports"] / f"{sub}_core_contrast_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        return {
            "subject": sub,
            "n_runs": len(RUNS),
            "manifest": str(manifest_path),
        }

    except Exception as e:
        logger.error(f"{sub} 核心 contrasts 一阶分析失败: {e}")
        logger.error(traceback.format_exc())
        return None


# =========================
# 主程序
# =========================
def main():
    logger.info("=" * 70)
    logger.info("开始实验2核心差值 contrasts 一阶 GLM 分析")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"MASTER_EVENT_FILE: {MASTER_EVENT_FILE}")
    logger.info(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    logger.info(f"SUBJECTS: {len(SUBJECTS)} | RUNS: {RUNS}")
    logger.info(f"核心 contrasts 数量: {len(CORE_CONTRASTS)}")
    logger.info("=" * 70)

    master_events = robust_load_master_events(MASTER_EVENT_FILE)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_subject)(sub, master_events) for sub in SUBJECTS
    )
    results = [r for r in results if r is not None]
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / "core_contrast_subject_summary.csv", index=False)

    # 保存 contrast 列表说明
    contrast_desc = pd.DataFrame([
        {"contrast": name, "label_cn": label_cn, "weights": str(weight_map)}
        for name, label_cn, weight_map in CORE_CONTRASTS
    ])
    contrast_desc.to_csv(OUTPUT_ROOT / "core_contrast_definitions.csv", index=False, encoding="utf-8-sig")

    logger.info("=" * 70)
    logger.info(f"完成。成功被试数: {len(results)}/{len(SUBJECTS)}")
    logger.info(f"汇总文件: {OUTPUT_ROOT / 'core_contrast_subject_summary.csv'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
