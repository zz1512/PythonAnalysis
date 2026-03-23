#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lss_main.py
LSS 分析主程序 - 调用glm_config.py 和 glm_utils.py
功能：
1. 兼容 Surface/Volume 的单试次建模 (伪装 NIfTI 策略)。
2. 直接输出包含 stimulus_content 的索引文件，实现一步到位的跨被试对齐。
"""

import os

# 1. 限制底层线程
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import gc
import logging
import argparse
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn import surface, image
from tqdm import tqdm

try:
    from glm_config import get_scenario_configs
    import glm_utils
except ImportError:
    print("Error: 请确保 glm_config.py 和 glm_utils.py 在同一目录下")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

WRITE_MODES = ("overwrite", "append", "sync")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSS 分析：支持断点续跑与索引增量写入")
    p.add_argument("subject_file", nargs="?", default=None, help="可选：被试列表文件（txt/空白分隔）")
    p.add_argument(
        "--write-mode",
        choices=WRITE_MODES,
        default="overwrite",
        help="overwrite=覆盖重建；append=仅追加新生成 trial；sync=仅同步已存在 beta 到索引",
    )
    return p.parse_args()


def _merge_append(existing: pd.DataFrame, new: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new], axis=0, ignore_index=True)
    if out.empty:
        return out
    miss = [c for c in key_cols if c not in out.columns]
    if miss:
        raise ValueError(f"索引缺少判重列: {miss}")
    out = out.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return out


def load_subjects_from_file(subject_file):
    """从文本文件中读取被试列表，兼容带表头/多列的空白分隔格式。"""
    subject_file = Path(subject_file)
    if not subject_file.exists():
        raise FileNotFoundError(f"被试列表文件不存在: {subject_file}")

    subjects = []
    with subject_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            first_col = line.split()[0]

            # 跳过常见表头，例如: sub_id mean_FD
            if i == 0 and first_col.lower() in {"sub", "sub_id", "subject", "subject_id"}:
                continue

            # 统一为 sub- 前缀格式
            if not first_col.startswith("sub-"):
                first_col = f"sub-{first_col}"

            subjects.append(first_col)

    if not subjects:
        raise ValueError(f"被试列表文件为空或未解析到有效被试: {subject_file}")

    # 去重并排序，保证稳定执行顺序
    return sorted(set(subjects))


def clean_bids_val(val):
    """清理 BIDS 表格中的字符串 (去除 .png 后缀, 去除空格)"""
    s = str(val).strip()
    if s.endswith('.png'):
        s = s[:-4]
    return s


def validate_subject_task_presence(subjects, config):
    """校验每个被试是否同时具备 EMO/SOC 的核心输入文件。"""
    presence_rows = []
    valid_subjects = []

    for sub in subjects:
        task_presence = {}
        missing_msgs = []

        for run in config.RUNS:
            task_label = config.RUN_MAP[run]["task"]
            try:
                fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
                fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)

                conf_ok = conf_path.exists()
                if not conf_ok:
                    cand = list(conf_path.parent.glob(f"*{task_label}*confounds_timeseries.tsv"))
                    conf_ok = bool(cand)

                run_ok = fmri_path.exists() and event_path.exists() and conf_ok
                task_presence[task_label] = run_ok

                if not run_ok:
                    missing_msgs.append(
                        f"{task_label}:fmri={fmri_path.exists()},events={event_path.exists()},confounds={conf_ok}"
                    )
            except Exception as e:
                task_presence[task_label] = False
                missing_msgs.append(f"{task_label}:get_paths_failed({e})")

        emo_ok = task_presence.get("EMO", False)
        soc_ok = task_presence.get("SOC", False)
        both_ok = emo_ok and soc_ok

        presence_rows.append(
            {
                "subject": sub,
                "emo_inputs_ok": emo_ok,
                "soc_inputs_ok": soc_ok,
                "inputs_both_ok": both_ok,
                "input_check_note": "; ".join(missing_msgs) if missing_msgs else "",
            }
        )
        if both_ok:
            valid_subjects.append(sub)

    return valid_subjects, presence_rows


def save_beta_as_gifti(beta_data, output_path):
    """辅助函数：将计算出的 Numpy 数组保存为 .gii 文件"""
    data_flat = beta_data.flatten().astype(np.float32)
    da = nb.gifti.GiftiDataArray(data_flat, intent='NIFTI_INTENT_ESTIMATE')
    img = nb.gifti.GiftiImage(darrays=[da])
    nb.save(img, str(output_path))


def process_single_trial_lss(args):
    """单个 Trial 建模处理函数"""
    (trial_idx, fmri_input, events_df, confounds_df,
     out_dir, unique_label, mask_img, config) = args

    try:
        stage = "prepare_input"
        # 1. 准备数据对象
        if config.DATA_SPACE == 'volume':
            # Volume 模式：直接传路径
            input_img = fmri_input
            current_mask = mask_img
        else:
            # ====================================================
            # 【Surface 模式】：伪装成 NIfTI + 强制全掩膜
            # ====================================================
            # Surface workaround: wrap (n_vert, n_time) as a fake 4D NIfTI and use an all-ones mask
            # so nilearn does not auto-mask away vertices.
            surf_data = surface.load_surf_data(fmri_input)

            # 强制转为 float32
            try:
                surf_data = surf_data.astype(np.float32)
            except ValueError:
                surf_data = pd.to_numeric(surf_data.flatten(), errors='coerce').reshape(surf_data.shape)
                surf_data = np.nan_to_num(surf_data).astype(np.float32)

            # 确保是 2D (Vertices, Time)
            if surf_data.ndim == 1:
                surf_data = surf_data[:, np.newaxis]

            n_vert, n_time = surf_data.shape

            # 伪 4D 体数据 (Vertices, 1, 1, Time)
            fake_vol_data = surf_data.reshape((n_vert, 1, 1, n_time))

            affine = np.eye(4)
            input_img = nb.Nifti1Image(fake_vol_data, affine=affine)

            # 全 1 Mask：避免 nilearn 自动 mask 抽空
            mask_data = np.ones((n_vert, 1, 1), dtype=np.int8)
            current_mask = nb.Nifti1Image(mask_data, affine=affine)
            # ====================================================

        # 2. 构建事件
        stage = "create_lss_events"
        # LSS design: 1 target regressor (this trial) + 1 nuisance regressor (all other trials).
        lss_events = glm_utils.create_lss_events(trial_idx, events_df)

        # 3. 建模
        stage = "init_model"
        model = FirstLevelModel(
            t_r=config.TR,
            slice_time_ref=0.5,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            noise_model="ar1",
            standardize=True,
            hrf_model="spm + derivative",
            drift_model="cosine",
            high_pass=1 / 128.0,
            mask_img=current_mask,
            minimize_memory=False,
            verbose=0
        )

        # 4. 拟合
        stage = "fit"
        model.fit(input_img, events=lss_events, confounds=confounds_df)

        # 5. 提取 Beta
        stage = "compute_contrast"
        beta_img = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        # 6. 保存结果
        stage = "save"
        ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
        # 加上 hemi 后缀
        hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == 'surface' else ""
        fname = f"beta_{unique_label}{hemi_suffix}{ext}"

        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if config.DATA_SPACE == 'volume':
            beta_img.to_filename(out_path)
        else:
            beta_data = beta_img.get_fdata()
            save_beta_as_gifti(beta_data, out_path)

        return {"status": "success", "file": fname}

    except Exception as e:
        logger.error(f"Trial {trial_idx} error ({stage}): {e}")
        return {"status": "fail", "stage": stage, "error": str(e)}


def _build_expected_results(
    sub: str,
    run: int,
    task_label: str,
    events: pd.DataFrame,
    out_dir: Path,
    hemi_suffix: str,
    ext: str,
    cond_col: Optional[str],
    emo_col: Optional[str],
) -> Tuple[List[Path], List[dict]]:
    expected_files: List[Path] = []
    expected_results: List[dict] = []
    for idx in range(len(events)):
        row = events.iloc[idx]
        if str(row.get("trial_type", "")) == "Other":
            continue
        fname = f"beta_{row['unique_label']}{hemi_suffix}{ext}"
        expected_files.append(out_dir / fname)
        raw_cond = clean_bids_val(row[cond_col]) if cond_col else "Unknown"
        raw_emo = clean_bids_val(row[emo_col]) if emo_col else "Unknown"
        # Cross-subject alignment key (task + condition + emotion).
        stimulus_content = f"{task_label}_{raw_cond}_{raw_emo}"
        expected_results.append(
            {
                "subject": sub,
                "run": int(run),
                "task": str(task_label),
                "label": str(row["unique_label"]),
                "file": str(fname),
                "stimulus_content": str(stimulus_content),
                "raw_condition": str(raw_cond),
                "raw_emotion": str(raw_emo),
            }
        )
    return expected_files, expected_results


def sync_one_run(sub: str, run: int, config):
    # Sync mode: do not run GLM; only scan existing outputs and rebuild aligned index entries.
    audit = []
    try:
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)
        if not fmri_path.exists() or not event_path.exists():
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": config.RUN_MAP[run]["task"],
                    "status": "fail",
                    "stage": "sync_missing_inputs",
                    "error": "fmri_or_events_not_found",
                    "fmri_path": str(fmri_path),
                    "events_path": str(event_path),
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}
    except Exception as e:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": config.RUN_MAP[run]["task"] if run in config.RUN_MAP else "",
                "status": "fail",
                "stage": "sync_get_paths",
                "error": str(e),
            }
        )
        return {"meta": [], "audit": audit}

    task_type = config.RUN_MAP[run]["folder_task"]
    task_label = config.RUN_MAP[run]["task"]
    out_dir = config.OUTPUT_ROOT / task_type / sub
    ext = ".gii" if config.DATA_SPACE == "surface" else ".nii.gz"
    hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == "surface" else ""

    try:
        events = pd.read_csv(event_path, sep="\t")
        if events.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "sync_read_events",
                    "error": "events_empty",
                    "events_path": str(event_path),
                }
            )
            return {"meta": [], "audit": audit}
        col_map = {c.lower(): c for c in events.columns}
        cond_col = col_map.get("condition")
        emo_col = col_map.get("emotion")
        events = glm_utils.reclassify_events(events)
        events["unique_label"] = events["trial_type"] + "_tr" + events.index.astype(str)
        expected_files, expected_results = _build_expected_results(
            sub=sub,
            run=int(run),
            task_label=str(task_label),
            events=events,
            out_dir=Path(out_dir),
            hemi_suffix=str(hemi_suffix),
            ext=str(ext),
            cond_col=cond_col,
            emo_col=emo_col,
        )
        present = []
        for f, r in zip(expected_files, expected_results):
            if Path(f).exists():
                present.append(r)
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "success",
                "stage": "sync",
                "n_present": int(len(present)),
                "n_expected": int(len(expected_results)),
            }
        )
        return {"meta": present, "audit": audit}
    except Exception as e:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "fail",
                "stage": "sync_failed",
                "error": str(e),
            }
        )
        return {"meta": [], "audit": audit}


def process_one_run(sub, run, config, record_existing_outputs: bool = True):
    audit = []
    # 1. 路径获取
    try:
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)

        if not conf_path.exists():
            cand = list(conf_path.parent.glob(f"*{config.RUN_MAP[run]['task']}*confounds_timeseries.tsv"))
            if cand:
                conf_path = cand[0]
            else:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": config.RUN_MAP[run]["task"],
                        "status": "fail",
                        "stage": "missing_confounds",
                        "error": "confounds_not_found",
                        "fmri_path": str(fmri_path),
                        "events_path": str(event_path),
                        "confounds_path": str(conf_path),
                    }
                )
                return {"meta": [], "audit": audit}

        if not fmri_path.exists() or not event_path.exists():
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": config.RUN_MAP[run]["task"],
                    "status": "fail",
                    "stage": "missing_inputs",
                    "error": "fmri_or_events_not_found",
                    "fmri_path": str(fmri_path),
                    "events_path": str(event_path),
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}
    except Exception:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": config.RUN_MAP[run]["task"] if run in config.RUN_MAP else "",
                "status": "fail",
                "stage": "get_paths",
                "error": "get_paths_failed",
            }
        )
        return {"meta": [], "audit": audit}

    # 2. 准备目录
    task_type = config.RUN_MAP[run]['folder_task']
    task_label = config.RUN_MAP[run]['task']  # EMO or SOC
    out_dir = config.OUTPUT_ROOT / task_type / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 加载数据
    try:
        # 读取 Events
        events = pd.read_csv(event_path, sep='\t')
        if events.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "read_events",
                    "error": "events_empty",
                    "events_path": str(event_path),
                }
            )
            return {"meta": [], "audit": audit}

        # 预先处理列名 (大小写兼容)
        col_map = {c.lower(): c for c in events.columns}
        cond_col = col_map.get('condition')
        emo_col = col_map.get('emotion')

        # 运行分类逻辑 (生成 trial_type)
        events = glm_utils.reclassify_events(events)

        # 加载 Confounds
        confounds = glm_utils.load_complex_confounds(str(conf_path))
        if confounds.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "load_confounds",
                    "error": "confounds_empty",
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}

        # 生成基础标签 (基于位置)
        events['unique_label'] = events['trial_type'] + '_tr' + events.index.astype(str)
    except Exception:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "fail",
                "stage": "prepare_inputs",
                "error": "prepare_inputs_failed",
                "events_path": str(event_path),
                "confounds_path": str(conf_path),
            }
        )
        return {"meta": [], "audit": audit}

    # 4. 预加载 Volume 数据 + Mask (仅 Volume)
    run_mask = None
    fmri_input = str(fmri_path)
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            try:
                fmri_input = nb.load(str(fmri_path))
            except Exception:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "status": "fail",
                        "stage": "load_fmri",
                        "error": "fmri_load_failed",
                        "fmri_path": str(fmri_path),
                    }
                )
                return {"meta": [], "audit": audit}

            try:
                mask_img = nb.load(str(config.MNI_MASK_PATH))
                run_mask = image.resample_to_img(mask_img, fmri_input, interpolation="nearest")
            except Exception:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "status": "fail",
                        "stage": "mask",
                        "error": "mask_load_or_resample_failed",
                        "mask_path": str(config.MNI_MASK_PATH),
                    }
                )
                return {"meta": [], "audit": audit}
        else:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "mask",
                    "error": "mask_missing",
                    "mask_path": str(config.MNI_MASK_PATH) if config.MNI_MASK_PATH else "",
                }
            )
            return {"meta": [], "audit": audit}

    # 5. 循环处理 Trial
    results = []

    # 始终全量跑
    indices = range(len(events))

    # 5.1 预检查：如果全部 trial 结果已存在，则跳过本 run
    ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
    hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == 'surface' else ""
    expected_files, expected_results = _build_expected_results(
        sub=sub,
        run=int(run),
        task_label=str(task_label),
        events=events,
        out_dir=Path(out_dir),
        hemi_suffix=str(hemi_suffix),
        ext=str(ext),
        cond_col=cond_col,
        emo_col=emo_col,
    )

    if expected_files and all(p.exists() for p in expected_files):
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "skipped",
                "stage": "all_trial_outputs_exist",
                "file_count": len(expected_files),
            }
        )
        return {"meta": expected_results if bool(record_existing_outputs) else [], "audit": audit}

    for idx in indices:
        try:
            row = events.iloc[idx]

            # 跳过无关事件
            if row['trial_type'] == 'Other':
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "skipped",
                        "stage": "trial_type_other",
                    }
                )
                continue

            # === [核心新增]：提取对齐信息 ===
            # 直接从当前 row 提取 condition 和 emotion
            raw_cond = "Unknown"
            raw_emo = "Unknown"

            if cond_col:
                raw_cond = clean_bids_val(row[cond_col])
            if emo_col:
                raw_emo = clean_bids_val(row[emo_col])

            # 生成唯一的内容对齐标签: Task_Condition_Emotion
            # e.g., EMO_PassiveLook_Negative_001_Negative
            stimulus_content = f"{task_label}_{raw_cond}_{raw_emo}"
            # ===============================

            # 如果结果已存在则跳过
            fname = f"beta_{row['unique_label']}{hemi_suffix}{ext}"
            if (out_dir / fname).exists():
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "skipped",
                        "stage": "output_exists",
                        "file": fname,
                    }
                )
                if bool(record_existing_outputs):
                    results.append(
                        {
                            "subject": sub,
                            "run": run,
                            "task": task_label,
                            "label": row["unique_label"],
                            "file": fname,
                            "stimulus_content": stimulus_content,
                            "raw_condition": raw_cond,
                            "raw_emotion": raw_emo,
                        }
                    )
                continue

            # 执行建模
            args = (idx, fmri_input, events, confounds, out_dir, row['unique_label'], run_mask, config)
            res = process_single_trial_lss(args)

            if res and res.get("status") == "success":
                # 将元数据和对齐信息一起写入结果
                results.append({
                    'subject': sub,
                    'run': run,
                    'task': task_label,
                    'label': row['unique_label'],  # 原始位置标签 (Passive_Neutral_tr0)
                    'file': res['file'],  # 生成的文件名
                    'stimulus_content': stimulus_content,  # [NEW] 内容对齐标签
                    'raw_condition': raw_cond,  # [NEW] 原始条件
                    'raw_emotion': raw_emo  # [NEW] 原始情绪
                })
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "success",
                        "stage": "trial",
                        "file": str(res.get("file", "")),
                    }
                )
            else:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "fail",
                        "stage": str(res.get("stage", "trial")) if isinstance(res, dict) else "trial",
                        "error": str(res.get("error", "")) if isinstance(res, dict) else "unknown",
                    }
                )
        except Exception:
            continue

    gc.collect()
    return {"meta": results, "audit": audit}


def run_main():
    args = parse_args()
    subject_file = Path(args.subject_file).expanduser() if args.subject_file is not None else None
    run_lss(subject_file=subject_file, write_mode=str(args.write_mode))


def run_lss(subject_file: Optional[Path], write_mode: str) -> None:
    subjects_from_file = None
    if subject_file is not None:
        subjects_from_file = load_subjects_from_file(Path(subject_file))
        logger.info(f"检测到被试列表文件: {subject_file}")
        logger.info(f"从文件读取被试数量: {len(subjects_from_file)}")
    write_mode = str(write_mode)
    if write_mode not in WRITE_MODES:
        raise ValueError(f"write_mode 不支持: {write_mode}")

    scenarios = get_scenario_configs()

    for cfg in scenarios:
        scenario_name = f"{cfg.DATA_SPACE}_{cfg.HEMI}" if cfg.DATA_SPACE == 'surface' else "volume"
        logger.info(f"\n>>> 正在处理: {scenario_name} (并行数: {cfg.N_JOBS})")

        # DEBUG 逻辑：限制被试数
        original_subjects = list(subjects_from_file) if subjects_from_file is not None else list(cfg.SUBJECTS)
        subjects = list(original_subjects)
        if getattr(cfg, 'DEBUG', False):
            debug_n_sub = getattr(cfg, 'DEBUG_N_SUBJECTS', 30)
            subjects = subjects[:min(debug_n_sub, len(subjects))]
            total_n = len(subjects_from_file) if subjects_from_file is not None else len(cfg.SUBJECTS)
            logger.info(f"[DEBUG] 限制被试数量: {len(subjects)} / {total_n}")

        checked_subjects, presence_rows = validate_subject_task_presence(subjects, cfg)
        # Strictly require each subject to have both EMO and SOC inputs before running LSS.
        logger.info(f"通过 EMO+SOC 双任务输入校验: {len(checked_subjects)} / {len(subjects)}")
        tasks = [(s, r) for s in checked_subjects for r in cfg.RUNS]
        if not tasks:
            status_file = cfg.OUTPUT_ROOT / f"lss_subject_status_{cfg.SCENARIO_ID}.csv"
            pd.DataFrame(presence_rows).assign(
                lss_status="fail",
                lss_note="输入校验失败，未进入LSS建模",
            ).to_csv(status_file, index=False)
            logger.info(f"LSS 被试状态已保存: {status_file}")
            continue

        all_meta = []
        all_audit = []
        try:
            record_existing = bool(write_mode) == "overwrite"
            if write_mode == "append":
                record_existing = False
            if cfg.PARALLEL:
                if write_mode == "sync":
                    res_list = Parallel(n_jobs=cfg.N_JOBS)(
                        delayed(sync_one_run)(s, r, cfg)
                        for s, r in tqdm(tasks, desc=f"{scenario_name} sync 并行进度")
                    )
                else:
                    res_list = Parallel(n_jobs=cfg.N_JOBS)(
                        delayed(process_one_run)(s, r, cfg, record_existing_outputs=bool(record_existing))
                        for s, r in tqdm(tasks, desc=f"{scenario_name} 并行进度")
                    )
                for r in res_list:
                    if isinstance(r, dict):
                        all_meta.extend(r.get("meta", []))
                        all_audit.extend(r.get("audit", []))
            else:
                for s, r in tqdm(tasks, desc=f"{scenario_name} 串行进度"):
                    if write_mode == "sync":
                        res = sync_one_run(s, r, cfg)
                    else:
                        res = process_one_run(s, r, cfg, record_existing_outputs=bool(record_existing))
                    if isinstance(res, dict):
                        all_meta.extend(res.get("meta", []))
                        all_audit.extend(res.get("audit", []))

            out_file = cfg.OUTPUT_ROOT / f"lss_index_{cfg.SCENARIO_ID}_aligned.csv"
            new_df = pd.DataFrame(all_meta) if all_meta else pd.DataFrame()
            key_cols = ["subject", "run", "task", "label", "file"]
            if write_mode == "overwrite":
                # Overwrite mode: replace the aligned index with this run's meta only.
                if not new_df.empty:
                    new_df.to_csv(out_file, index=False)
                    logger.info(f"全量对齐索引已保存: {out_file}")
            else:
                # Append/sync mode: merge by a stable key to avoid duplicates.
                existing_df = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame()
                merged = _merge_append(existing_df, new_df, key_cols=key_cols)
                if not merged.empty:
                    merged.to_csv(out_file, index=False)
                    logger.info(f"对齐索引已更新: {out_file} (mode={write_mode}, rows={int(merged.shape[0])})")

            if all_audit:
                audit_file = cfg.OUTPUT_ROOT / f"lss_audit_{cfg.SCENARIO_ID}.csv"
                audit_new = pd.DataFrame(all_audit)
                if write_mode == "overwrite" or not audit_file.exists():
                    audit_new.to_csv(audit_file, index=False)
                else:
                    audit_old = pd.read_csv(audit_file) if audit_file.exists() else pd.DataFrame()
                    audit_merged = pd.concat([audit_old, audit_new], axis=0, ignore_index=True)
                    audit_merged.to_csv(audit_file, index=False)
                logger.info(f"LSS 审计日志已保存: {audit_file} (mode={write_mode})")

            # 生成被试级成功/失败清单（要求同时具备 EMO 与 SOC 的有效结果）
            # Subject-level status: EMO and SOC must both have at least one success/skipped audit entry.
            audit_file = cfg.OUTPUT_ROOT / f"lss_audit_{cfg.SCENARIO_ID}.csv"
            audit_df = pd.read_csv(audit_file) if audit_file.exists() else pd.DataFrame(all_audit)
            success_like = {"success", "skipped"}
            status_rows = []
            for row in presence_rows:
                sub = row["subject"]
                sub_audit = audit_df[audit_df["subject"] == sub] if not audit_df.empty else pd.DataFrame()

                emo_ok = bool(
                    not sub_audit.empty
                    and ((sub_audit["task"] == "EMO") & (sub_audit["status"].isin(success_like))).any()
                )
                soc_ok = bool(
                    not sub_audit.empty
                    and ((sub_audit["task"] == "SOC") & (sub_audit["status"].isin(success_like))).any()
                )

                lss_ok = bool(row["inputs_both_ok"] and emo_ok and soc_ok)
                note = []
                if not row["inputs_both_ok"]:
                    note.append("输入校验未通过")
                if row["inputs_both_ok"] and not emo_ok:
                    note.append("EMO无成功/跳过记录")
                if row["inputs_both_ok"] and not soc_ok:
                    note.append("SOC无成功/跳过记录")

                status_rows.append(
                    {
                        "subject": sub,
                        "emo_inputs_ok": row["emo_inputs_ok"],
                        "soc_inputs_ok": row["soc_inputs_ok"],
                        "inputs_both_ok": row["inputs_both_ok"],
                        "emo_lss_ok": emo_ok,
                        "soc_lss_ok": soc_ok,
                        "lss_status": "success" if lss_ok else "fail",
                        "lss_note": "；".join(note) if note else "EMO/SOC均通过",
                        "input_check_note": row["input_check_note"],
                    }
                )

            status_file = cfg.OUTPUT_ROOT / f"lss_subject_status_{cfg.SCENARIO_ID}.csv"
            pd.DataFrame(status_rows).to_csv(status_file, index=False)
            logger.info(f"LSS 被试成功/失败清单已保存: {status_file}")

        except Exception as e:
            logger.error(f"处理失败: {e}")


if __name__ == "__main__":
    run_main()
