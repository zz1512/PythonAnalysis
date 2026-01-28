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
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn import surface
from tqdm import tqdm

try:
    from glm_config import get_scenario_configs
    import glm_utils
except ImportError:
    print("Error: 请确保 glm_config.py 和 glm_utils.py 在同一目录下")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_bids_val(val):
    """清理 BIDS 表格中的字符串 (去除 .png 后缀, 去除空格)"""
    s = str(val).strip()
    if s.endswith('.png'):
        s = s[:-4]
    return s


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


def process_one_run(sub, run, config):
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

    # 4. Mask (仅 Volume)
    run_mask = None
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            run_mask = str(config.MNI_MASK_PATH)
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
    fmri_input = str(fmri_path)
    results = []

    # 始终全量跑
    indices = range(len(events))

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
    scenarios = get_scenario_configs()

    for cfg in scenarios:
        scenario_name = f"{cfg.DATA_SPACE}_{cfg.HEMI}" if cfg.DATA_SPACE == 'surface' else "volume"
        logger.info(f"\n>>> 正在处理: {scenario_name} (并行数: {cfg.N_JOBS})")

        # DEBUG 逻辑：限制被试数
        subjects = list(cfg.SUBJECTS)
        if getattr(cfg, 'DEBUG', False):
            debug_n_sub = getattr(cfg, 'DEBUG_N_SUBJECTS', 30)
            subjects = subjects[:min(debug_n_sub, len(subjects))]
            logger.info(f"[DEBUG] 限制被试数量: {len(subjects)} / {len(cfg.SUBJECTS)}")

        tasks = [(s, r) for s in subjects for r in cfg.RUNS]
        if not tasks:
            continue

        all_meta = []
        all_audit = []
        try:
            if cfg.PARALLEL:
                res_list = Parallel(n_jobs=cfg.N_JOBS)(
                    delayed(process_one_run)(s, r, cfg)
                    for s, r in tqdm(tasks, desc=f"{scenario_name} 并行进度")
                )
                for r in res_list:
                    if isinstance(r, dict):
                        all_meta.extend(r.get("meta", []))
                        all_audit.extend(r.get("audit", []))
            else:
                for s, r in tqdm(tasks, desc=f"{scenario_name} 串行进度"):
                    res = process_one_run(s, r, cfg)
                    if isinstance(res, dict):
                        all_meta.extend(res.get("meta", []))
                        all_audit.extend(res.get("audit", []))

            if all_meta:
                # 输出文件名改为 _aligned.csv，表示这已经是包含对齐信息的最终版索引
                out_file = cfg.OUTPUT_ROOT / f"lss_index_{cfg.SCENARIO_ID}_aligned.csv"
                pd.DataFrame(all_meta).to_csv(out_file, index=False)
                logger.info(f"全量对齐索引已保存: {out_file}")

            if all_audit:
                audit_file = cfg.OUTPUT_ROOT / f"lss_audit_{cfg.SCENARIO_ID}.csv"
                pd.DataFrame(all_audit).to_csv(audit_file, index=False)
                logger.info(f"LSS 审计日志已保存: {audit_file}")

        except Exception as e:
            logger.error(f"处理失败: {e}")


if __name__ == "__main__":
    run_main()
