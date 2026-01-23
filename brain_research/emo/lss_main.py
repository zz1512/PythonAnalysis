#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lss_main.py
LSS 分析主程序 - 终极伪装版 (Fake NIfTI Strategy)
彻底解决 GiftiImage 属性缺失报错 (shape, _data_cache, _dataobj)
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
    from glm_config import Config, get_scenario_configs
    import glm_utils
except ImportError:
    print("Error: 请确保 glm_config.py 和 glm_utils.py 在同一目录下")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def save_beta_as_gifti(beta_data, output_path):
    """
    辅助函数：将计算出的 Numpy 数组保存为 .gii 文件
    beta_data shape 可能是 (Vertices, 1, 1) 需要展平
    """
    data_flat = beta_data.flatten().astype(np.float32)
    # 创建 Gifti 数据数组
    da = nb.gifti.GiftiDataArray(data_flat, intent='NIFTI_INTENT_ESTIMATE')
    # 创建 Gifti 图片对象
    img = nb.gifti.GiftiImage(darrays=[da])
    # 保存
    nb.save(img, str(output_path))


def process_single_trial_lss(args):
    """单个 Trial 处理函数"""
    (trial_idx, fmri_input, events_df, confounds_df,
     out_dir, unique_label, mask_img, config) = args

    try:
        # 1. 准备数据对象
        if config.DATA_SPACE == 'volume':
            # Volume 模式：直接传路径，使用外部传入的 mask_img
            input_img = fmri_input
            current_mask = mask_img
        else:
            # ====================================================
            # 【Surface 模式】：伪装成 NIfTI + 强制全掩膜
            # ====================================================
            # 1. 读取 Surface 数据
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

            # 2. 变形为 4D 伪体积 (Vertices, 1, 1, Time)
            fake_vol_data = surf_data.reshape((n_vert, 1, 1, n_time))

            # 3. 包装成 NIfTI 对象
            affine = np.eye(4)
            input_img = nb.Nifti1Image(fake_vol_data, affine=affine)

            # 4. 【核心修复】创建一个全 1 的 Mask
            # 告诉 Nilearn：这里面全是脑子，没有空气，别给我 Mask 没了
            mask_data = np.ones((n_vert, 1, 1), dtype=np.int8)
            current_mask = nb.Nifti1Image(mask_data, affine=affine)
            # ====================================================

        # 2. 构建事件
        lss_events = glm_utils.create_lss_events(trial_idx, events_df)

        # 3. 建模
        model = FirstLevelModel(
            t_r=config.TR,
            slice_time_ref=0.5,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            noise_model="ar1",
            standardize=True,
            hrf_model="spm + derivative",
            drift_model="cosine",
            high_pass=1 / 128.0,
            mask_img=current_mask,  # 现在 Surface 也有了强制 Mask
            minimize_memory=False,
            verbose=0
        )

        # 4. 拟合
        model.fit(input_img, events=lss_events, confounds=confounds_df)

        # 5. 提取 Beta
        beta_img = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        # 6. 保存结果
        ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
        if config.DATA_SPACE == 'surface':
            fname = f"beta_{unique_label}_{config.HEMI}{ext}"
        else:
            fname = f"beta_{unique_label}{ext}"

        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if config.DATA_SPACE == 'volume':
            beta_img.to_filename(out_path)
        else:
            # Surface: 解包保存
            beta_data = beta_img.get_fdata()
            save_beta_as_gifti(beta_data, out_path)

        return {"status": "success", "file": fname}

    except Exception as e:
        logger.error(f"Trial {trial_idx} error: {e}")
        # import traceback
        # logger.debug(traceback.format_exc())
        return None

def process_one_run(sub, run, config):
    # 1. 路径获取
    try:
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)

        if not conf_path.exists():
            cand = list(conf_path.parent.glob(f"*{config.RUN_MAP[run]['task']}*confounds_timeseries.tsv"))
            if cand:
                conf_path = cand[0]
            else:
                return []

        if not fmri_path.exists() or not event_path.exists():
            return []
    except:
        return []

    # 2. 准备目录
    task_type = config.RUN_MAP[run]['folder_task']
    out_dir = config.OUTPUT_ROOT / task_type / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 加载数据
    try:
        events = pd.read_csv(event_path, sep='\t')
        if events.empty: return []
        events = glm_utils.reclassify_events(events)

        confounds = glm_utils.load_complex_confounds(str(conf_path))
        if confounds.empty: return []

        events['unique_label'] = events['trial_type'] + '_tr' + events.index.astype(str)
    except:
        return []

    # 4. Mask (仅 Volume)
    run_mask = None
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            run_mask = str(config.MNI_MASK_PATH)
        else:
            return []

    # 5. 传递给 Trial 的数据参数
    # 直接传路径字符串，让子进程去读取并伪装
    fmri_input = str(fmri_path)

    results = []
    is_debug = getattr(config, 'DEBUG', False)
    indices = range(min(3, len(events))) if is_debug else range(len(events))

    for idx in indices:
        try:
            row = events.iloc[idx]
            if row['trial_type'] == 'Other': continue

            args = (idx, fmri_input, events, confounds, out_dir, row['unique_label'], run_mask, config)
            res = process_single_trial_lss(args)

            if res:
                results.append({
                    'subject': sub, 'run': run,
                    'task': config.RUN_MAP[run]['task'],
                    'label': row['unique_label'],
                    'file': res['file']
                })
        except:
            continue

    gc.collect()
    return results


def run_main():
    scenarios = get_scenario_configs()

    for cfg in scenarios:
        scenario_name = f"{cfg.DATA_SPACE}_{cfg.HEMI}" if cfg.DATA_SPACE == 'surface' else "volume"
        logger.info(f"\n>>> 正在处理: {scenario_name} (并行数: {cfg.N_JOBS})")

        tasks = [(s, r) for s in cfg.SUBJECTS for r in cfg.RUNS]
        if not tasks: continue

        all_meta = []
        try:
            if cfg.PARALLEL:
                res_list = Parallel(n_jobs=cfg.N_JOBS)(
                    delayed(process_one_run)(s, r, cfg) for s, r in tqdm(tasks, desc=f"{scenario_name} 并行进度")
                )
                for r in res_list:
                    if r: all_meta.extend(r)
            else:
                for s, r in tqdm(tasks, desc=f"{scenario_name} 串行进度"):
                    res = process_one_run(s, r, cfg)
                    if res: all_meta.extend(res)

            if all_meta:
                out_file = cfg.OUTPUT_ROOT / f"lss_index_{cfg.SCENARIO_ID}.csv"
                pd.DataFrame(all_meta).to_csv(out_file, index=False)
                logger.info(f"索引已保存: {out_file}")

        except Exception as e:
            logger.error(f"处理失败: {e}")


if __name__ == "__main__":
    run_main()