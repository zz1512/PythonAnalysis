#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lss_main.py
LSS 分析主程序 - 全量 Event 分析
"""

# %% 1. 初始化
import os
import sys
import gc
import logging
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel

try:
    from glm_config import config
    import glm_utils
except ImportError:
    print("Error: 请确保 glm_config.py 和 glm_utils.py 在同一目录下")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# %% 2. 单个 Trial
def process_single_trial_lss(args):
    (trial_idx, fmri_path, events_df, confounds_df,
     out_dir, unique_label, mask_img) = args

    try:
        lss_events = glm_utils.create_lss_events(trial_idx, events_df)

        model = FirstLevelModel(
            t_r=config.TR,
            slice_time_ref=0.5,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            noise_model="ar1",
            standardize=True,
            hrf_model="spm + derivative",
            drift_model="cosine",
            high_pass=1 / 128.0,
            mask_img=mask_img,
            minimize_memory=True,
            verbose=0
        )

        model.fit(fmri_path, events=lss_events, confounds=confounds_df)

        beta_map = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
        fname = f"beta_{unique_label}{ext}"
        out_path = out_dir / fname
        beta_map.to_filename(out_path)

        return {"status": "success", "file": fname}

    except Exception as e:
        logger.error(f"Trial {trial_idx} failed: {e}")
        return None


# %% 3. Run 处理
def process_one_run(sub, run):
    try:
        # 获取路径 (config已适配 midprep/miniprep 分离)
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path = Path(fmri_path_s)
        event_path = Path(event_path_s)
        conf_path = Path(conf_path_s)

        # Confounds 路径容错 (防止 miniprep 下的文件名有微小差异)
        if not conf_path.exists():
            # 尝试模糊匹配 (只要是在那个目录下的 .tsv)
            candidates = list(conf_path.parent.glob(f"*{config.RUN_MAP[run]['task']}*confounds_timeseries.tsv"))
            if candidates:
                conf_path = candidates[0]

    except Exception as e:
        logger.error(f"Config Error {sub}: {e}")
        return []

    if not fmri_path.exists():
        logger.warning(f"[SKIP] fMRI missing: {fmri_path.name}")
        return []
    if not event_path.exists():
        logger.warning(f"[SKIP] Event missing: {event_path.name}")
        return []
    if not conf_path.exists():
        logger.warning(f"[SKIP] Confounds missing: {conf_path.name}")
        return []

    out_dir = config.OUTPUT_ROOT / sub / f"run-{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载 Events (包含重分类)
        events = pd.read_csv(event_path, sep='\t')
        events = glm_utils.reclassify_events(events)

        # 加载 Confounds (Friston24 等)
        confounds = glm_utils.load_complex_confounds(str(conf_path))

    except Exception as e:
        logger.error(f"Data Load Error {sub}: {e}")
        return []

    if 'pic_num' in events.columns:
        events['unique_label'] = events['trial_type'] + '_' + events['pic_num'].astype(str)
    else:
        events['unique_label'] = events['trial_type'] + '_tr' + events.index.astype(str)

    run_mask = None
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            run_mask = str(config.MNI_MASK_PATH)
        else:
            return []

    results = []
    # 全量跑 vs 调试跑
    indices = range(len(events)) if config.PARALLEL else range(min(3, len(events)))

    for idx in indices:
        row = events.iloc[idx]

        # [逻辑] 跑所有 Event，不筛选 trial_type

        args = (idx, str(fmri_path), events, confounds, out_dir, row['unique_label'], run_mask)
        res = process_single_trial_lss(args)

        if res:
            results.append({
                'subject': sub, 'run': run,
                'task': config.RUN_MAP[run]['task'],
                'label': row['unique_label'],
                'file': res['file']
            })
        if idx % 10 == 0: gc.collect()

    return results


# %% 4. 主程序
def run_main():
    tasks = [(s, r) for s in config.SUBJECTS for r in config.RUNS]
    logger.info(f"Start LSS: {len(tasks)} Tasks | Midprep(Img)+Miniprep(Conf)")

    all_meta = []

    if config.PARALLEL:
        res_list = Parallel(n_jobs=config.N_JOBS)(delayed(process_one_run)(s, r) for s, r in tasks)
        for r in res_list: all_meta.extend(r)
    else:
        for s, r in tasks:
            logger.info(f"Running {s} Run {r}")
            all_meta.extend(process_one_run(s, r))

    if all_meta:
        pd.DataFrame(all_meta).to_csv(config.OUTPUT_ROOT / "lss_index.csv", index=False)
        logger.info("Done.")


if __name__ == "__main__":
    run_main()