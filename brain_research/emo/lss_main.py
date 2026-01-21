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
from tqdm import tqdm

try:
    from glm_config import Config, get_scenario_configs
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
     out_dir, unique_label, mask_img, config) = args

    try:
        # 验证输入参数
        if not fmri_path:
            logger.error(f"Trial {trial_idx} failed: fMRI 路径为空")
            return None
        
        if events_df is None or events_df.empty:
            logger.error(f"Trial {trial_idx} failed: 事件数据为空")
            return None
        
        if confounds_df is None:
            logger.error(f"Trial {trial_idx} failed: 头动数据为空")
            return None

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

        # 确保fmri_path是字符串格式（nilearn需要）
        fmri_path_str = str(fmri_path) if isinstance(fmri_path, Path) else fmri_path
        model.fit(fmri_path_str, events=lss_events, confounds=confounds_df)

        beta_map = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
        # 在文件名中添加场景标识符，避免左右脑结果覆盖
        if config.DATA_SPACE == 'surface':
            fname = f"beta_{unique_label}_{config.HEMI}{ext}"
        else:
            fname = f"beta_{unique_label}{ext}"
        out_path = out_dir / fname
        
        # 确保输出目录存在
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        beta_map.to_filename(out_path)

        return {"status": "success", "file": fname}

    except Exception as e:
        logger.error(f"Trial {trial_idx} failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# %% 3. Run 处理
def process_one_run(sub, run, config):
    try:
        # 获取路径 (config已适配 midprep/miniprep 分离)
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path = Path(fmri_path_s)
        event_path = Path(event_path_s)
        conf_path = Path(conf_path_s)

        # 验证路径格式
        if not isinstance(fmri_path, Path) or not isinstance(event_path, Path) or not isinstance(conf_path, Path):
            logger.error(f"Config Error {sub}: 路径格式错误")
            return []

        # Confounds 路径容错 (防止 miniprep 下的文件名有微小差异)
        if not conf_path.exists():
            # 尝试模糊匹配 (只要是在那个目录下的 .tsv)
            candidates = list(conf_path.parent.glob(f"*{config.RUN_MAP[run]['task']}*confounds_timeseries.tsv"))
            if candidates:
                conf_path = candidates[0]
                logger.info(f"使用备选 Confounds 路径: {conf_path.name}")
            else:
                logger.warning(f"[SKIP] Confounds missing: {conf_path.name}，且未找到备选文件")
                return []

    except Exception as e:
        logger.error(f"Config Error {sub}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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

    # 获取任务类型 (emo 或 soc)
    task_type = config.RUN_MAP[run]['folder_task']
    
    # 生成输出目录: BASE_DATA_DIR / lss_results / {task_type} / {sub}
    out_dir = config.OUTPUT_ROOT / task_type / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载 Events (包含重分类)
        events = pd.read_csv(event_path, sep='\t')
        if events.empty:
            logger.warning(f"[SKIP] 事件数据为空: {event_path.name}")
            return []
        events = glm_utils.reclassify_events(events)

        # 加载 Confounds (Friston24 等)
        confounds = glm_utils.load_complex_confounds(str(conf_path))
        if confounds.empty:
            logger.warning(f"[SKIP] 头动数据为空: {conf_path.name}")
            return []

    except Exception as e:
        logger.error(f"Data Load Error {sub}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []

    # 只使用trial_type和索引创建唯一标签，不使用pic_num
    events['unique_label'] = events['trial_type'] + '_tr' + events.index.astype(str)

    run_mask = None
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            run_mask = str(config.MNI_MASK_PATH)
        else:
            logger.warning(f"[SKIP] 未找到 MNI 掩码文件")
            return []

    results = []
    # 全量跑 vs 调试跑
    if config.DEBUG:
        indices = range(min(3, len(events)))
        logger.info(f"Debug mode: 处理前 {len(indices)} 个事件")
    else:
        indices = range(len(events))
        # 只打印Run级别的进度，不打印Trial级别的详细日志
        logger.debug(f"Processing {len(indices)} events for {sub} run {run}")

    for idx in indices:
        try:
            row = events.iloc[idx]

            # 跳过 'Other' 类型的事件
            if row['trial_type'] == 'Other':
                logger.debug(f"跳过 'Other' 类型的事件: {idx}")
                continue

            # 确保路径格式一致
            fmri_path_str = str(fmri_path)  # nilearn需要字符串路径
            out_dir_path = out_dir if isinstance(out_dir, Path) else Path(out_dir)
            
            args = (idx, fmri_path_str, events, confounds, out_dir_path, row['unique_label'], run_mask, config)
            res = process_single_trial_lss(args)

            if res:
                results.append({
                    'subject': sub, 'run': run,
                    'task': config.RUN_MAP[run]['task'],
                    'label': row['unique_label'],
                    'file': res['file']
                })
            if idx % 10 == 0: gc.collect()
        except Exception as e:
            logger.error(f"处理事件 {idx} 时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 继续处理下一个事件
            continue

    if not results:
        logger.warning(f"[SKIP] 未生成任何结果: {sub} run {run}")
    else:
        logger.info(f"完成处理 {len(results)} 个事件: {sub} run {run}")
    
    return results


# %% 4. 主程序
def run_main():
    try:
        # 获取所有场景的配置实例
        scenarios = get_scenario_configs()
        logger.info(f"开始处理 {len(scenarios)} 个场景")

        for i, config in enumerate(scenarios):
            scenario_name = f"{config.DATA_SPACE}{'_' + config.HEMI if config.DATA_SPACE == 'surface' else ''}"
            logger.info(f"\n=== 处理场景 {i+1}/{len(scenarios)}: {scenario_name} ===")

            try:
                tasks = [(s, r) for s in config.SUBJECTS for r in config.RUNS]
                logger.info(f"Start LSS: {len(tasks)} Tasks | Midprep(Img)+Miniprep(Conf)")

                if not tasks:
                    logger.warning("No tasks to process. Check your subject list and run configuration.")
                    continue

                all_meta = []

                if config.PARALLEL:
                    try:
                        # 使用tqdm显示并行处理的进度
                        logger.info(f"开始并行处理 {len(tasks)} 个任务")
                        res_list = Parallel(n_jobs=config.N_JOBS)(
                            delayed(process_one_run)(s, r, config) for s, r in tqdm(tasks, desc=f"{scenario_name} 处理进度")
                        )
                        for r in res_list:
                            if r:
                                all_meta.extend(r)
                    except Exception as e:
                        logger.error(f"并行处理失败: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        # 回退到串行处理
                        logger.info("回退到串行处理模式")
                        for s, r in tqdm(tasks, desc=f"{scenario_name} 串行处理进度"):
                            logger.info(f"Running {s} Run {r}")
                            res = process_one_run(s, r, config)
                            if res:
                                all_meta.extend(res)
                else:
                    # 使用tqdm显示串行处理的进度
                    for s, r in tqdm(tasks, desc=f"{scenario_name} 处理进度"):
                        logger.info(f"Running {s} Run {r}")
                        res = process_one_run(s, r, config)
                        if res:
                            all_meta.extend(res)

                if all_meta:
                    try:
                        # 在lss_results根目录下生成包含场景标识符的索引文件
                        out_file = config.OUTPUT_ROOT / f"lss_index_{config.SCENARIO_ID}.csv"
                        pd.DataFrame(all_meta).to_csv(out_file, index=False)
                        logger.info(f"完成场景 {scenario_name}。生成索引文件: {out_file}")
                        logger.info(f"处理完成: {len(all_meta)} 个结果")
                    except Exception as e:
                        logger.error(f"保存索引文件失败: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                else:
                    logger.warning(f"场景 {scenario_name} 未生成任何结果。请检查数据和配置。")

            except Exception as e:
                logger.error(f"处理场景 {scenario_name} 时失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info("\n=== 所有场景处理完成 ===")
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_main()