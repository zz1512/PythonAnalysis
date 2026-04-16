#!/usr/bin/env python3
"""
run_rsa_optimized.py
RSA 分析主程序 (Top-K & Item-wise 增强版) - 修复警告版

用途
- 对前后测（Pre: run1-2 / Post: run5-6）的 LSS 单 trial beta 做 ROI-level RSA。
- 输出：
  - summary 统计（用于快速看交互）
  - item-wise 明细（用于 LMM：`similarity ~ condition * time + (1|subject) + (1|item)`）

输入（来自 rsa_config.py）
- `${PYTHON_METAPHOR_ROOT}/lss_betas_final/lss_metadata_index_final.csv`：trial 索引表
- `${PYTHON_METAPHOR_ROOT}/stimuli_template.csv`：刺激模板（确保 trial 对齐）
- ROI masks（例如 `${PYTHON_METAPHOR_ROOT}/roi_masks_final/*.nii.gz`）

输出（来自 rsa_config.py 的 OUTPUT_DIR）
- `rsa_itemwise_details.csv`：item-wise 数据（强烈建议保留，用于 LMM）
- `rsa_summary_stats.csv`：汇总统计

关键实现点（为什么这么写）
- `get_sorted_files()`：严格按 `stimuli_template.csv` 的顺序取 beta 文件，避免 trial 对齐错误
- 对未知/缺失匹配直接 raise：宁可早失败也不要悄悄错配导致假阳性
"""

import numpy as np
import pandas as pd
from pathlib import Path
# [修复1] 修改导入路径: 从 maskers 导入
from nilearn.maskers import NiftiMasker
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
import rsa_config as cfg
import logging
import time

# 配置日志
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_sorted_files(sub, stage, lss_df, template_df):
    """获取排序好的 Beta 文件路径"""
    runs = [1, 2] if stage == 'Pre' else [5, 6]

    # [修复2] 加上 .copy()，创建独立副本，消除 SettingWithCopyWarning
    sub_df = lss_df[
        (lss_df['subject'] == sub) &
        (lss_df['run'].isin(runs))
        ].copy()

    # 确保列是字符串格式并去空格 (这是你报错的那一行逻辑)
    sub_df['unique_label'] = sub_df['unique_label'].astype(str).str.strip()
    sub_df['condition'] = sub_df['condition'].astype(str).str.strip()

    sorted_files = []

    for _, row in template_df.iterrows():
        target_word = str(row['word_label']).strip()

        match = sub_df[
            (sub_df['unique_label'] == target_word)
        ]

        # 严格模式：必须找到且唯一
        if len(match) == 1:
            row_data = match.iloc[0]
            fname = row_data['beta_file']

            # 如果 CSV 里存的是绝对路径，直接用
            if Path(fname).is_absolute():
                fpath = Path(fname)
            else:
                # 如果是文件名，需要手动拼接 sub-xx 和 run-x 目录
                # 结构: LSS_ROOT / sub-xx / run-x / filename
                sub_folder = row_data['subject']  # e.g., sub-01
                run_folder = f"run-{row_data['run']}"  # e.g., run-1

                fpath = cfg.LSS_META_FILE.parent / sub_folder / run_folder / fname
            sorted_files.append(str(fpath))
        else:
            # 如果之前校验脚本通过了，这里理论上不会进
            # 但为了防止程序崩，可以抛出异常或记录日志
            logger.warning(f"Mismatch for {sub} {stage}: {target_word} (Matches: {len(match)})")
            # 这里如果不 append，列表长度就不对，后面会报错。
            # 建议 raise error 让用户去查数据
            raise ValueError(f"Data Mismatch in Execution: {sub} {target_word}")

    return sorted_files


def extract_itemwise_data(sim_mat, template_df, condition_name):
    """
    [新增] 提取项目级 (Item-wise) 数据
    返回: list of dicts [{'pair_id': 1, 'similarity': 0.5}, ...]
    """
    indices = template_df[template_df['condition'] == condition_name].index.tolist()
    if not indices: return []

    # 获取 pair_id 列表 (假设每两个是一个pair)
    # 取步长为2，获取 Topic 的行号
    topic_indices = indices[0::2]

    item_data = []
    for idx_in_matrix in topic_indices:
        # 获取矩阵中该 Topic 的 Pair_ID
        # 注意 template_df 的 index 可能和矩阵 index (0-179) 对应
        # template_df.iloc[idx_in_matrix] 是第 idx 行
        pair_id = template_df.iloc[idx_in_matrix]['pair_id']
        word_label = template_df.iloc[idx_in_matrix]['word_label']

        # 提取相似度: (i, i+1)
        sim_val = sim_mat[idx_in_matrix, idx_in_matrix + 1]

        item_data.append({
            'pair_id': pair_id,
            'word_label': word_label,  # 记录 Topic 的词，方便认
            'similarity': sim_val
        })

    return item_data


def process_subject_roi(sub, roi_name, roi_mask_path, lss_df, template_df):
    """
    处理单个被试、单个 ROI (集成 Top-K 逻辑)
    """
    results_summary = []  # 存平均值 (用于做 Bar Plot)
    results_itemwise = []  # 存每个配对的值 (用于做散点图/LMM)

    try:
        # 1. 初始化 Masker (用于提取全 ROI 数据)
        # 注意：这里先不 standardize，等选完体素后再 standardize 可能更稳妥，
        # 但 nilearn masker 通常是一步到位的。
        # 策略：先用 Masker 提取 Raw Signal，手动选列，再 Z-score。
        masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=False)

        # ================= Top-K 核心逻辑 =================
        voxel_indices = None

        if cfg.USE_TOP_K:
            # A. 找到该被试对应的一阶 T-map
            contrast_name = cfg.ROI_CONTRAST_MAP.get(roi_name)
            # 假设文件名结构: sub-01_Metaphor_gt_Spatial_stat.nii.gz (根据你实际GLM输出修改)
            # 这里使用了 glob 来模糊匹配，防止后缀差异
            t_map_candidates = list(cfg.GLM_1ST_DIR.glob(f"{sub}/*{contrast_name}*.nii*"))

            if not t_map_candidates:
                logger.warning(f"{sub} {roi_name}: T-map not found. Fallback to all voxels.")
            else:
                t_map_path = t_map_candidates[0]

                # B. 提取 ROI 内所有体素的 T 值
                # fit_transform 输入一张图，返回 [1, n_voxels]
                t_values = masker.fit_transform(str(t_map_path)).flatten()

                # C. 排序并取 Top-K 索引
                # argsort 从小到大排，取最后 K 个
                if len(t_values) > cfg.TOP_K_VOXELS:
                    top_k_indices = np.argsort(t_values)[-cfg.TOP_K_VOXELS:]
                    voxel_indices = top_k_indices
                else:
                    # 如果 ROI 本来就很小，就全取
                    voxel_indices = np.arange(len(t_values))
        # ================================================

        for stage in ['Pre', 'Post']:
            # 2. 获取文件
            files = get_sorted_files(sub, stage, lss_df, template_df)

            # 3. 提取数据 [n_trials, n_voxels_in_roi]
            time_series = masker.fit_transform(files)

            # 4. 应用 Top-K 筛选
            if voxel_indices is not None:
                time_series = time_series[:, voxel_indices]

            # 5. 手动标准化 (Z-score across voxels for each trial)
            # RSA 标准操作：让每个 Trial 的 pattern 均值为0
            # axis=1 (across voxels)
            time_series = (time_series - time_series.mean(axis=1, keepdims=True)) / (
                        time_series.std(axis=1, keepdims=True) + 1e-8)

            # 6. 计算 RDM
            rdm_vec = pdist(time_series, metric=cfg.METRIC)
            sim_mat = 1 - squareform(rdm_vec)  # 转为相似度

            # 7. 提取 Item-wise Data (高优需求 2)
            # 分别提取 Metaphor 和 Spatial 的每对数据
            for cond in ['Metaphor', 'Spatial']:
                items = extract_itemwise_data(sim_mat, template_df, cond)
                for item in items:
                    results_itemwise.append({
                        'subject': sub,
                        'roi': roi_name,
                        'stage': stage,
                        'condition': cond,
                        'pair_id': item['pair_id'],
                        'word_label': item['word_label'],
                        'similarity': item['similarity']
                    })

            # 8. 提取 Summary Data (Bar Plot 用)
            # 计算基线均值
            base_idx = template_df[template_df['condition'] == 'Baseline'].index
            base_sim = np.nan
            if len(base_idx) > 0:
                base_sub = sim_mat[np.ix_(base_idx, base_idx)]
                mask_diag = ~np.eye(base_sub.shape[0], dtype=bool)
                base_sim = base_sub[mask_diag].mean()

            # 计算隐喻/空间均值 (直接从 itemwise 聚合，保证一致性)
            meta_mean = np.mean([x['similarity'] for x in results_itemwise if
                                 x['stage'] == stage and x['condition'] == 'Metaphor' and x['subject'] == sub])
            spatial_mean = np.mean([x['similarity'] for x in results_itemwise if
                                    x['stage'] == stage and x['condition'] == 'Spatial' and x['subject'] == sub])

            results_summary.append({
                'subject': sub,
                'roi': roi_name,
                'stage': stage,
                'Sim_Metaphor': meta_mean,
                'Sim_Spatial': spatial_mean,
                'Sim_Baseline': base_sim,
                'n_voxels': time_series.shape[1]  # 记录用了多少个体素
            })

    except Exception as e:
        logger.error(f"Error processing {sub} {roi_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], []

    return results_summary, results_itemwise


def main():
    logger.info(">>> 开始 RSA 优化版分析 (Top-K & Item-wise) <<<")
    t0 = time.time()

    lss_df = pd.read_csv(cfg.LSS_META_FILE)
    lss_df['unique_label'] = lss_df['unique_label'].astype(str)
    template_df = pd.read_csv(cfg.STIMULI_TEMPLATE)

    tasks = []
    for sub in cfg.SUBJECTS:
        for roi_name, roi_path in cfg.ROI_MASKS.items():
            tasks.append((sub, roi_name, roi_path))

    # 并行执行
    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_subject_roi)(sub, name, path, lss_df, template_df)
        for sub, name, path in tasks
    )

    # 拆解结果
    final_summary = []
    final_itemwise = []

    for res_sum, res_item in results:
        final_summary.extend(res_sum)
        final_itemwise.extend(res_item)

    # 保存 Summary (用于 ANOVA/t-test)
    if final_summary:
        df_sum = pd.DataFrame(final_summary)
        out_sum = cfg.OUTPUT_DIR / "rsa_summary_stats.csv"
        df_sum.to_csv(out_sum, index=False)
        logger.info(f"Summary 结果已保存: {out_sum}")

    # 保存 Item-wise (用于 云雨图/LMM/散点图)
    if final_itemwise:
        df_item = pd.DataFrame(final_itemwise)
        out_item = cfg.OUTPUT_DIR / "rsa_itemwise_details.csv"
        df_item.to_csv(out_item, index=False)
        logger.info(f"Item-wise 结果已保存: {out_item}")

    logger.info(f"耗时: {time.time() - t0:.2f}秒")


if __name__ == "__main__":
    main()
