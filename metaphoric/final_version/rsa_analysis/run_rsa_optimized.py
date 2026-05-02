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
import argparse
import traceback

# 配置日志
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_CONDITION_ORDER = ["Metaphor", "Spatial", "Baseline"]
MAIN_RSA_CONDITIONS = {"Metaphor", "Spatial"}
BASELINE_CONDITION = "Baseline"
CONDITION_ALIASES = {
    "metaphor": "Metaphor",
    "yy": "Metaphor",
    "yyw": "Metaphor",
    "yyew": "Metaphor",
    "spatial": "Spatial",
    "kj": "Spatial",
    "kjw": "Spatial",
    "kjew": "Spatial",
    "baseline": "Baseline",
    "base": "Baseline",
    "bl": "Baseline",
    "jx": "Baseline",
    "nonlink": "Baseline",
    "no_link": "Baseline",
    "unlinked": "Baseline",
}


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


def build_condition_pairs(template_df, condition_name):
    subset = template_df[template_df['condition_norm'] == condition_name].copy()
    if subset.empty:
        return []

    sort_col = 'sort_id' if 'sort_id' in subset.columns else None
    pair_specs = []
    for pair_id, pair_rows in subset.groupby('pair_id', sort=False):
        if sort_col is not None:
            pair_rows = pair_rows.sort_values(sort_col)
        indices = pair_rows.index.to_list()
        if len(indices) != 2:
            raise ValueError(
                f"Template pair_id={pair_id} under {condition_name} must contain exactly 2 rows, got {len(indices)}."
            )
        pair_specs.append(
            {
                'pair_id': pair_id,
                'index_a': int(indices[0]),
                'index_b': int(indices[1]),
                'word_label': str(pair_rows.iloc[0]['word_label']),
                'partner_label': str(pair_rows.iloc[1]['word_label']),
            }
        )
    return pair_specs


def get_available_conditions(template_df):
    """Return analysis conditions in a stable paper-friendly order."""
    available = {str(item).strip() for item in template_df["condition_norm"].dropna().unique()}
    ordered = [cond for cond in DEFAULT_CONDITION_ORDER if cond in available]
    return ordered


def normalize_template_conditions(template_df):
    out = template_df.copy()
    out["condition_norm"] = out["condition"].astype(str).str.strip().str.lower().map(CONDITION_ALIASES)
    missing = out["condition_norm"].isna()
    if missing.any():
        unknown = sorted(out.loc[missing, "condition"].astype(str).unique().tolist())
        raise ValueError(f"Unknown template conditions: {unknown}")
    return out


def extract_baseline_itemwise_data(sim_mat, template_df):
    """
    Baseline follows the same pair_id logic as the learned conditions when
    the template encodes A/B pseudo-pairs (e.g., jx_1 with jx_2).
    """
    pair_specs = build_condition_pairs(template_df, BASELINE_CONDITION)
    if not pair_specs:
        return []

    item_data = []
    for pair_spec in pair_specs:
        sim_val = sim_mat[pair_spec['index_a'], pair_spec['index_b']]
        item_data.append({
            "pair_id": pair_spec['pair_id'],
            "item": f"Baseline:{pair_spec['pair_id']}",
            "word_label": pair_spec['word_label'],
            "partner_label": pair_spec['partner_label'],
            "similarity": sim_val,
        })
    return item_data


def extract_itemwise_data(sim_mat, template_df, condition_name):
    """
    [新增] 提取项目级 (Item-wise) 数据
    返回: list of dicts [{'pair_id': 1, 'similarity': 0.5}, ...]
    """
    pair_specs = build_condition_pairs(template_df, condition_name)
    if not pair_specs:
        return []

    item_data = []
    for pair_spec in pair_specs:
        sim_val = sim_mat[pair_spec['index_a'], pair_spec['index_b']]
        item_data.append({
            'pair_id': pair_spec['pair_id'],
            'item': f"{condition_name}:{pair_spec['pair_id']}",
            'word_label': pair_spec['word_label'],
            'partner_label': pair_spec['partner_label'],
            'similarity': sim_val
        })

    return item_data


def process_subject_roi(sub, roi_name, roi_mask_path, lss_df, template_df, *, allow_partial=False):
    """
    处理单个被试、单个 ROI (集成 Top-K 逻辑)
    """
    results_summary = []  # 主结果 summary（Metaphor / Spatial）
    results_itemwise = []  # 主结果 item-wise（默认主链包含 Metaphor / Spatial / Baseline）
    baseline_summary = []  # baseline 控制单独导出，便于单独汇报
    baseline_itemwise = []  # 同时保留 baseline 单独文件，便于显式控制分析

    try:
        available_conditions = get_available_conditions(template_df)

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
            # 默认提取模板里存在的全部核心条件：Metaphor / Spatial / Baseline
            for cond in available_conditions:
                if cond == "Baseline":
                    items = extract_baseline_itemwise_data(sim_mat, template_df)
                else:
                    items = extract_itemwise_data(sim_mat, template_df, cond)
                for item in items:
                    record = {
                        'subject': sub,
                        'roi': roi_name,
                        'stage': stage,
                        'condition': cond,
                        'pair_id': item['pair_id'],
                        'item': item['item'],
                        'word_label': item['word_label'],
                        'partner_label': item['partner_label'],
                        'similarity': item['similarity'],
                        'pairing_scheme': 'template_pair' if cond == BASELINE_CONDITION else 'learned_pair',
                    }
                    results_itemwise.append(record)
                    if cond == BASELINE_CONDITION:
                        baseline_itemwise.append(record)

            # 8. 提取 Summary Data (Bar Plot 用)
            # 直接从 itemwise 聚合，保证主 summary 与 LMM 输入完全一致。
            def _mean_for(cond_name):
                values = [
                    x['similarity'] for x in results_itemwise
                    if x['stage'] == stage and x['condition'] == cond_name and x['subject'] == sub
                ]
                return float(np.mean(values)) if values else np.nan

            def _baseline_mean():
                values = [
                    x['similarity'] for x in baseline_itemwise
                    if x['stage'] == stage and x['subject'] == sub
                ]
                return float(np.mean(values)) if values else np.nan

            meta_mean = _mean_for('Metaphor')
            spatial_mean = _mean_for('Spatial')

            results_summary.append({
                'subject': sub,
                'roi': roi_name,
                'stage': stage,
                'Sim_Metaphor': meta_mean,
                'Sim_Spatial': spatial_mean,
                'n_voxels': time_series.shape[1]  # 记录用了多少个体素
            })
            if available_conditions and BASELINE_CONDITION in available_conditions:
                baseline_summary.append({
                    'subject': sub,
                    'roi': roi_name,
                    'stage': stage,
                    'Sim_Baseline': _baseline_mean(),
                    'n_voxels': time_series.shape[1],
                    'pairing_scheme': 'template_pair',
                })

    except Exception as e:
        logger.error(f"Error processing {sub} {roi_name}: {e}")
        failure = {
            "roi_set": cfg.ROI_SET,
            "subject": sub,
            "roi": roi_name,
            "roi_mask_path": str(roi_mask_path),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        return [], [], [], [], failure

    return results_summary, results_itemwise, baseline_summary, baseline_itemwise, None


def _expected_pair_counts(template_df):
    counts = {}
    for condition in get_available_conditions(template_df):
        if condition == BASELINE_CONDITION:
            counts[condition] = len(build_condition_pairs(template_df, BASELINE_CONDITION))
        else:
            counts[condition] = len(build_condition_pairs(template_df, condition))
    return counts


def _build_completeness_manifest(itemwise_rows, subjects, roi_names, template_df):
    expected_counts = _expected_pair_counts(template_df)
    observed = pd.DataFrame(itemwise_rows)
    rows = []
    for sub in subjects:
        for roi in roi_names:
            for stage in ["Pre", "Post"]:
                for condition, expected_pairs in expected_counts.items():
                    if observed.empty:
                        subset = observed
                    else:
                        subset = observed[
                            observed["subject"].astype(str).eq(str(sub))
                            & observed["roi"].astype(str).eq(str(roi))
                            & observed["stage"].astype(str).eq(stage)
                            & observed["condition"].astype(str).eq(condition)
                        ]
                    observed_pairs = int(subset["pair_id"].nunique()) if not subset.empty and "pair_id" in subset.columns else 0
                    rows.append(
                        {
                            "roi_set": cfg.ROI_SET,
                            "subject": sub,
                            "roi": roi,
                            "stage": stage,
                            "condition": condition,
                            "expected_pairs": int(expected_pairs),
                            "observed_pairs": observed_pairs,
                            "complete": bool(observed_pairs == int(expected_pairs)),
                        }
                    )
    return pd.DataFrame(rows)


def _write_qc_table(frame, filename):
    out_local = cfg.OUTPUT_DIR / filename
    frame.to_csv(out_local, sep="\t", index=False)
    root_qc = cfg.OUTPUT_DIR.parent
    out_root = root_qc / filename
    frame.to_csv(out_root, sep="\t", index=False)
    tagged = root_qc / f"{Path(filename).stem}_{cfg.ROI_SET}{Path(filename).suffix}"
    frame.to_csv(tagged, sep="\t", index=False)
    logger.info(f"QC saved: {out_local}")


def main():
    parser = argparse.ArgumentParser(description="Run ROI-level RSA with explicit completeness QC.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Continue when a subject x ROI cell fails; failed cells are written to rsa_failed_cells.tsv. Default is fail-fast.",
    )
    args = parser.parse_args()

    logger.info(">>> 开始 RSA 优化版分析 (Top-K & Item-wise) <<<")
    t0 = time.time()

    lss_df = pd.read_csv(cfg.LSS_META_FILE)
    lss_df['unique_label'] = lss_df['unique_label'].astype(str)
    template_df = pd.read_csv(cfg.STIMULI_TEMPLATE).reset_index(drop=True)
    template_df = normalize_template_conditions(template_df)

    tasks = []
    for sub in cfg.SUBJECTS:
        for roi_name, roi_path in cfg.ROI_MASKS.items():
            tasks.append((sub, roi_name, roi_path))

    # 并行执行
    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_subject_roi)(sub, name, path, lss_df, template_df, allow_partial=args.allow_partial)
        for sub, name, path in tasks
    )

    # 拆解结果
    final_summary = []
    final_itemwise = []
    final_baseline_summary = []
    final_baseline_itemwise = []
    failed_cells = []

    for res_sum, res_item, res_base_sum, res_base_item, failure in results:
        final_summary.extend(res_sum)
        final_itemwise.extend(res_item)
        final_baseline_summary.extend(res_base_sum)
        final_baseline_itemwise.extend(res_base_item)
        if failure:
            failed_cells.append(failure)

    failed_df = pd.DataFrame(
        failed_cells,
        columns=["roi_set", "subject", "roi", "roi_mask_path", "error_type", "error_message", "traceback"],
    )
    _write_qc_table(failed_df, "rsa_failed_cells.tsv")
    completeness_df = _build_completeness_manifest(
        final_itemwise,
        cfg.SUBJECTS,
        list(cfg.ROI_MASKS.keys()),
        template_df,
    )
    _write_qc_table(completeness_df, "rsa_completeness_manifest.tsv")
    incomplete = completeness_df[~completeness_df["complete"]]
    if not incomplete.empty:
        logger.warning(f"RSA completeness has {len(incomplete)} incomplete subject x ROI x stage x condition cells.")
        if not args.allow_partial:
            raise RuntimeError(
                "RSA completeness check failed. Re-run with --allow-partial only if you intend to keep partial outputs."
            )

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

    if final_baseline_summary:
        df_base_sum = pd.DataFrame(final_baseline_summary)
        out_base_sum = cfg.OUTPUT_DIR / "rsa_baseline_summary_stats.csv"
        df_base_sum.to_csv(out_base_sum, index=False)
        logger.info(f"Baseline summary 结果已保存: {out_base_sum}")

    if final_baseline_itemwise:
        df_base_item = pd.DataFrame(final_baseline_itemwise)
        out_base_item = cfg.OUTPUT_DIR / "rsa_baseline_itemwise_details.csv"
        df_base_item.to_csv(out_base_item, index=False)
        logger.info(f"Baseline item-wise 结果已保存: {out_base_item}")

    logger.info(f"耗时: {time.time() - t0:.2f}秒")


if __name__ == "__main__":
    main()
