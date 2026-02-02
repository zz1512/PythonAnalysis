#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_subject_to_group.py
功能：计算每个被试与群体平均激活模式的相似性 (Subject-to-Group Correlation)。
特点：
1. 采用 Leave-One-Out (LOO) 策略，去除自身的循环偏差。
2. 综合多个公共刺激 (Top K Stimuli) 的结果，避免单张图片的偶然性。
3. 快速定位 "Bad Subjects" (离群值)。
"""

import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from qc_utils import load_gifti_flatten, resolve_beta_path, safe_corr

# ================= 配置区 =================
# 必须与你服务器的实际路径一致
ALIGNED_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"
LSS_ROOT = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"


# ==========================================

def run_group_check(aligned_csv, lss_root, task, n_stimuli=3, min_frac=0.9):
    print(f"--- Starting Subject-to-Group Check (Task: {task}) ---")

    df = pd.read_csv(aligned_csv)
    df = df[df["task"] == task].copy()

    # 1. 找出覆盖率最高的 Top K 刺激
    cnt = Counter(df["stimulus_content"].tolist())
    all_subs = df["subject"].unique()
    threshold = int(len(all_subs) * min_frac)

    # 筛选出至少覆盖 90% 被试的刺激，并按出现次数排序
    common_candidates = [k for k, v in cnt.most_common() if v >= threshold]

    if not common_candidates:
        print("❌ 错误: 没有找到满足阈值的公共刺激。")
        return

    # 取前 N 个，不足则取全部
    target_stimuli = common_candidates[:n_stimuli]
    print(f"选取 Top {len(target_stimuli)} 个刺激进行验证: {target_stimuli}")

    # 用于存储每个被试的平均相关性
    # Key: subject, Value: list of correlations (one per stimulus)
    subject_scores = {sub: [] for sub in all_subs}

    # 2. 逐个刺激进行 LOO 分析
    for stim in target_stimuli:
        print(f"\n处理刺激: {stim} ...")

        # 2.1 加载该刺激下所有被试的数据
        stim_df = df[df["stimulus_content"] == stim]

        # 暂存数据: list of (subject, vector)
        loaded_data = []

        for _, row in tqdm(stim_df.iterrows(), total=len(stim_df), desc="Loading"):
            p = resolve_beta_path(lss_root, task, row["subject"], row["run"], row["file"])
            if p:
                vec = load_gifti_flatten(p)
                if vec is not None:
                    loaded_data.append((row["subject"], vec))

        if not loaded_data:
            continue

        # 2.2 构建矩阵 (N_subs, N_vertices)
        subs_in_batch = [x[0] for x in loaded_data]
        matrix = np.stack([x[1] for x in loaded_data])

        # 2.3 计算 Leave-One-Out Mean
        # sum_vec: 所有人的总和
        sum_vec = matrix.sum(axis=0)
        n = matrix.shape[0]

        # 遍历每一个人，计算 (总和 - 自己) / (N-1)
        for i, sub in enumerate(subs_in_batch):
            my_vec = matrix[i]
            # LOO Mean: 除去自己后的平均
            loo_mean = (sum_vec - my_vec) / (n - 1)

            r = safe_corr(my_vec, loo_mean)
            subject_scores[sub].append(r)

    # 3. 汇总结果
    final_stats = []
    for sub, scores in subject_scores.items():
        if scores:
            # 取该被试在不同刺激下的平均表现
            avg_r = np.nanmean(scores)
            final_stats.append({'subject': sub, 'r_group': avg_r, 'n_stims': len(scores)})

    res_df = pd.DataFrame(final_stats).sort_values('r_group')

    # 4. 打印报告
    print(f"\n=== Subject-to-Group Correlation Summary ({len(res_df)} subjects) ===")
    print(res_df['r_group'].describe())

    # 找出离群值 (例如 < 0.1 或 Mean - 2SD)
    mean_val = res_df['r_group'].mean()
    std_val = res_df['r_group'].std()
    cutoff = mean_val - 2 * std_val
    # 设定一个硬阈值作为双重保险
    hard_limit = 0.1

    bad_subs = res_df[(res_df['r_group'] < cutoff) | (res_df['r_group'] < hard_limit)]

    print(f"\n📉 表现最差的 10 个被试:")
    print(res_df.head(10))

    print(f"\n⚠️  建议剔除的离群被试 (r < {max(cutoff, hard_limit):.3f}): {len(bad_subs)} 个")
    if not bad_subs.empty:
        print(bad_subs['subject'].tolist())

    # 保存结果
    out_file = f"quality_check_subject_to_group_{task}.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\n详细结果已保存至: {out_file}")


if __name__ == "__main__":
    # 检查 EMO 任务，取前 3 个最常见的刺激做平均
    run_group_check(
        aligned_csv=ALIGNED_CSV,
        lss_root=LSS_ROOT,
        task="EMO",
        n_stimuli=3
    )
