#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_missing_files.py
功能：
1. 分析为什么矩阵计算时大量被试被剔除。
2. 检查硬盘上是否真的缺失了 Beta 文件。
3. 统计哪些刺激 (Trial) 是"重灾区"。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import os

# ================= 配置区 =================
# 1. 索引文件路径 (请修改为你正在分析的那个 aligned csv)
# 这里填写 Volume 的索引文件
ALIGNED_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"

# 2. LSS 结果根目录
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")

# 3. 任务
TASK = "EMO"


# ==========================================

def resolve_path(row):
    """尝试构建文件路径"""
    # 路径策略 1: task/sub/file
    p1 = LSS_ROOT / row['task'].lower() / row['subject'] / row['file']
    if p1.exists(): return p1

    # 路径策略 2: task/sub/run/file
    p2 = LSS_ROOT / row['task'].lower() / row['subject'] / f"run-{row['run']}" / row['file']
    if p2.exists(): return p2

    # 路径策略 3: EMO/sub/file (大写)
    p3 = LSS_ROOT / row['task'] / row['subject'] / row['file']
    if p3.exists(): return p3

    # 返回最可能的路径 (策略2) 用于打印报错
    return p2


def main():
    print(f"正在读取索引: {ALIGNED_CSV}")
    if not Path(ALIGNED_CSV).exists():
        print("❌ 文件不存在，请检查路径。")
        return

    df = pd.read_csv(ALIGNED_CSV)
    df = df[df["task"] == TASK].copy()

    # 1. 确定公共刺激目标
    # 假设我们还是用 90% 的阈值，看看哪些刺激是“应该存在”的
    all_subjects = df['subject'].unique()
    print(f"总被试数: {len(all_subjects)}")

    stim_counts = df['stimulus_content'].value_counts()
    threshold = int(len(all_subjects) * 0.9)
    # 目标刺激列表
    target_stimuli = stim_counts[stim_counts >= threshold].index.tolist()
    target_stimuli.sort()

    print(f"目标公共刺激数 (Target Stimuli): {len(target_stimuli)}")
    print(f"  (定义为至少 {threshold} 人拥有的刺激)")

    # 2. 建立内存索引
    # Key: (subject, stimulus_content) -> Row Data
    db = {}
    for _, row in df.iterrows():
        db[(row['subject'], row['stimulus_content'])] = row

    # 3. 开始诊断
    missing_stats = defaultdict(list)  # Key: Stimulus, Value: [List of Subjects missing it]
    subject_stats = defaultdict(list)  # Key: Subject, Value: [List of missing stimuli]

    print("\n>>> 开始逐个文件核查硬盘...")

    for sub in tqdm(all_subjects):
        for stim in target_stimuli:
            key = (sub, stim)

            # 情况 A: CSV 里压根没记录 (LSS 没跑出来)
            if key not in db:
                missing_stats[stim].append(sub)
                subject_stats[sub].append(stim)
                continue

            # 情况 B: CSV 有记录，但硬盘没文件 (被删了或路径不对)
            row = db[key]
            fpath = resolve_path(row)

            if not fpath.exists():
                missing_stats[stim].append(sub)
                subject_stats[sub].append(stim)

    # 4. 输出诊断报告
    n_perfect = len(all_subjects) - len(subject_stats)
    print(f"\n{'=' * 20} 诊断报告 {'=' * 20}")
    print(f"完美被试 (所有文件齐全): {n_perfect}")
    print(f"缺失被试 (至少缺1个文件): {len(subject_stats)}")

    print("\n--- 🚨 最常缺失的刺激 (Top 10 Missing Stimuli) ---")
    # 按照缺失人数排序
    sorted_missing_stim = sorted(missing_stats.items(), key=lambda x: len(x[1]), reverse=True)
    for stim, subs in sorted_missing_stim[:10]:
        print(f"  {stim:<40} : 缺失 {len(subs)} 人")

    print("\n--- 👤 缺失最严重的被试 (Top 10 Bad Subjects) ---")
    # 按照缺失文件数排序
    sorted_bad_subs = sorted(subject_stats.items(), key=lambda x: len(x[1]), reverse=True)
    for sub, missing_list in sorted_bad_subs[:10]:
        print(f"  {sub:<15} : 缺失 {len(missing_list)} / {len(target_stimuli)} 个文件")

    # 5. 给出挽救建议
    print("\n" + "=" * 40)
    print("💡 挽救建议 (Action Plan)")

    # 策略 A: 剔除刺激
    if sorted_missing_stim:
        worst_stim, worst_subs = sorted_missing_stim[0]
        if len(worst_subs) > 50:
            print(f"1. 【强烈建议】剔除刺激 '{worst_stim}'！")
            print(f"   它导致了 {len(worst_subs)} 人被剔除。如果在代码中删掉这个刺激，这些人可能就回来了。")

    # 策略 B: 检查路径
    print("2. 如果所有人都显示缺失，请检查代码里的 resolve_path 逻辑是否匹配你的文件夹结构。")

    # 策略 C: 重新运行 LSS
    print("3. 检查那些缺失严重的被试，LSS 运行日志是否有报错。")


if __name__ == "__main__":
    main()