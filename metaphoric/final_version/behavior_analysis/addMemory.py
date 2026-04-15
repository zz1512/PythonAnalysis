#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run7行为数据预处理脚本

功能描述:
    为run-7的行为数据添加memory列，用于标识被试是否记忆了特定刺激。
    基于action列的值（3.0表示记忆）生成二进制标识。

处理流程:
    1. 扫描所有被试文件夹（sub-01到sub-28）
    2. 查找每个被试的run-7事件文件
    3. 添加memory列（action=3.0时为1，否则为0）
    4. 保存更新后的文件

作者: 研究团队
版本: 1.0
日期: 2024
"""

import os
import glob
import pandas as pd

# === 配置参数 ===
BASE_DIR = r"E:/python_metaphor/data_events"  # 事件数据根目录
SUBJECT_RANGE = range(1, 29)     # 被试范围：sub-01到sub-28
TARGET_RUN = 7                   # 目标run编号
MEMORY_ACTION_VALUE = "3.0"      # 表示记忆的action值
def main():
    """
    主处理函数：为所有被试的run-7数据添加memory列
    """
    # === 文件收集阶段 ===
    file_list = []
    print(f"开始扫描被试文件夹: {BASE_DIR}")
    
    # 遍历所有被试文件夹
    for i in SUBJECT_RANGE:
        sub_folder = f"sub-{i:02d}"  # 格式化被试编号（如：sub-01, sub-02）
        folder_path = os.path.join(BASE_DIR, sub_folder)

        # 检查被试文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"跳过不存在的文件夹: {sub_folder}")
            continue

        # 查找目标run的事件文件
        pattern = os.path.join(folder_path, f"{sub_folder}_run-{TARGET_RUN}_events.tsv")
        tsv_files = glob.glob(pattern)
        
        if tsv_files:
            file_list.append(tsv_files[0])
            print(f"找到文件: {os.path.basename(tsv_files[0])}")
        else:
            print(f"未找到{sub_folder}的run-{TARGET_RUN}文件")
    
    print(f"\n共找到 {len(file_list)} 个文件需要处理")
    
    # === 数据处理阶段 ===
    processed_count = 0
    
    for file_path in file_list:
        try:
            # 读取原始事件文件
            df = pd.read_csv(file_path, sep="\t")
            print(f"\n处理文件: {os.path.basename(file_path)}")
            print(f"原始数据形状: {df.shape}")
            
            # 添加memory列：基于action值生成二进制标识
            # action=3.0表示被试记住了该刺激，标记为1；否则为0
            df["memory"] = (df["action"].astype(str) == MEMORY_ACTION_VALUE).astype(int)
            
            # 统计memory列的分布
            memory_count = df["memory"].sum()
            total_trials = len(df)
            print(f"记忆试次: {memory_count}/{total_trials} ({memory_count/total_trials*100:.1f}%)")
            
            # 保存更新后的文件
            df.to_csv(file_path, sep="\t", index=False)
            processed_count += 1
            print(f"文件已更新保存")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {processed_count}/{len(file_list)} 个文件")

if __name__ == "__main__":
    main()