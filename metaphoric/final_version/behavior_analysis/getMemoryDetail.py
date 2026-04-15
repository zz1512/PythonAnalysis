#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run7行为数据详细信息提取脚本

功能描述:
    为run-7的行为数据添加详细的记忆信息，包括图片名称和记忆动作详情。
    通过匹配TSV事件文件和CSV参考文件，提取记忆测试的反应和反应时数据。

处理流程:
    1. 扫描所有被试的run-7事件文件（TSV格式）
    2. 查找对应的参考文件（CSV格式）
    3. 生成图片文件名（trial_type + pic_num + .jpg）
    4. 从参考文件中提取记忆测试的反应和反应时
    5. 添加action和action_time列到事件文件

数据映射逻辑:
    - 优先使用ciyu列匹配eword7_RESP和eword7_RT
    - 如果ciyu匹配失败，使用word5_word列匹配word7_RESP和word7_RT
    - action列：3表示回忆成功，4表示回忆失败

作者: 研究团队
版本: 1.0
日期: 2024
"""

import os
import glob
import pandas as pd

# === 配置参数 ===
BASE_DIR = r"E:/python_metaphor/data_events"  # 事件数据根目录
TARGET_RUN = 7                   # 目标run编号
SUBJECT_RANGE = range(1, 29)     # 被试范围：sub-01到sub-28
def main():
    """
    主处理函数：为所有被试的run-7数据添加详细记忆信息
    """
    # === 文件映射阶段 ===
    file_mapping = {}
    print(f"开始扫描被试文件夹: {BASE_DIR}")
    
    # 遍历所有被试文件夹，建立TSV和CSV文件的映射关系
    for i in SUBJECT_RANGE:
        sub_folder = f"sub-{i:02d}"  # 格式化被试编号
        folder_path = os.path.join(BASE_DIR, sub_folder)

        # 检查被试文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"跳过不存在的文件夹: {sub_folder}")
            continue

        # 查找run-7的事件文件（TSV）和参考文件（CSV）
        tsv_pattern = os.path.join(folder_path, f"{sub_folder}_run-{TARGET_RUN}_events.tsv")
        csv_pattern = os.path.join(folder_path, f"{sub_folder}_run-{TARGET_RUN}_events.csv")
        
        tsv_files = glob.glob(tsv_pattern)
        csv_files = glob.glob(csv_pattern)

        # 确保TSV和CSV文件都存在才进行处理
        if tsv_files and csv_files:
            file_mapping[tsv_files[0]] = csv_files[0]
            print(f"找到文件对: {os.path.basename(tsv_files[0])} <-> {os.path.basename(csv_files[0])}")
        else:
            missing = []
            if not tsv_files: missing.append("TSV")
            if not csv_files: missing.append("CSV")
            print(f"跳过{sub_folder}: 缺少{'/'.join(missing)}文件")
    
    print(f"\n共找到 {len(file_mapping)} 对文件需要处理")

    # === 数据处理阶段 ===
    processed_count = 0
    
    for tsv_path, csv_path in file_mapping.items():
        try:
            print(f"\n处理文件: {os.path.basename(tsv_path)}")
            
            # 读取事件文件（TSV格式）
            df = pd.read_csv(tsv_path, sep="\t")
            print(f"原始数据形状: {df.shape}")
            
            # 生成图片文件名：trial_type + pic_num + .jpg
            df["pic_out"] = df["trial_type"].astype(str) + df["pic_num"].astype(str) + ".jpg"
            print(f"生成了 {len(df)} 个图片文件名")
            
            # 读取参考文件（CSV格式）
            ref_df = pd.read_csv(csv_path)
            print(f"参考文件形状: {ref_df.shape}")
            
            # === 建立映射字典 ===
            # 初始化映射字典
            ciyu_action_map = {}
            ciyu_time_map = {}
            word_action_map = {}
            word_time_map = {}
            
            # 方案1：使用ciyu列映射eword7的反应和反应时
            if {"ciyu", "eword7_RESP"}.issubset(ref_df.columns):
                ciyu_action_map = dict(zip(ref_df["ciyu"].astype(str), ref_df["eword7_RESP"]))
                ciyu_time_map = dict(zip(ref_df["ciyu"].astype(str), ref_df["eword7_RT"]))
                print(f"建立ciyu映射: {len(ciyu_action_map)} 条记录")
            
            # 方案2：使用word5_word列映射word7的反应和反应时
            if {"word5_word", "word7_RESP"}.issubset(ref_df.columns):
                word_action_map = dict(zip(ref_df["word5_word"].astype(str), ref_df["word7_RESP"]))
                word_time_map = dict(zip(ref_df["word5_word"].astype(str), ref_df["word7_RT"]))
                print(f"建立word映射: {len(word_action_map)} 条记录")
            
            # === 应用映射 ===
            # 优先使用ciyu映射，失败时使用word映射
            action_from_ciyu = df["pic_out"].astype(str).map(ciyu_action_map)
            time_from_ciyu = df["pic_out"].astype(str).map(ciyu_time_map)
            
            action_from_word = df["pic_out"].astype(str).map(word_action_map)
            time_from_word = df["pic_out"].astype(str).map(word_time_map)
            
            # 合并映射结果：优先ciyu，缺失时用word填补
            df["action"] = action_from_ciyu.where(action_from_ciyu.notna(), action_from_word)
            df["action_time"] = time_from_ciyu.where(time_from_ciyu.notna(), time_from_word)
            
            # 统计映射成功率
            action_success = df["action"].notna().sum()
            time_success = df["action_time"].notna().sum()
            total_trials = len(df)
            
            print(f"action映射成功: {action_success}/{total_trials} ({action_success/total_trials*100:.1f}%)")
            print(f"action_time映射成功: {time_success}/{total_trials} ({time_success/total_trials*100:.1f}%)")
            
            # 保存更新后的文件
            df.to_csv(tsv_path, sep="\t", index=False)
            processed_count += 1
            print(f"文件已更新保存: {tsv_path}")
            
        except Exception as e:
            print(f"处理文件 {tsv_path} 时出错: {e}")
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {processed_count}/{len(file_mapping)} 个文件")

if __name__ == "__main__":
    main()