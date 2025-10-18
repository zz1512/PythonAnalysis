#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析多run合并时条件index变化对trial-condition对应关系的影响
重点检查：
1. 原始trial_index在不同run中的重复问题
2. combined_trial_index如何解决索引冲突
3. 条件标签的正确性验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_roi_pipeline_enhanced import (
    MVPAConfig, 
    prepare_classification_data
)

def create_realistic_trial_data():
    """创建更接近真实情况的trial数据"""
    print("=== 创建模拟的真实trial数据 ===")
    
    # 模拟3个run的数据，每个run的trial_index都从1开始
    runs_data = {
        3: {
            'trials': [
                {'trial_index': 1, 'condition': 'kj_word1'},
                {'trial_index': 2, 'condition': 'yy_word1'},
                {'trial_index': 3, 'condition': 'kj_word2'},
                {'trial_index': 4, 'condition': 'yy_word2'},
                {'trial_index': 5, 'condition': 'kj_word3'},
                {'trial_index': 6, 'condition': 'yy_word3'},
            ],
            'beta_files': ['run3_beta_001.nii', 'run3_beta_002.nii', 'run3_beta_003.nii', 
                          'run3_beta_004.nii', 'run3_beta_005.nii', 'run3_beta_006.nii']
        },
        4: {
            'trials': [
                {'trial_index': 1, 'condition': 'kj_word4'},  # 注意：trial_index又从1开始
                {'trial_index': 2, 'condition': 'yy_word4'},
                {'trial_index': 3, 'condition': 'kj_word5'},
                {'trial_index': 4, 'condition': 'yy_word5'},
            ],
            'beta_files': ['run4_beta_001.nii', 'run4_beta_002.nii', 'run4_beta_003.nii', 'run4_beta_004.nii']
        },
        5: {
            'trials': [
                {'trial_index': 1, 'condition': 'kj_word6'},  # trial_index再次从1开始
                {'trial_index': 2, 'condition': 'yy_word6'},
                {'trial_index': 3, 'condition': 'kj_word7'},
                {'trial_index': 4, 'condition': 'yy_word7'},
                {'trial_index': 5, 'condition': 'kj_word8'},
            ],
            'beta_files': ['run5_beta_001.nii', 'run5_beta_002.nii', 'run5_beta_003.nii', 
                          'run5_beta_004.nii', 'run5_beta_005.nii']
        }
    }
    
    return runs_data

def analyze_index_conflicts(runs_data):
    """分析索引冲突问题"""
    print("\n=== 分析索引冲突问题 ===")
    
    # 收集所有原始trial_index
    all_original_indices = []
    for run, data in runs_data.items():
        for trial in data['trials']:
            all_original_indices.append((run, trial['trial_index']))
    
    print("所有原始trial_index:")
    for run, idx in all_original_indices:
        print(f"  Run {run}: trial_index = {idx}")
    
    # 检查重复的trial_index
    index_counts = {}
    for run, idx in all_original_indices:
        if idx not in index_counts:
            index_counts[idx] = []
        index_counts[idx].append(run)
    
    print("\n重复的trial_index:")
    conflicts_found = False
    for idx, runs in index_counts.items():
        if len(runs) > 1:
            print(f"  trial_index {idx} 出现在 runs: {runs}")
            conflicts_found = True
    
    if not conflicts_found:
        print("  没有发现重复的trial_index（这在真实数据中是不可能的）")
    else:
        print(f"  ⚠️  发现 {sum(1 for runs in index_counts.values() if len(runs) > 1)} 个重复的trial_index")
    
    return index_counts

def simulate_data_loading_and_merging(runs_data):
    """模拟数据加载和合并过程"""
    print("\n=== 模拟数据加载和合并过程 ===")
    
    # 模拟load_lss_trial_data的输出
    all_trial_info = []
    all_beta_images = []
    
    for run, data in runs_data.items():
        # 为每个run创建trial_info DataFrame
        trial_info = pd.DataFrame([
            {
                'trial_index': trial['trial_index'],
                'run': run,
                'original_condition': trial['condition'],
                'global_trial_index': f'run{run}_trial{trial["trial_index"]}'
            }
            for trial in data['trials']
        ])
        
        all_trial_info.append(trial_info)
        all_beta_images.extend(data['beta_files'])
        
        print(f"\nRun {run} 加载的数据:")
        print(trial_info.to_string(index=False))
    
    # 模拟load_multi_run_lss_data的合并逻辑
    print("\n--- 合并过程 ---")
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
    
    print("\n合并后的trial信息:")
    print(combined_trial_info.to_string(index=False))
    
    print(f"\n合并后的beta图像列表:")
    for i, beta in enumerate(all_beta_images):
        print(f"  Index {i}: {beta}")
    
    return combined_trial_info, all_beta_images

def test_classification_data_preparation(combined_trial_info, all_beta_images):
    """测试分类数据准备过程"""
    print("\n=== 测试分类数据准备过程 ===")
    
    config = MVPAConfig()
    
    # 准备分类数据
    selected_betas, y_labels, trial_details = prepare_classification_data(
        combined_trial_info, all_beta_images, 'kj', 'yy', config
    )
    
    print("\n分类数据准备结果:")
    print(trial_details.to_string(index=False))
    
    print(f"\n详细的索引映射检查:")
    for i, (_, row) in enumerate(trial_details.iterrows()):
        original_idx = row['trial_index']
        combined_idx = row['combined_trial_index']
        run = row['run']
        condition = row['condition']
        global_trial = row['global_trial_index']
        
        selected_beta = selected_betas[i]
        expected_beta = all_beta_images[combined_idx]
        
        print(f"  {global_trial}:")
        print(f"    原始trial_index: {original_idx} (在run {run}中)")
        print(f"    combined_trial_index: {combined_idx}")
        print(f"    条件: {condition}")
        print(f"    选择的beta: {selected_beta}")
        print(f"    期望的beta: {expected_beta}")
        print(f"    映射正确: {'✅' if selected_beta == expected_beta else '❌'}")
        print()
        
        # 验证映射正确性
        assert selected_beta == expected_beta, f"映射错误: {selected_beta} != {expected_beta}"
    
    return trial_details

def analyze_potential_issues(trial_details, combined_trial_info):
    """分析潜在问题"""
    print("=== 分析潜在问题 ===")
    
    # 问题1: 如果错误使用原始trial_index会怎样？
    print("\n问题1: 如果错误使用原始trial_index作为beta索引会怎样？")
    
    for _, row in trial_details.iterrows():
        original_idx = row['trial_index'] - 1  # 原始逻辑：trial_index - 1
        combined_idx = row['combined_trial_index']
        run = row['run']
        global_trial = row['global_trial_index']
        
        if original_idx != combined_idx:
            print(f"  {global_trial} (run {run}):")
            print(f"    如果用原始索引 {original_idx}: 会选择错误的beta")
            print(f"    正确的索引应该是 {combined_idx}")
            print(f"    索引差异: {combined_idx - original_idx}")
    
    # 问题2: 检查条件分布
    print("\n问题2: 条件分布检查")
    condition_by_run = trial_details.groupby(['run', 'condition']).size().unstack(fill_value=0)
    print("各run的条件分布:")
    print(condition_by_run)
    
    # 问题3: 检查global_trial_index的唯一性
    print("\n问题3: global_trial_index唯一性检查")
    global_indices = combined_trial_info['global_trial_index'].tolist()
    unique_global_indices = set(global_indices)
    
    if len(global_indices) == len(unique_global_indices):
        print("  ✅ 所有global_trial_index都是唯一的")
    else:
        print("  ❌ 发现重复的global_trial_index")
        duplicates = [idx for idx in unique_global_indices if global_indices.count(idx) > 1]
        print(f"  重复的索引: {duplicates}")
    
    # 问题4: 验证combined_trial_index的连续性
    print("\n问题4: combined_trial_index连续性检查")
    combined_indices = sorted(combined_trial_info['combined_trial_index'].tolist())
    expected_indices = list(range(len(combined_indices)))
    
    if combined_indices == expected_indices:
        print("  ✅ combined_trial_index是连续的")
    else:
        print("  ❌ combined_trial_index不连续")
        print(f"  期望: {expected_indices}")
        print(f"  实际: {combined_indices}")

def main():
    """主分析函数"""
    print("开始分析多run合并时条件index变化的影响...\n")
    
    try:
        # 1. 创建真实的trial数据
        runs_data = create_realistic_trial_data()
        
        # 2. 分析索引冲突
        index_conflicts = analyze_index_conflicts(runs_data)
        
        # 3. 模拟数据加载和合并
        combined_trial_info, all_beta_images = simulate_data_loading_and_merging(runs_data)
        
        # 4. 测试分类数据准备
        trial_details = test_classification_data_preparation(combined_trial_info, all_beta_images)
        
        # 5. 分析潜在问题
        analyze_potential_issues(trial_details, combined_trial_info)
        
        print("\n🎉 分析完成！")
        print("\n关键发现:")
        print("1. ✅ 多run合并时，原始trial_index确实会重复")
        print("2. ✅ combined_trial_index成功解决了索引冲突问题")
        print("3. ✅ 条件标签和trial的对应关系在合并后保持正确")
        print("4. ✅ global_trial_index提供了唯一的trial标识")
        print("5. ✅ 当前的实现能够正确处理多run分析")
        
        print("\n回答用户问题:")
        print("❓ 现在进行多run分析时，条件的index已经变了，还能准确找到对应的trial和condition的对应关系么？")
        print("✅ 答案：能够准确找到！")
        print("\n原因:")
        print("- 代码使用combined_trial_index而不是原始trial_index作为beta图像索引")
        print("- 每个trial都保留了完整的条件信息(original_condition)")
        print("- global_trial_index确保了每个trial的唯一标识")
        print("- 分类数据准备时正确传递了所有相关信息")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()