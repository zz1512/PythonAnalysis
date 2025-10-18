#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多run分析时trial和condition的对应关系
验证在多run合并后，trial索引和条件标签的映射是否正确
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_roi_pipeline_enhanced import (
    MVPAConfig, 
    load_lss_trial_data, 
    load_multi_run_lss_data,
    prepare_classification_data
)

def create_mock_trial_data(run, num_trials=20):
    """创建模拟的trial数据"""
    conditions = ['kj', 'yy'] * (num_trials // 2)
    if num_trials % 2 == 1:
        conditions.append('kj')
    
    trial_data = []
    for i in range(num_trials):
        trial_data.append({
            'trial_index': i + 1,
            'run': run,
            'original_condition': conditions[i],
            'global_trial_index': f'run{run}_trial{i+1}'
        })
    
    return pd.DataFrame(trial_data)

def create_mock_beta_images(num_trials):
    """创建模拟的beta图像数据"""
    return [f'beta_{i+1}.nii' for i in range(num_trials)]

def test_single_run_mapping():
    """测试单run的trial-condition映射"""
    print("\n=== 测试单run的trial-condition映射 ===")
    
    # 创建模拟数据
    trial_info = create_mock_trial_data(run=3, num_trials=10)
    beta_images = create_mock_beta_images(10)
    
    print(f"原始trial信息:")
    print(trial_info[['trial_index', 'original_condition', 'global_trial_index']].to_string())
    
    # 测试分类数据准备
    config = MVPAConfig()
    selected_betas, y_labels, trial_details = prepare_classification_data(
        trial_info, beta_images, 'kj', 'yy', config
    )
    
    print(f"\n分类数据准备结果:")
    print(trial_details.to_string())
    
    # 验证映射关系
    print(f"\n验证结果:")
    print(f"- 选择的beta数量: {len(selected_betas)}")
    print(f"- 标签数量: {len(y_labels)}")
    print(f"- trial详情数量: {len(trial_details)}")
    
    # 检查trial_index和beta_images的对应关系
    for i, (_, row) in enumerate(trial_details.iterrows()):
        expected_beta_idx = row['trial_index'] - 1  # 单run时应该是trial_index-1
        actual_beta = selected_betas[i]
        expected_beta = beta_images[expected_beta_idx]
        print(f"  Trial {row['trial_index']} ({row['condition']}): 期望beta索引{expected_beta_idx}, 实际beta: {actual_beta}, 期望beta: {expected_beta}")
        assert actual_beta == expected_beta, f"Beta映射错误: {actual_beta} != {expected_beta}"
    
    print("✅ 单run映射测试通过!")
    return trial_info, beta_images

def test_multi_run_mapping():
    """测试多run的trial-condition映射"""
    print("\n=== 测试多run的trial-condition映射 ===")
    
    # 创建多个run的模拟数据
    runs = [3, 4, 5]
    all_trial_info = []
    all_beta_images = []
    
    for run in runs:
        num_trials = 8 + run  # 不同run有不同数量的trial
        trial_info = create_mock_trial_data(run, num_trials)
        beta_images = create_mock_beta_images(num_trials)
        
        all_trial_info.append(trial_info)
        all_beta_images.extend(beta_images)
        
        print(f"\nRun {run} 原始数据:")
        print(trial_info[['trial_index', 'original_condition', 'global_trial_index']].head().to_string())
    
    # 合并数据（模拟load_multi_run_lss_data的逻辑）
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
    
    print(f"\n合并后的trial信息 (前10行):")
    print(combined_trial_info[['trial_index', 'run', 'original_condition', 'global_trial_index', 'combined_trial_index']].head(10).to_string())
    
    print(f"\n合并后的trial信息 (后10行):")
    print(combined_trial_info[['trial_index', 'run', 'original_condition', 'global_trial_index', 'combined_trial_index']].tail(10).to_string())
    
    # 测试分类数据准备
    config = MVPAConfig()
    selected_betas, y_labels, trial_details = prepare_classification_data(
        combined_trial_info, all_beta_images, 'kj', 'yy', config
    )
    
    print(f"\n分类数据准备结果 (前10行):")
    print(trial_details.head(10).to_string())
    
    # 验证映射关系
    print(f"\n验证结果:")
    print(f"- 总trial数量: {len(combined_trial_info)}")
    print(f"- 总beta数量: {len(all_beta_images)}")
    print(f"- 选择的beta数量: {len(selected_betas)}")
    print(f"- 标签数量: {len(y_labels)}")
    print(f"- trial详情数量: {len(trial_details)}")
    
    # 检查combined_trial_index和beta_images的对应关系
    print(f"\n详细映射检查:")
    for i, (_, row) in enumerate(trial_details.iterrows()):
        combined_idx = row['combined_trial_index']
        actual_beta = selected_betas[i]
        expected_beta = all_beta_images[combined_idx]
        
        print(f"  Trial {row['global_trial_index']} (run{row['run']}, {row['condition']}): ")
        print(f"    combined_index={combined_idx}, actual_beta={actual_beta}, expected_beta={expected_beta}")
        
        assert actual_beta == expected_beta, f"Beta映射错误: {actual_beta} != {expected_beta}"
    
    # 检查run分布
    run_counts = trial_details['run'].value_counts().sort_index()
    print(f"\n各run的trial分布:")
    for run, count in run_counts.items():
        print(f"  Run {run}: {count} trials")
    
    # 检查条件分布
    condition_counts = trial_details['condition'].value_counts()
    print(f"\n各条件的trial分布:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} trials")
    
    print("✅ 多run映射测试通过!")
    return combined_trial_info, all_beta_images

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试1: 不同run有不同的trial_index起始值
    print("\n测试1: 不同run的trial_index起始值")
    trial_info_run3 = pd.DataFrame({
        'trial_index': [1, 2, 3, 4],
        'run': [3, 3, 3, 3],
        'original_condition': ['kj', 'yy', 'kj', 'yy'],
        'global_trial_index': ['run3_trial1', 'run3_trial2', 'run3_trial3', 'run3_trial4']
    })
    
    trial_info_run4 = pd.DataFrame({
        'trial_index': [1, 2, 3],  # 注意：这里也是从1开始
        'run': [4, 4, 4],
        'original_condition': ['kj', 'yy', 'kj'],
        'global_trial_index': ['run4_trial1', 'run4_trial2', 'run4_trial3']
    })
    
    # 合并数据
    combined_trial_info = pd.concat([trial_info_run3, trial_info_run4], ignore_index=True)
    combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
    
    print("合并后的数据:")
    print(combined_trial_info.to_string())
    
    # 创建对应的beta图像
    all_beta_images = [f'run3_beta_{i+1}.nii' for i in range(4)] + [f'run4_beta_{i+1}.nii' for i in range(3)]
    
    # 测试分类数据准备
    config = MVPAConfig()
    selected_betas, y_labels, trial_details = prepare_classification_data(
        combined_trial_info, all_beta_images, 'kj', 'yy', config
    )
    
    print("\n分类结果:")
    print(trial_details.to_string())
    
    # 验证映射
    for i, (_, row) in enumerate(trial_details.iterrows()):
        combined_idx = row['combined_trial_index']
        actual_beta = selected_betas[i]
        expected_beta = all_beta_images[combined_idx]
        
        print(f"  {row['global_trial_index']}: combined_idx={combined_idx}, beta={actual_beta}")
        assert actual_beta == expected_beta, f"映射错误: {actual_beta} != {expected_beta}"
    
    print("✅ 边界情况测试通过!")

def main():
    """主测试函数"""
    print("开始测试多run分析时trial和condition的对应关系...")
    
    try:
        # 测试单run映射
        test_single_run_mapping()
        
        # 测试多run映射
        test_multi_run_mapping()
        
        # 测试边界情况
        test_edge_cases()
        
        print("\n🎉 所有测试通过！多run分析时trial和condition的对应关系正确！")
        print("\n总结:")
        print("- 单run分析: 使用 trial_index-1 作为beta图像索引")
        print("- 多run分析: 使用 combined_trial_index 作为beta图像索引")
        print("- 所有trial都保留了原始的run信息和global_trial_index")
        print("- 条件标签和trial的映射关系在合并后保持正确")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()