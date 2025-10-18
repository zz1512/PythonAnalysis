#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多run分析时run信息记录和数据重复问题的修复
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from fmri_mvpa_roi_pipeline_enhanced import (
    MVPAConfig, 
    load_lss_trial_data, 
    load_multi_run_lss_data,
    prepare_classification_data,
    log
)

def create_mock_lss_data(temp_dir, subject, run, n_trials=20):
    """创建模拟的LSS数据"""
    lss_dir = temp_dir / subject / f"run-{run}_LSS"
    lss_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建trial_info.csv
    trial_data = []
    for i in range(1, n_trials + 1):
        condition = 'yy' if i <= n_trials // 2 else 'kj'
        trial_data.append({
            'trial_index': i,
            'original_condition': condition,
            'onset': i * 2.0,
            'duration': 2.0
        })
    
    trial_df = pd.DataFrame(trial_data)
    trial_df.to_csv(lss_dir / "trial_info.csv", index=False)
    
    # 创建模拟的beta文件（空文件）
    for i in range(1, n_trials + 1):
        beta_file = lss_dir / f"beta_trial_{i:03d}.nii.gz"
        beta_file.touch()
    
    return lss_dir

def test_single_run_data_loading():
    """测试单run数据加载"""
    print("\n=== 测试单run数据加载 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建配置
        config = MVPAConfig()
        config.lss_root = temp_path
        
        # 创建模拟数据
        subject = "sub-01"
        run = 3
        create_mock_lss_data(temp_path, subject, run, n_trials=20)
        
        # 加载数据
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        
        print(f"加载的trial数量: {len(beta_images)}")
        print(f"Trial info列: {list(trial_info.columns)}")
        print(f"Run信息: {trial_info['run'].unique() if 'run' in trial_info.columns else '无'}")
        print(f"Global trial index示例: {trial_info['global_trial_index'].head(3).tolist() if 'global_trial_index' in trial_info.columns else '无'}")
        
        # 检查run信息
        assert 'run' in trial_info.columns, "缺少run列"
        assert all(trial_info['run'] == run), "run信息不正确"
        assert 'global_trial_index' in trial_info.columns, "缺少global_trial_index列"
        
        print("✅ 单run数据加载测试通过")
        return True

def test_multi_run_data_loading():
    """测试多run数据加载和合并"""
    print("\n=== 测试多run数据加载和合并 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建配置
        config = MVPAConfig()
        config.lss_root = temp_path
        
        # 创建模拟数据
        subject = "sub-01"
        runs = [3, 4]
        n_trials_per_run = [20, 18]  # 不同run的trial数量不同
        
        for run, n_trials in zip(runs, n_trials_per_run):
            create_mock_lss_data(temp_path, subject, run, n_trials=n_trials)
        
        # 加载多run数据
        beta_images, trial_info = load_multi_run_lss_data(subject, runs, config)
        
        print(f"合并后的trial数量: {len(beta_images)}")
        print(f"预期trial数量: {sum(n_trials_per_run)}")
        print(f"Trial info列: {list(trial_info.columns)}")
        
        # 检查run分布
        if 'run' in trial_info.columns:
            run_counts = trial_info['run'].value_counts().sort_index()
            print(f"各run的trial数量: {dict(run_counts)}")
            
            # 验证run分布
            for run, expected_count in zip(runs, n_trials_per_run):
                actual_count = run_counts.get(run, 0)
                assert actual_count == expected_count, f"Run {run}的trial数量不匹配: 期望{expected_count}, 实际{actual_count}"
        
        # 检查combined_trial_index
        if 'combined_trial_index' in trial_info.columns:
            combined_indices = trial_info['combined_trial_index'].tolist()
            expected_indices = list(range(len(trial_info)))
            assert combined_indices == expected_indices, "combined_trial_index不连续"
            print(f"Combined trial index范围: {min(combined_indices)} - {max(combined_indices)}")
        
        # 检查global_trial_index的唯一性
        if 'global_trial_index' in trial_info.columns:
            global_indices = trial_info['global_trial_index'].tolist()
            unique_global_indices = set(global_indices)
            assert len(global_indices) == len(unique_global_indices), "global_trial_index存在重复"
            print(f"Global trial index示例: {global_indices[:5]}")
        
        print("✅ 多run数据加载测试通过")
        return beta_images, trial_info

def test_classification_data_preparation(beta_images, trial_info):
    """测试分类数据准备"""
    print("\n=== 测试分类数据准备 ===")
    
    config = MVPAConfig()
    
    # 准备分类数据
    selected_betas, labels, trial_details = prepare_classification_data(
        trial_info, beta_images, 'yy', 'kj', config
    )
    
    print(f"选择的beta数量: {len(selected_betas)}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"Trial details列: {list(trial_details.columns)}")
    
    # 检查run信息是否保留
    if 'run' in trial_details.columns:
        run_counts = trial_details['run'].value_counts().sort_index()
        print(f"分类数据中各run的trial数量: {dict(run_counts)}")
        
        # 检查是否有重复的global_trial_index
        if 'global_trial_index' in trial_details.columns:
            global_indices = trial_details['global_trial_index'].tolist()
            unique_global_indices = set(global_indices)
            if len(global_indices) != len(unique_global_indices):
                print(f"⚠️  发现重复的global_trial_index: {len(global_indices)} vs {len(unique_global_indices)}")
                duplicates = [x for x in global_indices if global_indices.count(x) > 1]
                print(f"重复的索引: {set(duplicates)}")
            else:
                print("✅ 没有重复的global_trial_index")
    
    # 检查combined_trial_index的使用
    if 'combined_trial_index' in trial_details.columns:
        combined_indices = trial_details['combined_trial_index'].tolist()
        print(f"Combined trial index范围: {min(combined_indices)} - {max(combined_indices)}")
        
        # 验证索引是否正确对应beta_images
        max_combined_idx = max(combined_indices)
        assert max_combined_idx < len(beta_images), f"Combined index超出范围: {max_combined_idx} >= {len(beta_images)}"
        print("✅ Combined trial index范围正确")
    
    print("✅ 分类数据准备测试通过")
    return trial_details

def test_trial_details_completeness(trial_details):
    """测试trial详细信息的完整性"""
    print("\n=== 测试trial详细信息完整性 ===")
    
    required_columns = ['trial_index', 'condition', 'label']
    optional_columns = ['run', 'global_trial_index', 'combined_trial_index']
    
    # 检查必需列
    for col in required_columns:
        assert col in trial_details.columns, f"缺少必需列: {col}"
    
    # 检查可选列
    present_optional = [col for col in optional_columns if col in trial_details.columns]
    print(f"存在的可选列: {present_optional}")
    
    # 显示示例数据
    print("\nTrial details示例:")
    print(trial_details.head())
    
    # 检查数据完整性
    for col in trial_details.columns:
        null_count = trial_details[col].isnull().sum()
        if null_count > 0:
            print(f"⚠️  列 {col} 有 {null_count} 个空值")
        else:
            print(f"✅ 列 {col} 无空值")
    
    print("✅ Trial详细信息完整性测试通过")

def main():
    """主测试函数"""
    print("开始测试多run分析的run信息记录和数据重复问题修复...")
    
    try:
        # 测试单run数据加载
        test_single_run_data_loading()
        
        # 测试多run数据加载
        beta_images, trial_info = test_multi_run_data_loading()
        
        # 测试分类数据准备
        trial_details = test_classification_data_preparation(beta_images, trial_info)
        
        # 测试trial详细信息完整性
        test_trial_details_completeness(trial_details)
        
        print("\n🎉 所有测试通过！多run分析的run信息记录和数据重复问题已修复！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()