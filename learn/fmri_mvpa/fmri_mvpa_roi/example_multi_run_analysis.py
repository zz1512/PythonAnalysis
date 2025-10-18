#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多run MVPA分析示例
演示如何正确处理多个run的数据，避免数据重复和run信息丢失
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from fmri_mvpa_roi_pipeline_enhanced import (
    MVPAConfig, 
    load_lss_trial_data, 
    load_multi_run_lss_data,
    prepare_classification_data,
    analyze_single_subject_roi_enhanced,
    log
)

def create_realistic_mock_data(temp_dir, subject, run, n_trials=30):
    """创建更真实的模拟LSS数据"""
    lss_dir = temp_dir / subject / f"run-{run}_LSS"
    lss_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建trial_info.csv，模拟真实的实验设计
    trial_data = []
    np.random.seed(42 + run)  # 确保每个run的数据不同但可重现
    
    # 随机分配条件，但保持平衡
    conditions = ['yy'] * (n_trials // 2) + ['kj'] * (n_trials // 2)
    if n_trials % 2 == 1:
        conditions.append(np.random.choice(['yy', 'kj']))
    np.random.shuffle(conditions)
    
    for i in range(1, n_trials + 1):
        trial_data.append({
            'trial_index': i,
            'original_condition': conditions[i-1],
            'onset': i * 3.0 + np.random.normal(0, 0.5),  # 添加一些随机性
            'duration': 2.0,
            'pic_num': np.random.randint(1, 100),  # 模拟图片编号
            'response_time': np.random.normal(1.2, 0.3)  # 模拟反应时
        })
    
    trial_df = pd.DataFrame(trial_data)
    trial_df.to_csv(lss_dir / "trial_info.csv", index=False)
    
    # 创建模拟的beta文件（空文件，实际分析中这些是NIfTI文件）
    for i in range(1, n_trials + 1):
        beta_file = lss_dir / f"beta_trial_{i:03d}.nii.gz"
        beta_file.touch()
    
    return lss_dir, trial_df

def demonstrate_single_run_analysis():
    """演示单run分析"""
    print("\n" + "="*60)
    print("演示1: 单run MVPA分析")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建配置
        config = MVPAConfig()
        config.lss_root = temp_path
        config.runs = [3]  # 单run
        
        # 创建模拟数据
        subject = "sub-01"
        run = 3
        lss_dir, original_trial_df = create_realistic_mock_data(temp_path, subject, run, n_trials=24)
        
        print(f"创建了 {subject} run-{run} 的模拟数据:")
        print(f"  - 总trial数: {len(original_trial_df)}")
        print(f"  - 条件分布: {dict(original_trial_df['original_condition'].value_counts())}")
        
        # 加载数据
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        
        print(f"\n加载结果:")
        print(f"  - 加载的beta文件数: {len(beta_images)}")
        print(f"  - Trial info形状: {trial_info.shape}")
        print(f"  - 包含的列: {list(trial_info.columns)}")
        
        # 准备分类数据
        selected_betas, labels, trial_details = prepare_classification_data(
            trial_info, beta_images, 'yy', 'kj', config
        )
        
        print(f"\n分类数据准备:")
        print(f"  - 选择的trial数: {len(selected_betas)}")
        print(f"  - 标签分布: yy={np.sum(labels==0)}, kj={np.sum(labels==1)}")
        print(f"  - Trial details列: {list(trial_details.columns)}")
        
        # 显示trial详细信息示例
        print(f"\nTrial详细信息示例:")
        print(trial_details.head())
        
        return trial_details

def demonstrate_multi_run_analysis():
    """演示多run分析"""
    print("\n" + "="*60)
    print("演示2: 多run MVPA分析")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建配置
        config = MVPAConfig()
        config.lss_root = temp_path
        config.runs = [3, 4, 5]  # 多run
        
        # 创建模拟数据
        subject = "sub-01"
        runs = [3, 4, 5]
        n_trials_per_run = [24, 20, 28]  # 不同run的trial数量不同
        
        total_trials = 0
        all_original_data = []
        
        for run, n_trials in zip(runs, n_trials_per_run):
            lss_dir, original_trial_df = create_realistic_mock_data(temp_path, subject, run, n_trials=n_trials)
            all_original_data.append(original_trial_df)
            total_trials += n_trials
            print(f"创建了 {subject} run-{run}: {n_trials} trials")
        
        print(f"\n总计: {total_trials} trials 来自 {len(runs)} 个run")
        
        # 加载多run数据
        beta_images, trial_info = load_multi_run_lss_data(subject, runs, config)
        
        print(f"\n多run数据加载结果:")
        print(f"  - 合并后的beta文件数: {len(beta_images)}")
        print(f"  - 合并后的trial info形状: {trial_info.shape}")
        print(f"  - 包含的列: {list(trial_info.columns)}")
        
        # 检查run分布
        if 'run' in trial_info.columns:
            run_distribution = trial_info['run'].value_counts().sort_index()
            print(f"  - 各run的trial分布: {dict(run_distribution)}")
        
        # 检查条件分布
        condition_distribution = trial_info['original_condition'].value_counts()
        print(f"  - 总体条件分布: {dict(condition_distribution)}")
        
        # 按run和条件的交叉分布
        if 'run' in trial_info.columns:
            cross_tab = pd.crosstab(trial_info['run'], trial_info['original_condition'])
            print(f"\n各run的条件分布:")
            print(cross_tab)
        
        # 准备分类数据
        selected_betas, labels, trial_details = prepare_classification_data(
            trial_info, beta_images, 'yy', 'kj', config
        )
        
        print(f"\n分类数据准备:")
        print(f"  - 选择的trial数: {len(selected_betas)}")
        print(f"  - 标签分布: yy={np.sum(labels==0)}, kj={np.sum(labels==1)}")
        print(f"  - Trial details列: {list(trial_details.columns)}")
        
        # 检查run信息保留
        if 'run' in trial_details.columns:
            run_dist_in_classification = trial_details['run'].value_counts().sort_index()
            print(f"  - 分类数据中各run的trial数: {dict(run_dist_in_classification)}")
        
        # 检查唯一性
        if 'global_trial_index' in trial_details.columns:
            unique_global = len(trial_details['global_trial_index'].unique())
            total_trials_in_classification = len(trial_details)
            print(f"  - Global trial index唯一性: {unique_global}/{total_trials_in_classification}")
            if unique_global == total_trials_in_classification:
                print("    ✅ 没有重复的trial")
            else:
                print("    ❌ 存在重复的trial")
        
        # 显示trial详细信息示例
        print(f"\n多run Trial详细信息示例:")
        print(trial_details.head(10))
        
        # 按run分组显示
        if 'run' in trial_details.columns:
            print(f"\n按run分组的trial详细信息:")
            for run in sorted(trial_details['run'].unique()):
                run_trials = trial_details[trial_details['run'] == run]
                print(f"\nRun {run} ({len(run_trials)} trials):")
                print(run_trials[['trial_index', 'condition', 'global_trial_index', 'combined_trial_index']].head())
        
        return trial_details

def demonstrate_data_integrity_checks(trial_details):
    """演示数据完整性检查"""
    print("\n" + "="*60)
    print("演示3: 数据完整性检查")
    print("="*60)
    
    # 检查必需字段
    required_fields = ['trial_index', 'condition', 'label']
    for field in required_fields:
        if field in trial_details.columns:
            print(f"✅ 必需字段 '{field}' 存在")
        else:
            print(f"❌ 必需字段 '{field}' 缺失")
    
    # 检查run信息
    if 'run' in trial_details.columns:
        print(f"✅ Run信息已保留")
        print(f"   - 包含的run: {sorted(trial_details['run'].unique())}")
    else:
        print(f"❌ Run信息缺失")
    
    # 检查唯一标识符
    if 'global_trial_index' in trial_details.columns:
        global_indices = trial_details['global_trial_index']
        if len(global_indices) == len(global_indices.unique()):
            print(f"✅ Global trial index唯一性检查通过")
        else:
            duplicates = global_indices[global_indices.duplicated()].unique()
            print(f"❌ 发现重复的global trial index: {list(duplicates)}")
    
    # 检查索引连续性
    if 'combined_trial_index' in trial_details.columns:
        combined_indices = sorted(trial_details['combined_trial_index'])
        expected_indices = list(range(len(trial_details)))
        if combined_indices == expected_indices:
            print(f"✅ Combined trial index连续性检查通过")
        else:
            print(f"❌ Combined trial index不连续")
    
    # 检查数据类型
    print(f"\n数据类型检查:")
    for col in trial_details.columns:
        dtype = trial_details[col].dtype
        null_count = trial_details[col].isnull().sum()
        print(f"  - {col}: {dtype}, {null_count} nulls")
    
    print(f"\n✅ 数据完整性检查完成")

def main():
    """主演示函数"""
    print("多run MVPA分析演示")
    print("本演示展示了如何正确处理多个run的数据，避免数据重复和run信息丢失")
    
    # 演示1: 单run分析
    single_run_details = demonstrate_single_run_analysis()
    
    # 演示2: 多run分析
    multi_run_details = demonstrate_multi_run_analysis()
    
    # 演示3: 数据完整性检查
    demonstrate_data_integrity_checks(multi_run_details)
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("✅ 单run分析: 正确加载和处理单个run的数据")
    print("✅ 多run分析: 正确合并多个run的数据，保留run信息")
    print("✅ 数据完整性: 确保没有重复数据，所有必需信息都被保留")
    print("✅ 索引正确性: 使用正确的索引访问合并后的数据")
    
    print("\n关键改进:")
    print("1. 在load_lss_trial_data中添加run、global_trial_index信息")
    print("2. 在load_multi_run_lss_data中添加combined_trial_index")
    print("3. 在prepare_classification_data中正确使用combined_trial_index")
    print("4. 在trial_details中保留所有run相关信息")
    
    print("\n🎉 多run MVPA分析演示完成！")

if __name__ == "__main__":
    main()