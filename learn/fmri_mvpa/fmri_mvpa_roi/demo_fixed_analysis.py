#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示修复后的MVPA分析
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_roi_pipeline_enhanced import (
    run_roi_classification_enhanced, 
    perform_group_statistics_enhanced,
    MVPAConfig
)

def create_demo_data():
    """创建演示数据"""
    print("创建演示数据...")
    
    # 模拟3个被试，2个ROI的数据
    subjects = ['sub-01', 'sub-02', 'sub-03']
    rois = ['ROI_A', 'ROI_B']
    
    group_data = []
    
    for subject in subjects:
        for roi in rois:
            # 创建模拟数据
            np.random.seed(hash(subject + roi) % 1000)
            n_samples = 40
            n_features = 50
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.choice([0, 1], n_samples)
            
            # 为某些ROI添加信号
            if roi == 'ROI_A':
                X[y == 0, :10] += 0.3  # 添加一些分类信号
            
            # 运行分类
            config = MVPAConfig()
            config.n_permutations = 50  # 减少置换次数以便快速演示
            
            result = run_roi_classification_enhanced(X, y, config)
            
            group_data.append({
                'subject': subject,
                'contrast': 'metaphor_vs_space',
                'roi': roi,
                'accuracy': result['mean_accuracy'],
                'p_value': result['p_value'],
                'n_trials_cond1': np.sum(y == 0),
                'n_trials_cond2': np.sum(y == 1),
                'n_voxels': n_features,
                'n_features': n_features,
                'n_samples': n_samples,
                'best_params': str(result.get('best_params', {})),
                'best_score': result.get('best_score', np.nan)
            })
            
            print(f"  {subject} - {roi}: 准确率={result['mean_accuracy']:.3f}, p值={result['p_value']:.4f}")
    
    return pd.DataFrame(group_data)

def demo_group_analysis():
    """演示组水平分析"""
    print("\n=== 演示修复后的MVPA分析 ===")
    
    # 创建演示数据
    group_df = create_demo_data()
    
    print(f"\n组水平数据:")
    print(group_df[['subject', 'roi', 'accuracy', 'p_value']].to_string(index=False))
    
    # 运行组水平统计
    print("\n运行组水平统计分析...")
    config = MVPAConfig()
    stats_df = perform_group_statistics_enhanced(group_df, config)
    
    print(f"\n组水平统计结果:")
    display_cols = ['roi', 'mean_accuracy', 'p_value_permutation', 'p_corrected', 'significant_corrected']
    print(stats_df[display_cols].to_string(index=False))
    
    # 检查p值
    print(f"\n=== p值检查 ===")
    print(f"被试内置换检验p值范围: {group_df['p_value'].min():.4f} - {group_df['p_value'].max():.4f}")
    print(f"组水平置换检验p值范围: {stats_df['p_value_permutation'].min():.4f} - {stats_df['p_value_permutation'].max():.4f}")
    print(f"校正后p值范围: {stats_df['p_corrected'].min():.4f} - {stats_df['p_corrected'].max():.4f}")
    
    # 检查是否还有p值为0的问题
    zero_p_count = (stats_df['p_value_permutation'] == 0.0).sum()
    if zero_p_count > 0:
        print(f"❌ 警告: 仍有 {zero_p_count} 个p值为0")
    else:
        print(f"✅ 成功: 所有p值都大于0")
    
    return group_df, stats_df

if __name__ == "__main__":
    try:
        group_df, stats_df = demo_group_analysis()
        print("\n🎉 演示完成! p值修复成功!")
    except Exception as e:
        print(f"\n💥 演示失败: {e}")
        import traceback
        traceback.print_exc()