#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSS数据的Searchlight MVPA分析示例

本脚本展示如何使用更新后的searchlight管道分析LSS（Least Squares Separate）数据。
LSS数据来自fmri_lss_enhance.py的输出。

主要功能:
1. 加载LSS trial-wise beta图像
2. 执行searchlight MVPA分析
3. 支持多mask分析
4. 生成统计结果和可视化

作者: Assistant
日期: 2024
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_searchlight_pipeline_enhanced import (
    SearchlightMVPAConfig,
    run_group_searchlight_analysis,
    log
)

def setup_lss_searchlight_config():
    """
    设置LSS数据的Searchlight MVPA分析配置
    
    返回:
        SearchlightMVPAConfig: 配置好的分析参数
    """
    print("设置LSS Searchlight MVPA分析配置...")
    
    # 创建基础配置
    config = SearchlightMVPAConfig()
    
    # LSS数据特定设置已在配置类中完成:
    # - config.lss_root = Path(r"../../learn_LSS")  # 匹配fmri_lss_enhance.py的OUTPUT_ROOT
    # - config.subjects = [f"sub-{i:02d}" for i in range(1, 29)]  # sub-01到sub-28
    # - config.runs = [3, 4]  # 匹配LSS分析的runs
    
    print(f"✓ LSS数据路径: {config.lss_root}")
    print(f"✓ 被试范围: {config.subjects[0]} 到 {config.subjects[-1]} (总计{len(config.subjects)}个)")
    print(f"✓ 分析runs: {config.runs}")
    
    # 根据LSS数据调整对比条件
    # 注意: 这里的条件名需要匹配LSS数据中的'original_condition'字段
    config.contrasts = [
        ('target_vs_baseline', 'target', 'baseline'),  # 目标条件 vs 基线条件
        # 可以根据实际LSS数据中的条件添加更多对比
    ]
    
    print(f"✓ 对比条件: {config.contrasts}")
    
    # 设置searchlight参数
    config.searchlight_radius = 4.0  # 4mm半径
    config.min_voxels_in_sphere = 10
    
    # 设置SVM参数
    config.svm_param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear'],  # 使用线性核以提高速度
        'classifier__gamma': ['scale']
    }
    
    # 交叉验证设置
    config.cv_folds = 5
    config.cv_random_state = 42
    
    # 置换检验设置
    config.n_permutations = 100  # 可以根据计算资源调整
    config.permutation_random_state = 42
    
    # 并行处理设置
    config.n_jobs = 4  # 根据计算资源调整
    config.use_parallel = True
    
    print(f"✓ Searchlight半径: {config.searchlight_radius}mm")
    print(f"✓ 交叉验证折数: {config.cv_folds}")
    print(f"✓ 置换检验次数: {config.n_permutations}")
    print(f"✓ 并行进程数: {config.n_jobs}")
    
    return config

def run_single_mask_analysis():
    """
    运行单个mask的searchlight分析
    """
    print("\n=== 单Mask Searchlight分析 ===")
    
    config = setup_lss_searchlight_config()
    
    # 设置单个mask
    config.set_mask('gray_matter')  # 使用灰质mask
    print(f"使用mask: {config.mask_selection}")
    
    # 运行分析
    print("\n开始运行单mask searchlight分析...")
    results = run_group_searchlight_analysis(config)
    
    if results is not None:
        group_df, stats_df = results
        print(f"\n✓ 分析完成!")
        print(f"  总测试数: {len(stats_df)}")
        print(f"  显著结果数: {len(stats_df[stats_df['significant_corrected']])}")
        print(f"  结果保存在: {config.results_dir}")
        return results
    else:
        print("✗ 分析失败")
        return None

def run_multi_mask_analysis():
    """
    运行多mask的searchlight分析
    """
    print("\n=== 多Mask Searchlight分析 ===")
    
    config = setup_lss_searchlight_config()
    
    # 设置多个mask进行对比分析
    mask_list = [
        'gray_matter',           # 全脑灰质
        'language_network',      # 语言网络
        'broca',                # 布洛卡区
        'wernicke',             # 韦尼克区
        'angular_gyrus'         # 角回
    ]
    
    config.set_mask_list(mask_list)
    print(f"使用多个masks: {mask_list}")
    
    # 运行分析
    print("\n开始运行多mask searchlight分析...")
    results = run_group_searchlight_analysis(config)
    
    if results is not None:
        group_df, stats_df = results
        print(f"\n✓ 多mask分析完成!")
        print(f"  总测试数: {len(stats_df)}")
        print(f"  显著结果数: {len(stats_df[stats_df['significant_corrected']])}")
        
        # 按mask统计结果
        if 'mask_name' in stats_df.columns:
            mask_stats = stats_df.groupby('mask_name').agg({
                'significant_corrected': 'sum',
                'mean_accuracy': 'mean'
            }).round(3)
            print(f"\n各mask的结果统计:")
            print(mask_stats)
        
        print(f"  详细结果保存在: {config.results_dir}")
        return results
    else:
        print("✗ 多mask分析失败")
        return None

def run_custom_analysis():
    """
    运行自定义配置的分析
    """
    print("\n=== 自定义Searchlight分析 ===")
    
    config = setup_lss_searchlight_config()
    
    # 自定义设置
    config.subjects = config.subjects[:5]  # 只分析前5个被试以加快速度
    config.n_permutations = 50  # 减少置换次数以加快速度
    config.searchlight_radius = 3.0  # 使用较小的半径
    
    # 设置特定的语言相关mask
    config.set_mask('language_network')
    
    print(f"自定义设置:")
    print(f"  被试数量: {len(config.subjects)}")
    print(f"  置换次数: {config.n_permutations}")
    print(f"  Searchlight半径: {config.searchlight_radius}mm")
    print(f"  使用mask: {config.mask_selection}")
    
    # 运行分析
    print("\n开始运行自定义searchlight分析...")
    results = run_group_searchlight_analysis(config)
    
    if results is not None:
        group_df, stats_df = results
        print(f"\n✓ 自定义分析完成!")
        print(f"  分析被试数: {len(group_df['subject'].unique())}")
        print(f"  总测试数: {len(stats_df)}")
        print(f"  显著结果数: {len(stats_df[stats_df['significant_corrected']])}")
        
        if len(stats_df[stats_df['significant_corrected']]) > 0:
            best_result = stats_df[stats_df['significant_corrected']].nlargest(1, 'mean_accuracy')
            print(f"\n最佳结果:")
            print(f"  对比: {best_result['contrast'].iloc[0]}")
            print(f"  平均准确率: {best_result['mean_accuracy'].iloc[0]:.3f}")
            print(f"  p值: {best_result['p_value'].iloc[0]:.6f}")
        
        return results
    else:
        print("✗ 自定义分析失败")
        return None

def demonstrate_lss_integration():
    """
    演示LSS数据集成的完整流程
    """
    print("\n=== LSS数据集成演示 ===")
    
    # 展示LSS数据结构
    config = SearchlightMVPAConfig()
    
    print(f"LSS数据集成说明:")
    print(f"1. LSS数据路径: {config.lss_root}")
    print(f"   - 来源: fmri_lss_enhance.py的OUTPUT_ROOT")
    print(f"   - 结构: {{subject}}/run-{{run}}_LSS/")
    
    print(f"\n2. LSS输出文件:")
    print(f"   - beta_trial_XXX.nii.gz: 每个trial的beta图像")
    print(f"   - trial_info.csv: trial元数据信息")
    print(f"   - mask.nii.gz: 脑掩模文件")
    
    print(f"\n3. 数据加载适配:")
    print(f"   - 使用'original_condition'字段进行条件筛选")
    print(f"   - 使用'beta_file_path'字段获取beta图像路径")
    print(f"   - 支持多run数据合并")
    
    print(f"\n4. 分析流程:")
    print(f"   - 加载LSS trial数据")
    print(f"   - 准备分类数据")
    print(f"   - 执行searchlight MVPA")
    print(f"   - 生成统计结果和可视化")
    
    print(f"\n5. 支持的分析类型:")
    print(f"   - 单mask分析")
    print(f"   - 多mask对比分析")
    print(f"   - 自定义配置分析")

def main():
    """
    主函数 - 演示LSS数据的searchlight分析
    """
    print("LSS数据的Searchlight MVPA分析示例")
    print("=" * 60)
    
    # 演示LSS集成
    demonstrate_lss_integration()
    
    print("\n选择要运行的分析类型:")
    print("1. 单Mask分析 (推荐用于快速测试)")
    print("2. 多Mask分析 (全面对比不同脑区)")
    print("3. 自定义分析 (快速演示版本)")
    print("4. 仅显示配置信息")
    
    try:
        choice = input("\n请输入选择 (1-4, 默认4): ").strip() or "4"
        
        if choice == "1":
            run_single_mask_analysis()
        elif choice == "2":
            run_multi_mask_analysis()
        elif choice == "3":
            run_custom_analysis()
        elif choice == "4":
            config = setup_lss_searchlight_config()
            print(f"\n配置信息已显示。要运行实际分析，请选择选项1-3。")
        else:
            print(f"无效选择: {choice}")
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("LSS Searchlight分析示例完成")

if __name__ == "__main__":
    main()