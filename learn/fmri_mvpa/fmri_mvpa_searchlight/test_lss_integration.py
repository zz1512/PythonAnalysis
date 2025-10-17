#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LSS数据集成到Searchlight MVPA管道

本脚本测试更新后的searchlight管道是否能正确加载和处理LSS分析的输出数据。

作者: Assistant
日期: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_searchlight_pipeline_enhanced import (
    SearchlightMVPAConfig,
    load_lss_trial_data,
    load_multi_run_lss_data,
    prepare_classification_data,
    log
)

def test_config_setup():
    """测试配置设置是否正确匹配LSS输出"""
    print("\n=== 测试1: 配置设置 ===")
    
    config = SearchlightMVPAConfig()
    
    print(f"LSS根目录: {config.lss_root}")
    print(f"被试范围: {config.subjects[:3]}...{config.subjects[-3:]} (总计{len(config.subjects)}个)")
    print(f"Run编号: {config.runs}")
    print(f"对比条件: {config.contrasts}")
    
    # 检查LSS根目录是否存在
    if config.lss_root.exists():
        print(f"✓ LSS根目录存在: {config.lss_root}")
        
        # 列出可用的被试目录
        subject_dirs = [d for d in config.lss_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        print(f"✓ 发现{len(subject_dirs)}个被试目录")
        if subject_dirs:
            print(f"  示例: {subject_dirs[:3]}")
    else:
        print(f"✗ LSS根目录不存在: {config.lss_root}")
    
    return config

def test_single_subject_data_loading(config):
    """测试单个被试数据加载"""
    print("\n=== 测试2: 单被试数据加载 ===")
    
    # 选择第一个被试进行测试
    test_subject = config.subjects[0]
    test_run = config.runs[0]
    
    print(f"测试被试: {test_subject}, Run: {test_run}")
    
    # 测试单个run数据加载
    beta_images, trial_info = load_lss_trial_data(test_subject, test_run, config)
    
    if beta_images is not None and trial_info is not None:
        print(f"✓ 成功加载数据")
        print(f"  Beta图像数量: {len(beta_images)}")
        print(f"  Trial信息条数: {len(trial_info)}")
        print(f"  可用条件: {trial_info['original_condition'].unique()}")
        print(f"  Trial索引范围: {trial_info['trial_index'].min()}-{trial_info['trial_index'].max()}")
        
        # 显示前几条trial信息
        print("\n前3条trial信息:")
        print(trial_info[['trial_index', 'original_condition', 'onset', 'duration']].head(3))
        
        return beta_images, trial_info
    else:
        print(f"✗ 数据加载失败")
        return None, None

def test_multi_run_data_loading(config):
    """测试多run数据加载"""
    print("\n=== 测试3: 多Run数据加载 ===")
    
    test_subject = config.subjects[0]
    
    print(f"测试被试: {test_subject}, Runs: {config.runs}")
    
    # 测试多run数据加载
    all_beta_images, combined_trial_info = load_multi_run_lss_data(test_subject, config.runs, config)
    
    if all_beta_images is not None and combined_trial_info is not None:
        print(f"✓ 成功加载多run数据")
        print(f"  总Beta图像数量: {len(all_beta_images)}")
        print(f"  总Trial信息条数: {len(combined_trial_info)}")
        print(f"  可用条件: {combined_trial_info['original_condition'].unique()}")
        
        # 按run统计
        run_stats = combined_trial_info.groupby('run').size()
        print(f"  各run的trial数量: {dict(run_stats)}")
        
        return all_beta_images, combined_trial_info
    else:
        print(f"✗ 多run数据加载失败")
        return None, None

def test_classification_data_preparation(beta_images, trial_info, config):
    """测试分类数据准备"""
    print("\n=== 测试4: 分类数据准备 ===")
    
    # 获取可用条件
    available_conditions = trial_info['original_condition'].unique()
    print(f"可用条件: {available_conditions}")
    
    if len(available_conditions) >= 2:
        # 选择前两个条件进行测试
        cond1, cond2 = available_conditions[:2]
        print(f"测试对比: {cond1} vs {cond2}")
        
        # 准备分类数据
        selected_betas, labels, selected_trials = prepare_classification_data(
            trial_info, beta_images, cond1, cond2, config
        )
        
        if selected_betas is not None:
            print(f"✓ 成功准备分类数据")
            print(f"  选中的beta图像数量: {len(selected_betas)}")
            print(f"  标签数组形状: {labels.shape}")
            print(f"  标签分布: {np.bincount(labels)}")
            print(f"  条件1 ({cond1}): {np.sum(labels == 0)}个trial")
            print(f"  条件2 ({cond2}): {np.sum(labels == 1)}个trial")
            
            # 验证beta文件路径
            valid_paths = [Path(p).exists() for p in selected_betas[:5]]  # 检查前5个
            print(f"  前5个beta文件存在性: {valid_paths}")
            
            return selected_betas, labels, selected_trials
        else:
            print(f"✗ 分类数据准备失败")
    else:
        print(f"✗ 可用条件不足 (需要至少2个条件)")
    
    return None, None, None

def test_lss_directory_structure(config):
    """测试LSS目录结构"""
    print("\n=== 测试5: LSS目录结构检查 ===")
    
    if not config.lss_root.exists():
        print(f"✗ LSS根目录不存在: {config.lss_root}")
        return
    
    print(f"LSS根目录: {config.lss_root}")
    
    # 检查被试目录
    subject_dirs = [d for d in config.lss_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    print(f"发现{len(subject_dirs)}个被试目录")
    
    if subject_dirs:
        # 检查第一个被试的结构
        test_subject_dir = subject_dirs[0]
        print(f"\n检查被试目录: {test_subject_dir.name}")
        
        run_dirs = [d for d in test_subject_dir.iterdir() if d.is_dir() and 'run-' in d.name]
        print(f"  发现{len(run_dirs)}个run目录: {[d.name for d in run_dirs]}")
        
        if run_dirs:
            # 检查第一个run的内容
            test_run_dir = run_dirs[0]
            print(f"\n检查run目录: {test_run_dir.name}")
            
            # 检查关键文件
            trial_info_file = test_run_dir / "trial_info.csv"
            mask_file = test_run_dir / "mask.nii.gz"
            
            print(f"  trial_info.csv存在: {trial_info_file.exists()}")
            print(f"  mask.nii.gz存在: {mask_file.exists()}")
            
            # 检查beta文件
            beta_files = list(test_run_dir.glob("beta_trial_*.nii.gz"))
            print(f"  beta文件数量: {len(beta_files)}")
            if beta_files:
                print(f"  beta文件示例: {beta_files[:3]}")
            
            # 如果trial_info.csv存在，读取并显示信息
            if trial_info_file.exists():
                try:
                    trial_df = pd.read_csv(trial_info_file)
                    print(f"\n  trial_info.csv内容:")
                    print(f"    记录数: {len(trial_df)}")
                    print(f"    列名: {list(trial_df.columns)}")
                    if 'original_condition' in trial_df.columns:
                        print(f"    条件: {trial_df['original_condition'].unique()}")
                        print(f"    条件分布: {dict(trial_df['original_condition'].value_counts())}")
                except Exception as e:
                    print(f"    读取trial_info.csv失败: {e}")

def main():
    """主测试函数"""
    print("开始测试LSS数据集成到Searchlight MVPA管道")
    print("=" * 60)
    
    try:
        # 测试1: 配置设置
        config = test_config_setup()
        
        # 测试5: LSS目录结构检查
        test_lss_directory_structure(config)
        
        # 测试2: 单被试数据加载
        beta_images, trial_info = test_single_subject_data_loading(config)
        
        if beta_images is not None and trial_info is not None:
            # 测试3: 多run数据加载
            all_beta_images, combined_trial_info = test_multi_run_data_loading(config)
            
            if all_beta_images is not None and combined_trial_info is not None:
                # 测试4: 分类数据准备
                test_classification_data_preparation(all_beta_images, combined_trial_info, config)
        
        print("\n" + "=" * 60)
        print("LSS数据集成测试完成")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()