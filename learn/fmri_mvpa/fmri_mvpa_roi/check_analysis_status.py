#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析状态检查脚本

功能:
    - 检查当前分析进度
    - 验证结果文件完整性
    - 显示已完成和失败的被试
    - 提供继续分析的建议
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def check_analysis_status():
    """检查分析状态"""
    print("=" * 70)
    print("MVPA ROI 分析状态检查")
    print("=" * 70)
    
    # 定义路径
    results_dir = Path("../../../learn_mvpa/metaphor_ROI_MVPA_enhanced")
    backup_dir = results_dir / "backup"
    progress_file = backup_dir / "analysis_progress.json"
    subject_results_file = backup_dir / "subject_results.pkl"
    
    print(f"结果目录: {results_dir}")
    print(f"备份目录: {backup_dir}")
    print()
    
    # 检查目录是否存在
    if not results_dir.exists():
        print("❌ 结果目录不存在，可能还未开始分析")
        return
    
    if not backup_dir.exists():
        print("❌ 备份目录不存在，可能还未开始分析")
        return
    
    # 检查进度文件
    print("1. 分析进度检查")
    print("-" * 30)
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            
            print(f"✓ 进度文件存在")
            print(f"  最后更新时间: {progress.get('last_update', '未知')}")
            print(f"  总被试数: {progress.get('total_subjects', '未知')}")
            print(f"  已完成被试数: {len(progress.get('completed_subjects', []))}")
            print(f"  失败被试数: {len(progress.get('failed_subjects', []))}")
            
            completed = progress.get('completed_subjects', [])
            failed = progress.get('failed_subjects', [])
            
            if completed:
                print(f"  已完成被试: {completed}")
            
            if failed:
                print(f"  失败被试: {failed}")
                
        except Exception as e:
            print(f"❌ 读取进度文件失败: {e}")
    else:
        print("❌ 进度文件不存在")
        completed = []
        failed = []
    
    print()
    
    # 检查被试结果文件
    print("2. 被试结果检查")
    print("-" * 30)
    
    if subject_results_file.exists():
        try:
            with open(subject_results_file, 'rb') as f:
                subject_results = pickle.load(f)
            
            print(f"✓ 被试结果文件存在")
            print(f"  包含被试数: {len(subject_results)}")
            print(f"  被试列表: {list(subject_results.keys())}")
            
            # 检查每个被试的结果完整性
            for subject, results in subject_results.items():
                if isinstance(results, dict):
                    contrast_count = len(results)
                    roi_count = sum(len(contrast_results) for contrast_results in results.values())
                    print(f"    {subject}: {contrast_count}个对比, {roi_count}个ROI结果")
                else:
                    print(f"    {subject}: 结果格式异常")
                    
        except Exception as e:
            print(f"❌ 读取被试结果文件失败: {e}")
            subject_results = {}
    else:
        print("❌ 被试结果文件不存在")
        subject_results = {}
    
    print()
    
    # 检查最终结果文件
    print("3. 最终结果文件检查")
    print("-" * 30)
    
    result_files = [
        ("完整结果", "group_roi_mvpa_results_complete.csv"),
        ("组水平结果", "group_roi_mvpa_results.csv"),
        ("统计结果", "group_statistics_enhanced.csv"),
        ("HTML报告", "analysis_report.html"),
        ("分析总结", "analysis_summary.txt"),
        ("配置文件", "analysis_config.json")
    ]
    
    for desc, filename in result_files:
        filepath = results_dir / filename
        if filepath.exists():
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                    print(f"✓ {desc}: {filename} ({len(df)}行数据)")
                    if 'subject' in df.columns:
                        unique_subjects = df['subject'].nunique()
                        print(f"    包含 {unique_subjects} 个被试的数据")
                except Exception as e:
                    print(f"⚠️  {desc}: {filename} (读取失败: {e})")
            else:
                print(f"✓ {desc}: {filename}")
        else:
            print(f"❌ {desc}: {filename} (不存在)")
    
    print()
    
    # 分析建议
    print("4. 分析建议")
    print("-" * 30)
    
    all_subjects = [f"sub-{i:02d}" for i in range(1, 29)]  # sub-01 到 sub-28
    completed_subjects = list(subject_results.keys())
    missing_subjects = [s for s in all_subjects if s not in completed_subjects]
    
    if len(completed_subjects) == 0:
        print("🔄 建议: 开始新的分析")
        print("   命令: python fmri_mvpa_roi_resume_analysis.py")
    elif len(missing_subjects) == 0:
        print("✅ 所有被试分析完成！")
        if not (results_dir / "group_roi_mvpa_results_complete.csv").exists():
            print("🔄 建议: 重新生成最终报告")
            print("   命令: python fmri_mvpa_roi_resume_analysis.py")
    else:
        print(f"🔄 还有 {len(missing_subjects)} 个被试未完成分析")
        print(f"   缺失被试: {missing_subjects}")
        
        # 找到第一个缺失的被试
        first_missing = missing_subjects[0]
        print(f"🔄 建议: 从 {first_missing} 继续分析")
        print(f"   命令: python fmri_mvpa_roi_resume_analysis.py --resume-from {first_missing}")
        
        # 特殊情况：如果是从sub-17开始
        if first_missing == 'sub-17':
            print("   或者使用快速启动脚本:")
            print("   命令: python resume_from_sub17.py")
    
    # 检查失败的被试
    if 'failed' in locals() and failed:
        print(f"\n⚠️  注意: 有 {len(failed)} 个被试分析失败")
        print(f"   失败被试: {failed}")
        print("   建议检查这些被试的数据完整性")
    
    print()
    
    # 磁盘空间检查
    print("5. 磁盘空间检查")
    print("-" * 30)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(results_dir)
        
        print(f"总空间: {total // (1024**3):.1f} GB")
        print(f"已用空间: {used // (1024**3):.1f} GB")
        print(f"可用空间: {free // (1024**3):.1f} GB")
        
        if free < 5 * 1024**3:  # 小于5GB
            print("⚠️  警告: 可用磁盘空间不足5GB，可能影响分析")
        else:
            print("✓ 磁盘空间充足")
            
    except Exception as e:
        print(f"❌ 无法检查磁盘空间: {e}")
    
    print()
    print("=" * 70)
    print("检查完成")
    print("=" * 70)

def show_detailed_results():
    """显示详细的分析结果"""
    results_dir = Path("../../../learn_mvpa/metaphor_ROI_MVPA_enhanced")
    
    # 检查完整结果文件
    complete_results_file = results_dir / "group_roi_mvpa_results_complete.csv"
    if complete_results_file.exists():
        print("\n" + "=" * 50)
        print("详细分析结果")
        print("=" * 50)
        
        try:
            df = pd.read_csv(complete_results_file)
            
            print(f"总数据行数: {len(df)}")
            print(f"被试数量: {df['subject'].nunique()}")
            print(f"ROI数量: {df['roi'].nunique()}")
            print(f"对比条件数量: {df['contrast'].nunique()}")
            
            print("\n按被试统计:")
            subject_stats = df.groupby('subject').agg({
                'accuracy': ['mean', 'std'],
                'roi': 'count'
            }).round(3)
            print(subject_stats)
            
            print("\n按ROI统计:")
            roi_stats = df.groupby('roi').agg({
                'accuracy': ['mean', 'std'],
                'subject': 'count'
            }).round(3)
            print(roi_stats)
            
        except Exception as e:
            print(f"读取结果文件失败: {e}")
    
    # 检查统计结果文件
    stats_file = results_dir / "group_statistics_enhanced.csv"
    if stats_file.exists():
        print("\n" + "=" * 50)
        print("统计显著性结果")
        print("=" * 50)
        
        try:
            stats_df = pd.read_csv(stats_file)
            
            print(f"总测试数: {len(stats_df)}")
            
            if 'significant_uncorrected' in stats_df.columns:
                uncorrected_sig = stats_df['significant_uncorrected'].sum()
                print(f"显著结果数 (未校正): {uncorrected_sig}")
            
            if 'significant_corrected' in stats_df.columns:
                corrected_sig = stats_df['significant_corrected'].sum()
                print(f"显著结果数 (校正后): {corrected_sig}")
                
                if corrected_sig > 0:
                    print("\n显著结果详情 (校正后):")
                    sig_results = stats_df[stats_df['significant_corrected']]
                    for _, row in sig_results.iterrows():
                        print(f"  {row['roi']}: 准确率={row['mean_accuracy']:.3f}, "
                              f"t={row['t_stat']:.3f}, p={row.get('p_corrected', row.get('p_value', 'N/A')):.6f}")
            
        except Exception as e:
            print(f"读取统计文件失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查MVPA ROI分析状态')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细的分析结果')
    
    args = parser.parse_args()
    
    # 基本状态检查
    check_analysis_status()
    
    # 详细结果显示
    if args.detailed:
        show_detailed_results()

if __name__ == "__main__":
    main()