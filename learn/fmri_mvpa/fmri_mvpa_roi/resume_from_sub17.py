#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本：从第17个被试继续MVPA ROI分析

使用方法:
    python resume_from_sub17.py

功能:
    - 自动从sub-17开始继续分析
    - 合并之前1-16被试的结果
    - 生成1-28被试的完整报告
"""

import sys
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_roi_resume_analysis import ResumeConfig, run_resume_roi_analysis

def main():
    """主函数：从第17个被试开始继续分析"""
    print("=" * 60)
    print("从第17个被试继续MVPA ROI分析")
    print("=" * 60)
    
    # 创建配置，从sub-17开始分析
    config = ResumeConfig(
        start_subject=1,      # 总体被试范围1-28
        end_subject=28,       # 总体被试范围1-28
        resume_from='sub-28'  # 从sub-17开始继续
    )
    
    print(f"配置信息:")
    print(f"  - 总被试范围: sub-01 到 sub-28")
    print(f"  - 继续分析起点: sub-17")
    print(f"  - 将分析被试: {config.subjects[:5]}...{config.subjects[-2:]} (共{len(config.subjects)}个)")
    print(f"  - 结果保存目录: {config.results_dir}")
    print(f"  - 备份目录: {config.backup_dir}")
    print()
    
    # 检查是否有之前的结果
    if config.subject_results_file.exists():
        print("✓ 发现之前的分析结果，将自动合并")
    else:
        print("! 未发现之前的分析结果，将从头开始分析")
    
    if config.progress_file.exists():
        print("✓ 发现分析进度记录")
    
    print("\n开始分析...")
    print("-" * 60)
    
    try:
        # 运行分析
        results = run_resume_roi_analysis(config)
        
        if results is not None:
            group_df, stats_df, all_subject_results = results
            
            print("\n" + "=" * 60)
            print("分析完成！")
            print("=" * 60)
            print(f"✓ 总被试数: {len(all_subject_results)}")
            print(f"✓ 总测试数: {len(stats_df)}")
            # 兼容并补齐显著性列
            if 'significant_corrected' not in stats_df.columns:
                alpha = getattr(config, 'alpha_level', 0.05)
                if 'p_corrected' in stats_df.columns:
                    stats_df['significant_corrected'] = stats_df['p_corrected'] < alpha
                elif 'p_value' in stats_df.columns:
                    stats_df['significant_corrected'] = stats_df['p_value'] < alpha
                else:
                    stats_df['significant_corrected'] = False
            print(f"✓ 显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}")
            
            # 显示主要输出文件
            print("\n主要输出文件:")
            output_files = [
                ("完整结果", "group_roi_mvpa_results_complete.csv"),
                ("统计结果", "group_statistics_enhanced.csv"),
                ("HTML报告", "analysis_report.html"),
                ("分析总结", "analysis_summary.txt")
            ]
            
            for desc, filename in output_files:
                filepath = config.results_dir / filename
                if filepath.exists():
                    print(f"  ✓ {desc}: {filepath}")
                else:
                    print(f"  ✗ {desc}: {filepath} (未找到)")
            
            # 显著结果打印兼容t列名
            if len(stats_df[stats_df['significant_corrected']]) > 0:
                print("\n显著结果 (校正后):")
                sig_results = stats_df[stats_df['significant_corrected']]
                # 兼容t统计量列名：优先t_stat，其次t_statistic，若都缺失则用NaN
                t_col = 't_stat' if 't_stat' in stats_df.columns else (
                    't_statistic' if 't_statistic' in stats_df.columns else None
                )
                for _, row in sig_results.iterrows():
                    t_val = row[t_col] if t_col else float('nan')
                    print(f"  • {row['roi']}: 准确率={row['mean_accuracy']:.3f}, "
                          f"t={t_val:.3f}, p={row['p_corrected']:.6f}")
            else:
                print("\n未发现显著结果 (校正后)")
            
            print(f"\n详细报告请查看: {config.results_dir / 'analysis_report.html'}")
            
        else:
            print("\n分析失败！请检查日志文件获取详细错误信息。")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n分析被用户中断")
        print("进度已保存，可以稍后继续分析")
        return 1
    except Exception as e:
        print(f"\n\n分析过程中发生错误: {e}")
        logging.exception("详细错误信息:")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)