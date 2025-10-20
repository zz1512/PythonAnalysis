# main_analysis.py
# 主分析模块

import pandas as pd
import numpy as np
from pathlib import Path
from config import MVPAConfig
from utils import log, memory_cleanup
from data_loader import load_roi_masks, load_multi_run_lss_data
from classification import run_roi_classification_enhanced
from statistics import perform_group_statistics_corrected
from visualization import create_enhanced_visualizations, create_individual_roi_plots
from report_generator import generate_html_report

def run_group_roi_analysis_corrected(config):
    """运行组水平ROI分析（修正版本）"""
    log("开始组水平ROI分析", config)
    
    # 保存配置
    config.save_config()
    
    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    if not roi_masks:
        raise ValueError("未找到有效的ROI mask文件")
    
    log(f"成功加载 {len(roi_masks)} 个ROI masks", config)
    
    # 初始化结果存储
    all_results = []
    
    # 对每个被试进行分析
    for subject_id in config.subject_ids:
        log(f"\n处理被试: {subject_id}", config)
        
        try:
            # 加载该被试的LSS数据
            lss_data = load_multi_run_lss_data(config, subject_id)
            if lss_data is None:
                log(f"被试 {subject_id} 的LSS数据加载失败，跳过", config)
                continue
            
            # 对每个ROI进行分类分析
            for roi_name, roi_mask in roi_masks.items():
                log(f"  分析ROI: {roi_name}", config)
                
                try:
                    # 运行分类分析
                    result = run_roi_classification_enhanced(
                        lss_data, roi_mask, roi_name, subject_id, config
                    )
                    
                    if result is not None:
                        all_results.append(result)
                        log(f"    {roi_name}: 准确率 = {result['mean_accuracy']:.3f}", config)
                    else:
                        log(f"    {roi_name}: 分析失败", config)
                        
                except Exception as e:
                    log(f"    {roi_name} 分析出错: {str(e)}", config)
                    continue
            
            # 内存清理
            memory_cleanup()
            
        except Exception as e:
            log(f"被试 {subject_id} 处理失败: {str(e)}", config)
            continue
    
    if not all_results:
        raise ValueError("没有成功的分析结果")
    
    # 创建组水平数据框
    group_df = pd.DataFrame(all_results)
    log(f"\n组水平分析：共收集到 {len(group_df)} 个有效结果", config)
    log(f"涉及被试: {sorted(group_df['subject'].unique())}", config)
    log(f"涉及ROI: {sorted(group_df['roi'].unique())}", config)
    
    # 保存原始结果
    group_df.to_csv(config.results_dir / "group_results_raw.csv", index=False)
    
    # 执行组水平统计分析
    log("\n执行组水平统计分析...", config)
    stats_df = perform_group_statistics_corrected(group_df, config)
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistics_corrected.csv", index=False)
    
    # 生成可视化
    log("\n生成可视化图表...", config)
    try:
        create_enhanced_visualizations(group_df, stats_df, config)
        create_individual_roi_plots(group_df, stats_df, config)
        log("可视化生成完成", config)
    except Exception as e:
        log(f"可视化生成失败: {str(e)}", config)
    
    # 生成HTML报告
    log("\n生成HTML报告...", config)
    try:
        report_path = generate_html_report(group_df, stats_df, config)
        log(f"HTML报告生成完成: {report_path}", config)
    except Exception as e:
        log(f"HTML报告生成失败: {str(e)}", config)
    
    log("\n=== 分析完成 ===", config)
    log(f"结果保存在: {config.results_dir}", config)
    
    return config.results_dir

def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()
    
    try:
        # 运行分析
        results_dir = run_group_roi_analysis_corrected(config)
        print(f"\n✅ 分析完成！结果保存在: {results_dir}")
        return results_dir
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()