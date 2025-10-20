"""
fMRI MVPA ROI Pipeline - 模块化版本

这是重构后的模块化版本，将原来的单一文件拆分为多个功能模块：
- config.py: 配置管理
- utils.py: 工具函数
- data_loader.py: 数据加载
- feature_extraction.py: 特征提取
- classification.py: 分类分析
- statistics.py: 统计分析
- visualization.py: 可视化
- report_generator.py: 报告生成
- main_analysis.py: 主分析流程

使用方法：
    python fmri_mvpa_roi_pipeline_modular.py

作者: AI Assistant
日期: 2024
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import MVPAConfig
from utils import log, memory_cleanup
from data_loader import load_roi_masks, load_multi_run_lss_data
from classification import run_roi_classification_enhanced, prepare_classification_data
from feature_extraction import extract_roi_features
from group_statistics import perform_group_statistics_corrected
from visualization import create_enhanced_visualizations
from report_generator import generate_html_report

def run_group_roi_analysis_corrected(config):
    """运行组水平ROI分析（修正版本）"""
    log("开始组水平ROI分析", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")
    
    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    if not roi_masks:
        raise ValueError("未找到有效的ROI mask文件")
    
    log(f"成功加载 {len(roi_masks)} 个ROI masks", config)
    
    # 初始化结果存储
    all_results = []
    
    # 对每个被试进行分析
    for subject_id in config.subjects:
        log(f"\n处理被试: {subject_id}", config)
        
        try:
            # 加载该被试的LSS数据
            beta_images, trial_info = load_multi_run_lss_data(subject_id, config.runs, config)
            if beta_images is None or trial_info is None:
                log(f"被试 {subject_id} 的LSS数据加载失败，跳过", config)
                continue
            
            # 对每个对比条件进行分析
            for contrast_name, cond1, cond2 in config.contrasts:
                log(f"  分析对比: {contrast_name} ({cond1} vs {cond2})", config)
                
                # 准备分类数据
                selected_betas, labels, trial_details = prepare_classification_data(
                    trial_info, beta_images, cond1, cond2, config
                )
                
                if selected_betas is None:
                    log(f"跳过 {subject_id} {cond1} vs {cond2}: 数据准备失败", config)
                    continue
                
                # 对每个ROI进行分类分析
                for roi_name, roi_mask_path in roi_masks.items():
                    try:
                        # 提取ROI特征
                        X = extract_roi_features(selected_betas, roi_mask_path, config)
                        
                        if X is None or X.shape[1] == 0:
                            continue
                        
                        # 运行分类分析
                        result = run_roi_classification_enhanced(X, labels, config)
                        
                        if result is not None:
                            result.update({
                                'subject': subject_id,
                                'roi': roi_name,
                                'contrast': contrast_name,
                                'condition1': cond1,
                                'condition2': cond2,
                                'n_trials_cond1': np.sum(labels == 0),
                                'n_trials_cond2': np.sum(labels == 1)
                            })
                            all_results.append(result)
                            
                    except Exception as e:
                        log(f"      {roi_name} 分析出错: {str(e)}", config)
                        continue
            
            # 内存清理
            memory_cleanup()
            
        except Exception as e:
            log(f"被试 {subject_id} 处理失败: {str(e)}", config)
            continue
    
    # 如果有结果，进行组水平分析
    if all_results:
        log(f"\n开始组水平统计分析，共 {len(all_results)} 个结果", config)
        
        # 转换为DataFrame
        group_df = pd.DataFrame(all_results)
        
        # 保存个体结果
        group_df.to_csv(config.results_dir / "individual_roi_mvpa_results.csv", index=False)
        
        # 执行组水平统计分析
        stats_df = perform_group_statistics_corrected(group_df, config)
        
        # 保存组水平统计结果
        stats_df.to_csv(config.results_dir / "group_roi_mvpa_statistics.csv", index=False)
        
        # 创建可视化
        create_enhanced_visualizations(group_df, stats_df, config)
        
        # 生成HTML报告
        generate_html_report(group_df, stats_df, config)
        
        log(f"\n分析完成！结果保存在: {config.results_dir}", config)
        log(f"分析被试数: {len(set(group_df['subject']))}", config)
        log(f"ROI数量: {len(roi_masks)}", config)
        log(f"对比数量: {len(config.contrasts)}", config)
        
        return group_df, stats_df
    else:
        log("没有有效的分析结果", config)
        return None, None

def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()
    
    try:
        # 运行分析
        group_df, stats_df = run_group_roi_analysis_corrected(config)
        if group_df is not None:
            print(f"\n✅ 分析完成！结果保存在: {config.results_dir}")
            return config.results_dir
        else:
            print("\n❌ 分析失败：没有有效结果")
            return None
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()