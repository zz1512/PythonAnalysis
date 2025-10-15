#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版MVPA流水线使用示例
展示如何使用各个增强模块进行fMRI数据分析
"""

import os
import sys
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 导入增强模块
from enhanced_pipeline import EnhancedMVPAPipeline, create_enhanced_pipeline, run_quick_analysis
from global_config import GlobalConfig
from lss_enhanced import LSSAnalysisEnhanced
from quality_control import run_comprehensive_quality_check
from visualization import create_comprehensive_visualization_report

def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("=== 示例1: 基本使用方法 ===")
    
    # 创建增强版流水线
    pipeline = create_enhanced_pipeline(
        output_dir="./example_results_basic",
        enable_cache=True,
        enable_quality_control=True,
        enable_visualization=True,
        n_jobs=2  # 使用2个CPU核心
    )
    
    # 定义分析参数
    subjects = ["sub-01", "sub-02", "sub-03"]
    runs = ["run-1", "run-2"]
    
    try:
        # 运行完整分析
        results = pipeline.run_complete_analysis(
            subjects=subjects,
            runs=runs,
            skip_lss=False,  # 运行LSS分析
            skip_roi=True,   # 跳过ROI创建（需要额外配置）
            skip_mvpa=True   # 跳过MVPA分析（需要LSS结果）
        )
        
        # 查看分析状态
        status = pipeline.get_analysis_status()
        print(f"分析状态: {status}")
        
        print("基本分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
    
    finally:
        # 清理资源
        pipeline.cleanup()

def example_2_custom_config():
    """示例2: 使用自定义配置"""
    print("\n=== 示例2: 使用自定义配置 ===")
    
    # 创建自定义配置
    config = GlobalConfig.create_default_config()
    
    # 修改LSS配置
    config.lss_config.data_dir = "/path/to/your/data"  # 替换为实际数据路径
    config.lss_config.subjects = ["sub-01", "sub-02"]
    config.lss_config.runs = ["run-1"]
    config.lss_config.tr = 2.0
    config.lss_config.smoothing_fwhm = 6.0
    config.lss_config.n_jobs = 4
    
    # 修改MVPA配置
    config.mvpa_config.svm_params['C'] = 0.1
    config.mvpa_config.cv_params['cv_folds'] = 3
    config.mvpa_config.permutation_params['n_permutations'] = 500
    
    # 保存配置
    config_path = "./custom_config.json"
    config.save_config(config_path)
    print(f"自定义配置已保存到: {config_path}")
    
    # 使用自定义配置创建流水线
    pipeline = EnhancedMVPAPipeline(
        config_path=config_path,
        output_dir="./example_results_custom",
        enable_cache=True,
        enable_quality_control=True,
        enable_visualization=True
    )
    
    print("使用自定义配置的流水线已创建")
    
    # 清理
    pipeline.cleanup()

def example_3_lss_only():
    """示例3: 仅运行LSS分析"""
    print("\n=== 示例3: 仅运行LSS分析 ===")
    
    # 创建LSS配置
    from global_config import LSSConfig
    
    lss_config = LSSConfig(
        data_dir="/path/to/your/data",  # 替换为实际数据路径
        output_dir="./lss_only_results",
        subjects=["sub-01"],
        runs=["run-1"],
        task_name="metaphor",
        tr=2.0,
        smoothing_fwhm=6.0,
        n_jobs=2
    )
    
    # 创建LSS分析器
    lss_analyzer = LSSAnalysisEnhanced(
        config=lss_config,
        n_jobs=2
    )
    
    try:
        # 运行LSS分析
        print("开始LSS分析...")
        results = lss_analyzer.run_group_lss_analysis(
            subjects=lss_config.subjects,
            runs=lss_config.runs
        )
        
        print(f"LSS分析完成，处理了 {len(results)} 个被试")
        
    except Exception as e:
        print(f"LSS分析出现错误: {str(e)}")

def example_4_quality_control():
    """示例4: 数据质量控制"""
    print("\n=== 示例4: 数据质量控制 ===")
    
    # 示例数据文件路径（需要替换为实际路径）
    func_file = "/path/to/sub-01/func/sub-01_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    
    if os.path.exists(func_file):
        try:
            # 运行质量控制检查
            quality_results = run_comprehensive_quality_check(
                func_file=func_file,
                output_dir="./quality_control_results",
                subject_id="sub-01",
                run_id="run-1"
            )
            
            print("质量控制检查完成")
            print(f"质量评级: {quality_results.get('quality_rating', 'Unknown')}")
            print(f"平均SNR: {quality_results.get('mean_snr', 'N/A')}")
            print(f"平均tSNR: {quality_results.get('mean_tsnr', 'N/A')}")
            
        except Exception as e:
            print(f"质量控制检查出现错误: {str(e)}")
    else:
        print(f"数据文件不存在: {func_file}")
        print("请替换为实际的数据文件路径")

def example_5_visualization():
    """示例5: 结果可视化"""
    print("\n=== 示例5: 结果可视化 ===")
    
    # 创建示例MVPA结果数据
    example_results = {
        'roi_results': {
            'left_IFG': {
                'accuracy': 0.75,
                'permutation_pvalue': 0.001,
                'cv_scores': [0.7, 0.8, 0.75, 0.72, 0.78]
            },
            'right_IFG': {
                'accuracy': 0.68,
                'permutation_pvalue': 0.023,
                'cv_scores': [0.65, 0.7, 0.68, 0.66, 0.71]
            },
            'left_MTG': {
                'accuracy': 0.82,
                'permutation_pvalue': 0.0001,
                'cv_scores': [0.8, 0.85, 0.82, 0.81, 0.84]
            },
            'right_MTG': {
                'accuracy': 0.55,
                'permutation_pvalue': 0.234,
                'cv_scores': [0.5, 0.6, 0.55, 0.52, 0.58]
            }
        }
    }
    
    # 创建示例质量数据
    example_quality = {
        'sub-01': {
            'run-1': {
                'mean_snr': 45.2,
                'mean_tsnr': 78.5,
                'mean_fd': 0.12,
                'quality_rating': 'Good'
            },
            'run-2': {
                'mean_snr': 42.8,
                'mean_tsnr': 75.3,
                'mean_fd': 0.15,
                'quality_rating': 'Good'
            }
        },
        'sub-02': {
            'run-1': {
                'mean_snr': 38.9,
                'mean_tsnr': 68.2,
                'mean_fd': 0.28,
                'quality_rating': 'Fair'
            }
        }
    }
    
    try:
        # 生成综合可视化报告
        viz_results = create_comprehensive_visualization_report(
            analysis_results=example_results,
            quality_data=example_quality,
            config=None,
            output_dir="./visualization_results"
        )
        
        print("可视化报告生成完成")
        print("生成的文件:")
        for name, path in viz_results.items():
            print(f"  {name}: {path}")
            
    except Exception as e:
        print(f"可视化生成出现错误: {str(e)}")

def example_6_quick_analysis():
    """示例6: 快速分析"""
    print("\n=== 示例6: 快速分析 ===")
    
    try:
        # 使用快速分析函数
        results = run_quick_analysis(
            data_dir="/path/to/your/data",  # 替换为实际数据路径
            subjects=["sub-01"],
            runs=["run-1"],
            output_dir="./quick_analysis_results",
            enable_cache=True,
            enable_quality_control=True,
            enable_visualization=True,
            n_jobs=2
        )
        
        print("快速分析完成")
        print(f"结果摘要: {results.get('results_summary', {})}")
        
    except Exception as e:
        print(f"快速分析出现错误: {str(e)}")

def example_7_step_by_step():
    """示例7: 分步骤运行分析"""
    print("\n=== 示例7: 分步骤运行分析 ===")
    
    # 创建流水线
    pipeline = create_enhanced_pipeline(
        output_dir="./step_by_step_results",
        enable_cache=True,
        n_jobs=2
    )
    
    subjects = ["sub-01"]
    runs = ["run-1"]
    
    try:
        # 步骤1: 仅运行LSS分析
        print("步骤1: 运行LSS分析")
        lss_results = pipeline.run_lss_analysis(subjects, runs)
        print(f"LSS分析完成: {len(lss_results)} 个被试")
        
        # 步骤2: 运行质量控制
        print("\n步骤2: 运行质量控制")
        quality_results = pipeline.run_quality_control(subjects, runs)
        print(f"质量控制完成: {len(quality_results)} 个被试")
        
        # 步骤3: 生成可视化
        print("\n步骤3: 生成可视化")
        if pipeline.enable_visualization:
            viz_results = pipeline.generate_visualizations_and_reports()
            print(f"可视化完成: {len(viz_results)} 个文件")
        
        # 保存结果
        pipeline.save_final_results()
        print("\n所有步骤完成，结果已保存")
        
    except Exception as e:
        print(f"分步骤分析出现错误: {str(e)}")
    
    finally:
        pipeline.cleanup()

def main():
    """主函数：运行所有示例"""
    print("增强版MVPA流水线使用示例")
    print("=" * 50)
    
    # 注意：在实际使用前，请确保：
    # 1. 替换示例中的数据路径为实际路径
    # 2. 确保数据格式符合BIDS标准
    # 3. 安装所有必要的依赖包
    
    print("\n注意：这些示例使用模拟数据路径")
    print("在实际使用前，请替换为您的真实数据路径")
    print("\n可用的示例:")
    print("1. example_1_basic_usage() - 基本使用方法")
    print("2. example_2_custom_config() - 自定义配置")
    print("3. example_3_lss_only() - 仅LSS分析")
    print("4. example_4_quality_control() - 质量控制")
    print("5. example_5_visualization() - 结果可视化")
    print("6. example_6_quick_analysis() - 快速分析")
    print("7. example_7_step_by_step() - 分步骤分析")
    
    # 运行不需要真实数据的示例
    try:
        example_2_custom_config()  # 配置创建
        example_5_visualization()  # 可视化（使用模拟数据）
    except Exception as e:
        print(f"运行示例时出现错误: {str(e)}")
    
    print("\n示例演示完成！")
    print("要运行完整分析，请：")
    print("1. 准备符合BIDS格式的fMRI数据")
    print("2. 修改配置文件中的数据路径")
    print("3. 运行相应的示例函数")

if __name__ == "__main__":
    main()