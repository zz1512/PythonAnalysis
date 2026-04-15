#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Enhanced MVPA Pipeline v2.0
增强版MVPA流水线使用示例 - 最新版本

展示如何使用集成了最新LSS、ROI和MVPA优化算法的增强版流水线
"""

import sys
from pathlib import Path
import warnings
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from enhanced_pipeline import EnhancedMVPAPipeline
from global_config import GlobalConfig
from lss_optimized import run_optimized_lss_analysis
from roi_mask_creator import create_standard_rois
from mvpa_enhanced import run_enhanced_mvpa_analysis

warnings.filterwarnings('ignore')

def example_enhanced_pipeline_v2():
    """增强版流水线v2.0完整示例"""
    print("=== 增强版MVPA流水线 v2.0 完整示例 ===")
    
    # 创建默认配置
    config = GlobalConfig.create_default_config()
    
    # 可选：自定义配置参数
    config.lss.subjects = [f"sub-{i:02d}" for i in range(1, 6)]  # 5个被试
    config.lss.runs = [3, 4]  # 使用run 3和4
    config.mvpa.contrasts = [
        ("face_vs_house", "face", "house"),
        ("face_vs_object", "face", "object"),
        ("house_vs_object", "house", "object")
    ]
    
    # 创建增强版流水线
    pipeline = EnhancedMVPAPipeline(config=config)
    
    try:
        # 运行完整的增强版分析
        results = pipeline.run_complete_analysis_enhanced(
            subjects=config.lss.subjects,
            runs=config.lss.runs,
            contrasts=config.mvpa.contrasts,
            create_rois=True,
            enable_quality_control=True,
            enable_visualization=True
        )
        
        # 打印结果摘要
        print("\n=== 分析结果摘要 ===")
        print(f"LSS分析: {results['lss_results'].get('summary', {})}")
        print(f"ROI创建: {len(results['roi_results'].get('created_rois', []))} 个ROI")
        print(f"MVPA分析: {results['mvpa_results'].get('summary', {})}")
        print(f"总耗时: {results['analysis_info']['total_duration']}")
        
        return results
        
    except Exception as e:
        print(f"分析失败: {e}")
        return None
    
    finally:
        pipeline.cleanup()

def example_individual_components():
    """单独组件使用示例"""
    print("\n=== 单独组件使用示例 ===")
    
    config = GlobalConfig.create_default_config()
    
    # 1. 单独使用优化版LSS分析
    print("\n1. 优化版LSS分析")
    try:
        lss_results = run_optimized_lss_analysis(
            config=config.lss,
            subjects=config.lss.subjects[:2],  # 只分析前2个被试
            runs=config.lss.runs
        )
        print(f"LSS分析完成: {lss_results.get('summary', {})}")
    except Exception as e:
        print(f"LSS分析失败: {e}")
    
    # 2. 单独使用ROI创建器
    print("\n2. ROI mask创建")
    try:
        roi_results = create_standard_rois(
            config=config.roi,
            roi_types=['sphere', 'anatomical']
        )
        print(f"ROI创建完成: {len(roi_results.get('created_rois', []))} 个ROI")
    except Exception as e:
        print(f"ROI创建失败: {e}")
    
    # 3. 单独使用增强版MVPA分析
    print("\n3. 增强版MVPA分析")
    try:
        mvpa_results = run_enhanced_mvpa_analysis(config)
        print(f"MVPA分析完成: {mvpa_results.get('summary', {})}")
    except Exception as e:
        print(f"MVPA分析失败: {e}")

def example_custom_configuration():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = GlobalConfig.create_default_config()
    
    # LSS配置自定义
    config.lss.subjects = ["sub-01", "sub-02", "sub-03"]
    config.lss.runs = [1, 2, 3, 4]
    config.lss.n_jobs_glm = 4  # 增加GLM并行数
    config.lss.standardize = True
    config.lss.use_precomputed_mask = False
    
    # ROI配置自定义
    config.roi.sphere_coords = [(0, 0, 0), (20, -20, 10), (-20, -20, 10)]  # 自定义球形ROI坐标
    config.roi.sphere_radius = 8  # 8mm半径
    config.roi.create_anatomical_rois = True
    config.roi.create_activation_rois = True
    
    # MVPA配置自定义
    config.mvpa.cv_folds = 10  # 10折交叉验证
    config.mvpa.n_permutations = 1000  # 1000次置换检验
    config.mvpa.correction_method = 'bonferroni'  # 使用Bonferroni校正
    config.mvpa.n_jobs = 8  # 8个并行进程
    
    # 保存自定义配置
    config_file = "./custom_config_v2.json"
    config.save_config(config_file)
    print(f"自定义配置已保存: {config_file}")
    
    # 使用自定义配置运行分析
    pipeline = EnhancedMVPAPipeline(config=config)
    
    try:
        results = pipeline.run_complete_analysis_enhanced(
            enable_quality_control=True,
            enable_visualization=True
        )
        print("自定义配置分析完成")
        return results
    except Exception as e:
        print(f"自定义配置分析失败: {e}")
        return None
    finally:
        pipeline.cleanup()

def example_performance_optimization():
    """性能优化示例"""
    print("\n=== 性能优化示例 ===")
    
    config = GlobalConfig.create_default_config()
    
    # 启用所有性能优化选项
    config.lss.use_optimized_glm = True
    config.lss.enable_parallel_subjects = True
    config.lss.enable_parallel_runs = True
    config.lss.memory_efficient = True
    
    config.mvpa.use_parallel = True
    config.mvpa.memory_cache = True
    config.mvpa.batch_size = 100  # 增大批处理大小
    
    # 设置环境变量优化
    import os
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
    pipeline = EnhancedMVPAPipeline(config=config)
    
    try:
        # 监控性能
        import time
        start_time = time.time()
        
        results = pipeline.run_complete_analysis_enhanced()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n性能统计:")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"性能详情: {results.get('performance_stats', {})}")
        
        return results
        
    except Exception as e:
        print(f"性能优化分析失败: {e}")
        return None
    finally:
        pipeline.cleanup()

def example_error_recovery():
    """错误恢复示例"""
    print("\n=== 错误恢复示例 ===")
    
    config = GlobalConfig.create_default_config()
    
    # 故意设置一些可能导致错误的参数
    config.lss.subjects = ["sub-999"]  # 不存在的被试
    
    pipeline = EnhancedMVPAPipeline(config=config)
    
    try:
        # 尝试运行分析，预期会失败
        results = pipeline.run_complete_analysis_enhanced()
        
    except Exception as e:
        print(f"捕获到预期错误: {e}")
        
        # 检查错误统计
        error_stats = pipeline.performance_stats.get('errors', [])
        print(f"错误数量: {len(error_stats)}")
        
        # 尝试恢复：使用有效的被试ID
        print("\n尝试错误恢复...")
        config.lss.subjects = ["sub-01", "sub-02"]  # 使用有效的被试ID
        
        try:
            # 重新运行分析
            recovery_results = pipeline.run_complete_analysis_enhanced()
            print("错误恢复成功！")
            return recovery_results
        except Exception as recovery_error:
            print(f"错误恢复失败: {recovery_error}")
            return None
    
    finally:
        pipeline.cleanup()

def example_batch_processing():
    """批处理示例"""
    print("\n=== 批处理示例 ===")
    
    # 定义多个分析配置
    analysis_configs = [
        {
            'name': 'analysis_1',
            'subjects': ['sub-01', 'sub-02'],
            'runs': [1, 2],
            'contrasts': [('face_vs_house', 'face', 'house')]
        },
        {
            'name': 'analysis_2', 
            'subjects': ['sub-03', 'sub-04'],
            'runs': [3, 4],
            'contrasts': [('face_vs_object', 'face', 'object')]
        }
    ]
    
    all_results = {}
    
    for analysis_config in analysis_configs:
        print(f"\n运行分析: {analysis_config['name']}")
        
        # 创建配置
        config = GlobalConfig.create_default_config()
        config.lss.subjects = analysis_config['subjects']
        config.lss.runs = analysis_config['runs']
        config.mvpa.contrasts = analysis_config['contrasts']
        
        # 创建流水线
        pipeline = EnhancedMVPAPipeline(config=config)
        
        try:
            results = pipeline.run_complete_analysis_enhanced(
                create_rois=False,  # 只在第一次创建ROI
                enable_quality_control=True,
                enable_visualization=True
            )
            
            all_results[analysis_config['name']] = results
            print(f"分析 {analysis_config['name']} 完成")
            
        except Exception as e:
            print(f"分析 {analysis_config['name']} 失败: {e}")
            all_results[analysis_config['name']] = {'error': str(e)}
        
        finally:
            pipeline.cleanup()
    
    # 保存批处理结果
    batch_results_file = "./batch_analysis_results.json"
    with open(batch_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n批处理结果已保存: {batch_results_file}")
    return all_results

def example_configuration_comparison():
    """配置比较示例"""
    print("\n=== 配置比较示例 ===")
    
    # 创建两种不同的配置
    config_standard = GlobalConfig.create_default_config()
    config_optimized = GlobalConfig.create_default_config()
    
    # 优化配置
    config_optimized.lss.n_jobs_glm = 8
    config_optimized.lss.enable_parallel_subjects = True
    config_optimized.mvpa.n_jobs = 8
    config_optimized.mvpa.use_parallel = True
    
    configs = {
        'standard': config_standard,
        'optimized': config_optimized
    }
    
    comparison_results = {}
    
    for config_name, config in configs.items():
        print(f"\n测试配置: {config_name}")
        
        pipeline = EnhancedMVPAPipeline(config=config)
        
        try:
            import time
            start_time = time.time()
            
            results = pipeline.run_complete_analysis_enhanced(
                subjects=config.lss.subjects[:2],  # 只用2个被试进行比较
                create_rois=False,
                enable_quality_control=False,
                enable_visualization=False
            )
            
            end_time = time.time()
            
            comparison_results[config_name] = {
                'duration': end_time - start_time,
                'success': True,
                'performance_stats': results.get('performance_stats', {})
            }
            
            print(f"配置 {config_name} 完成，耗时: {end_time - start_time:.2f} 秒")
            
        except Exception as e:
            comparison_results[config_name] = {
                'duration': None,
                'success': False,
                'error': str(e)
            }
            print(f"配置 {config_name} 失败: {e}")
        
        finally:
            pipeline.cleanup()
    
    # 打印比较结果
    print("\n=== 配置比较结果 ===")
    for config_name, result in comparison_results.items():
        if result['success']:
            print(f"{config_name}: {result['duration']:.2f} 秒")
        else:
            print(f"{config_name}: 失败 - {result['error']}")
    
    return comparison_results

def main():
    """主函数"""
    print("Enhanced MVPA Pipeline v2.0 使用示例")
    print("=" * 60)
    
    # 运行各种示例
    try:
        # 1. 完整流水线示例
        example_enhanced_pipeline_v2()
        
        # 2. 单独组件示例
        example_individual_components()
        
        # 3. 自定义配置示例
        example_custom_configuration()
        
        # 4. 性能优化示例
        example_performance_optimization()
        
        # 5. 错误恢复示例
        example_error_recovery()
        
        # 6. 批处理示例
        example_batch_processing()
        
        # 7. 配置比较示例
        example_configuration_comparison()
        
    except Exception as e:
        print(f"示例执行过程中出现错误: {e}")
    
    print("\n=" * 60)
    print("所有示例执行完成！")
    print("\n使用说明:")
    print("1. 确保数据路径配置正确")
    print("2. 根据实际需求调整配置参数")
    print("3. 检查输出目录中的结果文件")
    print("4. 查看生成的HTML报告获取详细结果")
    print("5. 使用性能优化配置提高分析速度")
    print("6. 利用错误恢复机制处理异常情况")

if __name__ == "__main__":
    main()