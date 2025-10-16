#!/usr/bin/env python3
"""
fMRI GLM 增强分析运行脚本

这个脚本提供了一个简单的接口来运行优化后的fMRI GLM分析流程。
用户可以通过修改配置或命令行参数来自定义分析。
"""

import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config_enhance import AnalysisConfig, default_config
from fmri_glm_enhance import (
    run_first_level_analysis, 
    run_second_level_analysis, 
    create_visualization,
    run_complete_analysis,
    logger
)

def create_config_from_args(args) -> AnalysisConfig:
    """根据命令行参数创建配置"""
    config = AnalysisConfig()
    
    # 应用命令行参数
    if args.subjects:
        try:
            if '-' in args.subjects:
                # 范围格式：1-5
                parts = args.subjects.split('-')
                if len(parts) != 2:
                    raise ValueError("Range format should be 'start-end'")
                start, end = map(int, parts)
                if start > end or start < 1:
                    raise ValueError("Invalid subject range")
                config.SUBJECTS = [f"sub-{i:02d}" for i in range(start, end + 1)]
            else:
                # 逗号分隔格式：1,2,3
                subject_nums = [int(x.strip()) for x in args.subjects.split(',')]
                if any(num < 1 for num in subject_nums):
                    raise ValueError("Subject numbers must be positive")
                config.SUBJECTS = [f"sub-{i:02d}" for i in subject_nums]
        except ValueError as e:
            print(f"Error parsing subjects parameter: {e}")
            print("Expected format: '1-5' for range or '1,2,3' for specific subjects")
            sys.exit(1)
    
    if args.runs:
        try:
            config.RUNS = [int(x.strip()) for x in args.runs.split(',')]
            if any(run < 1 for run in config.RUNS):
                raise ValueError("Run numbers must be positive")
        except ValueError as e:
            print(f"Error parsing runs parameter: {e}")
            print("Expected format: '1,2,3,4' for comma-separated run numbers")
            sys.exit(1)
    
    if args.smoothing is not None:
        config.SMOOTHING_FWHM = args.smoothing if args.smoothing > 0 else None
    
    if args.output:
        config.OUTPUT_ROOT = Path(args.output)
    
    if args.no_multiprocess:
        config.USE_MULTIPROCESS = False
        config.MAX_WORKERS = 1
    elif args.workers:
        if args.workers < 1 or args.workers > 32:
            print("Error: Number of workers must be between 1 and 32")
            sys.exit(1)
        config.MAX_WORKERS = args.workers
    
    if args.fdr_alpha:
        if not (0 < args.fdr_alpha < 1):
            print("Error: FDR alpha must be between 0 and 1")
            sys.exit(1)
        config.SECOND_LEVEL_CONFIG["alpha"] = args.fdr_alpha
        config.SECOND_LEVEL_CONFIG["use_fdr"] = True
    
    if args.uncorrected_p:
        if not (0 < args.uncorrected_p < 1):
            print("Error: Uncorrected p-value must be between 0 and 1")
            sys.exit(1)
        config.SECOND_LEVEL_CONFIG["p_unc"] = args.uncorrected_p
        config.SECOND_LEVEL_CONFIG["use_fdr"] = False
    
    return config

def run_analysis_with_config(config: AnalysisConfig):
    """使用指定配置运行分析"""
    # 验证配置
    if not config.validate_config():
        logger.error("Configuration validation failed")
        return False
    
    # 显示配置摘要
    config.print_summary()
    
    # 创建输出目录
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 更新全局配置
    import fmri_glm_enhance
    fmri_glm_enhance.update_config(config)
    
    # 运行完整分析
    try:
        run_complete_analysis()
        return True
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="fMRI GLM Enhanced Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置运行所有被试
  python run_enhanced_analysis.py
  
  # 只分析前5个被试，不使用平滑
  python run_enhanced_analysis.py --subjects 1-5 --smoothing 0
  
  # 分析特定被试，使用4mm平滑，单进程模式
  python run_enhanced_analysis.py --subjects 1,3,5 --smoothing 4 --no-multiprocess
  
  # 使用未校正阈值进行二阶分析
  python run_enhanced_analysis.py --uncorrected-p 0.001
        """
    )
    
    # 数据选择参数
    parser.add_argument(
        '--subjects', '-s',
        help='被试选择：范围格式(1-5)或逗号分隔(1,2,3)'
    )
    
    parser.add_argument(
        '--runs', '-r',
        help='Run选择：逗号分隔(1,2,3,4)'
    )
    
    # 分析参数
    parser.add_argument(
        '--smoothing',
        type=float,
        help='平滑核大小(FWHM, mm)，设置为0表示不平滑'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出目录路径'
    )
    
    # 并行处理参数
    parser.add_argument(
        '--no-multiprocess',
        action='store_true',
        help='禁用多进程处理'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='并行工作进程数'
    )
    
    # 统计阈值参数
    parser.add_argument(
        '--fdr-alpha',
        type=float,
        help='FDR校正的alpha值'
    )
    
    parser.add_argument(
        '--uncorrected-p',
        type=float,
        help='未校正p值阈值（禁用FDR校正）'
    )
    
    # 其他选项
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='只显示配置信息，不运行分析'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    args = parser.parse_args()
    
    # 创建配置
    if any(vars(args).values()):
        config = create_config_from_args(args)
    else:
        config = default_config
    
    # 设置详细模式
    if args.verbose:
        config.LOG_LEVEL = 10  # DEBUG level
        config.VERBOSE = 2
    
    # 如果只是查看配置
    if args.config_only:
        config.print_summary()
        return
    
    # 运行分析
    success = run_analysis_with_config(config)
    
    if success:
        print("\n=== 分析完成 ===")
        print(f"结果保存在: {config.OUTPUT_ROOT}")
        print("\n主要输出文件:")
        print("- analysis_summary.json: 分析总结")
        print("- analysis.log: 详细日志")
        print("- sub-*/run-*_spm_style/: 一阶分析结果")
        print("- 2nd_level/: 二阶分析和可视化结果")
    else:
        print("\n=== 分析失败 ===")
        print("请检查日志文件获取详细错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()