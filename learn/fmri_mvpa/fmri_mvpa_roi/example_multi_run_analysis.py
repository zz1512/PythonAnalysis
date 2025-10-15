# example_multi_run_analysis.py
# 演示如何使用多run合并分析功能

from fmri_mvpa_roi_pipeline_enhanced import MVPAConfig, run_group_roi_analysis_enhanced
from pathlib import Path

def run_multi_run_example():
    """运行多run合并分析示例"""
    
    # 创建配置
    config = MVPAConfig()
    
    # 配置多run分析
    config.runs = [3, 4]  # 合并run3和run4的数据
    
    # 可以根据需要调整其他参数
    config.subjects = [f"sub-{i:02d}" for i in range(1, 5)]  # 4个被试
    config.n_permutations = 100  # 置换检验次数
    config.cv_folds = 5  # 交叉验证折数
    config.correction_method = 'fdr_bh'  # FDR校正
    config.use_parallel = True  # 启用并行处理
    
    # 设置数据路径（请根据实际情况修改）
    config.lss_root = Path(r"H:\PythonAnalysis\learn_LSS")
    config.roi_dir = Path(r"H:\PythonAnalysis\learn_mvpa\full_roi_mask")
    config.results_dir = Path(r"H:\PythonAnalysis\learn_mvpa\metaphor_ROI_MVPA_multi_run")
    
    # 对比条件
    config.contrasts = [
        ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间
    ]
    
    print("=== 多run合并MVPA分析配置 ===")
    print(f"被试: {config.subjects}")
    print(f"合并的runs: {config.runs}")
    print(f"对比条件: {config.contrasts}")
    print(f"结果目录: {config.results_dir}")
    print(f"预期每个被试总trial数: ~140 (70 per run × 2 runs)")
    print()
    
    # 运行分析
    try:
        results = run_group_roi_analysis_enhanced(config)
        
        if results is not None:
            group_df, stats_df = results
            print("\n=== 分析完成 ===")
            print(f"总测试数: {len(stats_df)}")
            print(f"显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}")
            print(f"详细报告: {config.results_dir / 'analysis_report.html'}")
            
            # 显示一些关键统计信息
            print("\n=== 数据统计 ===")
            subjects_analyzed = group_df['subject'].nunique()
            print(f"成功分析的被试数: {subjects_analyzed}")
            
            # 显示每个被试的trial数量（如果有的话）
            if 'n_samples' in group_df.columns:
                sample_counts = group_df.groupby('subject')['n_samples'].first()
                print("\n每个被试的trial数量:")
                for subject, n_samples in sample_counts.items():
                    print(f"  {subject}: {n_samples} trials")
            
            return True
        else:
            print("分析失败")
            return False
            
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_availability():
    """检查多run数据的可用性"""
    config = MVPAConfig()
    config.runs = [3, 4]
    
    print("=== 检查数据可用性 ===")
    
    for subject in config.subjects:
        print(f"\n检查 {subject}:")
        for run in config.runs:
            lss_dir = config.lss_root / subject / f"run-{run}_LSS"
            trial_map_path = lss_dir / "trial_map.csv"
            
            if lss_dir.exists():
                print(f"  ✓ run-{run} 目录存在: {lss_dir}")
                if trial_map_path.exists():
                    import pandas as pd
                    trial_info = pd.read_csv(trial_map_path)
                    print(f"    ✓ trial_map.csv 存在，包含 {len(trial_info)} 个trial")
                    
                    # 检查beta文件
                    beta_count = 0
                    for _, trial in trial_info.iterrows():
                        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
                        if beta_path.exists():
                            beta_count += 1
                    print(f"    ✓ 找到 {beta_count}/{len(trial_info)} 个beta文件")
                    
                    # 显示条件统计
                    if 'trial_condition' in trial_info.columns:
                        condition_counts = trial_info['trial_condition'].value_counts()
                        print(f"    条件分布: {dict(condition_counts)}")
                else:
                    print(f"    ✗ trial_map.csv 不存在")
            else:
                print(f"  ✗ run-{run} 目录不存在: {lss_dir}")

if __name__ == "__main__":
    print("fMRI MVPA 多run合并分析示例")
    print("=" * 50)
    
    # 首先检查数据可用性
    check_data_availability()
    
    print("\n" + "=" * 50)
    
    # 询问是否继续运行分析
    response = input("\n是否继续运行多run合并分析? (y/n): ")
    if response.lower() in ['y', 'yes', '是']:
        success = run_multi_run_example()
        if success:
            print("\n分析成功完成！")
        else:
            print("\n分析失败，请检查数据和配置。")
    else:
        print("\n分析已取消。")