# fmri_mvpa_roi_resume_analysis.py
# Enhanced version with FDR correction, parallel processing, memory optimization, and HTML reporting
# 支持断点续传的MVPA ROI分析脚本
# 可以从指定被试开始继续分析，并合并之前的结果生成完整报告

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging
import pickle
import os
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入原始脚本的函数
from fmri_mvpa_roi_pipeline_enhanced import (
    MVPAConfig, log, memory_cleanup, load_roi_masks, 
    analyze_single_subject_roi_enhanced, perform_group_statistics_enhanced,
    create_enhanced_visualizations, generate_html_report
)

class ResumeConfig(MVPAConfig):
    """支持断点续传的MVPA分析配置参数"""
    def __init__(self, start_subject=1, end_subject=28, resume_from=None):
        super().__init__()
        # 修改被试范围为1-28
        self.all_subjects = [f"sub-{i:02d}" for i in range(start_subject, end_subject + 1)]
        self.subjects = self.all_subjects  # 默认分析所有被试
        
        # 断点续传相关参数
        self.resume_from = resume_from  # 从哪个被试开始继续分析
        self.backup_dir = self.results_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 中间结果保存文件
        self.subject_results_file = self.backup_dir / "subject_results.pkl"
        self.progress_file = self.backup_dir / "analysis_progress.json"
        
        # 增强的错误处理和重试机制
        self.max_retries = 2
        self.retry_failed = True
        
        # 如果指定了resume_from，则只分析从该被试开始的被试
        if self.resume_from:
            try:
                resume_idx = int(self.resume_from.split('-')[1])
                self.subjects = [f"sub-{i:02d}" for i in range(resume_idx, end_subject + 1)]
                log(f"将从被试 {self.resume_from} 开始继续分析")
            except Exception as e:
                log(f"警告: 无法解析resume_from参数 {self.resume_from}: {e}，将分析所有被试")
    
    def save_progress(self, completed_subjects, failed_subjects=None, retry_subjects=None):
        """保存分析进度"""
        progress = {
            'completed_subjects': completed_subjects,
            'failed_subjects': failed_subjects or [],
            'retry_subjects': retry_subjects or [],
            'last_update': datetime.now().isoformat(),
            'total_subjects': len(self.all_subjects),
            'analysis_config': {
                'cv_folds': self.cv_folds,
                'n_permutations': self.n_permutations,
                'correction_method': self.correction_method,
                'alpha_level': self.alpha_level
            }
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def load_progress(self):
        """加载分析进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

def save_subject_results(subject_results, filepath):
    """保存被试结果到文件"""
    try:
        # 创建备份
        if filepath.exists():
            backup_path = filepath.with_suffix('.pkl.backup')
            filepath.rename(backup_path)
        
        with open(filepath, 'wb') as f:
            pickle.dump(subject_results, f)
        log(f"被试结果已保存到: {filepath}")
        
        # 删除备份文件
        backup_path = filepath.with_suffix('.pkl.backup')
        if backup_path.exists():
            backup_path.unlink()
            
    except Exception as e:
        log(f"保存被试结果失败: {e}")
        # 恢复备份
        backup_path = filepath.with_suffix('.pkl.backup')
        if backup_path.exists():
            backup_path.rename(filepath)
            log("已恢复备份文件")
        raise

def load_subject_results(filepath):
    """从文件加载被试结果"""
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            log(f"成功加载被试结果: {len(results)} 个被试")
            return results
        except Exception as e:
            log(f"加载被试结果失败: {e}")
            # 尝试加载备份
            backup_path = filepath.with_suffix('.pkl.backup')
            if backup_path.exists():
                try:
                    with open(backup_path, 'rb') as f:
                        results = pickle.load(f)
                    log(f"从备份文件加载被试结果: {len(results)} 个被试")
                    return results
                except Exception as e2:
                    log(f"加载备份文件也失败: {e2}")
    return {}

def merge_subject_results(existing_results, new_results):
    """合并被试结果"""
    merged = existing_results.copy()
    merged.update(new_results)
    log(f"合并结果: 原有 {len(existing_results)} 个，新增 {len(new_results)} 个，总计 {len(merged)} 个被试")
    return merged

def analyze_subject_with_retry(subject, runs, contrasts, roi_masks, config):
    """带重试机制的被试分析"""
    for attempt in range(config.max_retries + 1):
        try:
            log(f"分析被试 {subject} (尝试 {attempt + 1}/{config.max_retries + 1})")
            result = analyze_single_subject_roi_enhanced(
                subject, runs, contrasts, roi_masks, config
            )
            if result is not None:
                return result
            else:
                log(f"被试 {subject} 分析返回None (尝试 {attempt + 1})")
        except Exception as e:
            log(f"被试 {subject} 分析失败 (尝试 {attempt + 1}): {e}")
            if attempt < config.max_retries:
                log(f"将重试被试 {subject}")
                memory_cleanup()  # 清理内存
            else:
                log(f"被试 {subject} 达到最大重试次数，分析失败")
                log(f"错误详情: {traceback.format_exc()}")
    return None

def run_resume_roi_analysis(config):
    """运行支持断点续传的ROI分析"""
    log("开始支持断点续传的隐喻ROI MVPA分析", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")
    
    # 加载ROI掩模
    roi_masks = load_roi_masks(config)
    if roi_masks is None:
        return None
    
    # 加载已有的被试结果
    existing_results = load_subject_results(config.subject_results_file)
    log(f"加载了 {len(existing_results)} 个已完成被试的结果")
    
    # 加载分析进度
    progress = config.load_progress()
    if progress:
        log(f"上次分析进度: 已完成 {len(progress['completed_subjects'])}/{progress['total_subjects']} 个被试")
        log(f"已完成的被试: {progress['completed_subjects']}")
        if progress['failed_subjects']:
            log(f"之前失败的被试: {progress['failed_subjects']}")
    
    # 分析新的被试
    new_subject_results = {}
    completed_subjects = list(existing_results.keys())
    failed_subjects = []
    retry_subjects = []
    
    # 获取需要分析的被试列表
    subjects_to_analyze = [s for s in config.subjects if s not in existing_results]
    log(f"需要分析的被试: {subjects_to_analyze}")
    
    for i, subject in enumerate(subjects_to_analyze):
        log(f"开始分析被试: {subject} ({i+1}/{len(subjects_to_analyze)})")
        
        # 使用带重试机制的分析函数
        subject_results = analyze_subject_with_retry(
            subject, config.runs, config.contrasts, roi_masks, config
        )
        
        if subject_results is not None:
            new_subject_results[subject] = subject_results
            completed_subjects.append(subject)
            log(f"被试 {subject} 分析完成 ({len(new_subject_results)}/{len(subjects_to_analyze)})")
            
            # 定期保存中间结果
            if len(new_subject_results) % 3 == 0:  # 每3个被试保存一次
                merged_results = merge_subject_results(existing_results, new_subject_results)
                save_subject_results(merged_results, config.subject_results_file)
                config.save_progress(completed_subjects, failed_subjects, retry_subjects)
                log(f"中间结果已保存 (已完成 {len(completed_subjects)} 个被试)")
                memory_cleanup()  # 定期清理内存
        else:
            failed_subjects.append(subject)
            log(f"被试 {subject} 最终分析失败")
    
    # 如果启用重试失败的被试
    if config.retry_failed and failed_subjects:
        log(f"开始重试失败的被试: {failed_subjects}")
        retry_results = []
        for subject in failed_subjects[:]:
            log(f"重试被试: {subject}")
            subject_results = analyze_subject_with_retry(
                subject, config.runs, config.contrasts, roi_masks, config
            )
            if subject_results is not None:
                new_subject_results[subject] = subject_results
                completed_subjects.append(subject)
                failed_subjects.remove(subject)
                retry_results.append(subject)
                log(f"重试成功: {subject}")
        
        if retry_results:
            log(f"重试成功的被试: {retry_results}")
            retry_subjects.extend(retry_results)
    
    # 合并所有结果
    all_subject_results = merge_subject_results(existing_results, new_subject_results)
    
    # 保存最终的被试结果
    save_subject_results(all_subject_results, config.subject_results_file)
    config.save_progress(completed_subjects, failed_subjects, retry_subjects)
    
    if not all_subject_results:
        log("错误: 没有获得任何分析结果", config)
        return None
    
    log(f"总共完成 {len(all_subject_results)} 个被试的分析")
    log(f"新分析的被试: {list(new_subject_results.keys())}")
    if failed_subjects:
        log(f"最终失败的被试: {failed_subjects}")
    if retry_subjects:
        log(f"重试成功的被试: {retry_subjects}")
    
    # 收集组水平数据
    group_data = []
    for subject, subject_results in all_subject_results.items():
        for contrast_name, contrast_results in subject_results.items():
            for roi_name, roi_result in contrast_results.items():
                group_data.append({
                    'subject': subject,
                    'contrast': contrast_name,
                    'roi': roi_name,
                    'accuracy': roi_result['mean_accuracy'],
                    'std_accuracy': roi_result['std_accuracy'],
                    'p_value': roi_result['p_value'],
                    'n_trials_cond1': roi_result['n_trials_cond1'],
                    'n_trials_cond2': roi_result['n_trials_cond2'],
                    'n_voxels': roi_result['n_voxels'],
                    'n_features': roi_result['n_features'],
                    'n_samples': roi_result['n_samples'],
                    'best_params': str(roi_result.get('best_params', {})),
                    'best_score': roi_result.get('best_score', np.nan)
                })
    
    group_df = pd.DataFrame(group_data)
    
    # 保存完整的组水平结果
    final_results_file = config.results_dir / "group_roi_mvpa_results_complete.csv"
    group_df.to_csv(final_results_file, index=False)
    log(f"完整的组水平结果已保存到: {final_results_file}")
    
    # 数据质量检查
    log(f"数据质量检查:")
    log(f"  - 总数据点: {len(group_df)}")
    log(f"  - 平均准确率: {group_df['accuracy'].mean():.3f} ± {group_df['accuracy'].std():.3f}")
    log(f"  - 准确率范围: [{group_df['accuracy'].min():.3f}, {group_df['accuracy'].max():.3f}]")
    log(f"  - 缺失值: {group_df.isnull().sum().sum()}")
    
    # 组水平统计分析
    stats_df = perform_group_statistics_enhanced(group_df, config)
    
    # 创建可视化
    create_enhanced_visualizations(group_df, stats_df, config)
    
    # 生成HTML报告
    generate_html_report(group_df, stats_df, config)
    
    # 生成分析总结
    generate_analysis_summary(all_subject_results, group_df, stats_df, config, 
                            new_subject_results, failed_subjects, retry_subjects)
    
    log(f"\n断点续传MVPA分析完成!", config)
    log(f"结果保存在: {config.results_dir}", config)
    log(f"总分析被试数: {len(all_subject_results)}", config)
    log(f"本次新分析被试数: {len(new_subject_results)}", config)
    log(f"ROI数量: {len(roi_masks)}", config)
    log(f"对比数量: {len(config.contrasts)}", config)
    log(f"HTML报告: {config.results_dir / 'analysis_report.html'}", config)
    
    return group_df, stats_df, all_subject_results

def generate_analysis_summary(all_subject_results, group_df, stats_df, config, 
                            new_subject_results, failed_subjects, retry_subjects=None):
    """生成增强的分析总结报告"""
    summary_file = config.results_dir / "analysis_summary.txt"
    
    # 补齐缺失列，避免下游KeyError
    if 'significant_uncorrected' not in stats_df.columns:
        if 'p_value_permutation' in stats_df.columns:
            stats_df['significant_uncorrected'] = stats_df['p_value_permutation'] < config.alpha_level
        elif 'p_value' in stats_df.columns:
            stats_df['significant_uncorrected'] = stats_df['p_value'] < config.alpha_level
        else:
            stats_df['significant_uncorrected'] = False
    
    if 'significant_corrected' not in stats_df.columns:
        if 'p_corrected' in stats_df.columns:
            stats_df['significant_corrected'] = stats_df['p_corrected'] < config.alpha_level
        elif 'p_value' in stats_df.columns:
            stats_df['significant_corrected'] = stats_df['p_value'] < config.alpha_level
        else:
            stats_df['significant_corrected'] = False
    
    # t统计量列名兼容
    t_col = 't_stat' if 't_stat' in stats_df.columns else ('t_statistic' if 't_statistic' in stats_df.columns else None)
    retry_subjects = retry_subjects or []
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("隐喻ROI MVPA分析总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 被试分析情况:\n")
        f.write(f"   - 总被试数: {len(config.all_subjects)}\n")
        f.write(f"   - 成功分析被试数: {len(all_subject_results)}\n")
        f.write(f"   - 本次新分析被试数: {len(new_subject_results)}\n")
        f.write(f"   - 重试成功被试数: {len(retry_subjects)}\n")
        f.write(f"   - 失败被试数: {len(failed_subjects)}\n")
        f.write(f"   - 成功率: {len(all_subject_results)/len(config.all_subjects)*100:.1f}%\n")
        
        if new_subject_results:
            f.write(f"   - 本次新分析被试: {list(new_subject_results.keys())}\n")
        if retry_subjects:
            f.write(f"   - 重试成功被试: {retry_subjects}\n")
        if failed_subjects:
            f.write(f"   - 失败被试: {failed_subjects}\n")
        
        if new_subject_results:
            f.write(f"   - 本次新分析的被试: {list(new_subject_results.keys())}\n")
        
        if failed_subjects:
            f.write(f"   - 失败的被试: {failed_subjects}\n")
        
        f.write(f"\n2. 分析参数:\n")
        f.write(f"   - ROI数量: {len(load_roi_masks(config))}\n")
        f.write(f"   - 对比条件数: {len(config.contrasts)}\n")
        f.write(f"   - 交叉验证折数: {config.cv_folds}\n")
        f.write(f"   - 置换检验次数: {config.n_permutations}\n")
        f.write(f"   - 多重比较校正方法: {config.correction_method}\n")
        
        f.write(f"\n3. 统计结果:\n")
        f.write(f"   - 总测试数: {len(stats_df)}\n")
        f.write(f"   - 显著结果数 (校正前): {len(stats_df[stats_df['significant_uncorrected']])}\n")
        f.write(f"   - 显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}\n")
        
        # 按ROI统计显著结果
        if len(stats_df[stats_df['significant_corrected']]) > 0:
            f.write(f"\n4. 显著结果详情 (校正后):\n")
            sig_results = stats_df[stats_df['significant_corrected']]
            for _, row in sig_results.iterrows():
                t_val = row[t_col] if t_col else float('nan')
                f.write(f"   - {row['roi']}: 平均准确率={row['mean_accuracy']:.3f}, "
                       f"t={t_val:.3f}, p_corrected={row['p_corrected']:.6f}\n")
        
        f.write(f"\n5. 文件输出:\n")
        f.write(f"   - 完整结果: group_roi_mvpa_results_complete.csv\n")
        f.write(f"   - 统计结果: group_statistics_enhanced.csv\n")
        f.write(f"   - HTML报告: analysis_report.html\n")
        f.write(f"   - 被试结果备份: backup/subject_results.pkl\n")
        f.write(f"   - 分析进度: backup/analysis_progress.json\n")
    
    log(f"分析总结报告已保存到: {summary_file}", config)

def main():
    """主函数 - 支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='支持断点续传的隐喻ROI MVPA分析')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='从指定被试开始继续分析 (例如: sub-17)')
    parser.add_argument('--start-subject', type=int, default=1,
                       help='起始被试编号 (默认: 1)')
    parser.add_argument('--end-subject', type=int, default=28,
                       help='结束被试编号 (默认: 28)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ResumeConfig(
        start_subject=args.start_subject,
        end_subject=args.end_subject,
        resume_from=args.resume_from
    )
    
    # 运行分析
    results = run_resume_roi_analysis(config)
    
    if results is not None:
        group_df, stats_df, all_subject_results = results
        logging.info("\n=== 分析完成 ===\n")
        logging.info(f"总被试数: {len(all_subject_results)}")
        logging.info(f"总测试数: {len(stats_df)}")
        logging.info(f"显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}")
        logging.info(f"详细报告: {config.results_dir / 'analysis_report.html'}")
        logging.info(f"分析总结: {config.results_dir / 'analysis_summary.txt'}")
    else:
        logging.error("分析失败")

if __name__ == "__main__":
    main()