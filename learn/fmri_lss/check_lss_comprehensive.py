#!/usr/bin/env python3
"""
LSS分析结果综合校验工具
合并了CSV文件校验和Beta文件质量检查功能

功能:
1. 检查trial_info.csv文件的完整性和行数
2. 检查beta文件的质量和完整性
3. 生成详细的校验报告
4. 支持快速概览和全面检查两种模式
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from datetime import datetime

# 路径配置
OUTPUT_ROOT = Path(r"../../learn_LSS")
SUBJECTS = [f"sub-{i:02d}" for i in range(24, 25)]  # 可根据实际情况调整
RUNS = [3]  # 可根据实际情况调整
TRIALS_PER_RUN = 70  # 每个run的预期trial数量
MIN_ROWS_THRESHOLD = 70  # CSV文件最小行数阈值


class LSS_Validator:
    """LSS分析结果综合校验器"""
    
    def __init__(self, output_root=OUTPUT_ROOT, subjects=SUBJECTS, runs=RUNS, 
                 trials_per_run=TRIALS_PER_RUN, min_rows=MIN_ROWS_THRESHOLD):
        self.output_root = Path(output_root)
        self.subjects = subjects
        self.runs = runs
        self.trials_per_run = trials_per_run
        self.min_rows = min_rows
        
        # 初始化统计信息
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            # CSV相关统计
            'csv_total_files': 0,
            'csv_valid_files': 0,
            'csv_insufficient_rows': 0,
            'csv_missing_files': 0,
            'csv_error_files': 0,
            
            # Beta文件相关统计
            'beta_total_files': 0,
            'beta_valid_files': 0,
            'beta_invalid_files': 0,
            'beta_missing_files': 0,
            'beta_nan_issue_files': 0,
            'beta_zero_issue_files': 0,
            
            # 总体统计
            'total_subjects': 0,
            'total_runs': 0,
            'valid_subjects': 0,
            'problematic_subjects': []
        }
        
        self.detailed_results = {
            'csv_results': [],
            'beta_results': [],
            'subject_summaries': []
        }
    
    def check_csv_file(self, csv_path):
        """检查单个trial_info.csv文件"""
        try:
            # 快速计算CSV文件行数（减去标题行）
            with open(csv_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for row in f) - 1  # 减去标题行
            
            # 读取CSV内容进行更详细的检查
            df = pd.read_csv(csv_path)
            
            result = {
                'file_path': str(csv_path),
                'row_count': row_count,
                'expected_rows': self.trials_per_run,
                'has_sufficient_rows': row_count >= self.min_rows,
                'columns': list(df.columns),
                'has_required_columns': all(col in df.columns for col in ['trial_index', 'original_condition']),
                'unique_conditions': df['original_condition'].unique().tolist() if 'original_condition' in df.columns else [],
                'is_valid': True,
                'error': None
            }
            
            # 更新统计
            self.stats['csv_total_files'] += 1
            if result['has_sufficient_rows'] and result['has_required_columns']:
                self.stats['csv_valid_files'] += 1
            else:
                if not result['has_sufficient_rows']:
                    self.stats['csv_insufficient_rows'] += 1
                    
            return result
            
        except Exception as e:
            self.stats['csv_total_files'] += 1
            self.stats['csv_error_files'] += 1
            return {
                'file_path': str(csv_path),
                'row_count': 0,
                'expected_rows': self.trials_per_run,
                'has_sufficient_rows': False,
                'has_required_columns': False,
                'is_valid': False,
                'error': str(e)
            }
    
    def check_beta_quality(self, beta_path):
        """检查单个beta文件的质量"""
        try:
            img = nib.load(beta_path)
            data = img.get_fdata()
            
            # 计算质量指标
            quality_metrics = {
                "file_path": str(beta_path),
                "data_shape": data.shape,
                "data_range_min": float(np.nanmin(data)),
                "data_range_max": float(np.nanmax(data)),
                "data_mean": float(np.nanmean(data)),
                "data_std": float(np.nanstd(data)),
                "nan_count": int(np.isnan(data).sum()),
                "zero_count": int((data == 0).sum()),
                "total_voxels": int(data.size),
                "zero_ratio": float((data == 0).sum() / data.size),
                "nan_ratio": float(np.isnan(data).sum() / data.size)
            }
            
            # 质量标志
            quality_metrics["has_nan_issue"] = quality_metrics["nan_ratio"] > 0.1
            quality_metrics["has_zero_issue"] = quality_metrics["zero_ratio"] > 0.8
            quality_metrics["is_valid"] = not (quality_metrics["has_nan_issue"] or quality_metrics["has_zero_issue"])
            quality_metrics["error"] = None
            
            return quality_metrics
            
        except Exception as e:
            return {
                "file_path": str(beta_path),
                "error": str(e),
                "is_valid": False,
                "has_nan_issue": True,
                "has_zero_issue": True
            }
    
    def check_subject_run(self, subject, run, check_mode='comprehensive'):
        """检查单个被试单个run的所有文件"""
        run_dir = self.output_root / subject / f"run-{run}_LSS"
        
        if not run_dir.exists():
            print(f"  ❌ 目录不存在: {run_dir}")
            return None
        
        print(f"\n🔍 检查 {subject} | run-{run}")
        print("-" * 50)
        
        run_results = {
            'subject': subject,
            'run': run,
            'run_dir': str(run_dir),
            'csv_result': None,
            'beta_results': [],
            'summary': {}
        }
        
        # 1. 检查trial_info.csv
        csv_path = run_dir / "trial_info.csv"
        if csv_path.exists():
            csv_result = self.check_csv_file(csv_path)
            run_results['csv_result'] = csv_result
            
            if csv_result['is_valid'] and csv_result['has_sufficient_rows']:
                print(f"  ✅ CSV文件: {csv_result['row_count']} 行 (条件: {csv_result['unique_conditions']})")
            else:
                issues = []
                if not csv_result['has_sufficient_rows']:
                    issues.append(f"行数不足({csv_result['row_count']}<{self.min_rows})")
                if not csv_result['has_required_columns']:
                    issues.append("缺少必需列")
                if csv_result['error']:
                    issues.append(f"错误: {csv_result['error']}")
                print(f"  ⚠️  CSV文件问题: {', '.join(issues)}")
        else:
            print(f"  ❌ 缺失: trial_info.csv")
            self.stats['csv_missing_files'] += 1
        
        # 2. 检查beta文件
        beta_files = []
        missing_betas = []
        
        for trial in range(1, self.trials_per_run + 1):
            beta_path = run_dir / f"beta_trial_{trial:03d}.nii.gz"
            if beta_path.exists():
                beta_files.append(beta_path)
            else:
                missing_betas.append(f"beta_trial_{trial:03d}.nii.gz")
        
        if missing_betas:
            print(f"  ⚠️  缺失 {len(missing_betas)} 个beta文件")
            self.stats['beta_missing_files'] += len(missing_betas)
        
        if beta_files:
            if check_mode == 'comprehensive':
                # 全面检查所有beta文件
                print(f"  🔍 检查 {len(beta_files)} 个beta文件...")
                valid_betas = 0
                
                for beta_path in tqdm(beta_files, desc=f"检查 {subject}-run{run}", leave=False):
                    beta_result = self.check_beta_quality(beta_path)
                    beta_result['subject'] = subject
                    beta_result['run'] = run
                    run_results['beta_results'].append(beta_result)
                    
                    # 更新统计
                    self.stats['beta_total_files'] += 1
                    if beta_result.get('is_valid', False):
                        self.stats['beta_valid_files'] += 1
                        valid_betas += 1
                    else:
                        self.stats['beta_invalid_files'] += 1
                        if beta_result.get('has_nan_issue', False):
                            self.stats['beta_nan_issue_files'] += 1
                        if beta_result.get('has_zero_issue', False):
                            self.stats['beta_zero_issue_files'] += 1
                
                valid_ratio = valid_betas / len(beta_files)
                status = "✅" if valid_ratio > 0.9 else "⚠️" if valid_ratio > 0.7 else "❌"
                print(f"  {status} Beta文件: {valid_betas}/{len(beta_files)} 有效 ({valid_ratio:.1%})")
                
            elif check_mode == 'quick':
                # 快速检查：只检查第一个文件
                first_beta = beta_files[0]
                beta_result = self.check_beta_quality(first_beta)
                
                if beta_result.get('is_valid', False):
                    print(f"  ✅ Beta文件示例正常 (范围: [{beta_result['data_range_min']:.3f}, {beta_result['data_range_max']:.3f}])")
                else:
                    print(f"  ❌ Beta文件示例有问题")
                
                print(f"  📊 找到 {len(beta_files)}/{self.trials_per_run} 个beta文件")
        else:
            print(f"  ❌ 未找到任何beta文件")
        
        # 生成run汇总
        run_results['summary'] = {
            'csv_valid': run_results['csv_result']['is_valid'] if run_results['csv_result'] else False,
            'beta_count': len(beta_files),
            'beta_expected': self.trials_per_run,
            'beta_missing': len(missing_betas),
            'run_complete': len(beta_files) == self.trials_per_run and (run_results['csv_result']['is_valid'] if run_results['csv_result'] else False)
        }
        
        return run_results
    
    def comprehensive_check(self):
        """全面检查所有被试和run"""
        print("=" * 60)
        print("LSS分析结果综合校验 - 全面模式")
        print("=" * 60)
        print(f"检查路径: {self.output_root}")
        print(f"被试: {self.subjects}")
        print(f"Runs: {self.runs}")
        print(f"预期每run trial数: {self.trials_per_run}")
        
        self.reset_stats()
        
        for subject in self.subjects:
            subject_valid_runs = 0
            subject_total_runs = 0
            
            for run in self.runs:
                run_result = self.check_subject_run(subject, run, 'comprehensive')
                
                if run_result:
                    subject_total_runs += 1
                    self.stats['total_runs'] += 1
                    
                    if run_result['summary']['run_complete']:
                        subject_valid_runs += 1
                    
                    # 保存详细结果
                    if run_result['csv_result']:
                        self.detailed_results['csv_results'].append(run_result['csv_result'])
                    
                    self.detailed_results['beta_results'].extend(run_result['beta_results'])
            
            # 被试汇总
            if subject_total_runs > 0:
                self.stats['total_subjects'] += 1
                valid_ratio = subject_valid_runs / subject_total_runs
                
                if valid_ratio >= 0.8:
                    self.stats['valid_subjects'] += 1
                    status = "✅"
                else:
                    self.stats['problematic_subjects'].append(subject)
                    status = "⚠️" if valid_ratio > 0.5 else "❌"
                
                subject_summary = {
                    'subject': subject,
                    'total_runs': subject_total_runs,
                    'valid_runs': subject_valid_runs,
                    'valid_ratio': valid_ratio,
                    'status': status
                }
                
                self.detailed_results['subject_summaries'].append(subject_summary)
                print(f"\n{status} {subject}: {subject_valid_runs}/{subject_total_runs} 完整runs ({valid_ratio:.1%})")
        
        # 生成报告
        self.generate_comprehensive_report()
        
        return self.detailed_results, self.stats
    
    def quick_check(self):
        """快速检查概览"""
        print("🚀 LSS分析结果快速概览")
        print("=" * 40)
        
        self.reset_stats()
        
        for subject in self.subjects:
            print(f"\n{subject}:")
            
            for run in self.runs:
                run_dir = self.output_root / subject / f"run-{run}_LSS"
                
                if run_dir.exists():
                    # 检查CSV
                    csv_path = run_dir / "trial_info.csv"
                    csv_status = "✅" if csv_path.exists() else "❌"
                    
                    # 检查beta文件数量
                    beta_files = list(run_dir.glob("beta_trial_*.nii.gz"))
                    beta_status = "✅" if len(beta_files) == self.trials_per_run else "⚠️" if len(beta_files) > 0 else "❌"
                    
                    print(f"  run-{run}: CSV {csv_status} | Beta {beta_status} ({len(beta_files)}/{self.trials_per_run})")
                    
                    # 快速检查第一个beta文件质量
                    if beta_files:
                        first_beta = beta_files[0]
                        result = self.check_beta_quality(first_beta)
                        quality_status = "✅" if result.get('is_valid', False) else "❌"
                        print(f"    质量示例: {quality_status}")
                else:
                    print(f"  run-{run}: ❌ 目录不存在")
    
    def generate_comprehensive_report(self):
        """生成详细的综合报告"""
        print("\n" + "=" * 60)
        print("📊 LSS分析结果综合校验报告")
        print("=" * 60)
        
        # 总体统计
        print(f"\n📈 总体统计:")
        print(f"  被试数量: {self.stats['total_subjects']}")
        print(f"  Run数量: {self.stats['total_runs']}")
        print(f"  有效被试: {self.stats['valid_subjects']} ({self.stats['valid_subjects']/max(1, self.stats['total_subjects']):.1%})")
        
        # CSV文件统计
        print(f"\n📄 CSV文件统计:")
        print(f"  总CSV文件: {self.stats['csv_total_files']}")
        print(f"  有效CSV文件: {self.stats['csv_valid_files']} ({self.stats['csv_valid_files']/max(1, self.stats['csv_total_files']):.1%})")
        print(f"  行数不足文件: {self.stats['csv_insufficient_rows']}")
        print(f"  缺失CSV文件: {self.stats['csv_missing_files']}")
        print(f"  错误CSV文件: {self.stats['csv_error_files']}")
        
        # Beta文件统计
        print(f"\n🧠 Beta文件统计:")
        print(f"  总Beta文件: {self.stats['beta_total_files']}")
        print(f"  有效Beta文件: {self.stats['beta_valid_files']} ({self.stats['beta_valid_files']/max(1, self.stats['beta_total_files']):.1%})")
        print(f"  无效Beta文件: {self.stats['beta_invalid_files']}")
        print(f"  缺失Beta文件: {self.stats['beta_missing_files']}")
        print(f"  NaN问题文件: {self.stats['beta_nan_issue_files']}")
        print(f"  零值问题文件: {self.stats['beta_zero_issue_files']}")
        
        # 问题被试
        if self.stats['problematic_subjects']:
            print(f"\n⚠️  需要关注的被试 ({len(self.stats['problematic_subjects'])}个):")
            for subject in self.stats['problematic_subjects']:
                print(f"  - {subject}")
        
        # 保存详细报告
        self.save_reports()
    
    def save_reports(self):
        """保存详细报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        if not self.output_root.exists():
            print(f"\n⚠️  输出目录不存在: {self.output_root}")
            print("📁 创建报告目录: ./reports")
            report_dir = Path("./reports")
            report_dir.mkdir(exist_ok=True)
        else:
            report_dir = self.output_root
        
        # 保存CSV检查结果
        if self.detailed_results['csv_results']:
            csv_df = pd.DataFrame(self.detailed_results['csv_results'])
            csv_report_path = report_dir / f"csv_validation_report_{timestamp}.csv"
            csv_df.to_csv(csv_report_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 CSV校验报告已保存至: {csv_report_path}")
        
        # 保存Beta检查结果
        if self.detailed_results['beta_results']:
            beta_df = pd.DataFrame(self.detailed_results['beta_results'])
            beta_report_path = report_dir / f"beta_quality_report_{timestamp}.csv"
            beta_df.to_csv(beta_report_path, index=False, encoding="utf-8-sig")
            print(f"💾 Beta质量报告已保存至: {beta_report_path}")
        
        # 保存被试汇总
        if self.detailed_results['subject_summaries']:
            subject_df = pd.DataFrame(self.detailed_results['subject_summaries'])
            subject_report_path = report_dir / f"subject_summary_report_{timestamp}.csv"
            subject_df.to_csv(subject_report_path, index=False, encoding="utf-8-sig")
            print(f"💾 被试汇总报告已保存至: {subject_report_path}")
        
        # 保存统计汇总
        stats_df = pd.DataFrame([self.stats])
        stats_report_path = report_dir / f"validation_summary_{timestamp}.csv"
        stats_df.to_csv(stats_report_path, index=False, encoding="utf-8-sig")
        print(f"💾 统计汇总已保存至: {stats_report_path}")


def main():
    """主函数"""
    print("LSS分析结果综合校验工具")
    print("=" * 30)
    
    # 创建校验器实例
    validator = LSS_Validator()
    
    print("\n选择校验模式:")
    print("1. 全面校验（详细，包含所有文件质量检查）")
    print("2. 快速概览（快速，仅检查文件存在性和基本信息）")
    print("3. 自定义配置")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🔍 开始全面校验...")
        results, stats = validator.comprehensive_check()
        
    elif choice == "2":
        print("\n🚀 开始快速概览...")
        validator.quick_check()
        
    elif choice == "3":
        print("\n⚙️  自定义配置:")
        
        # 自定义路径
        custom_path = input(f"输出路径 (默认: {OUTPUT_ROOT}): ").strip()
        if custom_path:
            validator.output_root = Path(custom_path)
        
        # 自定义被试范围
        subject_range = input("被试范围 (格式: 开始-结束, 如 1-5, 默认: 24-24): ").strip()
        if subject_range and "-" in subject_range:
            start, end = map(int, subject_range.split("-"))
            validator.subjects = [f"sub-{i:02d}" for i in range(start, end + 1)]
        
        # 自定义runs
        runs_input = input("Run编号 (用逗号分隔, 如 3,4,5, 默认: 3): ").strip()
        if runs_input:
            validator.runs = [int(r.strip()) for r in runs_input.split(",")]
        
        # 选择模式
        mode = input("校验模式 (1=全面, 2=快速): ").strip()
        
        if mode == "1":
            print("\n🔍 开始自定义全面校验...")
            results, stats = validator.comprehensive_check()
        else:
            print("\n🚀 开始自定义快速概览...")
            validator.quick_check()
    
    else:
        print("\n无效选择，运行默认全面校验...")
        results, stats = validator.comprehensive_check()
    
    print("\n✅ 校验完成！")


if __name__ == "__main__":
    main()