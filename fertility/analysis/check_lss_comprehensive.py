#!/usr/bin/env python3
"""
check_lss_final.py
LSS 分析结果深度校验工具 (适配 RSA 专用流程)

主要升级:
1. 基于 trial_info.csv 驱动的文件校验 (确保元数据与文件一一对应)
2. 适配动态文件名 beta_trial-{idx}_{unique_label}.nii.gz
3. 增加 "空脑检测" (全0值) 和 "数值异常检测" (标准化后的Beta值不应过大)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import logging

# ===========================
# 1. 核心配置 (适配你的实验)
# ===========================
OUTPUT_ROOT = Path("C:/python_metaphor/lss_betas_final")
# 你的被试列表 (根据实际情况修改范围)
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
# LSS 分析涉及的 Run
RUNS = [1, 2, 5, 6]
# 预期每个 Run 的 Trial 数量 (前测/后测: 100个词)
EXPECTED_TRIALS = 100

# 质量控制阈值
QC_THRESHOLDS = {
    'min_valid_voxels': 1000,  # 至少有多少个体素是非0的
    'max_abs_value': 100.0,  # 标准化后的 Beta 值不应超过这个数 (超过通常是伪影)
    'nan_tolerance': 0.0,  # 不允许有 NaN
}


class LSS_Validator:
    def __init__(self, output_root=OUTPUT_ROOT):
        self.output_root = Path(output_root)
        self.reset_stats()

        # 日志配置
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger("LSS_Validator")

    def reset_stats(self):
        self.stats = {
            'subjects_checked': 0,
            'subjects_valid': 0,
            'total_files_expected': 0,
            'total_files_found': 0,
            'total_files_valid': 0,
            'errors': []
        }
        self.report_data = []

    def load_nifti_safe(self, path):
        """安全加载 NIfTI 并返回数据，失败返回 None"""
        try:
            img = nib.load(str(path))
            data = img.get_fdata()
            img.uncache()  # 释放内存
            return data
        except Exception as e:
            return None

    def validate_beta_data(self, data, file_name):
        """科学性数值检查"""
        issues = []
        is_valid = True

        # 1. 检查 NaN
        if np.isnan(data).any():
            issues.append("包含 NaN")
            is_valid = False

        # 2. 检查全零 (建模失败或 Mask 错误)
        non_zero_count = np.count_nonzero(data)
        if non_zero_count < QC_THRESHOLDS['min_valid_voxels']:
            issues.append(f"有效体素过少 ({non_zero_count}) - 可能是空脑")
            is_valid = False

        # 3. 检查极端值 (标准化后的 Beta 通常在 -10 到 10 之间)
        # 如果出现几百几千的值，通常是除零错误或运动伪影
        max_val = np.nanmax(np.abs(data))
        if max_val > QC_THRESHOLDS['max_abs_value']:
            issues.append(f"数值异常过大 (Max Abs: {max_val:.1f})")
            # 标记为警告但不一定无效，视情况而定

        return is_valid, issues, {
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'std': float(np.nanstd(data)),
            'non_zero_voxels': int(non_zero_count)
        }

    def check_run(self, sub, run, mode='comprehensive'):
        run_dir = self.output_root / sub / f"run-{run}"
        csv_path = run_dir / "trial_info.csv"

        run_report = {
            'subject': sub,
            'run': run,
            'status': 'PASS',
            'issues': [],
            'valid_ratio': 0.0
        }

        # 1. 检查目录和 CSV 是否存在
        if not run_dir.exists() or not csv_path.exists():
            run_report['status'] = 'MISSING'
            run_report['issues'].append("目录或 trial_info.csv 缺失")
            return run_report

        # 2. 读取 CSV 并验证完整性
        try:
            df = pd.read_csv(csv_path)
            # 检查行数 (Trial 数)
            if len(df) != EXPECTED_TRIALS:
                msg = f"Trial 数量不匹配 (预期 {EXPECTED_TRIALS}, 实际 {len(df)})"
                run_report['issues'].append(msg)
                run_report['status'] = 'WARNING'

            # 检查关键列
            required_cols = ['beta_file', 'unique_label', 'trial_index']
            if not all(col in df.columns for col in required_cols):
                run_report['status'] = 'FAIL'
                run_report['issues'].append("CSV 缺少关键列 (beta_file 等)")
                return run_report

        except Exception as e:
            run_report['status'] = 'FAIL'
            run_report['issues'].append(f"CSV 读取失败: {e}")
            return run_report

        # 3. 文件一致性检查 (CSV vs Disk)
        # 我们的 LSS 代码会在 CSV 记录生成的文件名，以此为准
        expected_files = df['beta_file'].tolist()
        self.stats['total_files_expected'] += len(expected_files)

        valid_files_count = 0
        pbar = tqdm(expected_files, desc=f"Checking {sub} Run {run}", leave=False)

        for fname in pbar:
            # 处理可能的路径差异 (CSV里存的可能是绝对路径或相对路径)
            fpath = run_dir / Path(fname).name

            if not fpath.exists():
                run_report['issues'].append(f"文件丢失: {fname}")
                continue

            self.stats['total_files_found'] += 1

            # 4. 深度质量检查 (Comprehensive 模式)
            if mode == 'comprehensive':
                data = self.load_nifti_safe(fpath)
                if data is None:
                    run_report['issues'].append(f"文件损坏: {fname}")
                    continue

                is_valid, data_issues, metrics = self.validate_beta_data(data, fname)

                if not is_valid:
                    run_report['issues'].append(f"{fname}: {'; '.join(data_issues)}")
                else:
                    valid_files_count += 1
            else:
                # Quick 模式只检查存在性
                valid_files_count += 1

        # 汇总 Run 结果
        total = len(expected_files)
        run_report['valid_ratio'] = valid_files_count / total if total > 0 else 0
        self.stats['total_files_valid'] += valid_files_count

        if run_report['valid_ratio'] < 1.0:
            if run_report['status'] == 'PASS':
                run_report['status'] = 'FAIL' if run_report['valid_ratio'] < 0.9 else 'WARNING'

        return run_report

    def run(self, mode='comprehensive'):
        print(f"🚀 开始 LSS 校验 (模式: {mode})")
        print(f"📂 根目录: {self.output_root}")
        print(f"📋 预期: {len(SUBJECTS)} 被试 x {len(RUNS)} Runs x {EXPECTED_TRIALS} Trials")
        print("-" * 60)

        for sub in SUBJECTS:
            self.stats['subjects_checked'] += 1
            sub_issues_count = 0

            for run in RUNS:
                res = self.check_run(sub, run, mode)
                self.report_data.append(res)

                # 打印非 PASS 的结果
                if res['status'] != 'PASS':
                    sub_issues_count += 1
                    print(f"\n❌ [{sub} Run {run}] {res['status']}")
                    for issue in res['issues'][:5]:  # 只显示前5个错误
                        print(f"   - {issue}")
                    if len(res['issues']) > 5:
                        print(f"   - ... (还有 {len(res['issues']) - 5} 个问题)")

            if sub_issues_count == 0:
                self.stats['subjects_valid'] += 1
                print(f"✅ {sub} ... 通过")

        self.generate_summary()

    def generate_summary(self):
        print("\n" + "=" * 60)
        print("📊 最终校验报告")
        print("=" * 60)

        success_rate = self.stats['subjects_valid'] / self.stats['subjects_checked'] if self.stats[
                                                                                            'subjects_checked'] > 0 else 0
        print(f"被试通过率: {self.stats['subjects_valid']}/{self.stats['subjects_checked']} ({success_rate:.1%})")

        file_rate = self.stats['total_files_valid'] / self.stats['total_files_expected'] if self.stats[
                                                                                                'total_files_expected'] > 0 else 0
        print(f"文件完整性: {self.stats['total_files_valid']}/{self.stats['total_files_expected']} ({file_rate:.1%})")

        print("\n⚠️  主要问题汇总:")
        all_issues = [issue for r in self.report_data for issue in r['issues']]
        if not all_issues:
            print("  无显著问题。完美！")
        else:
            # 简单统计最常见的问题
            from collections import Counter
            # 简化错误信息以便统计 (去掉具体文件名)
            simple_issues = []
            for i in all_issues:
                if "数值异常" in i:
                    simple_issues.append("数值异常 (Spikes)")
                elif "有效体素过少" in i:
                    simple_issues.append("空脑/Mask错误")
                elif "文件丢失" in i:
                    simple_issues.append("文件丢失")
                elif "Trial 数量" in i:
                    simple_issues.append("Trial 数量不匹配")
                else:
                    simple_issues.append("其他错误")

            for k, v in Counter(simple_issues).most_common():
                print(f"  - {k}: {v} 次")

        # 保存 CSV 报告
        df = pd.DataFrame(self.report_data)
        out_csv = self.output_root / "validation_summary_final.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n📄 详细报告已保存: {out_csv}")


if __name__ == "__main__":
    # 使用方法:
    # 1. 修改顶部的 SUBJECTS 和 OUTPUT_ROOT
    # 2. 运行 python check_lss_final.py

    validator = LSS_Validator()

    # mode='quick' 只检查文件是否存在，速度快
    # mode='comprehensive' 读取每个 NIfTI 检查数值，速度慢但严谨
    validator.run(mode='comprehensive')