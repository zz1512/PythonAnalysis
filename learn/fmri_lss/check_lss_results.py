import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm  # 进度条工具

# 路径配置
OUTPUT_ROOT = Path(r"../../learn_LSS")
SUBJECTS = [f"sub-{i:02d}" for i in range(24, 25)]  # 假设你有28个被试
RUNS = [3]  # 根据你的配置
TRIALS_PER_RUN = 70  # 每个run有70个trial


def check_beta_quality(beta_path):
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

        return quality_metrics

    except Exception as e:
        return {
            "file_path": str(beta_path),
            "error": str(e),
            "is_valid": False,
            "has_nan_issue": True,
            "has_zero_issue": True
        }


def batch_check_subject_run(subject, run):
    """批量检查单个被试单个run的所有beta文件"""
    run_dir = OUTPUT_ROOT / subject / f"run-{run}_LSS"

    if not run_dir.exists():
        return None

    print(f"\n🔍 检查 {subject} | run-{run}")
    print("-" * 50)

    # 查找所有beta文件
    beta_files = []
    for trial in range(1, TRIALS_PER_RUN + 1):
        beta_path = run_dir / f"beta_trial_{trial:03d}.nii.gz"
        if beta_path.exists():
            beta_files.append(beta_path)
        else:
            print(f"  ⚠️  缺失: beta_trial_{trial:03d}.nii.gz")

    if not beta_files:
        print("  ❌ 未找到任何beta文件")
        return None

    # 批量检查质量
    quality_results = []
    for beta_path in tqdm(beta_files, desc=f"检查 {subject}-run{run}", leave=False):
        result = check_beta_quality(beta_path)
        quality_results.append(result)

    return quality_results


def comprehensive_batch_quality_check():
    """全面批量质量检查"""
    print("=" * 60)
    print("LSS Beta文件批量质量检查")
    print("=" * 60)

    all_results = []
    summary_stats = {
        "total_subjects": 0,
        "total_runs": 0,
        "total_beta_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "nan_issue_files": 0,
        "zero_issue_files": 0,
        "missing_files": 0
    }

    for subject in SUBJECTS:
        subject_valid_files = 0
        subject_total_files = 0

        for run in RUNS:
            # 检查该run
            run_results = batch_check_subject_run(subject, run)

            if run_results:
                summary_stats["total_runs"] += 1

                for result in run_results:
                    summary_stats["total_beta_files"] += 1
                    subject_total_files += 1

                    if result.get("is_valid", False):
                        summary_stats["valid_files"] += 1
                        subject_valid_files += 1
                    else:
                        summary_stats["invalid_files"] += 1

                    if result.get("has_nan_issue", False):
                        summary_stats["nan_issue_files"] += 1

                    if result.get("has_zero_issue", False):
                        summary_stats["zero_issue_files"] += 1

                    # 添加识别信息
                    result["subject"] = subject
                    result["run"] = run
                    all_results.append(result)

            # 计算该run的缺失文件
            run_dir = OUTPUT_ROOT / subject / f"run-{run}_LSS"
            if run_dir.exists():
                expected_files = TRIALS_PER_RUN
                actual_files = len(list(run_dir.glob("beta_trial_*.nii.gz")))
                missing = expected_files - actual_files
                summary_stats["missing_files"] += max(0, missing)

        if subject_total_files > 0:
            valid_ratio = subject_valid_files / subject_total_files
            status = "✅" if valid_ratio > 0.9 else "⚠️" if valid_ratio > 0.7 else "❌"
            print(f"  {status} {subject}: {subject_valid_files}/{subject_total_files} 有效文件 ({valid_ratio:.1%})")
            summary_stats["total_subjects"] += 1

    # 生成详细报告
    generate_detailed_report(all_results, summary_stats)

    return all_results, summary_stats


def generate_detailed_report(all_results, summary_stats):
    """生成详细质量报告"""
    print("\n" + "=" * 60)
    print("📊 批量质量检查报告")
    print("=" * 60)

    # 总体统计
    print(f"\n📈 总体统计:")
    print(f"  被试数量: {summary_stats['total_subjects']}")
    print(f"  Run数量: {summary_stats['total_runs']}")
    print(f"  Beta文件总数: {summary_stats['total_beta_files']}")
    print(
        f"  有效文件: {summary_stats['valid_files']} ({summary_stats['valid_files'] / max(1, summary_stats['total_beta_files']):.1%})")
    print(
        f"  无效文件: {summary_stats['invalid_files']} ({summary_stats['invalid_files'] / max(1, summary_stats['total_beta_files']):.1%})")
    print(f"  NaN问题文件: {summary_stats['nan_issue_files']}")
    print(f"  零值问题文件: {summary_stats['zero_issue_files']}")
    print(f"  缺失文件: {summary_stats['missing_files']}")

    # 如果有无效文件，显示详细信息
    invalid_results = [r for r in all_results if not r.get("is_valid", True)]
    if invalid_results:
        print(f"\n⚠️  发现 {len(invalid_results)} 个有问题文件:")
        for result in invalid_results[:10]:  # 只显示前10个
            print(f"  ❌ {result['subject']}-run{result['run']}: {Path(result['file_path']).name}")
            if 'error' in result:
                print(f"     错误: {result['error']}")
            else:
                issues = []
                if result.get('has_nan_issue'): issues.append("NaN过多")
                if result.get('has_zero_issue'): issues.append("零值过多")
                print(f"     问题: {', '.join(issues)}")

        if len(invalid_results) > 10:
            print(f"  ... 还有 {len(invalid_results) - 10} 个有问题文件")

    # 保存详细结果到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = OUTPUT_ROOT / "beta_quality_report.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 详细报告已保存至: {output_file}")

        # 保存汇总统计
        summary_file = OUTPUT_ROOT / "beta_quality_summary.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"💾 汇总统计已保存至: {summary_file}")


def quick_quality_overview():
    """快速质量概览（不检查每个文件）"""
    print("🚀 快速质量概览")
    print("=" * 40)

    for subject in SUBJECTS:
        print(f"\n{subject}:")
        for run in RUNS:
            run_dir = OUTPUT_ROOT / subject / f"run-{run}_LSS"
            if run_dir.exists():
                beta_files = list(run_dir.glob("beta_trial_*.nii.gz"))
                print(f"  run-{run}: {len(beta_files)}/{TRIALS_PER_RUN} 个文件")

                # 快速检查第一个文件
                if beta_files:
                    first_file = beta_files[0]
                    result = check_beta_quality(first_file)
                    if result.get("is_valid", False):
                        print(
                            f"    ✅ 示例文件正常 (范围: [{result['data_range_min']:.3f}, {result['data_range_max']:.3f}])")
                    else:
                        print(f"    ❌ 示例文件有问题")
            else:
                print(f"  run-{run}: ❌ 目录不存在")


# 运行批量检查
if __name__ == "__main__":
    # 安装tqdm（如果还没有安装）
    # pip install tqdm

    print("选择检查模式:")
    print("1. 全面批量检查（详细，但较慢）")
    print("2. 快速概览（快速，但信息有限）")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        all_results, summary_stats = comprehensive_batch_quality_check()
    elif choice == "2":
        quick_quality_overview()
    else:
        print("无效选择，运行全面检查...")
        all_results, summary_stats = comprehensive_batch_quality_check()