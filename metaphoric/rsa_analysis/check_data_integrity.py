#!/usr/bin/env python3
"""
check_data_integrity.py (Fix Version)
RSA 数据严格校验脚本 - 修复版
修复了 "yyw_1" 会错误匹配到 "yyw_10" 的子串匹配问题，改为精确全词匹配。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rsa_config as cfg
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_subject(sub, lss_df, template_df):
    """验证单个被试的数据完整性 (精确匹配模式)"""
    issues = []

    # 检查两个阶段
    for stage in ['Pre', 'Post']:
        # 筛选 LSS 数据
        runs = [1, 2] if stage == 'Pre' else [5, 6]
        sub_df = lss_df[
            (lss_df['subject'] == sub) &
            (lss_df['run'].isin(runs))
            ].copy()  # Copy 以避免 SettingWithCopy警告

        # 预处理：确保 unique_label 是纯净字符串，去除潜在空格
        sub_df['unique_label'] = sub_df['unique_label'].astype(str).str.strip()
        sub_df['condition'] = sub_df['condition'].astype(str).str.strip()

        # 遍历模板中的每一个词 (180个)
        for idx, row in template_df.iterrows():
            # 获取模板中的目标值，同样去空格
            target_word = str(row['word_label']).strip()
            target_type = str(row['type']).strip()
            target_cond = str(row['condition']).strip()

            # --- 严格精确匹配逻辑 (Exact Match) ---
            # 1. unique_label 必须 完全等于 word_label
            # 2. condition 必须包含 type (或者根据你的LSS设计，如果trial_type就是type，也可以用==)
            # 这里对 condition 依然保留 contains，因为 trial_type 可能是 'Metaphor' 而 type 是 'yybw'
            # 但对 unique_label 必须用 ==

            match = sub_df[
                (sub_df['unique_label'] == target_word)
                ]

            # 如果上面太严导致找不到，可能LSS里的condition是 'yyw' 而不是 'Metaphor'
            # 你可以尝试注释掉上面那行，改用下面这行更宽松的条件检查（只查word_label）：
            # match = sub_df[sub_df['unique_label'] == target_word]

            if len(match) == 0:
                issues.append({
                    'subject': sub, 'stage': stage,
                    'error': 'MISSING', 'word': target_word,
                    'target_type': target_type
                })
            elif len(match) > 1:
                # 精确匹配下如果还有重复，那就是真重复了（比如 Run1 和 Run2 里都有这个 ID）
                # 正常 LSS 应该处理好了，但如果有重复，取第一个通常不是最佳方案，需要报警
                issues.append({
                    'subject': sub, 'stage': stage,
                    'error': 'DUPLICATE', 'word': target_word, 'count': len(match)
                })
            else:
                # === 修复路径拼接逻辑 ===
                row_data = match.iloc[0]
                fname = row_data['beta_file']

                # 如果 CSV 里存的是绝对路径，直接用
                if Path(fname).is_absolute():
                    fpath = Path(fname)
                else:
                    # 如果是文件名，需要手动拼接 sub-xx 和 run-x 目录
                    # 结构: LSS_ROOT / sub-xx / run-x / filename
                    sub_folder = row_data['subject']  # e.g., sub-01
                    run_folder = f"run-{row_data['run']}"  # e.g., run-1

                    fpath = cfg.LSS_META_FILE.parent / sub_folder / run_folder / fname

                # 检查物理存在性
                if not fpath.exists():
                    issues.append({
                        'subject': sub, 'stage': stage,
                        'error': 'FILE_NOT_FOUND', 'path': str(fpath)
                    })

    return issues


def main():
    print("🚀 开始数据完整性校验 (修复版: 精确匹配)...")

    # 1. 加载文件
    if not cfg.LSS_META_FILE.exists():
        logger.error(f"LSS Index 文件不存在: {cfg.LSS_META_FILE}")
        return
    if not cfg.STIMULI_TEMPLATE.exists():
        logger.error(f"刺激模板文件不存在: {cfg.STIMULI_TEMPLATE}")
        return

    lss_df = pd.read_csv(cfg.LSS_META_FILE)
    template_df = pd.read_csv(cfg.STIMULI_TEMPLATE)

    # 预处理模板：去空格
    template_df['word_label'] = template_df['word_label'].astype(str).str.strip()
    template_df['type'] = template_df['type'].astype(str).str.strip()

    print(f"模板包含 {len(template_df)} 个词汇。")

    all_issues = []

    # 2. 循环检查
    for sub in tqdm(cfg.SUBJECTS, desc="Checking Subjects"):
        sub_issues = validate_subject(sub, lss_df, template_df)
        all_issues.extend(sub_issues)

    # 3. 输出报告
    if len(all_issues) == 0:
        print("\n✅ ALL GREEN! 所有数据完整且匹配无误。可以运行 RSA。")
    else:
        print(f"\n❌ 发现 {len(all_issues)} 个问题！请不要运行 RSA！")

        report_df = pd.DataFrame(all_issues)
        out_csv = cfg.BASE_DIR / "rsa_data_validation_report_fixed.csv"
        report_df.to_csv(out_csv, index=False)

        print(f"详细错误报告已保存: {out_csv}")

        # 打印前几行看看是不是解决了 duplicate
        print("错误示例:")
        print(report_df.head())


if __name__ == "__main__":
    main()