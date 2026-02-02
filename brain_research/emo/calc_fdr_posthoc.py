import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.multitest import multipletests

# ================= 配置区 =================
# 输入文件路径 (替换成你刚才截图里的那个文件路径)
INPUT_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out/parcel_joint_analysis_dev_models_perm_fwer__surface_L__all.csv"

# 输出文件路径 (自动加个后缀)
input_path = Path(INPUT_CSV)
OUTPUT_CSV = input_path.parent / f"{input_path.stem}_with_FDR.csv"

# FDR 校正的阈值 (通常为 0.05，用于标记 True/False，但我们会保留具体 Q值)
ALPHA = 0.05


# =========================================

def apply_fdr_by_group(df):
    """
    按模型分组计算 FDR
    这样做是为了避免 M_nn 和 M_div 的差结果拖累 M_conv 的好结果
    """
    # 1. 检查是否有 p_perm 列
    if 'p_perm' not in df.columns:
        # 兼容性处理：之前的脚本可能叫 p_perm_one_tailed
        if 'p_perm_one_tailed' in df.columns:
            p_col = 'p_perm_one_tailed'
        else:
            raise ValueError("找不到 P 值列 (p_perm 或 p_perm_one_tailed)")
    else:
        p_col = 'p_perm'

    # 2. 定义处理函数
    def _fdr_core(group_df):
        p_vals = group_df[p_col].values

        # 处理 NaN (以防万一)
        clean_p = np.nan_to_num(p_vals, nan=1.0)

        # 执行 Benjamini-Hochberg FDR 校正
        # reject: 是否拒绝原假设 (True/False)
        # p_corrected: 校正后的 P 值 (也叫 Q-value)
        reject, p_corrected, _, _ = multipletests(clean_p, alpha=ALPHA, method='fdr_bh')

        group_df['p_fdr'] = p_corrected
        group_df['is_significant_fdr'] = reject
        return group_df

    # 3. 按 model 列分组应用
    print(f"正在对以下模型分别进行 FDR 校正: {df['model'].unique()}")
    df_result = df.groupby('model', group_keys=False).apply(_fdr_core)

    return df_result


def main():
    if not input_path.exists():
        print(f"错误: 找不到文件 {INPUT_CSV}")
        return

    print(f"读取文件: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 计算 FDR
    df_fdr = apply_fdr_by_group(df)

    # 保存
    df_fdr.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ FDR 计算完成！")
    print(f"结果已保存至: {OUTPUT_CSV}")

    # --- 简报 ---
    print("\n" + "=" * 40)
    print("FDR 校正结果简报 (Q < 0.05)")
    print("=" * 40)

    for model in df_fdr['model'].unique():
        sub_df = df_fdr[df_fdr['model'] == model]
        n_sig = sub_df['is_significant_fdr'].sum()
        min_q = sub_df['p_fdr'].min()
        print(f"模型: {model}")
        print(f"  - 显著脑区数: {n_sig} / {len(sub_df)}")
        print(f"  - 最小 Q-value: {min_q:.6f}")

        if n_sig > 0:
            top_parcels = sub_df[sub_df['is_significant_fdr']].sort_values('p_fdr').head(5)
            print(f"  - Top 5 显著脑区 ID: {top_parcels['parcel_id'].tolist()}")
        print("-" * 20)


if __name__ == "__main__":
    main()