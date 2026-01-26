import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from tqdm import tqdm

# ====== 辅助函数 ======

def load_gifti_flatten(path):
    g = nib.load(path)
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32, copy=False)

def safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa, bb = a[m], b[m]
    # 方差为0直接返回 NaN
    if aa.std() == 0 or bb.std() == 0:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return np.nan
    return float(np.dot(aa, bb) / denom)

def resolve_beta_path(lss_root, task, subject, run, rel_file):
    # 兼容大小写路径
    paths_to_try = [
        os.path.join(lss_root, task, subject, rel_file),
        os.path.join(lss_root, task, subject, f"run-{run}", rel_file),
        os.path.join(lss_root, task.lower(), subject, rel_file),
        os.path.join(lss_root, task.lower(), subject, f"run-{run}", rel_file)
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
    return None

# ====== 主函数 ======

def subject_to_mean_check(aligned_csv, lss_root, task, min_frac=0.9, max_subjects=None):
    print(f"\n{'='*60}")
    print(f"开始执行 Subject-to-Mean 检查 | Task: {task} | 模式: LOO(leave-one-out)")

    df = pd.read_csv(aligned_csv)
    df = df[df["task"] == task].copy()

    subs = sorted(df["subject"].unique().tolist())
    total_subs_in_csv = len(subs)

    if max_subjects:
        subs = subs[:max_subjects]
        print(f"配置限制: 仅分析前 {len(subs)} 名被试 (总数: {total_subs_in_csv})")
    else:
        print(f"分析所有被试: {len(subs)} 名")

    # 1) 选公共刺激
    cnt = Counter(df["stimulus_content"].tolist())
    need = int(np.ceil(min_frac * len(subs)))
    common = [k for k, v in cnt.items() if v >= need]

    if not common:
        print(f"❌ 错误: 未找到满足 {min_frac*100:.1f}% 覆盖率的公共刺激。")
        print(f"   Top 5 出现频率最高的刺激: {cnt.most_common(5)}")
        raise RuntimeError("No common stimulus_content found.")

    stim = common[0]
    print(f"✅ 选定公共刺激: {stim}")
    print(f"   覆盖率: {cnt[stim]}/{len(subs)} 被试 (阈值需 ≥ {need})")

    # 2) 加载数据（记录每个被试对应的 beta 路径，方便排查重复）
    betas = []
    used_subs = []
    used_paths = []

    print("\n>>> 正在加载 Beta 文件...")
    for sub in tqdm(subs, desc="Loading"):
        row = df[(df["subject"] == sub) & (df["stimulus_content"] == stim)]
        if row.empty:
            continue
        row = row.iloc[0]

        p = resolve_beta_path(lss_root, task, sub, row["run"], row["file"])
        if not p:
            continue

        data = load_gifti_flatten(p)

        # 简单数据有效性提示（不终止）
        if np.isnan(data).all():
            print(f"⚠️ 警告: 被试 {sub} 数据全为 NaN: {p}")
            continue

        if np.all(data == 0):
            print(f"⚠️ 警告: 被试 {sub} 数据全为 0: {p}")
            # 仍然允许进入（你也可以选择 continue）
            # continue

        betas.append(data)
        used_subs.append(sub)
        used_paths.append(p)

    if not betas:
        raise RuntimeError("未能加载任何有效数据文件！")

    # 3) 计算群组平均图统计（这里先做全体均值用于统计展示）
    betas = np.stack(betas, axis=0)  # [N, V]
    N, V = betas.shape

    mean_beta_all = betas.mean(axis=0)

    print(f"\n>>> 群组平均图 (Group Mean Map) 统计 (All-sub mean, 仅用于统计展示):")
    print(f"   - Shape: {mean_beta_all.shape}")
    print(f"   - Min  Value: {np.min(mean_beta_all):.6f}")
    print(f"   - Max  Value: {np.max(mean_beta_all):.6f}")
    print(f"   - Mean Value: {np.mean(mean_beta_all):.6f}")
    print(f"   - Non-zero vertices: {np.count_nonzero(mean_beta_all)}")

    if np.all(mean_beta_all == 0):
        print("❌ 严重警告: 群组平均图全为 0！可能所有被试数据无效或完全相互抵消。")

    # 4) LOO 相关：corr(sub_i, mean(all_except_i))
    print("\n>>> 正在计算 Subject-to-Mean 相关性 (LOO, leave-one-out)...")
    sum_beta = betas.sum(axis=0)

    rs = []
    for i in range(N):
        loo_mean = (sum_beta - betas[i]) / (N - 1)
        rs.append(safe_corr(betas[i], loo_mean))
    rs = np.array(rs, dtype=float)

    # 5) 汇总报告
    print(f"\n{'='*30} 最终结果 (LOO) {'='*30}")
    print(f"有效被试数 (N): {len(used_subs)}")

    mean_r = np.nanmean(rs)
    std_r = np.nanstd(rs)

    print(f"相关系数统计 (Pearson r):")
    print(f"  ● Mean   : {mean_r:.4f}")
    print(f"  ● Std Dev: {std_r:.4f}")
    print(f"  ● Median : {np.nanmedian(rs):.4f}")
    print(f"  ● Min    : {np.nanmin(rs):.4f}")
    print(f"  ● Max    : {np.nanmax(rs):.4f}")

    # DataFrame 输出
    res_df = pd.DataFrame({
        'subject': used_subs,
        'r_group_loo': rs,
        'beta_path': used_paths
    }).sort_values('r_group_loo')

    print(f"\n📉 表现最差的 5 个被试 (可能需要剔除):")
    print(res_df.head(5)[["subject", "r_group_loo"]].to_string(index=False))

    print(f"\n📈 表现最好的 5 个被试 (检查是否存在路径重复/异常高相关):")
    print(res_df.tail(5)[["subject", "r_group_loo"]].to_string(index=False))

    # 6) 额外诊断：检查 beta_path 是否重复（重复意味着多个被试读取了同一个文件）
    print("\n>>> 额外诊断: 检查 beta_path 是否存在重复...")
    dup_paths = res_df["beta_path"].duplicated(keep=False)
    if dup_paths.any():
        dup_df = res_df[dup_paths].sort_values("beta_path")
        print("⚠️ 发现重复路径（多个被试可能读到同一个 beta 文件）:")
        print(dup_df[["subject", "r_group_loo", "beta_path"]].to_string(index=False))
    else:
        print("✅ 未发现重复 beta_path。")

    # 7) 打印 r 最大被试的路径，便于你直接核对文件
    best_row = res_df.iloc[-1]
    print("\n>>> r 最大被试详情（用于核对文件是否异常/重复/软链接）:")
    print(f"   subject: {best_row['subject']}")
    print(f"   r_group_loo: {best_row['r_group_loo']:.4f}")
    print(f"   beta_path: {best_row['beta_path']}")

    print(f"{'='*60}\n")

    return used_subs, rs, res_df

# ====== 示例调用 ======
if __name__ == "__main__":
    subject_to_mean_check(
        aligned_csv="/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv",
        lss_root="/public/home/dingrui/fmri_analysis/zz_analysis/lss_results",
        task="EMO",
        min_frac=0.9
    )
