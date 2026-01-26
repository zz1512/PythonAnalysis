#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_group_mean.py
功能：
1. 读取 aligned_csv 中指定任务和条件的 Beta 文件。
2. 计算 29 人 Group Mean Beta Map。
3. 绘制并保存到图片，用于肉眼核查激活模式是否符合生物学预期。
"""

import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区 =================
ALIGNED_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"
LSS_ROOT = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"
TASK = "EMO"
# 选取一个最常见的刺激内容进行平均
TARGET_CONTENT = "EMO_Passive_Neutral"  # 或者根据你的 csv 里的内容调整，比如 EMO_PassiveLook_Neutral...


# ==========================================

def resolve_path(row):
    """构建文件路径"""
    # 尝试构建路径，根据你的目录结构可能需要微调
    # 假设结构: LSS_ROOT / task / sub / file
    p = Path(LSS_ROOT) / row['task'].lower() / row['subject'] / row['file']
    if not p.exists():
        # 尝试加 run
        p = Path(LSS_ROOT) / row['task'].lower() / row['subject'] / f"run-{row['run']}" / row['file']
    return p


def main():
    print(f"读取索引: {ALIGNED_CSV}")
    df = pd.read_csv(ALIGNED_CSV)

    # 1. 筛选出特定条件的记录
    # 模糊匹配，只要包含 EMO 和 Neutral 的就行
    df_target = df[
        (df['task'] == TASK) &
        (df['stimulus_content'].str.contains('Neutral', case=False))
        ]

    if df_target.empty:
        print("未找到符合条件的记录，请检查筛选条件。")
        return

    # 去重：每个被试只取一张图（防止某人看多次权重过大）
    df_target = df_target.drop_duplicates(subset=['subject'])
    print(f"筛选出 {len(df_target)} 个被试的数据用于平均。")

    # 2. 加载数据
    betas = []
    valid_subs = []

    for _, row in tqdm(df_target.iterrows(), total=len(df_target)):
        p = resolve_path(row)
        if p.exists():
            try:
                gii = nib.load(str(p))
                data = gii.darrays[0].data
                betas.append(data)
                valid_subs.append(row['subject'])
            except:
                pass

    if not betas:
        print("没有成功加载任何数据。")
        return

    # 3. 计算平均
    # stack shape: (N_subs, N_vertices)
    beta_matrix = np.stack(betas, axis=0)
    mean_map = np.mean(beta_matrix, axis=0)

    print(f"计算完成。平均图范围: Min={mean_map.min():.4f}, Max={mean_map.max():.4f}")

    # 4. 绘图 (fsaverage 模板)
    print("正在绘图...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')  # 这里只是为了显示，网格可能不用对齐，Nilearn会自动插值

    # 注意：你的数据是 fsLR 32k，Nilearn 绘图默认是 fsaverage。
    # 直接画可能会报错顶点不匹配。
    # 既然我们只是看个大概，我们直接保存成 .gii 文件，建议你下载下来用 Workbench 或 Mango 看。
    # 但为了脚本能跑，我们可以尝试用 nilearn 的 plot_surf_stat_map (需要 mesh 对应)

    # === 方案 B: 简单的保存为 PNG (如果不匹配 mesh 会很难看，所以我们只保存数据) ===
    # 还是保存成 .gii 最稳妥，你可以下载到本地用软件看

    save_path = Path(f"group_mean_{TASK}_N{len(valid_subs)}.func.gii")

    # 创建 Gifti 对象
    da = nib.gifti.GiftiDataArray(mean_map.astype(np.float32), intent='NIFTI_INTENT_ESTIMATE')
    img = nib.gifti.GiftiImage(darrays=[da])
    nib.save(img, str(save_path))
    print(f"✅ 平均图已保存: {save_path}")
    print("👉 请将此文件下载到本地，使用 Connectome Workbench (wb_view) 查看。")
    print("👉 预期：Visual Cortex (脑后部) 应该有明显的正值 (红色)。")

    # === 尝试画一个简单的图 (如果顶点数恰好是 10242 或者是标准 fsLR) ===
    # 你的数据是 32k (32492 vertices)?
    if mean_map.shape[0] > 30000:
        print("提示: 数据似乎是 fsLR 32k 格式。")
        print("Python 直接绘图需要 fsLR 的 mesh 文件。")
        print("最快验证方法：看数值分布。")

        # 简单统计视觉区 (大致索引，不一定准，仅供参考)
        # fsLR 左脑视觉区大致在后面
        # 我们对比一下 前1000个点(通常是皮层) vs 后面的点
        print(f"前 10 个顶点值: {mean_map[:10]}")
        print(f"中间 10 个顶点值: {mean_map[15000:15010]}")


if __name__ == "__main__":
    main()