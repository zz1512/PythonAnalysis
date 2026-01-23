#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_beta_quality_random.py
功能：随机抽取 5 个左脑 Surface Beta 文件进行深度体检
"""

import os
import glob
import random
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from nilearn import surface

# ================= 配置区 =================
# 你的输出目录
RESULT_DIR = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"


# ==========================================

def check_surface_file(filepath, index):
    filename = os.path.basename(filepath)
    print(f"\n{'=' * 60}")
    print(f"[{index}/5] 正在检查文件: {filename}")
    print(f"路径: {filepath}")

    try:
        # 1. 加载数据
        # Surface 数据通常是一维数组
        data = surface.load_surf_data(filepath)

        # 2. 维度检查
        # fsLR 32k 模板通常是 32492 个顶点
        print(f"  - 形状 (Vertices): {data.shape}")

        # 3. 数值统计
        max_val = np.max(data)
        min_val = np.min(data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        print(f"  - Max:  {max_val:.4f}")
        print(f"  - Min:  {min_val:.4f}")
        print(f"  - Mean: {mean_val:.4f} (应该接近 0)")
        print(f"  - Std:  {std_val:.4f}")

        # 4. 零值检查 (Mask 外可能为0，但 Surface 通常全是有效值)
        zeros = np.sum(data == 0)
        total = data.size
        print(f"  - 零值数量: {zeros} ({zeros / total * 100:.2f}%)")

        # 5. 逻辑判断
        is_valid = True
        if np.all(data == 0):
            print("  ❌ 严重错误: 所有值均为 0 (计算失败)")
            is_valid = False
        elif np.isnan(np.sum(data)):
            print("  ❌ 严重错误: 包含 NaN (空值)")
            is_valid = False
        elif abs(max_val) > 100 or abs(min_val) > 100:
            print("  ⚠️ 警告: 数值异常巨大 (可能模型发散)")
        else:
            print("  ✅ 数值范围合理")

        # 6. 绘制直方图 (判断数据分布形态)
        if is_valid:
            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=100, color='#4c72b0', alpha=0.8)
            plt.title(f"Distribution: {filename}\nMean={mean_val:.3f}, Std={std_val:.3f}")
            plt.xlabel("Beta Value")
            plt.ylabel("Vertex Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"  ❌ 读取或校验失败: {e}")


def main():
    print(f"正在搜索目录: {RESULT_DIR} ...")

    # 1. 搜索所有左脑 Surface 文件 (*_L.gii)
    # 使用 recursive=True 遍历所有子文件夹
    all_files = glob.glob(f"{RESULT_DIR}/**/*_L.gii", recursive=True)

    total_files = len(all_files)
    print(f"共找到 {total_files} 个左脑 Beta 文件。")

    if total_files == 0:
        print("❌ 未找到任何文件，请检查路径或等待程序运行一会。")
        return

    # 2. 随机抽取 5 个
    sample_size = 5
    if total_files > sample_size:
        selected_files = random.sample(all_files, sample_size)
    else:
        selected_files = all_files
        print(f"文件不足 5 个，将全部检查。")

    # 3. 逐个检查
    for i, fpath in enumerate(selected_files):
        check_surface_file(fpath, i + 1)


if __name__ == "__main__":
    main()