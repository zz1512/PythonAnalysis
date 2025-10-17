#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI统计图像分析脚本：显著性阈值处理与可视化

功能描述:
    对SPM生成的统计图像进行阈值处理、显著性检验和可视化展示。
    支持FDR多重比较校正和簇大小阈值过滤。

主要功能:
    1. 加载SPM统计图像（T值图）
    2. 应用FDR校正和簇大小阈值
    3. 生成正交切面可视化
    4. 导出显著激活簇表格

输入文件:
    - SPM统计图像：spmT_0001.nii.gz（YY vs KJ对比的T值图）

输出结果:
    - 阈值处理后的统计图像可视化
    - 显著激活簇的详细信息表格

统计参数:
    - 显著性水平：α = 0.05
    - 多重比较校正：FDR（False Discovery Rate）
    - 簇大小阈值：≥20个体素
    - 检验类型：双侧检验

注意事项:
    - 需要确保统计图像路径正确
    - MNI152模板用作解剖背景
    - 可根据需要调整阈值参数

作者: 研究团队
版本: 1.0
日期: 2024
"""

from nilearn import image, plotting, datasets
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
import pandas as pd

# === 配置参数 ===
stat_path = r"../metaphor/Activation/2nd_level/run3/yy_vs_kj/spmT_0001.nii.gz"  # SPM统计图像路径

# === 数据加载 ===
print(f"正在加载统计图像: {stat_path}")
fmri_img = image.load_img(stat_path)
fmri_data = fmri_img.get_fdata()
print(f"图像形状: {fmri_data.shape}")
print(f"数据范围: [{fmri_data.min():.3f}, {fmri_data.max():.3f}]")
# === 统计阈值处理 ===
# 加载MNI152标准模板作为解剖背景
print("\n正在加载MNI152模板...")
mni_t1 = datasets.load_mni152_template()

# 应用FDR多重比较校正和簇大小阈值
print("正在应用统计阈值处理...")
thresh_img, threshold = threshold_stats_img(
    stat_path, 
    alpha=0.05,              # 显著性水平
    height_control='fdr',    # FDR校正方法
    cluster_threshold=20,    # 最小簇大小（体素数）
    two_sided=True          # 双侧检验
)
print(f"FDR校正后的统计阈值: {threshold:.4f}")

# === 可视化展示 ===
print("生成统计图像可视化...")
display = plotting.plot_stat_map(
    thresh_img,              # 阈值处理后的统计图像
    bg_img=mni_t1,          # 解剖背景图像
    display_mode='ortho',    # 正交切面显示模式
    cut_coords=None,        # 自动选择切面坐标
    black_bg=True,          # 黑色背景
    dim=0.3,                # 背景图像透明度
    title="YY vs KJ: FDR校正后的激活图"
)
plotting.show()

# === 激活簇分析 ===
print("\n正在生成激活簇表格...")
try:
    # 获取显著激活簇的详细信息
    table = get_clusters_table(
        thresh_img, 
        stat_threshold=threshold, 
        cluster_threshold=20
    )
    
    print("\n=== 显著激活簇信息 ===")
    print(table)
    
    # 可选：保存簇表格到CSV文件
    save_table = False  # 设置为True以保存文件
    if save_table:
        output_file = "significant_clusters.csv"
        pd.DataFrame(table).to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n激活簇表格已保存至: {output_file}")
        
except Exception as e:
    print(f"生成激活簇表格时出错: {e}")
    print("可能原因：没有找到满足阈值条件的激活簇")

print("\n分析完成！")