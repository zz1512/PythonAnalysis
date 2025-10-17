#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载常用的fMRI脑区mask文件
用于searchlight MVPA分析
"""

import os
import urllib.request
import numpy as np
from nilearn import datasets, image
from nilearn.image import new_img_like
import os
from nilearn.datasets import fetch_atlas_harvard_oxford
import nibabel as nib
from pathlib import Path

def create_language_masks():
    """
    创建语言处理相关脑区的mask文件
    使用Harvard-Oxford皮层图谱和MNI152模板
    """
    print("开始下载和创建语言处理脑区mask文件...")
    
    # 下载Harvard-Oxford皮层图谱
    ho_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_atlas.maps
    labels = ho_atlas.labels
    
    print(f"图谱标签: {labels[:10]}...")  # 显示前10个标签
    
    # 下载灰质层mask
    print("正在下载MNI152灰质层mask...")
    mni_template = datasets.load_mni152_template(resolution=2)
    gm_mask = datasets.load_mni152_gm_mask(resolution=2)
    
    # 保存灰质层mask
    gm_mask.to_filename('mni152_gm_mask.nii.gz')
    print("已创建: mni152_gm_mask.nii.gz")
    
    # 创建各个脑区的mask
    atlas_data = atlas_img.get_fdata()
    
    # 定义脑区映射 (基于Harvard-Oxford图谱的标签索引)
    brain_regions = {
        'broca_area': [6, 7],  # 下额回 (Inferior Frontal Gyrus)
        'wernicke_area': [20, 21],  # 上颞回 (Superior Temporal Gyrus)
        'angular_gyrus': [30],  # 角回 (Angular Gyrus)
        'inferior_frontal_gyrus': [6, 7],  # 下额回
        'superior_temporal_gyrus': [20, 21],  # 上颞回
        'middle_temporal_gyrus': [19],  # 中颞回 (Middle Temporal Gyrus)
        'precuneus': [15],  # 楔前叶 (Precuneus)
        'posterior_parietal_cortex': [14, 30]  # 后顶叶皮层 (包括上顶叶小叶和角回)
    }
    
    for region_name, indices in brain_regions.items():
        print(f"正在创建 {region_name} mask...")
        
        # 创建二进制mask
        mask_data = np.zeros_like(atlas_data)
        for idx in indices:
            mask_data[atlas_data == idx] = 1
        
        # 创建NIfTI图像
        mask_img = new_img_like(atlas_img, mask_data)
        
        # 保存mask文件
        filename = f"{region_name}.nii.gz"
        mask_img.to_filename(filename)
        print(f"已创建: {filename}")
    
    # 创建综合语言网络mask (结合多个语言相关脑区)
    print("正在创建综合语言网络mask...")
    language_indices = [6, 7, 20, 21, 19, 30, 15]  # 结合多个语言相关脑区
    language_mask_data = np.zeros_like(atlas_data)
    for idx in language_indices:
        language_mask_data[atlas_data == idx] = 1
    
    language_network_img = new_img_like(atlas_img, language_mask_data)
    language_network_img.to_filename('language_network_mask.nii.gz')
    print("已创建: language_network_mask.nii.gz")
    
    # 创建空间认知网络mask (楔前叶 + 后顶叶)
    print("正在创建空间认知网络mask...")
    spatial_indices = [15, 14, 30]  # 楔前叶 + 后顶叶相关区域
    spatial_mask_data = np.zeros_like(atlas_data)
    for idx in spatial_indices:
        spatial_mask_data[atlas_data == idx] = 1
    
    spatial_network_img = new_img_like(atlas_img, spatial_mask_data)
    spatial_network_img.to_filename('spatial_network_mask.nii.gz')
    print("已创建: spatial_network_mask.nii.gz")
    
    # 创建灰质层mask
    print("正在创建灰质层mask...")
    # 使用MNI152灰质概率图创建二值化灰质mask
    gm_prob = datasets.load_mni152_gm_template(resolution=2)
    
    # 阈值化灰质概率图 (概率 > 0.2)
    gm_data = gm_prob.get_fdata()
    gm_binary_data = (gm_data > 0.2).astype(int)
    gm_binary_img = new_img_like(gm_prob, gm_binary_data)
    gm_binary_img.to_filename('gray_matter_mask.nii.gz')
    print("已创建: gray_matter_mask.nii.gz")
    
    # 创建皮层灰质mask (结合Harvard-Oxford皮层图谱)
    print("正在创建皮层灰质mask...")
    # 创建皮层mask (排除皮下结构)
    cortical_data = np.zeros_like(atlas_data)
    # 包含所有皮层区域 (索引1-48为皮层区域)
    for idx in range(1, 49):
        cortical_data[atlas_data == idx] = 1
    
    # 将灰质mask重采样到与atlas相同的空间
    gm_resampled = image.resample_to_img(gm_binary_img, atlas_img, interpolation='nearest')
    gm_resampled_data = gm_resampled.get_fdata()
    
    # 与灰质mask相交
    cortical_gm_data = cortical_data * gm_resampled_data
    cortical_gm_img = image.new_img_like(atlas_img, cortical_gm_data)
    cortical_gm_img.to_filename('cortical_gray_matter_mask.nii.gz')
    print("已创建: cortical_gray_matter_mask.nii.gz")
    
    print("\n所有mask文件创建完成!")
    print("文件列表:")
    for file in Path('.').glob('*.nii.gz'):
        print(f"  - {file.name}")

def main():
    """
    主函数
    """
    print("=== fMRI脑区Mask下载器 ===")
    print("正在创建常用的语言处理和空间认知脑区mask...")
    
    try:
        create_language_masks()
        print("\n✅ 所有mask文件下载和创建完成!")
        print("这些mask文件可用于searchlight MVPA分析")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查网络连接和nilearn库是否正确安装")

if __name__ == "__main__":
    main()