#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_mask.py
功能 - 根据已有图谱制作Mask：读取 Tian Subcortex 图谱，将其转化为二值化 Mask (0和1)，用于 GLM 分析
"""

import os
from pathlib import Path
from nilearn.image import load_img, math_img

# ================= 配置区 =================
# 这里填你截图中存放 Mask 的目录
ATLAS_DIR = Path("/public/home/dingrui/tools/masks_atlas")

# 这是你截图中那个对应的 3D NIfTI 文件名 (不要用 dscalar, 用这个对应的 nii.gz)
SOURCE_ATLAS = "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"

# 输出的 Mask 文件名
OUTPUT_MASK = "Tian_Subcortex_S2_3T_Binary_Mask.nii.gz"
# ==========================================

def make_binary_mask():
    src_path = ATLAS_DIR / SOURCE_ATLAS
    out_path = ATLAS_DIR / OUTPUT_MASK
    
    print(f"正在读取图谱: {src_path}")
    
    if not src_path.exists():
        print(f"错误: 找不到文件 {src_path}")
        print("请检查文件名是否与截图中的完全一致 (注意 2009cAsym 那部分)")
        return

    # 1. 加载图谱
    atlas_img = load_img(str(src_path))
    
    # 2. 二值化处理
    # 逻辑: 原图中值大于0的地方设为1，等于0的地方保持0
    print("正在进行二值化处理...")
    mask_img = math_img('img > 0', img=atlas_img)
    
    # 3. 保存
    mask_img.to_filename(str(out_path))
    print(f"成功! 二值化 Mask 已保存至: {out_path}")
    print(f"请在 glm_config.py 中将 MNI_MASK_PATH 设置为该路径。")

if __name__ == "__main__":
    make_binary_mask()