# feature_extraction.py
# 特征提取模块

import numpy as np
from nilearn.maskers import NiftiMasker
from utils import log

def extract_roi_features(beta_images, roi_mask_path, config):
    """从beta图像中提取ROI特征"""
    try:
        # 使用NiftiMasker进行特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )
        
        # 提取特征
        X = masker.fit_transform(beta_images)
        
        return X
        
    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        return None