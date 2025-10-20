# feature_extraction.py
# 特征提取模块

from nilearn.maskers import NiftiMasker
from utils import log

def extract_roi_timeseries_optimized(functional_imgs, roi_mask_path, config, confounds=None):
    """优化的ROI时间序列提取"""
    try:
        # 使用NiftiMasker进行高效的特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            detrend=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )
        
        # 提取时间序列
        timeseries = masker.fit_transform(functional_imgs, confounds=confounds)
        
        return timeseries
        
    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        return None