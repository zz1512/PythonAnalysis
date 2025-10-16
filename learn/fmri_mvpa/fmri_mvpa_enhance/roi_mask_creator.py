#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Mask创建器 - 增强版
基于create_optimized_roi_masks.py进行优化
提供多种ROI创建方法和质量控制
"""

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.plotting import plot_roi, plot_stat_map
import os
import pandas as pd
import re
from scipy.spatial.distance import cdist
from scipy.ndimage import label
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

class OptimizedROIMaskCreator:
    """优化版ROI Mask创建器"""
    
    def __init__(self, config):
        self.config = config
        self.roi_config = config.roi
        
        # 创建输出目录
        Path(self.roi_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化Harvard-Oxford模板
        self.ho_template = None
        self._load_ho_template()
        
        # 存储创建的ROI信息
        self.roi_info = []
        
    def _load_ho_template(self):
        """加载Harvard-Oxford模板"""
        try:
            # 加载皮层和皮层下结构
            ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            
            self.ho_cort = ho_cort
            self.ho_sub = ho_sub
            
            logging.info("Harvard-Oxford模板加载成功")
            
        except Exception as e:
            logging.warning(f"Harvard-Oxford模板加载失败: {e}")
            self.ho_cort = None
            self.ho_sub = None
    
    def create_sphere_roi(self, center_coords: Tuple[float, float, float], 
                         radius: float = None, 
                         roi_name: str = None) -> nib.Nifti1Image:
        """创建球形ROI"""
        if radius is None:
            radius = self.roi_config.sphere_radius
            
        if roi_name is None:
            roi_name = f"sphere_{center_coords[0]}_{center_coords[1]}_{center_coords[2]}"
        
        try:
            # 使用nilearn创建球形ROI
            from nilearn.image import new_img_like
            from nilearn.datasets import load_mni152_template
            
            # 获取MNI模板作为参考
            template = load_mni152_template(resolution=2)
            
            # 创建球形mask
            sphere_mask = image.new_img_like(
                template,
                np.zeros(template.shape),
                affine=template.affine
            )
            
            # 计算球形区域
            coords = np.array(center_coords).reshape(1, -1)
            sphere_img = image.new_img_like(
                template,
                image.coord_transform(*coords.T, template.affine.T)[0]
            )
            
            # 使用nilearn的球形ROI功能
            from nilearn.maskers import NiftiSpheresMasker
            masker = NiftiSpheresMasker(
                seeds=[center_coords],
                radius=radius,
                allow_overlap=True,
                standardize=False
            )
            
            # 创建球形mask
            mask_data = np.zeros(template.shape)
            
            # 计算每个体素到中心的距离
            x, y, z = np.mgrid[0:template.shape[0], 0:template.shape[1], 0:template.shape[2]]
            coords_voxel = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            coords_world = image.coord_transform(coords_voxel[:, 0], coords_voxel[:, 1], coords_voxel[:, 2], template.affine)
            coords_world = np.column_stack(coords_world)
            
            # 计算距离
            distances = np.sqrt(np.sum((coords_world - np.array(center_coords))**2, axis=1))
            
            # 创建球形mask
            sphere_mask_1d = distances <= radius
            mask_data = sphere_mask_1d.reshape(template.shape)
            
            sphere_roi = image.new_img_like(template, mask_data.astype(int))
            
            # 质量检查
            n_voxels = np.sum(mask_data)
            if n_voxels < self.roi_config.min_voxels_sphere:
                logging.warning(f"球形ROI {roi_name} 体素数过少: {n_voxels}")
            
            # 记录ROI信息
            self.roi_info.append({
                'name': roi_name,
                'type': 'sphere',
                'center_coords': center_coords,
                'radius': radius,
                'n_voxels': int(n_voxels),
                'volume_mm3': float(n_voxels * np.prod(template.header.get_zooms()[:3]))
            })
            
            logging.info(f"创建球形ROI: {roi_name}, 中心: {center_coords}, 半径: {radius}mm, 体素数: {n_voxels}")
            
            return sphere_roi
            
        except Exception as e:
            logging.error(f"创建球形ROI失败: {e}")
            return None
    
    def create_anatomical_roi(self, region_name: str, 
                             hemisphere: str = 'both',
                             threshold: float = None) -> nib.Nifti1Image:
        """基于解剖模板创建ROI"""
        if threshold is None:
            threshold = self.roi_config.anatomical_threshold
            
        if self.ho_cort is None or self.ho_sub is None:
            logging.error("Harvard-Oxford模板未加载")
            return None
        
        try:
            # 搜索匹配的区域
            roi_img = None
            found_region = None
            
            # 在皮层模板中搜索
            for i, label in enumerate(self.ho_cort.labels):
                if region_name.lower() in label.lower():
                    atlas_img = load_img(self.ho_cort.maps)
                    roi_data = (atlas_img.get_fdata() == i) * 100
                    roi_img = image.new_img_like(atlas_img, roi_data)
                    found_region = label
                    break
            
            # 在皮层下模板中搜索
            if roi_img is None:
                for i, label in enumerate(self.ho_sub.labels):
                    if region_name.lower() in label.lower():
                        atlas_img = load_img(self.ho_sub.maps)
                        roi_data = (atlas_img.get_fdata() == i) * 100
                        roi_img = image.new_img_like(atlas_img, roi_data)
                        found_region = label
                        break
            
            if roi_img is None:
                logging.error(f"未找到匹配的解剖区域: {region_name}")
                return None
            
            # 应用阈值
            roi_img = image.threshold_img(roi_img, threshold=threshold)
            
            # 半球分离
            if hemisphere in ['left', 'right'] and self.roi_config.apply_hemisphere_separation:
                roi_img = self._apply_hemisphere_mask(roi_img, hemisphere)
            
            # 质量检查
            roi_data = roi_img.get_fdata()
            n_voxels = np.sum(roi_data > 0)
            
            if n_voxels < self.roi_config.min_voxels_anatomical:
                logging.warning(f"解剖ROI {region_name} 体素数过少: {n_voxels}")
            
            # 记录ROI信息
            self.roi_info.append({
                'name': f"{region_name}_{hemisphere}",
                'type': 'anatomical',
                'region_name': found_region,
                'hemisphere': hemisphere,
                'threshold': threshold,
                'n_voxels': int(n_voxels),
                'volume_mm3': float(n_voxels * np.prod(roi_img.header.get_zooms()[:3]))
            })
            
            logging.info(f"创建解剖ROI: {found_region} ({hemisphere}), 体素数: {n_voxels}")
            
            return roi_img
            
        except Exception as e:
            logging.error(f"创建解剖ROI失败: {e}")
            return None
    
    def create_activation_roi(self, stat_map: nib.Nifti1Image,
                             threshold: float = None,
                             cluster_size: int = None,
                             roi_name: str = None) -> nib.Nifti1Image:
        """基于激活图创建ROI"""
        if threshold is None:
            threshold = self.roi_config.activation_threshold
        if cluster_size is None:
            cluster_size = self.roi_config.cluster_size_threshold
        if roi_name is None:
            roi_name = f"activation_t{threshold}"
        
        try:
            # 应用阈值
            thresholded_map = image.threshold_img(stat_map, threshold=threshold)
            
            # 聚类分析
            from nilearn.reporting import get_clusters_table
            
            clusters_table = get_clusters_table(
                thresholded_map,
                stat_threshold=threshold,
                cluster_threshold=cluster_size
            )
            
            if len(clusters_table) == 0:
                logging.warning(f"未找到满足条件的激活cluster")
                return None
            
            # 创建二值化ROI
            roi_data = (thresholded_map.get_fdata() > threshold).astype(int)
            roi_img = image.new_img_like(thresholded_map, roi_data)
            
            # 质量检查
            n_voxels = np.sum(roi_data)
            
            if n_voxels < self.roi_config.min_voxels_activation:
                logging.warning(f"激活ROI {roi_name} 体素数过少: {n_voxels}")
            
            # 记录ROI信息
            self.roi_info.append({
                'name': roi_name,
                'type': 'activation',
                'threshold': threshold,
                'cluster_size': cluster_size,
                'n_clusters': len(clusters_table),
                'n_voxels': int(n_voxels),
                'volume_mm3': float(n_voxels * np.prod(roi_img.header.get_zooms()[:3]))
            })
            
            logging.info(f"创建激活ROI: {roi_name}, 阈值: {threshold}, 聚类数: {len(clusters_table)}, 体素数: {n_voxels}")
            
            return roi_img
            
        except Exception as e:
            logging.error(f"创建激活ROI失败: {e}")
            return None
    
    def _apply_hemisphere_mask(self, roi_img: nib.Nifti1Image, hemisphere: str) -> nib.Nifti1Image:
        """应用半球mask"""
        try:
            roi_data = roi_img.get_fdata()
            
            # 获取图像中心
            center_voxel = np.array(roi_img.shape) // 2
            
            if hemisphere == 'left':
                # 保留左半球（x < center）
                roi_data[center_voxel[0]:, :, :] = 0
            elif hemisphere == 'right':
                # 保留右半球（x > center）
                roi_data[:center_voxel[0], :, :] = 0
            
            return image.new_img_like(roi_img, roi_data)
            
        except Exception as e:
            logging.error(f"应用半球mask失败: {e}")
            return roi_img
    
    def create_multiple_sphere_rois(self, coordinates_list: List[Tuple[float, float, float]],
                                   names_list: List[str] = None,
                                   radius: float = None) -> Dict[str, nib.Nifti1Image]:
        """批量创建球形ROI"""
        if names_list is None:
            names_list = [f"sphere_{i+1}" for i in range(len(coordinates_list))]
        
        if len(names_list) != len(coordinates_list):
            raise ValueError("名称列表和坐标列表长度不匹配")
        
        rois = {}
        for coords, name in zip(coordinates_list, names_list):
            roi = self.create_sphere_roi(coords, radius, name)
            if roi is not None:
                rois[name] = roi
        
        return rois
    
    def save_roi(self, roi_img: nib.Nifti1Image, filename: str) -> str:
        """保存ROI到文件"""
        output_path = Path(self.roi_config.output_dir) / filename
        
        # 确保文件扩展名
        if not str(output_path).endswith('.nii.gz'):
            output_path = output_path.with_suffix('.nii.gz')
        
        roi_img.to_filename(str(output_path))
        logging.info(f"ROI已保存: {output_path}")
        
        return str(output_path)
    
    def save_all_rois(self, rois_dict: Dict[str, nib.Nifti1Image]) -> Dict[str, str]:
        """批量保存ROI"""
        saved_paths = {}
        for name, roi_img in rois_dict.items():
            filename = f"{name}_roi.nii.gz"
            path = self.save_roi(roi_img, filename)
            saved_paths[name] = path
        
        return saved_paths
    
    def create_roi_quality_report(self) -> pd.DataFrame:
        """创建ROI质量报告"""
        if not self.roi_info:
            logging.warning("没有ROI信息可用于生成报告")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.roi_info)
        
        # 添加质量评估
        df['quality_score'] = 0
        
        for idx, row in df.iterrows():
            score = 0
            
            # 体素数评估
            if row['n_voxels'] >= 100:
                score += 3
            elif row['n_voxels'] >= 50:
                score += 2
            elif row['n_voxels'] >= 20:
                score += 1
            
            # 体积评估
            if 'volume_mm3' in row:
                if self.roi_config.min_roi_volume <= row['volume_mm3'] <= self.roi_config.max_roi_volume:
                    score += 2
            
            df.at[idx, 'quality_score'] = score
        
        # 保存报告
        report_path = Path(self.roi_config.output_dir) / "roi_quality_report.csv"
        df.to_csv(report_path, index=False)
        logging.info(f"ROI质量报告已保存: {report_path}")
        
        return df
    
    def visualize_roi(self, roi_img: nib.Nifti1Image, 
                     title: str = "ROI Visualization",
                     save_path: str = None) -> None:
        """可视化ROI"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.roi_config.plot_figsize)
            
            # 绘制ROI
            plot_roi(
                roi_img,
                title=title,
                axes=axes.flatten(),
                colorbar=True,
                cmap=self.roi_config.colormap
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.roi_config.plot_dpi, bbox_inches='tight')
                logging.info(f"ROI可视化已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logging.error(f"ROI可视化失败: {e}")
    
    def get_roi_summary(self) -> Dict[str, Any]:
        """获取ROI创建总结"""
        if not self.roi_info:
            return {'total_rois': 0, 'types': {}, 'total_volume': 0}
        
        df = pd.DataFrame(self.roi_info)
        
        summary = {
            'total_rois': len(df),
            'types': df['type'].value_counts().to_dict(),
            'total_volume': df['volume_mm3'].sum() if 'volume_mm3' in df.columns else 0,
            'average_volume': df['volume_mm3'].mean() if 'volume_mm3' in df.columns else 0,
            'total_voxels': df['n_voxels'].sum(),
            'average_voxels': df['n_voxels'].mean()
        }
        
        return summary

def create_standard_rois(config, coordinates_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, str]:
    """创建标准ROI集合的便捷函数"""
    creator = OptimizedROIMaskCreator(config)
    
    rois = {}
    for name, coords in coordinates_dict.items():
        roi = creator.create_sphere_roi(coords, roi_name=name)
        if roi is not None:
            rois[name] = roi
    
    # 保存所有ROI
    saved_paths = creator.save_all_rois(rois)
    
    # 生成质量报告
    creator.create_roi_quality_report()
    
    # 打印总结
    summary = creator.get_roi_summary()
    logging.info(f"ROI创建完成: {summary}")
    
    return saved_paths

if __name__ == "__main__":
    from .global_config import GlobalConfig
    
    # 创建配置
    config = GlobalConfig.create_default_config()
    
    # 示例坐标（MNI空间）
    example_coords = {
        'left_hippocampus': (-30, -15, -15),
        'right_hippocampus': (30, -15, -15),
        'left_ifg': (-45, 25, 5),
        'right_ifg': (45, 25, 5)
    }
    
    # 创建ROI
    saved_paths = create_standard_rois(config, example_coords)
    print(f"创建了 {len(saved_paths)} 个ROI")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")