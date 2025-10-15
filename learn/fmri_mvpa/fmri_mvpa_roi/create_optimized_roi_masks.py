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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')


# ==================== 配置参数 ====================
class ROIMaskConfig:
    """ROI mask生成配置类"""

    def __init__(self,
                 clusters_file=None,
                 output_dir=None,
                 sphere_radius=8,
                 min_voxels_sphere=50,
                 anatomical_threshold=25,
                 min_voxels_anatomical=100,
                 activation_threshold=3.1,
                 cluster_size_threshold=10,
                 min_voxels_activation=20,
                 network_threshold=0.2,
                 min_voxels_network=200,
                 max_roi_volume=10000,
                 min_roi_volume=100,
                 overlap_threshold=0.8):

        # 文件路径配置
        self.clusters_file = clusters_file
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'optimized_roi_masks')

        # 球形ROI配置
        self.sphere_radius = sphere_radius  # mm，球形ROI半径
        self.sphere_radius_mm = sphere_radius  # 兼容性别名
        self.min_voxels_sphere = min_voxels_sphere  # 球形ROI最小体素数

        # 解剖ROI配置
        self.anatomical_threshold = anatomical_threshold  # 解剖ROI概率阈值
        self.min_voxels_anatomical = min_voxels_anatomical  # 解剖ROI最小体素数

        # 激活ROI配置
        self.activation_threshold = activation_threshold  # 激活阈值（t值）
        self.cluster_size_threshold = cluster_size_threshold  # 最小cluster大小（体素数）
        self.min_voxels_activation = min_voxels_activation  # 激活ROI最小体素数

        # 功能网络配置
        self.network_threshold = network_threshold  # 网络mask阈值
        self.min_voxels_network = min_voxels_network  # 网络ROI最小体素数

        # 质量控制配置
        self.max_roi_volume = max_roi_volume  # ROI最大体积（mm³）
        self.min_roi_volume = min_roi_volume  # ROI最小体积（mm³）
        self.overlap_threshold = overlap_threshold  # ROI重叠检测阈值

        # Harvard-Oxford模板配置
        self.ho_template = None  # 将在初始化时加载

        # 生成文件路径
        self.mask_info_file = os.path.join(self.output_dir, 'mask_information.csv')
        self.quality_report_file = os.path.join(self.output_dir, 'roi_quality_report.csv')

        # 半球分离配置
        self.apply_hemisphere_separation = True

        # 可视化配置
        self.plot_dpi = 150
        self.plot_figsize = (15, 10)
        self.colormap = 'hot'

    def load_activation_data(self, activation_table_path):
        """加载激活数据

        期望的CSV格式包含以下字段:
        - 'Cluster ID': 激活cluster的ID
        - 'X', 'Y', 'Z': MNI坐标
        - 'Label(HO)': Harvard-Oxford图谱标签
        """
        try:
            if activation_table_path and os.path.exists(activation_table_path):
                df = pd.read_csv(activation_table_path)

                # 验证必需的字段
                required_fields = ['Cluster ID', 'X', 'Y', 'Z']
                missing_fields = [field for field in required_fields if field not in df.columns]

                if missing_fields:
                    logging.warning(f"CSV文件缺少必需字段: {missing_fields}")
                    logging.warning(f"可用字段: {list(df.columns)}")
                    return None

                # 过滤有效的激活点（有坐标和Cluster ID）
                valid_activations = df.dropna(subset=['Cluster ID', 'X', 'Y', 'Z'])
                logging.info(f"加载激活数据: {len(valid_activations)} 个有效激活点 (总共 {len(df)} 行)")

                return valid_activations
            else:
                logging.info("未提供激活数据文件或文件不存在")
                return None
        except Exception as e:
            logging.error(f"加载激活数据时出错: {e}")
            return None

    def load_ho_template(self):
        """加载Harvard-Oxford模板作为统一参考空间"""
        if self.ho_template is None:
            try:
                # 加载Harvard-Oxford图谱
                ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                self.ho_template = load_img(ho_atlas['maps'])
                logging.info("Harvard-Oxford模板加载成功")
                logging.info(f"模板形状: {self.ho_template.shape}")
                logging.info(f"模板仿射矩阵:\n{self.ho_template.affine}")
            except Exception as e:
                logging.warning(f"无法加载Harvard-Oxford模板: {e}")
                # 备用方案：使用MNI模板
                try:
                    self.ho_template = datasets.load_mni152_template(resolution=2)
                    logging.info("使用MNI模板作为备用")
                except Exception as e2:
                    logging.error(f"完全无法加载模板: {e2}")
                    raise
        return self.ho_template

    def create_output_dir(self):
        """创建输出目录"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"输出目录已创建: {self.output_dir}")
        except Exception as e:
            raise ValueError(f"无法创建输出目录 {self.output_dir}: {e}")

    def validate_config(self):
        """验证配置参数"""
        assert self.sphere_radius > 0, "球形ROI半径必须大于0"
        assert self.min_voxels_sphere > 0, "最小体素数必须大于0"
        assert 0 < self.anatomical_threshold <= 100, "解剖ROI阈值必须在0-100之间"
        assert self.min_roi_volume < self.max_roi_volume, "最小体积必须小于最大体积"
        logging.info("配置验证通过")


def create_sphere_roi(coord, radius, template_img=None, config=None):
    """
    创建球形ROI

    Parameters:
    -----------
    coord : tuple
        MNI坐标 (x, y, z)
    radius : float
        球形半径（mm）
    template_img : nibabel image, optional
        模板图像
    config : ROIMaskConfig, optional
        配置对象

    Returns:
    --------
    roi_img : nibabel image
        球形ROI图像
    quality_info : dict
        质量信息
    """
    if template_img is None:
        if config and config.ho_template is not None:
            template_img = config.ho_template
        else:
            template_img = datasets.load_mni152_template(resolution=2)

    # 创建球形ROI
    sphere_data = create_precise_sphere_mask(coord, radius, template_img.affine, template_img.shape)

    # 修复：将数据转换为兼容的数据类型
    sphere_data = sphere_data.astype(np.int32)  # 使用int32而不是int64

    roi_img = nib.Nifti1Image(sphere_data, template_img.affine)

    # 计算质量信息
    voxel_count = np.sum(sphere_data)
    voxel_volume = np.prod(template_img.header.get_zooms()[:3])  # mm³
    roi_volume = voxel_count * voxel_volume

    quality_info = {
        'voxel_count': int(voxel_count),
        'roi_volume_mm3': float(roi_volume),
        'center_coord': coord,
        'radius_mm': radius,
        'is_valid': voxel_count >= (config.min_voxels_sphere if config else 50)
    }

    return roi_img, quality_info


def create_activation_roi_optimized(activation_coords, template_img=None, config=None):
    """
    基于完整激活cluster创建ROI，而非单点激活

    Parameters:
    -----------
    activation_coords : list of tuples
        激活坐标列表 [(x1,y1,z1), (x2,y2,z2), ...]
    template_img : nibabel image, optional
        模板图像
    config : ROIMaskConfig, optional
        配置对象

    Returns:
    --------
    roi_img : nibabel image
        激活ROI图像
    quality_info : dict
        质量信息
    """
    if template_img is None:
        if config and config.ho_template is not None:
            template_img = config.ho_template
        else:
            template_img = datasets.load_mni152_template(resolution=2)

    cluster_threshold = config.cluster_size_threshold if config else 10

    # 创建cluster mask
    cluster_mask = np.zeros(template_img.shape[:3])

    for coord in activation_coords:
        # 为每个激活点创建小球形ROI
        sphere_mask = create_precise_sphere_mask(coord, 6, template_img.affine, template_img.shape)
        cluster_mask = np.maximum(cluster_mask, sphere_mask)

    # 应用cluster阈值
    if np.sum(cluster_mask) < cluster_threshold:
        warnings.warn(f"Activation cluster size ({np.sum(cluster_mask)}) below threshold ({cluster_threshold})")

    # 修复：将数据转换为兼容的数据类型
    cluster_mask = cluster_mask.astype(np.int32)  # 使用int32而不是int64

    # 创建激活ROI图像
    roi_img = image.new_img_like(template_img, cluster_mask)

    # 计算质量信息
    voxel_count = np.sum(cluster_mask)
    voxel_volume = np.prod(template_img.header.get_zooms()[:3])  # mm³
    roi_volume = voxel_count * voxel_volume

    quality_info = {
        'voxel_count': int(voxel_count),
        'roi_volume_mm3': float(roi_volume),
        'n_activation_coords': len(activation_coords),
        'cluster_threshold': cluster_threshold,
        'is_valid': voxel_count >= (config.min_voxels_activation if config else 20)
    }

    return roi_img, quality_info


def create_network_roi(network_name='dmn', config=None):
    """
    创建功能网络ROI，统一使用Harvard-Oxford空间

    Parameters:
    -----------
    network_name : str, default='dmn'
        网络名称 ('dmn', 'language')
    config : ROIMaskConfig, optional
        配置对象

    Returns:
    --------
    roi_img : nibabel image
        网络ROI图像
    quality_info : dict
        质量信息
    """
    try:
        # 加载网络图谱
        if network_name.lower() == 'dmn':
            # 使用Yeo网络或其他DMN图谱
            yeo_atlas = datasets.fetch_atlas_yeo_2011()
            network_img = load_img(yeo_atlas['thick_7'])
            network_label = 7  # DMN通常是第7个网络
        elif network_name.lower() == 'language':
            # 可以使用其他语言网络图谱
            yeo_atlas = datasets.fetch_atlas_yeo_2011()
            network_img = load_img(yeo_atlas['thick_7'])
            network_label = 6  # 语言网络
        else:
            raise ValueError(f"Unsupported network: {network_name}")

        # 应用阈值
        threshold = config.network_threshold if config else 0.2
        network_data = network_img.get_fdata()
        mask_data = (network_data == network_label).astype(np.int32)  # 修复：使用int32

        # 创建网络ROI图像
        network_roi_img = image.new_img_like(network_img, mask_data)

        # 统一重采样到Harvard-Oxford空间
        if config and config.ho_template is not None:
            roi_img = resample_to_img(network_roi_img, config.ho_template, interpolation='nearest')
        else:
            roi_img = network_roi_img

        # 计算质量信息
        voxel_count = np.sum(roi_img.get_fdata() > 0)
        voxel_volume = np.prod(roi_img.header.get_zooms()[:3])  # mm³
        roi_volume = voxel_count * voxel_volume

        quality_info = {
            'voxel_count': int(voxel_count),
            'roi_volume_mm3': float(roi_volume),
            'network_name': network_name,
            'threshold': threshold,
            'is_valid': voxel_count >= (config.min_voxels_network if config else 200)
        }

        return roi_img, quality_info

    except Exception as e:
        logging.error(f"创建网络ROI失败: {e}")
        # 返回空ROI
        if config and config.ho_template is not None:
            template = config.ho_template
        else:
            template = datasets.load_mni152_template()
        empty_img = image.new_img_like(template, np.zeros(template.shape[:3], dtype=np.int32))
        quality_info = {
            'voxel_count': 0,
            'roi_volume_mm3': 0.0,
            'network_name': network_name,
            'threshold': 0,
            'is_valid': False,
            'error': str(e)
        }
        return empty_img, quality_info


def check_roi_overlap(roi1_img, roi2_img, threshold=0.5, reference_img=None):
    """
    检查两个ROI之间的重叠程度

    Parameters:
    -----------
    roi1_img, roi2_img : nibabel images
        要比较的ROI图像
    threshold : float, default=0.5
        重叠阈值
    reference_img : nibabel image, optional
        参考图像，用于重采样

    Returns:
    --------
    overlap_ratio : float
        重叠比例
    is_overlapping : bool
        是否超过阈值
    """
    try:
        # 获取图像数据
        roi1_data = roi1_img.get_fdata() > 0
        roi2_data = roi2_img.get_fdata() > 0

        # 检查形状是否匹配
        if roi1_data.shape != roi2_data.shape:
            logging.warning(f"ROI形状不匹配 - ROI1: {roi1_data.shape}, ROI2: {roi2_data.shape}")

            # 如果提供了参考图像，则重采样到相同空间
            if reference_img is not None:
                logging.info("正在重采样ROI到参考图像空间...")
                roi1_resampled = resample_to_img(roi1_img, reference_img, interpolation='nearest')
                roi2_resampled = resample_to_img(roi2_img, reference_img, interpolation='nearest')

                roi1_data = roi1_resampled.get_fdata() > 0
                roi2_data = roi2_resampled.get_fdata() > 0
            else:
                # 如果没有参考图像，尝试重采样到第一个ROI的空间
                logging.info("正在重采样ROI2到ROI1空间...")
                roi2_resampled = resample_to_img(roi2_img, roi1_img, interpolation='nearest')
                roi2_data = roi2_resampled.get_fdata() > 0

        # 再次检查形状是否匹配
        if roi1_data.shape != roi2_data.shape:
            logging.error(f"重采样后形状仍不匹配 - ROI1: {roi1_data.shape}, ROI2: {roi2_data.shape}")
            return 0.0, False

        intersection = np.sum(roi1_data & roi2_data)
        union = np.sum(roi1_data | roi2_data)

        overlap_ratio = intersection / union if union > 0 else 0
        is_overlapping = overlap_ratio > threshold

        return overlap_ratio, is_overlapping
    except Exception as e:
        logging.error(f"计算ROI重叠失败: {e}")
        return 0.0, False


def validate_roi_quality(roi_img, config, roi_name=""):
    """
    验证ROI质量

    Parameters:
    -----------
    roi_img : nibabel image
        ROI图像
    config : ROIMaskConfig
        配置对象
    roi_name : str
        ROI名称

    Returns:
    --------
    quality_report : dict
        质量报告
    """
    try:
        roi_data = roi_img.get_fdata()
        voxel_count = np.sum(roi_data > 0)
        voxel_volume = np.prod(roi_img.header.get_zooms()[:3])
        roi_volume = voxel_count * voxel_volume

        # 检查ROI是否为空
        is_empty = voxel_count == 0

        # 检查ROI大小是否合理
        is_too_small = roi_volume < config.min_roi_volume
        is_too_large = roi_volume > config.max_roi_volume

        # 检查ROI连通性
        labeled_roi, n_components = label(roi_data > 0)
        is_fragmented = n_components > 3  # 超过3个连通分量认为过于分散

        # 检查ROI形状
        if voxel_count > 0:
            coords = np.where(roi_data > 0)
            centroid = [np.mean(coords[i]) for i in range(3)]
            max_distance = np.max(
                [np.sqrt(np.sum((np.array([coords[0][i], coords[1][i], coords[2][i]]) - centroid) ** 2))
                 for i in range(len(coords[0]))])
            is_elongated = max_distance > 20  # 最大距离超过20个体素认为过于细长
        else:
            is_elongated = False

        quality_report = {
            'roi_name': roi_name,
            'voxel_count': int(voxel_count),
            'roi_volume_mm3': float(roi_volume),
            'n_components': int(n_components),
            'is_empty': is_empty,
            'is_too_small': is_too_small,
            'is_too_large': is_too_large,
            'is_fragmented': is_fragmented,
            'is_elongated': is_elongated,
            'is_valid': not (is_empty or is_too_small or is_too_large or is_fragmented or is_elongated)
        }

        return quality_report

    except Exception as e:
        logging.error(f"ROI质量验证失败 {roi_name}: {e}")
        return {
            'roi_name': roi_name,
            'voxel_count': 0,
            'roi_volume_mm3': 0.0,
            'n_components': 0,
            'is_empty': True,
            'is_too_small': True,
            'is_too_large': False,
            'is_fragmented': False,
            'is_elongated': False,
            'is_valid': False,
            'error': str(e)
        }


def create_anatomical_roi_optimized(region_name, hemisphere='both', config=None):
    """
    创建优化的解剖ROI，使用Harvard-Oxford图谱

    Parameters:
    -----------
    region_name : str
        解剖区域名称
    hemisphere : str, default='both'
        半球选择 ('left', 'right', 'both')
    config : ROIMaskConfig, optional
        配置对象

    Returns:
    --------
    roi_img : nibabel image
        解剖ROI图像
    quality_info : dict
        质量信息
    """
    try:
        # 加载Harvard-Oxford图谱
        ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = load_img(ho_atlas['maps'])
        labels = ho_atlas['labels']

        # 查找匹配的区域
        atlas_data = atlas_img.get_fdata()
        # 修复：使用兼容的数据类型
        mask_data = np.zeros(atlas_img.shape[:3], dtype=np.int32)

        for i, label in enumerate(labels):
            if region_name.lower() in label.lower():
                region_mask = (atlas_data == i).astype(np.int32)  # 修复：使用int32
                mask_data = np.maximum(mask_data, region_mask)

        # 应用半球分离
        if hemisphere in ['left', 'right']:
            mask_data = apply_hemisphere_separation(mask_data, f"{region_name}_{hemisphere[0].upper()}",
                                                    atlas_img.affine, atlas_img.shape)

        # 创建ROI图像
        roi_img = image.new_img_like(atlas_img, mask_data)

        # 计算质量信息
        voxel_count = np.sum(mask_data)
        voxel_volume = np.prod(atlas_img.header.get_zooms()[:3])
        roi_volume = voxel_count * voxel_volume

        quality_info = {
            'voxel_count': int(voxel_count),
            'roi_volume_mm3': float(roi_volume),
            'region_name': region_name,
            'hemisphere': hemisphere,
            'is_valid': voxel_count >= (config.min_voxels_anatomical if config else 100)
        }

        return roi_img, quality_info

    except Exception as e:
        logging.error(f"创建解剖ROI失败 {region_name}: {e}")
        # 返回空ROI
        if config and config.ho_template is not None:
            template = config.ho_template
        else:
            template = datasets.load_mni152_template()
        empty_img = image.new_img_like(template, np.zeros(template.shape[:3], dtype=np.int32))
        quality_info = {
            'voxel_count': 0,
            'roi_volume_mm3': 0.0,
            'region_name': region_name,
            'hemisphere': hemisphere,
            'is_valid': False,
            'error': str(e)
        }
        return empty_img, quality_info


def create_activation_cluster_mask(activation_df, template_img, config):
    """基于激活cluster的MNI坐标创建球形区域并合并为完整mask

    Args:
        activation_df: 包含激活数据的DataFrame，需要有'X', 'Y', 'Z'字段
        template_img: 模板图像
        config: 配置对象
    """
    if activation_df is None or len(activation_df) == 0:
        return None

    cluster_mask = np.zeros(template_img.shape, dtype=np.int32)  # 修复：指定数据类型

    for idx, row in activation_df.iterrows():
        # 使用大写字段名，与原始数据格式一致
        coords = [row['X'], row['Y'], row['Z']]
        sphere_mask = create_precise_sphere_mask(
            coords, config.sphere_radius_mm, template_img.affine, template_img.shape
        )
        cluster_mask = np.maximum(cluster_mask, sphere_mask)

    return cluster_mask


def create_precise_sphere_mask(center_coord, radius_mm, affine, shape):
    """
    创建精确的球形mask

    Parameters:
    -----------
    center_coord : tuple
        中心坐标 (x, y, z) in mm
    radius_mm : float
        半径（mm）
    affine : array
        仿射变换矩阵
    shape : tuple
        图像形状

    Returns:
    --------
    sphere_mask : array
        球形mask
    """
    try:
        # 创建体素坐标网格
        i, j, k = np.meshgrid(np.arange(shape[0]),
                              np.arange(shape[1]),
                              np.arange(shape[2]), indexing='ij')

        # 转换为世界坐标
        voxel_coords = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
        world_coords = affine @ voxel_coords

        # 计算到中心点的距离
        distances = np.sqrt(np.sum((world_coords[:3] - np.array(center_coord).reshape(-1, 1)) ** 2, axis=0))

        # 创建球形mask
        sphere_mask = (distances <= radius_mm).reshape(shape)

        # 修复：返回int32类型而不是默认的bool或int64
        return sphere_mask.astype(np.int32)

    except Exception as e:
        logging.error(f"创建球形mask失败: {e}")
        return np.zeros(shape, dtype=np.int32)


def generate_quality_report_optimized(quality_reports, config):
    """
    生成ROI质量报告

    Parameters:
    -----------
    quality_reports : list
        质量报告列表
    config : ROIMaskConfig
        配置对象

    Returns:
    --------
    summary_stats : dict
        汇总统计
    """
    if not quality_reports:
        return {}

    try:
        df = pd.DataFrame(quality_reports)

        summary_stats = {
            'total_rois': len(df),
            'valid_rois': len(df[df['is_valid']]),
            'invalid_rois': len(df[~df['is_valid']]),
            'avg_voxel_count': df['voxel_count'].mean(),
            'avg_volume_mm3': df['roi_volume_mm3'].mean()
        }

        # 保存详细报告
        df.to_csv(config.quality_report_file, index=False)
        logging.info(f"质量报告已保存至: {config.quality_report_file}")

        return summary_stats

    except Exception as e:
        logging.error(f"生成质量报告失败: {e}")
        return {}


def detect_roi_overlaps(roi_masks, config):
    """
    检测ROI之间的重叠

    Parameters:
    -----------
    roi_masks : dict
        ROI mask字典
    config : ROIMaskConfig
        配置对象

    Returns:
    --------
    overlap_pairs : list
        重叠的ROI对列表
    """
    overlap_pairs = []
    roi_names = list(roi_masks.keys())

    # 获取参考图像（使用Harvard-Oxford模板）
    reference_img = config.load_ho_template()

    try:
        for i, roi1_name in enumerate(roi_names):
            for j, roi2_name in enumerate(roi_names[i + 1:], i + 1):
                overlap_ratio, is_overlapping = check_roi_overlap(
                    roi_masks[roi1_name],
                    roi_masks[roi2_name],
                    config.overlap_threshold,
                    reference_img=reference_img  # 传递参考图像用于重采样
                )

                if is_overlapping:
                    overlap_pairs.append({
                        'roi1': roi1_name,
                        'roi2': roi2_name,
                        'overlap_ratio': overlap_ratio
                    })

        return overlap_pairs

    except Exception as e:
        logging.error(f"检测ROI重叠失败: {e}")
        return []


def ensure_consistent_space(roi_img, reference_img):
    """
    确保ROI图像与参考图像在相同的空间

    Parameters:
    -----------
    roi_img : nibabel image
        ROI图像
    reference_img : nibabel image
        参考图像

    Returns:
    --------
    resampled_img : nibabel image
        重采样后的ROI图像
    """
    try:
        # 检查是否已经在相同空间
        if roi_img.shape == reference_img.shape and np.allclose(roi_img.affine, reference_img.affine):
            return roi_img

        # 重采样到参考图像空间
        logging.info(f"重采样ROI: {roi_img.shape} -> {reference_img.shape}")
        resampled_img = resample_to_img(roi_img, reference_img, interpolation='nearest')
        return resampled_img

    except Exception as e:
        logging.error(f"重采样ROI失败: {e}")
        return roi_img


def _save_and_record_roi(roi_img, roi_name, roi_type, quality_info, config, roi_masks, mask_info, quality_reports, extra_info=None):
    """Helper function to save ROI, record info, and update reports."""
    if not quality_info.get('is_valid', True):
        logging.warning(f"{roi_name} 质量不合格，跳过")
        quality_reports.append(quality_info)
        return

    try:
        roi_path = os.path.join(config.output_dir, f"{roi_name}.nii.gz")
        nib.save(roi_img, roi_path)

        roi_masks[roi_name] = roi_img
        quality_info.update({
            'roi_name': roi_name,
            'roi_type': roi_type,
            'file_path': roi_path
        })
        quality_reports.append(quality_info)

        info = {
            'roi_name': roi_name,
            'roi_type': roi_type,
            'n_voxels': quality_info['voxel_count'],
            'file_path': roi_path
        }
        if extra_info:
            info.update(extra_info)
        mask_info.append(info)

        logging.info(f"已保存: {roi_name}.nii.gz ({quality_info['voxel_count']} voxels)")
    except Exception as e:
        logging.error(f"保存和记录ROI失败 {roi_name}: {e}")


def create_optimized_metaphor_masks(config=None):
    """
    优化版本：基于隐喻fMRI激活数据和Harvard-Oxford图谱创建ROI masks
    统一使用Harvard-Oxford空间

    Parameters:
    -----------
    config : ROIMaskConfig
        配置对象

    Returns:
    --------
    roi_masks : dict
        ROI mask字典
    mask_info : list
        mask信息列表
    quality_reports : list
        质量报告列表
    """

    # 初始化配置
    if config is None:
        config = ROIMaskConfig()

    try:
        config.validate_config()
        config.create_output_dir()
    except ValueError as e:
        logging.error(f"配置错误: {e}")
        return {}, [], []

    # 加载Harvard-Oxford模板 - 这里确保模板被正确加载
    try:
        template_img = config.load_ho_template()
        if template_img is None:
            raise ValueError("无法加载Harvard-Oxford模板，无法继续创建ROI masks")
    except Exception as e:
        logging.error(f"模板加载失败: {e}")
        return {}, [], []

    # 初始化存储
    roi_masks = {}
    mask_info = []
    quality_reports = []

    logging.info("Creating optimized ROI masks...")
    logging.info(f"Using Harvard-Oxford template space: {template_img.shape}")

    # 读取激活数据（如果提供了路径）
    df = None
    if config.clusters_file and os.path.exists(config.clusters_file):
        df = config.load_activation_data(config.clusters_file)
        if df is not None:
            logging.info(f"Found {len(df)} valid activation clusters")
    else:
        logging.info("No activation data provided or file does not exist, creating only anatomical and network ROIs")

    # 1. 优化的球形ROI创建
    logging.info("\n--- Creating spherical ROIs ---")
    if df is not None and not df.empty and 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row.get('X')) and pd.notna(row.get('Y')) and pd.notna(row.get('Z')):
                try:
                    # 处理Cluster ID，去掉字母后缀（与原始代码一致）
                    cluster_id = str(row.get('Cluster ID', idx))
                    cluster_id = re.sub(r'[^0-9]', '', cluster_id) or str(idx)

                    # 创建ROI名称
                    if 'Label(HO)' in row and pd.notna(row['Label(HO)']):
                        label_part = str(row['Label(HO)']).split(';')[0].strip()
                        clean_label = re.sub(r'[^\w\s]', '', label_part).replace(' ', '_')
                        clean_label = clean_label[:20]  # 限制长度
                        roi_name = f"cluster_{cluster_id}_{clean_label}"
                    else:
                        roi_name = f"cluster_{cluster_id}"

                    # 创建优化的球形mask
                    coords = (float(row['X']), float(row['Y']), float(row['Z']))
                    sphere_img, sphere_quality = create_sphere_roi(coords, config.sphere_radius, template_img, config)

                    _save_and_record_roi(
                        sphere_img, f"{roi_name}_sphere", 'sphere', sphere_quality, config,
                        roi_masks, mask_info, quality_reports,
                        extra_info={'center_coord': f"({coords[0]}, {coords[1]}, {coords[2]})", 'radius_mm': config.sphere_radius}
                    )

                except Exception as e:
                    logging.error(f"创建球形ROI失败 (行 {idx}): {e}")
    else:
        logging.info("  Skipping spherical ROI creation (no valid activation data)")

    # 2. 优化的解剖ROI创建
    logging.info("\n--- Creating anatomical ROIs ---")
    anatomical_regions = {
        'Precuneus': 'Precuneus',
        'PosteriorCingulate': 'Cingulate',
        'AngularGyrus': 'Angular',
        'MiddleFrontalGyrus': 'Middle Frontal',
        'InferiorFrontalGyrus': 'Inferior Frontal'
    }

    for region_key, region_search in anatomical_regions.items():
        for hemisphere in ['left', 'right']:
            try:
                anat_img, anat_quality = create_anatomical_roi_optimized(
                    region_search, hemisphere=hemisphere, config=config
                )
                anat_img_resampled = ensure_consistent_space(anat_img, template_img)
                anat_name = f"{region_key}_{hemisphere[0].upper()}"

                _save_and_record_roi(
                    anat_img_resampled, f"{anat_name}_anatomical", 'anatomical', anat_quality, config,
                    roi_masks, mask_info, quality_reports,
                    extra_info={'region_name': region_key, 'hemisphere': hemisphere}
                )

            except Exception as e:
                logging.error(f"创建解剖ROI失败 {region_key}_{hemisphere}: {e}")

    # 3. 功能网络ROI
    logging.info("\n--- Creating network ROIs ---")
    networks = ['dmn', 'language']

    for network_name in networks:
        try:
            network_img, network_quality = create_network_roi(network_name, config)

            _save_and_record_roi(
                network_img, f"{network_name}_network", 'network', network_quality, config,
                roi_masks, mask_info, quality_reports,
                extra_info={'network_name': network_name}
            )

        except Exception as e:
            logging.error(f"创建网络ROI失败 {network_name}: {e}")

    # 保存mask信息到CSV
    try:
        mask_df = pd.DataFrame(mask_info)
        mask_df.to_csv(config.mask_info_file, index=False)
        logging.info(f"Mask信息已保存至: {config.mask_info_file}")
    except Exception as e:
        logging.error(f"保存mask信息失败: {e}")

    # 生成质量报告
    try:
        summary_stats = generate_quality_report_optimized(quality_reports, config)
        logging.info(f"质量报告统计: {summary_stats}")
    except Exception as e:
        logging.error(f"生成质量报告失败: {e}")

    # 检测ROI重叠
    try:
        overlap_report = detect_roi_overlaps(roi_masks, config)
        if overlap_report:
            overlap_df = pd.DataFrame(overlap_report)
            overlap_file = os.path.join(config.output_dir, "roi_overlap_report.csv")
            overlap_df.to_csv(overlap_file, index=False)
            logging.info(f"重叠报告已保存至: {overlap_file}")
        else:
            logging.info("未检测到显著的ROI重叠")
    except Exception as e:
        logging.error(f"检测ROI重叠失败: {e}")

    logging.info(f"\n=== Optimized ROI creation complete ===")
    logging.info(f"Successfully created {len(roi_masks)} ROI masks")
    logging.info(f"Output directory: {config.output_dir}")

    return roi_masks, mask_info, quality_reports


def apply_hemisphere_separation(region_mask, roi_name, affine, shape):
    """
    优化的左右半球分离逻辑
    """
    try:
        # 找出所有非零体素的索引
        voxel_indices = np.array(np.where(region_mask > 0))
        if voxel_indices.shape[1] == 0:
            return region_mask

        # 将体素索引转换为世界坐标
        world_coords = image.coord_transform(voxel_indices[0], voxel_indices[1], voxel_indices[2], affine)
        x_coords = world_coords[0]

        # 根据ROI名称和x坐标确定要保留的体素
        if roi_name.endswith('_L'):  # 左半球 (x < 0)
            valid_mask = x_coords < 0
        elif roi_name.endswith('_R'):  # 右半球 (x > 0)
            valid_mask = x_coords > 0
        else:
            return region_mask  # 不需要分离

        # 创建新的mask
        new_mask_data = np.zeros_like(region_mask)
        valid_indices = tuple(voxel_indices[:, valid_mask])
        new_mask_data[valid_indices] = region_mask[valid_indices]

        return new_mask_data

    except Exception as e:
        logging.error(f"半球分离失败 for {roi_name}: {e}")
        return region_mask


def main():
    """
    主函数：演示如何使用优化的ROI mask创建功能
    """
    import argparse
    parser = argparse.ArgumentParser(description='Create optimized ROI masks for fMRI analysis.')
    parser.add_argument('--clusters_file', type=str, default='h:\\PythonAnalysis\\learn_batch\\2nd_level\\run3\\yy_vs_kj\\clusters_table.csv', help='Path to the clusters table CSV file.')
    parser.add_argument('--output_dir', type=str, default='h:\\PythonAnalysis\\learn_mvpa\\full_roi_mask', help='Directory to save the output ROI masks.')
    parser.add_argument('--sphere_radius', type=int, default=8, help='Radius of the spherical ROIs in mm.')
    parser.add_argument('--min_voxels_sphere', type=int, default=50, help='Minimum number of voxels for a spherical ROI to be valid.')
    parser.add_argument('--min_voxels_anatomical', type=int, default=100, help='Minimum number of voxels for an anatomical ROI to be valid.')
    args = parser.parse_args()

    logging.info("=== 创建优化版ROI Masks ===")

    # 创建配置
    config = ROIMaskConfig(
        clusters_file=args.clusters_file,
        output_dir=args.output_dir,
        sphere_radius=args.sphere_radius,
        min_voxels_sphere=args.min_voxels_sphere,
        min_voxels_anatomical=args.min_voxels_anatomical
    )

    logging.info(f"配置信息:")
    logging.info(f"  球形ROI半径: {config.sphere_radius}mm")
    logging.info(f"  输出目录: {config.output_dir}")
    logging.info(f"  解剖ROI阈值: {config.anatomical_threshold}")
    logging.info(f"  网络ROI阈值: {config.network_threshold}")

    # 创建ROI masks
    try:
        roi_masks, mask_info, quality_reports = create_optimized_metaphor_masks(config)
        logging.info(f"\n成功创建 {len(roi_masks)} 个ROI masks")

        # 统计信息
        valid_rois = sum(1 for q in quality_reports if q.get('is_valid', False))
        logging.info(f"质量合格的ROI数量: {valid_rois}/{len(quality_reports)}")

    except Exception as e:
        logging.error(f"创建ROI masks失败: {e}")

    logging.info("\n=== ROI创建完成 ===")


if __name__ == "__main__":
    main()