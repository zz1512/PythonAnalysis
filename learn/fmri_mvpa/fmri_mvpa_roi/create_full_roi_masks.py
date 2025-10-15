import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.plotting import plot_roi, plot_stat_map
import os
import pandas as pd
import re


def create_metaphor_masks(activation_table_path, output_dir=r'H:\PythonAnalysis\learn_mvpa\full_roi_mask'):
    """
    基于隐喻fMRI激活数据和Harvard-Oxford图谱创建ROI masks
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载Harvard-Oxford图谱
    print("Loading Harvard-Oxford atlas...")
    ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = load_img(ho_atlas['maps'])
    labels = ho_atlas['labels']

    # 获取体素尺寸信息
    voxel_dims = atlas_img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)  # 单个体素的体积(mm³)
    print(f"Atlas voxel dimensions: {voxel_dims} mm, Voxel volume: {voxel_volume:.2f} mm³")

    # 读取激活数据
    df = pd.read_csv(activation_table_path)

    # 获取图谱数据数组
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    shape = atlas_img.shape

    # 创建空的mask字典来存储结果
    masks = {}

    print("Creating ROI masks...")

    # 1. 基于激活坐标创建球形ROI (6mm半径)
    sphere_masks = {}
    for idx, row in df.iterrows():
        if pd.notna(row['Cluster ID']) and pd.notna(row['X']):
            # 处理Cluster ID，去掉字母后缀
            cluster_id = str(row['Cluster ID']).rstrip('abcdefghijklmnopqrstuvwxyz')

            # 创建更简洁的ROI名称
            label_part = row['Label(HO)'].split(';')[0].strip()
            # 清理标签名称，移除特殊字符
            clean_label = re.sub(r'[^\w\s]', '', label_part).replace(' ', '_')

            roi_name = f"cluster_{cluster_id}_{clean_label}"

            # 创建球形mask (6mm半径)
            coords = [row['X'], row['Y'], row['Z']]

            # 简单的球形mask创建
            sphere_data = np.zeros(shape)
            radius_mm = 10

            # 将毫米坐标转换为体素坐标
            vox_coords = np.linalg.inv(affine).dot(np.array([*coords, 1]))[:3]
            vox_coords = np.round(vox_coords).astype(int)

            # 计算体素尺寸（用于准确的距离计算）
            voxel_size = np.abs(affine.diagonal()[:3])

            # 计算在体素空间中的半径（近似值）
            # 假设各向同性体素，取平均体素尺寸
            avg_voxel_size = np.mean(voxel_size)
            radius_vox = int(np.ceil(radius_mm / avg_voxel_size))

            # 创建球形区域 - 扩大搜索范围以适应10mm半径
            for i in range(max(0, vox_coords[0] - radius_vox), min(shape[0], vox_coords[0] + radius_vox + 1)):
                for j in range(max(0, vox_coords[1] - radius_vox), min(shape[1], vox_coords[1] + radius_vox + 1)):
                    for k in range(max(0, vox_coords[2] - radius_vox), min(shape[2], vox_coords[2] + radius_vox + 1)):
                        # 计算毫米距离（更准确）
                        world_coord_i = affine.dot(np.array([i, j, k, 1]))[:3]
                        world_coord_center = np.array(coords)
                        dist_mm = np.linalg.norm(world_coord_i - world_coord_center)

                        if dist_mm <= radius_mm:
                            sphere_data[i, j, k] = 1

            sphere_img = nib.Nifti1Image(sphere_data, affine)

            # 计算体素统计信息
            n_voxels = np.sum(sphere_data > 0)
            volume_mm3 = n_voxels * voxel_volume

            # 更新mask大小分类标准以适应更大的ROI
            if n_voxels == 0:
                size_category = "Empty"
            elif n_voxels < 20:  # 调整阈值
                size_category = "Very Small"
            elif n_voxels < 80:  # 调整阈值
                size_category = "Small"
            elif n_voxels < 150:  # 调整阈值
                size_category = "Medium"
            else:
                size_category = "Large"

            print(f"  {roi_name}_sphere: {n_voxels} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

            # 保存球形mask
            sphere_path = os.path.join(output_dir, f"{roi_name}_sphere.nii.gz")
            nib.save(sphere_img, sphere_path)
            masks[roi_name + '_sphere'] = {
                'path': sphere_path,
                'n_voxels': n_voxels,
                'volume_mm3': volume_mm3,
                'size_category': size_category
            }

    print("--- Spherical ROIs completed ---")

    # 2. 基于HO图谱的解剖ROI
    anatomical_masks = {}

    # 定义解剖ROI映射
    anatomical_rois = {
        'Precuneus_L': ['Precuneous Cortex'],
        'Precuneus_R': ['Precuneous Cortex'],
        'PosteriorCingulate': ['Cingulate Gyrus, posterior division'],
        'AnteriorCingulate': ['Cingulate Gyrus, anterior division'],
        'Insula_L': ['Insular Cortex'],
        'Insula_R': ['Insular Cortex'],
        'TemporalPole_L': ['Temporal Pole'],
        'TemporalPole_R': ['Temporal Pole'],
        'Fusiform_L': ['Temporal Fusiform Cortex, posterior division'],
        'Fusiform_R': ['Temporal Fusiform Cortex, posterior division'],
        'FrontalPole_L': ['Frontal Pole'],
        'FrontalPole_R': ['Frontal Pole'],
        'LateralOccipital_L': ['Lateral Occipital Cortex, superior division'],
        'LateralOccipital_R': ['Lateral Occipital Cortex, superior division'],
        'SuperiorTemporal_L': ['Superior Temporal Gyrus, posterior division'],
        'SuperiorTemporal_R': ['Superior Temporal Gyrus, posterior division'],
        'InferiorFrontal_L': ['Inferior Frontal Gyrus, pars opercularis'],
        'InferiorFrontal_R': ['Inferior Frontal Gyrus, pars opercularis'],
        'Parahippocampal_L': ['Parahippocampal Gyrus, posterior division'],
        'Parahippocampal_R': ['Parahippocampal Gyrus, posterior division']
    }

    for roi_name, region_names in anatomical_rois.items():
        mask_data = np.zeros(shape)

        for region_name in region_names:
            # 在标签中查找匹配的区域
            for i, label in enumerate(labels):
                if region_name in label:
                    # 提取该区域
                    region_mask = (atlas_data == i).astype(float)

                    # 根据左右半球筛选
                    if roi_name.endswith('_L'):  # 左半球
                        # 简单的左右分离：x坐标小于中线的体素
                        mid_x = shape[0] // 2
                        region_mask[mid_x:, :, :] = 0
                    elif roi_name.endswith('_R'):  # 右半球
                        mid_x = shape[0] // 2
                        region_mask[:mid_x, :, :] = 0

                    mask_data = np.maximum(mask_data, region_mask)

        n_voxels = np.sum(mask_data > 0)

        if n_voxels > 0:
            anatomical_img = nib.Nifti1Image(mask_data, affine)
            anatomical_masks[roi_name] = anatomical_img

            volume_mm3 = n_voxels * voxel_volume

            # 判断mask大小类别
            if n_voxels < 100:
                size_category = "Small"
            elif n_voxels < 500:
                size_category = "Medium"
            elif n_voxels < 1000:
                size_category = "Large"
            else:
                size_category = "Very Large"

            print(f"  {roi_name}_anatomical: {n_voxels} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

            # 保存解剖mask
            anat_path = os.path.join(output_dir, f"{roi_name}_anatomical.nii.gz")
            nib.save(anatomical_img, anat_path)
            masks[roi_name + '_anatomical'] = {
                'path': anat_path,
                'n_voxels': n_voxels,
                'volume_mm3': volume_mm3,
                'size_category': size_category
            }

    print("--- Anatomical ROIs completed ---")

    # 3. 创建功能网络mask
    network_masks = {}

    # DMN网络
    dmn_regions = ['Precuneous Cortex', 'Cingulate Gyrus, posterior division',
                   'Cingulate Gyrus, anterior division', 'Medial Frontal Cortex']
    dmn_mask = np.zeros(shape)
    for region in dmn_regions:
        for i, label in enumerate(labels):
            if region in label:
                dmn_mask = np.maximum(dmn_mask, (atlas_data == i).astype(float))

    n_voxels_dmn = np.sum(dmn_mask > 0)
    if n_voxels_dmn > 0:
        dmn_img = nib.Nifti1Image(dmn_mask, affine)
        network_masks['DMN'] = dmn_img
        dmn_path = os.path.join(output_dir, "DMN_network.nii.gz")
        nib.save(dmn_img, dmn_path)

        volume_mm3 = n_voxels_dmn * voxel_volume
        size_category = "Very Large" if n_voxels_dmn > 1000 else "Large"
        print(f"  DMN_network: {n_voxels_dmn} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

        masks['DMN_network'] = {
            'path': dmn_path,
            'n_voxels': n_voxels_dmn,
            'volume_mm3': volume_mm3,
            'size_category': size_category
        }

    # 语言网络
    language_regions = ['Inferior Frontal Gyrus, pars opercularis',
                        'Superior Temporal Gyrus, posterior division',
                        'Middle Temporal Gyrus', 'Inferior Temporal Gyrus']
    language_mask = np.zeros(shape)
    for region in language_regions:
        for i, label in enumerate(labels):
            if region in label:
                language_mask = np.maximum(language_mask, (atlas_data == i).astype(float))

    n_voxels_language = np.sum(language_mask > 0)
    if n_voxels_language > 0:
        language_img = nib.Nifti1Image(language_mask, affine)
        network_masks['Language'] = language_img
        language_path = os.path.join(output_dir, "Language_network.nii.gz")
        nib.save(language_img, language_path)

        volume_mm3 = n_voxels_language * voxel_volume
        size_category = "Very Large" if n_voxels_language > 1000 else "Large"
        print(f"  Language_network: {n_voxels_language} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

        masks['Language_network'] = {
            'path': language_path,
            'n_voxels': n_voxels_language,
            'volume_mm3': volume_mm3,
            'size_category': size_category
        }

    print("--- Network ROIs completed ---")

    # 4. 创建正负激活组合mask
    # 正激活组合
    positive_clusters = df[df['Peak Stat'] > 0]
    positive_mask = np.zeros(shape)
    for _, row in positive_clusters.iterrows():
        if pd.notna(row['X']):
            # 简单地将所有正激活区域合并
            coords = [row['X'], row['Y'], row['Z']]
            vox_coords = np.linalg.inv(affine).dot(np.array([*coords, 1]))[:3]
            vox_coords = np.round(vox_coords).astype(int)

            if all(0 <= c < s for c, s in zip(vox_coords, shape)):
                positive_mask[vox_coords[0], vox_coords[1], vox_coords[2]] = 1

    n_voxels_positive = np.sum(positive_mask > 0)
    positive_img = nib.Nifti1Image(positive_mask, affine)
    positive_path = os.path.join(output_dir, "positive_activation.nii.gz")
    nib.save(positive_img, positive_path)

    volume_mm3 = n_voxels_positive * voxel_volume
    size_category = "Very Large" if n_voxels_positive > 1000 else "Large"
    print(f"  positive_activation: {n_voxels_positive} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

    masks['positive_activation'] = {
        'path': positive_path,
        'n_voxels': n_voxels_positive,
        'volume_mm3': volume_mm3,
        'size_category': size_category
    }

    # 负激活组合
    negative_clusters = df[df['Peak Stat'] < 0]
    negative_mask = np.zeros(shape)
    for _, row in negative_clusters.iterrows():
        if pd.notna(row['X']):
            coords = [row['X'], row['Y'], row['Z']]
            vox_coords = np.linalg.inv(affine).dot(np.array([*coords, 1]))[:3]
            vox_coords = np.round(vox_coords).astype(int)

            if all(0 <= c < s for c, s in zip(vox_coords, shape)):
                negative_mask[vox_coords[0], vox_coords[1], vox_coords[2]] = 1

    n_voxels_negative = np.sum(negative_mask > 0)
    negative_img = nib.Nifti1Image(negative_mask, affine)
    negative_path = os.path.join(output_dir, "negative_activation.nii.gz")
    nib.save(negative_img, negative_path)

    volume_mm3 = n_voxels_negative * voxel_volume
    size_category = "Very Large" if n_voxels_negative > 1000 else "Large"
    print(f"  negative_activation: {n_voxels_negative} voxels, {volume_mm3:.1f} mm³ [{size_category}]")

    masks['negative_activation'] = {
        'path': negative_path,
        'n_voxels': n_voxels_negative,
        'volume_mm3': volume_mm3,
        'size_category': size_category
    }

    print("--- Activation ROIs completed ---")

    # 5. 保存mask信息表
    mask_info = []
    for mask_name, mask_data in masks.items():
        mask_info.append({
            'mask_name': mask_name,
            'file_path': mask_data['path'],
            'n_voxels': mask_data['n_voxels'],
            'volume_mm3': mask_data['volume_mm3'],
            'size_category': mask_data['size_category']
        })

    mask_df = pd.DataFrame(mask_info)
    mask_df.to_csv(os.path.join(output_dir, 'mask_information.csv'), index=False)

    print(f"Created {len(masks)} mask files in {output_dir}")
    print(f"Mask information saved to {os.path.join(output_dir, 'mask_information.csv')}")

    return masks, mask_df


def visualize_masks(masks_dict, output_dir='./metaphor_masks/visualization'):
    """
    可视化生成的masks
    """
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_anat, plot_roi

    # 加载标准模板用于背景
    mni_template = datasets.load_mni152_template()

    for mask_name, mask_data in masks_dict.items():
        try:
            mask_path = mask_data['path'] if isinstance(mask_data, dict) else mask_data

            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 三个视角
            display = plot_roi(mask_path, bg_img=mni_template,
                               display_mode='ortho', title=mask_name,
                               axes=axes, black_bg=True)

            plt.savefig(os.path.join(output_dir, f"{mask_name}_visualization.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Could not visualize {mask_name}: {e}")

    print(f"Visualizations saved to {output_dir}")


# 执行mask创建
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_path = r"H:\PythonAnalysis\learn_batch\2nd_level\run3\yy_vs_kj\clusters_table.csv"

    # 创建masks
    masks, mask_info = create_metaphor_masks(csv_path)

    # 可视化masks（可选）
    print("Creating visualizations...")
    visualize_masks(masks)

    # 打印摘要信息
    print("\n=== Mask Creation Summary ===")
    print(f"Total masks created: {len(masks)}")

    # 统计不同大小的mask数量
    size_categories = {}
    for mask_data in masks.values():
        if isinstance(mask_data, dict):
            category = mask_data['size_category']
            size_categories[category] = size_categories.get(category, 0) + 1

    print("\nMask size distribution:")
    for category, count in size_categories.items():
        print(f"  {category}: {count} masks")

    # 按类型统计
    print("\nMask categories:")
    categories = {}
    for name in masks.keys():
        if '_sphere' in name:
            categories['sphere'] = categories.get('sphere', 0) + 1
        elif '_anatomical' in name:
            categories['anatomical'] = categories.get('anatomical', 0) + 1
        elif '_network' in name:
            categories['network'] = categories.get('network', 0) + 1
        elif 'activation' in name:
            categories['activation'] = categories.get('activation', 0) + 1

    for category, count in categories.items():
        print(f"  {category}: {count} masks")