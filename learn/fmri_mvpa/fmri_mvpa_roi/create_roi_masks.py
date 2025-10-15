import numpy as np
from pathlib import Path
from nilearn import image, datasets
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 配置参数 ==========
ROI_OUTPUT_DIR = Path(r"H:\PythonAnalysis\learn_mvpa\ROI_masks")
ROI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 根据隐喻加工的神经机制研究，重新定义关键脑区
# 基于Harvard-Oxford图谱的实际标签名称
ROI_DEFINITIONS = {
    # 额下回 - 语义处理和隐喻理解的关键区域
    'IFG_tri_L': {'atlas': 'HO', 'label': 'Inferior Frontal Gyrus, pars triangularis', 'index': 5},
    'IFG_tri_R': {'atlas': 'HO', 'label': 'Inferior Frontal Gyrus, pars triangularis', 'index': 5},
    'IFG_oper_L': {'atlas': 'HO', 'label': 'Inferior Frontal Gyrus, pars opercularis', 'index': 6},
    'IFG_oper_R': {'atlas': 'HO', 'label': 'Inferior Frontal Gyrus, pars opercularis', 'index': 6},

    # 颞中回 - 语义记忆和隐喻整合
    'MTG_ant_L': {'atlas': 'HO', 'label': 'Middle Temporal Gyrus, anterior division', 'index': 11},
    'MTG_ant_R': {'atlas': 'HO', 'label': 'Middle Temporal Gyrus, anterior division', 'index': 11},
    'MTG_post_L': {'atlas': 'HO', 'label': 'Middle Temporal Gyrus, posterior division', 'index': 12},
    'MTG_post_R': {'atlas': 'HO', 'label': 'Middle Temporal Gyrus, posterior division', 'index': 12},

    # 角回 - 隐喻的语义整合和抽象思维
    'AG_L': {'atlas': 'HO', 'label': 'Angular Gyrus', 'index': 21},
    'AG_R': {'atlas': 'HO', 'label': 'Angular Gyrus', 'index': 21},

    # 额上回 - 执行功能和抽象推理
    'SFG_L': {'atlas': 'HO', 'label': 'Superior Frontal Gyrus', 'index': 3},
    'SFG_R': {'atlas': 'HO', 'label': 'Superior Frontal Gyrus', 'index': 3},

    # 额外添加：缘上回 - 语义处理和隐喻理解
    'SMG_ant_L': {'atlas': 'HO', 'label': 'Supramarginal Gyrus, anterior division', 'index': 19},
    'SMG_ant_R': {'atlas': 'HO', 'label': 'Supramarginal Gyrus, anterior division', 'index': 19},
    'SMG_post_L': {'atlas': 'HO', 'label': 'Supramarginal Gyrus, posterior division', 'index': 20},
    'SMG_post_R': {'atlas': 'HO', 'label': 'Supramarginal Gyrus, posterior division', 'index': 20},
}


def log(message):
    logging.info(message)


def fetch_ho_atlas():
    """获取Harvard-Oxford图谱"""
    logging.info("加载Harvard-Oxford皮质图谱...")
    try:
        # 获取Harvard-Oxford皮质概率图谱
        ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

        logging.info(f"HO图谱文件: {ho_atlas.maps}")
        logging.info(f"HO标签数量: {len(ho_atlas.labels)}")

        # 加载图谱图像
        atlas_img = image.load_img(ho_atlas.maps)
        atlas_data = atlas_img.get_fdata()

        logging.info(f"HO图谱形状: {atlas_img.shape}")
        logging.info(f"HO数据范围: [{np.min(atlas_data)}, {np.max(atlas_data)}]")
        logging.info(f"HO数据唯一值: {np.unique(atlas_data)}")

        return ho_atlas, atlas_img
    except Exception as e:
        logging.error(f"加载Harvard-Oxford图谱失败: {e}")
        return None, None


def create_roi_mask_ho(roi_name, roi_def, ho_atlas, atlas_img):
    """使用Harvard-Oxford图谱创建ROI掩模"""
    try:
        label_name = roi_def['label']
        target_index = roi_def['index']

        logging.info(f"创建ROI: {roi_name} ({label_name}, 索引: {target_index})")

        # 获取图谱数据
        atlas_data = atlas_img.get_fdata()

        # 创建二值掩模 (HO图谱中每个体素的值就是区域索引)
        roi_data = (atlas_data == target_index).astype(np.int32)
        voxel_count = np.sum(roi_data)

        logging.info(f"初始体素数: {voxel_count}")

        if voxel_count == 0:
            logging.warning(f"警告: ROI '{roi_name}' 初始体素数为0!")
            return None

        # 分离左右半球
        # 在MNI空间中，左半球x坐标<0，右半球x坐标>0
        affine = atlas_img.affine
        shape = atlas_img.shape

        # 创建坐标网格
        i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        # 将体素坐标转换为世界坐标（MNI坐标）
        x_coords = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]

        # 根据ROI名称确定半球
        if roi_name.endswith('_L'):
            # 左半球：x < 0
            hemisphere_mask = x_coords < 0
            logging.info("选择左半球")
        elif roi_name.endswith('_R'):
            # 右半球：x > 0
            hemisphere_mask = x_coords > 0
            logging.info("选择右半球")
        else:
            # 双侧
            hemisphere_mask = np.ones_like(roi_data, dtype=bool)
            logging.info("选择双侧")

        # 应用半球掩模
        roi_data_hemisphere = roi_data * hemisphere_mask
        final_voxel_count = np.sum(roi_data_hemisphere)

        logging.info(f"应用半球分离后体素数: {final_voxel_count}")

        if final_voxel_count == 0:
            logging.warning(f"警告: 应用半球分离后ROI '{roi_name}' 体素数为0!")
            return None

        # 创建新的图像
        roi_img = image.new_img_like(atlas_img, roi_data_hemisphere)

        logging.info(f"成功创建ROI: {roi_name} - 最终体素数: {final_voxel_count}")
        return roi_img

    except Exception as e:
        logging.error(f"创建ROI失败 {roi_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def create_all_roi_masks():
    """创建所有ROI掩模"""
    logging.info("开始创建ROI掩模...")

    # 加载Harvard-Oxford图谱
    ho_atlas, atlas_img = fetch_ho_atlas()
    if ho_atlas is None:
        logging.error("无法加载Harvard-Oxford图谱，终止程序")
        return {}

    created_masks = {}
    creation_report = []

    for roi_name, roi_def in ROI_DEFINITIONS.items():
        roi_mask = create_roi_mask_ho(roi_name, roi_def, ho_atlas, atlas_img)

        if roi_mask is not None:
            # 保存掩模文件
            output_path = ROI_OUTPUT_DIR / f"ROI_{roi_name}.nii.gz"
            roi_mask.to_filename(output_path)
            created_masks[roi_name] = output_path

            # 记录创建信息
            mask_data = roi_mask.get_fdata()
            n_voxels = np.sum(mask_data > 0)
            creation_report.append({
                'roi_name': roi_name,
                'file_path': str(output_path),
                'n_voxels': n_voxels,
                'status': '成功'
            })
            logging.info(f"保存ROI: {roi_name} -> {output_path} ({n_voxels}个体素)")
        else:
            creation_report.append({
                'roi_name': roi_name,
                'file_path': 'N/A',
                'n_voxels': 0,
                'status': '失败'
            })
            logging.error(f"失败: {roi_name}")

    # 保存创建报告
    import pandas as pd
    report_df = pd.DataFrame(creation_report)
    report_df.to_csv(ROI_OUTPUT_DIR / "roi_creation_report.csv", index=False, encoding='utf-8-sig')

    # 创建ROI路径配置文件
    import json
    roi_config = {
        'roi_directory': str(ROI_OUTPUT_DIR),
        'available_rois': list(created_masks.keys()),
        'atlas_used': 'Harvard-Oxford Cortical Atlas (cort-maxprob-thr25-2mm)',
        'roi_descriptions': {
            'IFG_tri': '额下回三角部 - 语义选择和隐喻理解',
            'IFG_oper': '额下回眶部 - 语义处理和句法整合',
            'MTG_ant': '颞中回前部 - 语义记忆检索',
            'MTG_post': '颞中回后部 - 语义整合',
            'AG': '角回 - 隐喻语义整合和抽象思维',
            'SFG': '额上回 - 执行功能和抽象推理',
            'SMG_ant': '缘上回前部 - 语义处理',
            'SMG_post': '缘上回后部 - 语义理解'
        }
    }
    with open(ROI_OUTPUT_DIR / "roi_config.json", 'w') as f:
        json.dump(roi_config, f, indent=2, ensure_ascii=False)

    logging.info(f"\nROI掩模创建完成!")
    logging.info(f"成功创建: {len(created_masks)}/{len(ROI_DEFINITIONS)} 个ROI")
    logging.info(f"输出目录: {ROI_OUTPUT_DIR}")

    return created_masks


def visualize_roi_masks(roi_masks):
    """可视化创建的ROI掩模"""
    try:
        logging.info("生成ROI可视化...")

        # 为每个ROI生成单独的图像
        for roi_name, mask_path in roi_masks.items():
            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                display_mode = ['ortho', 'z', 'x']
                titles = [f'{roi_name} - Ortho', f'{roi_name} - Axial', f'{roi_name} - Sagittal']

                for i, (mode, title) in enumerate(zip(display_mode, titles)):
                    plot_roi(
                        mask_path,
                        title=title,
                        display_mode=mode,
                        axes=axes[i],
                        annotate=False,
                        cmap='coolwarm'
                    )

                plt.tight_layout()
                plt.savefig(ROI_OUTPUT_DIR / f"ROI_{roi_name}_visualization.png",
                            dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"生成可视化: {roi_name}")

            except Exception as e:
                logging.error(f"可视化 {roi_name} 失败: {e}")

        # 创建所有ROI的叠加图
        if len(roi_masks) > 0:
            try:
                all_masks = list(roi_masks.values())
                plot_roi(
                    all_masks,
                    title="All Metaphor ROIs - Harvard-Oxford Atlas",
                    display_mode='ortho',
                    output_file=ROI_OUTPUT_DIR / "all_rois_overlay.png",
                    cmap='Set3'
                )
                logging.info("生成所有ROI叠加图")
            except Exception as e:
                logging.error(f"创建叠加图失败: {e}")

        logging.info("ROI可视化完成")

    except Exception as e:
        logging.error(f"可视化失败: {e}")


def check_roi_quality(roi_masks):
    """检查ROI质量"""
    log("\n检查ROI质量...")

    quality_report = []

    for roi_name, mask_path in roi_masks.items():
        roi_img = image.load_img(mask_path)
        roi_data = roi_img.get_fdata()

        n_voxels = np.sum(roi_data > 0)
        voxel_size = np.prod(roi_img.header.get_zooms())  # 体素体积 (mm³)
        volume_mm3 = n_voxels * voxel_size

        quality_report.append({
            'roi_name': roi_name,
            'n_voxels': n_voxels,
            'voxel_volume_mm3': voxel_size,
            'total_volume_mm3': volume_mm3,
            'quality': 'Good' if n_voxels >= 50 else 'Warning'
        })

        status = "✓" if n_voxels >= 50 else "⚠"
        logging.info(f"{status} {roi_name}: {n_voxels} 体素, {volume_mm3:.2f} mm³")

        if n_voxels < 10:
            logging.warning(f"警告: {roi_name} 体素数量较少，可能影响后续分析")

    # 保存质量报告
    import pandas as pd
    quality_df = pd.DataFrame(quality_report)
    quality_df.to_csv(ROI_OUTPUT_DIR / "roi_quality_report.csv", index=False, encoding='utf-8-sig')

    return quality_report


if __name__ == "__main__":
    # 创建所有ROI掩模
    created_masks = create_all_roi_masks()

    # 检查ROI质量
    if created_masks:
        check_roi_quality(created_masks)
        visualize_roi_masks(created_masks)
    else:
        logging.error("没有成功创建任何ROI掩模")

    logging.info(f"\nROI掩模创建流程完成!")
    print(f"请检查目录: {ROI_OUTPUT_DIR}")
    if created_masks:
        print(f"成功创建的ROI: {list(created_masks.keys())}")