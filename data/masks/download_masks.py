#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载常用的fMRI脑区mask文件
用于searchlight MVPA分析
统一为 MNI152(91×109×91) 空间
"""

import numpy as np
from nilearn import datasets, image
from nilearn.image import new_img_like, resample_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from pathlib import Path


def create_91x109x91_template():
    """直接创建(91×109×91)空间的模板"""
    print("创建MNI152(91×109×91)目标空间模板...")

    # 使用Harvard-Oxford图谱作为参考，因为它已经是91×109×91
    ho_atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    template_img = ho_atlas.maps

    print(f"目标模板shape: {template_img.shape}")
    return template_img


def resample_to_target_space(source_img, target_img, interpolation='nearest'):
    """将图像重采样到目标空间"""
    return resample_img(source_img,
                        target_affine=target_img.affine,
                        target_shape=target_img.shape[:3],
                        interpolation=interpolation,
                        force_resample=True)


def validate_and_resample_mask(mask_path, target_template, expected_shape=(91, 109, 91)):
    """验证mask维度，如果不正确则重采样"""
    mask_img = image.load_img(str(mask_path))
    current_shape = mask_img.shape[:3]

    if current_shape == expected_shape:
        print(f"  ✓ {mask_path.name} 维度正确: {current_shape}")
        return mask_img
    else:
        print(f"  ⚠ {mask_path.name} 维度不正确: {current_shape} -> 重采样到 {expected_shape}")
        resampled_img = resample_to_target_space(mask_img, target_template, 'nearest')
        # 确保二值化
        resampled_data = resampled_img.get_fdata()
        resampled_data_binary = (resampled_data > 0.5).astype(np.int32)
        resampled_img = new_img_like(target_template, resampled_data_binary)
        resampled_img.to_filename(mask_path)
        print(f"  ✓ 重采样完成: {mask_path.name} -> {resampled_img.shape}")
        return resampled_img


def create_language_masks():
    print("=== 开始创建 MNI152(91×109×91) 空间的mask文件 ===")

    # 创建目标空间模板
    target_template = create_91x109x91_template()
    expected_shape = target_template.shape[:3]

    # 确保目标模板是我们期望的维度
    if expected_shape != (91, 109, 91):
        print(f"警告: 目标模板维度为 {expected_shape}，不是期望的 (91, 109, 91)")
        print("将强制重采样到 (91, 109, 91)")
        # 如果模板不是我们想要的，创建一个空的91x109x91图像作为模板
        empty_data = np.zeros((91, 109, 91), dtype=np.int32)
        target_template = new_img_like(target_template, empty_data)

    # =====================================================
    # ✅ Step 1. 创建 MNI152(91×109×91) 灰质 mask
    # =====================================================
    print("下载并重采样MNI152灰质mask到91×109×91空间...")
    gm_mask_img_orig = datasets.load_mni152_gm_mask(resolution=2)
    gm_mask_img = resample_to_target_space(gm_mask_img_orig, target_template, 'nearest')  # 使用nearest避免插值警告

    # 二值化处理
    gm_data = gm_mask_img.get_fdata()
    gm_data_binary = (gm_data > 0.5).astype(np.int32)
    gm_mask_img = new_img_like(target_template, gm_data_binary)

    gm_mask_img.to_filename("gray_matter_mask_91x109x91.nii.gz")
    print(f"已创建: gray_matter_mask_91x109x91.nii.gz, shape: {gm_mask_img.shape}")

    # =====================================================
    # ✅ Step 2. 使用 Harvard-Oxford 图谱
    # =====================================================
    print("使用Harvard-Oxford皮层图谱...")
    ho_atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = ho_atlas.maps
    labels = ho_atlas.labels
    print(f"图谱shape: {atlas_img.shape}")

    # 确保图谱在目标空间
    if atlas_img.shape[:3] != expected_shape:
        print("重采样Harvard-Oxford图谱到目标空间...")
        atlas_img = resample_to_target_space(atlas_img, target_template, 'nearest')

    atlas_data = atlas_img.get_fdata().astype(np.int32)
    print(f"最终图谱shape: {atlas_img.shape}")

    # =====================================================
    # ✅ Step 3. 创建皮层灰质mask
    # =====================================================
    print("正在创建皮层灰质mask...")
    cortical_data = np.zeros(atlas_data.shape, dtype=np.int32)
    for idx in range(1, 49):  # Harvard-Oxford皮层索引
        cortical_data[atlas_data == idx] = 1

    gm_data = gm_mask_img.get_fdata()
    cortical_gm_data = cortical_data * (gm_data > 0)
    cortical_gm_img = new_img_like(target_template, cortical_gm_data)
    cortical_gm_img.to_filename("cortical_gray_matter_mask.nii.gz")
    print(f"已创建: cortical_gray_matter_mask.nii.gz, shape: {cortical_gm_img.shape}")

    # =====================================================
    # ✅ Step 4. 创建语言与空间网络 mask
    # =====================================================
    brain_regions = {
        "broca_area": [6, 7],
        "wernicke_area": [20, 21],
        "angular_gyrus": [30],
        "middle_temporal_gyrus": [19],
        "precuneus": [15],
        "posterior_parietal_cortex": [14, 30],
    }

    for name, indices in brain_regions.items():
        mask_data = np.isin(atlas_data, indices).astype(np.int32)
        mask_img = new_img_like(target_template, mask_data)
        mask_img.to_filename(f"{name}.nii.gz")
        print(f"已创建: {name}.nii.gz, shape: {mask_img.shape}")

    # 语言网络
    language_indices = [6, 7, 19, 20, 21, 30, 15]
    language_mask = np.isin(atlas_data, language_indices).astype(np.int32)
    language_img = new_img_like(target_template, language_mask)
    language_img.to_filename("language_network_mask.nii.gz")
    print(f"已创建: language_network_mask.nii.gz, shape: {language_img.shape}")

    # 空间网络
    spatial_indices = [14, 15, 30]
    spatial_mask = np.isin(atlas_data, spatial_indices).astype(np.int32)
    spatial_img = new_img_like(target_template, spatial_mask)
    spatial_img.to_filename("spatial_network_mask.nii.gz")
    print(f"已创建: spatial_network_mask.nii.gz, shape: {spatial_img.shape}")

    # =====================================================
    # ✅ Step 5. 严格校验所有生成的mask文件
    # =====================================================
    print("\n" + "=" * 60)
    print("开始严格校验所有mask文件的维度...")
    print("=" * 60)

    mask_files = list(Path(".").glob("*.nii.gz"))
    validation_passed = True

    for mask_file in mask_files:
        try:
            validated_img = validate_and_resample_mask(mask_file, target_template, (91, 109, 91))
            final_shape = validated_img.shape[:3]
            if final_shape != (91, 109, 91):
                print(f"  ❌ {mask_file.name} 校验失败: {final_shape} != (91, 109, 91)")
                validation_passed = False
        except Exception as e:
            print(f"  ❌ {mask_file.name} 校验出错: {e}")
            validation_passed = False

    # =====================================================
    # ✅ Step 6. 最终验证报告
    # =====================================================
    print("\n" + "=" * 60)
    print("最终验证报告:")
    print("=" * 60)

    if validation_passed:
        print("✅ 所有mask文件校验通过!")
        print(f"\n✅ 生成的mask文件列表 (统一空间 91×109×91):")
        for f in Path(".").glob("*.nii.gz"):
            img = image.load_img(str(f))
            print(f"  - {f.name}, shape: {img.shape}")
    else:
        print("❌ 部分mask文件校验失败，请检查上述错误信息")

    return validation_passed


def main():
    print("=== fMRI脑区Mask下载器 (严格校验到91×109×91空间) ===")
    try:
        success = create_language_masks()
        if success:
            print("\n🎉 所有mask创建并校验完成，可直接用于Searchlight分析。")
        else:
            print("\n⚠️ mask创建完成，但存在校验问题，请检查输出。")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()