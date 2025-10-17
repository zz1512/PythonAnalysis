#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Searchlight MVPA分析使用示例
演示如何使用不同的脑区mask进行分析
"""

from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis
import logging

def example_whole_brain_analysis():
    """
    示例1: 全脑searchlight分析
    """
    print("=== 示例1: 全脑Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'whole_brain'
    config.use_custom_mask = False
    
    print(f"分析类型: {config.mask_selection}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    print(f"交叉验证: {config.cv_folds}折")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def example_language_network_analysis():
    """
    示例2: 语言网络searchlight分析
    """
    print("=== 示例2: 语言网络Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'language_network'
    config.use_custom_mask = True
    
    print(f"分析类型: {config.mask_selection}")
    print(f"使用mask: {config.get_selected_mask()}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def example_broca_area_analysis():
    """
    示例3: Broca区searchlight分析
    """
    print("=== 示例3: Broca区Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'broca'
    config.use_custom_mask = True
    
    print(f"分析类型: {config.mask_selection}")
    print(f"使用mask: {config.get_selected_mask()}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def example_spatial_network_analysis():
    """
    示例4: 空间认知网络searchlight分析
    """
    print("=== 示例4: 空间认知网络Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'spatial_network'
    config.use_custom_mask = True
    
    print(f"分析类型: {config.mask_selection}")
    print(f"使用mask: {config.get_selected_mask()}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def example_gray_matter_analysis():
    """
    示例5: 灰质层searchlight分析
    """
    print("=== 示例5: 灰质层Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'gray_matter'
    config.use_custom_mask = True
    
    print(f"分析类型: {config.mask_selection}")
    print(f"使用mask: {config.get_selected_mask()}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def example_cortical_gray_matter_analysis():
    """
    示例6: 皮层灰质searchlight分析
    """
    print("=== 示例6: 皮层灰质Searchlight分析 ===")
    
    config = SearchlightMVPAConfig()
    config.mask_selection = 'cortical_gray_matter'
    config.use_custom_mask = True
    
    print(f"分析类型: {config.mask_selection}")
    print(f"使用mask: {config.get_selected_mask()}")
    print(f"Searchlight半径: {config.searchlight_radius}mm")
    
    # 运行分析
    # results = run_group_searchlight_analysis(config)
    print("配置完成，可运行分析\n")

def show_available_masks():
    """
    显示所有可用的mask选项
    """
    print("=== 可用的脑区Mask ===")
    
    config = SearchlightMVPAConfig()
    
    print("1. 全脑分析:")
    print("   - mask_selection = 'whole_brain'")
    print("   - use_custom_mask = False")
    print()
    
    print("2. 预定义脑区mask:")
    for mask_name, mask_path in config.language_masks.items():
        status = "✅" if mask_path.exists() else "❌"
        print(f"   - {mask_name}: {mask_path.name} {status}")
    print()
    
    print("3. 使用方法:")
    print("   # 方法1: 直接设置")
    print("   config.mask_selection = 'mask_name'  # 选择特定mask")
    print("   config.use_custom_mask = True        # 启用自定义mask")
    print("   ")
    print("   # 方法2: 使用便捷方法 (推荐)")
    print("   config.set_mask('mask_name')         # 自动设置mask和相关参数")
    print("   ")
    print("   # 验证mask可用性")
    print("   is_valid, info = config.validate_mask_availability()")
    print("   available_masks = config.get_available_masks()")
    print()

def example_optimized_mask_usage():
    """
    示例7: 优化的mask使用方法演示
    """
    print("\n=== 示例7: 优化的Mask使用方法 ===")
    
    config = SearchlightMVPAConfig()
    
    print("1. 检查可用的mask:")
    available_masks = config.get_available_masks()
    print(f"   可用mask: {available_masks}")
    
    print("\n2. 使用便捷方法设置mask:")
    try:
        config.set_mask('broca')
        print(f"   成功设置Broca区mask")
        print(f"   当前配置: mask_selection={config.mask_selection}, use_custom_mask={config.use_custom_mask}")
    except ValueError as e:
        print(f"   设置失败: {e}")
    
    print("\n3. 验证mask配置:")
    is_valid, mask_info = config.validate_mask_availability()
    print(f"   验证结果: {mask_info}")
    print(f"   配置有效: {is_valid}")
    
    print("\n4. 切换到灰质层分析:")
    config.set_mask('gray_matter')
    is_valid, mask_info = config.validate_mask_availability()
    print(f"   新配置: {mask_info}")
    
    print("\n5. 尝试设置不存在的mask:")
    try:
        config.set_mask('nonexistent_mask')
    except ValueError as e:
        print(f"   预期错误: {e}")
    
    print("\n配置完成，可运行分析\n")

def main():
    """
    运行所有示例
    """
    print("🧠 Searchlight MVPA分析示例")
    print("=" * 50)
    
    # 显示可用mask
    show_available_masks()
    
    # 运行各种分析示例
    example_whole_brain_analysis()
    example_language_network_analysis()
    example_broca_area_analysis()
    example_spatial_network_analysis()
    example_gray_matter_analysis()
    example_cortical_gray_matter_analysis()
    example_optimized_mask_usage()
    
    print("💡 提示:")
    print("1. 取消注释 'results = run_group_searchlight_analysis(config)' 来实际运行分析")
    print("2. 确保LSS数据已准备好在 'data/lss_trial_data' 目录")
    print("3. 结果将保存在 'results/searchlight_mvpa' 目录")
    print("4. 可以根据需要调整searchlight半径、交叉验证折数等参数")
    print("5. 推荐使用 config.set_mask() 方法来设置mask，更加安全和便捷")

if __name__ == "__main__":
    main()