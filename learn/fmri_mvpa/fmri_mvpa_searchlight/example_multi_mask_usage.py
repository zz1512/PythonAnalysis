#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Mask分析使用示例
演示如何使用mask列表进行多个脑区的同时分析
"""

from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis

def example_multi_mask_analysis():
    """
    示例: 多mask分析 - 同时分析灰质和语言网络
    """
    print("🧠 示例: 多Mask分析")
    print("=" * 40)
    
    config = SearchlightMVPAConfig()
    
    # 设置多个mask进行分析
    mask_list = ['gray_matter', 'language_network', 'broca']
    config.set_mask(mask_list)
    
    print(f"设置的mask列表: {config.mask_list}")
    print(f"使用mask列表模式: {config.use_mask_list}")
    
    # 验证配置
    is_valid, info = config.validate_mask_availability()
    print(f"配置验证: {info}")
    
    if is_valid:
        print("\n开始多mask分析...")
        # 注意: 这里只是演示配置，实际运行需要真实的数据
        # results = run_group_searchlight_analysis(config)
        print("配置成功，可以开始分析")
        
        # 显示预期的结果结构
        print("\n预期结果结构:")
        for contrast_name, _, _ in config.contrasts:
            for mask_name in mask_list:
                result_key = f"{contrast_name}_{mask_name}"
                print(f"  - {result_key}")
    else:
        print(f"配置验证失败: {info}")

def example_language_regions_analysis():
    """
    示例: 语言相关脑区分析
    """
    print("\n🗣️ 示例: 语言相关脑区分析")
    print("=" * 40)
    
    config = SearchlightMVPAConfig()
    
    # 设置语言相关的多个脑区
    language_masks = ['broca', 'wernicke', 'angular_gyrus', 'language_network']
    config.set_mask(language_masks)
    
    print(f"语言脑区列表: {language_masks}")
    
    # 验证配置
    is_valid, info = config.validate_mask_availability()
    print(f"配置验证: {info}")
    
    if is_valid:
        print("\n语言脑区分析配置成功")
        print("将为每个语言脑区生成独立的分析结果")

def example_comparison_analysis():
    """
    示例: 对比分析 - 全脑 vs 特定脑区
    """
    print("\n⚖️ 示例: 对比分析")
    print("=" * 40)
    
    config = SearchlightMVPAConfig()
    
    # 设置对比分析的mask列表
    comparison_masks = ['whole_brain', 'gray_matter', 'language_network']
    config.set_mask(comparison_masks)
    
    print(f"对比分析mask: {comparison_masks}")
    
    # 验证配置
    is_valid, info = config.validate_mask_availability()
    print(f"配置验证: {info}")
    
    if is_valid:
        print("\n对比分析配置成功")
        print("可以比较全脑、灰质和语言网络的分析结果")

def example_error_handling():
    """
    示例: 错误处理
    """
    print("\n❌ 示例: 错误处理")
    print("=" * 40)
    
    config = SearchlightMVPAConfig()
    
    # 测试无效mask
    try:
        invalid_masks = ['gray_matter', 'nonexistent_mask', 'language_network']
        config.set_mask(invalid_masks)
        print("不应该到达这里")
    except ValueError as e:
        print(f"正确捕获错误: {e}")
    
    # 测试空列表
    try:
        config.set_mask([])
        print("不应该到达这里")
    except ValueError as e:
        print(f"正确捕获空列表错误: {e}")
    
    # 测试部分有效的mask列表
    print("\n测试部分有效的mask:")
    config = SearchlightMVPAConfig()
    # 假设某些mask文件不存在，这里只是演示验证逻辑
    valid_masks = ['whole_brain', 'gray_matter']
    config.set_mask(valid_masks)
    is_valid, info = config.validate_mask_availability()
    print(f"部分验证结果: {info}")

def show_multi_mask_usage_guide():
    """
    显示多mask使用指南
    """
    print("\n📖 多Mask使用指南")
    print("=" * 40)
    
    print("""
    1. 基本用法:
       config.set_mask(['gray_matter', 'language_network'])
    
    2. 支持的mask类型:
       - 'whole_brain': 全脑分析
       - 'gray_matter': 灰质层
       - 'language_network': 语言网络
       - 'broca': 布洛卡区
       - 'wernicke': 韦尼克区
       - 等等...
    
    3. 结果文件命名:
       - 单mask: contrast_name_accuracy.nii.gz
       - 多mask: contrast_name_maskname_accuracy.nii.gz
    
    4. 结果字典键名:
       - 单mask: contrast_name
       - 多mask: contrast_name_maskname
    
    5. 优势:
       - 一次运行，多个脑区结果
       - 便于比较不同脑区的分类效果
       - 节省计算时间和存储空间
    """)

def main():
    """
    主函数 - 运行所有示例
    """
    print("🚀 多Mask Searchlight MVPA分析示例")
    print("=" * 50)
    
    # 运行各种示例
    example_multi_mask_analysis()
    example_language_regions_analysis()
    example_comparison_analysis()
    example_error_handling()
    show_multi_mask_usage_guide()
    
    print("\n✅ 所有示例演示完成!")
    print("\n💡 提示:")
    print("   - 使用config.set_mask(mask_list)设置多个mask")
    print("   - 每个mask会生成独立的分析结果")
    print("   - 结果文件会自动添加mask名称后缀")
    print("   - 支持任意数量的mask组合分析")

if __name__ == "__main__":
    main()