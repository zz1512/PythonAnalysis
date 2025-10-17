#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的mask使用方法
"""

from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig

def test_mask_optimization():
    """
    测试mask使用方法的优化功能
    """
    print("🧪 测试优化后的Mask使用方法")
    print("=" * 40)
    
    config = SearchlightMVPAConfig()
    
    # 测试1: 获取可用mask列表
    print("\n1. 测试获取可用mask列表:")
    available_masks = config.get_available_masks()
    print(f"   可用mask数量: {len(available_masks)}")
    print(f"   mask列表: {available_masks[:5]}..." if len(available_masks) > 5 else f"   mask列表: {available_masks}")
    
    # 测试2: 便捷设置方法
    print("\n2. 测试便捷设置方法:")
    test_masks = ['whole_brain', 'broca', 'gray_matter']
    
    for mask_name in test_masks:
        try:
            config.set_mask(mask_name)
            is_valid, info = config.validate_mask_availability()
            status = "✅" if is_valid else "❌"
            print(f"   {mask_name}: {status} {info}")
        except ValueError as e:
            print(f"   {mask_name}: ❌ {e}")
    
    # 测试3: 错误处理
    print("\n3. 测试错误处理:")
    try:
        config.set_mask('nonexistent_mask')
        print("   ❌ 应该抛出错误但没有")
    except ValueError as e:
        print(f"   ✅ 正确捕获错误: {str(e)[:50]}...")
    
    # 测试4: 验证功能
    print("\n4. 测试验证功能:")
    config.set_mask('language_network')
    is_valid, info = config.validate_mask_availability()
    print(f"   语言网络mask验证: {'✅' if is_valid else '❌'} {info}")
    
    # 测试5: 全脑分析配置
    print("\n5. 测试全脑分析配置:")
    config.set_mask('whole_brain')
    selected_mask = config.get_selected_mask()
    print(f"   全脑分析mask路径: {selected_mask if selected_mask else 'None (正确)'}")
    print(f"   use_custom_mask: {config.use_custom_mask} (应该为False)")
    
    print("\n🎉 所有测试完成!")
    print("\n💡 优化要点:")
    print("   - 使用 config.set_mask() 方法更安全便捷")
    print("   - 自动验证mask文件存在性")
    print("   - 提供详细的错误信息和可用选项")
    print("   - 统一的mask选择逻辑")

if __name__ == "__main__":
    test_mask_optimization()