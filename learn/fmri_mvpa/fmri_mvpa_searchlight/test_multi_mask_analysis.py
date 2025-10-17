#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Mask分析功能测试脚本
测试多mask分析的各种功能和边界情况
"""

import sys
import traceback
from pathlib import Path
from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig

def test_basic_multi_mask_setup():
    """
    测试基本的多mask设置功能
    """
    print("\n🧪 测试1: 基本多mask设置")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 测试设置mask列表
        mask_list = ['gray_matter', 'language_network', 'broca']
        config.set_mask(mask_list)
        
        print(f"✅ 成功设置mask列表: {config.mask_list}")
        print(f"✅ 多mask模式已启用: {config.use_mask_list}")
        
        # 测试获取mask路径
        selected_masks = config.get_selected_mask()
        print(f"✅ 获取的mask路径: {len(selected_masks) if isinstance(selected_masks, list) else 1}个")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_mask_validation():
    """
    测试mask验证功能
    """
    print("\n🧪 测试2: Mask验证功能")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 测试有效mask列表
        valid_masks = ['whole_brain', 'gray_matter']
        config.set_mask(valid_masks)
        
        is_valid, info = config.validate_mask_availability()
        print(f"✅ 有效mask验证: {info}")
        
        # 测试包含无效mask的列表
        config = SearchlightMVPAConfig()
        mixed_masks = ['gray_matter', 'nonexistent_mask', 'whole_brain']
        
        try:
            config.set_mask(mixed_masks)
            print("❌ 应该抛出异常但没有")
            return False
        except ValueError as e:
            print(f"✅ 正确捕获无效mask错误: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_single_vs_multi_mask():
    """
    测试单mask和多mask模式的切换
    """
    print("\n🧪 测试3: 单mask vs 多mask模式")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 测试单mask模式
        config.set_mask('gray_matter')
        print(f"✅ 单mask模式: use_mask_list={config.use_mask_list}")
        print(f"✅ 当前mask: {config.mask_selection}")
        
        # 切换到多mask模式
        config.set_mask(['gray_matter', 'language_network'])
        print(f"✅ 多mask模式: use_mask_list={config.use_mask_list}")
        print(f"✅ mask列表: {config.mask_list}")
        
        # 再切换回单mask模式
        config.set_mask('broca')
        print(f"✅ 切换回单mask: use_mask_list={config.use_mask_list}")
        print(f"✅ 当前mask: {config.mask_selection}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_edge_cases():
    """
    测试边界情况
    """
    print("\n🧪 测试4: 边界情况")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 测试空列表
        try:
            config.set_mask([])
            print("❌ 空列表应该抛出异常")
            return False
        except ValueError as e:
            print(f"✅ 正确处理空列表: {e}")
        
        # 测试单元素列表
        config.set_mask(['gray_matter'])
        print(f"✅ 单元素列表处理: use_mask_list={config.use_mask_list}")
        
        # 测试重复mask
        config.set_mask(['gray_matter', 'gray_matter', 'language_network'])
        print(f"✅ 重复mask处理: {config.mask_list}")
        
        # 测试None输入
        try:
            config.set_mask(None)
            print("❌ None输入应该抛出异常")
            return False
        except (ValueError, TypeError) as e:
            print(f"✅ 正确处理None输入: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_get_available_masks():
    """
    测试获取可用mask列表
    """
    print("\n🧪 测试5: 获取可用mask")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        available_masks = config.get_available_masks()
        print(f"✅ 可用mask数量: {len(available_masks)}")
        print(f"✅ 可用mask列表: {available_masks[:5]}...")  # 只显示前5个
        
        # 验证预定义mask是否在列表中
        expected_masks = ['whole_brain', 'gray_matter', 'language_network']
        for mask in expected_masks:
            if mask in available_masks:
                print(f"✅ 找到预期mask: {mask}")
            else:
                print(f"⚠️ 未找到预期mask: {mask}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_mask_path_generation():
    """
    测试mask路径生成
    """
    print("\n🧪 测试6: Mask路径生成")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 测试多mask路径生成
        mask_list = ['whole_brain', 'gray_matter', 'language_network']
        config.set_mask(mask_list)
        
        selected_masks = config.get_selected_mask()
        print(f"✅ 生成的路径数量: {len(selected_masks)}")
        
        for i, (mask_name, mask_path) in enumerate(zip(mask_list, selected_masks)):
            if mask_name == 'whole_brain':
                if mask_path is None:
                    print(f"✅ 全脑mask正确处理: {mask_name} -> None")
                else:
                    print(f"⚠️ 全脑mask路径异常: {mask_name} -> {mask_path}")
            else:
                if mask_path and Path(mask_path).name.startswith(mask_name.split('_')[0]):
                    print(f"✅ mask路径正确: {mask_name} -> {Path(mask_path).name}")
                else:
                    print(f"⚠️ mask路径可能有问题: {mask_name} -> {mask_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_config_consistency():
    """
    测试配置一致性
    """
    print("\n🧪 测试7: 配置一致性")
    print("-" * 40)
    
    try:
        config = SearchlightMVPAConfig()
        
        # 设置多mask并检查所有相关属性
        mask_list = ['gray_matter', 'language_network']
        config.set_mask(mask_list)
        
        # 检查属性一致性
        checks = [
            (config.use_mask_list == True, "use_mask_list应为True"),
            (config.mask_list == mask_list, "mask_list应匹配输入"),
            (config.use_custom_mask == True, "use_custom_mask应为True"),
            (isinstance(config.get_selected_mask(), list), "get_selected_mask应返回列表")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """
    运行所有测试
    """
    print("🚀 开始多Mask分析功能测试")
    print("=" * 50)
    
    tests = [
        test_basic_multi_mask_setup,
        test_mask_validation,
        test_single_vs_multi_mask,
        test_edge_cases,
        test_get_available_masks,
        test_mask_path_generation,
        test_config_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 发生异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！多mask分析功能正常工作")
        return True
    else:
        print(f"⚠️ {total - passed} 个测试失败，请检查相关功能")
        return False

def main():
    """
    主函数
    """
    try:
        success = run_all_tests()
        
        print("\n💡 使用提示:")
        print("   - 使用 config.set_mask(['mask1', 'mask2']) 设置多mask")
        print("   - 使用 config.validate_mask_availability() 验证配置")
        print("   - 多mask结果会自动添加mask名称后缀")
        print("   - 参考 example_multi_mask_usage.py 获取更多示例")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())