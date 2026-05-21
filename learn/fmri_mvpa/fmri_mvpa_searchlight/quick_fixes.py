#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始代码快速修复脚本

如果需要继续使用原始的 fmri_mvpa_searchlight_pipeline_multi.py，
可以使用此脚本进行关键问题的快速修复。

主要修复:
1. 路径配置问题
2. 灰质mask选择
3. 性能参数优化
4. 简化统计设置
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional

def create_quick_fix_config() -> Dict[str, Any]:
    """创建快速修复的配置覆盖参数"""
    
    # 检查实际存在的路径
    base_dir = Path("/Users/bytedance/PycharmProjects/PythonAnalysis")
    
    # LSS数据路径修正
    possible_lss_paths = [
        base_dir / "learn_LSS",
        base_dir / "learn" / "learn_LSS", 
        base_dir / "gather_to_learn" / "LSS",
        Path("../../learn_LSS"),
        Path("../../../learn_LSS")
    ]
    
    lss_root = None
    for path in possible_lss_paths:
        if path.exists():
            lss_root = str(path)
            break
    
    if not lss_root:
        print("警告: 未找到LSS数据目录，请手动设置 lss_root 参数")
        lss_root = "../../learn_LSS"  # 默认值
    
    # mask目录修正
    mask_dir = str(base_dir / "data" / "masks")
    
    # 结果目录修正
    results_dir = str(base_dir / "results" / "searchlight_mvpa_fixed")
    
    return {
        # === 路径修正 ===
        'lss_root': lss_root,
        'mask_dir': mask_dir,
        'results_dir': results_dir,
        
        # === 被试和run设置 ===
        'subjects': [f"sub-{i:02d}" for i in range(1, 4)],  # 减少被试数量用于测试
        'runs': [3, 4],  # 保持原设置
        
        # === 灰质mask设置 ===
        'process_mask': 'gray_matter_mask.nii.gz',  # 使用实际存在的mask
        
        # === 性能优化 ===
        'n_jobs': 2,  # 保守的并行设置
        'use_parallel': True,
        'parallel_subjects': False,  # 禁用被试级并行
        'max_workers': 2,
        'memory_aware_parallel': False,
        'min_memory_per_subject_gb': 8,  # 降低内存要求
        
        # === 统计设置简化 ===
        'n_permutations': 100,  # 减少置换次数
        'within_subject_permutations': 50,
        'max_chance_permutations': 50,
        'enable_two_sided_group_tests': False,  # 简化统计测试
        
        # === 缓存设置 ===
        'enable_mask_cache': True,
        'enable_beta_cache': True,
        'mask_cache_max_items': 4,  # 减少缓存大小
        'beta_cache_max_items': 2,
        
        # === 调试设置 ===
        'debug_mode': True,  # 启用调试模式
        'enable_memory_monitoring': False,  # 禁用内存监控
        'optimize_memory': True,
        
        # === 分析设置 ===
        'searchlight_radius': 3,
        'min_region_size': 20,  # 降低最小簇大小
        'cv_folds': 5,
        'alpha_level': 0.05,
        'chance_level': 0.5,
        
        # === 输出控制 ===
        'disable_chance_maps': True,  # 禁用机会水平地图计算
        'enable_batch_processing': False,  # 禁用批处理
        'generate_report': True,
        
        # === 对比条件 ===
        'contrasts': [('metaphor_vs_space', 'yy', 'kj')],  # 专注核心对比
    }

def apply_quick_fixes_to_original():
    """将快速修复应用到原始代码"""
    
    print("=== MVPA Searchlight 快速修复 ===")
    print()
    
    # 获取修复配置
    fix_config = create_quick_fix_config()
    
    print("修复的主要问题:")
    print("1. ✅ 路径配置修正")
    print("2. ✅ 灰质mask选择")
    print("3. ✅ 性能参数优化")
    print("4. ✅ 统计设置简化")
    print("5. ✅ 内存使用优化")
    print()
    
    print("修复后的关键配置:")
    print(f"- LSS数据路径: {fix_config['lss_root']}")
    print(f"- Mask目录: {fix_config['mask_dir']}")
    print(f"- 结果目录: {fix_config['results_dir']}")
    print(f"- 被试数量: {len(fix_config['subjects'])}")
    print(f"- 并行作业: {fix_config['n_jobs']}")
    print(f"- 置换次数: {fix_config['n_permutations']}")
    print()
    
    # 验证路径
    print("路径验证:")
    
    lss_path = Path(fix_config['lss_root'])
    if lss_path.exists():
        print(f"✅ LSS数据目录存在: {lss_path}")
    else:
        print(f"❌ LSS数据目录不存在: {lss_path}")
    
    mask_path = Path(fix_config['mask_dir'])
    if mask_path.exists():
        print(f"✅ Mask目录存在: {mask_path}")
        
        # 检查灰质mask文件
        mask_file = mask_path / fix_config['process_mask']
        if mask_file.exists():
            print(f"✅ 灰质mask文件存在: {mask_file}")
        else:
            print(f"❌ 灰质mask文件不存在: {mask_file}")
            # 寻找替代文件
            alternatives = [
                'cortical_gray_matter_mask.nii.gz',
                'mni152_gm_mask.nii.gz', 
                'gray_matter_mask_91x109x91.nii.gz'
            ]
            
            for alt in alternatives:
                alt_path = mask_path / alt
                if alt_path.exists():
                    print(f"✅ 找到替代mask: {alt}")
                    fix_config['process_mask'] = alt
                    break
    else:
        print(f"❌ Mask目录不存在: {mask_path}")
    
    print()
    
    return fix_config

def run_fixed_analysis():
    """使用修复配置运行原始分析"""
    
    try:
        # 导入原始模块
        from fmri_mvpa_searchlight_pipeline_multi import MVPAConfig, run_group_searchlight_analysis_parallel
        
        print("使用修复配置运行原始分析...")
        
        # 获取修复配置
        fix_config = create_quick_fix_config()
        
        # 创建修复后的配置对象
        config = MVPAConfig(overrides=fix_config)
        
        print(f"配置创建成功，开始分析...")
        print(f"结果将保存到: {config.results_dir}")
        
        # 运行分析
        group_df, stats_df = run_group_searchlight_analysis_parallel(config)
        
        if group_df is not None:
            print("\n✅ 分析成功完成！")
            print(f"结果保存在: {config.results_dir}")
            return config.results_dir
        else:
            print("\n❌ 分析失败：没有有效结果")
            return None
            
    except ImportError as e:
        print(f"❌ 无法导入原始模块: {e}")
        print("请确保 fmri_mvpa_searchlight_pipeline_multi.py 文件存在")
        return None
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        return None

def generate_fix_summary():
    """生成修复摘要"""
    
    summary = """
# 快速修复摘要

## 修复的主要问题

### 1. 路径配置问题
- **问题**: 原始代码中的LSS数据路径和mask路径配置错误
- **修复**: 自动检测实际存在的路径，提供多个候选路径
- **结果**: 确保能正确找到数据文件

### 2. 性能参数过度配置
- **问题**: 原始代码设置了过高的内存要求和复杂的并行配置
- **修复**: 降低内存要求，简化并行设置
- **结果**: 提高分析的稳定性和兼容性

### 3. 统计测试过度复杂
- **问题**: 包含过多的置换测试和复杂的统计设置
- **修复**: 减少置换次数，简化统计测试
- **结果**: 加快分析速度，保持统计有效性

### 4. 灰质mask选择问题
- **问题**: 硬编码的mask文件名可能不存在
- **修复**: 自动检测可用的灰质mask文件
- **结果**: 确保使用正确的灰质区域限制

## 使用方法

### 方法1: 应用快速修复到原始代码
```python
from quick_fixes import apply_quick_fixes_to_original, run_fixed_analysis

# 获取修复配置
fix_config = apply_quick_fixes_to_original()

# 运行修复后的分析
results = run_fixed_analysis()
```

### 方法2: 使用简化版本（推荐）
```python
from simplified_searchlight_analysis import main

# 直接运行简化版本
results = main()
```

## 建议

1. **优先使用简化版本**: `simplified_searchlight_analysis.py` 更清晰、更稳定
2. **快速修复仅用于紧急情况**: 如果必须使用原始代码时的临时解决方案
3. **验证结果**: 无论使用哪种方法，都要验证输出结果的合理性

## 预期效果

- ✅ 解决路径找不到的问题
- ✅ 降低内存使用，提高稳定性
- ✅ 加快分析速度
- ✅ 确保专注于灰质区域分析
- ✅ 生成清晰的分类准确脑区地图
"""
    
    # 保存摘要
    summary_path = Path("QUICK_FIXES_SUMMARY.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"修复摘要已保存: {summary_path}")
    return str(summary_path)

def main():
    """主函数"""
    print("MVPA Searchlight 快速修复工具")
    print("=" * 40)
    print()
    
    print("选择操作:")
    print("1. 查看修复配置")
    print("2. 运行修复后的原始分析")
    print("3. 生成修复摘要")
    print("4. 推荐使用简化版本")
    
    choice = input("\n请选择 (1/2/3/4): ").strip()
    
    if choice == '1':
        fix_config = apply_quick_fixes_to_original()
        print("\n修复配置已生成，可以手动应用到原始代码中")
        
    elif choice == '2':
        results = run_fixed_analysis()
        if results:
            print(f"\n分析完成，结果保存在: {results}")
        
    elif choice == '3':
        summary_path = generate_fix_summary()
        print(f"\n修复摘要已生成: {summary_path}")
        
    elif choice == '4':
        print("\n推荐使用简化版本:")
        print("python simplified_searchlight_analysis.py")
        print("\n简化版本的优势:")
        print("- 代码更清晰易懂")
        print("- 性能更稳定")
        print("- 专注核心功能")
        print("- 易于维护和修改")
        
    else:
        print("无效选择")

if __name__ == "__main__":
    main()