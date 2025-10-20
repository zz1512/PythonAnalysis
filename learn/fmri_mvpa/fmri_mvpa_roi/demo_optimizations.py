#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示优化后的图片加载和字体设置功能

这个脚本演示了：
1. 增强的图片加载错误处理
2. 跨平台字体设置
3. HTML报告生成的鲁棒性
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from report_generator import image_to_base64, generate_image_html, generate_html_report
from visualization import setup_chinese_fonts, create_enhanced_visualizations
from config import MVPAConfig

def create_demo_data():
    """创建演示数据"""
    print("📊 创建演示数据...")
    
    # 创建模拟的组级数据
    np.random.seed(42)
    subjects = [f'sub-{i:02d}' for i in range(1, 21)]  # 20个被试
    rois = ['左侧海马', '右侧海马', '左侧杏仁核', '右侧杏仁核']
    
    group_data = []
    for subject in subjects:
        for roi in rois:
            # 模拟不同ROI的准确率
            base_acc = 0.55 if '海马' in roi else 0.52
            accuracy = base_acc + np.random.normal(0, 0.08)
            accuracy = np.clip(accuracy, 0.3, 0.9)  # 限制在合理范围内
            
            group_data.append({
                'subject': subject,
                'roi': roi,
                'contrast': 'metaphor_vs_space',
                'mean_accuracy': accuracy
            })
    
    group_df = pd.DataFrame(group_data)
    
    # 创建统计结果数据
    stats_data = []
    for roi in rois:
        roi_data = group_df[group_df['roi'] == roi]['mean_accuracy']
        mean_acc = roi_data.mean()
        sem_acc = roi_data.sem()
        
        # 模拟统计检验结果
        t_stat = (mean_acc - 0.5) / sem_acc * np.sqrt(len(roi_data))
        p_val = 0.01 if '海马' in roi else 0.15  # 海马区域显著
        cohens_d = (mean_acc - 0.5) / roi_data.std()
        
        stats_data.append({
            'roi': roi,
            'mean_accuracy': mean_acc,
            'sem_accuracy': sem_acc,
            't_statistic': t_stat,
            'p_value_permutation': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05,
            'ci_lower': mean_acc - 1.96 * sem_acc,
            'ci_upper': mean_acc + 1.96 * sem_acc,
            'n_subjects': len(roi_data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    print(f"✅ 创建了 {len(group_df)} 条数据记录，涵盖 {len(subjects)} 个被试和 {len(rois)} 个ROI")
    return group_df, stats_df

def demo_font_optimization():
    """演示字体优化功能"""
    print("\n🔤 演示字体优化功能...")
    
    # 测试字体设置
    font_success = setup_chinese_fonts()
    print(f"字体设置结果: {'✅ 成功' if font_success else '❌ 失败'}")
    print(f"当前字体: {plt.rcParams['font.sans-serif'][0]}")
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 测试数据
    rois = ['左侧海马', '右侧海马', '左侧杏仁核', '右侧杏仁核']
    accuracies = [0.62, 0.58, 0.51, 0.49]
    colors = ['#2E8B57', '#4682B4', '#CD853F', '#DC143C']
    
    bars = ax.bar(rois, accuracies, color=colors, alpha=0.7)
    ax.set_title('fMRI MVPA 分类准确率结果', fontsize=16, fontweight='bold')
    ax.set_ylabel('分类准确率', fontsize=12)
    ax.set_xlabel('脑区 (ROI)', fontsize=12)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='机会水平')
    ax.legend()
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    demo_img_path = Path('demo_font_test.png')
    plt.savefig(demo_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if demo_img_path.exists():
        print(f"✅ 中文字体测试图片已保存: {demo_img_path}")
        return str(demo_img_path)
    else:
        print("❌ 图片保存失败")
        return None

def demo_image_loading():
    """演示图片加载优化功能"""
    print("\n🖼️ 演示图片加载优化功能...")
    
    # 测试1: 正常图片加载
    demo_img = demo_font_optimization()
    if demo_img:
        print("\n测试正常图片加载:")
        b64_data = image_to_base64(demo_img)
        success = b64_data is not None
        print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
        
        # 清理测试文件
        Path(demo_img).unlink()
    
    # 测试2: 不存在的文件
    print("\n测试不存在文件的错误处理:")
    b64_data = image_to_base64("nonexistent_image.png")
    success = b64_data is None
    print(f"错误处理结果: {'✅ 正确处理' if success else '❌ 处理失败'}")
    
    # 测试3: HTML生成
    print("\n测试HTML图片标签生成:")
    html_with_image = generate_image_html("dummy_base64_data", "测试图片")
    html_without_image = generate_image_html(None, "缺失图片")
    
    print(f"有图片的HTML: {'✅ 包含img标签' if '<img' in html_with_image else '❌ 缺少img标签'}")
    print(f"无图片的HTML: {'✅ 包含错误提示' if '加载失败' in html_without_image else '❌ 缺少错误提示'}")

def demo_html_report():
    """演示HTML报告生成"""
    print("\n📄 演示HTML报告生成...")
    
    # 创建演示数据
    group_df, stats_df = create_demo_data()
    
    # 创建配置
    config = MVPAConfig()
    
    # 生成可视化（模拟）
    print("生成可视化图表...")
    try:
        # 设置字体
        setup_chinese_fonts()
        
        # 创建可视化
        create_enhanced_visualizations(group_df, stats_df, config)
        
        # 检查图片文件
        main_viz_path = Path('enhanced_analysis_visualizations.png')
        individual_viz_path = Path('individual_roi_detailed_plots.png')
        
        main_viz_b64 = image_to_base64(str(main_viz_path)) if main_viz_path.exists() else None
        individual_viz_b64 = image_to_base64(str(individual_viz_path)) if individual_viz_path.exists() else None
        
        print(f"主要可视化: {'✅ 生成成功' if main_viz_b64 else '❌ 生成失败'}")
        print(f"详细可视化: {'✅ 生成成功' if individual_viz_b64 else '❌ 生成失败'}")
        
        # 生成HTML报告
        print("生成HTML报告...")
        generate_html_report(group_df, stats_df, config, main_viz_b64, individual_viz_b64)
        
        # 检查报告文件
        report_path = Path('fmri_mvpa_analysis_report.html')
        if report_path.exists():
            print(f"✅ HTML报告生成成功: {report_path}")
            print(f"报告大小: {report_path.stat().st_size / 1024:.1f} KB")
            return str(report_path)
        else:
            print("❌ HTML报告生成失败")
            return None
            
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        return None

def main():
    """主演示函数"""
    print("🚀 开始演示优化后的功能...")
    print("=" * 60)
    
    try:
        # 演示各项功能
        demo_font_optimization()
        demo_image_loading()
        report_path = demo_html_report()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        
        print("\n📋 优化总结:")
        print("1. ✅ 图片加载: 增强了错误处理和文件大小检查")
        print("2. ✅ 字体设置: 支持跨平台中文字体自动检测")
        print("3. ✅ HTML生成: 增加了图片加载失败的优雅降级")
        print("4. ✅ 错误处理: 所有关键步骤都有详细的日志记录")
        
        if report_path:
            print(f"\n📊 生成的HTML报告: {report_path}")
            print("可以在浏览器中打开查看完整的分析结果")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()