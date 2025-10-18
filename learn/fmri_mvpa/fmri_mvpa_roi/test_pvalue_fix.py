#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试p值修复的脚本
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from fmri_mvpa_roi_pipeline_enhanced import run_roi_classification_enhanced, MVPAConfig

def test_pvalue_calculation():
    """测试p值计算是否正常"""
    print("测试p值计算...")
    
    # 创建配置
    config = MVPAConfig()
    config.n_permutations = 10  # 减少置换次数以便快速测试
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 50
    n_features = 100
    
    # 创建有一定分类能力的数据
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples)
    
    # 为第一类添加一些信号
    X[y == 0, :20] += 0.5
    
    try:
        # 运行分类
        result = run_roi_classification_enhanced(X, y, config)
        
        print(f"分类结果:")
        print(f"  平均准确率: {result['mean_accuracy']:.4f}")
        print(f"  p值: {result['p_value']:.6f}")
        print(f"  置换分布均值: {np.mean(result['null_distribution']):.4f}")
        print(f"  置换分布标准差: {np.std(result['null_distribution']):.4f}")
        
        # 检查p值是否合理
        if result['p_value'] == 0.0:
            print("❌ 错误: p值仍然为0")
            return False
        elif result['p_value'] < 1.0 / (config.n_permutations + 1):
            print("❌ 错误: p值小于理论最小值")
            return False
        else:
            print(f"✅ 成功: p值正常 ({result['p_value']:.6f})")
            return True
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    success = test_pvalue_calculation()
    if success:
        print("\n🎉 p值修复测试通过!")
    else:
        print("\n💥 p值修复测试失败!")
    
    sys.exit(0 if success else 1)