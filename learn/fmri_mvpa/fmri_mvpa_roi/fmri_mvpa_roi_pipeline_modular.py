#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI MVPA ROI Pipeline - 模块化版本

这是重构后的模块化版本，将原来的单一文件拆分为多个功能模块：
- config.py: 配置管理
- utils.py: 工具函数
- data_loader.py: 数据加载
- feature_extraction.py: 特征提取
- classification.py: 分类分析
- statistics.py: 统计分析
- visualization.py: 可视化
- report_generator.py: 报告生成
- main_analysis.py: 主分析流程

使用方法：
    python fmri_mvpa_roi_pipeline_modular.py

作者: AI Assistant
日期: 2024
"""

from main_analysis import main

if __name__ == "__main__":
    # 运行主分析
    main()