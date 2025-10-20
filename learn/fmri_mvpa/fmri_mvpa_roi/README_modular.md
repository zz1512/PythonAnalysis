# fMRI MVPA ROI Pipeline - 模块化版本

## 概述

这是fMRI MVPA ROI分析管道的模块化重构版本。原来的单一文件 `fmri_mvpa_roi_pipeline_corrected.py` 已被拆分为多个功能模块，提高了代码的可维护性和可扩展性。

## 模块结构

### 核心模块

1. **config.py** - 配置管理
   - `MVPAConfig`: 主配置类，包含所有分析参数
   - 支持配置保存和加载

2. **utils.py** - 工具函数
   - `log()`: 增强的日志记录函数
   - `memory_cleanup()`: 内存清理函数

3. **data_loader.py** - 数据加载
   - `load_roi_masks()`: 加载ROI mask文件
   - `load_lss_trial_data()`: 加载LSS trial数据
   - `load_multi_run_lss_data()`: 加载多run LSS数据

4. **feature_extraction.py** - 特征提取
   - `extract_roi_timeseries_optimized()`: 优化的ROI时间序列提取

5. **classification.py** - 分类分析
   - `run_roi_classification_enhanced()`: 增强的ROI分类分析
   - `prepare_classification_data()`: 分类数据准备

6. **statistics.py** - 统计分析
   - `perform_group_statistics_corrected()`: 组水平统计分析
   - `permutation_test_group_level()`: 置换检验

7. **visualization.py** - 可视化
   - `create_enhanced_visualizations()`: 创建增强可视化图表
   - `create_individual_roi_plots()`: 创建单个ROI详细图表

8. **report_generator.py** - 报告生成
   - `generate_html_report()`: 生成HTML报告
   - `image_to_base64()`: 图片转base64编码

9. **main_analysis.py** - 主分析流程
   - `run_group_roi_analysis_corrected()`: 主分析函数
   - `main()`: 程序入口点

### 入口文件

- **fmri_mvpa_roi_pipeline_modular.py** - 新的模块化入口文件
- **fmri_mvpa_roi_pipeline_corrected.py** - 原始文件（保持不变）

## 使用方法

### 基本使用

```bash
# 使用新的模块化版本
python fmri_mvpa_roi_pipeline_modular.py

# 或者直接运行主分析模块
python main_analysis.py
```

### 自定义配置

```python
from config import MVPAConfig
from main_analysis import run_group_roi_analysis_corrected

# 创建自定义配置
config = MVPAConfig()
config.subject_ids = ['sub-01', 'sub-02', 'sub-03']  # 自定义被试列表
config.cv_folds = 10  # 修改交叉验证折数
config.n_permutations = 2000  # 修改置换检验次数

# 运行分析
results_dir = run_group_roi_analysis_corrected(config)
```

### 单独使用模块

```python
# 只进行数据加载
from config import MVPAConfig
from data_loader import load_roi_masks, load_multi_run_lss_data

config = MVPAConfig()
roi_masks = load_roi_masks(config)
lss_data = load_multi_run_lss_data(config, 'sub-01')

# 只进行统计分析
from statistics import perform_group_statistics_corrected
import pandas as pd

group_df = pd.read_csv('group_results_raw.csv')
stats_df = perform_group_statistics_corrected(group_df, config)

# 只生成可视化
from visualization import create_enhanced_visualizations

create_enhanced_visualizations(group_df, stats_df, config)
```

## 模块化的优势

1. **可维护性**: 每个模块职责单一，便于维护和调试
2. **可扩展性**: 可以轻松添加新的分析方法或可视化类型
3. **可重用性**: 各模块可以独立使用，支持自定义分析流程
4. **可测试性**: 每个模块可以独立测试
5. **代码组织**: 逻辑清晰，便于理解和协作开发

## 依赖关系

```
main_analysis.py
├── config.py
├── utils.py
├── data_loader.py
│   ├── config.py
│   └── utils.py
├── classification.py
│   ├── config.py
│   ├── utils.py
│   └── feature_extraction.py
├── statistics.py
│   ├── config.py
│   └── utils.py
├── visualization.py
│   ├── config.py
│   └── utils.py
└── report_generator.py
    └── utils.py
```

## 输出文件

分析完成后，会在结果目录中生成以下文件：

- `group_results_raw.csv`: 原始分组结果
- `group_statistics_corrected.csv`: 统计分析结果
- `enhanced_analysis_visualizations.png`: 主要可视化图表
- `individual_roi_detailed_plots.png`: 单个ROI详细图表
- `enhanced_analysis_report.html`: 完整HTML报告
- `results_summary_chinese.csv`: 中文结果摘要
- `analysis_config.json`: 分析配置文件
- `analysis.log`: 分析日志文件

## 注意事项

1. 确保所有模块文件都在同一目录下
2. 原始文件 `fmri_mvpa_roi_pipeline_corrected.py` 保持不变
3. 新的模块化版本与原版本功能完全一致
4. 可以根据需要选择使用原版本或模块化版本

## 兼容性

- Python 3.7+
- 所有原有依赖库保持不变
- 输出格式和结果与原版本完全一致