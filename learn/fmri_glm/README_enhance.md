# fMRI GLM 增强分析流程

## 概述

这是一个优化的fMRI GLM分析流程，相比原始流程有以下改进：

### 主要优化

1. **移除独立平滑步骤**：不再需要单独运行`spm_smooth.py`，平滑处理集成到GLM分析中
2. **流程整合**：将一阶分析、二阶分析和可视化整合到单一流程中
3. **灵活配置**：支持通过配置文件或命令行参数调整分析参数
4. **增强错误处理**：更好的日志记录和错误恢复机制
5. **保持兼容性**：保留原始文件不变，新建enhance版本

### 技术改进

- **平滑处理**：使用nilearn的`FirstLevelModel`内置平滑功能（`smoothing_fwhm`参数）
- **内存优化**：避免生成大量中间文件
- **并行处理**：支持多进程加速分析
- **结果验证**：自动检查分析结果完整性

## 文件结构

```
learn/fmri_glm/
├── 原始文件（保持不变）
│   ├── spm_smooth.py          # 原始平滑脚本
│   ├── fmri_glm_pipeline.py   # 原始一阶分析
│   ├── fmri_glm_second.py     # 原始二阶分析
│   └── fmri_glm_plot.py       # 原始可视化
│
├── 增强版本（新建）
│   ├── fmri_glm_enhance.py    # 主分析脚本
│   ├── config_enhance.py      # 配置管理
│   ├── run_enhanced_analysis.py # 运行脚本
│   └── README_enhance.md      # 说明文档
```

## 快速开始

### 1. 使用默认配置

```bash
# 运行完整分析流程
python run_enhanced_analysis.py
```

### 2. 自定义参数

```bash
# 分析前5个被试，不使用平滑
python run_enhanced_analysis.py --subjects 1-5 --smoothing 0

# 分析特定被试，使用4mm平滑
python run_enhanced_analysis.py --subjects 1,3,5 --smoothing 4

# 使用未校正阈值
python run_enhanced_analysis.py --uncorrected-p 0.001
```

### 3. 编程接口

```python
from config_enhance import AnalysisConfig
from fmri_glm_enhance import run_complete_analysis

# 创建自定义配置
config = AnalysisConfig()
config.SUBJECTS = ["sub-01", "sub-02"]
config.SMOOTHING_FWHM = 4  # 4mm平滑

# 运行分析
run_complete_analysis()
```

## 配置选项

### 基本参数

- `SUBJECTS`: 被试列表（如：["sub-01", "sub-02"]）
- `RUNS`: Run列表（如：[1, 2, 3, 4]）
- `SMOOTHING_FWHM`: 平滑核大小（mm），设为None禁用平滑
- `OUTPUT_ROOT`: 输出目录路径

### GLM参数

- `TR`: 重复时间（秒）
- `SLICE_TIME_REF`: 切片时间参考
- `HRF_MODEL`: 血氧动力学响应函数模型
- `DRIFT_MODEL`: 漂移模型
- `HIGH_PASS`: 高通滤波截止频率

### 并行处理

- `USE_MULTIPROCESS`: 是否使用多进程
- `MAX_WORKERS`: 最大工作进程数

### 统计阈值

- `alpha`: FDR校正alpha值
- `p_unc`: 未校正p值阈值
- `cluster_threshold`: 聚类大小阈值

## 输出结果

分析完成后，结果保存在指定的输出目录中：

```
output/
├── analysis_summary.json      # 分析总结
├── analysis.log               # 详细日志
├── sub-01/
│   └── run-1_spm_style/       # 一阶分析结果
│       ├── beta_*.nii         # Beta系数图
│       ├── con_*.nii          # 对比图
│       ├── spmT_*.nii         # T统计图
│       └── model_info.json    # 模型信息
└── 2nd_level/                 # 二阶分析结果
    ├── group_analysis_results.nii
    ├── visualization/
    │   ├── orthogonal_slices.png
    │   ├── glass_brain.png
    │   └── activation_clusters.csv
    └── model_info.json
```

## 平滑处理说明

### 为什么可以移除独立平滑步骤？

1. **fMRIPrep输出质量**：fMRIPrep已经进行了高质量的预处理，包括运动校正、配准等
2. **nilearn内置平滑**：`FirstLevelModel`的`smoothing_fwhm`参数可以在GLM分析时直接应用平滑
3. **内存效率**：避免生成大量平滑后的中间文件
4. **灵活性**：可以轻松调整或禁用平滑参数

### 平滑参数建议

- **标准分析**：6mm FWHM（默认值）
- **高分辨率数据**：4mm FWHM
- **ROI分析**：考虑不使用平滑（设为None）
- **组分析**：通常需要适度平滑以提高信噪比

## 与原始流程的对比

| 方面 | 原始流程 | 增强流程 |
|------|----------|----------|
| 步骤数 | 4个独立步骤 | 1个整合流程 |
| 平滑处理 | 独立脚本 | 集成到GLM |
| 中间文件 | 大量平滑文件 | 最少必要文件 |
| 配置方式 | 硬编码参数 | 灵活配置 |
| 错误处理 | 基本 | 增强 |
| 并行处理 | 部分支持 | 全面支持 |
| 结果验证 | 手动 | 自动 |

## 故障排除

### 常见问题

1. **内存不足**
   - 减少并行进程数：`--workers 2`
   - 禁用多进程：`--no-multiprocess`

2. **数据路径错误**
   - 检查`config_enhance.py`中的路径设置
   - 确保fMRIPrep输出文件存在

3. **平滑参数选择**
   - 对于高分辨率数据，尝试较小的FWHM值
   - 对于ROI分析，考虑禁用平滑

### 日志分析

详细的分析日志保存在`analysis.log`文件中，包含：
- 每个步骤的执行时间
- 错误和警告信息
- 数据质量检查结果
- 统计结果摘要

## 扩展和自定义

### 添加新的对比

在`config_enhance.py`中修改`CONTRASTS`字典：

```python
CONTRASTS = {
    'task_vs_baseline': [1, 0],
    'custom_contrast': [1, -1],  # 新增对比
}
```

### 修改统计阈值

```python
SECOND_LEVEL_CONFIG = {
    "alpha": 0.01,           # 更严格的FDR阈值
    "cluster_threshold": 20,  # 更大的聚类阈值
}
```

### 自定义可视化

可以修改`fmri_glm_enhance.py`中的`create_visualization`函数来添加新的可视化选项。

## 性能优化建议

1. **使用SSD存储**：提高I/O性能
2. **合理设置并行数**：通常设为CPU核心数的50-75%
3. **监控内存使用**：大数据集可能需要调整批处理大小
4. **定期清理**：删除不需要的中间文件

## 技术支持

如果遇到问题，请：
1. 检查日志文件中的错误信息
2. 验证数据路径和文件完整性
3. 确认依赖包版本兼容性
4. 参考nilearn和SPM文档

---

*此增强版本保持与原始分析流程的结果兼容性，同时提供更好的用户体验和性能。*