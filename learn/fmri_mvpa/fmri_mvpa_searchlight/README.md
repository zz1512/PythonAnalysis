# fMRI MVPA Searchlight 分析工具

这是一个用于fMRI数据多体素模式分析(MVPA)的Searchlight分析工具，支持多种语言处理相关脑区的掩模分析，并已集成LSS (Least Squares Separate) 数据支持。

## 功能特点

- 🧠 支持全脑和特定脑区的Searchlight分析
- 🎯 预置9个语言处理相关脑区掩模
- 📊 自动生成统计结果和可视化报告
- 🔧 灵活的配置选项
- 📁 使用相对路径，便于项目迁移
- 🔗 **集成LSS数据支持** - 直接分析trial-wise beta图像
- 🔄 **多Mask分析** - 同时分析多个脑区并对比结果

## 预置脑区掩模

### 语言处理脑区

| 脑区 | 文件名 | 功能 |
|------|--------|------|
| Broca区 | `broca_area.nii.gz` | 语言产生 |
| Wernicke区 | `wernicke_area.nii.gz` | 语言理解 |
| 角回 | `angular_gyrus.nii.gz` | 语义处理 |
| 下额回 | `inferior_frontal_gyrus.nii.gz` | 语言加工 |
| 上颞回 | `superior_temporal_gyrus.nii.gz` | 听觉语言处理 |
| 中颞回 | `middle_temporal_gyrus.nii.gz` | 语义记忆 |
| 楔前叶 | `precuneus.nii.gz` | 空间认知 |
| 后顶叶皮层 | `posterior_parietal_cortex.nii.gz` | 注意和空间处理 |

### 网络掩模

| 网络 | 文件名 | 功能 |
|------|--------|------|
| 语言网络 | `language_network_mask.nii.gz` | 综合语言网络 |
| 空间认知网络 | `spatial_network_mask.nii.gz` | 空间认知网络 |

### 灰质层掩模

| 掩模类型 | 文件名 | 功能 |
|----------|--------|------|
| 全脑灰质层 | `gray_matter.nii.gz` | 基于MNI152模板的全脑灰质 |
| 皮层灰质 | `cortical_gray_matter.nii.gz` | 结合Harvard-Oxford皮层图谱的皮层灰质 |

## 快速开始

### 1. 安装依赖

```bash
pip install nilearn scikit-learn numpy pandas matplotlib seaborn statsmodels
```

### 2. 下载脑区掩模

掩模文件已自动下载到 `data/masks/` 目录。

### 3. 运行示例

```python
from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis

# 示例1: 全脑分析
config = SearchlightMVPAConfig()
results = run_group_searchlight_analysis(config)

# 示例2: 特定脑区分析 (推荐使用便捷方法)
config = SearchlightMVPAConfig()
config.set_mask('broca')  # 自动设置mask_selection和use_custom_mask
results = run_group_searchlight_analysis(config)

# 示例3: 语言网络分析
config = SearchlightMVPAConfig()
config.set_mask('language_network')
results = run_group_searchlight_analysis(config)

# 示例4: 灰质层分析
config = SearchlightMVPAConfig()
config.set_mask('gray_matter')
results = run_group_searchlight_analysis(config)

# 示例5: 验证mask可用性
config = SearchlightMVPAConfig()
available_masks = config.get_available_masks()
print(f"可用mask: {available_masks}")

# 设置mask并验证
config.set_mask('cortical_gray_matter')
is_valid, info = config.validate_mask_availability()
print(f"配置验证: {info}")

if is_valid:
    results = run_group_searchlight_analysis(config)
```

### 4. 查看示例

运行 `example_usage.py` 查看详细使用示例：

```bash
python example_usage.py
```

## 目录结构

```
project_root/
├── data/
│   ├── lss_trial_data/          # LSS分析结果数据
│   └── masks/                   # 脑区掩模文件
│       ├── broca_area.nii.gz
│       ├── wernicke_area.nii.gz
│       └── ...
├── results/
│   └── searchlight_mvpa/        # 分析结果输出
└── learn/fmri_mvpa/fmri_mvpa_searchlight/
    ├── fmri_mvpa_searchlight_pipeline_enhanced.py
    ├── example_usage.py
    └── README.md
```

## Mask使用方法 (优化版)

### 便捷方法 (推荐)

```python
config = SearchlightMVPAConfig()

# 1. 查看可用mask
available_masks = config.get_available_masks()
print(f"可用mask: {available_masks}")

# 2. 设置mask (自动配置相关参数)
config.set_mask('broca')  # 或其他mask名称

# 3. 验证配置
is_valid, info = config.validate_mask_availability()
print(f"配置状态: {info}")

# 4. 运行分析
if is_valid:
    results = run_group_searchlight_analysis(config)
```

### 传统方法 (仍然支持)

```python
config = SearchlightMVPAConfig()
config.mask_selection = 'broca'
config.use_custom_mask = True
results = run_group_searchlight_analysis(config)
```

### 错误处理

```python
try:
    config.set_mask('nonexistent_mask')
except ValueError as e:
    print(f"错误: {e}")
    print(f"可用选项: {config.get_available_masks()}")
```

## 多Mask分析 (新功能)

### 功能概述

多Mask分析允许您在一次运行中同时分析多个脑区，极大提高分析效率并便于结果比较。

### 基本用法

```python
from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis

# 创建配置
config = SearchlightMVPAConfig()

# 设置多个mask进行分析
mask_list = ['gray_matter', 'language_network', 'broca']
config.set_mask(mask_list)

# 运行分析
results = run_group_searchlight_analysis(config)
```

### 支持的分析场景

#### 1. 语言脑区对比分析
```python
# 分析多个语言相关脑区
language_masks = ['broca', 'wernicke', 'angular_gyrus', 'language_network']
config.set_mask(language_masks)
```

#### 2. 全脑vs特定脑区对比
```python
# 对比全脑和特定脑区的分类效果
comparison_masks = ['whole_brain', 'gray_matter', 'language_network']
config.set_mask(comparison_masks)
```

#### 3. 网络级别分析
```python
# 分析不同功能网络
network_masks = ['language_network', 'spatial_network']
config.set_mask(network_masks)
```

### 结果文件命名规则

#### 单Mask分析
- 准确率图: `contrast_name_accuracy.nii.gz`
- 统计图: `contrast_name_stats.nii.gz`

#### 多Mask分析
- 准确率图: `contrast_name_maskname_accuracy.nii.gz`
- 统计图: `contrast_name_maskname_stats.nii.gz`

### 结果数据结构

```python
# 多mask分析结果字典结构
results = {
    'subject_01': {
        'words_vs_pseudowords_gray_matter': {...},
        'words_vs_pseudowords_language_network': {...},
        'words_vs_pseudowords_broca': {...}
    }
}
```

### 配置验证

```python
# 验证多mask配置
is_valid, info = config.validate_mask_availability()
print(f"配置状态: {info}")

if is_valid:
    print("所有mask配置有效，可以开始分析")
else:
    print("配置验证失败，请检查mask文件")
```

### 优势特点

1. **效率提升**: 一次运行，多个脑区结果
2. **便于比较**: 统一参数下的多脑区对比
3. **节省时间**: 避免重复的数据加载和预处理
4. **结果一致**: 确保所有脑区使用相同的分析参数

### 注意事项

1. **内存使用**: 多mask分析会增加内存使用，建议监控系统资源
2. **存储空间**: 每个mask会生成独立的结果文件，注意磁盘空间
3. **分析时间**: 总分析时间与mask数量成正比
4. **结果解读**: 注意区分不同mask的结果文件

### 完整示例

参见 `example_multi_mask_usage.py` 文件，包含详细的使用示例和错误处理演示。

## 配置选项

### 主要参数

- `searchlight_radius`: Searchlight半径 (默认: 4.0mm)
- `cv_folds`: 交叉验证折数 (默认: 5)
- `mask_selection`: 选择的脑区掩模
- `use_custom_mask`: 是否使用自定义掩模
- `n_jobs`: 并行处理数 (默认: -1)

### 掩模选择

```python
# 全脑分析
config.mask_selection = 'whole_brain'
config.use_custom_mask = False

# 特定脑区分析
config.mask_selection = 'broca'  # 可选: broca, wernicke, angular_gyrus 等
config.use_custom_mask = True
```

## 输出结果

分析完成后，结果保存在 `results/searchlight_mvpa/` 目录：

- 统计图像文件 (NIfTI格式)
- HTML可视化报告
- 分析日志文件

## LSS数据集成 (新功能)

### 概述

本工具已完全集成LSS (Least Squares Separate) 数据支持，可直接分析来自 `fmri_lss_enhance.py` 的trial-wise beta图像。

### LSS数据结构

LSS分析输出的数据结构如下：
```
../../learn_LSS/
├── sub-01/
│   ├── run-3_LSS/
│   │   ├── beta_trial_001.nii.gz
│   │   ├── beta_trial_002.nii.gz
│   │   ├── ...
│   │   ├── trial_info.csv
│   │   └── mask.nii.gz
│   └── run-4_LSS/
│       └── ...
├── sub-02/
└── ...
```

### 使用LSS数据进行分析

```python
from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis

# 创建配置（已自动适配LSS数据）
config = SearchlightMVPAConfig()

# LSS数据路径已自动设置为: ../../learn_LSS
# 被试范围已自动设置为: sub-01 到 sub-28
# Run编号已自动设置为: [3, 4]

# 设置对比条件（使用LSS数据中的条件名）
config.contrasts = [
    ('target_vs_baseline', 'target', 'baseline'),
    # 根据实际LSS数据中的条件调整
]

# 运行分析
results = run_group_searchlight_analysis(config)
```

### LSS数据字段说明

`trial_info.csv` 包含以下关键字段：
- `trial_index`: trial索引（1-based）
- `original_condition`: 原始条件名称
- `beta_file`: beta图像文件名
- `onset`: 事件开始时间
- `duration`: 事件持续时间

### 示例脚本

运行LSS数据分析示例：
```bash
python example_lss_searchlight_analysis.py
```

测试LSS数据集成：
```bash
python test_lss_integration.py
```

### LSS集成优势

1. **Trial-wise分析**: 直接使用每个trial的beta图像
2. **自动数据加载**: 无需手动处理数据格式
3. **多run支持**: 自动合并多个run的数据
4. **条件灵活性**: 支持任意条件对比
5. **完整兼容**: 保持所有原有功能

## 注意事项

1. 确保LSS数据已准备好在 `../../learn_LSS/` 目录
2. LSS数据来源于 `fmri_lss_enhance.py` 的分析结果
3. 首次运行会自动创建必要的目录结构
4. 分析可能需要较长时间，建议在服务器上运行
5. 结果文件较大，注意磁盘空间
6. 对比条件名称需要匹配LSS数据中的 `original_condition` 字段

## 故障排除

### 常见问题

1. **掩模文件未找到**: 检查 `data/masks/` 目录是否存在所需文件
2. **内存不足**: 减少 `n_jobs` 参数或使用更小的掩模
3. **数据路径错误**: 确认LSS数据路径配置正确

### 获取帮助

如遇问题，请检查：
1. 依赖包是否正确安装
2. 数据路径是否正确
3. 掩模文件是否存在