# 增强版MVPA分析流水线

这是一个功能完整、高度优化的fMRI多体素模式分析(MVPA)流水线，集成了数据流优化、配置管理、错误处理、质量控制和可视化等多个增强功能。

## 🚀 主要特性

### 1. 流程优化
- **数据流优化**: 智能缓存机制，避免重复计算
- **配置管理**: 统一的配置文件管理系统
- **错误处理**: 完善的错误处理和容错机制
- **进度跟踪**: 实时分析进度监控

### 2. 性能增强
- **并行处理**: 支持多核并行计算
- **内存优化**: 智能内存管理和清理
- **批处理**: 大数据集的分批处理
- **性能监控**: 实时系统资源监控

### 3. 质量控制
- **数据质量评估**: SNR、tSNR、运动参数等指标
- **自动质量评级**: 基于多指标的质量评分
- **质量报告**: 详细的HTML质量报告

### 4. 可视化增强
- **脑图可视化**: 多视角脑激活图
- **交互式图表**: 基于Plotly的交互式结果展示
- **综合报告**: 自动生成HTML/PDF分析报告
- **组间比较**: 多组数据对比可视化

## 📁 文件结构

```
fmri_mvpa_enhance/
├── README.md                    # 本文档
├── example_config.json          # 示例配置文件
├── example_usage.py            # 使用示例
├── enhanced_pipeline.py        # 主流水线
├── global_config.py            # 全局配置管理
├── error_handling.py           # 错误处理模块
├── data_flow.py               # 数据流优化
├── lss_enhanced.py            # 增强LSS分析
├── quality_control.py         # 质量控制模块
└── visualization.py           # 可视化模块
```

## 🛠️ 安装要求

### 必需依赖
```bash
# 核心科学计算库
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0

# 神经影像处理
nibabel>=3.2.0
nilearn>=0.8.0

# 机器学习
scikit-learn>=1.0.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# 并行处理
joblib>=1.1.0

# 其他
psutil>=5.8.0
tqdm>=4.62.0
```

### 可选依赖
```bash
# PDF报告生成
weasyprint>=54.0

# 更好的进度条
rich>=10.0.0

# 内存分析
memory-profiler>=0.60.0
```

### 安装命令
```bash
# 使用pip安装
pip install numpy scipy pandas nibabel nilearn scikit-learn matplotlib seaborn plotly joblib psutil tqdm

# 可选依赖
pip install weasyprint rich memory-profiler
```

## 🚀 快速开始

### 1. 基本使用

```python
from enhanced_pipeline import create_enhanced_pipeline

# 创建流水线
pipeline = create_enhanced_pipeline(
    output_dir="./my_mvpa_results",
    enable_cache=True,
    enable_quality_control=True,
    enable_visualization=True
)

# 运行完整分析
results = pipeline.run_complete_analysis(
    subjects=["sub-01", "sub-02", "sub-03"],
    runs=["run-1", "run-2"]
)
```

### 2. 使用配置文件

```python
from enhanced_pipeline import EnhancedMVPAPipeline

# 使用自定义配置
pipeline = EnhancedMVPAPipeline(
    config_path="./my_config.json",
    output_dir="./results"
)

results = pipeline.run_complete_analysis()
```

### 3. 快速分析

```python
from enhanced_pipeline import run_quick_analysis

# 一行代码运行完整分析
results = run_quick_analysis(
    data_dir="/path/to/bids/data",
    subjects=["sub-01", "sub-02"],
    runs=["run-1", "run-2"],
    output_dir="./quick_results"
)
```

## ⚙️ 配置说明

### 配置文件结构

配置文件采用JSON格式，包含三个主要部分：

```json
{
  "lss_config": {
    "data_dir": "/path/to/bids/data",
    "subjects": ["sub-01", "sub-02"],
    "runs": ["run-1", "run-2"],
    "tr": 2.0,
    "smoothing_fwhm": 6.0,
    "n_jobs": -1
  },
  "roi_config": {
    "sphere_rois": [...],
    "anatomical_rois": [...],
    "network_rois": [...]
  },
  "mvpa_config": {
    "svm_params": {...},
    "cv_params": {...},
    "permutation_params": {...}
  }
}
```

### 主要配置参数

#### LSS配置 (lss_config)
- `data_dir`: BIDS格式数据目录
- `subjects`: 被试列表
- `runs`: run列表
- `tr`: 重复时间(秒)
- `smoothing_fwhm`: 平滑核大小(mm)
- `n_jobs`: 并行作业数(-1表示使用所有CPU)

#### ROI配置 (roi_config)
- `sphere_rois`: 球形ROI定义
- `anatomical_rois`: 解剖ROI定义
- `network_rois`: 功能网络ROI定义

#### MVPA配置 (mvpa_config)
- `svm_params`: SVM分类器参数
- `cv_params`: 交叉验证参数
- `permutation_params`: 置换检验参数

## 📊 输出结果

### 目录结构
```
output_directory/
├── analysis_summary.json       # 分析摘要
├── detailed_results.json       # 详细结果
├── default_config.json         # 使用的配置
├── pipeline.log               # 分析日志
├── lss_results/               # LSS分析结果
├── roi_masks/                 # ROI掩模文件
├── mvpa_results/              # MVPA分析结果
├── quality_reports/           # 质量控制报告
├── performance/               # 性能监控结果
├── visualizations/            # 可视化文件
├── reports/                   # 综合报告
└── cache/                     # 缓存文件
```

### 主要输出文件

1. **分析摘要** (`analysis_summary.json`)
   - 分析配置信息
   - 处理的被试和run数量
   - 各步骤完成状态
   - 输出目录信息

2. **质量报告** (`quality_reports/`)
   - 每个被试的数据质量评估
   - SNR、tSNR、运动参数等指标
   - 质量评级和建议

3. **可视化结果** (`visualizations/`)
   - 脑激活图
   - ROI分析结果图
   - 交互式仪表板
   - 组间比较图

4. **综合报告** (`reports/`)
   - HTML格式的完整分析报告
   - 包含所有图表和统计结果
   - 可直接在浏览器中查看

## 🔧 高级功能

### 1. 缓存机制

```python
# 启用缓存（默认）
pipeline = create_enhanced_pipeline(enable_cache=True)

# 清理缓存
pipeline.cache.cleanup()

# 查看缓存统计
stats = pipeline.cache.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
```

### 2. 错误处理

```python
# 查看错误统计
error_stats = pipeline.error_handler.get_error_stats()
print(f"错误数量: {error_stats['total_errors']}")

# 导出错误报告
pipeline.error_handler.export_error_report("./error_report.json")
```

### 3. 性能监控

```python
# 查看性能统计
if pipeline.enable_quality_control:
    perf_stats = pipeline.performance_monitor.get_current_stats()
    print(f"CPU使用率: {perf_stats['cpu_percent']:.1f}%")
    print(f"内存使用: {perf_stats['memory_mb']:.1f} MB")
```

### 4. 分步骤执行

```python
# 仅运行LSS分析
lss_results = pipeline.run_lss_analysis(subjects, runs)

# 仅运行质量控制
quality_results = pipeline.run_quality_control(subjects, runs)

# 仅生成可视化
viz_results = pipeline.generate_visualizations_and_reports()
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减少并行作业数
   pipeline = create_enhanced_pipeline(n_jobs=2)
   
   # 减少批处理大小
   config.lss_config.chunk_size = 500
   ```

2. **数据路径错误**
   ```python
   # 检查数据目录结构
   from pathlib import Path
   data_dir = Path("/path/to/data")
   print(list(data_dir.glob("sub-*/func/*.nii.gz")))
   ```

3. **依赖包问题**
   ```bash
   # 检查包版本
   pip list | grep -E "(nibabel|nilearn|sklearn)"
   
   # 更新包
   pip install --upgrade nibabel nilearn scikit-learn
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用小数据集测试
test_results = run_quick_analysis(
    data_dir="/path/to/data",
    subjects=["sub-01"],  # 仅一个被试
    runs=["run-1"],       # 仅一个run
    output_dir="./test_results"
)
```

## 📚 示例和教程

查看 `example_usage.py` 文件获取详细的使用示例：

- 基本使用方法
- 自定义配置
- 仅LSS分析
- 质量控制
- 结果可视化
- 快速分析
- 分步骤分析

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 📞 支持和反馈

如有问题或建议，请：

1. 查看本文档的故障排除部分
2. 检查 `example_usage.py` 中的示例
3. 提交 Issue 或 Pull Request

## 🔄 更新日志

### v1.0.0 (当前版本)
- 初始版本发布
- 完整的MVPA分析流水线
- 数据流优化和缓存机制
- 错误处理和质量控制
- 可视化和报告生成
- 性能监控和优化

---

**注意**: 这是一个增强版的MVPA分析流水线，在原有功能基础上添加了多项优化和增强功能。使用前请确保您的数据符合BIDS格式，并根据实际需求调整配置参数。