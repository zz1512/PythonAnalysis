# 增强版MVPA分析流水线 v2.0

这是一个功能完整、高度优化的fMRI多体素模式分析(MVPA)流水线，集成了最新的LSS、ROI和MVPA优化算法，提供生产级别的稳定性和性能。

## 🚀 主要特性

### 1. 核心算法优化
- **优化版LSS分析**: 集成最新单trial GLM算法，支持高效并行处理
- **智能ROI创建**: 支持球形、解剖学和激活ROI的批量创建和质量控制
- **增强版MVPA**: 先进的特征提取、分类分析和统计检验
- **算法整合**: 无缝集成三大核心模块的最新优化算法

### 2. 性能与扩展性
- **多级并行处理**: 被试级、run级和任务级并行优化
- **智能内存管理**: 自动内存优化和缓存策略
- **批处理支持**: 大规模数据集的高效处理
- **性能监控**: 实时资源使用和性能统计

### 3. 质量控制与验证
- **综合质量评估**: SNR、tSNR、运动参数等多维度指标
- **结果验证**: 自动化结果验证和一致性检查
- **质量报告**: 详细的HTML质量控制报告
- **可视化诊断**: 交互式质量诊断图表

### 4. 用户体验优化
- **配置管理v2.0**: 统一的配置系统，支持保存/加载
- **错误恢复**: 智能错误处理和分析恢复机制
- **进度监控**: 实时分析进度和状态跟踪
- **综合报告**: 自动生成HTML/PDF分析报告

## 📁 文件结构

```
fmri_mvpa_enhance/
├── README.md                    # 本文档
├── example_usage.py            # v2.0使用示例
├── enhanced_pipeline.py        # 增强版主流水线 v2.0
├── global_config.py            # 全局配置管理 v2.0
├── lss_optimized.py            # 优化版LSS分析器
├── roi_mask_creator.py         # 优化版ROI创建器
├── mvpa_enhanced.py            # 增强版MVPA分析器
├── error_handling.py           # 错误处理模块
├── data_flow.py               # 数据流优化
├── quality_control.py         # 质量控制模块
└── visualization.py           # 可视化模块
```

## 🛠️ 安装要求

### 必需依赖
```bash
# 核心科学计算库
numpy>=1.21.0
scipy>=1.8.0
pandas>=1.4.0

# 神经影像处理
nibabel>=3.2.0
nilearn>=0.9.0

# 机器学习
scikit-learn>=1.1.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# 并行处理和优化
joblib>=1.2.0
psutil>=5.9.0
tqdm>=4.64.0
```

### 可选依赖
```bash
# 报告生成
weasyprint>=56.0
jinja2>=3.0.0

# 性能分析
memory-profiler>=0.60.0
rich>=12.0.0

# 统计分析
statsmodels>=0.13.0
```

### 安装命令
```bash
# 基础安装
pip install numpy scipy pandas nibabel nilearn scikit-learn matplotlib seaborn plotly joblib psutil tqdm

# 完整安装（包含可选依赖）
pip install numpy scipy pandas nibabel nilearn scikit-learn matplotlib seaborn plotly joblib psutil tqdm weasyprint jinja2 memory-profiler rich statsmodels
```

## 🚀 快速开始

### 1. 增强版流水线v2.0（推荐）

```python
from enhanced_pipeline import EnhancedMVPAPipeline
from global_config import GlobalConfig

# 创建默认配置
config = GlobalConfig.create_default_config()

# 可选：自定义配置
config.lss.subjects = ["sub-01", "sub-02", "sub-03"]
config.lss.runs = [1, 2, 3, 4]
config.mvpa.contrasts = [
    ("face_vs_house", "face", "house"),
    ("face_vs_object", "face", "object")
]

# 创建增强版流水线
pipeline = EnhancedMVPAPipeline(config=config)

# 运行完整的增强版分析
results = pipeline.run_complete_analysis_enhanced(
    create_rois=True,
    enable_quality_control=True,
    enable_visualization=True
)

# 查看结果摘要
print(f"LSS分析: {results['lss_results'].get('summary', {})}")
print(f"ROI创建: {len(results['roi_results'].get('created_rois', []))} 个ROI")
print(f"MVPA分析: {results['mvpa_results'].get('summary', {})}")
print(f"总耗时: {results['analysis_info']['total_duration']}")

# 清理资源
pipeline.cleanup()
```

### 2. 单独组件使用

```python
from lss_optimized import run_optimized_lss_analysis
from roi_mask_creator import create_standard_rois
from mvpa_enhanced import run_enhanced_mvpa_analysis
from global_config import GlobalConfig

config = GlobalConfig.create_default_config()

# 1. 优化版LSS分析
lss_results = run_optimized_lss_analysis(
    config=config.lss,
    subjects=config.lss.subjects,
    runs=config.lss.runs
)

# 2. ROI mask创建
roi_results = create_standard_rois(
    config=config.roi,
    roi_types=['sphere', 'anatomical', 'activation']
)

# 3. 增强版MVPA分析
mvpa_results = run_enhanced_mvpa_analysis(config)
```

### 3. 自定义配置

```python
from global_config import GlobalConfig

# 创建自定义配置
config = GlobalConfig.create_default_config()

# LSS配置优化
config.lss.n_jobs_glm = 8  # 增加GLM并行数
config.lss.enable_parallel_subjects = True
config.lss.memory_efficient = True

# ROI配置自定义
config.roi.sphere_coords = [(0, 0, 0), (20, -20, 10), (-20, -20, 10)]
config.roi.sphere_radius = 8  # 8mm半径
config.roi.create_anatomical_rois = True

# MVPA配置优化
config.mvpa.cv_folds = 10  # 10折交叉验证
config.mvpa.n_permutations = 1000  # 1000次置换检验
config.mvpa.use_parallel = True
config.mvpa.n_jobs = 8

# 保存配置
config.save_config("./my_custom_config.json")

# 使用自定义配置
pipeline = EnhancedMVPAPipeline(config=config)
results = pipeline.run_complete_analysis_enhanced()
```

## ⚙️ 配置说明 v2.0

### 配置文件结构

增强版v2.0采用分层配置结构：

```python
# 全局配置
class GlobalConfig:
    version: str = "2.0.0"
    project_name: str = "Enhanced MVPA Pipeline"
    output_dir: str = "./enhanced_mvpa_results"
    cache_dir: str = "./cache"
    temp_dir: str = "./temp"
    
    # 性能监控
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    
    # 子配置
    lss: LSSConfig
    roi: ROIConfig  
    mvpa: MVPAConfig
```

### 主要配置参数

#### LSS配置增强版
```python
class LSSConfig:
    # 基础参数
    data_dir: str = "/path/to/bids/data"
    subjects: List[str] = field(default_factory=lambda: ["sub-01", "sub-02"])
    runs: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    # 优化参数
    n_jobs_glm: int = 4
    enable_parallel_subjects: bool = True
    enable_parallel_runs: bool = True
    memory_efficient: bool = True
    use_optimized_glm: bool = True
```

#### ROI配置增强版
```python
class ROIConfig:
    # ROI创建参数
    sphere_coords: List[Tuple[int, int, int]] = field(default_factory=list)
    sphere_radius: int = 6  # mm
    create_anatomical_rois: bool = True
    create_activation_rois: bool = False
    
    # 质量控制
    min_voxels: int = 10
    max_voxels: int = 1000
    quality_threshold: float = 0.8
```

#### MVPA配置增强版
```python
class MVPAConfig:
    # 分析参数
    contrasts: List[Tuple[str, str, str]] = field(default_factory=list)
    cv_folds: int = 5
    n_permutations: int = 100
    
    # 性能优化
    use_parallel: bool = True
    n_jobs: int = -1
    memory_cache: bool = True
    batch_size: int = 50
```

## 📊 输出结果 v2.0

### 增强版目录结构
```
output_directory/
├── config/                     # 配置文件
│   ├── global_config.json     # 全局配置快照
│   └── analysis_params.json   # 分析参数记录
├── logs/                      # 日志文件
│   ├── pipeline.log          # 主流水线日志
│   ├── lss_analysis.log      # LSS分析日志
│   ├── roi_creation.log      # ROI创建日志
│   └── mvpa_analysis.log     # MVPA分析日志
├── lss_results/              # LSS分析结果
│   ├── summary.json          # LSS结果摘要
│   ├── beta_maps/            # Beta系数图
│   └── quality_metrics/      # 质量指标
├── roi_masks/                # ROI掩模文件
│   ├── sphere_rois/          # 球形ROI
│   ├── anatomical_rois/      # 解剖ROI
│   ├── activation_rois/      # 激活ROI
│   └── quality_report.html   # ROI质量报告
├── mvpa_results/             # MVPA分析结果
│   ├── classification_results.json  # 分类结果
│   ├── statistical_maps/     # 统计图
│   ├── cross_validation/     # 交叉验证结果
│   └── permutation_tests/    # 置换检验结果
├── quality_control/          # 质量控制
│   ├── comprehensive_qc.html # 综合质量报告
│   ├── data_quality/         # 数据质量评估
│   └── analysis_validation/  # 分析验证
├── visualizations/           # 可视化文件
│   ├── brain_maps/           # 脑图可视化
│   ├── statistical_plots/    # 统计图表
│   ├── interactive_dashboard.html  # 交互式仪表板
│   └── summary_plots/        # 摘要图表
├── reports/                  # 综合报告
│   ├── comprehensive_report.html   # 完整HTML报告
│   ├── summary_report.html   # 摘要报告
│   └── analysis_summary.json # 分析摘要JSON
├── performance/              # 性能监控
│   ├── performance_stats.json    # 性能统计
│   ├── resource_usage.json  # 资源使用情况
│   └── timing_analysis.json # 时间分析
└── cache/                    # 缓存文件
    ├── lss_cache/           # LSS缓存
    ├── roi_cache/           # ROI缓存
    └── mvpa_cache/          # MVPA缓存
```

### 主要输出文件

1. **综合分析报告** (`reports/comprehensive_report.html`)
   - 完整的分析流程和结果
   - 交互式图表和统计结果
   - 质量控制和验证信息
   - 性能统计和资源使用

2. **质量控制报告** (`quality_control/comprehensive_qc.html`)
   - 数据质量评估结果
   - ROI质量验证
   - 分析结果一致性检查
   - 质量评级和建议

3. **性能监控结果** (`performance/`)
   - 各阶段耗时统计
   - 内存和CPU使用情况
   - 并行效率分析
   - 优化建议

## 🔧 高级功能 v2.0

### 1. 性能优化

```python
# 启用所有性能优化
config = GlobalConfig.create_default_config()
config.lss.use_optimized_glm = True
config.lss.enable_parallel_subjects = True
config.lss.enable_parallel_runs = True
config.lss.memory_efficient = True

config.mvpa.use_parallel = True
config.mvpa.memory_cache = True
config.mvpa.batch_size = 100

# 设置环境变量优化
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
```

### 2. 错误恢复

```python
# 错误恢复示例
try:
    results = pipeline.run_complete_analysis_enhanced()
except Exception as e:
    print(f"分析失败: {e}")
    
    # 检查错误统计
    error_stats = pipeline.performance_stats.get('errors', [])
    print(f"错误数量: {len(error_stats)}")
    
    # 尝试恢复
    config.lss.subjects = ["sub-01", "sub-02"]  # 使用有效的被试ID
    recovery_results = pipeline.run_complete_analysis_enhanced()
    print("错误恢复成功！")
```

### 3. 批处理分析

```python
# 批处理多个分析配置
analysis_configs = [
    {
        'name': 'analysis_1',
        'subjects': ['sub-01', 'sub-02'],
        'runs': [1, 2],
        'contrasts': [('face_vs_house', 'face', 'house')]
    },
    {
        'name': 'analysis_2',
        'subjects': ['sub-03', 'sub-04'], 
        'runs': [3, 4],
        'contrasts': [('face_vs_object', 'face', 'object')]
    }
]

all_results = {}
for analysis_config in analysis_configs:
    config = GlobalConfig.create_default_config()
    config.lss.subjects = analysis_config['subjects']
    config.lss.runs = analysis_config['runs']
    config.mvpa.contrasts = analysis_config['contrasts']
    
    pipeline = EnhancedMVPAPipeline(config=config)
    results = pipeline.run_complete_analysis_enhanced()
    all_results[analysis_config['name']] = results
    pipeline.cleanup()
```

### 4. 配置比较

```python
# 比较不同配置的性能
config_standard = GlobalConfig.create_default_config()
config_optimized = GlobalConfig.create_default_config()

# 优化配置
config_optimized.lss.n_jobs_glm = 8
config_optimized.lss.enable_parallel_subjects = True
config_optimized.mvpa.n_jobs = 8
config_optimized.mvpa.use_parallel = True

configs = {
    'standard': config_standard,
    'optimized': config_optimized
}

comparison_results = {}
for config_name, config in configs.items():
    pipeline = EnhancedMVPAPipeline(config=config)
    
    import time
    start_time = time.time()
    results = pipeline.run_complete_analysis_enhanced()
    end_time = time.time()
    
    comparison_results[config_name] = {
        'duration': end_time - start_time,
        'performance_stats': results.get('performance_stats', {})
    }
    
    pipeline.cleanup()

# 打印比较结果
for config_name, result in comparison_results.items():
    print(f"{config_name}: {result['duration']:.2f} 秒")
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 启用内存优化
   config.lss.memory_efficient = True
   config.mvpa.batch_size = 20  # 减少批处理大小
   config.mvpa.memory_cache = False  # 禁用内存缓存
   ```

2. **并行处理问题**
   ```python
   # 减少并行作业数
   config.lss.n_jobs_glm = 2
   config.mvpa.n_jobs = 2
   
   # 禁用某些并行功能
   config.lss.enable_parallel_subjects = False
   config.mvpa.use_parallel = False
   ```

3. **数据路径错误**
   ```python
   # 验证路径
   config.validate_paths()
   
   # 检查数据结构
   from pathlib import Path
   data_dir = Path(config.lss.data_dir)
   print(list(data_dir.glob("sub-*/func/*.nii.gz")))
   ```

### 调试模式

```python
# 启用详细日志
config.log_level = "DEBUG"
config.enable_performance_monitoring = True

# 使用小数据集测试
config.lss.subjects = ["sub-01"]  # 仅一个被试
config.lss.runs = [1]             # 仅一个run

# 运行测试分析
test_results = pipeline.run_complete_analysis_enhanced(
    create_rois=False,
    enable_quality_control=False,
    enable_visualization=False
)
```

## 📚 示例和教程

查看 `example_usage.py` 文件获取v2.0的详细使用示例：

- **增强版流水线v2.0完整示例**: 展示最新功能的完整使用
- **单独组件使用示例**: 如何单独使用LSS、ROI、MVPA模块
- **自定义配置示例**: 高级配置和参数调优
- **性能优化示例**: 最大化分析性能的配置
- **错误恢复示例**: 处理分析过程中的错误
- **批处理示例**: 大规模数据的批量处理
- **配置比较示例**: 不同配置的性能对比

## 🆕 v2.0 新特性

### 算法升级
- 集成了最新的LSS单trial分析算法
- 优化的ROI创建和质量控制算法
- 增强的MVPA分类和统计分析方法

### 性能提升
- 多级并行处理架构
- 智能内存管理和缓存策略
- 优化的数据流和计算效率

### 用户体验
- 统一的配置管理系统v2.0
- 增强的错误处理和恢复机制
- 实时进度监控和性能统计
- 综合质量控制和验证

### 扩展功能
- 批处理和配置比较功能
- 交互式可视化和报告生成
- 模块化设计支持灵活扩展

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 开发指南
- 遵循PEP 8代码风格
- 添加适当的文档字符串
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 📞 支持和反馈

如有问题或建议，请：

1. 查看本文档的故障排除部分
2. 检查 `example_usage.py` 中的v2.0示例
3. 查看日志文件获取详细错误信息
4. 提交 Issue 或 Pull Request

## 🔄 更新日志

### v2.0.0 (当前版本)
- **重大更新**: 集成最新LSS、ROI和MVPA优化算法
- **新增**: OptimizedLSSAnalyzer - 优化版LSS分析器
- **新增**: OptimizedROIMaskCreator - 优化版ROI创建器  
- **新增**: EnhancedMVPAAnalyzer - 增强版MVPA分析器
- **升级**: GlobalConfig v2.0 - 统一配置管理系统
- **升级**: EnhancedMVPAPipeline v2.0 - 增强版主流水线
- **新增**: run_complete_analysis_enhanced - 完整增强版分析方法
- **优化**: 多级并行处理和智能内存管理
- **增强**: 综合质量控制和结果验证
- **新增**: 性能监控和错误恢复机制
- **扩展**: 批处理和配置比较功能

### v1.0.0
- 初始版本发布
- 基础MVPA分析流水线
- 数据流优化和缓存机制
- 错误处理和质量控制
- 可视化和报告生成

---

**注意**: 这是增强版MVPA分析流水线v2.0，集成了最新的优化算法和先进功能。使用前请确保您的数据符合BIDS格式，并根据实际需求调整配置参数。建议从示例开始，逐步熟悉各项功能。