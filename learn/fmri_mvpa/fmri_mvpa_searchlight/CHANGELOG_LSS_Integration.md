# LSS数据集成更新日志

## 版本: LSS Integration v1.0
**日期**: 2024年
**作者**: Assistant

## 概述

本次更新将fMRI MVPA Searchlight分析工具与LSS (Least Squares Separate) 数据完全集成，使其能够直接分析来自 `fmri_lss_enhance.py` 的trial-wise beta图像。

## 主要更改

### 1. 配置类更新 (SearchlightMVPAConfig)

#### 路径配置适配
- **更新**: `self.lss_root = Path(r"../../learn_LSS")`
- **说明**: 匹配 `fmri_lss_enhance.py` 的 `OUTPUT_ROOT` 设置
- **影响**: 自动指向正确的LSS数据目录

#### 被试和实验设置适配
- **更新**: `self.subjects = [f"sub-{i:02d}" for i in range(1, 29)]`
- **更新**: `self.runs = [3, 4]`
- **说明**: 匹配LSS分析的被试范围和run设置
- **影响**: 确保分析范围与LSS数据一致

### 2. 数据加载函数更新

#### load_lss_trial_data() 函数增强
- **新增**: 详细的LSS数据结构检查
- **新增**: 错误处理和日志记录
- **更新**: 适配LSS的文件命名格式 `beta_trial_{trial_index:03d}.nii.gz`
- **新增**: 添加更多元数据字段 (`subject`, `beta_file_path`)
- **影响**: 更稳定和详细的数据加载过程

#### load_multi_run_lss_data() 函数保持
- **状态**: 无需修改，已兼容LSS数据结构
- **功能**: 继续支持多run数据合并

### 3. 数据处理函数更新

#### prepare_classification_data() 函数重构
- **关键更新**: 使用 `'original_condition'` 字段替代 `'condition'`
- **新增**: 使用 `'beta_file_path'` 字段获取beta图像路径
- **新增**: 更详细的错误信息和可用条件显示
- **新增**: beta图像数量验证
- **影响**: 完全适配LSS数据格式，提高数据处理的可靠性

### 4. 文档更新

#### README.md 增强
- **新增**: LSS数据集成章节
- **新增**: LSS数据结构说明
- **新增**: 使用示例和代码片段
- **新增**: LSS数据字段说明
- **新增**: 示例脚本使用指南
- **更新**: 注意事项和故障排除

### 5. 新增文件

#### test_lss_integration.py
- **功能**: 全面测试LSS数据集成
- **包含**: 配置测试、数据加载测试、分类数据准备测试
- **包含**: LSS目录结构检查
- **用途**: 验证集成是否正常工作

#### example_lss_searchlight_analysis.py
- **功能**: LSS数据分析的完整示例
- **包含**: 单mask分析示例
- **包含**: 多mask分析示例
- **包含**: 自定义配置分析示例
- **包含**: 交互式选择界面

#### CHANGELOG_LSS_Integration.md
- **功能**: 记录所有LSS集成相关的更改
- **用途**: 版本控制和更新追踪

## 技术细节

### LSS数据格式适配

#### 目录结构匹配
```
../../learn_LSS/          # 匹配fmri_lss_enhance.py的OUTPUT_ROOT
├── sub-01/
│   ├── run-3_LSS/        # 匹配LSS的命名格式
│   │   ├── beta_trial_001.nii.gz  # 匹配LSS的文件命名
│   │   ├── trial_info.csv         # LSS输出的元数据
│   │   └── mask.nii.gz           # LSS生成的脑掩模
│   └── run-4_LSS/
└── ...
```

#### 数据字段映射
| LSS字段 | Searchlight使用 | 说明 |
|---------|----------------|------|
| `trial_index` | trial识别 | 1-based索引 |
| `original_condition` | 条件筛选 | 替代原来的'condition' |
| `beta_file` | 文件名参考 | beta图像文件名 |
| `onset` | 时间信息 | 事件开始时间 |
| `duration` | 时间信息 | 事件持续时间 |

### 向后兼容性

- **保持**: 所有原有的多mask分析功能
- **保持**: 原有的配置选项和参数
- **保持**: 原有的输出格式和可视化
- **增强**: 数据加载的错误处理和日志记录

### 性能优化

- **优化**: 使用LSS提供的完整文件路径，减少路径构建开销
- **优化**: 增加数据验证步骤，减少运行时错误
- **优化**: 更详细的进度日志，便于监控长时间运行的分析

## 使用指南

### 快速开始

1. **确保LSS数据可用**:
   ```bash
   ls ../../learn_LSS/sub-01/run-3_LSS/
   # 应该看到: beta_trial_*.nii.gz, trial_info.csv, mask.nii.gz
   ```

2. **运行测试**:
   ```bash
   python test_lss_integration.py
   ```

3. **运行示例分析**:
   ```bash
   python example_lss_searchlight_analysis.py
   ```

### 自定义分析

```python
from fmri_mvpa_searchlight_pipeline_enhanced import SearchlightMVPAConfig, run_group_searchlight_analysis

# 创建配置
config = SearchlightMVPAConfig()

# 根据LSS数据设置对比条件
config.contrasts = [
    ('condition_a_vs_b', 'condition_a', 'condition_b'),
    # 条件名需要匹配LSS数据中的original_condition字段
]

# 运行分析
results = run_group_searchlight_analysis(config)
```

## 验证和测试

### 测试覆盖

- ✅ 配置设置验证
- ✅ LSS目录结构检查
- ✅ 单被试数据加载
- ✅ 多run数据合并
- ✅ 分类数据准备
- ✅ 条件筛选和标签生成
- ✅ beta图像路径验证

### 已知限制

1. **条件名依赖**: 对比条件必须匹配LSS数据中的 `original_condition` 字段
2. **路径依赖**: LSS数据必须位于 `../../learn_LSS/` 目录
3. **文件格式**: 依赖LSS的特定文件命名格式

## 未来改进

### 计划中的功能

1. **动态条件检测**: 自动检测LSS数据中的可用条件
2. **路径配置灵活性**: 支持自定义LSS数据路径
3. **批量分析**: 支持多个LSS数据集的批量分析
4. **结果对比**: LSS结果与传统GLM结果的对比分析

### 性能优化计划

1. **内存优化**: 大数据集的内存使用优化
2. **并行优化**: 更高效的并行处理策略
3. **缓存机制**: 中间结果缓存以支持断点续传

## 总结

本次LSS集成更新成功实现了:

1. **完全兼容**: 与 `fmri_lss_enhance.py` 输出的无缝集成
2. **功能保持**: 保留所有原有功能，包括多mask分析
3. **易用性**: 提供详细文档和示例脚本
4. **稳定性**: 增强的错误处理和数据验证
5. **可扩展性**: 为未来功能扩展奠定基础

这使得研究人员可以直接使用LSS分析的trial-wise数据进行高质量的MVPA searchlight分析，大大简化了分析流程并提高了分析效率。