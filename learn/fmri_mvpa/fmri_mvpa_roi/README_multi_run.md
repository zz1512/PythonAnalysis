# fMRI MVPA ROI 多run合并分析说明

## 概述

增强版的fMRI MVPA ROI分析脚本现在支持多run数据合并分析。这意味着您可以将来自不同run的trial数据合并在一起进行分类分析，从而增加样本量并提高统计功效。

## 主要特性

### 1. 多run数据合并
- 自动加载并合并指定runs的所有trial数据
- 保持每个trial的原始run信息
- 为每个trial分配唯一的全局标识符
- 自动统计各条件的trial数量

### 2. 兼容性
- 完全向后兼容单run分析
- 自动检测是单run还是多run配置
- 保持原有的所有分析功能

### 3. 增强的日志和报告
- 详细记录每个run的数据加载情况
- 在HTML报告中显示使用的runs信息
- 保存的trial详细信息文件包含run标识

## 使用方法

### 配置多run分析

```python
from fmri_mvpa_roi_pipeline_enhanced import MVPAConfig, run_group_roi_analysis_enhanced

# 创建配置
config = MVPAConfig()

# 设置要合并的runs
config.runs = [3, 4]  # 合并run3和run4的数据

# 其他配置保持不变
config.subjects = [f"sub-{i:02d}" for i in range(1, 5)]
config.contrasts = [('metaphor_vs_space', 'yy', 'kj')]

# 运行分析
results = run_group_roi_analysis_enhanced(config)
```

### 数据结构要求

您的数据应该按以下结构组织：

```
LSS_root/
├── sub-01/
│   ├── run-3_LSS/
│   │   ├── trial_map.csv
│   │   ├── beta_trial_001.nii.gz
│   │   ├── beta_trial_002.nii.gz
│   │   └── ...
│   └── run-4_LSS/
│       ├── trial_map.csv
│       ├── beta_trial_001.nii.gz
│       ├── beta_trial_002.nii.gz
│       └── ...
├── sub-02/
│   └── ...
└── ...
```

### 示例：run3和run4合并分析

假设您有：
- run3：每个被试70个trial
- run4：每个被试70个trial
- 合并后：每个被试140个trial

```python
# 运行示例脚本
python example_multi_run_analysis.py
```

## 输出文件说明

### 1. Trial详细信息文件
- 文件名格式：`{subject}_{contrast}_run3_run4_trial_details.csv`
- 包含每个trial的run信息、条件标签等

### 2. HTML报告
- 显示使用的runs信息："Runs: run-3, run-4 (Multi-run Combined Analysis)"
- 包含合并后的统计结果

### 3. 日志信息
```
[2024-01-01 12:00:00] [MVPA] 开始加载 sub-01 的多run数据: runs [3, 4]
[2024-01-01 12:00:01] [MVPA]   成功加载 run-3: 70个trial
[2024-01-01 12:00:02] [MVPA]   成功加载 run-4: 70个trial
[2024-01-01 12:00:03] [MVPA] 合并完成 sub-01: 总共140个trial (来自2个run)
[2024-01-01 12:00:04] [MVPA]   条件统计: {'yy': 70, 'kj': 70}
```

## 优势和注意事项

### 优势
1. **增加样本量**：合并多个run可以显著增加每个条件的trial数量
2. **提高统计功效**：更多的样本有助于提高分类准确率和统计显著性
3. **更稳定的结果**：减少由于单个run数据质量问题导致的分析失败

### 注意事项
1. **数据质量**：确保所有run的数据质量相当，避免某个run的噪声影响整体结果
2. **时间效应**：考虑run之间可能存在的时间效应或学习效应
3. **内存使用**：合并多个run会增加内存使用，大数据集时需要注意
4. **计算时间**：更多的trial意味着更长的计算时间

## 数据检查

在运行分析前，建议使用以下代码检查数据可用性：

```python
from example_multi_run_analysis import check_data_availability
check_data_availability()
```

这将检查：
- 每个run的目录是否存在
- trial_map.csv文件是否存在
- beta文件的完整性
- 各条件的trial分布

## 故障排除

### 常见问题

1. **某个run数据缺失**
   - 脚本会自动跳过缺失的run并继续分析
   - 日志中会显示跳过的run信息

2. **trial数量不匹配**
   - 不同run的trial数量可以不同
   - 脚本会自动处理并在日志中报告实际数量

3. **内存不足**
   - 减少batch_size参数
   - 关闭并行处理（设置use_parallel=False）
   - 减少同时分析的ROI数量

### 调试建议

1. 首先运行单run分析确保基本功能正常
2. 使用较少的被试和ROI进行测试
3. 检查日志文件中的详细错误信息
4. 确保所有依赖库版本兼容

## 性能优化

对于大规模多run分析，建议：

1. **启用并行处理**：`config.use_parallel = True`
2. **调整批处理大小**：`config.batch_size = 50`
3. **使用内存缓存**：`config.memory_cache = 'nilearn_cache'`
4. **合理设置进程数**：`config.n_jobs = 4`

## 联系和支持

如果您在使用多run合并分析功能时遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看生成的日志文件
3. 运行数据检查脚本确认数据完整性
4. 提供详细的错误信息和数据结构描述