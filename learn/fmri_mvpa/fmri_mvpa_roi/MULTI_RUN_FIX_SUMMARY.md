# 多run MVPA分析修复总结

## 问题描述

在执行多个run的fMRI MVPA分析时，发现以下问题：

1. **run信息丢失**：在trial详细信息中只记录了trial_index、condition、label，没有区分trial来源于哪个run
2. **数据重复风险**：由于缺少run标识，可能导致来自不同run的相同trial_index被误认为是重复数据
3. **索引错误**：在多run合并时，使用原始的trial_index作为beta_images的索引，可能导致索引越界或访问错误的数据
4. **追溯困难**：无法追溯每个trial的具体来源，影响结果解释和质量控制

## 根本原因分析

### 1. 数据加载阶段
- `load_lss_trial_data`函数虽然添加了run信息，但在后续处理中没有充分利用
- `load_multi_run_lss_data`函数合并数据时，没有建立正确的索引映射关系

### 2. 数据准备阶段
- `prepare_classification_data`函数在构建trial_details时，没有传递run相关信息
- 使用`trial['trial_index'] - 1`作为beta_images索引，在多run合并场景下不正确

### 3. 索引管理问题
- 缺少统一的索引管理机制
- 没有区分原始trial_index和合并后的索引

## 修复方案

### 1. 增强数据加载功能

#### 在`load_lss_trial_data`中添加run信息：
```python
# 添加run信息到trial数据中
trial_with_run = trial.copy()
trial_with_run['run'] = run
trial_with_run['global_trial_index'] = f"run{run}_trial{trial['trial_index']}"
valid_trials.append(trial_with_run)
```

#### 在`load_multi_run_lss_data`中添加合并索引：
```python
# 重新索引trial，确保每个trial有唯一标识
combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
```

### 2. 修复数据准备逻辑

#### 正确的索引使用：
```python
# 对于多run合并数据，使用combined_trial_index；对于单run数据，使用trial_index-1
if 'combined_trial_index' in trial:
    trial_idx = trial['combined_trial_index']
else:
    trial_idx = trial['trial_index'] - 1
```

#### 完整的trial信息保留：
```python
trial_detail = {
    'trial_index': trial['trial_index'],
    'condition': trial['original_condition'],
    'label': label
}
# 添加run信息（如果存在）
if 'run' in trial:
    trial_detail['run'] = trial['run']
if 'global_trial_index' in trial:
    trial_detail['global_trial_index'] = trial['global_trial_index']
if 'combined_trial_index' in trial:
    trial_detail['combined_trial_index'] = trial['combined_trial_index']
```

### 3. 建立多层索引体系

1. **原始索引**：`trial_index` - 每个run内的原始trial编号
2. **全局索引**：`global_trial_index` - 格式为"run{run}_trial{trial_index}"的唯一标识
3. **合并索引**：`combined_trial_index` - 多run合并后的连续索引，用于访问beta_images
4. **run标识**：`run` - 标识trial来源的run编号

## 修复效果验证

### 测试结果

#### 单run分析测试：
- ✅ 正确加载run信息
- ✅ 生成global_trial_index
- ✅ trial详细信息完整

#### 多run分析测试：
- ✅ 正确合并多个run的数据
- ✅ 各run的trial分布正确
- ✅ combined_trial_index连续且正确
- ✅ global_trial_index唯一性
- ✅ 没有数据重复

#### 数据完整性测试：
- ✅ 所有必需字段存在
- ✅ run信息完整保留
- ✅ 索引正确性验证
- ✅ 数据类型正确

### 演示结果示例

```
多run数据加载结果:
  - 合并后的beta文件数: 72
  - 合并后的trial info形状: (72, 7)
  - 各run的trial分布: {3: 24, 4: 20, 5: 28}
  - 总体条件分布: {'yy': 36, 'kj': 36}

分类数据准备:
  - 选择的trial数: 72
  - 标签分布: yy=36, kj=36
  - 分类数据中各run的trial数: {3: 24, 4: 20, 5: 28}
  - Global trial index唯一性: 72/72
    ✅ 没有重复的trial
```

## 技术改进

### 1. 数据结构优化
- 建立了完整的索引体系
- 保留了所有必要的元数据
- 确保了数据的可追溯性

### 2. 错误预防
- 防止了索引越界错误
- 避免了数据重复问题
- 提高了数据处理的鲁棒性

### 3. 功能增强
- 支持任意数量的run合并
- 兼容单run和多run分析
- 提供了详细的数据追溯信息

## 使用建议

### 1. 配置多run分析
```python
config = MVPAConfig()
config.runs = [3, 4, 5]  # 指定要分析的run
```

### 2. 检查数据完整性
- 运行`test_multi_run_fix.py`验证修复效果
- 运行`example_multi_run_analysis.py`查看演示

### 3. 结果文件说明
- trial详细信息文件现在包含完整的run信息
- 文件命名格式：`{subject}_{contrast}_run3_run4_run5_trial_details.csv`
- 包含列：trial_index, condition, label, run, global_trial_index, combined_trial_index

### 4. 质量控制
- 检查各run的trial分布是否合理
- 验证global_trial_index的唯一性
- 确认combined_trial_index的连续性

## 文件修改清单

### 主要修改文件
1. **fmri_mvpa_roi_pipeline_enhanced.py**
   - 修改`prepare_classification_data`函数
   - 增强trial信息保留逻辑
   - 修复索引使用错误

### 新增测试文件
2. **test_multi_run_fix.py**
   - 单run数据加载测试
   - 多run数据合并测试
   - 分类数据准备测试
   - 数据完整性验证

3. **example_multi_run_analysis.py**
   - 单run分析演示
   - 多run分析演示
   - 数据完整性检查演示

4. **MULTI_RUN_FIX_SUMMARY.md**
   - 问题分析和修复方案文档
   - 使用指南和最佳实践

## 后续建议

### 1. 进一步优化
- 考虑添加run间效应的建模
- 实现更灵活的run选择机制
- 添加run质量评估功能

### 2. 文档完善
- 更新README文档
- 添加多run分析的最佳实践指南
- 提供更多实际数据的使用示例

### 3. 测试扩展
- 添加更多边界情况的测试
- 实现自动化的回归测试
- 添加性能基准测试

---

**修复完成时间**：2024年10月18日  
**修复版本**：Enhanced Pipeline v2.1  
**测试状态**：✅ 全部通过  
**文档状态**：✅ 已完成  

🎉 **多run MVPA分析的run信息记录和数据重复问题已完全修复！**