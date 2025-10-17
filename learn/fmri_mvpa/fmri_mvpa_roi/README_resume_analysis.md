# 断点续传MVPA ROI分析使用指南

## 概述

`fmri_mvpa_roi_resume_analysis.py` 是一个支持断点续传功能的MVPA ROI分析脚本，可以在分析中断后从指定被试继续分析，并生成包含所有被试（1-28）的完整报告。

## 主要功能

1. **断点续传**: 从指定被试开始继续分析
2. **结果合并**: 自动合并之前的分析结果
3. **进度保存**: 自动保存分析进度和中间结果
4. **完整报告**: 生成包含所有被试的统计报告
5. **错误恢复**: 记录失败的被试，便于后续处理

## 使用方法

### 1. 从第17个被试继续分析

```bash
cd /Users/bytedance/PycharmProjects/PythonAnalysis/learn/fmri_mvpa/fmri_mvpa_roi
python fmri_mvpa_roi_resume_analysis.py --resume-from sub-17
```

### 2. 分析指定范围的被试

```bash
# 分析被试1-28（默认）
python fmri_mvpa_roi_resume_analysis.py

# 分析被试10-25
python fmri_mvpa_roi_resume_analysis.py --start-subject 10 --end-subject 25

# 从被试20开始分析到28
python fmri_mvpa_roi_resume_analysis.py --resume-from sub-20 --end-subject 28
```

### 3. 命令行参数说明

- `--resume-from`: 从指定被试开始继续分析（例如：sub-17）
- `--start-subject`: 起始被试编号（默认：1）
- `--end-subject`: 结束被试编号（默认：28）

## 输出文件说明

### 主要结果文件

1. **group_roi_mvpa_results_complete.csv**: 包含所有被试的完整MVPA结果
2. **group_statistics_enhanced.csv**: 组水平统计分析结果
3. **analysis_report.html**: 详细的HTML分析报告
4. **analysis_summary.txt**: 分析总结报告

### 备份和进度文件

1. **backup/subject_results.pkl**: 所有被试结果的备份文件
2. **backup/analysis_progress.json**: 分析进度记录
3. **analysis_config.json**: 分析配置参数

## 工作流程

### 第一次运行（完整分析）
```bash
python fmri_mvpa_roi_resume_analysis.py
```

### 中断后继续分析
如果在第17个被试时中断：
```bash
python fmri_mvpa_roi_resume_analysis.py --resume-from sub-17
```

### 检查分析进度
脚本会自动显示：
- 已完成的被试列表
- 失败的被试列表
- 当前分析进度

## 特殊情况处理

### 1. 数据修复后重新分析特定被试

如果某个被试的数据有问题，修复后需要重新分析：

```python
# 删除该被试的结果，重新分析
import pickle
from pathlib import Path

# 加载结果
results_file = Path("../../../learn_mvpa/metaphor_ROI_MVPA_enhanced/backup/subject_results.pkl")
with open(results_file, 'rb') as f:
    results = pickle.load(f)

# 删除有问题的被试结果
if 'sub-17' in results:
    del results['sub-17']
    
# 保存修改后的结果
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

# 然后重新运行分析
```

### 2. 强制重新分析所有被试

```bash
# 删除备份文件，重新开始
rm -rf ../../../learn_mvpa/metaphor_ROI_MVPA_enhanced/backup/
python fmri_mvpa_roi_resume_analysis.py
```

### 3. 只分析失败的被试

查看 `backup/analysis_progress.json` 文件中的 `failed_subjects` 列表，然后：

```bash
# 假设sub-15和sub-20失败了
python fmri_mvpa_roi_resume_analysis.py --resume-from sub-15 --end-subject 15
python fmri_mvpa_roi_resume_analysis.py --resume-from sub-20 --end-subject 20
```

## 结果验证

### 检查完整性

```python
import pandas as pd

# 读取完整结果
df = pd.read_csv("../../../learn_mvpa/metaphor_ROI_MVPA_enhanced/group_roi_mvpa_results_complete.csv")

# 检查被试数量
print(f"总被试数: {df['subject'].nunique()}")
print(f"被试列表: {sorted(df['subject'].unique())}")

# 检查每个被试的ROI数量
subject_roi_counts = df.groupby('subject')['roi'].nunique()
print(f"每个被试的ROI数量: {subject_roi_counts.describe()}")
```

### 查看统计结果

```python
# 读取统计结果
stats_df = pd.read_csv("../../../learn_mvpa/metaphor_ROI_MVPA_enhanced/group_statistics_enhanced.csv")

# 显著结果
sig_results = stats_df[stats_df['significant_corrected']]
print(f"显著的ROI数量: {len(sig_results)}")
print(sig_results[['roi', 'mean_accuracy', 't_stat', 'p_corrected']])
```

## 注意事项

1. **数据路径**: 确保LSS数据路径正确（`../../../learn_LSS`）
2. **ROI掩模**: 确保ROI掩模文件存在（`../../../learn_mvpa/full_roi_mask`）
3. **内存管理**: 大样本分析时注意内存使用，脚本会自动进行内存清理
4. **并行处理**: 默认使用多核并行处理，可在配置中调整
5. **备份重要**: 定期备份 `backup/` 目录中的文件

## 故障排除

### 常见错误

1. **路径错误**: 检查数据路径是否正确
2. **权限问题**: 确保有写入结果目录的权限
3. **内存不足**: 减少并行进程数或批处理大小
4. **数据缺失**: 检查特定被试的LSS数据是否完整

### 日志查看

```bash
# 查看分析日志
tail -f ../../../learn_mvpa/metaphor_ROI_MVPA_enhanced/analysis.log
```

## 联系支持

如果遇到问题，请检查：
1. 错误日志信息
2. 数据文件完整性
3. 配置参数设置
4. 系统资源使用情况