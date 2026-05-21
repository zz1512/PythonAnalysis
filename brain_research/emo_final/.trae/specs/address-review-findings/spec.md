# 同事 Review 问题修复与稳健性增强 Spec

## Why
同事 review 指出若干确定性风险点（潜在 bug、静默丢弃样本、统计可比性不足），这些问题会直接影响论文可辩护性与结果解释一致性。需要以“最小改动、向后兼容”为原则，补齐断言、记录元信息、输出必要的稳定性指标与对照结果。

## What Changes
- Step2：增强 spin test 结果的可验证性与可追溯性（shape 断言、记录 nulls 结构与 fallback 状态）
- Step4：行为合并改为可配置并显式报告丢失样本数，避免 inner join 静默丢人
- Step3：网络级 RC 输出增加 `n_roi_pairs`，并支持聚合方式（均值/中位数）与方差提示
- Step1：增加方向性一致性检查（仅提示，不强制失败），让“ΔISC 方向约定”与叙事一致性可快速核对
- Step2：fallback（随机置换）在输出中明确标注为 exploratory，并写入 CSV 列
- Step4：`reappraisal_success` 的 suffix 匹配增加白名单过滤，降低误匹配
- Step3：复杂性指标增强为更可解释的 distinctiveness（保留原指标作为补充）
- ISC：Mahalanobis 协方差估计提供可选 shrinkage（默认不改变旧结果）
- 负对照：Step2 负对照从单 SomMot 扩展为多网络可选（默认仍保留现状）
- 工程：可选把 Step1-4 集成进 `run_pipeline.py`（不改变默认运行行为）

## Impact
- Affected specs:
  - 统计可追溯性（spin/fallback 标注、样本丢失报告）
  - 网络级统计可比性（n_roi_pairs、median 聚合）
  - 行为锚定的可辩护性（success 构造更稳健）
  - ISC 估计稳健性（可选 shrinkage）
- Affected code:
  - `step2_gradient_shift.py`
  - `step4_maturity_index.py`
  - `step3_representational_connectivity.py`
  - `step1_task_dissociation.py`
  - `calc_roi_isc_by_age.py`（可选）
  - `run_pipeline.py`（可选）
  - `readall.md`（同步新增列/参数含义）

## ADDED Requirements

### Requirement: Spin Test Robustness Metadata
系统 SHALL 在 Step2 输出中显式记录 spin test 是否真正使用（vs fallback），并对 `nulls` 形状做断言以防 neuromaps 版本变化导致静默错配。

#### Scenario: neuromaps 可用
- **WHEN** neuromaps spin test 成功返回 nulls
- **THEN** 系统断言 `nulls` 的 parcel 维度为 200（与 Schaefer200 对齐），并在结果 CSV 中写入 `spin_test_available=True`、`spin_test_method=alexander_bloch`

#### Scenario: neuromaps 不可用或失败
- **WHEN** spin test 失败并回退到随机置换
- **THEN** 结果 CSV 写入 `spin_test_available=False`、`spin_test_method=random_permutation`，并将该 p 值标注为 exploratory（不应与 spin test p 值等价报告）

### Requirement: Behavior Merge Transparency
系统 SHALL 在 MI 的行为合并中避免静默丢弃被试，并在输出与日志中报告合并后可用于 MI~behavior 的样本量。

#### Scenario: 行为数据缺失
- **WHEN** 行为表缺少部分被试
- **THEN** 默认采用 left join，并打印/写入缺失数量与最终用于行为分析的 n

### Requirement: RC Pair Count Reporting & Robust Aggregation
系统 SHALL 在网络级 RC 统计中输出 `n_roi_pairs`，并支持至少两种聚合方式（mean 与 median）。

#### Scenario: 网络对规模差异
- **WHEN** 不同网络对包含不同数量的 ROI pairs
- **THEN** stats CSV 包含 `n_roi_pairs`；用户可选择 `--rc-aggregate median` 以降低极端值影响

### Requirement: Success Suffix Matching Safety
系统 SHALL 对 `*_passive`/`*_reappraisal` 的 suffix 匹配增加白名单关键词过滤（rating/emot/score 等），以避免误匹配无关列。

### Requirement: Complexity Metric Interpretability
系统 SHALL 提供至少一个更具解释性的复杂性/区分度指标（例如 pattern distinctiveness），并在输出中同时报告旧指标以便对照。

### Requirement: Optional Covariance Shrinkage
系统 SHALL 提供可选 Mahalanobis 协方差 shrinkage 方法（如 Ledoit-Wolf），默认保持现有 ridge+pinv 行为不变。

## MODIFIED Requirements

### Requirement: Output Schema Additivity
所有改动 SHALL 以“新增列/新增文件/新增参数”为主，保证不破坏既有文件名与默认运行结果（除非用户显式启用新参数）。

## REMOVED Requirements
（无）

