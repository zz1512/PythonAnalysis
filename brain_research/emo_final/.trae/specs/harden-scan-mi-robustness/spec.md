# 强化 SCAN 映射与成熟度指标稳健性 Spec

## Why
当前 Phase 2 叙事已具备顶刊结构，但审稿高风险点集中在：SCAN 映射的可辩护性与成熟模板（MI）的稳定性证据链不足。需要用最小增量补强关键方法学脆弱点，提升可复现性与审稿抗性。

## What Changes
- 增强 SCAN 映射：支持离线 atlas、阈值敏感性、映射质量报告与可视化产出
- 增强 SCAN 结果稳健性：加入随机/负对照映射基线，量化“SCAN 效应是否高于随机”
- 增强 MI 稳健性：成熟阈值敏感性、bootstrap 模板稳定性、leave-one-out 模板影响
- 增强行为锚定：提供重评成功度（success）指标的默认构造方案（如可配对）
- 增强 ISC 解释排他性：增加表征复杂性/区分度指标，区分“成熟化一致”与“低维简化”
- 增强发育模式表达：对关键结果增加非线性拟合选项（作为补充/敏感性）
- 文档同步：更新 `readall.md` 中与上述新增输出/参数一致的说明（不改变既有主线叙事）

## Impact
- Affected specs:
  - SCAN ROI 映射与报告（方法学可信度）
  - Step2 梯度+SCAN 结果呈现（敏感性/对照）
  - Step4 成熟度指标（稳定性/阈值敏感性）
  - 行为锚定指标（success 的操作化）
  - ISC 解释（复杂性/区分度补充分析）
- Affected code:
  - `utils_network_assignment.py`
  - `step1_task_dissociation.py`
  - `step2_gradient_shift.py`
  - `step3_representational_connectivity.py`
  - `step4_maturity_index.py`
  - （可选）`behavior/*` 中行为汇总脚本（仅当需要生成被试级 success 指标文件）
  - `readall.md`

## ADDED Requirements

### Requirement: SCAN Mapping Robustness
系统 SHALL 提供可复现、可报告的 SCAN→ROI 映射流程，并输出映射质量指标与敏感性结果。

#### Scenario: 离线 atlas 成功
- **WHEN** 用户提供本地 Schaefer atlas（或其 centroid 缓存文件）
- **THEN** `get_scan_roi_mapping()` 不依赖在线下载即可运行并产出映射表

#### Scenario: 阈值敏感性分析
- **WHEN** 用户指定多个阈值（例如 10/15/20/25mm）
- **THEN** 系统输出每个阈值下的命中 ROI 列表、距离分布摘要、以及主要统计量对阈值的稳定性摘要

#### Scenario: 映射质量报告
- **WHEN** 用户运行 SCAN QC 报告命令
- **THEN** 系统输出：
  - 每个 SCAN 区域的最近 parcel、距离（mm）、所属网络
  - 全部距离分布（均值/中位数/分位数）
  - “within_threshold” 比例与阈值曲线

### Requirement: SCAN Negative Control Baseline
系统 SHALL 提供至少一种负对照基线，用于证明“SCAN 效应”高于随机映射带来的期望值。

#### Scenario: 随机点/随机 parcel 负对照
- **WHEN** 用户指定随机次数 N（例如 1000）
- **THEN** 系统输出随机基线分布（如均值/分位数）以及观测 SCAN 效应在该分布下的经验 p 值

### Requirement: MI Template Stability
系统 SHALL 对成熟模板的稳定性进行量化，并在输出中报告稳定性指标与阈值敏感性。

#### Scenario: Bootstrap 模板稳定性
- **WHEN** 用户指定 bootstrap 次数 B（例如 500）
- **THEN** 系统输出：
  - 每个被试 MI 的 bootstrap 分布摘要（均值/CI）
  - 全局稳定性指标（例如 MI 的平均 SD / CV）

#### Scenario: 成熟阈值敏感性
- **WHEN** 用户指定多个成熟阈值（例如 16.5/17/17.5/18）
- **THEN** 系统输出每个阈值下的 MI~age、MI~behavior（partial）的关键统计量对照表

### Requirement: Behavior Anchor Enrichment
系统 SHALL 支持至少一种更贴近“重评成功”的行为指标构造方式，并保证与现有 `emot_rating` 兼容。

#### Scenario: 可配对时的 success 指标
- **WHEN** 同一刺激在 Passive 与 Reappraisal 可配对
- **THEN** 系统默认提供 `reappraisal_success = rating_passive - rating_reappraisal` 的被试级聚合指标（均值/中位数等可选）

### Requirement: Representational Complexity Control
系统 SHALL 提供一项或多项表征复杂性/区分度指标，用于辅助解释 ISC 上升的性质。

#### Scenario: 复杂性与年龄/MI 的一致性检验
- **WHEN** 用户对关键 ROI/网络运行复杂性分析
- **THEN** 输出复杂性指标随龄变化，以及与 ISC/MI 的关系（用于排除“低维简化”解释）

## MODIFIED Requirements

### Requirement: Step Scripts Output Schema
现有 step 脚本 SHALL 在不破坏既有输出字段/文件名的前提下，新增补充输出文件（以新文件或新增列的方式），并在控制台打印关键摘要（含 warning，而非静默跳过关键步骤）。

## REMOVED Requirements
（无）

