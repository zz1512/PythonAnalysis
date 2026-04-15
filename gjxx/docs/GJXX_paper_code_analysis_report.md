# GJXX 论文与代码对应分析报告

**Hippocampal-prefrontal orchestration supports higher-order learning for creative ideation**

*Nature Communications (2026) 17:464*

代码库与论文分析内容的全面对比审查 | 生成日期：2026-04-15

---

## 一、论文概述

### 1.1 研究背景与目的

本研究探究高阶学习（higher-order learning）促进后续创造性构思（creative ideation）的神经认知机制。研究采用"基于输出的定义"框架，通过学习后的创造性结果来定义高阶学习。核心假设：较低维度的海马体表征和较强的海马-vlPFC连接在创造性范例学习过程中将促进后续创造性构思。

### 1.2 实验设计

研究包含两个核心fMRI实验和多个补充行为实验：

- **实验1（Exp. 1）**：文本范例学习，27名行为被试 / 23名fMRI被试，学习阶段在扫描内，创造阶段在扫描外
- **实验2（Exp. 2）**：图片范例学习，39名被试，学习和创造阶段均在扫描内（稀疏采样）
- **补充实验**：Exp. S1a-S1c（对照组）、S2（SME）、S3（定向遗忘）、S4（R/K判断）、S5-S7（孵化效应）

核心范式：创造性范例学习（Phase 1）→ 创造性构思（Phase 2）→ 线索回忆（Phase 3）。每个试次根据后续创造性得分事后分类为高后续创造（HSC）或低后续创造（LSC）。

---

## 二、论文分析内容清单

以下列出论文中所有分析内容及其对应图表，按逻辑分组排列。

| 编号 | 类别 | 分析内容 | 对应图表 | 统计方法 |
|------|------|----------|----------|----------|
| A1 | 行为分析 | HSC vs LSC 主观创意判断差异 | Fig. 2B | 配对t检验 |
| A2 | 行为分析 | HSC vs LSC 线索回忆率差异 | Fig. 2C | 配对t检验 |
| A3 | 行为分析 | 创意范例学习 vs 对照组创造性得分 | Fig. 2D | Kruskal-Wallis |
| A4 | 行为分析 | 模仿-创造权衡分析 | 补充 | 回归分析 |
| B1 | ROI维度 | 左右海马 PCA 维度 (HSC vs LSC) | Fig. 3B-C | PCA + 配对t检验 |
| B2 | ROI GPS | 左右海马全局模式相似性 (HSC vs LSC) | Fig. 3D | Fisher z + 配对t检验 |
| B3 | ROI-ROI DSM | 海马-IFG 表征连接 (HSC vs LSC) | Fig. 3E-F | RDM相关 + 配对t检验 |
| B4 | 相关分析 | 海马维度与海马-IFG连接的负相关 | Fig. 3G | Pearson相关 |
| C1 | ROI维度 | Exp.2 左右海马 PCA 维度复制 | Fig. 4B | PCA + 配对t检验 |
| C2 | 脑-行为回归 | 合并Exp.1+2 维度与创造性表现相关 | Fig. 4C | Pearson相关 |
| D1 | 跨阶段迁移 | 统一PCA空间跨阶段相似性 (HSC vs LSC) | Fig. 5A-B | PCA + Spearman相关 |
| D2 | 跨阶段迁移 | 类别特定PCA空间跨阶段相似性 | Fig. 5C-D | PCA + NPS |
| E1 | 跨阶段迁移 | 前海马全维空间跨阶段相似性 | Fig. 6B | RDM相关 |
| E2 | 跨阶段迁移 | vlPFC全维/压缩空间跨阶段相似性 | Fig. 6C | RDM/PCA相关 |
| E3 | 跨阶段连接 | 海马-vlPFC 压缩/未压缩空间连接 | Fig. 6D-E | RDM相关 + t检验 |
| E4 | 跨阶段解码 | vlPFC SVM跨阶段解码 + 海马回归剥离 | Fig. 6F-G | SVM + 排列检验 |
| F1 | 稳健性 | 单变量激活控制 | 补充 | ANCOVA |
| F2 | 稳健性 | 模仿控制 | 补充 | 配对t检验 |
| F3 | 稳健性 | 项目特定创意潜力控制 | 补充 | 配对t检验 |
| F4 | 稳健性 | 记忆控制（仅分析成功回忆项目） | 补充 | bootstrap + t检验 |
| F5 | 稳健性 | 固定标准分类 | 补充 | 配对t检验 |
| G1 | Searchlight | 全脑 RD searchlight (HSC vs LSC) | 补充 | searchlight + t检验 |
| G2 | 稳健回归 | searchlight表征连接图回归到海马维度 | 补充 | 体素回归 + SVC |
| G3 | 种子连接 | 海马 PCA 连接性 searchlight | 补充 | PCA相似 + t检验 |
| H1 | SME分析 | Exp.S2 记忆成功神经关联 | 补充 | t检验 |
| H2 | 行为补充 | Exp.S3-S7 定向遗忘/R-K/孵化效应 | 补充 | t检验/ANOVA |

---

## 三、代码库功能清单

以下列出代码库中所有模块及其实现的分析功能。

| 文件名 | 阶段 | 功能描述 |
|--------|------|----------|
| events.py | Stage 01 | 事件重标记，排除极端项目 |
| first_level.py | Stage 02-03 | Item-level GLM + Condition-level GLM |
| patterns.py | Stage 04 | 单trial统计图堆叠为4D NIfTI |
| roi.py | Stage 05-06 | GPS分析 + ROI-to-ROI DSM相关 |
| rd.py | Stage 07 | 表征维度(RD)分析，含均衡化和searchlight |
| run_stage08_roi_robustness.py | Stage 08 | 7组ROI维度稳健性检验 |
| run_stage09_remembered_forgotten.py | Stage 09 | Remembered vs Forgotten 维度对比 |
| run_stage10_memory_group.py | Stage 10 | Memory-good vs Memory-poor 维度对比 |
| run_stage11_behavior_pooling.py | Stage 11 | 行为评分pooling维度分析 |
| run_stage12_dimension_searchlight.py | Stage 12 | 全脑维度searchlight分析 |
| run_stage13_seed_connectivity.py | Stage 13 | 海马种子区PCA连接性searchlight |
| group.py | Stage 14 | 组水平体素回归分析 |
| dimension_analysis_core.py | 核心库 | RDM-PCA维度估计、配对t检验等核心算法 |
| dimension_story_utils.py | 辅助库 | 稳健性、searchlight、种子连接性等高级功能 |
| cli.py | 入口 | 命令行接口，统一调度所有分析命令 |
| config.py | 配置 | 路径、GLM参数、RD参数等配置数据类 |
| utils.py | 工具 | Fisher z、GPS、PCA方差、配对t检验等工具函数 |

---

## 四、论文分析与代码实现对应表

以下表格逐项对比论文中的每个分析内容与代码实现的匹配情况。

**状态标注**：✅ 已实现、⚠️ 部分实现、❌ 缺失、ℹ️ 未明确

### 4.1 Exp. 1 核心分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| B1 | 左右海马 PCA 维度 (HSC vs LSC) | rd.py (run_group_rd / run_group_equalized_rd) | ✅ 已实现 | 支持多方差阈值(70%-90%)、bootstrap均衡化 |
| B2 | 左右海马 GPS (HSC vs LSC) | roi.py (run_group_gps) | ✅ 已实现 | Fisher z + 平均行相似性 |
| B3 | 海马-IFG 表征连接 (HSC vs LSC) | roi.py (run_group_roi_dsm_correlation) | ✅ 已实现 | RDM相关 + Fisher z |
| B4 | 海马维度与海马-IFG连接的负相关 | — | ❌ 缺失 | 代码中未实现此被试内相关分析 |

### 4.2 稳健性控制分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| F1 | 单变量激活控制 (ANCOVA) | — | ❌ 缺失 | 未实现ANCOVA控制分析 |
| F2 | 模仿控制 | — | ❌ 缺失 | 未实现排除模仿项目后的重新分析 |
| F3 | 项目特定创意潜力控制 | — | ❌ 缺失 | 未实现排除一致分类项目后的重新分析 |
| F4 | 记忆控制 (仅分析成功回忆项目) | run_stage09_remembered_forgotten.py | ⚠️ 部分实现 | 已有remembered/forgotten拆分，但缺少DMN激活控制和bootstrap均衡化 |
| F5 | 固定标准分类 | — | ❌ 缺失 | 未实现使用组平均中位数分类 |

### 4.3 ROI 稳健性检验（Stage 08）

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| R1 | Pattern PCA（不经RDM） | run_stage08_roi_robustness.py (hl_patternbutnotrdm) | ✅ 已实现 | 直接对pattern做PCA |
| R2 | 右海马 RDM-PCA 基线 | run_stage08_roi_robustness.py (hl_rdm_baseline) | ✅ 已实现 | 右海马作为基线对照 |
| R3 | 固定trial数重抽样 | run_stage08_roi_robustness.py (sametrial_rdm_fixed20) | ✅ 已实现 | 1000次重抽样 |
| R4 | 去均值 + 最小trial数 | run_stage08_roi_robustness.py (demean_pattern_min) | ✅ 已实现 | 多方差阈值(70%-90%) |
| R5 | 去均值 + 固定25 trial | run_stage08_roi_robustness.py (demean_pattern_fixed25) | ✅ 已实现 | 更严格的控制 |
| R6 | 体素mask过滤 | run_stage08_roi_robustness.py (voxelmask_pattern_fixed20) | ✅ 已实现 | 跨被试t检验去除激活差异体素 |
| R7 | PC窗口(3-5)解释方差和 | run_stage08_roi_robustness.py (pc_window_3_5_fixed20) | ✅ 已实现 | 排除前1-2个成分影响 |

### 4.4 记忆与行为分组分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| M1 | Remembered vs Forgotten 维度 | run_stage09_remembered_forgotten.py | ✅ 已实现 | 分别对两组做HSC vs LSC维度对比 |
| M2 | Memory-good vs Memory-poor 维度 | run_stage10_memory_group.py | ✅ 已实现 | 按记忆准确率分组后分析 |
| M3 | 行为评分pooling维度 | run_stage11_behavior_pooling.py | ✅ 已实现 | 跨被试汇总、高低分池、重抽样 |

### 4.5 全脑分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| G1 | 全脑 RD searchlight | run_stage12_dimension_searchlight.py | ✅ 已实现 | KD-Tree邻域，100体素，组水平t检验 |
| G2 | searchlight表征连接图回归到海马维度 | group.py (run_group_regression) | ⚠️ 部分实现 | 有体素回归，但缺少SVC小体积校正 |
| G3 | 海马 PCA 连接性 searchlight | run_stage13_seed_connectivity.py | ✅ 已实现 | PCA loading余弦相似度 |

### 4.6 Exp. 2 分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| C1 | Exp.2 海马维度复制 | rd.py | ℹ️ 未明确 | 代码支持，但未确认是否已用Exp.2数据跑过 |
| C2 | 合并Exp.1+2 维度与创造性相关 | group.py (run_group_regression) | ⚠️ 部分实现 | 有回归框架，但缺少Pearson被试间相关分析 |

### 4.7 跨阶段表征迁移分析（Exp. 2 核心）

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| D1 | 统一PCA空间跨阶段相似性 | — | ❌ 缺失 | 未实现学习-创造跨阶段PCA迁移 |
| D2 | 类别特定PCA空间跨阶段相似性 | — | ❌ 缺失 | 未实现分条件PCA投影迁移 |
| E1 | 前海马全维空间跨阶段相似性 | — | ❌ 缺失 | 未实现未压缩RDM跨阶段相关 |
| E2 | vlPFC全维/压缩空间跨阶段相似性 | — | ❌ 缺失 | 未实现 |
| E3 | 海马-vlPFC 压缩/未压缩连接 | — | ❌ 缺失 | 未实现学习和创造阶段分别的连接分析 |
| E4 | vlPFC SVM跨阶段解码 | — | ❌ 缺失 | 未实现SVM训练-测试、排列检验、海马回归剥离 |

### 4.8 补充分析

| 编号 | 论文分析内容 | 对应代码 | 状态 | 备注 |
|------|-------------|----------|------|------|
| H1 | Exp.S2 SME分析 | — | ❌ 缺失 | 未实现记忆成功神经关联分析 |
| H2 | Exp.S3-S7 行为补充实验 | — | ❌ 缺失 | 定向遗忘、R/K、孵化效应等均未实现 |
| P1 | PPI分析 (海马种子) | — | ❌ 缺失 | 未实现心理生理交互分析 |
| A1-A3 | 行为结果统计 (Fig.2B-D) | — | ❌ 缺失 | 未实现行为数据统计和可视化 |
| A4 | 模仿-创造权衡分析 | — | ❌ 缺失 | 未实现模仿相关回归分析 |

---

## 五、对应关系总结

| 状态 | 数量 | 涉及内容 |
|------|------|----------|
| ✅ 已实现 | 14 | Exp.1核心ROI分析、稳健性检验、记忆分组、行为pooling、全脑searchlight、种子连接性 |
| ⚠️ 部分实现 | 3 | 记忆控制(DMN激活、bootstrap)、脑-行为回归(SVC)、Exp.2复制确认 |
| ❌ 缺失 | 16 | 跨阶段迁移(4项)、SVM解码(1项)、稳健性控制(4项)、行为统计(4项)、SME(1项)、PPI(1项)、被试内相关(1项) |
| ℹ️ 未明确 | 1 | Exp.2数据是否已跑过 |

---

## 六、缺失分析详细说明

### 6.1 跨阶段表征迁移分析（Exp. 2 核心）

这是论文中最重要的缺失部分，对应 Figure 5 和 Figure 6 的大部分内容。包括：

- **统一PCA空间跨阶段相似性（Fig. 5A-B）**：将HSC和LSC在学习和创造阶段的海马激活合并投影到共享PCA空间，计算跨阶段PC分数的Spearman相关
- **类别特定PCA空间跨阶段相似性（Fig. 5C-D）**：分别对HSC和LSC条件进行PCA，将创造阶段数据投影到学习阶段的主轴上
- **前海马全维空间跨阶段相似性（Fig. 6B）**：在未压缩空间中计算跨阶段RDM的Pearson相关
- **vlPFC全维/压缩空间跨阶段相似性（Fig. 6C）**
- **海马-vlPFC压缩/未压缩连接（Fig. 6D-E）**：在学习和创造阶段分别分析

> **影响评估：这些分析是论文的核心贡献之一，证明低维海马表征能够从学习迁移到创造阶段。缺少这些分析会严重影响论文的可复现性。**

### 6.2 SVM跨阶段解码分析（Fig. 6F-G）

缺少以下内容：

- 全脑searchlight SVM解码：学习阶段训练区分HSC/LSC，创造阶段测试
- 10,000次排列检验（SnPM13），体素水平p<0.001，簇水平FDR校正q<0.05
- 海马低维成分回归剥离后重新解码
- 解码准确率与创造性表现的相关分析

> **影响评估：这是论文的关键证据，证明vlPFC的泛化解码能力依赖于海马低维表征。缺少此分析会削弱论文的核心论证力。**

### 6.3 稳健性控制分析

代码已有丰富的ROI稳健性检验（Stage 08，7组），但论文中报告的以下控制分析在代码中缺失：

- **单变量激活控制（ANCOVA）**：控制HSC/LSC单变量激活差异后，维度和连接效应是否仍然显著
- **模仿控制**：排除模仿/抄袭项目后重新分析
- **项目特定创意潜力控制**：排除一致被分类为HSC或LSC的项目
- **固定标准分类**：使用组平均中位数而非个体中位数进行HSC/LSC分类

> **影响评估：这些控制分析对论文的可信度至关重要，审稿人可能会要求提供。**

### 6.4 行为数据统计与可视化

论文中的行为结果（Figure 2B-D）和补充行为实验（Exp. S1-S7）均缺少对应的统计分析代码。包括：

- HSC vs LSC主观创意判断差异的配对t检验
- HSC vs LSC线索回忆率差异的配对t检验
- 多组创造性得分比较的Kruskal-Wallis检验
- 模仿-创造权衡回归分析
- Exp. S3-S7的各种行为统计

> **影响评估：行为结果是论文的基础，建议补充实现以确保完整可复现性。**

### 6.5 PPI分析

论文提到使用心理生理交互（PPI）分析以左侧海马为种子区域，但代码库中未实现PPI。代码中的种子连接性分析（Stage 13）基于PCA loading相似度，而非PPI。两者是不同的分析方法。

### 6.6 被试内相关分析

论文 Figure 3G 报告了海马维度与海马-IFG连接的负相关（r=-0.49, p=0.01）。这是一个被试内的相关分析，但代码中未实现。当前代码仅做了组水平的配对t检验，没有将维度差异分数与连接差异分数做被试内相关。

### 6.7 SME分析（Exp. S2）

论文补充分析中的Exp. S2（记忆成功的神经关联）是证明SME和SCE机制分离的关键证据。该分析包括海马单变量激活、GPS、维度和表征连接四个指标，均未在代码中实现。

---

## 七、改进建议

### 7.1 高优先级（影响论文核心结论）

- 补充跨阶段表征迁移分析模块（对应Fig. 5、Fig. 6B-E），包括统一PCA空间、类别特定PCA空间、全维RDM迁移、压缩/未压缩连接分析
- 补充SVM跨阶段解码分析（对应Fig. 6F-G），包括searchlight SVM、排列检验、海马回归剥离
- 补充稳健性控制分析（ANCOVA、模仿控制、固定标准分类等）

### 7.2 中优先级（影响可复现性）

- 补充行为数据统计分析模块（Fig. 2B-D、Exp. S1-S7）
- 补充被试内维度-连接相关分析（Fig. 3G）
- 确认Exp.2数据是否已用当前代码跑过，并补充合并分析（Fig. 4C）
- 补充SME分析（Exp. S2）

### 7.3 低优先级（工程优化）

- 考虑实现PPI分析模块（如果审稿人要求）
- 统一配置系统，去除硬编码路径（参见docs/92建议）
- 为每个分析模块添加单元测试，确保结果可复现

---

## 八、现有代码的优势

尽管存在上述缺失，现有代码库具有以下显著优势：

- **完整的Exp.1核心分析链路**：从事件重标记到组水平统计，覆盖14个阶段
- **丰富的稳健性检验套件**：7组不同策略的ROI稳健性分析，超越了论文报告的部分内容
- **良好的工程架构**：CLI统一入口、配置与逻辑分离、核心库与入口脚本分离
- **全脑分析能力**：searchlight维度图和种子连接性分析均已实现
- **多种记忆分组策略**：remembered/forgotten、memory-good/poor、行为pooling
