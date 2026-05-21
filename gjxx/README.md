# GJXX 认知神经科学分析流水线

## 一、研究背景与核心问题

本项目是一个 **fMRI 表征分析** 研究，核心研究问题是：

> **高语义一致性（HSC）和低语义一致性（LSC）条件下，大脑对记忆项目的表征维度是否存在差异？这种差异在哪些脑区最显著？又与个体的记忆行为表现如何关联？**

### 实验范式简介

- 被试（23 人，编号 sub-02 ~ sub-30）在 fMRI 扫描中完成 4 个 run 的记忆编码任务。
- 每个 trial 呈现一个记忆项目（example）。
- 项目按其所属类别的"语义一致性"被分为两组：
  - **HSC（High Semantic Consistency）**：类别内项目语义高度一致的条件
  - **LSC（Low Semantic Consistency）**：类别内项目语义较分散的条件
- 编码后进行记忆测验，记录每个项目是 **remembered**（记住）还是 **forgotten**（遗忘）。

### 核心假设

**HSC 条件下的神经表征维度低于 LSC 条件。** 直觉上，如果一个类别的成员在语义空间中非常一致，那么大脑只需较少的维度就能区分它们；而语义分散的类别需要更多维度来编码差异。这一假设受 Mack et al. (2020) 的启发，将"表征维度"（Representational Dimensionality, RD）作为衡量表征结构复杂度的指标。

---

## 二、数据分析的完整故事

整个分析流水线可以看作一个"从粗到细"的递进故事，分为三大章：

### 第一章：数据准备与基础统计（Stage 01–07）

这一阶段的目标是：从原始 fMRI 数据出发，为每个 trial 生成干净的神经活动 pattern，并做最基础的 HSC vs LSC 比较。

#### Stage 01：事件文件清洗（relabel-events）

首先需要清洗实验事件。原始设计中，有些项目在所有被试中正确率极高或极低（太容易/太难），这类项目对表征分析不具信息量。本阶段统计全被试的行为表现，将过于极端的项目标记为 `tl`（too-low）/ `th`（too-high），不参与后续建模。

#### Stage 02：Item-level GLM（item-glm）

对每个被试、每个 trial 单独拟合 GLM（即 LS-S 方法），得到每个 trial 的 beta/t-map。这是后续所有表征分析的基础——只有 trial-level 的 pattern 才能计算 trial 间的相似性矩阵。

支持多种"故事"（story），如 `no_too_easy_or_hard`（排除极端项目后的主分析）、`gps`（GPS 分析用）、`rd_all_examples`（全 example 回归用）等。

#### Stage 03：Condition-level GLM（activation-glm）

与 Stage 02 互补。这里不是每个 trial 单独建模，而是按条件（如 HSC-remembered / HSC-forgotten / LSC-remembered / LSC-forgotten）拟合激活 GLM，用于传统的激活幅度比较分析。

#### Stage 04：Pattern 堆叠（stack-patterns）

将 Stage 02 产出的单 trial t-map 按条件堆叠为 4D NIfTI 文件（如 `glm_T_stats_HSC.nii`、`glm_T_stats_LSC.nii`），便于后续批量读取。同时也生成 GPS 分析所需的合并文件 `glm_T_gps.nii`。

#### Stage 05：ROI GPS 分析（gps）

**GPS（Global Pattern Similarity）** 是一个衡量"组内条目表征是否趋同"的指标。在指定 ROI（如海马）内，计算 HSC 和 LSC 条件的 GPS 值并做组水平配对 t 检验。如果 HSC 的 GPS 更高，说明 HSC 类别的表征更加"聚拢"。

#### Stage 06：ROI-to-ROI DSM 相关（roi-dsm）

比较两个 ROI 之间的表征相似矩阵（DSM）相关性在 HSC 和 LSC 条件下是否不同。这可以回答"两个脑区之间的表征对齐程度，是否受语义一致性调节"。

#### Stage 07：Representational Dimensionality（rd）

本研究的核心指标。在 ROI 内：
1. 为 HSC 和 LSC 分别构建 trial-by-trial 相关距离 RDM
2. 对 RDM 做 PCA
3. 找到累积解释方差达到阈值（默认 80%）所需的最少成分数——即"表征维度"
4. 做 HSC vs LSC 的组水平配对 t 检验

如果 HSC 的 RD 显著低于 LSC，说明高语义一致性条件确实导致更低维、更紧凑的表征结构。

---

### 第二章：核心假设的深度验证（Stage 08–11）

基础分析找到了效应后，需要回答：这个效应稳健吗？不同的分析选择是否影响结论？记忆表现如何调节该效应？

#### Stage 08：ROI 稳健性套件（roi-robustness）

这是一个包含 7 组子分析的"压力测试"套件，用于验证 RD 效应在多种分析选择下是否一致：

| 子分析 | 做了什么 | 为什么做 |
|--------|----------|----------|
| `hl_patternbutnotrdm` | 直接对 pattern 做 PCA（不经过 RDM） | 对照：不用 RDM 中介时效应是否仍存在 |
| `hl_rdm_baseline` | 用右海马做标准 RDM-PCA | 验证效应不限于左海马 |
| `sametrial_rdm_fixed20` | 固定 20 个 trial 随机抽样 1000 次取均值 | 控制 HSC/LSC 的 trial 数不等 |
| `demean_pattern_min` | 先去均值中心化，再取最小 trial 数重抽样 | 排除条件间均值差异的影响 |
| `demean_pattern_fixed25` | 去均值 + 固定 25 trial 重抽样 | 同上，更严格的控制 |
| `voxelmask_pattern_fixed20` | 先去掉条件间激活显著不同的体素 | 排除"激活幅度差异"的混淆 |
| `pc_window_3_5_fixed20` | 只看第 3–5 主成分的解释方差 | 排除第 1–2 成分（可能反映噪声/均值）的影响 |

#### Stage 09：Remembered vs Forgotten（remembered-forgotten）

将编码项目按记忆测验结果拆分，分别计算 remembered 和 forgotten 试次的 HSC vs LSC 维度差异。这回答了：**维度效应是否只存在于被成功记住的项目？**

分析使用 `patterns_hlrf` 目录下的数据（已按 HSC-remembered、HSC-forgotten、LSC-remembered、LSC-forgotten 拆分）。

#### Stage 10：Memory-good vs Memory-poor（memory-group）

根据每个被试的整体记忆正确率，将被试分为 memory-good（正确率 > 20%）和 memory-poor 两组，分别检验 HSC vs LSC 维度效应。这回答了：**维度效应是否在记忆能力好的被试中更强？**

#### Stage 11：行为评分 Pooling（behavior-pooling）

一种跨被试的 pooling 分析。将全部被试的全部 trial 按行为评分（如 `jiaocha_score`）从高到低排序，取前 30% 和后 30% 作为"高分池"和"低分池"，分别重抽样计算 RDM 维度并做配对检验。这回答了：**项目级别的行为表现是否与表征维度相关？**

---

### 第三章：全脑空间与脑区间关系（Stage 12–14）

前两章聚焦于 ROI，第三章将视野扩展到全脑。

#### Stage 12：维度 Searchlight（dimension-searchlight）

在全脑每个体素邻域（默认 100 个最近邻体素）内计算 RDM-PCA 维度，生成 HSC 和 LSC 的全脑维度图，再做组水平配对 t 检验。这产生一张"哪些脑区的表征维度在条件间差异最大"的统计图。

#### Stage 13：种子区连接性 Searchlight（seed-connectivity）

以左海马为种子区（seed），计算种子区与全脑每个 searchlight 球之间的 PCA 结构相似度（PCA loadings 的 cosine similarity）。分别对 HSC 和 LSC 做此计算，再做配对比较。这回答了：**海马的表征结构在哪些脑区有"回声"？HSC 与 LSC 的连接性模式是否不同？**

#### Stage 14：组水平回归（group-regression）

将 Stage 12 或 Stage 13 产出的全脑统计图（如被试级维度图或连接性图）与行为变量做 voxelwise regression，找到"表征维度与行为表现相关的脑区"。

---

## 三、运行方式

唯一推荐入口：

```
python -m gjxx.cli --help
python -m gjxx.cli <command> --help
```

也可以直接运行模块：`python -m gjxx`（等价于 `python -m gjxx.cli`）。

Stage 08–13 也可以通过模块方式独立运行，例如 `python -m gjxx.run_stage08_roi_robustness --help`。

### 流水线阶段速查表

#### 数据生产（Stage 01–07）

| Stage | 命令 | 功能 |
|-------|------|------|
| 01 | `relabel-events` | 事件文件重标记：排除过易/过难项目 |
| 02 | `item-glm` | Item-level GLM：为每个 trial 拟合独立 GLM |
| 03 | `activation-glm` | Condition-level GLM：按条件拟合激活 GLM |
| 04 | `stack-patterns` | 堆叠 trial maps 为 4D pattern 图像 |
| 05 | `gps` | ROI GPS 分析：组水平 pattern similarity |
| 06 | `roi-dsm` | ROI-to-ROI DSM 相关分析 |
| 07 | `rd` | RD 分析（representational dimensionality） |

#### 进阶分析（Stage 08–14）

| Stage | 命令 | 功能 |
|-------|------|------|
| 08 | `roi-robustness` | ROI 维度性 + 稳健性检验套件 |
| 09 | `remembered-forgotten` | Remembered vs Forgotten 维度对比 |
| 10 | `memory-group` | Memory-good vs Memory-poor 维度对比 |
| 11 | `behavior-pooling` | 行为评分分组的跨被试维度分析 |
| 12 | `dimension-searchlight` | 全脑 RD 维度性 searchlight |
| 13 | `seed-connectivity` | 海马种子区 PCA-connectivity searchlight |
| 14 | `group-regression` | 组水平 voxelwise regression |

**补充命令（随时可用）：**

| 命令 | 功能 |
|------|------|
| `rd-equalized` | Equalized-trial RD 分析 |
| `rd-searchlight` | 单图 searchlight RD |
| `subject-scores` | 从事件汇总被试分数 |

### 阶段依赖关系

```
Stage 01 (relabel-events)
  └─ Stage 02 (item-glm) / Stage 03 (activation-glm)
       └─ Stage 04 (stack-patterns)
            ├─ Stage 05 (gps)
            ├─ Stage 06 (roi-dsm)
            ├─ Stage 07 (rd)
            ├─ Stage 08 (roi-robustness)
            ├─ Stage 09 (remembered-forgotten)
            ├─ Stage 10 (memory-group)
            ├─ Stage 11 (behavior-pooling)
            ├─ Stage 12 (dimension-searchlight)
            └─ Stage 13 (seed-connectivity)
                 └─ Stage 14 (group-regression)
```

---

## 四、配置与路径

配置优先级（从高到低）：

1. **命令行参数**（CLI）
2. **`--config-json` 配置文件**
3. **legacy preset 默认值**（见 `legacy_presets.py`，源自 MATLAB 工程的 Windows 盘符路径）

Stage 08–13 的默认路径收敛在 `legacy_presets.py` 的 `FLXX1_WINDOWS` 中。正式使用时请通过 CLI 参数或 config JSON 覆盖。

---

## 五、输出约定

所有任务至少输出：

- `summary.json`：任务摘要与统计结果
- `subject_outputs.csv`（若有逐被试输出）
- `selection_details.csv`（若有纳排逻辑）

---

## 六、核心模块

| 模块 | 职责 |
|------|------|
| `events.py` | 事件文件读写与重标记 |
| `first_level.py` | 一阶 GLM（item-level / activation） |
| `patterns.py` | Trial map 堆叠 |
| `roi.py` | GPS、ROI DSM 分析 |
| `rd.py` | RD、equalized RD、searchlight |
| `group.py` | 被试评分汇总、组回归 |
| `dimension_analysis_core.py` | 维度分析核心算法（PCA、RDM、t 检验等） |
| `dimension_story_utils.py` | 维度分析辅助函数（searchlight、connectivity 等） |
| `config.py` | 配置数据类（StudyPaths、GlmSettings 等） |
| `runtime_config.py` | 运行时配置解析（config JSON、路径优先级） |
| `legacy_presets.py` | 历史默认路径（MATLAB 工程兼容） |
| `utils.py` | 通用工具函数 |

---

## 七、专家评审：可补充与优化的方向

以下是从认知神经科学方法学角度对当前研究流水线的分析，供规划下一步研究参考。

### 7.1 统计与多重比较

| 当前状况 | 建议 |
|----------|------|
| 所有组水平检验使用未校正的配对 t 检验 | Stage 07/08 的 ROI 分析可补充 **FDR 或 Bonferroni 校正**（跨多个阈值、多个子分析） |
| Searchlight（Stage 12/13）输出未校正 p 图 | 建议增加 **cluster-level FWE 校正**（permutation-based）或 **TFCE**（Threshold-Free Cluster Enhancement） |
| 行为 pooling（Stage 11）用 bootstrap 配对 t 检验 | 考虑补充 **permutation test** 以获得精确的零分布 |

### 7.2 效应量与置信区间

| 当前状况 | 建议 |
|----------|------|
| 报告 t 值和 p 值，计算了 r² | 建议补充 **Cohen's d** 及其 **95% 置信区间**，这是期刊审稿人常要求的标准效应量指标 |
| Searchlight 只输出 t-map 和 p-map | 考虑额外输出 **Cohen's d map**，便于可视化效应量的空间分布 |

### 7.3 交叉验证与泛化性

| 当前状况 | 建议 |
|----------|------|
| Stage 08 用同一数据做分析和稳健性检验 | 考虑加入 **leave-one-subject-out 交叉验证**：用 N-1 个被试确定最优阈值/ROI，在第 N 个被试上验证 |
| 整体分析只有 FLXX1 一个数据集 | 若有条件，在独立数据集上做 **replication** 是最强的稳健性证据 |

### 7.4 表征几何的进一步刻画

| 当前状况 | 建议 |
|----------|------|
| 维度指标只用"PCA 累积解释 80% 方差的成分数" | 可补充 **参与比（Participation Ratio）** 作为连续性更好的维度指标（避免阈值敏感性） |
| 只比较维度数量 | 可增加 **RSA 模型比较**：构建 HSC/LSC 的"理想 RDM 模型"（如语义模型、知觉模型），用 RSA 检验哪种模型能更好解释观察到的 RDM |
| 只有 RDM-PCA 和 pattern-PCA 两种度量 | 考虑加入 **几何学指标**，如表征的有效维度、intrinsic dimensionality 估计（如 MLE 方法） |

### 7.5 脑区选择与全脑覆盖

| 当前状况 | 建议 |
|----------|------|
| ROI 分析集中在左/右海马 | 可补充 **内嗅皮层（entorhinal cortex）**、**海马旁回**、**前额叶**等与记忆编码和语义加工相关的区域 |
| Searchlight 分析没有解剖学分区的 ROI 汇总 | 建议增加 **atlas-based ROI 汇总**：用 AAL/Schaefer atlas 对 searchlight 结果做区域级平均，便于系统性报告 |

### 7.6 时间动态

| 当前状况 | 建议 |
|----------|------|
| 所有分析基于 trial-level GLM beta/t-map（时间维度被平均掉） | 如果数据允许，可考虑 **时间分辨的 RSA（time-resolved RSA）**，用滑动窗口或 FIR 模型捕捉表征维度的编码时程变化 |

### 7.7 记忆效应的细化

| 当前状况 | 建议 |
|----------|------|
| Stage 09 只区分 remembered / forgotten | 可以进一步按记忆信心（confidence）分级，检验 **高信心记忆 vs 低信心记忆** 的维度差异 |
| Stage 10 按正确率 > 20% 分组 | 阈值选择较低，考虑使用 **中位数分组** 或 **连续回归** 替代二分法 |
| 未区分编码成功（DM）效应 | 可补充 **subsequent memory 效应的维度分析**：编码时的维度差异能否预测该项目后续是否被记住？ |

### 7.8 工程与可复现性

| 当前状况 | 建议 |
|----------|------|
| 核心模块没有单元测试 | 建议对关键函数（如 `correlation_distance_rdm`、`pca_explained_variance`、`dimensionality_from_samples`）增加 **单元测试**，用已知矩阵验证输出 |
| 没有 `requirements.txt` 或 `pyproject.toml` | 建议增加依赖管理文件，固定 numpy/scipy/nibabel 版本 |
| 随机种子固定但不全局统一 | 各阶段的随机种子（13、23 等）建议统一通过配置管理 |

---

## 八、归档文档（历史参考）

- MATLAB（Dimension）流程梳理：[90_dimension_matlab_flow.md](docs/90_dimension_matlab_flow.md)
- MATLAB（GJXX_1_reanalysis）脚本盘点：[91_gjxx1_matlab_inventory.md](docs/91_gjxx1_matlab_inventory.md)
- 历史整理建议（供参考）：[92_pipeline_optimization_notes.md](docs/92_pipeline_optimization_notes.md)
