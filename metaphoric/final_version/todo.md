# 冲刺 NN / NC / NHB（最低 PNAS）总体方案与任务计划

> 本文件是 2026-04-27 重写版，本文件是**项目唯一的权威任务清单**，面向 Nature 子刊冲刺。
>
> 配套文档：
>
> - 核心结果口径：[`result.md`](./result.md)
> - 每步具体方法与结论（用于汇报）：[`result_walkthrough.md`](./result_walkthrough.md)
> - 怎么跑：[`README.md`](./README.md)
> - 每步为什么这么跑：[`readme_detail.md`](./readme_detail.md)

***

## 一、目标档次与投稿策略

分档（强绑定不同证据量）：

- **Nature Neuroscience / Neuron 冲刺档**：机制框架 + 网络方向性 + 个体差异跨样本/留出预测 + 学习动态 + MTL 子区证据，至少 1 个"新现象"命名权。
- **Nature Communications / Nature Human Behaviour 主目标档**：机制框架 + 个体差异预测（within 或 across）+ 学习动态 + 至少 1 条网络层证据。
- **PNAS 最低保底档**：多层 ROI × Model-RSA + baseline 控制 + 脑-行为 within-subject 耦合 + 学习动态 + 稳健性三件套。

投稿流程建议：**先内部写一版针对 NC/NHB 的稿件；如果 editor desk reject，退档到 Communications Biology / Cerebral Cortex / J Neurosci，不要先投保底档。**

### 强制边界（所有档次都必须遵守）

- **主文只允许 1 个 primary endpoint**：`similarity ~ condition × time + (1|subject) + (1|item)` 中的 `condition × time` 交互。其他全部标 Secondary / Exploratory。
- Secondary / Exploratory 必须在方法中声明层级与多重比较策略，避免"很多分析挑显著"。
- 所有结论必须能追溯到具体输出文件（表格 / 阈值图 / 统计图）。
- 任何"无差异"必须给 BF10 或 CI，不只写 "p > .05"。
- MVPA 放主文必须讲清防泄漏与置换策略；否则只能留 SI。
- 不要把所有分析都放主文：主文只服务 primary endpoint 的证据链。

### 分析层级（Confirmatory / Secondary / Exploratory）

- **Confirmatory（主文主结论）**：前后测 item-wise RSA + LMM（primary endpoint）+ Model-RSA / Δρ LMM（机制主线）+ baseline 控制
- **Secondary（主文或 SI）**：学习阶段 GLM 定位证据；脑-行为 within-subject 耦合；激活-表征解耦图；Searchlight RSA
- **Exploratory（SI）**：RD/GPS、MVPA、gPPI、有效连接、learning dynamics（若未作为 secondary）、中介、模型 RDM 扩展、CPM、LI、Career of Metaphor

交付物：`paper_outputs/analysis_plan.tsv` 必须完整列出以上三层及其多重比较策略（FWE / FDR / 置换）与阈值。

***

## 二、子刊级核心故事脉络（Story Arc）

### 2.1 一句话主张（升级版）

> 学习新颖隐喻（相较空间联结）诱发了一种特定方向的表征可塑性：大脑在预存语义几何之上"撕开"源-靶语义的共享成分，构建独立、可分离的关系表征（**representational differentiation / semantic unbinding**）；该过程沿 **左前颞语义枢纽 → 左 IFG 语义控制 → 左 AG/Precuneus 关系整合** 的网络动态展开，其 individual 强度在被试内项目层可预测后续记忆表现，并在学习阶段逐步形成。

### 2.2 五层叙事逻辑（What → Where → When → Why → So What）

| 层级              | 回答的问题                                            | 证据出处                                                                      | 当前状态                        |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------- | --------------------------- |
| **What**（现象）    | 隐喻学习后是否发生比空间学习更强的表征变化？方向是什么？                     | Step 5C item-wise RSA + LMM（四层 ROI）                                       | ✅ 已完成                       |
| **Where**（空间）   | 这种变化发生在哪些脑区？是否具选择性？                              | 学习阶段 GLM + Searchlight RSA                                                | ⚠️ GLM 已有，Searchlight **缺** |
| **When**（时间）    | 差异化在学习中如何形成？是否线性？                                | Learning dynamics（run-3/4 sliding window）                                 | ❌ 未做                        |
| **Why**（机制）     | differentiation 的维度是什么？pair 层级、关系层级还是 schema 层级？ | Model-RSA + relational / memory / reverse-pair 模型 + variance partitioning | ⚠️ 初步有，未扩展到关系层              |
| **So what**（意义） | differentiation 强度是否预测个体行为？通过什么路径？               | Within-subject item-level 耦合 + 中介 + （可选）CPM                               | ❌ 当前 between-subject 相关阴性   |

**子刊需要把所有五层都有强证据**，PNAS 至少四层，Communications Biology 三层即可。

### 2.3 Primary Endpoint（写进方法学的定义）

- 模型：`similarity ~ condition × time + (1|subject) + (1|item)`
- 主检验：`condition × time` 交互，考察 yy 的 pre→post 变化幅度是否显著不同于 kj / baseline
- 命名：采用 **representational differentiation** 而非 convergence / alignment
- 结论模板："在 ROI X 中，LMM 显示 condition×time 交互显著（b=..., SE=..., t=..., p=..., 95% CI \[...]），表明 yy 的表征相似性从 pre→post 的变化幅度显著大于 kj；在当前数据中该变化主要表现为 yy 的相似性下降（分化增强）。"
- 最低汇报要求：主文必须同时报告 yy / kj / baseline 三条件的 pre/post 方向与效应大小，用 baseline 封堵"纯重测/熟悉化"解释。

***

## 三、距离子刊的 5 个硬差距（诊断结论）

| # | 差距                                                              | 当前状态 | 优先级   |
| - | --------------------------------------------------------------- | ---- | ----- |
| 0a | **M7_memory 二值化技术债**（`recall_score` 是 0/1，RDM 退化为二值，"不显著"被误读风险） | ⚠️   | P0 致命（解读风险） |
| 0b | **Δρ LMM ROI 集合异质性技术债**（`main_functional` 混合 yy-强 / kj-强 ROI，LMM 平均效应不对应单一机制） | ⚠️   | P0 致命（解读风险） |
| 1 | **个体差异可预测性**（脑-行为阴性 r=-0.163, p=.417）                           | ❌    | P0 致命 |
| 2 | **机制维度浅**（仅 pair/语义/记忆三类模型，缺关系、缺 schema、缺 hippocampal subfield） | ⚠️   | P0 重大 |
| 3 | **缺学习动态**（run-3/4 sliding window 未做）                            | ❌    | P0 重大 |
| 4 | **方向性与网络层缺失**（gPPI / representational connectivity 未出结果）        | ❌    | P1 重要 |
| 5 | **N=28 偏小，无复制 / 留出 / BF / LOSO**                                | ⚠️   | P1 重要 |

***

## 四、统一输出目录约定（强制）

全部子刊冲刺期产出写到：

```
${PYTHON_METAPHOR_ROOT}/paper_outputs/
├── figures_main/           # 主文 4–7 张
├── figures_si/             # SI 图
├── tables_main/            # 主文表（含 figure_stats_map.tsv）
├── tables_si/              # SI 表
├── qc/                     # QC 记录（顶刊必审）
│   ├── lss_qc_summary.tsv
│   ├── pattern_qc_summary.tsv
│   ├── motion_scrubbing_summary.tsv
│   └── fake_handling_note.md
├── analysis_plan.tsv       # 分析层级表
└── figure_stats_map.tsv    # 每张图 ↔ 主统计检验 ↔ 校正 ↔ 输出文件
```

三类交付物目标：

- **核心结果（主文）**：4–7 张主图 + 每张图背后唯一主统计检验 + 一句结论
- **补充材料（SI）**：所有稳健性检验、控制验证、探索性分析的完整表格与图
- **可复现包**：代码 + 关键输出表 + ROI / mask / 模板（不一定公开原始数据）

***

## 五、QC 封堵（顶刊高频追问点，全部必做）

### 5.1 LSS 完整性与对齐 QC

- NaN / 空脑 / 极端值 / 缺失 trial 报告
- `pattern_root/sub-xx/pre_yy.nii.gz` 等文件齐全性
- `*_metadata.tsv` 的 trial 数与预期一致（至少能解释任何偏差）
- [ ] 输出 `paper_outputs/qc/lss_qc_summary.tsv`
- [ ] 输出 `paper_outputs/qc/pattern_qc_summary.tsv`
- [ ] 方法学段落："如何避免 trial 错配导致假阳性"

### 5.2 假词 / 过滤策略

- [ ] `paper_outputs/qc/fake_handling_note.md`（≤10 行）：说明 fake/nonword 在 stack\_patterns 时排除，以及为什么不能在 LSS 建模前排除

### 5.3 运动 / 去噪策略可审计

- [ ] 报告 scrubbing 规则（FD/DVARS 阈值、是否扩窗）
- [ ] 每被试被 scrub 的时间点比例
- [ ] 输出 `paper_outputs/qc/motion_scrubbing_summary.tsv`

***

## 六、分层 TODO 列表

### P0 — 子刊硬门槛（2–3 周内完成）

#### P0.0 M7_memory 二值化结构性问题修复（新发现的技术债）

**背景**：2026-04-27 审稿式自查发现，**M7_memory RDM 当前是二值矩阵，不是真正的连续回忆强度**——
- [`build_memory_strength_table.py`](./rsa_analysis/build_memory_strength_table.py) 从 run-7 events 的 `memory`（0/1）列派生 `recall_score`
- run-7 是**单次最终记忆测验**，每个 `real_word` 通常只测试 1 次
- 虽然脚本 L91-94 做了 `groupby('real_word').mean()`，但单 trial 均值仍是 0 或 1 → **`recall_score` 实际退化为二值**
- 所以 `RDM_M7[i,j] = |recall_i - recall_j|` 只产生 `{0, 1}` 两种距离，**M7 实际是二值 RDM，信息容量与 M1_condition 接近**

**后果（必须在主文和 SI 中诚实声明）**：
1. 当前"M7 在 7/7 ROI 不显著"**不能解读为 "回忆强度与神经几何无关"** —— 可能只是度量粒度不足导致的功效不足
2. M7 与 M1_condition 都是二值，**偏相关时两者相互残差化可能掩盖真实信号**
3. 在顶刊审稿人眼里，M7 必须**降级为 Secondary/Exploratory**，不能与 M2/M3/M8 并列作为主机制模型

**修复任务清单**：

- [ ] **诚实声明（最低必做）**：在论文方法学段落 + SI 明确写出 "**M7 在当前实验设计下为二值近似；不显著结果不等价于回忆强度与神经几何无关**"
- [ ] **主文降级**：把 M7 从主机制模型（与 M2/M3/M8 并列）降级为 Secondary 模型；主文 Fig 3 / Δρ LMM 的主机制只保留 M1/M2/M3/M8
- [ ] **方案 A（最优，推荐实现）：连续 confidence 指标**
  - 构造 `confidence_score = memory + w × (1 - normalized_log_rt)`，`w ∈ [0.3, 0.5]`（先做敏感性分析）
  - 在 `build_memory_strength_table.py` 基础上新增 `--score-mode confidence` 模式
  - 重跑 M7 → 命名为 **M7_continuous_confidence**
  - 与原 M7_binary 同时保留，**对比 Δρ 的差异**作为 SI 证据
- [ ] **方案 B（备选）：加入学习阶段行为**
  - 把 run-3/4 学习阶段的 item-level 正确率（多次重复，天然连续）纳入 recall_score
  - 命名为 **M7_learning_weighted**，作为 confidence 的补充来源
- [ ] **方案 C（可选，SI 级）：Bayesian IRT**
  - 用 1PL/2PL IRT 模型从 run-7 的 0/1 响应反推 item 的 latent difficulty
  - 作为第三种 M7 变体验证
- [ ] **P0.1 绕开 M7 的桥梁路径**：**trial-level LMM 直接把 `memory(0/1)` 作因变量、`Δ similarity` 作预测变量**——这是"脑-行为耦合"最强证据，**不受 M7 结构限制**（见下方 P0.1 更新）

**交付物**：
- [ ] 更新 [`build_memory_strength_table.py`](./rsa_analysis/build_memory_strength_table.py) 支持 `--score-mode {binary, confidence, learning_weighted, irt}` 四种模式
- [ ] 输出对比表 `paper_outputs/tables_si/table_m7_variants_comparison.tsv`（四种 M7 下的 Δρ 一致性）
- [ ] 方法学段落修订：M7 降级说明 + 二值化限制声明
- [ ] 若方案 A/B 的 M7 变体显著：作为 Fig 3 SI；若仍不显著：作为"当前未检出 memory-geometry 耦合"的 SI 证据

**优先级理由**：虽然不是"新增分析"，但这是**已做分析的一处解读风险**；顶刊审稿人会立刻质疑，**必须在投稿前修复或降级**。

#### P0.1 Within-subject item-level 脑-行为耦合（救场关键，兼顾 M7 问题绕行）

目标：
1. 把当前 between-subject 阴性相关替换为更敏感的 within-subject 项目级耦合，作为"So what"的核心
2. **同时绕开 P0.0 M7 二值化问题**——直接在 trial-level LMM 里用原始 `memory(0/1)` 作因变量，不经过 RDM 这一中间环节

实现思路：

1. 对每个 trial，构造 `item_similarity_post - item_similarity_pre` 作为该 item 的神经可塑性指标。
2. 对每个 trial，取 run-7 该 item 的 `memory`（0/1）或 per-item `log_rt_correct`。
3. 拟合 trial-level LMM：`memory ~ delta_similarity × condition + (1 + delta_similarity | subject) + (1 | item)`，对 `delta_similarity:condition[yy]` 做推断。
   - **注意**：这里 `memory` 是二值因变量，所以用 **binomial family + logit link（GLMM）**；`log_rt_correct` 版本用 Gaussian LMM
4. 对每位被试单独跑 within-subject logistic：`memory ~ delta_similarity`，再对 27 个被试的斜率做 one-sample t。两种方法互相复核。
5. **为什么这个路径比 M7 更强**：M7 把 memory 压成 RDM 再和 neural RDM 求 Spearman，经过两次非线性变换（取绝对差 + 秩相关）；trial-level LMM 直接用原始变量，**功效高、解释直接、不受 RDM 二值化影响**。

- [ ] 在 `brain_behavior/` 下新建 `itemlevel_brain_behavior.py`
- [ ] 跑 4 层 ROI × yy/kj 两个条件的 within-subject 耦合
- [ ] 输出 `paper_outputs/tables_main/table_itemlevel_coupling.tsv`
- [ ] 输出 `paper_outputs/figures_main/fig_itemlevel_coupling.png`（小提琴 + 每被试斜率点 + 组均值）
- [ ] 一句结论：在左 IFG / 左 temporal pole 等 ROI 中，更强的 pre→post item-level differentiation 预测该 trial 更好的记忆表现（yy 特异）

#### P0.2 Searchlight RSA 全脑（Where 的完整性）

目标：证明 differentiation 不是 ROI cherry-pick 的区域假象。

实现思路：

1. 复用已有 [`rd_searchlight.py`](./representation_analysis/rd_searchlight.py)，在全脑做 `pre→post` Δ similarity 对比（yy / kj / baseline 三条件）。
2. 统计：subject-level 先做 trial-level permutation，group-level 用 cluster-based permutation（5000 次）做 FWE。
3. 输出 `yy_deltasim`、`kj_deltasim`、`baseline_deltasim` 三张 thresholded map 和一张 `(yy - baseline) > (kj - baseline)` 对比图。
4. 提取显著 cluster 的 peak 坐标表，与 Huang 2023 ALE 元分析的隐喻网络坐标做距离匹配。

- [ ] 运行 `rd_searchlight.py` 三条件 × pre/post
- [ ] 跑 group-level cluster permutation（置换 5000 次）
- [ ] 输出 `paper_outputs/figures_main/fig_searchlight_delta.png`
- [ ] 输出 `paper_outputs/tables_main/table_searchlight_peaks.tsv`
- [ ] 与 Huang 2023 ALE、Liu 2024 DT 脑模式做 spatial correlation（置换 10000 次）

#### P0.3 Learning dynamics（When 闭环）

目标：证明 differentiation 是在学习中逐步形成的，不是测验时读出。

实现思路：

1. 运行已有的 [`learning_dynamics.py`](./temporal_dynamics/learning_dynamics.py)，参数 `--window-size 20 --compute-mvpa`，在 run-3/4 上做 sliding-window 内 pattern similarity 轨迹。
2. 对每位被试 × ROI 拟合 `similarity ~ window_index` 的斜率，条件为 yy 和 kj。
3. 组水平比较 `slope_yy - slope_kj`（paired t + BF10）。
4. 关键二次分析：individual `(slope_yy - slope_kj)` 是否预测 post-pre Δ similarity，以及是否预测 run-7 记忆。

- [ ] 跑 `learning_dynamics.py`，覆盖主功能 ROI + literature ROI
- [ ] 输出 `paper_outputs/tables_main/table_learning_slope.tsv`
- [ ] 输出 `paper_outputs/figures_main/fig_learning_dynamics.png`（多 ROI 轨迹 + 组斜率条）
- [ ] slope × behavior 相关（Spearman + bootstrap 95% CI）

#### P0.4 机制维度扩展：Relational Model（M9）+ Variance Partitioning

目标：回答 "differentiation 是什么维度的"——是 item-level 还是 relation-level。

实现思路：

1. 新增 `M9_relational`：对每对 yy pair 标注关系类型（如"空间→时间"、"物理→情感"等映射类别），构造 relational category RDM。若暂无人工标注，用 BERT 的 relational embedding（concat source-target 后 PCA 降维）近似。
2. 新增 `M10_source_domain` / `M11_target_domain`：分别对源域、目标域词的 BERT embedding 构造 RDM。
3. 在 `model_rdm_comparison.py` 基础上增加 variance partitioning：对 neural RDM 同时回归 M3 / M9 / M10 / M11，拆出每个成分的独立贡献率。
4. 关键假设检验：**post 阶段 M9 的独立贡献率应大于 pre 阶段**。

- [ ] 在 `rsa_analysis/` 下新增 `build_relational_rdm.py`
- [ ] 扩展 `model_rdm_comparison.py` 支持 variance partitioning
- [ ] 输出 `paper_outputs/tables_main/table_variance_partitioning.tsv`
- [ ] 输出 `paper_outputs/figures_main/fig_variance_partitioning.png`
- [ ] 一句结论：post 阶段 `M9_relational` 独立解释率显著高于 pre，隐喻学习建立了超越单词语义的关系层表征

#### P0.5 稳健性三件套（顶刊 desk-reject 防护）

- [ ] 对 primary endpoint（condition × time 交互）加 **Bayes Factor**（BF10），调用 [`utils/bayesian_stats.py`](./utils/bayesian_stats.py)
- [ ] **Leave-One-Subject-Out**：每次去掉 1 个被试，验证 yy × pre 交互仍显著
- [ ] **Split-half replication**：随机划分 14 + 14，验证方向一致性（1000 次重随机）
- [ ] **Bootstrap CI**：对 subject-level Δ similarity 做 bootstrap 95% CI（[`utils/bootstrap_ci.py`](./utils/bootstrap_ci.py)）
- [ ] 输出 `paper_outputs/tables_si/table_bayes_factors.tsv`、`table_loso.tsv`、`table_splithalf.tsv`、`table_bootstrap_ci.tsv`

#### P0.6 刺激语言学控制表（方法学必备）

目标：堵死"yy 和 kj 刺激不均衡导致差异"的质疑。

- [ ] 提取 yy 组和 kj 组刺激的：词频（BCC 语料库）、笔画数、具象性评分、熟悉度评分、imageability 评分
- [ ] 两独立样本 t 检验 + BF10
- [ ] 输出 `paper_outputs/tables_si/table_stimulus_control.tsv`
- [ ] 若不均衡，把相关变量加入 LMM 作为 covariate 重跑主分析

***

### P1 — 主目标（NC/NHB）必须档（3–5 周内完成）

#### P1.1 Hippocampal subfield + MTL repulsion/integration 对比

目标：与 Favila 2016 (Nature Communications) / Chanales 2017 / Schlichting 2015 (Nature) / Sun 2026 (Commun Biol) 对话。

实现思路：

1. 用 ASHS atlas 或 hippocampal subfield atlas，分出 CA1 / CA3-DG / subiculum / anterior-posterior hippocampus。
2. 在每个 subfield 内做 yy / kj 的 pre→post Δ similarity。
3. 关键假设：**CA3-DG 应呈现 repulsion（differentiation），CA1 应呈现更小的重组或 integration**。
4. 补 hippocampus × cortex 的 seed-based representational connectivity。

- [ ] 生成 hippocampal subfield mask（ASHS 或手工）
- [ ] 在 `rsa_analysis/` 下新增 `hippocampal_subfield_rsa.py`
- [ ] 输出 `paper_outputs/tables_main/table_hippocampal_subfield.tsv`
- [ ] 输出 `paper_outputs/figures_main/fig_hippocampal_subfield.png`

#### P1.2 gPPI + Representational Connectivity（Why 的网络层）

目标：把 "differentiation 怎么展开" 从"发生在哪里"升级到"以什么方向流动"。

实现思路：

1. **gPPI**：以学习期 `yy > kj` 的左前颞 peak 为 seed，检验其对左 IFG / 左 AG / 双侧 precuneus / hippocampus 的调制连接。
2. **Representational Connectivity（Basti 2020）**：对每对 ROI 计算 neural RDM 之间的 Spearman 相关，比较 pre 和 post。
3. **Effective Connectivity**（简化 Granger 或 beta-series 方向回归）：给出"左前颞 → 左 IFG → 左 AG"方向假设的证据。声明：这是近似方法，不等价 DCM。

- [ ] 跑 [`gPPI_analysis.py`](./connectivity_analysis/gPPI_analysis.py)，生成 seed × target 调制矩阵
- [ ] 实现 `representational_connectivity.py`（新脚本）
- [ ] 跑 [`effective_connectivity.py`](./connectivity_analysis/effective_connectivity.py) `--time learn --condition yy`
- [ ] 输出 `paper_outputs/figures_main/fig_network_directionality.png`（箭头图或 circular plot）
- [ ] 输出 `paper_outputs/tables_main/table_network_directionality.tsv`

#### P1.3 中介分析：activation → Δ similarity → memory

目标：把"并列证据"组织为"路径假设"。

实现思路：

1. 自变量 X：学习期 `yy > kj` 激活强度（subject-level ROI contrast）
2. 中介 M：`yy` 的 pre→post Δ similarity（subject-level）
3. 因变量 Y：run-7 `memory_yy - memory_kj`
4. [`mediation_analysis.py`](./brain_behavior/mediation_analysis.py) 的 bootstrap（5000 次）估计间接效应 95% CI

- [ ] 对 3–5 个关键 ROI 各做一遍
- [ ] FDR 校正跨 ROI
- [ ] 输出 `paper_outputs/tables_main/table_mediation.tsv`
- [ ] 输出 `paper_outputs/figures_main/fig_mediation_path.png`
- [ ] 声明：样本量限制，作为路径假设性证据

#### P1.4 Noise Ceiling（顶刊 RSA 标配）

- [ ] 对每个 ROI 的 Step 5C similarity 结果计算 lower / upper noise ceiling（leave-one-subject-out）
- [ ] 在 Figure 2 中叠加 noise ceiling 灰带
- [ ] 输出 `paper_outputs/tables_si/table_noise_ceiling.tsv`

#### P1.5 Cross-validated RSA（防 circular analysis）

- [ ] 改写 RSA 流程为 leave-one-run-out cross-validated pattern correlation
- [ ] 对比 CV vs non-CV 结果方向一致性
- [ ] 输出 `paper_outputs/tables_si/table_cv_rsa.tsv`

***

### P2 — NN/Neuron 冲刺档（可选，6–8 周内）

#### P2.1 CPM 跨被试预测

目标：类比 Ovando-Tellez 2022 (Commun Biol)、Liu 2024 (Commun Biol)，把 individual differentiation 作为可外推 predictor。

实现思路：

1. 每位被试学习阶段全脑功能连接矩阵（Schaefer 200 ROI）
2. Leave-one-out CPM 预测 `memory_yy - memory_kj` 或 `Δ similarity_yy - Δ similarity_kj`
3. 识别显著正/负 edge，与 DMN / FPCN / Salience 网络做 overlap 分析

- [ ] 实现 `cpm_prediction.py`
- [ ] 5000 次 permutation 显著性
- [ ] 输出 `paper_outputs/tables_si/table_cpm.tsv`
- [ ] 输出 `paper_outputs/figures_si/fig_cpm_edges.png`

#### P2.2 Parametric Modulation：新颖性 × 熟悉度梯度（Career of Metaphor）

目标：对接 Cardillo 2012 / Lai 2015 的 Career of Metaphor 理论。

- [ ] 对每个 yy trial 计算 exposure\_count（学习阶段第 k 次暴露）
- [ ] 用 [`parametric_modulation.py`](./glm_analysis/parametric_modulation.py) 重跑 GLM
- [ ] 预测：右半球激活随熟悉度衰减，左半球增强；pair similarity 轨迹呈非线性
- [ ] 输出 `paper_outputs/figures_si/fig_career_of_metaphor.png`

#### P2.3 Lateralization Index（LI）

目标：回应 Mashal 2007 / Cardillo 2018 的 LH vs RH 争议。

- [ ] 运行 [`lateralization_analysis.py`](./lateralization/lateralization_analysis.py)
- [ ] 对比 yy / kj 在 IFG / AG / pMTG / temporal pole 的 LI
- [ ] 输出 `paper_outputs/tables_si/table_lateralization.tsv`

#### P2.4 RD / GPS 几何层互补证据

- [ ] 运行 [`rd_analysis.py`](./representation_analysis/rd_analysis.py) 和 [`gps_analysis.py`](./representation_analysis/gps_analysis.py)
- [ ] 验证 RD 下降（分化增强）+ GPS 改变是否同向支持 RSA
- [ ] 输出 `paper_outputs/figures_si/fig_rd_gps_complement.png`
- [ ] 敏感性：RD 阈值 70–90%

***

### P3 — SI 整理与方法学防线（投稿前 1 周）

- [ ] 整理 `paper_outputs/qc/` 下全部 QC 文件（见第五节）
- [ ] 完成 `analysis_plan.tsv`（Confirmatory / Secondary / Exploratory + 多重比较策略）
- [ ] 完成 `figure_stats_map.tsv`（每张主图 ↔ 唯一主统计检验 ↔ 校正 ↔ 输出路径）
- [ ] 代码 + 衍生指标 + ROI mask + 刺激模板打包为可复现 release

***

## 七、主文图谱升级预案（冲 NC/NHB 口径）

| 图号        | 标题                                             | 承载证据                                                  | 状态             |
| --------- | ---------------------------------------------- | ----------------------------------------------------- | -------------- |
| Fig 1     | Paradigm + 行为优势                                | Run-7 memory + log\_rt\_correct + 证据链示意               | ✅ 已有 Fig 1 可复用 |
| Fig 2     | Representational differentiation（What + Where） | 四层 ROI Δ similarity + Searchlight map + noise ceiling | ⚠️ 需合并升级       |
| Fig 3     | Mechanism（Why）                                 | Model-RSA + M9 relational + variance partitioning     | ⚠️ 需补 M9       |
| Fig 4     | Network directionality（Why 的网络层）               | gPPI + representational connectivity + 箭头图            | ❌ 新增           |
| Fig 5     | Learning dynamics（When）                        | Sliding-window 轨迹 + slope × behavior                  | ❌ 新增           |
| Fig 6     | Brain-behavior coupling + mediation（So what）   | within-subject item-level 耦合 + mediation path         | ❌ 新增           |
| Fig 7（可选） | Hippocampal subfield + 激活-表征解耦                 | CA3-DG 分化 + activation vs reorganization 分离           | 当前 Fig 5 可并入   |

主文图上限按 NC/NHB 惯例为 6–7 张；若冲 NN，Fig 7 可进正文。

***

## 八、两条推荐排期

### 快路径：4 周冲 PNAS / Communications Biology 安全档

- **Week 1**：P0.1（within-subject 耦合）+ P0.6（刺激控制）+ QC 封堵
- **Week 2**：P0.2（searchlight）+ P0.3（learning dynamics）
- **Week 3**：P0.4（M9 relational + variance partitioning）+ P0.5（BF/LOSO/split-half/bootstrap）
- **Week 4**：P1.3（mediation）+ P1.4（noise ceiling）+ 图表收敛（Fig 1–6）

### 长路径：8 周冲 NC/NHB（必要时冲 NN）

- **Week 1–2**：P0 全部完成 + QC
- **Week 3–4**：P1.1（hippocampal subfield）+ P1.2（gPPI + representational connectivity）
- **Week 5**：P1.3（mediation）+ P1.4（noise ceiling）+ P1.5（CV-RSA）
- **Week 6**：P2.1（CPM）+ P2.2（Career of Metaphor）
- **Week 7**：P2.3（LI）+ P2.4（RD/GPS）+ 主文图 1–7 定稿
- **Week 8**：P3（SI + QC + 可复现包）+ cover letter

***

## 九、已完成工作存档（历史记录）

### 已完成 P0（主故事 + 稳健性）

- [x] 固定论文主文口径（yy 学习后 pre→post similarity 下降 + pair-based differentiation / semantic reweighting）
- [x] 细化行为分析（memory + log\_rt\_correct 双终点，`behavior_refined.py`）
- [x] 最小脑-行为 between-subject 相关（阴性：r = -0.163, p = .417）→ **待 P0.1 替换**
- [x] trial / voxel 稳健性（逐被试 exact\_match）
- [x] `atlas_robustness` 层（9 ROI 重复 yy 差异化）
- [x] 空间 ROI 负结果讨论定稿（`literature_spatial` 中 kj 无稳健 differentiation）
- [x] 激活-表征解耦图（Figure 5 旧版）
- [x] 主文 5 张图第一版

### 历史性结果支柱（继续沿用）

- Step 5C：yy 在四层 ROI 中均显著 pre→post similarity 下降
- Step 5D ROI 级：左 Precuneous / 左 IFG 的 M2/M8 镜像 + 左 temporal pole 的 M3 减弱
- Step 5D Δρ LMM：`main_functional` 中 M3 最稳定负 Δρ；`literature` 中 M2 仅趋势
- QC / reliability：不支持 post 整体测量崩坏

***

## 十、写作边界（所有档次都必须保留）

**继续使用**：

- "representational differentiation / reorganization"
- "pair-based differentiation + semantic reweighting"
- "行为收益与分化增强共同反映更精细的关系编码"

**新增子刊口径**：

- "semantic unbinding"（分化的具体机制）
- "relational-level re-coding"（变换维度）
- "network-level directionality"（网络方向性证据）
- "individual item-level coupling"（个体差异证据）

**继续禁止**：

- "学习后同 pair 更像 / convergence"
- "隐喻学习失败"
- "已经完全排除所有质量解释"
- "literature 层 Δρ 强裁决"
- "M7\_memory 当前主导"
- **"M7 不显著说明回忆强度与神经几何无关"**（M7 在当前数据下是二值 RDM，这一结论过度解读 —— 见 P0.0a）
- **把 M7 与 M2/M3/M8 并列作为主机制模型**（必须降级为 Secondary —— 见 P0.0a）
- **"`main_functional` M3 Δρ LMM p<.001 证明 M3 是主导机制"**（过度解读；该结果只反映跨异质 ROI 的方向一致性 —— 见 P0.0b）
- **把 Δρ LMM 当作机制定位证据**（机制定位由单 ROI 配对 t 承担；Δρ LMM 只做一致性检查 —— 见 P0.0b）

**M7 相关的新增必备写作口径（P0.0a 配套）**：

- "M7_memory 在当前实验设计下为二值近似（run-7 为单次最终测验），其不显著结果不能排除度量粒度限制"
- "脑-行为耦合的更精细检验见 within-subject trial-level LMM（见 P0.1），该方法绕开 RDM 二值化约束"
- "M7 的连续变体（M7_continuous_confidence / M7_learning_weighted / M7_IRT）作为 SI 敏感性分析"

**Δρ LMM 相关的新增必备写作口径（P0.0b 配套）**：

- "Δρ LMM 在合集 ROI 上估计的是跨异质 ROI 的方向一致性，不是机制强度"
- "机制定位由单 ROI 配对 t 承担（左 Precuneous / 左 IFG / 左颞极）；Δρ LMM 的作用是鲁棒性检查"
- "`main_functional` 合集混合了 Metaphor>Spatial 与 Spatial>Metaphor 两家族；主文报告家族分层 Δρ LMM 或仅 Metaphor>Spatial 子集合的 Δρ LMM"
- "literature 层 Δρ LMM `converged = false`，仅作趋势参考，不作定量裁决"

***

## 十一、风险与退路

- 若 **P0.1 within-subject 耦合** 仍阴性 → 退至 **M9 relational 的 post 独立解释率提升** 作为 "So what" 替代证据。
- 若 **P1.1 hippocampal subfield** 数据精度不够（ASHS mask 质量差）→ 退至 **anterior vs posterior hippocampus 粗分**，仍能讲通 Schlichting 2015 那条线。
- 若 **P1.2 gPPI** 效应弱 → 退至 **representational connectivity**（它更敏感，不依赖 BOLD 幅度）。
- 若 **Searchlight RSA** 在 cluster-FWE 下无显著 → 改用 **TFCE + 5000 permutation**，或降级为 ROI-based support figure。
- 若所有机制扩展都弱 → 退至 **PNAS 档**，以"多层 ROI × baseline × within-subject 耦合"作为主卖点。

***

*最后更新：2026-04-27，合并自 todo.md 与 task\_plan\_top\_journals.md*
