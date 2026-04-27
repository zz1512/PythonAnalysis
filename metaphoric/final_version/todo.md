# 子刊冲刺收敛版 TODO

最后更新：2026-04-27

这份文档是当前阶段唯一的任务清单。目标不是“把所有能做的分析都立刻做完”，而是：

1. 先修掉会影响解释的硬问题
2. 再补最能增强主故事的证据
3. 最后再做高风险、高成本的扩展分析

相关文档：

- 核心结果口径：[result.md](./result.md)
- 详细执行排期：[execution_plan.md](./execution_plan.md)
- 方法与设计说明：[readme_detail.md](./readme_detail.md)
- 结果逐步解读：[result_walkthrough.md](./result_walkthrough.md)

---

## 一、当前固定主线

当前不再重构的核心故事：

- 行为层：`yy` 比 `kj` 记得更好，而且正确提取更快
- 表征层：`yy` 在多层 ROI 中都表现出更强的 pre→post similarity 下降
- 机制层：当前最支持 `pair-based differentiation + semantic reweighting`
- 边界层：
  - 现有 between-subject 脑-行为相关为阴性，不能当主支柱
  - `M7_memory` 当前存在度量粒度问题，不能继续并列为主机制模型
  - `Δρ LMM` 只能作为跨 ROI 一致性检查，不能替代单 ROI 机制定位

后续所有新增分析都必须直接服务这条主线，而不是平行堆砌结果。

---

## 二、优先级原则

### A层：必须先完成

这些任务不做，后面的机制扩展会建立在不稳的地基上。

### B层：强烈建议完成

这些任务最有可能把稿件从“能投”提升到“更有竞争力”。

### C层：保留但后置

这些任务有价值，但高成本、高风险，或者当前不是解释闭环的最短路径。

---

## 三、A层：必须先完成

### A1. 修正 M7 的解释风险

目标：不再把当前近似二值的 `M7_memory` 当作主机制模型。

- [x] 在 [`rsa_analysis/build_memory_strength_table.py`](./rsa_analysis/build_memory_strength_table.py) 中新增 `--score-mode`
  - `binary`
  - `confidence`
  - `learning_weighted`
  - `irt`（可先占位，后实现）
- [x] 最低要求：先跑出 `M7_continuous_confidence`
- [x] 在 [`rsa_analysis/model_rdm_comparison.py`](./rsa_analysis/model_rdm_comparison.py) 中固定主解释口径为 `M1/M2/M3/M8`
- [x] `M7` 改为 secondary / sensitivity analysis，但继续保留在模型执行链中
- [ ] 输出：
  - `paper_outputs/tables_si/table_m7_variants_comparison.tsv`
- [x] 文稿要求：
  - 在方法和讨论中明确写出当前 `M7_binary` 的局限
  - 明确“不显著不等于 memory 与 neural geometry 无关”

### A2. 修正 Δρ LMM 的异质 ROI 平均问题

目标：避免把异质 ROI 混成一个 pooled 结果来做过度解释。

- [x] 在 [`rsa_analysis/delta_rho_lmm.py`](./rsa_analysis/delta_rho_lmm.py) 中新增 family split / ROI split 支持
- [ ] 最低要求：
  - [x] 按 `base_contrast` 或等价家族拆开 `main_functional`
  - [x] 至少提供：
    - pooled
    - family-split
  - [ ] leave-one-roi-out sensitivity
- [ ] 输出：
  - [ ] `paper_outputs/tables_si/table_delta_rho_family_split.tsv`
  - [ ] `paper_outputs/tables_si/table_delta_rho_leave_one_roi_out.tsv`
  - [ ] `paper_outputs/tables_si/table_delta_rho_random_slope.tsv`（如实现）
- [x] 文稿要求：
  - `Δρ LMM` 只写“跨 ROI 一致性”，不写成机制定位本体

### A3. 将脑-行为分析升级到 within-subject item level

目标：替代当前阴性的 between-subject 相关，给出更敏感、更直接的“so what”证据。

- [x] 新建 [`brain_behavior/itemlevel_brain_behavior.py`](./brain_behavior/itemlevel_brain_behavior.py)
- [x] 分析结构：
  - 因变量：run-7 `memory`（0/1）
  - 自变量：`item_similarity_post - item_similarity_pre`
  - 模型：trial-level GEE（within-subject item-level decomposition）
- [x] 最低要求：
  - `yy`
  - `kj`
  - 至少主功能 ROI
- [x] 推荐扩展：
  - literature ROI
  - 正确 trial RT 版本
- [x] 输出：
  - `paper_outputs/tables_main/table_itemlevel_coupling.tsv`
  - `paper_outputs/figures_main/fig_itemlevel_coupling.png`

### A4. 刺激控制表

目标：提前封住“yy / kj 刺激不均衡导致效应”的常见质疑。

- [ ] 整理：
  - 词频
  - 笔画
  - 具象性
  - 熟悉度
  - imageability
- [ ] 输出：
  - `paper_outputs/tables_si/table_stimulus_control.tsv`
- [ ] 如存在不平衡：
  - 将相关变量作为 covariate 重跑核心 LMM sensitivity

### A5. 统一输出目录到 `paper_outputs/`

目标：停止把结果散落在多个老目录中，保证后续图表、SI、主文可追踪。

- [ ] 为关键脚本提供统一输出入口或包装层：
  - behavior
  - RSA
  - Model-RSA
  - brain-behavior
  - QC
- [ ] 至少保证新增结果统一落到：
  - `paper_outputs/figures_main`
  - `paper_outputs/figures_si`
  - `paper_outputs/tables_main`
  - `paper_outputs/tables_si`
  - `paper_outputs/qc`

---

## 四、B层：强烈建议完成

### B1. Learning dynamics

目标：补上 “when”。

- [ ] 运行 [`temporal_dynamics/learning_dynamics.py`](./temporal_dynamics/learning_dynamics.py)
- [ ] 核心检验：
  - `similarity ~ window_index`
  - 比较 `slope_yy` 与 `slope_kj`
- [ ] 输出：
  - `paper_outputs/tables_main/table_learning_slope.tsv`
  - `paper_outputs/figures_main/fig_learning_dynamics.png`

### B2. Representational connectivity

目标：优先用更贴近主故事的方法补网络层证据，而不是一上来重押 gPPI / effective connectivity。

- [ ] 新建或完善 `representational_connectivity.py`
- [ ] 至少覆盖：
  - left temporal pole
  - left IFG
  - AG / Precuneus
  - hippocampus（如可行）
- [ ] 输出：
  - `paper_outputs/tables_main/table_repr_connectivity.tsv`
  - `paper_outputs/figures_main/fig_repr_connectivity.png`

### B3. Searchlight RSA

目标：补 “where”，但优先级低于 A 层和 dynamics。

- [ ] 运行 [`representation_analysis/rd_searchlight.py`](./representation_analysis/rd_searchlight.py)
- [ ] 最低要求：
  - `yy`
  - `kj`
  - `baseline`
  - group-level permutation
- [ ] 输出：
  - `paper_outputs/tables_main/table_searchlight_peaks.tsv`
  - `paper_outputs/figures_main/fig_searchlight_delta.png`

### B4. 稳健性三件套

目标：用最少但高价值的稳健性分析支撑主文。

- [ ] Bayes Factor for primary endpoint
- [ ] LOSO
- [ ] bootstrap CI
- [ ] split-half replication
- [ ] 输出：
  - `paper_outputs/tables_si/table_bayes_factors.tsv`
  - `paper_outputs/tables_si/table_loso.tsv`
  - `paper_outputs/tables_si/table_bootstrap_ci.tsv`
  - `paper_outputs/tables_si/table_splithalf.tsv`

### B5. Noise ceiling

- [ ] 对 Step 5C / 关键 Model-RSA 结果计算 noise ceiling
- [ ] 输出：
  - `paper_outputs/tables_si/table_noise_ceiling.tsv`
- [ ] 仅在图上作为支持信息，不替代主统计

---

## 五、C层：保留但后置

这些任务都保留，不删，但默认不进入最近两周的主执行线。

### C1. M9 relational + variance partitioning

- [ ] 关系类别 RDM
- [ ] source / target domain RDM
- [ ] variance partitioning
- [ ] 输出：
  - `paper_outputs/tables_main/table_variance_partitioning.tsv`
  - `paper_outputs/figures_main/fig_variance_partitioning.png`

说明：这是很有潜力的机制升级项，但定义和解释风险都高，不再强绑 A 层。

### C2. Hippocampal subfield / MTL 扩展

- [ ] subfield mask
- [ ] CA1 / CA3-DG / subiculum / anterior-posterior hippocampus
- [ ] 输出：
  - `paper_outputs/tables_main/table_hippocampal_subfield.tsv`
  - `paper_outputs/figures_main/fig_hippocampal_subfield.png`

### C3. gPPI

- [ ] 作为网络层补充证据保留
- [ ] 不再默认和 representational connectivity 同优先级

### C4. Effective connectivity

- [ ] 保留
- [ ] 默认 SI / exploratory

### C5. CPM

- [ ] 保留
- [ ] 默认后置到较高投稿目标再做

### C6. Career of Metaphor

- [ ] 保留
- [ ] 作为理论扩展，不阻塞当前主线

### C7. LI / RD / GPS

- [ ] 保留
- [ ] 默认 SI / 扩展模块

---

## 六、QC 与交付物

这些不是“可做可不做”，而是投稿前必须齐。

### QC

- [ ] `paper_outputs/qc/lss_qc_summary.tsv`
- [ ] `paper_outputs/qc/pattern_qc_summary.tsv`
- [ ] `paper_outputs/qc/motion_scrubbing_summary.tsv`
- [ ] `paper_outputs/qc/fake_handling_note.md`

### 交付管理

- [ ] `paper_outputs/analysis_plan.tsv`
- [ ] `paper_outputs/figure_stats_map.tsv`
- [ ] 主文图与 SI 图逐一绑定统计来源

---

## 七、按投稿档次的停止规则

### 到达“可投 PNAS / Communications Biology / Cerebral Cortex”门槛

满足以下即可进入写稿：

- A层全部完成
- B层至少完成 2 项
  - 优先 `Learning dynamics`
  - 优先 `Representational connectivity` 或 `Searchlight`

### 到达“可冲 NC / NHB”门槛

在上一层基础上，再至少满足：

- `Learning dynamics`
- 一条稳定的网络层证据
- 一条比现有 between-subject 更强的脑-行为证据

### 冲更高档前再考虑

- M9 relational
- hippocampal subfield
- CPM
- Career of Metaphor

---

## 八、最近两周的现实推进顺序

### 第 1 阶段

- [ ] A1 `M7`
- [ ] A2 `Δρ LMM`
- [ ] A3 item-level 脑-行为

### 第 2 阶段

- [ ] A4 刺激控制
- [ ] A5 `paper_outputs`
- [ ] B1 learning dynamics

### 第 3 阶段

- [ ] B2 representational connectivity
- [ ] B4 稳健性三件套
- [ ] B5 noise ceiling

### 第 4 阶段

- [ ] B3 searchlight
- [ ] C1 M9 relational
- [ ] 其余 C 层任务按时间和结果收益决定

---

## 九、已完成工作存档

以下内容已完成，不再重复立项，只在后续文稿中整合：

- [x] 主线口径固定
- [x] refined behavior
- [x] 主 ROI / literature / literature_spatial / atlas_robustness 的 Step 5C
- [x] 主 ROI / literature 的 Model-RSA 与 `Δρ LMM`
- [x] trial / voxel robustness
- [x] QC / reliability 第一轮
- [x] 主文关键图第一轮

这些是当前主线地基，不是当前待办。

---

## 十、执行边界

继续允许的表述：

- representational differentiation / reorganization
- pair-based differentiation
- semantic reweighting

继续禁止的表述：

- “学习后同 pair 更像”
- “M7 不显著说明记忆强度与神经几何无关”
- “Δρ LMM 直接证明主导机制”
- “已经完全排除所有质量解释”

---

## 十一、如果进度受阻，优先牺牲什么

按顺序后撤：

1. 先牺牲 `effective connectivity`
2. 再牺牲 `CPM`
3. 再牺牲 `Career of Metaphor`
4. 再牺牲 `LI / RD / GPS`
5. 最后才考虑牺牲 `Searchlight`

不应牺牲：

- `M7` 修复
- `Δρ LMM` 异质性处理
- item-level 脑-行为
- 刺激控制

因为这些直接关系到当前主文解释是否站得住。
