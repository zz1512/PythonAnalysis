# 收敛版执行计划

最后更新：2026-04-27

这份文档是 [`todo.md`](./todo.md) 的执行版。它的目标不是把所有工作平均铺开，而是用更现实的节奏推进：

1. 先修解释硬债
2. 再补主故事最短板
3. 然后做高收益的机制增强
4. 最后才考虑高成本扩展

---

## 一、执行原则

### 1. 先地基，后扩展

如果以下三件事没有完成，不应继续大规模扩展：

- `M7` 的解释降级与连续变体修复
- `Δρ LMM` 的异质 ROI 处理
- within-subject item-level 脑-行为分析

### 2. 每一阶段只设一个主目标

每个阶段可以并行做若干支持任务，但只能有一个“必须产出”的主结果。

### 3. 不丢任务，但可以后置任务

高成本任务保留，不删除；只有当更前面的结果稳定后才启动。

### 4. 统一写入 `paper_outputs/`

从这一版计划开始，新增分析默认都应落地到：

- `paper_outputs/figures_main`
- `paper_outputs/figures_si`
- `paper_outputs/tables_main`
- `paper_outputs/tables_si`
- `paper_outputs/qc`

---

## 二、阶段结构

### Phase A：修解释硬债

目标：让当前主故事在统计与解释上站稳。

包含：

- A1 `M7` 修复与降级
- A2 `Δρ LMM` family split
- A3 within-subject item-level 脑-行为
- A4 刺激控制
- A5 统一输出体系

### Phase B：补主文增强证据

目标：把稿件从“可投”提升到“有说服力”。

包含：

- B1 learning dynamics
- B2 representational connectivity
- B4 稳健性三件套
- B5 noise ceiling

### Phase C：补空间与机制外延

目标：补 where / why 的更高层级版本。

包含：

- B3 searchlight
- C1 M9 relational
- C2 hippocampal subfield

### Phase D：高风险高收益扩展

目标：只在前面都顺利时再冲。

包含：

- C3 gPPI
- C4 effective connectivity
- C5 CPM
- C6 Career of Metaphor
- C7 LI / RD / GPS

---

## 三、依赖关系

### 强依赖

- `M7` 修复 → 决定后续 Model-RSA 的主/辅模型口径
- `Δρ LMM` family split → 决定 Fig 3 / 机制段如何写
- item-level 脑-行为 → 决定 “so what” 是否能进入主文
- learning dynamics → 决定 “when” 是否能进入主文

### 弱依赖

- representational connectivity 不依赖 searchlight
- searchlight 不依赖 M9 relational
- hippocampal subfield 不依赖 CPM

### 默认后置

- gPPI
- effective connectivity
- CPM
- Career of Metaphor

这些都不应阻塞前面的主线闭环。

---

## 四、推荐时间表

这里不再假设每周都能完整跑完整包分析，而是按更稳妥的阶段推进。

### 第 1 周：只做解释硬债

主目标：

- 完成 `M7` 修复与降级口径

并行支持：

- 为 `Δρ LMM family split` 搭建代码入口
- 整理刺激控制数据源

本周验收：

- 有 `M7` 变体表
- 默认主机制模型中不再强绑 `M7`
- 文稿口径更新完

### 第 2 周：完成 Δρ LMM 异质性处理

主目标：

- 完成 `Δρ LMM family split`

并行支持：

- leave-one-roi-out
- random-slope sensitivity（若实现成本可控）

本周验收：

- 有 family-split 结果表
- 能明确说清 `main_functional` pooled fit 的边界

### 第 3 周：完成 item-level 脑-行为

主目标：

- 跑通 within-subject item-level 脑-行为耦合

并行支持：

- refined 行为表与 RSA 表的 trial-level 对齐
- 先只做主功能 ROI

本周验收：

- 有 `table_itemlevel_coupling.tsv`
- 有一版主图
- 能判断“so what”是否真正补强

### 第 4 周：刺激控制 + 输出体系

主目标：

- 统一 `paper_outputs/`

并行支持：

- 刺激控制表
- 若需要，主 LMM 加 covariate sensitivity

本周验收：

- 新增结果不再散落在老目录
- 刺激控制可写入方法和 SI

### 第 5 周：learning dynamics

主目标：

- 补 “when”

并行支持：

- slope-behavior
- 主图第一版

本周验收：

- 有 `table_learning_slope.tsv`
- 有 `fig_learning_dynamics.png`

### 第 6 周：representational connectivity + 稳健性

主目标：

- 补网络层最贴主线的证据

并行支持：

- BF
- LOSO
- bootstrap
- noise ceiling

本周验收：

- 有一条可写入主文的网络层证据
- 主结果稳健性表齐

### 第 7 周：searchlight / M9 二选一优先

主目标：

- 二选一：
  - 若你最需要 `where`，优先 searchlight
  - 若你最需要 `why`，优先 M9 relational

并行支持：

- 另一项只做搭架子，不强求当周完成

本周验收：

- 至少有一个高价值扩展进入可写状态

### 第 8 周：是否继续冲高

条件判断：

- 如果 Phase A + B 结果都稳，可以考虑进入：
  - hippocampal subfield
  - gPPI
  - CPM

- 如果前面仍有未闭环项，优先写稿，不再继续扩张

---

## 五、每阶段的停止规则

### Stop Rule 1

如果 `M7` 连续变体仍然不稳定：

- 不继续把精力投在 `M7`
- 直接把它降级到 SI
- 主文专注 `M1/M2/M3/M8`

### Stop Rule 2

如果 item-level 脑-行为仍然阴性：

- 不再把“强脑-行为耦合”当主卖点
- 主文改为组水平现象 + 机制重组
- 后续优先补 dynamics 或 network，而不是继续死磕相关

### Stop Rule 3

如果 representational connectivity 不稳：

- 不立刻转去 effective connectivity
- 先用 learning dynamics + searchlight 补足主文

### Stop Rule 4

如果 searchlight 长时间卡住：

- 直接降级到支持性 SI
- 不阻塞主稿

---

## 六、输出映射

### 主文优先输出

- `fig_itemlevel_coupling.png`
- `fig_learning_dynamics.png`
- `fig_repr_connectivity.png`
- 必要时 `fig_searchlight_delta.png`

### SI 优先输出

- `table_m7_variants_comparison.tsv`
- `table_delta_rho_family_split.tsv`
- `table_delta_rho_leave_one_roi_out.tsv`
- `table_stimulus_control.tsv`
- `table_bayes_factors.tsv`
- `table_loso.tsv`
- `table_bootstrap_ci.tsv`
- `table_noise_ceiling.tsv`

---

## 七、现实版投稿门槛

### 可投第一档：PNAS / Communications Biology / Cerebral Cortex

满足以下即可：

- Phase A 完成
- Phase B 至少完成两项
- 主文图与 SI 已经闭环

### 可冲更高一档：NC / NHB

建议额外满足：

- item-level 脑-行为显著
- learning dynamics 清楚
- 至少一条网络层证据清楚

### 更高风险目标：NN / Neuron

不作为当前默认目标，只在以下条件同时满足时考虑：

- 前述条件全部成立
- M9 relational 或 hippocampal subfield 给出强机制增益
- 至少一项高风险扩展真的显著且解释稳

---

## 八、每周检查模板

### 本周主目标

- [ ] 只写 1 项

### 本周支持任务

- [ ] 最多 2–3 项

### 本周输出

- [ ] 表
- [ ] 图
- [ ] 文稿口径更新

### 本周决策

- [ ] 继续当前路径
- [ ] 降级某个任务
- [ ] 推迟某个任务

---

## 九、当前建议的立刻行动

按顺序：

1. `M7`
2. `Δρ LMM family split`
3. item-level 脑-行为
4. 刺激控制
5. `paper_outputs`

在这 5 件事完成前，不建议重新把精力放到：

- gPPI
- effective connectivity
- CPM
- Career of Metaphor

这些工作都保留，但现在不该抢占主线资源。
