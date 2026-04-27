# 子刊冲刺 8 周详细执行规划

> 本文件是基于 [`todo.md`](./todo.md) 的**可执行工程排期**，面向 NN / NC / NHB（最低 PNAS）投稿。
>
> 配套文档：
> - 任务定义 + 验收标准：[`todo.md`](./todo.md)
> - 当前结果与口径：[`result.md`](./result.md)
> - 每步方法解读（汇报用）：[`result_walkthrough.md`](./result_walkthrough.md)
> - 怎么跑：[`README.md`](./README.md)
>
> **使用方式**：每天开工前勾选当天任务；每周三、周日做交付检查点；若进度落后 50%，触发第五节的风险应对决策树。

---

## 零、规划制定原则

1. **工程依赖优先**：有些任务是其他任务的前置依赖，必须先跑
2. **致命缺口前置**：P0.0a/b（技术债）和 P0.1（救场）放最前，防止主文故事在投稿前崩
3. **长尾并行**：计算密集型任务（searchlight / permutation / LSS）在后台跑，不阻塞前端分析
4. **每周两次"交付检查点"**：周三和周日，防止任务堆积
5. **留 20% buffer**：8 周计划里第 7–8 周留机动时间应对突发问题

---

## 一、任务依赖关系图

```
P0.0a (M7 修复)                   ──┐
P0.0b (Δρ LMM 修复)              ──┼──► 主文 Fig 3 重绘 (依赖)
P0.6 (刺激控制)                   ──┘

P0.1 (within-subject 耦合) ────────► P1.3 (中介分析) ────► Fig 6 主图
P0.2 (searchlight RSA) ─────────────► P1.4 (noise ceiling) ────► Fig 2 升级
P0.3 (learning dynamics) ───────────► Fig 5 主图
P0.4 (M9 relational + VP) ──────────► Fig 3 升级

P0.5 (BF/LOSO/split-half) ──────────► SI 表（不阻塞任何下游）

P1.1 (hippocampal subfield) ────────► Fig 7 (可选)
P1.2 (gPPI + repr-conn) ────────────► Fig 4 主图
P1.5 (CV-RSA)  ─────────────────────► SI 表

P2.1 (CPM) ─────────────────────────► SI
P2.2 (Career of Metaphor) ──────────► SI
P2.3 (LI) ──────────────────────────► SI
P2.4 (RD/GPS) ──────────────────────► SI

P3 (SI 整理 + QC + 可复现) ──────────► 投稿前 1 周集中
```

**关键依赖（不能逆转）**：

- **P0.0a + P0.0b 必须先做**，否则下游所有机制分析的解读都有风险
- **P0.1 必须先做**，否则 "So what" 是空的，新做的分析没有意义
- P1.3 依赖 P0.1 完成后的 item-level 耦合数据
- P1.4 noise ceiling 依赖 P0.2 searchlight 数据

---

## 二、8 周详细排期

### 🔴 Week 1：技术债清零 + 救场分析启动（最关键一周）

**目标**：把 P0.0a/b 两个解读风险修掉，**不做新分析只做修复**；同时启动 P0.1 的救场路径。

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P0.0a step 1**：扩展 `build_memory_strength_table.py` 支持 `--score-mode {binary, confidence, learning_weighted, irt}` | 4h | 代码完成 |
|  | **P0.0a step 2**：生成 `confidence_score = memory + 0.4 × (1 - normalized_log_rt)` | 2h | 数据表 |
| **Tue** | **P0.0a step 3**：重跑 Model-RSA，输出 `M7_binary` / `M7_continuous_confidence` / `M7_learning_weighted` 三个变体的 Δρ 对比表 | 6h | `table_m7_variants_comparison.tsv` |
| **Wed** | **P0.0b step 1**：扩展 `delta_rho_lmm.py --family-split`；按 `base_contrast` 分 `main_functional` 为两家族 | 4h | 代码完成 |
|  | **P0.0b step 2**：跑家族分层 Δρ LMM（两家族 × 5 模型），输出系数表 | 2h | `table_delta_rho_family_split.tsv` |
|  | **🔶 交付检查点 1**：M7 修复 + Δρ LMM 家族分层结果是否改变故事 | — | 决策：Fig 3 改造方案 |
| **Thu** | **P0.0b step 3**：仅用 "Metaphor>Spatial" 派生 ROI 重跑 Δρ LMM | 3h | 第二个 SI 表 |
|  | **P0.0b step 4**：随机斜率 `(C(model)\|roi)` 敏感性 + leave-one-ROI-out | 3h | `table_delta_rho_random_slope.tsv` |
| **Fri** | **P0.6**：刺激语言学控制表（词频、笔画、具象性、imageability） | 6h | `table_stimulus_control.tsv` |
|  | 若不均衡 → 把协变量加入 LMM 重跑主分析 | — | 主结果 sensitivity |
| **Sat** | **P0.1 step 1**：实现 `itemlevel_brain_behavior.py` 框架 | 5h | 代码完成 |
| **Sun** | **P0.1 step 2**：构造 `Δ similarity = item_sim_post - item_sim_pre` 的逐 trial 表 | 3h | item-level 表 |
|  | **🔶 交付检查点 2**：Week 1 全部技术债修复 + P0.1 脚手架就位 | — | Week 1 总结 |

**Week 1 产出清单**：

- [ ] `table_m7_variants_comparison.tsv`
- [ ] `table_delta_rho_family_split.tsv`
- [ ] `table_delta_rho_random_slope.tsv`
- [ ] `table_stimulus_control.tsv`
- [ ] `itemlevel_brain_behavior.py` 脚手架 + item-level 表

---

### 🔴 Week 2：救场完成 + 动态 / 搜索光启动

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P0.1 step 3**：拟合 trial-level GLMM：`memory ~ delta_similarity × condition + (1 + delta_similarity\|subject) + (1\|item)`，binomial family | 5h | 主 LMM 结果 |
| **Tue** | **P0.1 step 4**：within-subject logistic 逐被试版本，对 27 个斜率做 one-sample t 复核 | 3h | 复核表 |
|  | **P0.1 step 5**：4 层 ROI × yy/kj 两条件批量运行 | 3h | 完整结果表 |
| **Wed** | **P0.1 step 6**：绘制 `fig_itemlevel_coupling.png`（小提琴 + 被试斜率点 + 组均值） | 4h | 主 Fig 6 |
|  | **🔶 交付检查点 3**：P0.1 是否显著？救场是否成功？→ 影响后续策略 | — | 是否启动风险退路 |
| **Thu** | **P0.3 step 1**：运行 `learning_dynamics.py --window-size 20 --compute-mvpa` 在主功能 + literature ROI 上 | 6h（后台） | sliding-window 数据 |
| **Fri** | **P0.3 step 2**：拟合 `similarity ~ window_index` 斜率（yy / kj 各一组），paired t + BF10 | 4h | 组水平结果 |
|  | **P0.3 step 3**：slope × behavior 相关（Spearman + bootstrap 95% CI） | 2h | 个体差异结果 |
| **Sat** | **P0.3 step 4**：绘制 `fig_learning_dynamics.png`（多 ROI 轨迹 + 组斜率条） | 4h | 主 Fig 5 |
| **Sun** | **P0.2 step 1**：运行 `rd_searchlight.py` 三条件 × pre/post 单被试 map（**后台跑，可能 >12h**） | 12h（后台） | 单被试 map |
|  | **🔶 交付检查点 4**：Week 2 主救场 + 动态 + searchlight 启动 | — | Week 2 总结 |

**Week 2 产出**：P0.1 救场结果 + P0.3 完整结果 + P0.2 后台进行中

---

### 🟡 Week 3：机制深化（M9 + searchlight 完成）

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P0.2 step 2**：group-level cluster permutation（5000 次） | 6h（后台） | group map |
|  | **P0.4 step 1**：标注 yy pair 的关系类型（空间→时间 / 物理→情感等映射类别），或用 BERT relational embedding + PCA | 4h | `M9_relational` RDM 文件 |
| **Tue** | **P0.4 step 2**：构造 `M10_source_domain` / `M11_target_domain` RDM | 3h | 额外模型 RDM |
|  | **P0.4 step 3**：扩展 `model_rdm_comparison.py` 支持 variance partitioning（M3 / M9 / M10 / M11 联合回归，拆出独立贡献） | 4h | 代码 |
| **Wed** | **P0.4 step 4**：跑 variance partitioning，输出 pre vs post 各模型独立贡献率 | 3h | VP 表 |
|  | **P0.4 step 5**：关键假设检验：**post 阶段 M9 独立贡献率 > pre** | 2h | 主结论 |
|  | **🔶 交付检查点 5**：M9 是否显著？机制故事是否深化 | — | 决策 |
| **Thu** | **P0.4 step 6**：绘制 `fig_variance_partitioning.png`（甜甜圈 / 堆叠柱图） | 4h | 主 Fig 3 升级 |
| **Fri** | **P0.2 step 3**：提取 searchlight 显著 cluster peak，与 Huang 2023 ALE 元分析做 spatial correlation（10000 次置换） | 4h | 外部一致性证据 |
|  | **P0.2 step 4**：绘制 `fig_searchlight_delta.png` + 输出 peak 表 | 3h | 主 Fig 2 升级 |
| **Sat** | **P0.5 step 1**：对 primary endpoint 的 condition×time 交互加 Bayes Factor（调用 `utils/bayesian_stats.py`） | 4h | BF10 表 |
|  | **P0.5 step 2**：LOSO（去掉 1 被试循环 28 次，验证主效应仍显著） | 3h | LOSO 表 |
| **Sun** | **P0.5 step 3**：split-half replication（随机划 14+14，1000 次重随机） | 4h | split-half 表 |
|  | **P0.5 step 4**：bootstrap 95% CI（subject-level Δ similarity，5000 次） | 2h | bootstrap 表 |
|  | **🔶 交付检查点 6**：P0 全部完成 → 进入 P1 | — | Week 3 总结 |

**Week 3 结束时 P0 100% 完成**。此时已有主文 Fig 1–Fig 6 的第一版底稿（Fig 1 / 2 / 3 / 5 / 6）。

---

### 🟡 Week 4：网络层 + MTL（P1.1 + P1.2 启动）

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P1.1 step 1**：生成 hippocampal subfield mask（ASHS 或用 HCP 现成 subfield atlas 简化版） | 6h | mask 文件 |
| **Tue** | **P1.1 step 2**：实现 `hippocampal_subfield_rsa.py`，跑 CA1 / CA3-DG / subiculum / anterior-posterior hippocampus 的 pre→post Δ similarity | 5h | 代码 + 初步结果 |
| **Wed** | **P1.1 step 3**：检验 CA3-DG repulsion vs CA1 integration 假设 | 3h | 假设检验表 |
|  | **P1.1 step 4**：hippocampus × cortex 的 seed-based representational connectivity | 4h | 连接结果 |
|  | **🔶 交付检查点 7**：subfield 是否给出 CA3-DG differentiation 证据 | — | Fig 7 决策 |
| **Thu** | **P1.1 step 5**：绘制 `fig_hippocampal_subfield.png` | 4h | Fig 7 |
| **Fri** | **P1.2 step 1**：跑 `gPPI_analysis.py` 以学习期 `yy > kj` 左前颞 peak 为 seed，生成 seed × target 调制矩阵 | 6h | gPPI 结果 |
| **Sat** | **P1.2 step 2**：实现 `representational_connectivity.py`（Basti 2020 方法，ROI 对之间 neural RDM 的 Spearman 相关） | 5h | 代码 + 结果 |
| **Sun** | **P1.2 step 3**：跑 `effective_connectivity.py --time learn --condition yy`（简化 Granger / beta-series 方向回归） | 4h | EC 结果 |
|  | **🔶 交付检查点 8**：Week 4 网络层 + MTL 完成 | — | Week 4 总结 |

---

### 🟡 Week 5：网络层收尾 + 中介 + noise ceiling / CV-RSA

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P1.2 step 4**：整合 gPPI + representational connectivity + EC 三条证据，绘制 `fig_network_directionality.png`（箭头图 / circular plot） | 6h | 主 Fig 4 |
| **Tue** | **P1.3 step 1**：运行 `mediation_analysis.py` 对 3–5 个关键 ROI（左 IFG / 左 Precuneous / 左颞极 / 左前颞 / 左 AG）做 X=activation → M=Δ similarity → Y=memory 的 bootstrap 中介（5000 次） | 5h | 中介结果 |
| **Wed** | **P1.3 step 2**：FDR 跨 ROI 校正 + 绘制 `fig_mediation_path.png` | 4h | 中介主图 |
|  | **🔶 交付检查点 9**：中介是否显著 → 影响主文叙事 | — | 决策 |
| **Thu** | **P1.4 step 1**：对每个 ROI 的 Step 5C similarity 计算 lower / upper noise ceiling（leave-one-subject-out） | 5h | noise ceiling 表 |
|  | **P1.4 step 2**：在 Fig 2 中叠加 noise ceiling 灰带 | 2h | Fig 2 升级 |
| **Fri** | **P1.5 step 1**：改写 RSA 流程为 leave-one-run-out cross-validated pattern correlation | 5h | CV-RSA 代码 |
| **Sat** | **P1.5 step 2**：跑 CV-RSA，对比 CV vs non-CV 方向一致性 | 4h | `table_cv_rsa.tsv` |
| **Sun** | 主文第一版文字稿：**从 Week 1–5 的所有结果组装出 Methods + Results 初稿** | 6h | 论文 draft v1 |
|  | **🔶 交付检查点 10**：P1 全部完成 + 论文第一版初稿 | — | Week 5 总结 |

**Week 5 结束时**：如果冲 Communications Biology / Cerebral Cortex 已经足够，**可以选择在此投稿**；如冲 NC/NHB 继续 Week 6–8。

---

### 🟢 Week 6：NC/NHB 冲刺（P2.1 + P2.2）

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon–Tue** | **P2.1 step 1–3**：实现 `cpm_prediction.py`，用 Schaefer 200 ROI 功能连接矩阵做 LOO CPM 预测 `memory_yy - memory_kj` | 12h | CPM 代码 + 预测结果 |
| **Wed** | **P2.1 step 4**：5000 次 permutation 显著性 + 正 / 负 edge 与 DMN / FPCN / Salience 网络 overlap | 5h | CPM 显著性 |
| **Thu** | **P2.1 step 5**：绘制 `fig_cpm_edges.png` | 4h | CPM 图 |
| **Fri** | **P2.2 step 1**：对每个 yy trial 计算 `exposure_count`（学习阶段第 k 次暴露） | 3h | 元数据 |
|  | **P2.2 step 2**：用 `parametric_modulation.py` 重跑 GLM，exposure_count 作参数调制 | 4h | PM GLM 结果 |
| **Sat** | **P2.2 step 3**：预测验证：右半球激活随熟悉度衰减 / pair similarity 非线性轨迹 | 4h | 主结论 |
| **Sun** | **P2.2 step 4**：绘制 `fig_career_of_metaphor.png` | 3h | Career 图 |
|  | **🔶 交付检查点 11**：P2.1–P2.2 完成 | — | Week 6 总结 |

---

### 🟢 Week 7：NN 冲刺（P2.3 + P2.4）+ 主文定稿

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | **P2.3**：运行 `lateralization_analysis.py`，对比 yy / kj 在 IFG / AG / pMTG / temporal pole 的 LI | 5h | LI 表 |
| **Tue** | **P2.4 step 1**：运行 `rd_analysis.py`（阈值 70% / 80% / 90% 三档敏感性） | 5h | RD 结果 |
| **Wed** | **P2.4 step 2**：运行 `gps_analysis.py`，验证 RD 下降 + GPS 变化是否同向支持 RSA | 4h | GPS 结果 |
|  | **P2.4 step 3**：绘制 `fig_rd_gps_complement.png` | 3h | 互补图 |
| **Thu–Fri** | **主文 7 张图定稿**（Fig 1 行为 / Fig 2 RSA+searchlight+noise ceiling / Fig 3 机制+M9+VP / Fig 4 网络 / Fig 5 动态 / Fig 6 耦合+中介 / Fig 7 subfield） | 12h | 定稿图 |
| **Sat–Sun** | **Results + Discussion 完稿 v2**（根据所有新结果重写） | 10h | 论文 v2 |
|  | **🔶 交付检查点 12**：P2 完成 + 主文定稿 | — | Week 7 总结 |

---

### 🟢 Week 8：SI + QC + 投稿准备（P3）

| 天 | 任务 | 预计耗时 | 交付物 |
|---|---|---|---|
| **Mon** | 整理 `paper_outputs/qc/` 全部 QC 文件：`lss_qc_summary.tsv`、`pattern_qc_summary.tsv`、`motion_scrubbing_summary.tsv`、`fake_handling_note.md` | 5h | QC 全套 |
| **Tue** | 完成 `analysis_plan.tsv`（Confirmatory / Secondary / Exploratory 分层 + 多重比较策略） | 4h | 交付物 |
| **Wed** | 完成 `figure_stats_map.tsv`（每张主图 ↔ 唯一主统计检验 ↔ 校正 ↔ 输出路径） | 4h | 交付物 |
| **Thu** | SI 组装：约 20 个 SI 表 + 10 张 SI 图 | 6h | SI 文档 |
| **Fri** | 代码 + ROI mask + 刺激模板打包可复现 release（GitHub private repo 或 OSF） | 5h | 复现包 |
| **Sat** | **Cover letter 起草 + response-to-reviewer 预演**（按 reviewer 常见问题列 FAQ） | 6h | Cover |
| **Sun** | 全文交叉检查：所有 p 值 / CI / N / 图注 / 引用对齐 | 5h | 终稿 |
|  | **🔶 交付检查点 13**：准备投稿 NC/NHB | — | 投稿！ |

---

## 三、每周的"两次交付检查点"规则（防止堆积）

| 检查点 | 周三 | 周日 |
|---|---|---|
| 做什么 | 评估当周剩余风险 | 完整产出 + 写进 result_walkthrough.md |
| 强制问自己 | "有没有任务进度落后 50%？" | "主文的这个结论能不能写下来？" |
| 退路触发 | 若落后，砍 P2 层任务保 P0/P1 | 若失败，触发 todo.md §十一 风险与退路 |

---

## 四、并行化策略（节省时间关键）

### 后台任务（挂着跑，不阻塞）

- **P0.2 searchlight**：12–24h 后台，Week 2 周日启动
- **P1.1 subfield 分析**：可与 P1.2 并行
- **P1.2 gPPI + RepConn + EC**：三个脚本可独立跑
- **P2.1 CPM permutation**：5000 次置换建议过夜跑

### 串行必须等的

- P0.0a/b **必须先于** 所有机制分析解读
- P0.1 结果 **必须先于** P1.3 中介
- P0.2 结果 **必须先于** P1.4 noise ceiling 叠加

---

## 五、风险应对决策树

| 触发条件 | 决策 | 后续行动 |
|---|---|---|
| Week 1 末：P0.0b 家族分层发现两家族方向**相反** | 触发 | 主文机制叙事必须拆写；Fig 3 改两排小图 |
| Week 2 末：**P0.1 仍阴性** | 触发 | 退至 M9_relational 作为"So what"替代证据（见 todo.md §十一） |
| Week 3 末：**M9 不显著** | 触发 | 退至 PNAS 档，主卖点变成"多层 ROI × baseline × within-subject 耦合" |
| Week 4 末：**subfield 数据精度差** | 触发 | 退至 anterior-posterior hippocampus 粗分 |
| Week 5 末：**gPPI 效应弱** | 触发 | 退至 representational connectivity 为主 |
| Week 5 末：**Searchlight 在 cluster-FWE 无显著** | 触发 | 改用 TFCE + 5000 permutation，或降级为 ROI-based support figure |
| Week 6 初：**CPM 阴性** | 触发 | CPM 放 SI，不放主文 |

---

## 六、这周（Week 1）就要立刻做的 5 件事

按优先级顺序：

1. **今天**：扩展 `build_memory_strength_table.py` 的 `--score-mode`，产出 `M7_continuous_confidence` 表
2. **今天**：扩展 `delta_rho_lmm.py` 支持 `--family-split`
3. **明天**：跑两个家族的 Δρ LMM，看方向是否一致（**这决定主文 Fig 3 的结构**）
4. **明天**：开始 `itemlevel_brain_behavior.py` 的脚手架
5. **后天**：跑 P0.6 刺激控制表（最简单但必不可少）

---

## 七、里程碑概览（投稿节点）

| 节点 | 累计周数 | 可投稿档次 | 主要支柱证据 |
|---|---|---|---|
| M1 | Week 3 末 | **PNAS / Communications Biology** 保底档 | P0 全部完成：技术债 + within-subject 耦合 + searchlight + 动态 + M9 + 稳健性三件套 |
| M2 | Week 5 末 | **Nature Communications / NHB** 主目标档 | + P1.1 subfield + P1.2 网络方向性 + P1.3 中介 + P1.4 noise ceiling |
| M3 | Week 7 末 | **Nature Neuroscience / Neuron** 冲刺档 | + P2.1 CPM 跨被试预测 + P2.2 Career of Metaphor + P2.3 LI + P2.4 RD/GPS |
| M4 | Week 8 末 | 投稿 NC/NHB（默认主目标） | + P3 SI + QC + 可复现包 + cover letter |

**推荐投稿策略**：
- 冲到 M3 后直接投 **Nature Communications**
- 若 desk reject → 退到 Communications Biology / Cerebral Cortex（此时 M1 的证据已足够）
- 不在 M1 就先投保底档，浪费顶档机会

---

## 八、每日执行模板（复制到每天 standup）

```
日期：YYYY-MM-DD  |  第 X 周 Day Y

今日任务：
- [ ] 任务 1（预计 Xh）
- [ ] 任务 2（预计 Xh）

昨日产出：
- ...

阻塞项：
- ...

是否触发风险决策树？
- 否 / 是（原因：...，决策：...）
```

---

## 九、补充说明

- **所有交付物必须落盘到 `${PYTHON_METAPHOR_ROOT}/paper_outputs/` 下**（见 todo.md 第四节目录约定）
- **新增分析必须同步更新**：
  - `result_walkthrough.md` 新增"做了什么 / 怎么做 / 关键数字 / 一句结论"段落
  - `todo.md` 把对应 checkbox 打勾
  - `analysis_plan.tsv` 标注 Confirmatory / Secondary / Exploratory
- **遇到方法学不确定**：默认先做 Secondary 标注，投稿前再根据结果决定是否升级为主文 Confirmatory
- **Bayes Factor 阈值**：BF10 > 10 报告为 strong evidence，3 < BF10 < 10 为 moderate，其他为 inconclusive

---

*最后更新：2026-04-27；负责人：项目主笔*
