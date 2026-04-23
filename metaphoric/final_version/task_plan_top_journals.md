# Top-Journal Task Plan (PNAS / NN / Neuron)

本文件把“冲 PNAS / Nature Neuroscience / Neuron”的原则清单，落成可执行的任务计划。目标读者是项目执行者（你自己或研究助理），要求每项任务都能产出可验证的文件与“论文式一句话结论”。

配套文档：
- 短版怎么跑：`metaphoric/final_version/README.md`
- 教学版为什么这么跑：`metaphoric/final_version/readme_detail.md`

---

## 0. 目标与边界

目标分两档：
- PNAS 优先：主线证据链完整、统计推断严谨、替代解释封堵到位，图表收敛（4–6 张主图）。
- NN/Neuron 冲刺：在 PNAS 基础上增加“机制层解释”（模型RDM/连接性/方向性/中介路径择一为主），并增强广义影响力（稳健性、外部一致性或更强理论对话）。

边界（强制）：
- 主文只允许 1 个 primary endpoint（主推断只围绕它展开）。
- Secondary/Exploratory 必须标注层级与阈值，避免“很多分析挑显著”。
- 所有结论必须能追溯到具体输出文件（表格/统计图/阈值图）。

---

## 1. 产出物清单（最终交付）

你最终需要准备三类“交付物”：
- 核心结果（主文）：4–6 张主图 + 每张图背后唯一主统计检验 + 结果句。
- 补充材料（SI）：所有稳健性检验、控制验证、探索性分析的完整表格与图。
- 可复现包：代码 + 关键输出表（不一定公开原始数据，但至少可公开衍生指标与 ROI/mask/模板）。

建议统一一个“论文产出目录”（相对 `${PYTHON_METAPHOR_ROOT}`）：
- `paper_outputs/`
- `paper_outputs/figures_main/`
- `paper_outputs/figures_si/`
- `paper_outputs/tables_main/`
- `paper_outputs/tables_si/`
- `paper_outputs/qc/`

---

## 2. 主结论收敛（必须先做）

### Task 2.1 选定 Primary Endpoint（一次定死）

推荐 primary endpoint（与现有 pipeline 最一致、可解释性最强）：
- item-wise RSA 的 LMM：`similarity ~ condition * time + (1|subject) + (1|item)`
- 主检验：`condition × time` 交互效应（隐喻 yy 的 pre→post 变化幅度是否显著不同于 kj）。

方向性说明（根据当前阶段性结果更新）：
- 目前的 RSA 结果显示：yy 条件的 pair similarity 从 pre 到 post **更明显下降**，而 kj 变化较小或方向不同。
- 因此，本项目的主线更适合用 **representational differentiation（表征分化）** 来命名 primary effect，而不是默认“学习导致配对聚合/对齐”。

交付物：
- 一段“主检验定义”（写进方法或预注册式分析计划页）
- 一句论文式结论模板（示例）：
  - “在 ROI X 中，LMM 显示 `condition×time` 交互显著（b=..., SE=..., t=..., p=...），表明 yy 的表征相似性从 pre→post 的变化幅度显著大于 kj；在当前数据中该变化主要表现为 yy 的相似性下降（分化增强）。”

### Task 2.2 明确分析层级（Confirmatory/Secondary/Exploratory）

定义层级（建议）：
- Confirmatory（主文主结论）：前后测 RSA + LMM（primary endpoint）+ Model-RSA / Δρ LMM（机制主线）
- Secondary（主文或 SI）：GLM 的定位证据；脑-行为关联（预先指定 1 个行为指标）；激活-表征解离图
- Exploratory（SI）：RD/GPS、MVPA（ROI 或 searchlight）、gPPI、有效连接、学习动态、中介、模型RDM扩展等

交付物：
- 一页“分析层级表”（建议放 `paper_outputs/tables_si/analysis_plan.tsv`）
- 每层对应的多重比较策略（FWE/FDR/置换）与阈值

---

## 3. 数据与 QC 封堵（顶刊高频追问点）

### Task 3.1 LSS 完整性与对齐 QC（必须）

跑法参照 `README.md` Step 2–4。

必备 QC（写进 `paper_outputs/qc/`）：
- LSS 质量报告（NaN/空脑/极端值/缺失 trial）
- patterns 文件名与 trial 对齐检查：
  - `pattern_root/sub-xx/pre_yy.nii.gz` 等文件是否齐全
  - `*_metadata.tsv` 的 trial 数与预期一致（至少能解释任何偏差）

交付物：
- `paper_outputs/qc/lss_qc_summary.tsv`
- `paper_outputs/qc/pattern_qc_summary.tsv`
- 一段方法说明：“如何避免 trial 错配导致假阳性”

### Task 3.2 假词/过滤策略写清楚（必须）

要求：
- 说明 fake/nonword 在哪一步被排除（推荐：堆叠 patterns 时排除）
- 说明为什么不能在太早排除（例如 LSS 建模/元数据一致性）

交付物：
- `paper_outputs/qc/fake_handling_note.md`（可 10 行以内，写清策略与检查方式）

### Task 3.3 运动/去噪策略可审计（必须）

要求：
- 报告 scrubbing 规则（FD/DVARS 阈值、是否扩窗）
- 报告每被试被 scrub 的时间点比例（至少一个分布表）

交付物：
- `paper_outputs/qc/motion_scrubbing_summary.tsv`

---

## 4. 主线结果跑通与“论文句子化”

### Task 4.1 学习阶段 GLM（定位证据，主文或 SI）

目的：
- 回答 “哪里在参与”（Where）

交付物：
- `paper_outputs/figures_main/fig_glm_where.png`（或 SI）
- `paper_outputs/tables_main/table_glm_clusters.tsv`
- 一句结论：yy>kj 的关键网络/脑区（注意避免过度解剖故事）

### Task 4.2 Primary Endpoint：前后测 RSA + LMM（主文核心，baseline 必须纳入）

目的：
- 回答 “表征关系怎么变”（What）

交付物（来自 `rsa_analysis/` 输出 + 你汇总到 paper_outputs）：
- `paper_outputs/tables_main/table_rsa_lmm_primary.tsv`
- `paper_outputs/figures_main/fig_rsa_primary_interaction.png`
- 一句结论：交互效应（方向、统计量、CI；本项目当前更倾向“yy 的相似性下降更明显=分化增强”）
- 最低汇报要求：必须同时报告 `yy / kj / baseline` 三条件的 pre/post 方向与效应大小，用 baseline 封堵“纯重测/熟悉化”解释

### Task 4.2.5 激活-表征“解耦”可视化（强烈推荐，主文友好）

动机（根据当前阶段性结果）：
- 学习阶段 GLM 显示：`yy > kj` 的激活范围相对窄（主要左颞极），`kj > yy` 更广泛（后部视觉-顶叶-内侧颞叶网络）。
- 但 RSA 显示：隐喻（yy）的表征变化更强，而空间（kj）的变化更弱或方向不同。
- 这提供一个很强的顶刊叙事钩子：**“激活强弱”与“表征重塑强弱”可以解耦**。

交付物（主文或 SI，取决于图位）：
- `paper_outputs/figures_main/fig_activation_representation_dissociation.png`
- `paper_outputs/tables_si/table_activation_representation_dissociation.tsv`
- 一句结论：哪些 ROI 属于“低激活-高重塑”（语义枢纽）与“高激活-低重塑”（后部空间网络）

### Task 4.3 Model-RSA + Δρ LMM（主流程机制解释）

目的：
- 回答 “为什么会变”（Why）

当前定位（根据你的最新思路，已从“机制扩展”升为主流程）：
- Model-RSA 不再放在后面当可选加分项，而是与 RSA 主效应并列进入主文主线
- 推荐的主文证据链变为：
  - 行为差异
  - 学习阶段 GLM（Where）
  - 前后测 RSA + baseline 控制（What）
  - Model-RSA / Δρ LMM（Why）
  - 脑-行为关联（So what）

模型集合（当前主线）：
- M1_condition：条件结构基线
- M2_pair：配对“收敛/对齐”路径
- M8_reverse_pair：配对“分化增强”路径
- M3_embedding：预存语义基线
- M7_memory：记忆强度/回忆强度路径

关键解释原则：
- 不再预设学习后“同 pair 必然更相似”
- 当前结果显示 yy 的 pair similarity 更明显下降，因此需要把 **convergence vs differentiation** 作为并行竞争解释，而不是先验押注 M2
- `M2_pair` 与 `M8_reverse_pair` 属于互补编码，偏相关时不应互相作为控制变量；共线性必须显式报告

交付物：
- `paper_outputs/tables_main/table_model_rdm.tsv`
- `paper_outputs/tables_main/table_delta_rho_lmm.tsv`
- `paper_outputs/figures_main/fig_model_rdm.png`
- 一句结论：哪个模型最能解释 post / Δρ 的变化，以及这更支持 convergence 还是 differentiation 路径

### Task 4.4 RD/GPS（互补证据，主文或 SI）

目的：
- 回答 “几何结构怎么变”（How）

交付物：
- `paper_outputs/figures_main/fig_rd_gps_complement.png`
- `paper_outputs/tables_si/table_rd_gps_stats.tsv`
- 一句结论：RD 与 GPS 是否同向支持 RSA（例如 RD 降低 + GPS 升高）

### Task 4.5 脑-行为关联（解释力与一致性）

要求：
- 预先指定 1 个行为指标（推荐：Run-7 yy 的回忆表现；或 yy-kj 的记忆差值）
- 预先指定 1 个神经指标（推荐：yy 的 `Δ similarity = post − pre`；当前数据下通常为负值，代表分化增强）
- 做相关/回归并做 FDR（若多指标）

交付物：
- `paper_outputs/figures_main/fig_brain_behavior.png`（或 SI）
- `paper_outputs/tables_si/table_brain_behavior.tsv`
- 一句结论：分化增强（更负的 Δ similarity）是否预测更好的学习/记忆表现（强调方向与不确定性）

---

## 5. 控制验证（推荐，PNAS 友好；NN/Neuron 也会问）

### Task 5.1 baseline 控制验证（最小版本）

目标：
- 回答 “变化不是纯重测效应”

最小实现（至少 SI 必做；在当前“yy 相似性下降”模式下强烈建议提升到主文控制位）：
- 在 pre/post 加入 baseline 的 patterns
- 计算 `Δbaseline` 并检验 `Δyy/Δkj > Δbaseline`
- baseline 语义必须写清楚：本项目中 baseline = `jx`，不是学习配对材料，只作为前后测的无学习控制条件

交付物：
- `paper_outputs/tables_si/table_baseline_control.tsv`（若放主文可再复制到 tables_main）
- `paper_outputs/figures_si/fig_baseline_control.png`（若放主文可再复制到 figures_main）
- 一段解释：baseline 用来排除“纯重测/噪声变化”导致的假分化，不取代 yy-kj 主结论

---

## 6. 进阶可扩展研究（在主流程收敛后择优推进）

重新 review 你的当前思路，并结合：
- `七篇创造力fMRI研究结构化摘要.md`
- `隐喻研究7篇文献结构化摘要.md`
- `metaphoric_pdf/` 下的方案文档

当前最值得推进的进阶研究，不是“平铺所有扩展分析”，而是按与主故事的贴合度分层：

### A档：强推荐（最能增强当前主文）

1. **熟悉度/时间梯度参数调制**
   - 理论文献：Cardillo et al. (2012), Lai et al. (2015)
   - 价值：把 pre/post 二分结果提升成“隐喻熟悉化/重组轨迹”，直接对接 Career of Metaphor

2. **偏侧化指数分析（LI）**
   - 理论文献：Mashal et al. (2007), Lai et al. (2015), Cardillo et al. (2018)
   - 价值：直接回应隐喻研究最经典的 LH/RH 争议；如果主结果主要落在左额颞系统，这一项尤其有价值

3. **学习阶段动态时间窗分析**
   - 理论文献：Beaty et al. (2017), Heinonen et al. (2016)
   - 价值：回答“分化增强是何时形成的”，把纵向设计优势真正用起来

### B档：结果导向扩展（当主结果足够强时考虑）

4. **中介分析**
   - 理论文献：Zhu et al. (2017)
   - 价值：从并列证据提升到路径假设（例如：RSA/Model-RSA → behavior）

5. **有效连接/方向性分析**
   - 理论文献：De Pisapia et al. (2016), Beaty et al. (2017)
   - 价值：如果你要往 NN/Neuron 冲刺，这一项最像“机制增强”

### C档：可做但不建议当前优先推进

6. **情绪效价调节**
   - 理论文献：Subramaniam et al. (2013)
   - 价值：理论上有趣，但与当前实验主设计耦合较弱；除非已有现成 item-level 效价评分，否则优先级较低

### Option A gPPI（调节连接）

目标：
- 回答 “学习阶段 yy vs kj 是否调制 seed→target 的连接”

交付物：
- `paper_outputs/tables_si/table_gppi.tsv`
- `paper_outputs/figures_si/fig_gppi.png`

### Option B 有效连接（方向性线索）

目标：
- 给出方向性线索（探索性）

交付物：
- `paper_outputs/tables_si/table_effective_connectivity.tsv`
- 解释声明：这是近似方法，不等价 DCM

### Option C 中介（路径假设）

目标：
- 把并列证据组织成“路径假设”（探索性）

交付物：
- `paper_outputs/tables_si/table_mediation.tsv`
- `paper_outputs/figures_si/fig_mediation_path.png`
- 解释声明：样本量限制，作为补充

---

## 7. 稳健性与透明度（顶刊必备）

### Task 7.1 Bootstrap CI 与 BF10（至少用于主结论）

要求：
- 为 primary endpoint 的关键效应提供 CI（若 LMM 不便，可对 subject-level Δ 做 bootstrap CI 作为补充）
- 对“无差异”或关键控制结论提供 BF10（在 SI）

交付物：
- `paper_outputs/tables_si/table_bootstrap_ci.tsv`
- `paper_outputs/tables_si/table_bayes_factors.tsv`

### Task 7.2 敏感性分析（最多 2–3 个）

推荐组合：
- RD 阈值敏感性（70–90%）
- trial 数等量化重采样
- CV 方案敏感性（若 MVPA 进入主文）

交付物：
- `paper_outputs/tables_si/table_sensitivity.tsv`
- `paper_outputs/figures_si/fig_sensitivity.png`

---

## 8. 图表与写作对齐（最后一步做收敛）

### Task 8.1 主文图 4–6 张“强约束”

建议主文图顺序：
- Fig1：范式 + 证据链示意
- Fig2：行为 + 学习阶段 GLM（Where）
- Fig3：Primary RSA + baseline 控制（What）
- Fig4：Model-RSA / Δρ LMM（Why）
- Fig5：脑-行为（So what）
- Fig6（可选）：RD/GPS 或激活-表征解离图

### Task 8.2 每张图 1 个主统计检验

要求：
- 在 `paper_outputs/tables_main/figure_stats_map.tsv` 里列出：
  - Figure
  - Primary test
  - Multiple comparison policy
  - Effect size/CI
  - Output file path

---

## 9. 推荐排期（两种版本）

### 2 周版本（PNAS 优先）

Week 1：
- Task 2.1–2.2（主检验与层级）
- Task 3.1–3.3（QC 封堵）
- Task 4.2（RSA + LMM primary）
- Task 4.3（Model-RSA + Δρ LMM）

Week 2：
- Task 4.1（GLM where）
- Task 4.4（RD/GPS，若主结果足够强）
- Task 4.5（脑-行为）
- Task 7.1（CI/BF10 最小版）
- Task 8.1–8.2（主图收敛）

### 4 周版本（NN/Neuron 冲刺）

在 2 周版基础上追加：
- Task 6（进阶扩展：优先 A档，择 1 条做强机制）
- Task 7.2（敏感性分析 2–3 项）
- 更严格的探索/确认边界声明（方法写作与 SI 组织）

---

## 10. 执行提示（避免顶刊最常见“翻车点”）

- 不要把所有分析都放主文：主文只服务 primary endpoint 的证据链，其余放 SI。
- 任何“显著”必须能复现：输出文件路径 + 运行命令 + 参数记录必须可追溯。
- 任何“无差异”要用 BF10 或 CI 说清楚：不是“没显著=没效应”。
- 任何 MVPA 放主文必须讲清防泄漏与置换：否则审稿人会直接否定可信度。
- 当前稿件不要再把“模型RDM比较”放到最后再决定做不做；既然主流程已经确定以机制解释为目标，Model-RSA 就应与 RSA 主效应一起写进主文主线。
