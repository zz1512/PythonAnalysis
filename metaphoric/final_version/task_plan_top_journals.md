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
- 主检验：`condition × time` 交互效应（隐喻 yy 的 pre→post 变化是否显著大于 kj）。

交付物：
- 一段“主检验定义”（写进方法或预注册式分析计划页）
- 一句论文式结论模板（示例）：
  - “在 ROI X 中，LMM 显示 `condition×time` 交互显著（b=..., SE=..., t=..., p=...），表明 yy 的表征相似性变化大于 kj。”

### Task 2.2 明确分析层级（Confirmatory/Secondary/Exploratory）

定义层级（建议）：
- Confirmatory（主文主结论）：RSA + LMM（primary endpoint）
- Secondary（主文或 SI）：RD/GPS 的同向互补验证；GLM 的定位证据；脑-行为关联（预先指定 1 个行为指标）
- Exploratory（SI）：MVPA（ROI 或 searchlight）、gPPI、有效连接、学习动态、中介、模型RDM扩展等

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

### Task 4.2 Primary Endpoint：前后测 RSA + LMM（主文核心）

目的：
- 回答 “表征关系怎么变”（What）

交付物（来自 `rsa_analysis/` 输出 + 你汇总到 paper_outputs）：
- `paper_outputs/tables_main/table_rsa_lmm_primary.tsv`
- `paper_outputs/figures_main/fig_rsa_primary_interaction.png`
- 一句结论：交互效应（方向、统计量、CI）

### Task 4.3 RD/GPS（互补证据，主文或 SI）

目的：
- 回答 “几何结构怎么变”（How）

交付物：
- `paper_outputs/figures_main/fig_rd_gps_complement.png`
- `paper_outputs/tables_si/table_rd_gps_stats.tsv`
- 一句结论：RD 与 GPS 是否同向支持 RSA（例如 RD 降低 + GPS 升高）

### Task 4.4 脑-行为关联（解释力与一致性）

要求：
- 预先指定 1 个行为指标（例如 run7 accuracy 或 RT）
- 预先指定 1 个神经指标（例如 RSA 的 subject-level Δ 或 LMM 随机斜率提取）
- 做相关/回归并做 FDR（若多指标）

交付物：
- `paper_outputs/figures_main/fig_brain_behavior.png`（或 SI）
- `paper_outputs/tables_si/table_brain_behavior.tsv`
- 一句结论：神经变化是否预测行为增益（强调方向与不确定性）

---

## 5. 控制验证（推荐，PNAS 友好；NN/Neuron 也会问）

### Task 5.1 baseline 控制验证（最小版本）

目标：
- 回答 “变化不是纯重测效应”

最小实现（SI 即可）：
- 在 pre/post 加入 baseline 的 patterns
- 计算 `Δbaseline` 并检验 `Δyy/Δkj > Δbaseline`

交付物：
- `paper_outputs/tables_si/table_baseline_control.tsv`
- `paper_outputs/figures_si/fig_baseline_control.png`
- 一段解释：baseline 是控制验证，不取代 yy-kj 主结论

---

## 6. 机制扩展（NN/Neuron 冲刺：只选 1 条做主线）

从下面 4 条里选 1 条放主文，其余放 SI 或不做：
- 模型RDM比较（推荐优先）
- gPPI（连接调节）
- 有效连接（方向性近似）
- 中介分析（路径假设）

### Option A 模型RDM（推荐）

目标：
- 回答 “为什么隐喻会更重组”（理论解释）

交付物：
- `paper_outputs/tables_main/table_model_rdm.tsv`（或 SI）
- `paper_outputs/figures_main/fig_model_rdm.png`
- 一句结论：哪个模型最能解释神经RDM（并说明对理论的意义）

### Option B gPPI（调节连接）

目标：
- 回答 “学习阶段 yy vs kj 是否调制 seed→target 的连接”

交付物：
- `paper_outputs/tables_si/table_gppi.tsv`
- `paper_outputs/figures_si/fig_gppi.png`

### Option C 有效连接（方向性线索）

目标：
- 给出方向性线索（探索性）

交付物：
- `paper_outputs/tables_si/table_effective_connectivity.tsv`
- 解释声明：这是近似方法，不等价 DCM

### Option D 中介（路径假设）

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
- Fig3：Primary RSA（What）
- Fig4：RD/GPS（How）
- Fig5：脑-行为（So what）
- Fig6（可选）：机制线（Why）

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

Week 2：
- Task 4.1（GLM where）
- Task 4.3（RD/GPS）
- Task 4.4（脑-行为）
- Task 7.1（CI/BF10 最小版）
- Task 8.1–8.2（主图收敛）

### 4 周版本（NN/Neuron 冲刺）

在 2 周版基础上追加：
- Task 6（机制扩展：选 1 条做主线）
- Task 7.2（敏感性分析 2–3 项）
- 更严格的探索/确认边界声明（方法写作与 SI 组织）

---

## 10. 执行提示（避免顶刊最常见“翻车点”）

- 不要把所有分析都放主文：主文只服务 primary endpoint 的证据链，其余放 SI。
- 任何“显著”必须能复现：输出文件路径 + 运行命令 + 参数记录必须可追溯。
- 任何“无差异”要用 BF10 或 CI 说清楚：不是“没显著=没效应”。
- 任何 MVPA 放主文必须讲清防泄漏与置换：否则审稿人会直接否定可信度。

