# paper_story_extension：论文故事拓展落地规划

更新时间：2026-05-13
对应主结果文档：`metaphoric/final_version/result_new_meta_roi.md`
对应收束目录：
- `metaphoric/final_version/nc_converge/`（§28 收束）
- `metaphoric/final_version/hpc_subfield_three_axis/`（§29 三轴）

本文档不是新主线，而是把 `paper_analysis/`（50 篇文献拆解）和 `result_new_meta_roi.md`
已落地结果（§0–§27）做交叉，列出**可在不动主线的前提下拓宽论文的角度**与对应落地动作。
所有拓展遵守 nc_converge / hpc_subfield_three_axis 的隔离约束：
- 不回写既有分析产物。
- 新产物只写到 `paper_outputs/qc/<extension_module>/`。
- 主结果文档只通过对应 append 脚本在末尾追加；不重写 §0–§27。

---

## 0. 设计原则

1. **不改主线**：当前 Stage 5 主线（learning condition geometry → post trained-edge differentiation → retrieval rebound）保持不变。
2. **不替换主机制**：post 阶段 trained-edge differentiation 仍是最强机制层。
3. **新增分析只能进入三类位置**：
   - Introduction / Discussion 的理论挂钩段（A 类，无新分析）。
   - Result 的补充图 / 鲁棒性段（B 类，少量新代码）。
   - Supplementary / Discussion 的探索性段（C 类，新代码、不进入主结论）。
4. **NC 投稿优先级**：A 类必做；B 类按时间窗口选做；C 类只做能进 Discussion 的最少必要项。
5. **ROI 仍以 18 个 meta ROI 为锚**，subfield 由 `hpc_subfield_three_axis` 承担。

---

## 1. 拓展角度总览

| 编号 | 角度 | 类型 | 优先级 | 主要文献来源 | 兑现位置 |
| --- | --- | --- | --- | --- | --- |
| A1 | Pattern separation / differentiation 文献链挂钩 | A | P0（必做） | 论文 39 / 40 / 44 / 26-28 / 36-37 | Intro + Discussion |
| A2 | Schema / vmPFC 重写 memory bridge boundary | A | P0（必做） | 论文 38 / 45 | §23/24 文字、Discussion |
| B1 | Global Pattern Similarity（GPS）补充 | B | P1（强推） | 论文 11 | Result 2 补充图 |
| B2 | HPC × lateral temporal coordinated reinstatement | B | P2（锦上） | 论文 43 / 5 / 50 | Result 3 / Discussion |
| B3 | Favila 式 post > pre 单尾 differentiation | B | P1（强推） | 论文 39 / 40 / 44 | Result 2 主图说明 |
| C1 | Insight / novelty moderation | C | P1（强推） | 论文 24 / 47 | Result 4 supp 或 Discussion |
| C2 | Predictive geometry / cognitive map 升维 | C | P3（埋伏笔） | 论文 46 / 48 | Discussion 埋伏笔 |
| C3 | Drift-diffusion 行为升级 | C | P1（强推） | 论文 49 | Result 1 supp |
| C4 | DMN × ECN × 创造力网络同现 | C | P3（埋伏笔） | 论文 1 / 2 / 14 / 15 / 16 / 32 | Discussion 一段 |
| D1 | Trained-edge drop 的 whole-brain searchlight | D | P2（锦上） | 论文 30 / 41 | Methods + Supp |
| D2 | Manifold geodesic / 曲率几何 | D | P3（埋伏笔） | 论文 5 / 50 / 30 | Discussion 一句 |
| D3 | Inter-subject representational alignment（IS-RSA） | D | P2（锦上） | 论文 5 / 30 / 41 | Result 5 supp |

P0 = NC 投稿前必做；P1 = 强推（可独立成段）；P2 = 锦上添花；P3 = 仅 Discussion 提出。

---

## 2. P0 必做拓展（不动主分析，只补理论挂钩）

### 2.1 A1：把主机制挂到 pattern separation / differentiation 文献链

**问题**：当前 §3/§4 主结果"YY trained edge post-pre similarity drop"在论文里只与
Step5C condition×time 与 edge specificity 自洽，没有显式接到 hippocampal
pattern separation / cortical assimilation 的国际叙事。

**落地动作**：
1. 在 Introduction 增加一段 anchor，主轴为：
   - 海马 pattern separation（论文 26 / 27 / 28 / 36 / 37）。
   - Experience-dependent differentiation（论文 39）。
   - HPC separation × cortex assimilation 双系统（论文 40 / 44）。
2. 在 Discussion 用相同三组文献回扣：
   - 把 R hippocampus / R PPA-PHG 的 trained-edge drop 写成 differentiation。
   - 把 left semantic（temporal pole / IFG / AG）写成 cortical assimilation。
   - 把"YY 已有源概念 + 新目标"写成 prior-knowledge integration scenario。
3. 引用清单写进 `nc_converge/append_section28.py` 的 §28 模板，避免后期失同步。

**验收**：在 §28 预览中能看到 differentiation/assimilation 一对术语；在 §29 中 head/body/tail
结果与 DG/CA3/CA1 分离动机叙事接上。

**成本**：仅引用与 200–400 字 Discussion 段落，不需新分析。

### 2.2 A2：用 schema/vmPFC 重写 memory bridge 的 boundary 表述

**问题**：§23/§24 现有结论是"remembered YY items 有更大 rebinding difference score，
但主要由 lower post similarity 驱动，不是 retrieval similarity 更高"。当前文字写法
偏"边界条件 + 不强主张"，但其实可以**积极地**接到 schema 文献。

**落地动作**：
1. 在 §23/§24 的解读段补一句：该模式与 schema-consistent learning 预测一致——一旦
   schema 启用，item-level retrieval similarity 不必更高（论文 38 / 45）。
2. 把"endpoint boundary evidence"重命名为"schema-consistent endpoint pattern"。
3. 在 Discussion 写：post-stage separation as endpoint marker of successful
   schema integration；而不是"weak retrieval reinstatement"。
4. 不修改 §0 / §14 / §25 的主线；仅把语气从 boundary-only 升级为 boundary-with-theory。

**验收**：§23/§24 阅读体验从"reviewer 会问'那你 retrieval 怎么解释'"变成"我们已经把它放进 schema 框架"。

**成本**：纯文字。

---

## 3. P1 强推拓展（小代码，可独立成段）

### 3.1 B1：Global Pattern Similarity（GPS）补充

**目标**：回答 reviewer 必问的"YY 效应是 pair-specific 还是 category-level"。

**指标**：
```text
GPS(item) = mean over j ≠ i of corr(pattern_i, pattern_j)
GPS_within_condition / GPS_across_condition
```

**落地动作**：
- 新增模块：`metaphoric/final_version/extension_gps/`（参考 nc_converge 命名风格）。
- 复用 nc_converge 的 `shared_nc.py`（IO、FDR、condition_item_id、network mapping）。
- 输入：现有 item-level beta（与 Step5C 同源），不重抽。
- 输出：
  - `paper_outputs/qc/extension_gps/gps_long.tsv`
  - `paper_outputs/qc/extension_gps/gps_condition_time_model.tsv`
- 主分析：拟合 `GPS ~ C(condition) * C(time) + (1|subject)`，BH-FDR 与 Step5C 同 family 粒度。
- 与 §4 edge specificity 并列：GPS 显著 + edge specificity 显著 = 双重证据。

**验收**：
- 若 GPS 与 edge specificity 同向，Result 2 加一张 supp 图。
- 若 GPS 显著但 edge specificity 不显著的 ROI，列入 boundary。

**成本**：约 200 行代码 + 一张图。

### 3.2 B3：Favila 式 post > pre 单尾 differentiation

**目标**：把当前"YY trained drop > pseudo / KJ / non-edge"的对比写法升级到
Favila / Molitor 文献最直接的"post > pre 距离扩大"主张。

**落地动作**：
- 复用现有 `pair_similarity` 长表（来自 Step5C / S3）。
- 检验：对每个 (ROI, YY trained pair)，计算 `pre_distance = 1 - r_pre`、`post_distance = 1 - r_post`，
  做 paired one-sided test `H1: post > pre`。
- 输出：
  - `paper_outputs/qc/extension_favila_repulsion/repulsion_test.tsv`
  - 与 §4 edge specificity 同 ROI 列表对齐。
- BH-FDR 在 (network, subROI) 内做。

**验收**：在 R hippocampus / R PPA-PHG / R PPC-SPL / R temporal pole 中，
trained pair 的 post > pre 单尾显著（与 §3 Step5C 同向）。

**成本**：约 100 行代码；可挂在 nc_converge `b2` 后面或独立模块。

### 3.3 C1：Insight / novelty moderation

**目标**：让 YY vs KJ 差异不只是材料类别差，而是 insight/novelty 强度调节。

**落地动作**：
- 已有占位：`nc_converge/c3_novelty_familiarity_moderation.py`，目前**不在默认 run_all 中**。
- 把它纳入 `run_all_nc_converge.py`（在 D 之后、A6 之前）。
- 输入：item-level YY novelty / insight 评分（若 metadata 不完整，先补 manifest）。
- 模型：`trained_edge_drop ~ novelty_z * C(condition) + (1|subject) + (1|condition_item_id)`。
- 输出：
  - `paper_outputs/qc/nc_converge/c3_novelty_familiarity_moderation/novelty_moderation.tsv`
- 与 §28 storyline reframing 联动：若 novelty × YY interaction 显著，
  在 Discussion 写"YY 优势部分由 insight-triggered cortical representational change 驱动"
  （论文 24 / 47 锚点）。

**验收**：moderation 显著或不显著都可写——显著时进入 Result，不显著时作为 boundary。

**成本**：c3 已实现；只需补 manifest 与编排顺序。

### 3.4 C3：Drift-diffusion 行为分析

**目标**：把 §2 的 RT 从"补充指标"升级为"机制可解释指标"。

**落地动作**：
- 新增 `metaphoric/final_version/extension_ddm/`，使用 `hddm` 或 `pyddm`。
- 数据：现有 trial-level RT + accuracy（与 §2 同源）。
- 模型：分别拟合 YY / KJ 的 drift rate v、boundary a、non-decision t；group-level 比较。
- 输出：
  - `paper_outputs/qc/extension_ddm/ddm_group_estimates.tsv`
  - `paper_outputs/qc/extension_ddm/ddm_yy_vs_kj.tsv`
- 接到主线：若 YY drift rate 更高，写"语义检索更有效"，与 left semantic assimilation 一致；
  若 boundary 更小，写"更宽松的决策边界"，作为 boundary。

**验收**：DDM 至少给出一个稳定的 YY vs KJ 参数差异，可写进 §2 supp。

**成本**：建模约 200 行；hddm 拟合时间是主要瓶颈。

---

## 4. P2 锦上添花拓展（中等代码）

### 4.1 B2：HPC × lateral temporal coordinated reinstatement

**目标**：把 §3（R hippocampus）+ §10（R temporal pole MVPA）+ §9（retrieval rebound）
在回路层面绑成一条"hippocampo-lateral-temporal coordination"链。

**落地动作**：
- 复用现有 item-level beta；ROI 选 R hippocampus、R temporal pole（必要时加 hpc subfield head/body/tail）。
- 指标：
  - 阶段内的 trial-by-trial pair-similarity correlation（HPC vs temporal pole）。
  - 阶段间（learning → post → retrieval）coupling shift。
- 输出：
  - `paper_outputs/qc/extension_repr_coupling_hpc_tp/coupling_long.tsv`
  - `paper_outputs/qc/extension_repr_coupling_hpc_tp/coupling_stage_test.tsv`
- 与 nc_converge `b3_repr_coupling` 区分：b3 是 network composite × network composite；
  本扩展是 ROI × ROI 的细粒度。

**验收**：YY 在 retrieval 阶段 HPC × temporal pole coupling 高于 KJ，或高于 pre 阶段。

**成本**：约 250 行代码 + 一张图。

### 4.2 D1：Trained-edge drop whole-brain searchlight

**目标**：回应"为什么只看 18 个 meta ROI"的 reviewer 标准问题。

**落地动作**：
- 新增 `metaphoric/final_version/extension_searchlight_drop/`。
- 在 group level 跑 trained-edge drop searchlight（半径建议 8mm，z-scored）。
- 输出：
  - `paper_outputs/qc/extension_searchlight_drop/searchlight_zmap.nii.gz`
  - cluster table：`searchlight_clusters.tsv`
- 控制：cluster-level FDR；输出与 18 个 meta ROI 的 overlap（百分比）。

**验收**：searchlight 主峰落在 meta ROI 之内或邻近 → 支持当前 ROI 选择。

**成本**：约 300 行 + 计算时间。

### 4.3 D3：IS-RSA（Inter-Subject Representational Alignment）

**目标**：检验"哪些被试学得更像同一张图"，并把它接到 memory accuracy。

**落地动作**：
- 计算 subject-level Δ(post-pre) edge similarity 矩阵；
  在被试间做 RSA，再把 alignment score 与 memory accuracy 相关。
- 输出：
  - `paper_outputs/qc/extension_isrsa/isrsa_alignment.tsv`
  - `paper_outputs/qc/extension_isrsa/isrsa_behavior_link.tsv`

**验收**：alignment score 与 YY memory accuracy 显著相关 → Result 5 supp。

**成本**：约 200 行。

---

## 5. P3 概念升维（仅 Discussion 埋伏笔，不在本轮做分析）

### 5.1 C2：Predictive geometry / cognitive map
- 论文 46 / 48 的 successor representation / cognitive map 框架。
- Discussion 埋伏笔："YY trained edges may form a learned cognitive map of relational
  mappings; future work could test whether post-stage geometry is predictive-coded."
- 不做分析。

### 5.2 C4：DMN × ECN × 创造力网络对话
- 论文 1 / 2 / 14 / 15 / 16 / 32 的 DMN-ECN coupling 叙事。
- Discussion 一段："our IFG / temporal pole / AG findings align with creativity-network
  dialogue between DMN and ECN during novel relational learning."
- 不做分析。

### 5.3 D2：Manifold geodesic / 曲率几何
- 论文 5 / 50 / 30 的 geodesic vs euclidean。
- Discussion 一句："V1 MDS visualisation suggests post-stage manifold reorganisation
  beyond pairwise distance; future work could quantify geodesic curvature."
- 不做分析。

---

## 6. 与现有 §28 / §29 的衔接

### 6.1 §28（nc_converge/append_section28.py）
- A1 / A2 在 §28 模板中体现为：
  - 主线段加入 differentiation × assimilation 一对术语。
  - Memory bridge 段重写为 schema-consistent endpoint。
- B1 / B3 / C1 / C3 若执行完成，§28 增加 1 段 "Robustness and Specificity Extensions"，
  分别引用 GPS / Favila one-sided / novelty moderation / DDM。
- 不修改 nc_converge 的 A/B/D 主流程脚本；新代码全部进入新模块或现有 c3。

### 6.2 §29（hpc_subfield_three_axis/append_section29.py）
- A1 在 §29 中绑到 head/body/tail × DG/CA3/CA1 的机制描述。
- B2 / B3 若执行，§29 加一句"hippocampo-lateral-temporal coordination is most
  pronounced in head/body subfields during retrieval"。
- D1 searchlight 输出与 N1 evidence table 并列：用 searchlight 验证 hpc subfield
  effect 不是 ROI 选择产物。

---

## 7. 时间窗口建议（不预测具体天数，只列依赖关系）

依赖最少的先做（可并行）：
- A1 文献清单 + Intro/Discussion 段落草稿。
- A2 §23/§24 重写。
- B3 Favila one-sided（依赖 pair_similarity 长表，已就绪）。
- C1 novelty moderation（c3 已实现，只缺 manifest 接入）。

依赖 nc_converge 主流程跑完后再做：
- B1 GPS（与 §3/§4 主表对齐 family）。
- C3 DDM（与 §2 行为表对齐）。

依赖 hpc_subfield 跑完后再做：
- B2 HPC × temporal pole coupling（subfield 列接入）。
- D1 searchlight（验证 ROI 选择）。
- D3 IS-RSA。

仅文字（Discussion）：
- C2 / C4 / D2 埋伏笔，等其他全部完成后一次性补。

---

## 8. 立项决策表

| 拓展 | 是否立项 | 备注 |
| --- | --- | --- |
| A1 | 默认立项 | NC 投稿前必须挂理论 |
| A2 | 默认立项 | 重写文字即可 |
| B1 GPS | 待用户确认 | 是否纳入 P1 主推 |
| B3 Favila one-sided | 待用户确认 | 强推；代码量小 |
| C1 novelty moderation | 待用户确认 | c3 已存在，只差启用 |
| C3 DDM | 待用户确认 | 依赖 hddm/pyddm 环境 |
| B2 HPC × temporal pole | 待用户确认 | P2 |
| D1 searchlight | 待用户确认 | 计算成本最高 |
| D3 IS-RSA | 待用户确认 | P2 |
| C2 / C4 / D2 | 默认进 Discussion | 不做分析 |

---

## 9. 下一步动作建议

1. 用户先就第 8 节立项决策表勾选 P1 中要做的拓展。
2. 勾选后，对 A1 / A2 立刻在 §28 / §29 模板里埋好引用与术语，避免后续重写。
3. P1 中被勾选的拓展，每个对应一个新模块目录（命名前缀 `extension_*`），
   独立编排器、独立 FDR family、独立 isolation_check。
4. 主结果文档 `result_new_meta_roi.md` 仍只通过 §28 / §29 追加；
   本拓展规划本身不写入主文档，但被各扩展模块的 README 引用。
