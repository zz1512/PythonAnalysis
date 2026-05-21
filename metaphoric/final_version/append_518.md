# append_518.md — 2026-05-18 NC converge 增量分析说明

本文件记录 2026-05-18 在 `metaphoric/final_version/nc_converge/` 中新增的脚本，以及它们要回答的具体问题。本批新增不修改任何已有脚本和已有结果输出，仅在 sandbox `paper_outputs/qc/nc_converge/` 下产出新的 TSV。

---

## 1. 背景与起因

老师/审稿追问：

> "post-stage separation 更强的 YY items 往往具有更高的原始 pre-pair similarity，但 post separation 不是由 novelty 或单一材料维度决定。这能不能引申为：YY 学习对学习前更相似的句对有更强的精细化分离作用？"

已有结果给出的边界：

- C1（[c1_input_output_transformation.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/c1_input_output_transformation.py)）报告 `pre_pair_similarity_z` 主效应 +0.69 ~ +0.73（q < 1e-89），说明 pre→post 强正向；但 `pre × condition` 交互不显著（hpc-spatial q≈0.751，semantic q≈0.450）。
- §28 R2 high/low profile（[result_new_meta_roi.md L2858](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md#L2858)）：YY 内 high vs low post-separation items 的 pre_pair_similarity 差异极大（hpc-spatial low vs high estimate = -0.6269, p = 6.69e-06；semantic estimate = -0.6917, p = 7.38e-07）。

两条证据方向一致但还不能直接写成"YY 对 pre-相似句对有更强分化"，因为 C1 的 `pre × condition` slope 没过 FDR。需要一个 **confirmatory tertile design**：在每个 network × condition 内按 pre_pair_similarity 分位划分 low/mid/high，再看 YY 在 high tertile 是否相对 low tertile 出现 condition-specific 的额外分化（i.e. `condition[T.yy] × pre_tertile[T.high]`）。

`pair_similarity` 来源已多次确认是 **脑表征相似度**（Fisher-z(Pearson(pattern_word1, pattern_word2))，源自 [build_learning_post_memory_item_table.py L214-L232](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/build_learning_post_memory_item_table.py#L214-L232)），不是 BERT / 词向量 / 文本相似度。本批新增分析仍然只用脑表征相似度。

---

## 2. 新增脚本

### `c1b_pre_similarity_tertile.py` — pre-similarity tertile moderation

[c1b_pre_similarity_tertile.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/c1b_pre_similarity_tertile.py)

**问题**

YY 在"学习前 ROI pattern 已经更相似"的句对上，是否产生比 KJ 更强的 post-stage representational separation？

**输入**

- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`（与 C1/C3 同一上游表）

**预处理（`prepare`）**

1. ROI 层 raw 列 `pre_pair_similarity` / `post_pair_similarity` 与 `post_edge_specificity`（缺失时 fallback 到 `trained_edge_drop`）按 `subject × network × condition_item_id` 聚合到 raw 网络均值；
2. 三个 raw outcome 各自做 **单次全局 z**（沿用 §28.4 修复后单位策略，避免 z-of-z 镜像）；
3. 在每个 `network × condition` 内按 `pre_pair_similarity_network_raw` 划分 tertile：默认 quantile 30 / 70 切，写入 `pre_tertile ∈ {low, mid, high}`。
   - 关键：tertile 在 condition 内独立划分。这样 KJ 的 high tertile 与 YY 的 high tertile 对应的是**各自分布上**的 high；如果改成跨 condition 全局划分，YY 整体相似度与 KJ 不同会让 high tertile 偏向某一 condition，从而把 condition main effect 混进 tertile 维度。

**模型（`fit_models`）**

```text
post_edge_differentiation_z ~ C(condition, Treatment('kj'))
                              * C(pre_tertile, Treatment('low'))
                              + sentence_char_len_z + word_frequency_mean_z
                              + stroke_count_mean_z + valence_mean_z
                              + arousal_mean_z
                              + (1 | subject) + (1 | condition_item_id)
```

第二个 outcome 同样 design，把左边换成 `post_pair_similarity_z`，作为对 C1 input-output slope 的离散版交叉验证。两个 model 走 `shared_nc.fit_formula`（即 reviewer_supp 的 mixed-effects + OLS-HC3 fallback 链）。

关键回归项：

| 关键项 | 解释 |
| --- | --- |
| `condition[T.yy]` | KJ low vs YY low 的差异（确认 condition main effect 在 low pre tertile 是否仍存在） |
| `pre_tertile[T.high]` | KJ low vs KJ high 的差异（pre 越相似 → post 是否仍更相似） |
| `condition[T.yy]:pre_tertile[T.high]` | YY 在 high tertile 的 **额外** separation（核心 confirmatory term） |
| `condition[T.yy]:pre_tertile[T.mid]` | mid tertile sanity check |

**焦点对照（`focal_contrasts`）**

不依赖回归框架，直接做 cell mean 差：

- `yy_high_minus_low` = YY high tertile mean − YY low tertile mean
- `kj_high_minus_low` = KJ high tertile mean − KJ low tertile mean
- `did_yy_minus_kj_high_low` = (YY high − YY low) − (KJ high − KJ low)

每个 contrast 用 Welch SE，输出 `estimate / se / stat / p`，整张表跑 BH-FDR。

**输出（写入 `paper_outputs/qc/nc_converge/c1b_pre_similarity_tertile/`）**

| 文件 | 内容 |
| --- | --- |
| `pre_tertile_descriptives.tsv` | network × condition × tertile 的 raw / z 均值与样本数 |
| `pre_tertile_model_coefficients.tsv` | 两条 model 的全部系数 + 状态 + BH-FDR |
| `focal_contrasts.tsv` | YY/KJ high-low + DiD 三组对照 |
| `pre_tertile_manifest.tsv` | 输入路径、行数、tertile 分位参数 |

**Go/No-Go 判读**

执行后建议这样读结果（分两层互斥结论）：

1. **支持升级为主机制**：在两个 network 中都看到
   - `condition[T.yy]:pre_tertile[T.high]` 估计为负且 q < 0.05（YY 在 high pre-similarity 中分化更强），并且
   - `did_yy_minus_kj_high_low` 估计为负且 q < 0.05；
   则可以在文中写"YY learning preferentially differentiates pairs that started from higher pre-learning representational similarity"。
2. **维持 boundary 写法**：如果 DiD 不显著，但 `pre_tertile[T.high]` 主效应仍稳健，则保持原结论"YY 的 high-separation items 倾向落在 high pre-similarity tail，但 condition-specific differentiation slope 与 KJ 不可区分"。

**安全策略**

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/c1b_pre_similarity_tertile/`，不写其他目录；
- `safe_output_path` 拒绝覆盖已有文件；
- 不修改 C1/C2/C3 任何代码或输出；
- 入口仍走 `shared_nc` 的 `add_common_args / default_config`，与 nc_converge 其他脚本一致。

---

## 3. 与既有脚本的关系

| 脚本 | 关系 |
| --- | --- |
| `c1_input_output_transformation.py` | C1b 是 C1 的 confirmatory 离散版本：把连续 `pre × condition` slope 替换为 tertile × condition interaction，避免 slope 局部线性假设并直接给出 high vs low cell-mean DiD。 |
| `c2_global_similarity_centrality.py` | 不耦合。C1b 仍以 `pre_pair_similarity` 为 moderator，C2 以 `global_similarity_change` 为 moderator。 |
| `c3_novelty_familiarity_moderation.py` | C1b 与 C3 互补。C3 检验 novelty 等材料维度是否解释 post separation；C1b 检验 pre-stage 脑表征相似度是否解释 YY-specific post separation。 |
| `a4_post_memory_component.py` | 沿用同一单位策略（raw → 单次全局 z），不影响 A4 输出。 |
| `a6_fdr_tiering.py` | C1b 输出可后续纳入 A6 family（建议挂在 Tier B `module=c1b::network=...`），但当前批不修改 a6。 |

---

## 4. 运行示例（数据机上执行）

```bash
cd metaphoric/final_version/nc_converge
python c1b_pre_similarity_tertile.py \
    --base-dir E:/python_metaphor
```

输出会落到 `E:/python_metaphor/paper_outputs/qc/nc_converge/c1b_pre_similarity_tertile/`。如果要测试不同 tertile 切分，可加 `--low-quantile 0.25 --high-quantile 0.75`。

---

## 5. 文档建议（拿到结果后）

- 在 §28.7 末尾追加一段 confirmatory 结果，按上面 Go/No-Go 模板写。
- 如 DiD 显著，在 result discussion 中明确把"YY 对 pre-相似句对的额外 differentiation"作为 supplementary mechanism；如果 DiD 不显著，则维持现有 boundary 写法，但保留 R2 profile 的描述性证据。
- 不建议把 C1b 升为主机制：它仍然是 supplementary moderation，主结论仍是 YY post-stage separation 的 condition main effect。

---

## 6. `d1_learning_understand_like_bridge.py` — 学习阶段 understand / like 行为 → 神经/记忆 桥模型

[d1_learning_understand_like_bridge.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d1_learning_understand_like_bridge.py)

### 6.1 起因

§32.4 重新接入 run3/run4 后，R2 profile 已经检验了 RT-derived `learning_fluency_shift` 与 high vs low post-separation items 的关系，但 run3/run4 行为并不只有 RT —— [build_learning_behavior_table.py L98-L104](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/build_learning_behavior_table.py#L98-L104) 同时记录了：

- `run3_understand_yes`：sentence3_RESP=3 ⇒ 1，否则 0（被试是否报告"理解"）；
- `run4_like_yes`：sentence4_RESP=3 ⇒ 1，否则 0（是否报告"喜好"）。

这两列在主管线里目前**只出现在** [learning_behavior_models.py L67-L68](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/learning_behavior_models.py#L67-L68) 的 `~ condition` GEE 主效应模型里，回答的是"YY 是否更容易被报告 understand / like"。它们**没有**进入：

- post-stage representational separation（post_edge_specificity / post_pair_similarity）的桥模型；
- retrieval memory（remembered_strict）的桥模型；
- §32 R2 profile / fluency mediator 路线；
- A4/A5/A6 / C1/C2/C3 任一 NC 收敛模型。

因此"理解 / 喜好行为是否与 post-separation / 记忆稳定相关、是否具备 condition-specific moderation"目前是**完全空白**，且 reviewer 几乎一定会问。本批补一个独立桥脚本，数据 schema 已齐（覆盖率 98.57% / 97.86%，[result_new_meta_roi.md](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md) §32 已有交代）。

### 6.2 输入

- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`（与 C1/C2/C3/C1b 同源；同时含 understand/like、RT、fluency、post_edge_specificity、pre/post pair_similarity、memory/remembered_strict、材料 covariates）。

### 6.3 预处理（`prepare`）

1. ROI 层 raw 列 `pre_pair_similarity` / `post_pair_similarity` / `post_edge_specificity`（缺失时 fallback `trained_edge_drop`）按 `subject × network × condition_item_id` 聚合到 raw 网络均值；
2. 对三个 raw outcome 各自做**单次全局 z**（与 §28.4 修复后口径一致）；
3. RT 列直接用上游已经 within `subject × condition` z 过的 `run3_rt_z_subject_condition` / `run4_rt_z_subject_condition`（[reactivation_to_post_edge.py L93-L102](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/reactivation_to_post_edge.py#L93-L102) 同列名），**不再二次 z**；
4. `learning_fluency_shift` 单次全局 z 写入 `learning_fluency_shift_z`，作为 RT-side 控制；
5. understand/like 二元 0/1 直接进；item-level 聚合到 network 后可能出现 0/0.5/1 三档，`learning_response_profile` 标签按 ≥0.5 阈值映射为 `{understand,no_understand} × {like,no_like}` 共 4 cell；
6. memory outcome 与 [stage4_subsequent_memory_rework.py L138-L144](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/stagewise_mechanism/stage4_subsequent_memory_rework.py#L138-L144) 对齐：优先用 `remembered_strict`，缺失时由 `memory_score` 等于 1.0 / 0.0 推导。

### 6.4 模型

四组互补 model（每个 network 独立）：

**A. understand / like → post-stage separation**

```text
post_edge_differentiation_z ~
    C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)
  + pre_pair_similarity_z + RT covariates + material covariates
```

confirmatory term：`C(condition)[T.yy]:run3_understand_yes` / `...:run4_like_yes`。回答"YY 中被报告 understand / like 的句对，是否在 post 阶段表现出额外的 representational separation"。

**B. understand / like → post pair similarity（input-output 交叉验证）**

```text
post_pair_similarity_z ~ pre_pair_similarity_z
  + C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)
  + RT covariates + material covariates
```

控制 pre 之后看 understand/like 是否对 post 表征几何还有 condition-specific 残差贡献。该模型同时是 §28.7 (C1) 框架的扩展——把 understand/like 当作离散 moderator 加入 input-output transformation。

**C. understand / like → memory（retrieval 行为终点，binomial GLMM），两版**

- C-direct：

```text
memory_strict ~ C(condition, Treatment('kj'))
              * (run3_understand_yes + run4_like_yes)
              + pre_pair_similarity_z + RT covariates + material covariates
```

- C-with-neural：在 C-direct 基础上加入 `post_edge_differentiation_z` 作为 mediator。

通过比较两版中 `run3_understand_yes` / `run4_like_yes` 系数的衰减幅度，读取"理解/喜好对 retrieval memory 的关系是否经过 post-stage separation 的中介"。这是一个条件中介读法（不做正式 indirect effect 计算，避免引入 bootstrap 计算量），但在 NC 层面足以支持讨论。

**D. profile descriptive（`learning_response_profile_descriptives.tsv`）**

按 `network × condition × learning_response_profile` 4 cell 报告：

- post_edge_differentiation 的 raw mean / sd / z mean
- post_pair_similarity raw mean
- memory_prop（remembered_strict 的均值）
- learning_fluency_shift 均值
- n / n_subjects / n_items

这是 NC supplementary table 友好的描述性截面，能在 reviewer 追问"4 个组合是否有 outlier cell"时直接给出。

### 6.5 输出（沙箱 `paper_outputs/qc/nc_converge/d1_learning_understand_like_bridge/`）

| 文件 | 内容 |
| --- | --- |
| `understand_like_to_post_separation.tsv` | 模型 A 的全部系数 + 状态 + BH-FDR |
| `understand_like_to_post_pair_similarity.tsv` | 模型 B 的全部系数 + 状态 + BH-FDR |
| `understand_like_to_memory_direct.tsv` | 模型 C-direct 的全部系数（binomial logit）+ BH-FDR |
| `understand_like_to_memory_with_neural.tsv` | 模型 C-with-neural 的全部系数 + BH-FDR |
| `learning_response_profile_descriptives.tsv` | 4-cell 描述性表 |
| `manifest.tsv` | 输入路径、行数、各关键列 nonnull 计数（含 understand_yes / like_yes / memory_strict 覆盖率） |

每张 model 表内部 BH-FDR family 自然按 `network` 分（与 a6 family 粒度一致）。`q_for_ok_rows` 自动跳过 `singular / not_converged / nonfinite_se / single_class_outcome / too_few_rows` 行。

### 6.6 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d1_learning_understand_like_bridge/`，不写其他目录；
- `safe_output_path` 拒绝覆盖已有文件；
- 不修改 [learning_behavior_models.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/learning_behavior_models.py)、[reactivation_to_post_edge.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/reactivation_to_post_edge.py)、[stage4_subsequent_memory_rework.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/stagewise_mechanism/stage4_subsequent_memory_rework.py) 等任一上游脚本与既有输出；
- 入口与 c1b 一致，仍走 `shared_nc` 的 `add_common_args / default_config`。

### 6.7 与既有脚本的关系

| 脚本 | 关系 |
| --- | --- |
| `learning_behavior_models.py` | D1 不重复 GEE `~ condition` 的 main effect（已有）；只把 understand/like 提升为 moderator / mediator predictor 进入神经与 memory 模型。 |
| `c1_input_output_transformation.py` | D1 模型 B 是 C1 的扩展：在 input-output transformation 中加入 understand/like × condition；两者 outcome 单位一致（pre→单次全局 z）。 |
| `c1b_pre_similarity_tertile.py` | 互补：C1b 用 pre similarity tertile 做 moderator，D1 用 understand/like 做 moderator。两者可在结果讨论中合并写为 "geometric (pre-similarity) 与 metacognitive (understand/like) 两条 moderator 路径"。 |
| `stage4_subsequent_memory_rework.py` | memory outcome 命名与口径完全对齐（`remembered_strict`），保证 D1 的 memory_strict 与 stage4 的 binomial 模型可直接对照。 |
| `a6_fdr_tiering.py` | D1 输出可后续挂入 `module=d1::network=...` 的 family；本批不修改 a6。 |

### 6.8 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d1_learning_understand_like_bridge.py \
    --base-dir E:/python_metaphor
```

如需指定其它输入：`--input /path/to/item_mechanism_table.tsv`。

### 6.9 文档建议（拿到结果后）

- **A 显著**（understand_yes 或 like_yes × YY 对 post_edge_differentiation_z 显著）：
  - 在 §32 / §28 间加一节 "Metacognitive moderators of YY post-stage separation"，把 understand/like 与 RT-fluency 一起作为 learning-stage 行为 moderator 报告。
- **C-direct 显著但 C-with-neural 衰减**（understand/like 系数在 C-with-neural 中变小或失显）：
  - 写为"post-stage representational separation 部分中介 understand/like → retrieval memory"；
  - 可在 discussion 提及 metacognitive monitoring → 海马 representational geometry → memory 的三段链。
- **A 不显著但 C-direct 显著**：
  - 写为"like_yes 直接预测 retrieval 成功，但不经过 post-stage 几何"，对应 affect-to-memory 独立通道（与 reward/curiosity 文献链接）。
- **A 与 C 都不显著**：
  - 加强 §32.4 的边界结论 "neither RT-based fluency nor categorical understand/like predicts post-separation or memory at item level"，作为 reviewer 关心的 negative evidence。

不建议在确认结果之前改任何主表叙事；本脚本仅产出 supplementary evidence，不动主结论。

---

## 7. `d2_voxel_trajectory_alignment.py` — voxel-level pre-post-retrieval 重建方向

[d2_voxel_trajectory_alignment.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d2_voxel_trajectory_alignment.py)

### 7.1 起因

§31.5 R3 ([r3_pre_post_retrieval_geometry.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/paper_analysis/revision_relation_edge/r3_pre_post_retrieval_geometry.py)) 当前只用三个标量 `pre_pair_similarity / post_pair_similarity / retrieval_pair_similarity`，派生 `d_pre_post / d_post_retrieval / return_to_pre_index / retrieval_new_state_index` 等绝对值差与 1D 数轴指标；`vector_alignment_placeholder` 直接写死 `status="not_available_no_voxel_pattern_input_in_revision_master"`。§31.5 自己已声明：

> 不能证明 retrieval 是完全回到 pre，也不能强证 new-state reconstruction；voxel-level vector alignment 在本轮不可用。

这是 [result_new_meta_roi.md](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md) 中**唯一一处主动声明"未完成"的几何分析**，对应用户原句问题：

> 比较 KJ 和 YY 在 pre→post→retrieval 中，词对表征相似度，在脑网络中，从分离到重建的过程中，重建的方向是否相似。

D2 闭合该 caveat：在 voxel-level 直接做 vector-alignment，回答方向问题。spec 与 task 单独存放在 [.trae/specs/add-voxel-trajectory-alignment/](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/.trae/specs/add-voxel-trajectory-alignment/)。

### 7.2 输入（三档 fallback）

按优先级：

1. `--pattern-long-tsv` 指向的 long 表（subject / network / roi / stage / condition_item_id / word_index / voxel / value），pivot 还原 pattern matrix；
2. `--pattern-root` 下按 `{stage}/sub-XX/roi-XX_pair-XX.npz` 装载（npz 内 `pattern` 形状 `(2, n_voxel)`，0 行 = word_index 0，1 行 = word_index 1）；
3. 两者都不可用 ⇒ manifest 写 status=`no_pattern_input`，所有 cosine 留 NaN，**脚本不重新提取 pattern**（与 spec §3 一致），等数据机后续决定补哪条入口。

item table 入口与 C1 / C1b / D1 完全一致：`paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`，提取 condition / memory_strict / pair word ids / 材料 covariates。

### 7.3 计算细节

**pair state 与 differentiation**（spec §4.1）：

每个 (subject, roi, stage, condition_item_id) 取 pair 内两词 voxel pattern m1, m2 ∈ R^V，

- `s = (m1 + m2) / 2`（pair centroid）
- `Δ = m1 - m2`（differentiation 轴；方向按 `word_index` 升序固定）

三个 stage 各算一次 ⇒ (s_pre, s_post, s_retrieval) 与 (Δ_pre, Δ_post, Δ_retrieval)。三个 stage voxel 数不一致直接 status=`shape_mismatch` 跳过该 pair，不强行 padding。

**五件套 cosine + 两个投影**（spec §4.2，item-level）：

| 名称 | 公式 | 解读 |
| --- | --- | --- |
| `return_to_pre_cos` | `cos(s_retrieval - s_post, s_pre - s_post)` | >0 ⇒ retrieval 朝 pre 拉回 |
| `continued_diff_cos` | `cos(s_retrieval - s_post, s_post - s_pre)` | >0 ⇒ retrieval 沿 post 方向继续 |
| `bend_cos` | `cos(s_post - s_pre, s_retrieval - s_post)` | trajectory 拐角 |
| `pre_retrieval_axis_cos` | `cos(s_retrieval - s_pre, s_post - s_pre)` | retrieval 终点是否仍沿 pre→post 方向 |
| `differentiation_axis_cos` | `cos(Δ_retrieval - Δ_post, Δ_post - Δ_pre)` | differentiation 轴方向（**直接回答用户问句**） |
| `pre_recovery_projection` | `(s_retrieval - s_post) · (s_pre - s_post) / ||s_pre - s_post||` | retrieval 朝 pre 的投影长度 |
| `post_alignment_projection` | `(s_retrieval - s_pre) · (s_post - s_pre) / ||s_post - s_pre||` | retrieval 在 pre→post 轴上的位置 |

`||a|| · ||b|| == 0` 时 cosine 返回 NaN，进入 status=`zero_norm`。

**主模型**（spec §4.5）：

每个 network × outcome 独立跑

```text
cos_outcome ~ C(condition, Treatment('kj')) * memory_strict
            + sentence_char_len_z + word_frequency_mean_z
            + stroke_count_mean_z + valence_mean_z + arousal_mean_z
            + (1|subject) + (1|condition_item_id)
```

走 `shared_nc.fit_formula(family="gaussian")`（即 reviewer_supp 的 mixed-effects + OLS-HC3 fallback 链），BH-FDR family 按 `(network, outcome)` 切，与 a6 family 粒度一致。`q_for_ok_rows` 自动跳过 `singular / not_converged / nonfinite_se / single_class_outcome / too_few_rows` 行。

**跨条件 group direction agreement**（spec §4.3 + §4.4）：

对每个 (subject, network, condition) 在 condition 内做 item 平均得到 mean displacement vector `v̄_post`、`v̄_retrieval`、`ū_retrieval`。within-subject 跨条件 cosine：

- `cos_within_subject(v̄_post_KJ, v̄_post_YY)`
- `cos_within_subject(v̄_retrieval_KJ, v̄_retrieval_YY)`
- `cos_within_subject(ū_retrieval_KJ, ū_retrieval_YY)`

跨被试做 one-sample t（H0: 均值 = 0；用 erf 近似 normal CDF 避免 scipy 依赖），同时跑 N=200 permutation chance baseline（`--n-permutations` / `--rng-seed=42`），打乱 condition 内 pair-to-condition 标签得到 95% CI 与 `p_perm`。permutation 仅作 supplementary baseline，主统计仍由 mixed-effects + one-sample t 给出。

### 7.4 输出（沙箱 `paper_outputs/qc/nc_converge/d2_voxel_trajectory_alignment/`）

| 文件 | 内容 |
| --- | --- |
| `voxel_pattern_manifest.tsv` | 每个 (subject, roi, stage, pair) pattern 路径、voxel 数、状态（ok / missing / shape_mismatch / no_pattern_input） |
| `trajectory_cosines_item.tsv` | item-level 五件套 + 两个 projection；含 subject / network / condition / condition_item_id / memory_strict / 各 cos / 各 projection / status |
| `trajectory_cosine_models.tsv` | 每个 network × outcome 的 mixed-effects 全部系数 + 状态 + BH-FDR |
| `group_direction_agreement.tsv` | subject × network × outcome 的 within-subject cross-condition cosine + permutation baseline |
| `group_direction_agreement_summary.tsv` | network × outcome 的 mean / SE / one-sample t / p_t / p_perm / q_bh |
| `descriptives.tsv` | network × condition 的均值 cosine（与 R3 描述性表对齐） |
| `manifest.tsv` | 输入路径、行数、覆盖率、permutation N、rng_seed |

不写 figure。

### 7.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d2_voxel_trajectory_alignment/`，不写其他目录；
- `safe_output_path` 拒绝覆盖既有文件；
- **不修改** R3 脚本与其输出 `paper_outputs/qc/revision_relation_edge/r3_pre_post_retrieval_geometry/`、§31.5 已落盘文本，以及 nc_converge a/b/c 系列与 hpc_subfield_three_axis 模块；
- **不重新提取** voxel pattern（脚本只读，对 pattern 入口缺失走 fallback / NaN 标记）；
- 入口与 c1b / d1 一致，仍走 `shared_nc.add_common_args / default_config`。

### 7.6 与既有脚本/章节的关系

| 现有脚本/章节 | D2 与之关系 |
| --- | --- |
| §31.5 R3 (`r3_pre_post_retrieval_geometry.py`) | D2 闭合 §31.5 自承的 voxel vector alignment caveat；R3 输出保持不动，§31.5 文本等执行结果回填。 |
| `a4_post_memory_component.py` | 单位策略一致：raw 直接进 cosine（cosine 已天然归一），(1|subject)+(1|condition_item_id)，不做 z-of-z。 |
| `c1_input_output_transformation.py` | 互补：C1 在标量层看 pre→post 输入-输出；D2 在 voxel 层看 pre→post→retrieval 方向。 |
| `c1b_pre_similarity_tertile.py` | 互补 moderator 路径：C1b 用 pre similarity tertile 做切分；D2 用 trajectory direction 做切分。 |
| `d1_learning_understand_like_bridge.py` | 互补：D1 看 metacognitive moderator → post / memory；D2 看 retrieval 方向 × memory 交互。 |
| `a6_fdr_tiering.py` | D2 输出可后续挂入 `module=d2::network=...` 的 family；本批不修改 a6。 |

### 7.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d2_voxel_trajectory_alignment.py \
    --base-dir E:/python_metaphor
```

按需指定 pattern 入口：

```bash
python d2_voxel_trajectory_alignment.py \
    --base-dir E:/python_metaphor \
    --pattern-long-tsv E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv
# 或
python d2_voxel_trajectory_alignment.py \
    --base-dir E:/python_metaphor \
    --pattern-root E:/python_metaphor/paper_outputs/qc/lss_voxel_patterns
```

开发机自检（无 voxel pattern）：

```bash
python d2_voxel_trajectory_alignment.py --base-dir /tmp/nc_d2_dry --allow-empty
```

已通过：`python -m py_compile` ✓、`--help` ✓、dry-run 落 7 张 TSV ✓。

### 7.8 文档建议（拿到结果后，按 spec §8 Go/No-Go）

- **YY `return_to_pre_cos` 显著 > 0；KJ ≈ 0**：写 "YY retrieval reinstates pre-learning representational state；trained-edge separation 在 retrieval 阶段被部分回拉"。
- **YY `continued_diff_cos` 显著 > 0；KJ ≈ 0**：写 "YY retrieval 沿 post differentiation 方向继续推进，schema consolidation 而非 reactivation"。
- **`differentiation_axis_cos` 在 YY 显著 > 0；KJ ≈ 0**：直接写 "YY 重建方向沿 post differentiation 轴延展"，最直接回答用户问句。
- **跨 condition direction agreement cosine ≈ 0 或 < 0**：写 "KJ 与 YY retrieval direction 不一致，magnitude 与 direction 共同支持 condition-specific reorganization"。
- **全部 NS**：维持 §31.5 边界结论，并写 "voxel-level direction analyses do not yield additional condition-specific evidence beyond magnitude" 作为 negative evidence。

不在结果回填前主动改 §31.5 主线文字；D2 仅作为 §31.5 末尾的 supplementary direction-specific mechanism。

