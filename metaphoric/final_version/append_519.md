# append_519.md — 2026-05-19 NC converge 增量分析说明（Dm 全景：Where / When / How-fine / How-broad + ERS revived）

本文件记录 2026-05-19 在 `metaphoric/final_version/nc_converge/` 中新增的 6 个脚本，以及它们要回答的具体问题。本批新增不修改任何已有脚本和已有结果输出，仅在 sandbox `paper_outputs/qc/nc_converge/` 下产出新的 TSV；唯一例外是 `voxel_pattern_export.py` 写入 `paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv`，并由 `safe_output_path` 拒绝覆盖既有文件（spec §3.1 显式白名单）。

spec / tasks / checklist 单独存放在 [.trae/specs/expand-dm-narrative/](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/.trae/specs/expand-dm-narrative/)。

---

## 1. 背景与起因

`metaphoric/final_version/result_new_meta_roi.md` 当前 Dm 主线由 §28.4
([a4_post_memory_component.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/a4_post_memory_component.py))
承担，结论为「YY 的 memory 优势主要来自 post-stage representational separation；retrieval pair_similarity 与 ERS 的 reactivation 证据较弱」（[result_new_meta_roi.md L13-L19](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md#L13-L19) 与 [L1721-L1724](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md#L1721-L1724)）。

但已有分析在五个方向上**有数据但没做完**或**做过被撤下**，构成 NC reviewer 几乎一定会追问的盲区：

| 维度 | 现状 | 缺口 |
| --- | --- | --- |
| Where（解剖特化） | hpc_subfield_three_axis 只产出 §29 描述性 narrative | 没有 head/body/tail × condition × memory 的 a4 同构 Dm 模型 |
| When（动态轨迹） | A4 stage_long 只有 pre / post / retrieval 三时点 | 学习 run3 / run4 与 post run5 / run6 没拆开看 Dm slope |
| How-fine（评分粒度） | A4 outcome 仅 memory_strict (binary) / memory_lenient | memory_score (0/0.5/1) 与 memory_successes (0/1/2) 没进 ordinal 模型 |
| How-broad（跨网络综合） | A4 按 network 分别报告，未做 multi-network composite | reviewer 会问 single-network cherry-pick 风险 |
| Reactivation pivot | encoding_retrieval_similarity.py 与 reviewer_supp/m4_correct_ers.py 都做过 ERS 但被撤下 | 缺 own−other 基准 + memory moderation + network 化整理 |

本批补 6 个**互不耦合**但讲同一条故事的脚本，整体故事题为：

> **Where, When, How-fine, How-broad — YY 的 subsequent memory 效应在记忆几何上的全景**

ERS 复活作为 reactivation pivot，与 Where/When/How-fine/How-broad 共同闭合 §31.5 与 §28.4 的边界。

---

## 2. `voxel_pattern_export.py` — 解锁跨脚本 voxel 入口

[voxel_pattern_export.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/voxel_pattern_export.py)

### 2.1 起因

D2 ([d2_voxel_trajectory_alignment.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d2_voxel_trajectory_alignment.py)) 与本批 D3 / D5 都需要 voxel-level pattern 长表 `voxel_pattern_long.tsv`，但仓库中**没有现成的生成脚本**：D2 只在没有该文件时 fallback 到 npz pattern_root，等同于 D3 / D5 也只能在数据机临时拼装。本脚本是最薄的 helper，把 [stack_patterns.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/representation_analysis/stack_patterns.py) 已写出的 4D pattern + meta ROI mask 转成 long 形式，给 D2 / D3 / D5 共用。

### 2.2 输入

| 输入 | 路径 |
| --- | --- |
| pattern 4D + metadata | `pattern_root/sub-XX/{pre,post,learn,retrieval}_{yy,kj}.nii.gz` 与 `*_metadata.tsv`（[stack_patterns.py L267-L291](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/representation_analysis/stack_patterns.py#L267-L291)） |
| meta ROI mask | `roi_library/meta_sources/`（与 hpc_subfield_three_axis / encoding_retrieval_similarity 一致） |
| pair 模板 | `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv` 取 `subject / condition / condition_item_id / pair_word_index_0 / pair_word_index_1` |

### 2.3 计算细节

1. 枚举 `sub-XX/{stage}_{cond}_metadata.tsv`，按路径推断 `subject / stage / condition`；
2. 用 nibabel 读取 4D pattern + ROI mask，按 (subject, roi, stage, condition_item_id, word_index) 跨 trial 做平均（与 D2 单位策略一致）；
3. pivot 为 long 格式 (subject / network / roi / stage / condition_item_id / word_index / voxel / value)；
4. 写入 `paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv`，**仅在该文件不存在时写入**；存在则 `long_tsv_status="exists_skip_no_overwrite"`，不覆盖；
5. dev 友好：nibabel 缺失或 mask / pattern_root 缺失时返回空表（`--allow-empty` 模式），不抛异常。

### 2.4 输出

| 文件 | 内容 |
| --- | --- |
| `paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv` | 跨沙箱白名单（spec §3.1 唯一例外）；列含 subject / network / roi / stage / condition_item_id / word_index / voxel / value |
| `paper_outputs/qc/nc_converge/voxel_pattern_export/manifest.tsv` | (subject, roi, stage, pair) 计数与 voxel 数；含 `long_tsv_status` |

### 2.5 安全策略

- 写到 sandbox 外仅此一文件，路径在脚本内**显式硬编码**；
- `safe_output_path` 拒绝覆盖既有 long-tsv（保护现有 D2 fallback 数据不被污染）；
- 不修改 stack_patterns / run_lss / encoding_retrieval_similarity 任一上游脚本与既有输出；
- 入口与 c1b / d1 / d2 一致，走 `shared_nc.add_common_args / default_config`。

### 2.6 运行示例

```bash
cd metaphoric/final_version/nc_converge
python voxel_pattern_export.py --base-dir E:/python_metaphor

# 自定义 pattern_root 与 mask root
python voxel_pattern_export.py \
    --base-dir E:/python_metaphor \
    --pattern-root E:/python_metaphor/paper_outputs/qc/lss_voxel_patterns \
    --mask-root E:/python_metaphor/roi_library/meta_sources

# 开发机 dry-run
python voxel_pattern_export.py --base-dir /tmp/nc_export_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run ✓。

---

## 3. `d3_encoding_retrieval_similarity_revived.py` — ERS 复活 (own−other × memory × condition)

[d3_encoding_retrieval_similarity_revived.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d3_encoding_retrieval_similarity_revived.py)

### 3.1 起因

[encoding_retrieval_similarity.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/encoding_retrieval_similarity.py) 与 [reviewer_supp/m4_correct_ers.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/reviewer_supp/m4_correct_ers.py) 都做过 ERS（encoding–retrieval similarity），但 §31.5 撤下了 ERS 主线，原因有三：

1. 老 ERS 缺 own−other baseline，无法排除 condition-level pattern drift；
2. m4 跑过 own−other 但落在 reviewer_supp 副线，没上 §28.4 的 Dm framework；
3. 老 ERS 没把 memory 作为 moderator，无法回答"YY remembered items 是否 reinstatement 更强"。

D3 闭合三条诊断，把 own−other × condition × memory 三因素一次性放进 a4 同构的 mixed-effects model，并写 pre→retrieval baseline 作为 sanity check。

### 3.2 输入

- `paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv`（D2 入口；本批由 voxel_pattern_export 解锁）；
- 缺失时 fallback 到 `--pattern-root` npz（与 D2 同 fallback）；
- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv` 提取 condition / memory_strict / memory_lenient / memory_prop / memory_successes 与材料 covariates。

### 3.3 计算细节（spec §4.1）

**pair-level pattern**：每 (subject, roi, stage, condition_item_id) 把两词 voxel 向量平均得 pair pattern（与 m4_correct_ers 同口径）。

**ERS 三件套**（核心 outcome 行级）：

| 名称 | 公式 | 解读 |
| --- | --- | --- |
| `ers_post_to_retrieval_own` | `Fisher-z(corr(pair_post_i, pair_retrieval_i))` | item own reinstatement |
| `ers_post_to_retrieval_other` | `mean_{j ≠ i, condition(j) == condition(i)} Fisher-z(corr(pair_post_i, pair_retrieval_j))` | **condition 内** other baseline |
| `ers_post_to_retrieval_diff` | `own − other` | 核心 outcome；与 m4 同方向但 condition 内 baseline |
| `ers_pre_to_retrieval_*` | 同上但 pair_post → pair_pre | baseline 对照（pre 阶段是否本身已经达到 retrieval 相似度） |

**单位策略**：spec §2.3 显式禁止 own / other 单独条件内 z（避免 condition main effect 漏出去）；own−other 直接进模型。

**模型**（每 network 独立）：

```text
ers_post_to_retrieval_diff ~ C(condition, Treatment('kj')) * memory_var
                           + sentence_char_len_z + word_frequency_mean_z
                           + valence_mean_z + arousal_mean_z + stroke_count_mean_z
                           + (1|subject) + (1|condition_item_id)
```

`memory_var` ∈ {memory_strict, memory_lenient, memory_prop, memory_successes} 各跑一次；前三与 a4 一致，第四作为 ordinal 近似的健壮性检查。

**confirmatory term**：`C(condition)[T.yy]:memory_strict`，回答"YY remembered items 的 own−other reinstatement 是否高于 KJ"。

**baseline 对照**：`ers_pre_to_retrieval_diff` 跑同一 design，作为「pre 阶段就有的相似度漂移」的 sanity check。

### 3.4 输出

| 文件 | 内容 |
| --- | --- |
| `ers_item.tsv` | item-level (subject, network, roi, condition_item_id, memory_*) × {own, other, diff} × {post→retrieval, pre→retrieval} |
| `ers_models.tsv` | 每 network × outcome × memory_var 的 mixed-effects 全部系数 + 状态 + BH-FDR |
| `ers_baseline_pre_to_retrieval.tsv` | pre→retrieval baseline 模型表 |
| `ers_descriptives.tsv` | network × condition × memory 的均值 ERS（与 m4 描述性表对齐） |
| `manifest.tsv` | 输入路径、pattern source、行数、覆盖率 |

### 3.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d3_encoding_retrieval_similarity_revived/`；
- 不修改 encoding_retrieval_similarity.py / m4_correct_ers.py / r3_pre_post_retrieval_geometry.py 任一上游脚本与既有输出；
- 不重新提取 voxel pattern；voxel 入口缺失时返回空表。

### 3.6 与既有脚本/章节的关系

| 现有脚本/章节 | D3 与之关系 |
| --- | --- |
| §31.5 caveat | D3 在 caveat 末尾追加 "Encoding–retrieval reinstatement (own − other) revisited under memory moderation" 段，闭合 reactivation pivot |
| `m4_correct_ers.py` | 单位与 own−other 公式一致，但 D3 走 a4 同构 design：condition × memory mixed-effects；m4 沙箱保留不动 |
| `encoding_retrieval_similarity.py` | D3 独立产出，不复用其 main；temp_outputs 不改 |
| `a4_post_memory_component.py` | 复用 a4 同构 covariates / formula 骨架；不进 a4 沙箱 |

### 3.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d3_encoding_retrieval_similarity_revived.py --base-dir E:/python_metaphor

# 关闭 pre→retrieval baseline（默认开启）
python d3_encoding_retrieval_similarity_revived.py --base-dir E:/python_metaphor --no-include-pre-baseline

# 开发机 dry-run
python d3_encoding_retrieval_similarity_revived.py --base-dir /tmp/nc_d3_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run 落 5 张 TSV ✓。

### 3.8 文档建议（拿到结果后，按 spec §7 Go/No-Go）

- **YY × memory_strict 在 own−other diff 上显著正向**：写 "YY remembered items show higher post→retrieval own−other reinstatement than KJ; reactivation evidence revived under same−other baseline"；
- **不显著**：维持 §31.5 边界 "Even under same−other correction, post→retrieval reinstatement does not differ between conditions; consistent with §28.4 separation-driven account"。

---

## 4. `a4b_ordinal_dm.py` — 三档评分粒度的 Dm

[a4b_ordinal_dm.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/a4b_ordinal_dm.py)

### 4.1 起因

A4 的 outcome 只有 `memory_strict (binary)` / `memory_lenient` / `memory_prop (gaussian)`，但 [build_learning_post_memory_item_table.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/build_learning_post_memory_item_table.py) 同时记录了 `memory_score` ∈ {0, 0.5, 1} 与 `memory_successes` ∈ {0, 1, 2} 两个 ordinal 评分。reviewer 几乎一定会问 "binary collapse 是否丢失了中间档信息"。A4b 把 ordinal logit 显式接入 a4 同构 design，并提供 binomial / gaussian sanity 双跑 fallback。

### 4.2 输入

复用 [a4_post_memory_component.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/a4_post_memory_component.py) 的 `load / prepare / usable_covariates`：

- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`（与 c1 / c1b / d1 / a4 同源）；
- prepare 后保留 `post_pair_similarity_network_z` / `retrieval_pair_similarity_network_z` 与材料 covariates。

### 4.3 计算细节（spec §4.2）

**模型**（每 network 独立）：

```text
memory_score ~ C(condition, Treatment('kj')) * post_pair_similarity_network_z
             + C(condition) * retrieval_pair_similarity_network_z
             + sentence_char_len_z + word_frequency_mean_z + stroke_count_mean_z
             + valence_mean_z + arousal_mean_z
             + (1|subject) + (1|condition_item_id)
```

`memory_score` ∈ {0, 0.5, 1} 走 statsmodels `OrderedModel`（distr="logit"）；`memory_successes` ∈ {0, 1, 2} 走同 ordered logit。

**fallback 链**：

1. `OrderedModel` 不可用（statsmodels 未装 / patsy 缺失） → status = `ordered_model_unavailable: ...`；
2. `OrderedModel` 拟合失败（`design_matrix_failed` / `ordered_fit_failed`） → 自动 fallback；
3. 自动 fallback 在 `--ordinal-engine ordered_model` 模式下额外跑 `binomial(memory_strict)` 与 `gaussian(memory_score)` 双跑作为 sanity；fallback rows 通过 `fallback` 列标注。

不引入新依赖（`mord` 不会装）；只用 statsmodels 自带 OrderedModel。

### 4.4 输出

| 文件 | 内容 |
| --- | --- |
| `ordinal_dm_models.tsv` | 主 engine 模型系数 + 状态 + BH-FDR；含 fallback 行 |
| `binary_sanity_models.tsv` | `--ordinal-engine ordered_model` 时附带跑一次 binomial sanity |
| `ordinal_dm_descriptives.tsv` | network × condition × memory_outcome 的 n / mean / sd / n_subjects / n_items |
| `manifest.tsv` | engine / 输入路径数 / 行数 / subject 数 / networks |

### 4.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/a4b_ordinal_dm/`；
- 复用 a4 的 `prepare`，不修改 a4 任何代码与沙箱；
- BH-FDR 仅对 `status == "ok"` 行做校正。

### 4.6 与既有脚本/章节的关系

| 现有脚本 | A4b 与之关系 |
| --- | --- |
| `a4_post_memory_component.py` | A4b 复用其 `prepare / load / usable_covariates`，在同一 z 列上跑 ordered logit；不进 a4 沙箱 |
| `stage4_subsequent_memory_rework.py` | memory outcome 命名一致（memory_score / memory_successes 来自 stage4）；a4b 不重写 memory 列 |
| `a6_fdr_tiering.py` | a4b 输出可后续挂入 `module=a4b::network=...` family；本批不改 a6 |

### 4.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python a4b_ordinal_dm.py --base-dir E:/python_metaphor

# 强制走 binomial sanity（不跑 OrderedModel）
python a4b_ordinal_dm.py --base-dir E:/python_metaphor --ordinal-engine binomial_sanity

# 开发机 dry-run
python a4b_ordinal_dm.py --base-dir /tmp/nc_a4b_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run ✓。

### 4.8 文档建议（拿到结果后）

- **OrderedModel `YY × post_pair_similarity_z` 显著且方向与 a4 binary 一致**：写 "Granularity-graded Dm confirms YY × post separation interaction across rating levels"；
- **不显著但 binary sanity 显著**：写 "Binary Dm sufficient; ordinal does not gain additional sensitivity"，作为 negative robustness。

---

## 5. `a4c_multinetwork_composite.py` — 跨网络 composite Dm

[a4c_multinetwork_composite.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/a4c_multinetwork_composite.py)

### 5.1 起因

A4 按 network 分别报告 hpc-spatial 与 semantic/metaphor 的 post / retrieval 相似度模型，但 reviewer 在 NC 层面会追问"是不是 single-network cherry-pick？跨网络综合是否还能稳定预测 memory？"。A4c 把两个 network 的 4 个 z 列（post + retrieval × hpc + semantic）拼成 4 维特征，做 PCA + cross-network mean，得到 composite predictor 进 a4 同构 binomial model，并与单 network 模型对比 ΔAIC / ΔBIC / Δpseudo-R²。

### 5.2 输入

- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`（同 a4 prepare 后的 item-level z 列）；
- `paper_outputs/qc/nc_converge/a4_post_memory_component/memory_component_model.tsv`（**只读**，用于 ΔAIC 对比；缺失时跳过对比但保留 composite 主模型）。

### 5.3 计算细节（spec §4.3）

**composite 构造**：

1. 把 (subject, condition_item_id, network) × {post, retrieval} pivot 成宽表 `{post, retrieval}__{hpc_spatial, semantic}` 共 4 列；
2. 同 (subject, condition_item_id) 多 ROI 取均值；
3. 4 列做 numpy SVD（不依赖 sklearn）→ `composite_score_pca1` / `composite_score_pca2`；
4. 4 列直接平均 → `composite_score_mean`（cross-network mean，作为 sanity）。

**模型**：

```text
memory_strict ~ C(condition, Treatment('kj')) * composite_score_pca1
              + sentence_char_len_z + word_frequency_mean_z + stroke_count_mean_z
              + valence_mean_z + arousal_mean_z
              + (1|subject) + (1|condition_item_id)
```

`composite_score_pca2` 与 `composite_score_mean` 作为 sensitivity 各跑一次；`memory_lenient` / `memory_prop` 同 design 各跑一次。

**对比**：从 `a4_post_memory_component/memory_component_model.tsv` 读出 a4 单 network 模型的 AIC / BIC / n_obs，按 outcome × network 对比 composite_pca1 与之的 ΔAIC / ΔBIC，落 `composite_vs_network_comparison.tsv`。

### 5.4 输出

| 文件 | 内容 |
| --- | --- |
| `composite_models.tsv` | composite predictor 模型系数 + 状态 + BH-FDR |
| `composite_vs_network_comparison.tsv` | composite vs a4 单 network 的 ΔAIC / ΔBIC（缺 a4 表时返回 status 占位） |
| `composite_loadings.tsv` | PCA loadings + explained_variance_ratio（含 SVD failure 占位） |
| `composite_item_table.tsv` | (subject, condition_item_id, condition) × 4 z 列 + 3 composite 分数 + memory 列 |
| `manifest.tsv` | n_input_paths / n_composite_rows / n_subjects / a4_network_table 路径 |

### 5.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/a4c_multinetwork_composite/`；
- a4 沙箱**只读不写**（comparison 表只读 a4 的 model.tsv）；
- 不改 a4 任一脚本或输出。

### 5.6 与既有脚本/章节的关系

| 现有脚本 | A4c 与之关系 |
| --- | --- |
| `a4_post_memory_component.py` | A4c 复用其 prepare 后的 z 列；并对比其 model AIC/BIC；不进 a4 沙箱 |
| `a4b_ordinal_dm.py` | A4b 在 outcome 维度扩展（评分粒度），A4c 在 predictor 维度扩展（跨网络综合）；正交互补 |
| `c1_input_output_transformation.py` | 无耦合；C1 仍按 network 报告，A4c 是 NC layer 的 cross-network composite 二级输出 |

### 5.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python a4c_multinetwork_composite.py --base-dir E:/python_metaphor

# 指定 a4 网络结果表用于 ΔAIC 对比
python a4c_multinetwork_composite.py \
    --base-dir E:/python_metaphor \
    --a4-network-table E:/python_metaphor/paper_outputs/qc/nc_converge/a4_post_memory_component/memory_component_model.tsv

# 开发机 dry-run
python a4c_multinetwork_composite.py --base-dir /tmp/nc_a4c_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run 落 5 张 TSV ✓。

### 5.8 文档建议（拿到结果后）

- **composite_pca1 在 binomial(memory_strict) 上显著且 ΔAIC > 2 vs 任一 single-network**：写 "Multi-network composite predicts memory beyond either network alone (ΔAIC > 2)"；
- **不显著或 ΔAIC < 0**：写 "Single-network models suffice; composite does not improve fit"，作为 robustness。

---

## 6. `d4_subfield_dm.py` — HPC subfield × condition × memory

[d4_subfield_dm.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d4_subfield_dm.py)

### 6.1 起因

[hpc_subfield_three_axis](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/hpc_subfield_three_axis/) 模块产出 head/body/tail × L/R 6 段的 anterior-posterior beta，但当前结果表 §29 只给出描述性 narrative，没有进 a4 同构的 Dm 模型；reviewer 一定会问"YY × memory 的效应在 hippocampus long-axis 上是否非均匀"。D4 把 6 段 beta pair similarity 接入 a4 同构 condition × memory 框架，回答 anterior vs posterior subfield specificity。

### 6.2 输入

- `paper_outputs/qc/hpc_subfield_three_axis/s2_subfield_extract_beta/beta_long.tsv`（每行：subject / run_phase / condition / item_id / subROI / beta_vector_path）；
- 每行 `beta_vector_path` 指向 NPZ（key="beta"），按需懒加载；
- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv` 提供 memory_strict / memory_lenient / memory_prop / memory_successes 与材料 covariates。

### 6.3 计算细节（spec §4.4）

**subROI → (segment, hemisphere) 解析**：从 subROI 名称推断 segment ∈ {head, body, tail} 与 hemisphere ∈ {L, R}（共 6 段）。**不**重新切 CA1/CA3/DG（hpc_subfield 模块也未提供该切分；spec §4.4 显式锁定）。

**pair similarity**：每 (subject, subROI, run_phase, condition_item_id) 内 item beta 向量两两 Fisher-z(Pearson)，与 retrieval_pair_similarity 同公式；在 (subROI, run_phase) 内做 z（与 hpc_subfield_three_axis 内部口径一致）。

**模型 1**（per-segment × hemisphere，仅 post run_phase）：

```text
memory_strict ~ C(condition, Treatment('kj')) * pair_similarity_z + material covariates + (1|subject) + (1|condition_item_id)
```

每 (segment, hemisphere) 独立跑；memory_strict / memory_lenient 各一次。

**模型 2**（pooled three-way）：

```text
memory_strict ~ C(condition, Treatment('kj')) * pair_similarity_z * C(segment, Treatment('body'))
              + C(hemisphere)
              + material covariates
              + (1|subject) + (1|condition_item_id)
```

confirmatory term：`C(condition)[T.yy]:pair_similarity_z:C(segment)[T.head]` / `[T.tail]` 三阶交互。

### 6.4 输出

| 文件 | 内容 |
| --- | --- |
| `subfield_models_per_segment.tsv` | (segment, hemisphere) × outcome 的模型系数 + 状态 + BH-FDR |
| `subfield_pooled_three_way.tsv` | pooled 三阶交互模型的全部系数 + BH-FDR |
| `subfield_descriptives.tsv` | (segment, hemisphere, condition, run_phase) × n / mean_raw / sd_raw / mean_z / n_subjects |
| `pair_similarity_long.tsv` | item-level (subject, subROI, run_phase, segment, hemisphere, condition_item_id, pair_similarity_raw, _z, n_items) + memory + covariates |
| `beta_qc.tsv` | 每 (subject, subROI, run_phase, item_id) NPZ 装载状态 + n_voxels |
| `manifest.tsv` | 输入路径、行数、segments / hemispheres |

### 6.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d4_subfield_dm/`；
- 不修改 hpc_subfield_three_axis 任一脚本与其输出沙箱；只读 beta_long.tsv 与按 path 装载 NPZ；
- 开发机 dry-run 下输入缺失时返回空表，不抛异常。

### 6.6 与既有脚本/章节的关系

| 现有脚本/章节 | D4 与之关系 |
| --- | --- |
| §29 hpc_subfield narrative | D4 在 §28.4 末尾追加 a4 同构 subfield Dm，不改 §29 描述性 narrative |
| `a4_post_memory_component.py` | 同 design / 同 covariates；D4 把 ROI 维度替换为 segment × hemisphere |
| `c1_input_output_transformation.py` | 无耦合；C1 是 meta ROI input-output，D4 是 subfield Dm |

### 6.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d4_subfield_dm.py --base-dir E:/python_metaphor

# 自定义入口
python d4_subfield_dm.py \
    --base-dir E:/python_metaphor \
    --beta-long E:/python_metaphor/paper_outputs/qc/hpc_subfield_three_axis/s2_subfield_extract_beta/beta_long.tsv \
    --item-table E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv

# 开发机 dry-run
python d4_subfield_dm.py --base-dir /tmp/nc_d4_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run 落 6 张 TSV ✓。

### 6.8 文档建议（拿到结果后）

- **pooled three-way `YY × pair_similarity_z × segment[T.head]` 显著**：写 "YY × memory × subfield interaction localizes to anterior hippocampus (head)"；
- **`× segment[T.tail]` 显著**：写 "...localizes to posterior (tail)"；
- **三阶不显著但 per-segment head 模型 YY × pair_similarity_z 显著**：写 "anterior subfield carries YY-specific Dm signal under exploratory threshold"；
- **全部不显著**：写 "No subfield specificity; effect homogeneous along long-axis"，加强 §29 边界结论。

---

## 7. `d5_run_level_trajectory_dm.py` — run3 / run4 / run5 / run6 Dm slope

[d5_run_level_trajectory_dm.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d5_run_level_trajectory_dm.py)

### 7.1 起因

A4 stage_long 只有 pre / post / retrieval 三个时点，但 stack_patterns metadata 保留 run 列；learning (run3 / run4) 与 post (run5 / run6) 完全可拆。reviewer 几乎一定会问 "YY 的 Dm divergence 是从 run3 学习阶段就开始累积，还是在 post 才出现？" D5 在 voxel_pattern_long.tsv 上按 run 拆 4 个时点重算 pair similarity，跑 condition × run × memory 三阶交互（连续 slope + categorical 各一次）。

### 7.2 输入

- `paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv`（D2 入口；含 metadata.run）；缺失时 `--allow-empty` 返回空表；
- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv` 提供 memory_strict / memory_lenient / covariates。

### 7.3 计算细节（spec §4.5）

**run 拆分**：long 表中保留 run 列（命名兼容 `run / metadata_run / run_index / run_label`），强制转为 int 后筛选 `run ∈ {3, 4, 5, 6}`；run3 / run4 来自 learning，run5 / run6 来自 post。

**pair similarity**：每 (subject, network, roi, run, condition_item_id) 把两词 voxel pattern 做 Fisher-z(Pearson)（与 retrieval_pair_similarity 同公式）；在 (network, run) 内做 z（与 a4 单位策略一致）。

**模型**（每 network 独立）：

连续 slope：

```text
pair_similarity_z ~ C(condition, Treatment('kj')) * run_index * memory_strict
                  + sentence_char_len_z + word_frequency_mean_z + stroke_count_mean_z
                  + valence_mean_z + arousal_mean_z
                  + (1|subject) + (1|condition_item_id)
```

categorical 4 levels：

```text
pair_similarity_z ~ C(condition, Treatment('kj')) * C(run, Treatment(3)) * memory_strict
                  + material covariates
                  + (1|subject) + (1|condition_item_id)
```

confirmatory term：`C(condition)[T.yy]:run_index:memory_strict`（连续）+ `C(condition)[T.yy]:C(run)[T.5]:memory_strict` 等（categorical）。

### 7.4 输出

| 文件 | 内容 |
| --- | --- |
| `run_level_trajectory_long.tsv` | (subject, network, roi, run, condition_item_id) × pair_similarity_raw / _z + memory + covariates |
| `run_level_models.tsv` | network × outcome × {linear_run_slope, categorical_run} 的模型系数 + 状态 + BH-FDR |
| `run_level_descriptives.tsv` | (network, condition, run) × n / mean_raw / sd_raw / mean_z / n_subjects |
| `manifest.tsv` | 输入路径、pattern_source、行数、subject 数、runs_present |

### 7.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d5_run_level_trajectory_dm/`；
- 不修改 stack_patterns / a4 / r3 任一上游脚本；
- 不重新提取 voxel pattern；voxel 入口缺失时返回空表（与 D2 / D3 一致）。

### 7.6 与既有脚本/章节的关系

| 现有脚本/章节 | D5 与之关系 |
| --- | --- |
| `a4_post_memory_component.py` | A4 stage 级三时点；D5 在 learning + post 内拆 4 run，互补 |
| `d2_voxel_trajectory_alignment.py` | D2 在 stage 三点做 cosine direction；D5 在 4 run 做 pair similarity slope；同 voxel 入口共生 |
| `e4_learning_activation_sme.py`（518result §7.1） | E4 是 run-level activation 入口；D5 是 run-level representation 入口；横向对照 representation vs activation 的 run-level 一致性 |

### 7.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d5_run_level_trajectory_dm.py --base-dir E:/python_metaphor

# 指定 voxel_pattern_long.tsv
python d5_run_level_trajectory_dm.py \
    --base-dir E:/python_metaphor \
    --pattern-long-tsv E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv

# 开发机 dry-run
python d5_run_level_trajectory_dm.py --base-dir /tmp/nc_d5_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run 落 4 张 TSV ✓。

### 7.8 文档建议（拿到结果后）

- **`YY × run_index × memory_strict` 显著正向且 categorical 中 run3 / run4 系数已显著**：写 "YY Dm slope diverges from KJ as early as run3/4 (learning), continues into run5/6"；
- **仅 run5 / run6 显著**：写 "Dm divergence emerges only at run5/6 (post); learning-stage representational dynamics homogeneous"；
- **全部不显著**：写 "No run-level Dm trajectory specificity; A4 stage-level summary suffices"。

---

## 8. 与 518result 收敛点对照（数据机执行前的预读）

[518result.md](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md) 已在 ROI 层面给出三条收敛证据，本批 6 脚本在解释时**应吸收**这些 prior，但**不改代码骨架**：

1. **post_edge → memory 的 4 ROI focal subset**（[518result §2.5](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L424-L432)）：`meta_L_PPC_SPL` / `meta_R_PPC_SPL` / `meta_R_precuneus` / `meta_R_pMTG_pSTS`。
   - **D3 / A4c / D5 在解读时**：除了全 network 主表，可在 manifest 后人工 filter 4 ROI focal subset，作为 secondary 证据。
2. **trajectory decomposition 与 condition / memory 无 FDR 显著交互**（[518result §3.3](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L526-L572)）：
   - **D3 主线读法应优先**「item-level memory × ERS」而非「condition × ERS 三阶」；如 condition × memory × ERS 全 NS，按 spec §7 的 "维持写法" 处理，**不**强行改写主线。
3. **learning behavior 与 memory 的 condition-dependent relationship**（[518result §2.4](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L351-L399)）：`run3_understand_yes` 主效应（KJ baseline）正向，`YY × run3_understand_yes` 稳定负向。
   - **D5 解读时**：如 D5 显示 YY run3 / run4 representation 与 memory 没有显著交互，与 [E4](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L811-L908) "learning activation SME 不稳定" 一致，可联合写 "learning 阶段神经表征与 activation 都没有稳定 YY-specific Dm 累积"。
4. **E4 中 `meta_R_hippocampus run3` activation → memory 显著**：[518result §7.1.2](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L856-L884)。
   - **D4 subfield Dm 解读时**：与 E4 是 representation vs activation 的不同维度，二者横向对照（subfield pair similarity 是 within-pair 几何，E4 是 ROI mean activation；不冲突）。

> 总体策略：T1-T6 主结果走 spec §6 / §7 的 Go/No-Go，4-ROI focal subset / E4-E7 收敛仅作为 secondary supporting evidence。如本批数据机执行结果与 518result 收敛点全部一致，即可将故事整合写为：

> 隐喻学习的 subsequent memory 优势在记忆几何上的全景由 5 个维度共同锚定：(Where) post-stage edge differentiation 集中在 PPC/precuneus 与 right pMTG-pSTS（518result + a4c focal-ROI sanity）；(When) representation 在 run-level 没有稳定 YY-specific Dm 累积（D5 + E4 联合）；(How-fine) ordinal Dm 与 binary Dm 一致（a4b）；(How-broad) cross-network composite 与 single-network 等价或更优（a4c）；(Reactivation pivot) own−other ERS 在 memory moderation 下提供 §31.5 reactivation caveat 的闭合（d3）。anterior vs posterior subfield 维度由 d4 给出 supplementary specificity，不进主线。

---

## 9. 运行顺序与依赖

```text
voxel_pattern_export.py     ── 必须最先跑（D2 / D3 / D5 共用 voxel_pattern_long.tsv）
        │
        ├── d3_encoding_retrieval_similarity_revived.py
        ├── d5_run_level_trajectory_dm.py
        └── d2_voxel_trajectory_alignment.py（已执行，append_518.md §7）

a4b_ordinal_dm.py            ── 独立（只依赖 a4 prepare 与 item_mechanism_table.tsv）
a4c_multinetwork_composite.py ── 依赖 a4 输出 memory_component_model.tsv（仅读）
d4_subfield_dm.py            ── 依赖 hpc_subfield_three_axis/s2_subfield_extract_beta/
```

数据机执行命令模板（按顺序）：

```bash
cd metaphoric/final_version/nc_converge
python voxel_pattern_export.py        --base-dir E:/python_metaphor
python d3_encoding_retrieval_similarity_revived.py --base-dir E:/python_metaphor
python d5_run_level_trajectory_dm.py  --base-dir E:/python_metaphor
python a4b_ordinal_dm.py              --base-dir E:/python_metaphor
python a4c_multinetwork_composite.py  --base-dir E:/python_metaphor
python d4_subfield_dm.py              --base-dir E:/python_metaphor
```

---

## 10. 文档建议（数据机执行后回填）

按 spec §6：

- §28.4 末尾追加一节 "Subsequent-memory robustness across rating granularity, networks, subfields, and learning runs"，按 a4b → a4c → d4 → d5 顺序展开；
- §31.5 caveat 段后面写 "Encoding–retrieval reinstatement (own − other) revisited under memory moderation"，引用 d3；
- 不在 result_new_meta_roi.md 主线中改写 a4 / c1 / d1 / d2 任一段；
- 不在确认结果之前改任何主表叙事；本批 6 脚本仅产出 supplementary evidence，与 518result E1-E7 一起作为 reviewer 追问的 robustness layer。

---

## 11. `d1c_per_subject_multivariate_dm.py` — 个体级纯行为 understand/like → memory_strict 多元回归

[d1c_per_subject_multivariate_dm.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d1c_per_subject_multivariate_dm.py)

### 11.1 起因

[result_new_meta_roi.md §19.1 L668-L685](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md#L668-L685) 已经报告了 yy vs kj 在 understand / like 自身比例上的差异（understand 趋势 KJ ≥ YY、like 上 YY > KJ 显著），并在 [518result.md §2.4](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L351-L399) 给出 `condition × understand_yes` / `condition × like_yes` 在 memory 上的 GLMM 系数。但仓库现有 understand/like → memory 模型（[d1_learning_understand_like_bridge.py L503-L509](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d1_learning_understand_like_bridge.py#L503-L509) / [e2_all_roi_learning_bridge.py L199-L205](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/e2_all_roi_learning_bridge.py#L199-L205) / [e6_focused_learning_to_post_memory_rois.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/e6_focused_learning_to_post_memory_rois.py#L180-L186)）**全部带 pre_pair_similarity_z + RT + 材料 covariates + 部分版本带神经 mediator**，缺以下三块：

1. 真·"纯行为单变量"模型 `memory_strict ~ run3_understand_yes + run4_like_yes`（无任何 covariate / mediator）；
2. yy / kj subset 各自的 within-condition simple slope（不靠交互项手动反算）；
3. condition × understand_yes / like_yes 的 2×2 cell 命中率描述表（§28.4 / §2.4 都没给 cell 比例）。

D1c 用 **per-subject 两阶段多元回归** 闭合：每被试在三个 subset (yy / kj / pooled) 内对自己的 60 / 60 / 120 trials 跑 `memory ~ understand + like`，得到 28 套 β；再跨被试做 one-sample t-test (β vs 0) + Wilcoxon + paired t-test (β_yy vs β_kj)。这是 d1 GLMM 在被试-斜率分布维度的互补读数（不冲突，不替代）。

### 11.2 输入

- `paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`（提供 subject / condition / condition_item_id / remembered_strict 等）；
- `paper_outputs/qc/learning_reactivation_mapping/learning_behavior_item.tsv`（提供 run3_understand_yes / run4_like_yes 0/1 列）；
- 不读神经表，不依赖 voxel pattern；本脚本是**纯行为链**。

### 11.3 计算细节

**Stage 1 — 个体多元回归**：

主模型（默认）：

```text
memory_strict ~ run3_understand_yes + run4_like_yes
```

可选交互模型（`--include-interaction` 开启时额外跑一轮，主表保留）：

```text
memory_strict ~ run3_understand_yes + run4_like_yes + run3_understand_yes:run4_like_yes
```

设计要点：

- subset ∈ {yy, kj, pooled}；pooled 不做条件约束；
- 主引擎 `statsmodels.api.Logit`；quasi-separation / convergence failure / nonfinite SE / design rank deficient 时 fallback 到 OLS（memory_strict 当 0/1 数值，β 解释为 linear probability，p 用 t 分布）；
- 每被试每 subset 仅在 (a) 三列 (predictor + outcome) 全非空、(b) outcome 至少 2 个 class、(c) ≥ `--min-trials`（默认 12）时拟合；否则 status = `too_few_rows` / `single_class_outcome` / `design_rank_deficient`；
- **不做任何 covariate / 神经 mediator 加入**（这是与 d1 / e2 / e6 的核心差异）；
- 交互项开启时由于 KJ 内 `understand=0.946` 极偏，4-cell 不平衡可能让某 subject 的某 subset cell 完全空、`U:L` 列与 `U` 完全共线；脚本通过 `np.linalg.matrix_rank` 检查 + `design_rank_deficient` 标记跳过该被试-subset，不影响主模型行的拟合。

**被试可用性筛选**（响应用户的 26/28 经验）：

- 只接受 `run3_understand_yes + run4_like_yes + memory_strict` 三列同时**至少各有一条非空**的被试；
- 缺任一列的全部记入 `subject_eligibility.tsv` 并写明 `reason: all_null:run3_understand_yes` 等；
- 此筛选通过后再进入 Stage 1 拟合。

**Stage 2 — 跨被试聚合**：

每 (subset, predictor) 对：

- `scipy.stats.ttest_1samp(beta, 0.0)`（参数）；
- `scipy.stats.wilcoxon(beta)`（非参 sanity）；

paired YY vs KJ（仅在被试**同时**通过 yy 与 kj 两次拟合时入分子样本）：

- `scipy.stats.ttest_rel(beta_yy, beta_kj)`；
- `scipy.stats.wilcoxon(beta_yy − beta_kj)`；

BH-FDR family：所有 `status == "ok"` 的 inference 行一起做一次校正（含 one-sample 与 paired，共 8 行；family 粒度与 d1 不同——d1 是 network × outcome，d1c 是 subset × predictor）。

### 11.4 输出

| 文件 | 内容 |
| --- | --- |
| `per_subject_betas.tsv` | (subject, subset, predictor) × {beta, se, p, engine, status, n_obs}；engine ∈ {logit, ols_fallback, None}；status 包含 fallback 原因 |
| `group_inference.tsv` | (subset, predictor, test) × {n, mean_beta, t, p, wilcoxon_W, wilcoxon_p, q_bh, status}；含 paired_yy_minus_kj 行 |
| `cell_means_2x2.tsv` | (predictor, condition, predictor_value) × {n_items, n_subjects, memory_strict_mean / sd / se}；解决 §2.4 / §28.4 缺 cell 命中率描述的问题 |
| `subject_eligibility.tsv` | dropped subject id + 缺失列；用于回填"26 / 28 进入分析"叙事 |
| `manifest.tsv` | 输入路径、merge_status、n_subjects_eligible / dropped、fits_logit / fits_ols_fallback / fits_skipped 计数 |

### 11.5 安全策略

- 输出沙箱锁定 `paper_outputs/qc/nc_converge/d1c_per_subject_multivariate_dm/`；
- 不修改 d1 / e2 / e6 / e7 / build_learning_behavior_table 任一脚本与输出；
- 不依赖 reviewer_shared.fit_formula 或 mixed-effects 引擎，避免 d1 系列已有的 mixed singular fallback 噪声；统计推断走最朴素 ttest_1samp / ttest_rel / wilcoxon。

### 11.6 与既有脚本/章节的关系

| 现有脚本/章节 | D1c 与之关系 |
| --- | --- |
| [d1_learning_understand_like_bridge.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/d1_learning_understand_like_bridge.py) | d1 = item-level GLMM 群体平均；d1c = 被试-level 多元回归被试斜率分布。同一行为链的两种统计读法 |
| [e2_all_roi_learning_bridge.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/nc_converge/e2_all_roi_learning_bridge.py) | e2 是 18 ROI 的同 design 复核；d1c 与之正交（d1c 不分 ROI / 不进神经层） |
| [learning_behavior_models.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/brain_behavior/learning_reactivation/learning_behavior_models.py) | 那个跑反方向 `understand ~ condition`；d1c 跑正方向 `memory ~ understand + like`，读出 understand→memory 的纯行为链 |
| [518result.md §2.4](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L351-L399) | §2.4 给 GLMM 交互项系数；d1c 给"被试中位数 β / 几个被试满足该方向"的稳健性，回答"GLMM 显著是否被少数被试拉的"|
| [result_new_meta_roi.md §19.1](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/result_new_meta_roi.md#L668-L685) | §19.1 给 condition→understand/like 的 GEE；d1c 把 understand/like→memory 的反向链一并落档 |

### 11.7 运行示例

```bash
cd metaphoric/final_version/nc_converge
python d1c_per_subject_multivariate_dm.py --base-dir E:/python_metaphor

# 自定义最小 trial 阈值（被试样本太小时收紧）
python d1c_per_subject_multivariate_dm.py --base-dir E:/python_metaphor --min-trials 20

# 开发机 dry-run
python d1c_per_subject_multivariate_dm.py --base-dir /tmp/d1c_dry --allow-empty
```

已通过：`python -m py_compile` ✓ / `--help` ✓ / dry-run 落 5 张 TSV ✓。

### 11.8 文档建议（拿到结果后）

- **pooled 下 understand β > 0 且 t-test 显著**：写 "Across subjects, items endorsed as understood predict higher memory strict at paired-t level"；
- **paired_yy_minus_kj 下 understand β 显著负**：与 [518result §2.4](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/518result.md#L351-L399) `YY × understand` 负向交互一致，写 "Per-subject β confirms understand-memory slope is steeper for KJ than YY"；
- **like β 在 yy / kj 各自显著正**：写 "Subjective liking predicts memory in both conditions"；如 paired yy vs kj 不显著则补 "no condition difference in liking-driven Dm slope"；
- **cell_means_2x2.tsv 用法**：直接报告 "YY / understand=1 命中率 = X% (SE = ...)，KJ / understand=1 = Y%，YY 优势 = Δ%"；§28.4 与 §2.4 至此都补上 cell-level 描述。
- **若大量被试 fallback 到 OLS**（fits_ols_fallback / fits_logit 比例 > 0.3）：在文档中说明"由于 understand_yes 在 KJ 内均值 0.946，部分被试出现 quasi-separation；OLS fallback 给 linear probability slope，方向与 logit 一致"。

---

