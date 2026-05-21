# result_story_revision

## 1. 当前主线

这轮 relation-edge revision 没有推翻原主线，而是把故事进一步收束为：

```text
Behavior:
YY memory advantage

Learning:
YY/KJ condition-level semantic geometry emerges

Post:
YY trained relation edges show specific differentiation

Retrieval:
YY pair structure rebounds in a task-driven manner, with semantic-hpc coordination

Memory:
remembered YY items show stronger prior post-stage separation
```

一句话版本：

> 隐喻学习的关键不是让两个概念简单变得更相似，而是把 trained relation edge 从原有语义邻域中分化出来；这种分化后的关系边可以在提取阶段被任务依赖地重新组织，并且后续被记住的 YY items 主要表现为更强的 prior post-stage separation。

英文版本：

> Successful metaphor learning does not simply increase similarity between associated concepts. Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood, creating relation-edge representations that can be task-dependently reconstructed during retrieval.

## 2. Source-first 更正

这轮重新审计后，之前关于 run3/run4 行为只能作为 placeholder 的判断需要更正。精简 `*_events.tsv` 中确实没有学习阶段反应，但原始 E-Prime CSV 中存在完整字段：

```text
run3: sentence3_RESP / sentence3_RT / sentence3_ACC
run4: sentence4_RESP / sentence4_RT / sentence4_ACC
```

已更新 `r0_build_revision_master_table.py`，现在 revision master 会接入源 CSV 抽取出的：

```text
run3_understand
run4_like
run3_rt
run4_rt
run3_rt_z
run4_rt_z
learning_fluency_shift
```

覆盖率良好：

```text
run3_understand: 98.57%
run4_like: 97.86%
run3_rt_z: 98.57%
run4_rt_z: 97.86%
learning_fluency_shift: 96.43%
```

同时新增 source audit：

```text
paper_analysis/revision_relation_edge/r0a_source_inventory.py
paper_analysis/revision_relation_edge/r0b_compare_source_recompute.py
```

`r0b` 将从 `pattern_root` 和 `roi_library` 重新计算得到的 retrieval geometry 与原锁定上游表比较，结果完全一致：

```text
retrieval_pair_metrics rows: 32760 vs 32760
all key rows match: true
all audited numeric columns identical at 1e-12: true
```

因此，当前版本可以说：run3/run4 行为已经从源 CSV 接入；retrieval geometry 已经过 source-level 复算验证；其他 neural summary 仍是 locked upstream neural summaries，若要达到完全 source-only，需要继续对 MVPA 和其他上游表做同类复算审计。

## 3. 新增分析怎样回答问题

### 3.1 材料语义距离和新颖性

R1 不支持“YY memory advantage 由 novelty 或 semantic distance 简单解释”。

memory model 中：

```text
hpc-spatial YY x novelty: estimate = +0.0119, q = 0.897
semantic YY x novelty:    estimate = +0.0044, q = 0.932

hpc-spatial YY x pre_pair_similarity: q = 0.250
semantic YY x pre_pair_similarity:    q = 0.775
```

post separation model 中，pre-pair similarity 与 post separation 有很强总体关系：

```text
hpc-spatial pre_pair_similarity: estimate = +0.7309, q = 2.84e-89
semantic pre_pair_similarity:    estimate = +0.6952, q = 5.08e-91
```

YY-specific novelty moderation 对 post separation 只有边界支持：

```text
hpc-spatial YY x novelty: estimate = +0.2210, across-model q = 0.0593
semantic YY x novelty:    estimate = +0.2077, q = 0.0599
```

写作含义：novelty 可以作为 supplementary/boundary 线索，但不能升为主机制。

### 3.2 High vs low post-separation YY items

R2 的稳定结果是 high-separation YY items 主要具有更高的 pre-pair similarity。

top/bottom 30% split：

```text
hpc-spatial low vs high pre_pair_similarity: estimate = -0.6269, p = 6.69e-06
semantic low vs high pre_pair_similarity:    estimate = -0.6917, p = 7.38e-07
```

接入 run3/run4 源行为后，learning behavior proxy 没有解释 high/low post separation：

```text
hpc-spatial learning_fluency_shift low - high: estimate = +0.039, p = 0.777
semantic learning_fluency_shift low - high:    estimate = +0.045, p = 0.754
```

写作含义：可以回答“哪些隐喻更容易分化”这个问题，但答案不是 run3/run4 行为流畅性，而是更强的 pre-existing pair structure。

### 3.3 Pre-post-retrieval 轨迹几何

R3 用 scalar pair-similarity proxy 描述三阶段轨迹。描述上，YY 在两个网络中都表现为更大的 post drop 和 retrieval rebound：

```text
hpc-spatial YY: post_drop_raw = 0.0724, retrieval_rebound_raw = 0.0589
hpc-spatial KJ: post_drop_raw = 0.0072, retrieval_rebound_raw = 0.0107

semantic YY: post_drop_raw = 0.0616, retrieval_rebound_raw = 0.0616
semantic KJ: post_drop_raw = -0.0063, retrieval_rebound_raw = 0.0082
```

但 return-to-pre / new-state reconstruction 没有得到强统计支持：

```text
hpc-spatial return_to_pre_index YY effect: q = 0.561
semantic return_to_pre_index YY effect:    q = 0.957
```

写作含义：可以写 task-driven rebound，但不能写 retrieval 证明回到 pre，也不能写证明进入新几何状态。voxel-level vector alignment 仍需要更底层 pattern 分析。

### 3.4 Semantic-hpc network coupling

R4 最强新增结果是 retrieval rebound 的 semantic-hpc coordination：

```text
semantic_retrieval_rebound_z -> hpc_spatial_retrieval_rebound_z
estimate = +0.6977, q = 5.57e-63
```

但 YY-specific interaction 和 coupling-memory bridge 不稳定：

```text
YY-specific interaction: estimate = -0.1095, p = 0.0392, q = 0.196
YY x retrieval coupling -> memory: estimate = +0.0119, q = 0.783
YY x post coupling -> memory:      estimate = +0.0018, q = 0.944
```

写作含义：支持 retrieval-stage inter-network coordination，不支持 YY-specific causal communication，也不支持 coupling 直接预测 memory。

### 3.5 MVPA stage-state evidence

MVPA 应作为 stage-state evidence，而不是 edge-reorganization mechanism evidence。

```text
pre:       n_q_lt_05 = 0, best_q = 0.0983
learning:  n_q_lt_05 = 10, best_q = 0.000036
post:      n_q_lt_05 = 2, best_q = 0.0354
retrieval: n_q_lt_05 = 4, best_q = 0.000030
behavior bridge: not stable
```

写作含义：RSA 负责解释 pair/edge geometry，MVPA 负责说明 YY/KJ condition information 在 learning 和 retrieval 阶段被调动。

## 4. Evidence tier

### Tier A: 主文核心

```text
Behavior YY memory advantage
Learning condition-level semantic geometry
Post YY trained-edge differentiation / Step5C
Stage trajectory: post drop + retrieval rebound
Memory component: remembered YY = stronger prior post separation
Retrieval semantic-hpc coordination
```

### Tier B: 主文补充或重要补充材料

```text
Material moderation boundary
High/low post-separation item profile
Trajectory geometry proxy
MVPA stage-state summary
Mahalanobis / crossnobis-style robustness proxy
Source-level retrieval geometry audit
```

### Tier C: Supplementary / boundary

```text
Novelty moderation if uncorrected/borderline
Learning behavior proxy -> post separation
Network coupling -> memory
Return-to-pre / new-state geometry claim
MVPA-behavior bridge
HPC long-axis / MDS / AP gradient
Relation-vector and directional mapping
```

## 5. 当前最稳写法

最终 story 应避免完整因果链：

```text
learning trace -> post separation -> retrieval re-binding -> memory
```

推荐写成阶段性表征链：

```text
learning condition geometry
-> post trained-edge differentiation
-> retrieval task-driven rebound and network coordination
-> remembered YY items show stronger prior post-stage separation
```

尤其要避免把 retrieval-post difference score 写成 pure retrieval reinstatement。后续记住的 YY items 更稳的行为桥接是 stronger prior post-stage separation，而不是 retrieval similarity 本身更高。
