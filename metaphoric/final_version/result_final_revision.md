# result_final_revision

本文档是基于最新 source-first revision 后的 Results/Discussion 中文草稿。它不替代正式投稿文本，但可以作为下一版 `result_final.md` 的结构基础。

## Results

### 1. 行为结果：YY 隐喻条件具有稳定记忆优势

行为结果仍然是全文的入口和锚点。YY 条件在后续记忆中稳定优于 KJ 条件，并且这一优势不能简单解释为速度-准确权衡。根据已有行为模型，YY accuracy 约为 0.680，高于 KJ 的 0.588；mixed logit 中 YY 条件效应稳定，`beta = 0.398, p = 2.38e-06`。在加入 novelty、familiarity、comprehensibility、difficulty 等材料协变量后，YY memory advantage 仍保持稳定。

这一结果在写作中只承担行为锚点功能：它说明隐喻学习确实带来更好的记忆表现，但不能单独证明具体神经机制。

### 2. 学习阶段：YY/KJ condition-level semantic geometry 被调动

学习阶段的主文表述应保持收束：learning 阶段出现 YY/KJ condition-level semantic geometry，但目前不应写成稳定的 item-specific learning trace。

已有 learning condition geometry 支持 semantic/metaphor network 在学习阶段区分 YY/KJ 条件；MVPA 也提供 stage-state 层面的补充证据。最新 MVPA 汇总显示：

```text
learning: n_q_lt_05 = 10, best_q = 0.000036
pre:      n_q_lt_05 = 0, best_q = 0.0983
post:     n_q_lt_05 = 2, best_q = 0.0354
run7:     n_q_lt_05 = 4, best_q = 0.000030
```

因此，MVPA 最适合写成：YY/KJ condition information 在 learning 和 retrieval 阶段被调动；它不承担 edge-reorganization mechanism 的主证据。RSA 负责解释 pair/edge geometry，MVPA 负责说明阶段性 condition information。

### 3. Post 阶段：YY trained relation edges 出现特异性 differentiation

Post 阶段仍是全文的机制中心。核心结论不是 YY 让两个词简单更相似，而是 YY trained relation edge 在学习后从原有语义邻域中分化出来。

已有 Step5C / edge specificity 结果支持这一点：右海马、左右海马、右 temporal pole、左 IFG，以及 PPA/PHG、PPC/SPL 等 hpc-spatial / scene-context 与 semantic/metaphor ROI 中，YY trained edge 的 pre-post drop 明显强于 KJ 或 baseline pseudo-edge。主文写作中应把它称为 trained-edge differentiation 或 relation-edge reorganization，而不是泛化为严格的 hippocampal pattern separation。

### 4. 材料边界：哪些 YY items 更容易发生 post separation？

新增 R1/R2 主要回答材料调节问题。结论是：YY memory advantage 不能由 novelty 或 semantic distance 简单解释；post separation 更强的 YY items 主要具有更高 pre-pair similarity，但 run3/run4 行为 proxy 不能解释这种差异。

R1 memory model 中，YY-specific novelty moderation 不稳定：

```text
hpc-spatial YY x novelty -> memory: estimate = +0.0119, q = 0.897
semantic YY x novelty -> memory:    estimate = +0.0044, q = 0.932
```

YY-specific pre_pair_similarity 对 memory 也不稳定：

```text
hpc-spatial q = 0.250
semantic q = 0.775
```

但 post separation 与 pre-pair similarity 有强总体关系：

```text
hpc-spatial pre_pair_similarity -> post separation:
estimate = +0.7309, q = 2.84e-89

semantic pre_pair_similarity -> post separation:
estimate = +0.6952, q = 5.08e-91
```

R2 high/low profile 显示，high-separation YY items 的稳定特征是更高 pre-pair similarity：

```text
hpc-spatial low vs high: estimate = -0.6269, p = 6.69e-06
semantic low vs high:    estimate = -0.6917, p = 7.38e-07
```

source-first 更正后，run3/run4 学习行为已从原始 E-Prime CSV 接入 master table，而不再是 placeholder。覆盖率为：

```text
run3_understand: 98.57%
run4_like: 97.86%
learning_fluency_shift: 96.43%
```

但这些 learning behavior proxy 并不能解释 high vs low post separation：

```text
hpc-spatial learning_fluency_shift low - high:
estimate = +0.039, p = 0.777

semantic learning_fluency_shift low - high:
estimate = +0.045, p = 0.754
```

因此，当前最稳写法是：更强 post separation 倾向出现于 pre-existing pair structure 更强的 YY items，但这种分化并不能被 novelty、memory score 或 run3/run4 行为流畅性简单解释。

### 5. Pre-post-retrieval 轨迹：YY 显示 post drop 与 retrieval rebound，但不能强证 return-to-pre

阶段轨迹仍支持 YY 的 post drop 与 retrieval rebound。新增 R3 用 scalar pair-similarity proxy 描述 pre-post-retrieval 三角关系：

```text
hpc-spatial YY:
post_drop_raw = 0.0724
retrieval_rebound_raw = 0.0589

hpc-spatial KJ:
post_drop_raw = 0.0072
retrieval_rebound_raw = 0.0107

semantic YY:
post_drop_raw = 0.0616
retrieval_rebound_raw = 0.0616

semantic KJ:
post_drop_raw = -0.0063
retrieval_rebound_raw = 0.0082
```

但 trajectory geometry model 没有支持强 return-to-pre 或 new-state reconstruction 结论：

```text
hpc-spatial return_to_pre_index YY effect: q = 0.561
semantic return_to_pre_index YY effect:    q = 0.957
```

因此，主文可以写 retrieval-stage task-driven rebound，但不能写 retrieval 完全恢复 pre 几何，也不能写已证明形成新几何方向。voxel-level displacement vector alignment 仍属于未来或补充分析。

### 6. Retrieval 阶段：pair structure rebound 伴随 semantic-hpc coordination

Retrieval 阶段的核心仍是 pair-level structure 以 task-driven rebound 形式重新出现。新增 R4 进一步显示，semantic/metaphor network 的 retrieval rebound 与 hpc-spatial network 的 retrieval rebound 强耦合：

```text
semantic_retrieval_rebound_z -> hpc_spatial_retrieval_rebound_z
estimate = +0.6977, q = 5.57e-63
```

这说明 retrieval 阶段存在很强的 predefined inter-network coordination。但 YY-specific interaction 没有通过校正：

```text
YY-specific interaction:
estimate = -0.1095, p = 0.0392, q = 0.196
```

coupling-memory bridge 也不稳定：

```text
YY x retrieval coupling -> memory: q = 0.783
YY x post coupling -> memory:      q = 0.944
```

因此，这一部分应写成 retrieval-stage semantic-hpc coordination，而不是 YY-specific causal communication，也不是 memory 的新强桥接。

### 7. Memory bridge：remembered YY items 主要表现为 stronger prior post-stage separation

行为桥接应继续以 A4 memory component model 为中心。当前最稳结论是：后续被记住的 YY items 在 retrieval 前已经表现出更强的 prior post-stage separation，而不是 retrieval similarity 本身更高。

换言之，不能把结果写成：

```text
retrieval reinstatement explains memory
```

更合适的写法是：

```text
remembered YY items had stronger prior post-stage separation
```

这与 relation-edge differentiation 的主线一致：成功记忆并不是因为 retrieval 阶段简单恢复原始相似性，而是与 post 阶段已经形成的更分化 relation-edge state 有关。

### 8. Source-level robustness：retrieval geometry 已经从源 NIfTI 复算验证

为了避免 revision 分析只依赖二次汇总表，本轮新增 source-first audit。run3/run4 行为已经从原始 E-Prime CSV 接入；retrieval geometry 已从 `pattern_root/*.nii.gz` 和 `roi_library` masks 重新计算，并与原锁定上游表比较。

复算结果：

```text
new retrieval_pair_metrics rows: 32760
old retrieval_pair_metrics rows: 32760
all key rows match: true
all audited numeric columns identical at 1e-12: true
```

这说明当前 revision 使用的 retrieval pair similarity、post/retrieval difference、pair reinstatement 等指标，与源 NIfTI 复算结果一致。需要保留的边界是：MVPA 和其他 upstream neural summaries 尚未逐一完成同级别 source-recompute audit，因此仍应标记为 locked upstream neural summaries。

## Discussion

### 1. 隐喻学习不是简单相似性增强，而是 relation-edge differentiation

当前最稳理论解释是：隐喻学习不会简单把两个概念拉近，而是在 post 阶段将 trained relation edge 从原有语义邻域中分化出来。这个说法比“similarity increase”更符合结果，因为 YY 的关键效应是 post-stage differentiation，而不是 post 阶段 pair similarity 更高。

### 2. 材料因素提供边界，而不是解释主效应

材料分析回答了“哪些隐喻更容易分化”的问题，但没有产生新的主机制。更高 pre-pair similarity 的 YY items 更容易表现出 high post separation；然而 novelty、familiarity、difficulty、memory score 和 run3/run4 learning fluency 都不能稳定解释 post separation 的差异。因此，材料因素应作为 boundary / supplementary analysis，而不是主文因果机制。

### 3. Retrieval 是 task-driven rebound，但几何方向仍需谨慎

Retrieval 阶段的 pair structure rebound 和 semantic-hpc coordination 支持 task-driven reconstruction 的表述。但 R3 只能提供 scalar proxy，不能证明 retrieval 回到 pre，也不能证明 retrieval 形成新状态。主文应避免使用过强的几何方向语言。

### 4. Memory endpoint 是 prior post-stage separation

后续记住的 YY items 主要表现为 stronger prior post-stage separation。这可以作为 relation-edge differentiation 与行为记忆优势之间的谨慎桥接。但不能写成 post separation causes memory，也不能写成 retrieval reinstatement explains memory。

### 5. KJ 是 active spatial-relational comparison

KJ 不能写成 null control。KJ 也有 spatial-relational learning 和 pair structure，只是没有表现出 YY 那种稳定的 post differentiation + retrieval rebound + memory bridge 组合。因此，主文应把 KJ 写成 active spatial-relational comparison。

### 6. 证据边界需要显式保留

以下内容应进入 Supplementary 或 boundary 叙述：

```text
learning behavior proxy -> post separation
novelty-only account
return-to-pre / new-state geometry proof
network coupling -> memory
MVPA-behavior bridge
HPC long-axis / AP gradient / MDS
relation-vector and directional mapping
```

这些结果不是失败，而是帮助主线避免过度扩张：当前论文最强的贡献是 relation-edge differentiation 与 task-driven retrieval coordination，而不是完整因果链。

## Recommended Figure Logic

```text
Fig 1. Design + YY memory advantage
Fig 2. Learning condition-level geometry + MVPA stage-state inset
Fig 3. Post YY trained-edge differentiation
Fig 4. Material boundary and high/low post-separation item profile
Fig 5. Stage trajectory: post drop + retrieval rebound
Fig 6. Retrieval semantic-hpc coordination
Fig 7. Memory component: stronger prior post-stage separation
Supplement. Source-first audit and retrieval geometry recompute
```
