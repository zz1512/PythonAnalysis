# Results / Discussion 故事线重构

## 2026-05-20 append_518/append_519 更新说明

518/519 的新分析不改变主线，但让故事边界更清楚。518 把 ROI-level learning boundary、pre-similarity boundary、post-edge-to-memory bridge 和 trajectory decomposition 补齐：pre-existing pair similarity 强烈影响 post-stage separation，但 YY-specific moderation 不稳定；run3 understand / run4 like 不能稳定预测 post separation 或 post pair similarity；post_edge_differentiation_z 与 memory 的桥接主要集中在 `meta_L_PPC_SPL`、`meta_R_PPC_SPL`、`meta_R_precuneus`、`meta_R_pMTG_pSTS`；trajectory decomposition 没有给出 return-to-pre / new-state 的稳定 ROI 证据。

519 进一步补上 memory endpoint 和 robustness boundary。ordinal Dm 支持 A4 的核心方向：YY remembered items 更符合 stronger prior post-stage separation，而不是更高 post similarity；semantic 网络中 YY x post_pair_similarity_network_z = -0.2085, q = 1.71e-10，hpc-spatial 中 = -0.1395, q = 1.99e-06。与此同时，ERS revival、run-level post run5/6 trajectory、per-subject multivariate Dm 和 composite/subfield analyses 都没有形成新的强机制主线。它们应作为 robustness / boundary，不应改写主故事。

## 2026-05-18 source-first 更新说明

这次 relation-edge revision 后，`result_story.md` 需要同步更新，但不需要推翻原有故事结构。更新原因有三点：

1. run3/run4 学习行为已经从原始 E-Prime CSV 接入，不再是 placeholder。源字段为 `sentence3_RESP / sentence3_RT / sentence3_ACC` 和 `sentence4_RESP / sentence4_RT / sentence4_ACC`。更新后的 revision master 中，`run3_understand`、`run4_like`、`run3_rt_z`、`run4_rt_z` 和 `learning_fluency_shift` 覆盖率约为 96-99%。
2. 接入 run3/run4 后，high vs low post-separation item profile 已重跑。结果显示 learning behavior proxy 不能解释 YY post separation 的 item 差异：`learning_fluency_shift` 在 hpc-spatial 中 `p = 0.777`，在 semantic/metaphor 中 `p = 0.754`。
3. retrieval geometry 已经从 `E:/python_metaphor/pattern_root/` 和 `E:/python_metaphor/roi_library/manifest.tsv` 重新计算，并与原锁定上游 `retrieval_pair_metrics.tsv` 完全一致：32760 行全部匹配，关键数值列在 1e-12 阈值下差异为 0。

因此，当前故事仍应保持为阶段性表征链：

```text
learning condition geometry
-> post trained-edge differentiation
-> retrieval task-driven rebound and semantic-hpc coordination
-> remembered YY items show stronger prior post-stage separation
```

需要新增的边界表述是：学习阶段行为 proxy 已经可用，但并不能解释哪些 YY items 更容易出现 post-stage separation；当前更稳的 item-level 线索仍是 high-separation YY items 具有更高 pre-pair similarity。retrieval geometry 已通过 source-level recompute 验证，但 MVPA 和其他 upstream neural summaries 仍应标记为 locked upstream neural summaries，除非后续继续逐一 source-recompute audit。

## 1. 最终主线

### 1.1 一句话主张

隐喻学习的关键不是让两个概念简单变得更相似，而是把 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 能在 retrieval 阶段被任务依赖地重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。

英文版本：Successful metaphor learning does not simply increase similarity between associated concepts. Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood, creating relation-edge representations that can be task-dependently reconstructed during retrieval.

### 1.2 最稳五段链条

当前主线仍然保持阶段性表征链，而不是完整因果中介链：

1. Behavior：YY 隐喻条件表现出稳定记忆优势，并且不是速度-准确权衡。
2. Learning：学习阶段出现 YY/KJ condition-level semantic geometry，但 item-specific trace 不足以主线化。
3. Post：学习后 isolated-word 阶段出现 YY-specific trained-edge differentiation，核心网络是 hpc-spatial / scene-context 与 semantic/metaphor systems。
4. Retrieval：retrieval 阶段 pair structure 以 task-driven rebound 方式重新出现，同时 semantic-hpc retrieval rebound coupling 很强。
5. Memory：remembered YY items 主要表现为 stronger prior post-stage separation，而不是 pure retrieval reinstatement。

### 1.3 本轮 revision 对故事的改变

本轮 relation_edge_story_revision 没有推翻旧主线，而是把旧主线从“YY similarity drop + retrieval rebound”推进到三个更审稿友好的问题：哪些隐喻更容易分化；retrieval rebound 是回到 pre 还是任务驱动的重新组织；semantic/metaphor 与 hpc-spatial network 是否协同支持 retrieval-stage reconstruction。

新增结果给出的答案是保守但有价值的：high-separation YY items 主要具有更高 pre-pair similarity；novelty 与 semantic distance 不能解释 YY memory advantage；trajectory geometry scalar proxy 不能强证 return-to-pre 或 new-state reconstruction；semantic retrieval rebound 强预测 hpc-spatial retrieval rebound，但 YY-specific coupling 与 coupling-memory bridge 不稳定。

### 1.4 不能写成的强因果链

不要写成 learning trace -> post separation -> retrieval re-binding -> memory。也不要写成 post separation causes memory、retrieval reinstatement explains memory、semantic-hpc coupling is YY-specific causal communication、trajectory geometry proves new-state reconstruction、novelty explains YY memory advantage。

append_518/519 后还不能写成：run3 understand / run4 like 解释 post separation 或 memory advantage；ordinal/composite Dm 发现了新的 memory mechanism；ERS reinstatement explains memory；post run5/6 run-level accumulation explains memory；HPC subfield 或 long-axis 证明了清晰 AP gradient。

最稳写法是：learning condition geometry -> post trained-edge differentiation -> retrieval task-driven rebound / network coordination -> remembered YY items show stronger prior post-stage separation。

### 1.5 新的解释原则：共同结构不等于简单耦合

可以借鉴另一个 ISC-age 研究的逻辑：两个层面都带有同一个背景结构，并不意味着二者之间一定存在全样本、全 ROI、单一方向的简单耦合。在那个研究中，Brain ISC 和 Behavior ISC 都与 Age model 显著，但 Brain ISC ~ Behavior ISC 不显著；这说明共同发展结构可能分布在不同成分、条件或尺度上。

对应到本研究，pre-existing pair similarity 像一个背景结构源：它强烈约束 post-stage separation，也帮助界定哪些 items 更容易分化；但这不等于 learning behavior、retrieval ERS、semantic-hpc coupling 或 run-level trajectory 必须与 memory 形成简单线性中介链。negative coupling / null moderation 结果不是故事失败，而是说明隐喻学习更像 stage-specific representational organization，而不是 single monotonic brain-behavior pathway。

## 2. Results 重构方案

### Result 1. Behavior: YY memory advantage

保留既有行为入口：YY accuracy = 0.680，高于 KJ = 0.588；mixed logit beta = 0.398, p = 2.38e-06。加入 novelty、familiarity、comprehensibility、difficulty 等协变量后仍稳定，beta 约 0.548-0.553, p < 3.45e-07。RT 上 YY 不慢于 KJ，KJ = 1100 ms，YY = 1070 ms；logRT beta = -0.026, p = .009，协变量模型 p = .022。

写作功能：行为优势是 anchor，不是机制证明。主图：figure_final/fig_behavior_accuracy_rt.png。

### Result 2. Learning: condition-level semantic geometry

保留 learning 阶段的降调表述：semantic/metaphor network 出现 condition-level geometry，beta = +0.0131, p = 0.00858, q = 0.0172。item-specific same-pair trace 和 learning-to-post bridge 不稳定，因此 learning 只写成 condition-level representational geometry。

新增 MVPA 作为 stage-state inset：learning decoding 有 10 个 q < .05 rows，best q = 0.000036。MVPA 只说明 YY/KJ condition information 在 learning 阶段被调动，不承担 edge-reorganization 机制。

### Result 3. Post: YY-specific trained-edge differentiation

这是主文核心机制。Step5C 中右海马 estimate = 0.125, p = 9.02e-08, q = 2.0e-06；右 temporal pole estimate = 0.159, p = 2.00e-06, q = 3.2e-05；左海马 estimate = 0.092, p = 1.07e-05, q = 1.07e-04；左 IFG estimate = 0.105, p = 1.90e-05, q = 1.52e-04。

edge specificity 进一步支持 trained-edge differentiation：右海马 YY trained > baseline pseudo effect = 0.125, q = 1.73e-05；YY trained > KJ trained effect = 0.093, q = 1.66e-04。右 PPA/PHG effect = 0.090, q = 3.80e-05；右 PPC/SPL effect = 0.081, q = 2.50e-04。

写作功能：post 阶段是机制中心，但不直接写成 generic pattern separation。

### Result 4. Material boundary: what kinds of YY items separate more?

新增 r1/r2 放在 post 机制之后，作为“哪些隐喻更容易分化”的材料边界。

R1 memory moderation 显示 YY memory advantage 不能由 novelty 或 semantic distance 简单解释。YY × novelty 对 memory 不稳定：hpc-spatial estimate = +0.0119, q = 0.897；semantic/metaphor estimate = +0.0044, q = 0.932。YY × pre_pair_similarity 对 memory 也不稳定：hpc-spatial q = 0.250；semantic/metaphor q = 0.775。

R1 post separation model 显示 post separation 强烈依赖 pre-pair similarity 的总体结构：hpc-spatial estimate = +0.7309, q = 2.84e-89；semantic/metaphor estimate = +0.6952, q = 5.08e-91。YY-specific novelty moderation 对 post separation 只有边界性支持：hpc-spatial estimate = +0.2210, within-model q = 0.0494, across-model q = 0.0593；semantic/metaphor q = 0.0599。

R2 high/low profile 显示 high-separation YY items 最稳定的材料特征是更高 pre_pair_similarity。top/bottom 30% split 中，low vs high estimate 在 hpc-spatial 为 -0.6269, p = 6.69e-06，在 semantic/metaphor 为 -0.6917, p = 7.38e-07。novelty、familiarity、valence、arousal 与 memory score 没有稳定 profile。

写作功能：可以说 stronger post separation tends to arise from items with stronger pre-existing pair structure, but is not reducible to novelty or later memory。

### Result 5. Stage trajectory and trajectory geometry

保留原 stage trajectory：hpc-spatial 中 YY post-stage drop 明显，condition[T.yy] = -0.0580, q = 5.39e-07；pre interaction = +0.0652, q = 3.93e-06；retrieval interaction = +0.0482, q = 6.18e-04。semantic/metaphor 中 condition[T.yy] = -0.0688, q = 1.58e-08；pre interaction = +0.0680, q = 4.64e-06；retrieval interaction = +0.0534, q = 3.46e-04。

新增 r3 trajectory geometry 使用 scalar pair-similarity proxy。描述上，YY 在两个网络中都有更大的 post drop 与 retrieval rebound：hpc-spatial 中 YY post_drop_raw = 0.0724、retrieval_rebound_raw = 0.0589；KJ 分别为 0.0072 与 0.0107。semantic/metaphor 中 YY post_drop_raw = 0.0616、retrieval_rebound_raw = 0.0616；KJ 分别为 -0.0063 与 0.0082。

但 r3 不能强证 retrieval 是 return-to-pre 或 new-state reconstruction：hpc-spatial return_to_pre_index YY effect q = 0.561；semantic/metaphor q = 0.957；voxel-level vector alignment 本轮不可用。因此写作上应说：trajectory proxy supports the post-drop/retrieval-rebound description, but does not prove whether retrieval returns to pre or forms a new state。

### Result 6. Retrieval: task-driven rebound and semantic-hpc coordination

保留 retrieval 主结果：右海马 retrieval pair similarity rebound effect = +0.111, p = 7.02e-09；run7 YY/KJ decoding accuracy = 0.565, p = 3.47e-07。

更新 r4 network coupling：semantic retrieval rebound 强预测 hpc-spatial retrieval rebound，estimate = +0.6977, q = 5.57e-63。这是本轮最强新增支持。YY-specific interaction 为 estimate = -0.1095, p = 0.0392, q = 0.196，不能写成 YY-specific coupling。stage coupling 中 YY × retrieval estimate = +0.1526, p = 0.0213, q = 0.128，只能作为趋势。memory coupling 不稳定：YY × retrieval coupling q = 0.783；YY × post coupling q = 0.944。

写作功能：retrieval 阶段支持 inter-network coordination，但不是 YY-specific causal communication，也不是 memory 的新强桥接。

### Result 7. Memory bridge: stronger prior post-stage separation

保留 A4 作为行为桥接核心。YY memory advantage 主效应稳定：hpc-spatial memory_prop condition YY = +0.106, q = 2.20e-52；memory_strict = +0.565, q = 2.00e-111。semantic/metaphor memory_prop = +0.107, q = 9.10e-43；memory_strict = +0.570, q = 8.84e-91。

post similarity 与 YY memory 的交互为负：hpc-spatial strict = -0.0177, q = 4.78e-04；semantic/metaphor strict = -0.0261, q = 3.41e-06；lenient hpc = -0.290, q = 1.71e-14；lenient semantic = -0.358, q = 1.93e-18。retrieval similarity 的 memory 证据较弱或网络特异。因此 memory bridge 必须写成 remembered YY items show stronger prior post-stage separation，而不是 retrieval reinstatement explains memory。

518 的 ROI-level bridge 把这个 memory endpoint 局部化：post_edge_differentiation_z 预测 memory 的 FDR-significant ROI 主要为 `meta_L_PPC_SPL`、`meta_R_PPC_SPL`、`meta_R_precuneus`、`meta_R_pMTG_pSTS`。学习行为项本身不支持正向机制解释：YY x run3_understand_yes 和 YY x run4_like_yes 在 direct memory model 及加入 post-edge 后均为 18/18 ROI FDR 显著负向，不能写成 YY 中“理解/喜欢越高记忆越好”。

519 的 ordinal Dm 是 A4 的稳健性补强。ordered memory 中，semantic 网络 YY x post_pair_similarity_network_z = -0.2085, q = 1.71e-10；hpc-spatial = -0.1395, q = 1.99e-06。semantic YY x retrieval_pair_similarity_network_z 为正，estimate = +0.1082, q = 7.15e-04，但 ERS revival 没有给出 memory moderation 的 FDR 证据，因此 retrieval 仍不能升格为主要 memory explanation。

### Result 8. Robustness and exploratory boundaries

B2 仍称为 Mahalanobis/crossnobis-style robustness proxy，不称 strict crossnobis。C1 支持控制 pre similarity 后的 output-level differentiation，但 slope moderation 不稳定。C2 支持 hpc-spatial global centrality interpretation，但 memory-specific centrality 不稳定。C3 不支持 novelty-only explanation。HPC long-axis 支持 broad hippocampal involvement，但不支持 AP gradient 或 subfield mechanism。

518/519 新增边界：pre-similarity tertile 对 post_edge_differentiation_z 是强总体效应，high vs low 与 mid vs low 均为 18/18 ROI FDR 显著，但 YY-specific tertile interaction 为 0/18 ROI FDR；trajectory return_to_pre_cos、new_state_fraction、return_new_log_ratio、differentiation_axis_cos 的 condition/memory/interaction 均无稳定 FDR 证据；activation SME 和 beta-series connectivity SME 不能作为主线；D1c per-subject multivariate Dm、D3 ERS revival、D5 run-level post trajectory 均没有形成 FDR-significant group-level memory mechanism。D4 subfield 有 pooled YY x pair 负向证据，但 per-seg strict focal 不稳定，只能作为 supplementary anatomical robustness。

写作功能：这些 null / boundary 结果应被组织为“共同结构不必产生简单耦合”的证据。它们排除了几条过度直接的解释路径：learning behavior proxy、retrieval reinstatement、run-level accumulation 和 subfield gradient 都不能单独解释 memory endpoint；主线因此应保持为阶段性表征链，而不是线性中介链。

## 3. 推荐 Results 顺序

1. Behavioral memory advantage.
2. Learning condition-level semantic geometry and MVPA stage-state evidence.
3. Post-learning YY-specific trained-edge differentiation.
4. Material boundary: high-separation YY items and novelty/distance moderation.
5. Stage trajectory and trajectory geometry boundary.
6. Retrieval rebound and semantic-hpc coordination.
7. Memory bridge: prior post-stage separation.
8. Robustness and exploratory anatomical boundaries.

## 4. Discussion 重构方案

### Discussion 1. Relation-edge differentiation, not simple similarity increase

核心理论段落要明确：隐喻学习不是让两个概念简单更相似，而是使 trained relation edge 从原有语义邻域中分化出来。post-stage differentiation 是机制中心。

### Discussion 2. Materials modulate but do not explain the effect

新增 r1/r2 可以让 Discussion 回答“什么样的隐喻更容易分化”。最稳说法是：更高 pre-pair similarity 的 YY items 更容易出现 high post separation，但 novelty、difficulty、memory score 等单一材料维度不能解释核心效应。

### Discussion 3. Retrieval is task-driven rebound, but geometry remains bounded

retrieval 阶段 pair structure rebound 与 semantic-hpc coordination 支持 task-driven reconstruction。r3 不能证明 return-to-pre，也不能强证 new-state vector geometry，因此 Discussion 要避免“完全恢复 pre”或“证明新状态”的绝对说法。

### Discussion 4. Memory endpoint is prior post-stage separation

行为桥接继续写成 remembered YY items had stronger prior post-stage separation。r4 的 coupling-memory bridge 不稳定，因此不新增“network coupling predicts memory”的主张。

518/519 后，这一点更稳而不是更强因果化：ordinal Dm 复制了 post similarity x YY memory 的负向方向，ROI-level bridge 主要落在 bilateral PPC/SPL、right precuneus 和 right pMTG/pSTS；但 ERS、composite、run-level accumulation 和 learning behavior proxy 没有提供新的中介链。因此 Discussion 应强调 memory endpoint 是 prior post-stage separation 的行为关联，而不是完整 causal pathway。

### Discussion 5. Why null coupling does not break the story

可以在 Discussion 中加入一个方法论式段落：本研究中多个层面都显示有结构，pre-existing pair similarity 约束 post separation，YY condition 产生 memory advantage，post-edge differentiation 与 memory 有局部 bridge，retrieval 阶段也有 rebound / network coordination；但这些结构不必在 learning behavior、ERS、run-level trajectory 或全局 brain-behavior coupling 中表现为同一条简单线性路径。

推荐英文表述：

```text
The absence of a simple learning-behavior, reinstatement, or run-level coupling does not weaken the representational account. Instead, it indicates that metaphor learning is organized by stage-specific representational structure rather than by a single monotonic brain-behavior pathway.
```

这个段落的功能是把 518/519 的 negative results 转成解释边界：它们说明哪些更简单的路径不能解释 memory endpoint，同时保护主线不被迫写成完整因果中介。

### Discussion 6. KJ active comparison and evidence boundaries

KJ 是 active spatial-relational comparison，不是 null control。MVPA 是 stage-state evidence；HPC long-axis 是 exploratory robustness；B2 不是 strict crossnobis；MDS 不作为统计证据。

新增边界也要写清楚：run3/run4 行为有可用来源，但不能解释 high-separation YY item profile；ordinal 和 binary Dm 支持 memory bridge，但 composite lenient 结果、subfield pooled effect、activation/connectivity SME 只能放在 sensitivity 或 supplement；post run5/6 run-level trajectory 因为与 run3/run4 学习任务粒度不同，不能被解释成学习期到测试期的连续积累曲线。
