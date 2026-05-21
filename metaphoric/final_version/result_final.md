# 结果与讨论终稿草案

本文的核心结论是：隐喻学习的关键不是让两个概念简单变得更相似，而是把 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 能在 retrieval 阶段被任务依赖地重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。当前结果仍不支持一条完整的 item-specific learning trace -> post separation -> retrieval rebinding -> memory 因果链，更适合写成阶段性表征重组。

![Final figure contact sheet](figure_final/fig_result_final_contact_sheet.png)

## Results

### 1. YY 隐喻学习产生稳定的行为记忆优势

行为结果首先提供了清楚的 endpoint：YY 隐喻关系学习带来了可靠的后续记忆优势。YY 条件的记忆正确率为 0.680，高于 KJ 空间关系条件的 0.588。混合 logit 模型中，YY 相对于 KJ 的条件效应稳定显著，beta = 0.398, p = 2.38e-06。加入 novelty、familiarity、comprehensibility 和 difficulty 等材料协变量后，YY 优势仍然稳定，beta 约为 0.548-0.553，p < 3.45e-07。

反应时结果排除了速度-准确权衡解释。KJ 条件平均反应时约为 1100 ms，YY 条件约为 1070 ms；logRT 模型中 YY 反应反而略快，beta = -0.026, p = .009，在协变量模型中仍保持显著，p = .022。因此，YY 的记忆优势不是以更慢反应或额外决策时间为代价获得的。

![Behavior accuracy and RT](figure_final/fig_behavior_accuracy_rt.png)

### 2. 学习阶段建立 condition-level semantic geometry，而不是稳定 item-specific trace

学习阶段最稳定的神经证据来自 condition-level geometry，而不是 item-specific trace。semantic/metaphor network 中，YY/KJ 条件层面的表征结构在学习阶段显著出现，beta = +0.0131, p = 0.00858, q = 0.0172。这说明学习任务在语义/隐喻相关网络中组织出了条件层面的几何结构。

![Learning dynamics semantic/metaphor network](figure_final/fig_learning_dynamics_meta_metaphor.png)

相比之下，item-specific same-pair trace 以及 learning-to-post bridge 的证据更弱，不适合作为主线。因此，learning 阶段应写成 condition-level semantic geometry，而不是每个 YY pair 从 learning 到 post 再到 retrieval 的稳定 neural trace。

MVPA 在这里可以作为 stage-state evidence。新整理的 MVPA summary 显示，learning 阶段有 10 个 q < .05 的 decoding rows，best q = 0.000036；retrieval 阶段也有稳定 decoding，best q = 0.000030。MVPA 的作用是说明 YY/KJ condition information 在哪些阶段被调动，而不是解释 edge geometry 如何重组。

![MVPA stage-state summary](figure_final/fig_revision_mvpa_stage_state_summary.png)

### 3. 学习后 isolated-word 阶段出现 YY-specific trained-edge differentiation

最强的机制证据仍然来自 post isolated-word 阶段。Step5C ROI-level RSA 显示，YY trained relation edges 在多个预定义 ROI 中出现显著 differentiation，尤其集中在 hippocampal-spatial / scene-context network 和部分语义控制区域。右海马效应最强，estimate = 0.125, p = 9.02e-08, q = 2.0e-06；右 temporal pole 也显著，estimate = 0.159, p = 2.00e-06, q = 3.2e-05。左海马 estimate = 0.092, p = 1.07e-05, q = 1.07e-04；左 IFG estimate = 0.105, p = 1.90e-05, q = 1.52e-04。

![Step5C main ROI effects](figure_final/fig_step5c_main.png)

edge specificity 进一步说明这些效应不是一般 isolated-word 稳定性。右海马中，YY trained edges 显著强于 baseline pseudo edges，effect = 0.125, p = 4.33e-07, q = 1.73e-05；也显著强于 KJ trained edges，effect = 0.093, p = 1.25e-05, q = 1.66e-04。右 PPA/PHG 中 YY trained > KJ trained，effect = 0.090, p = 1.90e-06, q = 3.80e-05；右 PPC/SPL 中同样显著，effect = 0.081, p = 3.12e-05, q = 2.50e-04。

![Edge specificity effects](figure_final/fig_edge_specificity.png)

word-stability control 进一步确认该结果不是 isolated words 的一般可重复性。

![Word stability control](figure_final/fig_word_stability.png)

因此，post 阶段应作为全文核心机制：YY 学习后，单独呈现的词不只是保留词身份，而是带上了训练过的 relation-edge 信息。这个结果不应被简单命名为 generic pattern separation；更准确的表述是 YY-specific trained-edge differentiation。

### 4. 材料调节分析显示：post separation 与 pre-existing pair structure 有关，但不能由 novelty 解释

本轮 revision 新增 r1/r2，用来回答“哪些隐喻更容易分化”。首先，memory moderation model 不支持 novelty 或 semantic distance 解释 YY memory advantage。YY x novelty 对 memory 不稳定：hpc-spatial estimate = +0.0119, q = 0.897；semantic/metaphor estimate = +0.0044, q = 0.932。YY x pre_pair_similarity 对 memory 也不稳定：hpc-spatial estimate = -0.0491, q = 0.250；semantic/metaphor estimate = -0.0120, q = 0.775。因此不能写成更 novel 或原始语义距离更合适的隐喻记忆更好。

![Material moderation effects](figure_final/fig_revision_material_moderation_effects.png)

post separation model 显示，post separation 与 pre_pair_similarity 有很强总体关系：hpc-spatial estimate = +0.7309, q = 2.84e-89；semantic/metaphor estimate = +0.6952, q = 5.08e-91。YY-specific pre-similarity interaction 不稳定，hpc-spatial q = 0.751，semantic/metaphor q = 0.450。YY-specific novelty moderation 对 post separation 只有边界性支持：hpc-spatial estimate = +0.2210, within-model q = 0.0494, across-model q = 0.0593；semantic/metaphor q = 0.0599。因此 novelty 可以作为补充讨论线索，但不能升级为主机制。

high/low post-separation profile 给出更清楚的材料线索：high-separation YY items 最稳定的特征是更高 pre_pair_similarity。top/bottom 30% split 中，low vs high estimate 在 hpc-spatial 为 -0.6269, p = 6.69e-06，在 semantic/metaphor 为 -0.6917, p = 7.38e-07。median split 方向一致。novelty、familiarity、valence、arousal 与 memory score 没有形成稳定 high/low profile。

![High vs low post-separation profile](figure_final/fig_revision_high_low_separation_profile.png)

source-first 审计后，run3/run4 学习阶段行为已经从原始 E-Prime CSV 接入，不再是 placeholder。run3 使用 sentence3_RESP / sentence3_RT / sentence3_ACC，run4 使用 sentence4_RESP / sentence4_RT / sentence4_ACC。更新后的 revision master 中，run3_understand 覆盖率为 98.57%，run4_like 为 97.86%，learning_fluency_shift 为 96.43%。接入这些源行为字段后，high vs low post-separation profile 已重新运行；结果显示 learning_fluency_shift 并不能解释 YY post separation 的 item 差异。top/bottom 30% split 中，hpc-spatial low - high estimate = +0.039, p = 0.777；semantic/metaphor low - high estimate = +0.045, p = 0.754。

因此，这一节最适合写成边界和材料线索：更强 post-stage separation 往往出现在 pre-existing pair structure 更强的 YY items 上，但核心 YY effect 不能被 novelty、semantic distance、run3/run4 行为流畅性或单一材料维度简单解释。

### 5. 阶段轨迹显示 post separation 与 retrieval rebound；trajectory geometry 不能强证 return-to-pre

阶段轨迹模型显示，YY 的关键特征不是全程相似性更高或更低，而是 post 阶段下降、retrieval 阶段回升。hpc-spatial network 中，YY post-stage drop 显著，condition[T.yy] = -0.0580, q = 5.39e-07；pre interaction = +0.0652, q = 3.93e-06；retrieval interaction = +0.0482, q = 6.18e-04。semantic/metaphor network 中方向相同：condition[T.yy] = -0.0688, q = 1.58e-08；pre interaction = +0.0680, q = 4.64e-06；retrieval interaction = +0.0534, q = 3.46e-04。

![Four-phase trajectory semantic/metaphor network](figure_final/fig_four_phase_trajectory_meta_metaphor.png)

![Four-phase trajectory hpc-spatial network](figure_final/fig_four_phase_trajectory_meta_spatial.png)

新增 r3 trajectory geometry 使用 scalar pair-similarity proxy 进一步检查 retrieval 是回到 pre，还是进入新的关系状态。描述统计显示，YY 在两个网络中都有更大的 post drop 与 retrieval rebound：hpc-spatial 中 YY post_drop_raw = 0.0724、retrieval_rebound_raw = 0.0589，而 KJ 分别为 0.0072 与 0.0107；semantic/metaphor 中 YY post_drop_raw = 0.0616、retrieval_rebound_raw = 0.0616，而 KJ 分别为 -0.0063 与 0.0082。

![Pre-post-retrieval triangle metrics](figure_final/fig_revision_pre_post_retrieval_triangle.png)

但是，trajectory metric 模型没有支持强 return-to-pre 或强 new-state reconstruction。hpc-spatial return_to_pre_index 的 YY effect q = 0.561；semantic/metaphor q = 0.957；memory interaction 也不稳定。voxel-level displacement vector alignment 在本轮不可用。因此，这一节应作为边界证据：YY 的 post drop / retrieval rebound 轨迹清楚，但当前 scalar proxy 不能证明 retrieval 完全回到 pre，也不能证明它形成新的几何状态。

![Displacement direction proxy](figure_final/fig_revision_displacement_vector_alignment.png)

### 6. Retrieval 阶段出现 task-driven rebound 和 semantic-hpc coordination

retrieval 阶段的结果显示，pair-level structure 会在任务需求下重新出现。右海马 retrieval pair similarity rebound 显著，effect = +0.111, p = 7.02e-09。run7 YY/KJ decoding 也重新出现，MVPA accuracy = 0.565, p = 3.47e-07。这说明 retrieval 不是 post differentiation 的简单延续，而是关系结构在任务上下文中被重新调动。

![Relation behavior prediction and retrieval effects](figure_final/fig_relation_behavior_prediction.png)

本轮 r4 network coupling revision 强化了 retrieval-stage coordination 的证据。semantic retrieval rebound 强预测 hpc-spatial retrieval rebound，estimate = +0.6977, q = 5.57e-63。这是本轮最强新增支持，说明 semantic/metaphor 与 hpc-spatial networks 在 retrieval 阶段的 pair-structure rebound 高度协同。

![Network coupling revision](figure_final/fig_revision_network_coupling.png)

不过，YY-specific coupling interaction 没有通过校正，estimate = -0.1095, p = 0.0392, q = 0.196；stage coupling 中 YY x retrieval 也只是趋势，estimate = +0.1526, p = 0.0213, q = 0.128。memory coupling 不稳定：YY x retrieval coupling q = 0.783，YY x post coupling q = 0.944。因此，这一结果应写成 retrieval-stage inter-network coordination，而不是 YY-specific causal communication 或 coupling-memory bridge。

![Representational connectivity](figure_final/fig_repr_connectivity.png)

### 7. 后续记忆主要对应 stronger prior post-stage separation

A4 memory component model 仍是行为桥接核心。YY memory advantage 主效应稳定存在。在 hpc-spatial network 中，memory_prop 模型的 YY 条件效应为 +0.106, q = 2.20e-52，memory_strict 模型为 +0.565, q = 2.00e-111。在 semantic/metaphor network 中，memory_prop 条件效应为 +0.107, q = 9.10e-43，memory_strict 为 +0.570, q = 8.84e-91。

关键是，post similarity 与 YY memory 的交互为负。hpc-spatial strict 模型中 interaction = -0.0177, q = 4.78e-04；semantic/metaphor strict 模型中 interaction = -0.0261, q = 3.41e-06。lenient 版本中方向更强：hpc-spatial interaction = -0.290, q = 1.71e-14；semantic/metaphor interaction = -0.358, q = 1.93e-18。这说明 remembered YY items 主要表现为 post 阶段 pair similarity 更低，也就是 stronger prior post-stage separation。

![A4 memory components](figure_final/fig_result_final_a4_memory_components.png)

retrieval similarity 的 memory 证据较弱且网络特异。semantic/metaphor network 中 retrieval similarity 与 memory 有一定正相关，strict = +0.129, q = 3.99e-06，proportional = +0.0131, q = 0.0210；但 hpc-spatial proportional 模型中 retrieval 项接近零，+0.0016, q = 0.768。因此，行为桥接不能写成 retrieval reinstatement 越强所以越记得。更准确的结论是：remembered YY items 主要具有 stronger prior post-stage separation，retrieval similarity 只提供有限且网络特异的额外证据。

![Mechanism behavior prediction](figure_final/fig_mechanism_behavior_prediction.png)

### 8. 稳健性与探索性分析限定解释边界

为了避免 revision 只依赖二次汇总表，本轮增加了 source-first audit。学习阶段行为已经从原始 E-Prime CSV 接入；retrieval geometry 也已经从 E:/python_metaphor/pattern_root/ 的 NIfTI pattern 和 E:/python_metaphor/roi_library/manifest.tsv 的 ROI masks 重新计算，并与原锁定上游 retrieval_pair_metrics.tsv 比较。复算结果显示，新旧 retrieval_pair_metrics 都为 32,760 行，所有 subject x roi_set x roi x condition x pair_id key rows 完全匹配，pre_pair_similarity、post_pair_similarity、retrieval_pair_similarity、retrieval_minus_post_pair_similarity、retrieval_minus_pre_pair_similarity、post_minus_pre_pair_similarity 以及 pair reinstatement 指标的 max_abs_diff 都为 0.0。因此，当前 revision 使用的 retrieval pair geometry 已经通过 source-level recompute 验证。需要保留的边界是：MVPA 和其他 upstream neural summaries 尚未逐一完成同级别 source-recompute audit，因此仍应标记为 locked upstream neural summaries。

![Source-first audit](figure_final/fig_source_first_audit.png)

B2 当前应称为 Mahalanobis/crossnobis-style robustness proxy，而不是 strict crossnobis。它的价值是检查方向是否与主结果一致，而不是替代主 RSA 结果。YY 在两个网络中都表现出负向 distance-like proxy：hpc-spatial YY proxy = -0.0138, corr = -0.0746；semantic/metaphor YY proxy = -0.0102, corr = -0.0668。KJ 方向相反或接近零。

![B2 Mahalanobis proxy](figure_final/fig_result_final_b2_mahalanobis_proxy.png)

C1 input-output transformation 显示，在控制 pre_pair_similarity 后，YY post differentiation 仍然存在：hpc-spatial condition effect = -0.176, q = 0.0229；semantic/metaphor condition effect = -0.186, q = 0.0229。但 pre-to-post slope moderation 不稳定，因此不能写成 YY 改变了完整转换函数的斜率。C2 支持 hpc-spatial global centrality decrease 的解释，但 memory-specific centrality 不稳定。C3 没有支持 novelty-only explanation。

![C1 C2 C3 boundaries](figure_final/fig_result_final_c1_c2_c3_boundaries.png)

hpc_subfield_three_axis 只能作为探索性扩展。QC 显示 112 个 head/body/tail masks 通过体素数阈值，56 个为 qc_warn；S2 构建 68,880 个 beta vectors；S3 混合模型出现 singular matrix。S5 长轴模型显示左右半球都有 YY post differentiation 主效应：left = -0.0720, q = 1.10e-04；right = -0.0837, q = 4.28e-05；pooled = -0.0779, q = 1.90e-06。但 anterior-posterior interactions 不稳定：left q = 0.141，right q = 0.792，pooled q = 0.241。因此，这部分支持 broad hippocampal involvement，但不支持可靠 AP gradient 或 subfield-specific mechanism。

![HPC long-axis summary](figure_final/fig_result_final_hpc_long_axis_summary.png)

## Discussion

### 1. 隐喻学习形成 relation-edge differentiation，而不是简单增加相似性

本研究最稳的理论贡献是：隐喻学习不是简单增强两个概念之间的神经相似性。YY 的关键模式是 learning 阶段的 condition-level semantic geometry、post 阶段的 trained-edge differentiation，以及 retrieval 阶段的 task-driven rebound。这个阶段性模式说明，隐喻学习更像是把既有语义几何转化为分化的 relation-edge representation。

### 2. 材料特征提供边界，而不是替代机制

新增材料分析帮助回答哪些隐喻更容易分化。high-separation YY items 往往具有更强 pre-existing pair structure，这提示隐喻重组不是凭空发生，而是依赖已有语义邻域。但 novelty、difficulty、familiarity、valence、arousal、memory score 以及 run3/run4 learning behavior proxy 都没有形成稳定 high/low profile，YY memory advantage 也不能由 novelty 或 semantic distance 简单解释。因此，材料特征和学习阶段行为指标应作为边界和线索，而不是替代神经机制。

### 3. Retrieval 是 task-driven rebound，但不能过度写成 return-to-pre 或 new-state proof

retrieval 阶段的 pair structure rebound、run7 decoding 和 semantic-hpc coupling 都支持任务驱动的重新调动。尤其是 semantic retrieval rebound 强预测 hpc-spatial retrieval rebound，说明两个网络在 retrieval 阶段协同恢复关系结构。不过，trajectory geometry 目前只是 scalar proxy，不能证明 retrieval 完全回到 pre，也不能强证进入新的几何状态。因此最稳表述是 task-driven reconstruction / rebound，而不是 strict reinstatement 或 definitive new-state re-binding。

### 4. Memory advantage 更像 schema-consistent endpoint

YY memory advantage 稳定，但 A4 显示 remembered YY items 的主要神经特征是 stronger prior post-stage separation。retrieval similarity 的 memory 证据有限且网络特异，r4 coupling-memory bridge 也不稳定。因此，memory 不应写成 retrieval reinstatement 的直接结果，而应写成与 post-stage relation-edge differentiation 一致的 endpoint pattern。

### 5. KJ 是 active comparison，稳健性分析限定过度解释

KJ 是 active spatial-relational comparison condition，不是 null control。KJ 也有 pair structure，只是没有 YY 那种清晰的 post differentiation + retrieval rebound 轨迹。B2、C1、C2、C3、HPC long-axis 与 MVPA 的作用，是让主线边界更清楚：B2 不是 strict crossnobis；C1 不是完整 slope transformation；C2 不是 memory centrality mechanism；C3 不支持 novelty-only explanation；HPC long-axis 不是 subfield mechanism；MVPA 是 stage-state evidence，不是 edge geometry evidence。

### 6. 最终模型

综合所有结果，最终模型可以写成：YY 隐喻学习首先在 semantic/metaphor network 中建立 condition-level geometry；随后在 post isolated-word 阶段，trained relation edge 从原有语义邻域中分化出来，尤其体现在 hippocampal-spatial / scene-context network；retrieval 阶段，pair-level structure 被任务需求重新组织，表现为 retrieval rebound、YY/KJ decoding 和 semantic-hpc coordination；后续被记住的 YY items 主要表现为 stronger prior post-stage separation。

因此，当前最稳的一句话是：隐喻学习的关键不是让两个概念更相似，而是把 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 能在 retrieval 阶段被任务依赖地重新组织。

## 主文图件建议

Fig 1：behavior。
Fig 2：learning condition geometry + MVPA stage-state inset。
Fig 3：post trained-edge differentiation + word identity control。
Fig 4：stage trajectory + trajectory geometry boundary。
Fig 5：memory component / prior post separation。
Fig 6：semantic-hpc retrieval coupling + robustness summary。
Supplement：source-first audit / retrieval geometry recompute；material and learning-behavior boundary；HPC long-axis exploratory robustness。

## 避免使用的表述

避免写：learning trace caused post separation；post separation caused memory；retrieval reinstatement explains memory；novelty explains YY memory advantage；trajectory geometry proves return-to-pre or new-state reconstruction；semantic-hpc coupling is YY-specific causal communication；KJ is null control；HPC long-axis proves subfield mechanism。
