# new_way.md

最后更新：2026-05-03

本文档是在当前 `result_new.md` 阶段性结果基础上，为冲击 PNAS 目标制定的新分析路线。核心判断是：当前结果已经形成稳定主线，但还缺少一个足够有解释力的计算机制。下一步不应继续单纯围绕 `post - pre pair similarity` 下降重复扩展，而应把主效应升级为“创造性关系学习如何重塑语义关系几何”的机制故事。

## 0. 当前出发点

当前最稳的结果链条：

- 行为层：`YY` 相比 `KJ` 有更好的最终记忆/提取表现，且正确 trial 反应更快。
- 神经表征层：三套 ROI 中 `YY/Metaphor` 均表现出稳定的 pre-to-post pair similarity 下降，`KJ` 与 baseline 不呈现同等下降。
- 稳健性层：bootstrap、LOSO、split-half 与 noise ceiling 支持 Step 5C 主效应方向稳定。
- 边界层：Model-RSA、item-level 脑-行为、S1、representational connectivity、gPPI 当前较弱或阴性。

当前主要限制：

- Model-RSA 主要只有 `M3_embedding` 与 `M8_reverse_pair` 两类核心模型，机制空间偏窄。
- 后续脑-行为与网络分析过度围绕 `post - pre similarity` 下降展开，容易只重复主效应，而不是发现新的机制。
- 当前故事可以支持“隐喻学习诱发表征重组/分化”，但还不足以强写“发现了隐喻学习的计算机制”。

## 1. 推荐总路线

主攻方案：

> 关系向量几何（Relational Vector Geometry）

核心主张从：

> 学习后两个词是否更像？

升级为：

> 大脑是否学习了从 cue 到 target 的关系方向？

PNAS 目标叙事：

> Creative relational learning reshapes the neural geometry of semantic memory.

中文表述：

> 隐喻联结学习不是简单地让两个概念变得更相似，而是让人脑形成可记忆、可区分的关系向量表征；这种关系几何在语义控制与组合语义网络中形成，并与最终记忆表现相关。

这一路线的优势：

- 直接贴合当前范式：每个 YY/KJ trial 本来就是一个 cue-target pair。
- 能自然解释当前 similarity 下降：下降不再被解释成“pair 没学成”，而是“学习后两个词在神经空间中获得更明确的方向性关系，导致局部表征分化”。
- 新增工程量可控：可复用现有 `pattern_root`、ROI masks、`model_rdm_comparison.py`、`brain_behavior` 与 learning dynamics 框架。
- 比“语义认知地图”方案更高效，不需要立刻重建完整 manifold / graph / community 分析。

## 2. 分析包结构

推荐把下一轮分析拆成三个相互支撑的核心包，并把师兄旧探索中可参考的方法收敛为两个 legacy-informed 补充包：

| 包 | 目标 | 论文角色 |
| --- | --- | --- |
| A. Relation-vector Model-RSA | 检验神经 RDM 是否编码 cue-to-target 关系方向 | 主机制 |
| B. Relation geometry -> behavior | 检验关系几何是否预测最终记忆或 YY-KJ 行为差异 | 脑-行为桥接 |
| C. Learning-stage trajectory | 检验关系几何是否在学习阶段逐步形成 | 过程证据 |
| D. Cross-stage reinstatement / RDM stability | 吸收师兄 `cross_similarity` / `cross_RDM` 思路，检验学习期表征是否延续到 post/retrieval | 过程与稳健性补充 |
| E. Legacy network confirmation | 吸收师兄 whole-brain gPPI / beta-series connectivity 的强线索，聚焦左海马-PCC/Precuneus/AG-IFG 网络 | 探索性网络补充 |

原始最小可行版本设想为完成 A + B；若 C 也支持，PNAS 叙事会明显增强。基于 2026-05-02 的实际执行结果，A 已经提供新增机制线索，但 B/C 未形成强支持，因此当前优先级已调整为 A+ 深挖。

### 2.1 2026-05-02 执行后修订判断

A0/A1/A2 已完成后，relation-vector Model-RSA 出现多处 primary-family FDR 显著结果，但方向主要为 `post - pre < 0`，因此当前更准确的机制表述是：

> relation-vector realignment / decoupling

而不是：

> relation-vector alignment enhancement

B1/B2 的 subject-level 脑-行为桥接目前没有严格通过校正；C1/C2 的 run3-run4 learning trajectory 也没有提供稳定轨迹证据。因此下一步优先级应从原来的 “A -> B -> C” 调整为：

```text
A 已成立但需解释清楚
-> A+ 深挖 YY-KJ 特异性与网络分工
-> 只有 A+ 支持后，再决定是否继续做 pair-level B
```

换句话说，现在最值得推进的不是继续扩展 B/C，而是沿 A 下挖：

- YY 相对 KJ 的 relation decoupling 是否更强？
- 哪些 ROI 同时表现出 Step5C pair differentiation 和 A relation decoupling？
- relation decoupling 是否呈现 semantic/metaphor network 与 spatial/context network 的分工？
- hippocampus 是否作为 relation binding / memory-binding 节点参与 YY 的 learned-edge differentiation 与 retrieval reinstatement？当前 `literature_spatial` 已含左右 hippocampus，应优先单独汇总，而不是先盲目新增 mask。
- 师兄旧结果中较好的 left hippocampus gPPI 与 cross-stage similarity 是否能在当前 QC/FDR 口径下作为 SI 机制补充，而不是另起一条主线？

## 3. 分析 A：Relation-vector Model-RSA

### 3.1 理论问题

当前 Model-RSA 的 `M3_embedding` 问的是“神经表征是否像静态语义距离”，`M8_reverse_pair` 问的是“是否出现 pair 分化结构”。它们还没有直接刻画“隐喻关系本身”。

Relation-vector 模型要问：

> 每个 pair 的 cue-to-target 语义转换方向，是否成为学习后神经表征的一部分？

### 3.2 模型定义

对每个 pair：

```text
relation_vector_i = embedding(target_i) - embedding(cue_i)
```

构建 pair × pair 模型 RDM：

```text
M_relation_vector[i, j] = 1 - cosine(relation_vector_i, relation_vector_j)
```

可比较的模型版本：

- `M9_relation_vector_direct`：`target - cue`，主模型。
- `M9_relation_vector_reverse`：`cue - target`，方向性对照。
- `M9_relation_vector_abs`：`abs(target - cue)` 或 undirected relation，对照非方向关系。
- `M9_relation_vector_length`：关系向量长度差异，控制“语义距离大小”。
- `M3_embedding_pair_distance`：原始 pair 两词 embedding 距离，控制普通语义距离。

主假设：

- `M9_relation_vector_direct` 在 `YY` post 阶段或 post-pre delta 中优于普通 `M3_embedding`。
- 该效应优先出现在左 IFG、ATL/temporal pole、AG、pMTG 等语义控制/组合语义 ROI。
- `KJ` 可作为对照：若 KJ 更依赖空间/视觉网络，则 relation-vector 效应应较弱或出现在不同 ROI。

### 3.3 代码实现路径

优先新增一个独立脚本，不直接大改现有 `model_rdm_comparison.py`：

```text
rsa_analysis/relation_vector_rsa.py
```

输入：

- `paper_outputs/qc/model_rdm_results_<roi_set>/model_rdm_subject_metrics.tsv` 可作为格式参考。
- `pattern_root/sub-xx/{pre,post}_{yy,kj}.nii.gz`
- `pattern_root/sub-xx/{pre,post}_{yy,kj}_metadata.tsv`
- `rsa_analysis/build_stimulus_embeddings.py` 生成的 embedding 表，或已有 embedding TSV。
- ROI masks 来自 `roi_library/manifest.tsv`。

输出：

```text
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_subject_metrics.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_group_summary.tsv
paper_outputs/tables_main/table_relation_vector_rsa.tsv
paper_outputs/tables_si/table_relation_vector_rsa_fdr.tsv
paper_outputs/figures_main/fig_relation_vector_rsa.png
```

### 3.4 最小统计模型

Subject-level 指标：

```text
rho_neural_relation = Spearman(neural_RDM, relation_vector_RDM)
delta_rho = rho_post - rho_pre
```

组水平：

```text
delta_rho ~ 1
```

并按 ROI set / ROI / condition / model 做 FDR：

- primary family：`M9_relation_vector_direct`, `M9_relation_vector_abs`
- control family：`M3_embedding_pair_distance`, `M9_relation_vector_length`
- secondary family：reverse direction 与其他探索模型

主文只报告 primary family。

### 3.5 决策阈值

可以进入主机制结果的标准：

- 至少一套独立 ROI set 中，`M9_relation_vector_direct` 通过 primary-family FDR。
- 效应方向可解释：若 post 或 delta 中 relation alignment 增强，可写作通用语义关系编码增强；若 post-pre 下降，则写作 relation-vector realignment / decoupling，即学习后神经关系几何从通用 embedding 结构中脱耦。
- 结果不是单一 ROI 偶然显著，最好在语义控制/组合语义网络中成簇出现。

如果只出现未校正趋势：

- 可写作“relation-vector exploratory evidence”，但不能作为 PNAS 主机制。

## 4. 分析 B：Relation geometry 与行为

### 4.1 理论问题

当前 item-level 脑-行为主要问：

> similarity 下降是否预测 memory？

下一步应改为：

> 关系几何是否预测最终记忆？

也就是说，不再把 `delta_similarity` 当作唯一神经指标，而是引入 `relation_alignment`。

### 4.2 可检验指标

Subject-level：

```text
relation_alignment_pre
relation_alignment_post
relation_alignment_delta = post - pre
```

Item/pair-level：

```text
pair_relation_fit_pre
pair_relation_fit_post
pair_relation_fit_delta
```

行为指标：

- `run7_memory_accuracy`
- `run7_correct_retrieval_efficiency`
- `YY - KJ` 行为差异

### 4.3 推荐模型

Subject-level 最小模型：

```text
YY_minus_KJ_behavior ~ YY_minus_KJ_relation_alignment
```

或：

```text
behavior_value ~ relation_alignment + condition
```

Item-level 增强模型：

```text
memory ~ relation_fit_pre
       + relation_fit_post
       + relation_fit_delta
       + condition
       + relation_fit_post:condition
```

若使用 GEE：

```text
groups = subject
family = binomial
```

若使用混合模型：

```text
memory ~ relation_fit * condition + (1|subject) + (1|pair_id)
```

### 4.4 代码实现路径

新增：

```text
brain_behavior/relation_behavior_prediction.py
```

复用：

- `brain_behavior/itemlevel_brain_behavior.py` 的行为 trial join 逻辑。
- `brain_behavior/mechanism_behavior_prediction.py` 的 subject-level 行为汇总和 FDR 逻辑。

输出：

```text
paper_outputs/qc/relation_behavior_long.tsv
paper_outputs/tables_main/table_relation_behavior_prediction.tsv
paper_outputs/tables_si/table_relation_behavior_prediction_fdr.tsv
paper_outputs/figures_main/fig_relation_behavior_prediction.png
```

### 4.5 决策阈值

强结果：

- `relation_alignment_post` 或 `relation_alignment_delta` 能预测 `YY - KJ` memory / efficiency，且通过 planned family FDR。

中等结果：

- 只在语义 ROI 或某个行为指标上未校正显著，可作为机制线索。

阴性结果：

- 仍可保留主 RSA 机制，但不能写“关系几何解释行为差异”。

## 4A. 分析 A+：Relation-vector 深挖与 YY-KJ 特异性定位

### 4A.1 新问题

A0/A1/A2 已经证明 relation-vector geometry 在多个 ROI 中发生显著 pre-to-post 变化，但当前最强结果大多表现为 alignment 下降，且 `YY` 与 `KJ` 都有显著项。因此下一轮不应继续只问：

> 哪些 ROI 的 relation-vector alignment 下降？

而应改问：

> 哪些 ROI 中，YY 相对 KJ 的 relation geometry realignment / decoupling 更强？哪些 ROI 同时表现出 pair 内分化和 relation-level 重组？

这个问题能把 A 从“关系学习的一般机制”推进到“隐喻学习相对空间学习的特异机制”。

### 4A.2 A3：YY-KJ relation decoupling contrast

核心指标：

```text
relation_delta = post_alignment - pre_alignment
relation_decoupling = pre_alignment - post_alignment
YY_minus_KJ_decoupling = decoupling_YY - decoupling_KJ
```

统计单位为 `subject × ROI × model`。组水平检验：

```text
YY_minus_KJ_decoupling ~ 1
```

主检验模型：

- `M9_relation_vector_direct`
- `M9_relation_vector_abs`

主要输出：

```text
paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_subject.tsv
paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_group_fdr.tsv
paper_outputs/tables_main/table_relation_vector_condition_contrast.tsv
paper_outputs/figures_main/fig_relation_vector_condition_contrast.png
```

解释标准：

- 若 `YY_minus_KJ_decoupling > 0` 且通过 primary-family FDR，说明该 ROI 的 relation geometry 重组更偏隐喻学习。
- 若不显著但多个语义 ROI 同向，可作为机制趋势。
- 若 KJ 更强，说明 relation-vector realignment 主要反映一般关系学习或空间关系学习，不宜写成 YY 特异。

### 4A.3 A4：A × Step5C conjunction

单独 A 可能是关系学习通用机制；Step5C 是 YY 特异 pair similarity 下降。最有价值的证据是二者汇合：

```text
Step5C: YY pair differentiation > KJ
A:      YY relation decoupling > KJ
```

构建 ROI-level conjunction table：

```text
pair_differentiation_score = pre_post_similarity_drop_YY - pre_post_similarity_drop_KJ
relation_decoupling_score = relation_decoupling_YY - relation_decoupling_KJ
```

输出：

```text
paper_outputs/qc/relation_vector_conjunction/relation_step5c_conjunction.tsv
paper_outputs/tables_main/table_relation_step5c_conjunction.tsv
paper_outputs/figures_main/fig_relation_step5c_conjunction.png
```

主文价值：

> 找到既发生 pair 内分化、又发生 relation-level 重组的 ROI。

如果这些 ROI 集中在 temporal pole、pMTG、IFG、AG/pSTS 或 OPA/PPA/RSC/PPC，将比单独 A 或单独 Step5C 更有机制解释力。

### 4A.4 A5：Network-level dissociation

把 ROI 按理论网络聚合，而不是只看单 ROI：

```text
semantic/metaphor network: IFG, temporal pole/ATL, pMTG, AG, pSTS
hippocampal-binding network: hippocampus, parahippocampal/PPA, RSC/PCC, precuneus
spatial/context network: OPA, PPA, RSC, PPC, precuneus
main functional families: Metaphor_gt_Spatial, Spatial_gt_Metaphor
```

推荐模型：

```text
relation_decoupling ~ condition * network + (1|subject) + (1|ROI)
```

重点检验：

- `YY > KJ` 是否集中在 semantic/metaphor network。
- `KJ > YY` 是否集中在 spatial/context network。
- hippocampal-binding network 是否更像 YY 的 relation-binding / retrieval 支持系统，而不是单纯空间网络。
- main functional 的 `Metaphor_gt_Spatial` 与 `Spatial_gt_Metaphor` family 是否呈现可解释分工。

输出：

```text
paper_outputs/qc/relation_vector_network/relation_vector_network_lmm_params.tsv
paper_outputs/tables_main/table_relation_vector_network_dissociation.tsv
paper_outputs/figures_main/fig_relation_vector_network_dissociation.png
```

### 4A.5 A+ 决策标准

强结果：

- A3 中至少一个理论 ROI 或一个 network-level contrast 通过 primary-family FDR；
- A4 中存在 ROI 同时满足 Step5C pair differentiation 和 A relation decoupling；
- A5 显示语义网络与空间网络存在 condition × network 分工。

中等结果：

- A3/A5 未校正显著但方向集中；
- A4 conjunction 排名靠前 ROI 与理论一致。

弱结果：

- YY-KJ contrast 无方向；
- A 与 Step5C 不重合；
- network model 无 condition × network 结构。

若 A+ 为强或中等，论文主线可写为：

> 隐喻学习具有 pair-level differentiation 的特异效应，并在语义/空间网络中伴随 relation geometry realignment；KJ 的 relation-vector 变化则说明这一机制部分属于更一般的关系学习。

若 A+ 为弱，则 A 阶段应降级为：

> relation-vector realignment 是学习任务的一般结果，不能独立支撑隐喻特异机制。

## 4B. 分析 A++：边特异性、baseline、单词稳定性与回忆阶段扩展

老师新增想法中，最应该优先整合的是能直接回答 Step5C / A 结果解释漏洞的问题：当前结果不能只说明“组成过 pair 的两个词 post-pre 相似性下降”，还要说明这种下降不是全局噪声、不是所有词都变散、不是 baseline 也同样发生，而是与学习建立过的关系边有关。

因此 A++ 的核心问题改为：

> 学习后表征分化是否具有 learned-edge specificity：训练过的 YY pair 相比未连接词、baseline pseudo-pair 和 KJ pair，是否发生更强、更可解释的关系边重组？

### 4B.1 P0：trained edge vs untrained non-edge

新增比较不只看原始词对，还要看每个词与“没有被连接过的其他词”的相似性变化。最小模型：

```text
similarity_change ~ edge_status * condition + ROI/network + subject
edge_status = trained_edge / untrained_nonedge
```

关键解释：
- 如果 `trained_edge` 下降，而 `untrained_nonedge` 不下降或上升，说明结果更像 learned relation reorganization。
- 如果 trained 与 untrained 都下降，说明可能是全局表征漂移、注意/噪声或阶段差异。
- 如果 untrained 上升但 trained 下降，证据最强：学习让被绑定的关系边从一般语义邻域中分化出来。

计划输出：

```text
paper_outputs/qc/edge_specificity/edge_similarity_long.tsv
paper_outputs/qc/edge_specificity/edge_specificity_group_fdr.tsv
paper_outputs/tables_main/table_edge_specificity.tsv
paper_outputs/figures_main/fig_edge_specificity.png
```

### 4B.2 P0：baseline pseudo-edge

baseline 不能只作为“没有 YY/KJ 学习”的背景条件，也要按之前 `jx1-jx2` 这类固定组合方式构建 pseudo-pair。这样可以检验：

```text
YY trained_edge drop > KJ trained_edge drop > baseline pseudo_edge drop
```

或更保守地检验：

```text
YY trained_edge drop > baseline pseudo_edge drop
```

解释标准：
- baseline pseudo-edge 不显著下降：支持学习关系边特异性。
- baseline 也下降：主文不能把 pair drop 写成学习特异效应，只能写成任务阶段或测量阶段的共同变化。

### 4B.3 P0：single-word stability

之前 run1/2/5/6 主要看 pair similarity 的 post-pre 变化，还需要看单个词自身表征是否稳定：

```text
same_word_stability = corr(pattern_word_pre, pattern_word_post)
word_neighborhood_change = mean(similarity(word, all_other_words))_post
                         - mean(similarity(word, all_other_words))_pre
```

最有力的模式是：
- same-word stability 保持稳定或较高；
- trained pair similarity 下降；
- word-to-untrained-neighbors 不同步下降。

这可以把故事从“表征变差/噪声变大”改写为“词身份仍可保留，但学习过的关系边发生重组”。

计划输出：

```text
paper_outputs/qc/word_stability/word_stability_long.tsv
paper_outputs/qc/word_stability/word_neighborhood_change.tsv
paper_outputs/tables_main/table_word_stability.tsv
paper_outputs/figures_main/fig_word_stability.png
```

### 4B.4 P1：run7 回忆阶段几何

run7 回忆阶段展示了 YY/KJ 词，可以作为第三时间点，不只看 pre/post：

```text
pre -> post -> retrieval
```

优先问题：
- 学习后的 pair differentiation 在 retrieval 阶段是否保留、恢复或进一步增强？
- retrieval 阶段成功回忆 trial 是否有更强 cue-target reinstatement？
- post-to-retrieval stability 是否预测最终回忆成功？

计划输出：

```text
paper_outputs/qc/retrieval_geometry/retrieval_pair_similarity_long.tsv
paper_outputs/qc/retrieval_prediction/retrieval_prediction_group_fdr.tsv
paper_outputs/tables_main/table_retrieval_geometry.tsv
paper_outputs/tables_main/table_retrieval_prediction.tsv
paper_outputs/figures_main/fig_retrieval_geometry.png
paper_outputs/figures_main/fig_retrieval_prediction.png
```

### 4B.5 P1：学习阶段 driver 与中介/调节

“学习阶段哪些机制让词相似性变差”可以做，但不建议一开始就直接上中介模型。更稳妥的路径是先筛选 driver，再决定是否做 mediation/moderation。

第一步 driver screening：

```text
post_pre_pair_differentiation ~ learning_stage_metric * condition
```

候选 `learning_stage_metric`：
- run3/run4 隐喻句子激活强度或 YY-KJ 激活差；
- learning-stage pair/relation alignment slope；
- repeated-exposure similarity change；
- run3/run4 中 cue-target 或 pair-pattern 的稳定性；
- 学习阶段行为反应时/正确率/置信度变化。

第二步才做中介或调节：

```text
learning_metric -> pair_differentiation -> run7_memory
```

或：

```text
pair_differentiation -> run7_memory
moderated by learning_metric / condition
```

验收标准：只有当 `learning_metric -> pair_differentiation` 和 `pair_differentiation -> memory` 至少有方向一致的证据时，才把中介写入主文；否则作为探索性补充。

### 4B.6 P2：run3/run4 激活特异性与 ROI 覆盖审计

run3/run4 激活分析更适合作为上游机制变量，而不是单独再开一条主线。推荐先回答：

> 哪些 ROI 在学习阶段对隐喻句子表现出 YY 特异激活，并且这种激活是否预测后续 pair differentiation / relation decoupling？

ROI 覆盖则做成审计表，不建议盲目加 ROI。审计表至少包含：

```text
meta-analysis region
current ROI covered?
ROI source
missing / partial / covered
recommendation
```

优先核查区域：
- metaphor/semantic：IFG, ATL/temporal pole, pMTG, AG, pSTS, dmPFC/medial frontal。
- spatial/context：OPA, PPA, RSC, PPC, hippocampus, precuneus/PCC。

只有当 meta-analysis 明确支持、当前 ROI 未覆盖、且该区域与理论主线有关时，才新增 exploratory ROI set。

## 4C. 师兄 legacy 方法的可参考融合

师兄旧仓库里真正值得融合的不是旧 MATLAB/SPM 代码本身，而是三个可转换成 `final_version` 口径的问题：跨阶段表征是否延续、学习期网络是否支持关系重组、这些网络线索是否与最终记忆相关。旧结果作为 hypothesis generator，进入新版后必须重新使用统一的 `pattern_root`、ROI manifest、subject-level QC、planned family/FDR 或 permutation。

### 4C.1 P1：cross-stage same-pair reinstatement

对应师兄旧线索：
- `H:\metaphor\cross_similarity\new_measure_dsm_corr.m`
- `H:\metaphor\LSS\words_with_juzi\run3_with_run56`
- `H:\metaphor\LSS\words_with_juzi\run3_with_run7`

旧方法核心是把同一 pair/item 在两个阶段的 pattern 做相关，例如 run3 vs run4、run3 vs post、run3 vs run7。新版应改成同一套 pair-discrimination 口径，而不是只算 raw same-pair similarity：

```text
same_pair_cross_stage_similarity = sim(stage_a_i, stage_b_i)
different_pair_cross_stage_similarity = mean(sim(stage_a_i, stage_b_j), j != i)
cross_stage_reinstatement = same_pair - different_pair
```

优先比较：
- `learn_run3 -> learn_run4`：学习过程中是否形成稳定 pair/句子表征。
- `learn_run3/4 -> post`：学习期表征是否迁移到后测词表征。
- `post -> retrieval/run7`：后测几何是否在回忆阶段保留。

计划输出：

```text
paper_outputs/qc/cross_stage_reinstatement/cross_stage_reinstatement_long.tsv
paper_outputs/qc/cross_stage_reinstatement/cross_stage_reinstatement_group_fdr.tsv
paper_outputs/tables_si/table_cross_stage_reinstatement.tsv
paper_outputs/figures_si/fig_cross_stage_reinstatement.png
```

解释标准：
- 如果 YY 的 `learn -> post/retrieval` reinstatement 高于 KJ，且与 Step5C/relation-vector 结果同向，可作为“学习期关系表征延续”的过程证据。
- 如果仅 run3 -> run4 显著，而不延续到 post/retrieval，只能说明 repeated exposure / 学习期稳定性。
- 如果不显著，不影响 Step5C 主线。

### 4C.2 P1：cross-stage RDM stability / realignment

对应师兄旧线索：
- `H:\metaphor\cross_RDM\new_measure_dsm_corr.m`
- `H:\metaphor\cross_RDM\rca_crossstage.m`

旧方法看的是两个阶段的整体 RDM 是否相似。新版可把它变成 relation-vector 的互补分析：学习后不是简单“两个词更像/更不像”，而是整个关系结构是否从学习期迁移、保持或重排。

核心指标：

```text
rdm_stability = corr(vectorize(RDM_stage_a), vectorize(RDM_stage_b))
rdm_realignment = rdm_stability_post_related - rdm_stability_pre_related
```

优先比较：
- `RDM_run3` vs `RDM_run4`
- `RDM_learn` vs `RDM_post`
- `RDM_pre` vs `RDM_post`

计划输出：

```text
paper_outputs/qc/cross_stage_rdm/cross_stage_rdm_subject.tsv
paper_outputs/qc/cross_stage_rdm/cross_stage_rdm_group_fdr.tsv
paper_outputs/tables_si/table_cross_stage_rdm_stability.tsv
paper_outputs/figures_si/fig_cross_stage_rdm_stability.png
```

决策阈值：
- 只作为 SI 或机制补充，除非 YY 在独立 ROI set 中出现稳定、校正后、方向可解释的 `learn -> post` RDM continuity。
- 若结果与 relation-vector decoupling 相冲突，优先解释为不同层级指标：global RDM continuity 与 local relation-vector realignment 可以并存。

### 4C.3 P1：hippocampal-binding ROI 与 legacy whole-brain gPPI / beta-series connectivity confirmation

对应师兄旧线索：
- `H:\metaphor\gppi\lHPC_seed\scripts\run3.m`
- `H:\metaphor\gppi\output_run3\second_lhip\yy-kj\cluster_fwe_001_peaks.csv`
- `H:\metaphor\gppi\output_run4\second_lhip\kj-yy\cluster_FWE_peaks.csv`
- `H:\metaphor\BSC\seed_to_voxel_correlation beta_run34.ipynb`
- `H:\metaphor\BSC\F_run3\fwe_data_peaks.csv`

师兄最值得保留的网络假设是：学习阶段 `YY > KJ` 时，左海马与 PCC/Precuneus 的 gPPI 耦合增强；旧结果中 run3 left-HPC seed 的主要峰包括 PCC `[6, -38, 6]` 与 Precuneus `[2, -50, 52]`。这条线与“关系学习 / 情景记忆 / 默认网络参与隐喻联结”叙事吻合。由于当前 `literature_spatial` 已经包含 `litspat_L_hippocampus` 和 `litspat_R_hippocampus`，下一步优先不是新增海马 mask，而是把海马从 spatial/context ROI 中单独提升为 hippocampal-binding family 汇总；whole-brain legacy gPPI 仍作为 exploratory confirmation，原因是旧 pipeline seed、contrast、reverse seed 与阈值尝试较多。

新版优先做两层：

```text
P1a ROI-level hippocampal-binding summary:
  ROIs: litspat_L_hippocampus, litspat_R_hippocampus, PHG/PPA, RSC/PCC, precuneus
  analyses: Step5C, edge specificity, word stability, relation-vector A/A+, retrieval geometry

P1b legacy-confirmation gPPI:
  seed: left hippocampus
  targets/scopes: PCC/Precuneus, AG/MTG, IFG/MFG, hippocampal-context network
```

新版不建议复刻全部旧 gPPI，而应做一个 planned legacy-confirmation：

```text
seed: left hippocampus
stage: learn/run3 or learn run3+run4
contrast: PPI_yy - PPI_kj
targets/scopes: PCC/Precuneus, AG/MTG, IFG/MFG, hippocampal-context network
```

计划输出：

```text
paper_outputs/qc/legacy_gppi/legacy_gppi_subject.tsv
paper_outputs/qc/legacy_gppi/legacy_gppi_group_fdr.tsv
paper_outputs/qc/legacy_gppi/legacy_gppi_wholebrain_peaks.tsv
paper_outputs/tables_si/table_legacy_gppi_lhip.tsv
paper_outputs/figures_si/fig_legacy_gppi_lhip.png
```

解释标准：
- 若 hippocampus 在 A++ 或 retrieval geometry 中显示 YY learned-edge / reinstatement 相关效应，可作为 relation binding 的重要机制节点，即使 gPPI 不显著也有报告价值。
- 若 left-HPC -> PCC/Precuneus 的 `YY > KJ` 通过 planned target family，作为 SI 网络证据支持“学习期海马-默认网络协作”。
- 若只复现未校正趋势，写作 legacy exploratory network clue。
- 若不复现，不影响主文；说明当前更严格 ROI/FDR 口径下网络证据不足。

### 4C.4 暂不融合的旧网络做法

以下旧做法暂不进入新版主线：
- 从显著 gPPI 簇反向做 seed 的 reverse PPI：存在 circularity 风险。
- 多 seed、多阶段、多阈值的 whole-brain 搜索：适合内部探索，不适合主文。
- 旧 BSC top-k 或 seed-to-voxel beta correlation 直接当主网络结果：可作为 beta-series connectivity sensitivity，但必须重新命名、重新校正。
- 旧 `selfreportmemory` / `true_memory` 的简单 ROI 相关：新版已有 refined behavior 与 GEE/LMM，旧行为相关只作为 sensitivity。

## 5. 分析 C：学习阶段 relation trajectory

### 5.1 理论问题

PNAS 级别故事最好不只回答“学习前后变了”，还要回答：

> 这种关系几何是在学习过程中如何形成的？

当前已有 learning dynamics 图，但口径仍偏 repeated-exposure similarity。下一步应追踪 relation alignment。

### 5.2 数据需求

优先检查是否已有：

```text
pattern_root/sub-xx/learn_yy.nii.gz
pattern_root/sub-xx/learn_kj.nii.gz
pattern_root/sub-xx/learn_yy_metadata.tsv
pattern_root/sub-xx/learn_kj_metadata.tsv
```

若有 exposure / run / trial_order：

- 可做 exposure-level trajectory。

若没有细粒度 exposure：

- 最小版本做 run3 vs run4 或 early vs late。

### 5.3 指标

```text
relation_alignment_early
relation_alignment_late
trajectory_slope = late - early
```

主假设：

- YY 的 relation alignment 在学习阶段逐步增强。
- 学习阶段 relation slope 预测 post relation alignment 或 run7 memory。

### 5.4 代码实现路径

新增：

```text
temporal_dynamics/relation_learning_dynamics.py
```

复用：

- `temporal_dynamics/learning_dynamics.py` 的 metadata / window 处理。
- `rsa_analysis/relation_vector_rsa.py` 的 relation RDM 构建函数。

输出：

```text
paper_outputs/qc/relation_learning_dynamics_long.tsv
paper_outputs/tables_main/table_relation_learning_dynamics.tsv
paper_outputs/tables_si/table_relation_learning_dynamics_subject.tsv
paper_outputs/figures_main/fig_relation_learning_dynamics.png
```

### 5.5 决策阈值

强结果：

- YY relation alignment 随学习显著增强，并预测 Run7 行为。

中等结果：

- 只显示方向性 trajectory，可作为过程补充。

阴性结果：

- 不影响 pre/post 主机制；C 包降级为 SI 或不写。

## 6. 暂缓但保留的扩展

以下方向有价值，但不作为当前最小冲刺路线：

### 6.1 语义认知地图 / Conceptual Map

优点是 PNAS 味道强，缺点是工程量大，且当前实验没有显式 inference/generalization 测试。可作为背景框架使用，但暂不做完整 graph/manifold/community 分析。

可后续扩展：

- semantic network distance
- local density
- community bridging
- cognitive-map graph embedding

### 6.2 Schema Model-RSA

可以给 YY/KJ pair 人工标注关系 schema，例如：

- 情绪-具象
- 人格特质-物体属性
- 功能关系
- 空间/路径关系
- 社会关系
- 感知属性映射

但这需要额外评分和一致性检验。若 relation-vector 结果不强，可以作为第二轮机制补充。

### 6.3 SSDD / 感知运动语义维度

中文 SSDD 等数据库可提供视觉、运动、社会性、情绪、时间、空间维度。该方向适合解释 YY/KJ 的语义特征差异，但会引入较多材料覆盖和匹配问题。暂列为第二优先级。

## 7. 推荐执行顺序

当前最优路径不是继续平均推进 B/C/D，而是先把 A 的解释力补强。执行优先级如下：

### 第 1 阶段：A0-A2 已完成结果的定位

- [x] relation-vector Model-RSA 已经提供新增机制线索。
- [x] 当前更合适的解释是 `relation-vector realignment / decoupling`，不是 alignment enhancement。
- [ ] 在 `result_new.md` 中明确：A 是机制线索，仍需 A++ 证明它不是全局相似性下降。

### 第 2 阶段：P0，先做 learned-edge specificity

- [ ] 新增 `rsa_analysis/edge_specificity_rsa.py`。
- [ ] 比较 `trained_edge`、`untrained_nonedge` 与 `baseline_pseudo_edge`。
- [ ] 明确 YY 的 pair drop 是否强于 KJ 与 baseline。
- [ ] 输出 `table_edge_specificity.tsv` 与 `fig_edge_specificity.png`。
- [ ] 若 trained edge 下降、untrained non-edge 不下降或上升，把 Step5C 解释升级为“learned relation edge differentiation”。

### 第 3 阶段：P0，补 single-word stability

- [ ] 新增 `rsa_analysis/word_stability_analysis.py`。
- [ ] 计算 same-word pre/post stability。
- [ ] 计算 word-to-all-neighbors 的局部邻域变化。
- [ ] 判断 pair similarity 下降是否能与单词身份稳定共存。
- [ ] 若成立，排除“表征整体变差/噪声变大”的弱解释。

### 第 4 阶段：A+ 深挖 ROI 与网络分工

- [ ] 新增 `rsa_analysis/relation_vector_condition_contrast.py`。
- [ ] 计算 `YY_minus_KJ_decoupling`，定位 YY 相对 KJ 更强的 relation realignment ROI。
- [ ] 新增 `rsa_analysis/relation_step5c_conjunction.py`。
- [ ] 汇合 Step5C pair differentiation 与 relation-vector decoupling，找共同机制 ROI。
- [ ] 新增 `rsa_analysis/relation_vector_network_lmm.py`。
- [ ] 检验 semantic/metaphor network 与 spatial/context network 的 condition × network 分工。

### 第 5 阶段：P1，run7 回忆阶段几何与预测

- [ ] 新增 `rsa_analysis/retrieval_pair_similarity.py`。
- [ ] 比较 pre、post、retrieval/run7 的 pair similarity trajectory。
- [ ] 新增 `brain_behavior/retrieval_success_prediction.py`。
- [ ] 检验 retrieval-stage ROI pattern、cue-target reinstatement 或 post-to-retrieval stability 是否预测回忆成功。

### 第 6 阶段：P1，学习阶段 driver 与中介/调节筛选

- [ ] 新增 `temporal_dynamics/learning_driver_analysis.py`。
- [ ] 先检验 learning-stage metric 是否预测 post-pre pair differentiation。
- [ ] 若 driver path 成立，再做 mediation/moderation。
- [ ] 若不成立，只把学习阶段结果作为过程性补充，不强写因果链。

### 第 7 阶段：P1，融合师兄 cross-stage 表征方法

- [ ] 新增 `temporal_dynamics/cross_stage_reinstatement.py`，把师兄 `cross_similarity` 转成 same-vs-different pair reinstatement。
- [ ] 新增 `rsa_analysis/cross_stage_rdm_stability.py`，把师兄 `cross_RDM` 转成阶段间 RDM stability / realignment。
- [ ] 优先检验 `learn -> post` 与 `post -> retrieval`，不要只停在 run3 -> run4。
- [ ] 若 YY 的跨阶段延续强于 KJ，作为过程证据；若阴性，降级为 SI 边界结果。

### 第 8 阶段：P1，hippocampal-binding family 与 legacy network confirmation

- [ ] 不先新增海马 mask；先确认并单独汇总现有 `litspat_L_hippocampus` / `litspat_R_hippocampus`。
- [ ] 在 Step5C、A++ edge specificity、word stability、relation-vector A+ 与 retrieval geometry 中增加 hippocampal-binding family 输出。
- [ ] 新增 `connectivity_analysis/legacy_lhip_gppi_confirmation.py` 或扩展现有 `gPPI_analysis.py`。
- [ ] 以 left hippocampus 为 planned seed，检验 `PPI_yy - PPI_kj`。
- [ ] planned target family 包含 PCC/Precuneus、AG/MTG、IFG/MFG 与 hippocampal-context network。
- [ ] 只把可复现、校正后或 planned family 支持的结果写入 SI；旧 whole-brain 峰只作为 hypothesis generator。

### 第 9 阶段：P2，激活特异性与 ROI 覆盖审计

- [ ] 用 run3/run4 激活确认隐喻句子特异性，并作为 driver 候选。
- [ ] 新增 `roi_analysis/roi_coverage_audit.py`。
- [ ] 对照 meta-analysis region 检查当前 metaphor/spatial ROI 是否覆盖充分。
- [ ] 只在理论必要且元分析支持时新增 exploratory ROI。

### 第 10 阶段：论文重写

- [ ] 更新 `result_new.md`，新增 A++ 结果节。
- [ ] 更新一页概览，把“相似性下降”改写为“learned-edge differentiation / relation geometry realignment”。
- [ ] 准备给老师汇报的两版路线：
  - 稳妥版：Step 5C RSA 主结果 + A++ learned-edge specificity。
  - PNAS 冲刺版：learned-edge specificity + relation-vector realignment + cross-stage/retrieval evidence + legacy-confirmed hippocampal network support。

## 8. 推荐图表

如果 relation-vector 路线成立，主文图可以重排为：

### Figure 1：任务与核心行为

- YY/KJ 学习范式。
- Run7 memory 与 RT。

### Figure 2：学习后神经表征重组

- Step 5C ROI-level RSA 主效应。
- 三套 ROI robustness。

### Figure 3：Relation-vector Model-RSA

- 关系向量模型示意。
- pre/post 或 delta relation alignment。
- ROI/network 分布。

### Figure 4：Relation geometry predicts memory

- subject-level 或 item-level relation alignment 与行为。
- 若只有趋势，降为 SI。

### Figure 5：Learning-stage trajectory

- early-late relation alignment。
- trajectory 到 post geometry / behavior 的预测。

SI：

- baseline control
- old Model-RSA (`M3`, `M8`)
- searchlight
- representational connectivity
- legacy cross-stage reinstatement / RDM stability
- legacy-confirmation gPPI / beta-series connectivity
- RD/GPS
- robustness/noise ceiling

## 9. 预期结果与解释分叉

### 情况 A：Relation-vector 结果强

论文主张升级为：

> 隐喻联结学习通过形成方向性关系几何来重塑语义记忆。

这时可以认真准备 PNAS 冲刺。

### 情况 B：Relation-vector 结果中等，行为桥接弱

论文主张为：

> 隐喻学习诱发表征分化，并伴随方向性关系几何的局部机制线索。

目标更现实地落在 NeuroImage / Cerebral Cortex / Communications Biology。

### 情况 C：Relation-vector 结果弱

保留当前 Step 5C 主线：

> 隐喻学习诱发表征分化/重组，但当前模型尚不足以解释其具体计算机制。

下一步转向 schema 或 SSDD 语义维度模型。

## 10. 给老师汇报时的建议表述

可以这样汇报：

> 当前主结果已经稳定：隐喻学习相比空间联结学习引发更强的表征重组，表现为多个 ROI 中 pair similarity 的学习后下降。这个结果不支持简单的 pair convergence，而更像是 representational differentiation。为了冲击 PNAS，我们下一步不再继续围绕 similarity 下降做重复分析，而是引入 relation-vector geometry，检验大脑是否学习了 cue-to-target 的方向性关系表征。如果该关系几何能解释最终记忆或学习阶段轨迹，论文可以从“隐喻学习改变 RSA”升级为“创造性关系学习重塑语义记忆的神经几何”。

## 11. 当前不需要修改的事项

根据最新讨论，以下 review findings 暂不作为修改任务：

- `current_roi_set()` 默认值问题：接受当前流程，后续靠命令和日志显式记录 ROI set。
- 聚合脚本依赖 `METAPHOR_ROI_SET`：不作为当前阻断项。
- searchlight mismatch manifest：当前已有配对 QC，失败时不额外优化。
- baseline searchlight：baseline 使用特定匹配 `pair_id`，不新增 `template_pair_id`。

这些事项不影响 relation-vector 主路线。
