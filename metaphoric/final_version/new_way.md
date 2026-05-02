# new_way.md

最后更新：2026-05-02

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

推荐把下一轮分析拆成三个相互支撑的包：

| 包 | 目标 | 论文角色 |
| --- | --- | --- |
| A. Relation-vector Model-RSA | 检验神经 RDM 是否编码 cue-to-target 关系方向 | 主机制 |
| B. Relation geometry -> behavior | 检验关系几何是否预测最终记忆或 YY-KJ 行为差异 | 脑-行为桥接 |
| C. Learning-stage trajectory | 检验关系几何是否在学习阶段逐步形成 | 过程证据 |

最小可行版本只需要完成 A + B。若 C 也支持，PNAS 叙事会明显增强。

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
- 效应方向与理论一致：post 或 delta 中 relation alignment 增强。
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

### 第 1 阶段：最小机制验证

- [ ] 确认 embedding 表覆盖所有 YY/KJ word labels。
- [ ] 新增 `rsa_analysis/relation_vector_rsa.py`。
- [ ] 跑 `main_functional`、`literature`、`literature_spatial` 三套 ROI。
- [ ] 输出 `table_relation_vector_rsa.tsv` 和 FDR 表。
- [ ] 判断 `M9_relation_vector_direct` 是否成为比现有 `M3/M8` 更好的机制模型。

### 第 2 阶段：行为桥接

- [ ] 新增 `brain_behavior/relation_behavior_prediction.py`。
- [ ] 做 subject-level `relation_alignment -> YY-KJ behavior`。
- [ ] 做 item-level `relation_fit -> memory`。
- [ ] 明确结果是否支持“关系几何解释行为差异”。

### 第 3 阶段：学习过程

- [ ] 检查 learning-stage patterns 与 metadata。
- [ ] 新增 `temporal_dynamics/relation_learning_dynamics.py`。
- [ ] 做 early-late 或 exposure-level relation trajectory。
- [ ] 检验 trajectory 是否预测 post relation alignment 或 Run7 memory。

### 第 4 阶段：论文重写

- [ ] 更新 `result_new.md`，新增 relation-vector 机制结果节。
- [ ] 更新一页概览，把 Model-RSA 从 `M8_reverse_pair` 弱机制改为 relation geometry 主机制，若结果支持。
- [ ] 准备给老师汇报的两版路线：
  - 稳妥版：Step 5C RSA 主结果 + relation-vector exploratory mechanism。
  - PNAS 冲刺版：relation-vector 主机制 + behavior bridge + learning trajectory。

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
- gPPI
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
