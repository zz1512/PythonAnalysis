# 当前结果整理：隐喻联结学习的行为与神经表征结果

最后更新：2026-05-03

本文档用于客观汇总当前 `final_version` 的主要分析结果、对应图片和源文件。当前结果按“行为优势 -> ROI 来源 -> 单 ROI 表征变化 -> learned edge 机制 -> RDM 机制解释 -> 行为/全脑/网络补充”的链路排列。所有 ROI 推断以单 ROI 为基本单位；跨 ROI 或 network 汇总只作为描述性稳健性/补充证据，不作为替代单 ROI 统计的主结果。

## 1. 数据与执行状态

当前正式 ROI 范围为三套：`main_functional`、`literature`、`literature_spatial`。本轮已基于最新代码重跑 Step 5C RSA、RSA LMM、Model-RSA、Delta-rho LMM、item-level brain-behavior、S1 机制到行为预测、representational connectivity、robustness suite 和 noise ceiling。新增 A 阶段 relation-vector Model-RSA 已完成 A0/A1/A2：构建 pair-level semantic relation-vector model RDM，并在三套 ROI 中检验 pre/post neural geometry 与该模型的关系。2026-05-03 继续完成 A++ learned-edge specificity、single-word stability、A3 YY-KJ relation decoupling contrast、A4 Step5C/A++ × relation-vector conjunction、A5 network-level dissociation。两个 searchlight 使用 `--reuse-subject-maps`，复用既有 subject-level `.nii.gz`，刷新 group-level permutation/FWE、peak table、QC 和图。gPPI 本轮执行 `export_gppi_summary.py`，基于既有 raw gPPI 结果刷新 summary 表。

`Step 5C RSA` 三套 ROI set 均完成，failed cell 与 incomplete cell 均为 0。

| ROI set | Subjects | ROIs | Item-wise rows | Baseline rows | Failed cells | Incomplete cells | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_functional` | 28 | 7 | 35280 | 7840 | 0 | 0 | 30.92 min |
| `literature` | 28 | 12 | 60480 | 13440 | 0 | 0 | 49.01 min |
| `literature_spatial` | 28 | 12 | 60480 | 13440 | 0 | 0 | 48.37 min |

源文件：
- [step5c_rsa_summary.tsv](E:/python_metaphor/paper_outputs/logs/step5c_rsa_20260501_200250/step5c_rsa_summary.tsv)
- [rerun_after_step5c logs](E:/python_metaphor/paper_outputs/logs/rerun_after_step5c_20260501_234434)
- [result_update_stats.txt](E:/python_metaphor/paper_outputs/logs/rerun_after_step5c_20260501_234434/result_update_stats.txt)

## 2. 行为结果

行为主终点为 run-7 的最终记忆正确率 `memory`，补充终点为正确 trial 的反应时 `log_rt_correct`。行为主分析包含 27 名被试、3780 条 `YY/KJ` run-7 trial。`memory` 使用按被试聚类的 binomial GEE；`log_rt_correct` 使用 Gaussian mixed model。

`YY` 条件最终记忆正确率高于 `KJ`：`YY = 0.680`，`KJ = 0.588`，差值 `+0.092`。GEE 模型中，`YY` 相对 `KJ` 的 log-odds 更高，`b = 0.398`, `SE = 0.084`, `z = 4.72`, `p = 2.37e-06`，odds ratio 约 `1.49`。

正确 trial 反应时也显示 `YY` 更快：`YY = 1070.43 ms`，`KJ = 1100.29 ms`，差值约 `-29.86 ms`。`log_rt_correct ~ C(condition)` 模型中，`YY` 的反应时更短，`b = -0.026`, `SE = 0.010`, `z = -2.61`, `p = .009`。

行为图已直接读取模型结果并标注显著性：memory 为 `***`，correct-trial RT 为 `**`。

![行为正确率和反应时](E:/python_metaphor/paper_outputs/figures_main/fig_behavior_accuracy_rt.png)

源文件：
- [behavior_trials.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv)
- [subject_condition_summary.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_condition_summary.tsv)
- [run7_memory/behavior_lmm_params.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/lmm/run7_memory/behavior_lmm_params.tsv)
- [run7_log_rt_correct/behavior_lmm_params.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/lmm/run7_log_rt_correct/behavior_lmm_params.tsv)
- [fig_behavior_accuracy_rt.png](E:/python_metaphor/paper_outputs/figures_main/fig_behavior_accuracy_rt.png)

## 3. 材料控制结果

材料控制当前只保留既有历史评分结果，作为刺激合理性说明，不进入后续行为或神经模型。历史评分覆盖部分正式刺激：句子评分覆盖 `19/35` 个 YY cue 和 `17/35` 个 KJ cue；词对评分覆盖 `18/35` 个 YY pair、`18/35` 个 KJ pair，并覆盖 `20/20` 个 baseline pair。

在可用子集中，YY/KJ 的可理解度相近（`6.60` vs `6.53`, `p = .665`），难度相近（`1.67` vs `1.61`, `p = .663`），人工熟悉度无显著差异（`4.54` vs `4.85`, `p = .300`）。词对关联度和词对熟悉度在 YY/KJ/baseline 间无显著差异（association `p = .666`；pair familiarity `p = .975`）。YY 句子新颖性高于 KJ（`4.07` vs `2.53`, `p = 6.1e-05`）。

本节当前无单独结果图。

源文件：
- [table_stimulus_control_from_materials.tsv](E:/python_metaphor/paper_outputs/tables_si/table_stimulus_control_from_materials.tsv)
- [table_stimulus_balance_materials_available.tsv](E:/python_metaphor/paper_outputs/tables_si/table_stimulus_balance_materials_available.tsv)
- [stimulus_control_materials_coverage.tsv](E:/python_metaphor/paper_outputs/qc/stimulus_control/stimulus_control_materials_coverage.tsv)

## 4. 学习阶段 GLM 与 ROI 来源

学习阶段 GLM 使用 run-3/4 数据。`Metaphor > Spatial` 的显著簇主要位于左前颞/颞极语义相关区域，包括左侧颞极 MNI `[-52, 6, -24]`，cluster size `1592 mm3`，以及左前颞/颞极邻近区域 MNI `[-38, 12, -36]`，cluster size `392 mm3`。这些簇构成 `main_functional` 中两个 `Metaphor_gt_Spatial` ROI。

`Spatial > Metaphor` 的显著簇主要位于后部视觉、内侧颞叶和顶叶网络，包括左后部梭状回 MNI `[-28, -38, -16]`，cluster size `4056 mm3`；左楔前叶 MNI `[-10, -54, 8]`，cluster size `3648 mm3`；右楔前叶 MNI `[12, -52, 10]`，cluster size `2544 mm3`；右后部海马旁回 MNI `[28, -32, -18]`，cluster size `2440 mm3`；左外侧枕叶上部 MNI `[-38, -78, 30]`，cluster size `2200 mm3`。这些簇构成 `main_functional` 中五个 `Spatial_gt_Metaphor` ROI。

本节当前无单独汇总图；ROI 后续结果图见第 5 节。

## 5. ROI-level RSA 主结果（Step 5C）

Step 5C 使用 pre/post 阶段 LSS beta，在 ROI 内计算 item-wise pattern similarity。对 `YY/KJ`，核心指标是学习 pair 内两个词的 similarity；对 baseline，使用模板中预定义的 `jx` 伪配对作为控制。统计模型为 LMM：`similarity ~ condition * time + (1|subject) + (1|item)`。

旧版文档曾给出同一 ROI set 下跨 ROI 的描述性均值；这只能作为粗略方向检查，不应作为主统计解释。当前主图和派生表已改为单 ROI 展示：每个 ROI 分别给出 `Metaphor`、`Spatial`、`Baseline` 的 `post - pre`，并在 heatmap 中标注该 ROI 的 planned-interaction q。

`main_functional` 单 ROI 过程如下。负值表示 post 相似性低于 pre。

| ROI | Metaphor post-pre | Spatial post-pre | Baseline post-pre | Metaphor q |
| --- | ---: | ---: | ---: | ---: |
| `Prec-L1` | -0.1102 | -0.0671 | -0.0175 | 0.00158 |
| `Prec-L2` | -0.0874 | -0.0384 | -0.0208 | 0.0593 |
| `PHG-R` | -0.0630 | +0.0062 | +0.0263 | 0.000737 |
| `Fus-L` | -0.0618 | +0.0127 | +0.0130 | 0.000737 |
| `TP-L2` | -0.0612 | +0.0198 | +0.0526 | 0.000129 |
| `LOC-L` | -0.0603 | -0.0131 | +0.0001 | 0.0529 |
| `TP-L1` | -0.0543 | +0.0219 | +0.0537 | 0.000129 |

因此 Step 5C 的准确表述是：`main_functional` 中 5/7 个 ROI 的 `Metaphor × Pre` planned interaction 通过 `q < .05`，7/7 个 ROI 达到 `q < .10`；方向均为 `Metaphor` pair similarity 的 pre-to-post 下降。`literature` 与 `literature_spatial` 仍提供独立 ROI set 的稳健性检查，但写作时不应把它们压成“同类 ROI 平均效应”。

![Step 5C RSA 主图](E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png)

![三套 ROI 结果稳健性图](E:/python_metaphor/paper_outputs/figures_main/fig_roi_set_robustness.png)

源文件：
- [rsa_itemwise_details.csv: main_functional](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature_spatial](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv)
- [table_rsa_lmm_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rsa_lmm_fdr.tsv)
- [table_step5c_condition_summary.tsv](E:/python_metaphor/paper_outputs/figures_main/table_step5c_condition_summary.tsv)
- [table_step5c_roi_delta.tsv](E:/python_metaphor/paper_outputs/figures_main/table_step5c_roi_delta.tsv)
- [table_step5c_roi_condition_group.tsv](E:/python_metaphor/paper_outputs/figures_main/table_step5c_roi_condition_group.tsv)
- [fig_step5c_main.png](E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png)
- [fig_roi_set_robustness.png](E:/python_metaphor/paper_outputs/figures_main/fig_roi_set_robustness.png)

### 5.1 Learned relation-edge specificity：训练过的关系边是否特异性分化

A++ 分析直接回到 pre/post 4D pattern 与 ROI mask，比较三类 pair/edge：`YY/KJ` 中真实训练过的 cue-target edge、同一条件内未连接过的 untrained non-edge、以及 baseline 中由模板固定构造的 pseudo-edge。核心指标为 `drop_pre_minus_post = pre - post`；因此正值表示学习后 pair similarity 下降更强。该分析用于回答 Step 5C 的关键解释漏洞：相似性下降是否是 learned relation edge 特异的，而不是所有词之间都同步变散。

QC 通过：`edge_similarity_long.tsv` 共 10416 行，`edge_delta_subject.tsv` 共 5208 行，失败 cell 为 0；edge 计数为 `YY/KJ trained edge = 35`、`YY/KJ untrained non-edge = 2380`、`baseline pseudo-edge = 20`、`baseline untrained non-edge = 760`。主表包含 124 个 primary contrast，其中 `q_bh_primary_family < .05` 的结果为 119 个，`q < .10` 的结果为 124 个。新版脚本额外输出 `edge_process_group.tsv` 和 `edge_contrast_process_components.tsv`，用于显示每个显著比较背后的 pre、post 和 drop 过程值。

最强结果如下：

| ROI set | ROI | Contrast | Mean effect | t | p | q |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `literature_spatial` | `litspat_L_PPC` | `yy_specificity_minus_kj_specificity` | 0.0829 | 6.62 | 4.15e-07 | 1.99e-05 |
| `literature` | `lit_L_temporal_pole` | `yy_specificity_minus_kj_specificity` | 0.1220 | 6.21 | 1.23e-06 | 3.27e-05 |
| `literature` | `lit_R_IFG` | `yy_trained_drop_minus_kj_trained_drop` | 0.0967 | 6.04 | 1.89e-06 | 3.27e-05 |
| `literature` | `lit_L_AG` | `yy_trained_drop_minus_baseline_pseudo_drop` | 0.1042 | 6.01 | 2.04e-06 | 3.27e-05 |
| `literature` | `lit_L_temporal_pole` | `yy_trained_drop_minus_kj_trained_drop` | 0.1217 | 5.66 | 5.21e-06 | 5.43e-05 |
| `literature` | `lit_R_temporal_pole` | `yy_specificity_minus_kj_specificity` | 0.1251 | 5.35 | 1.18e-05 | 9.35e-05 |

这些比较背后的过程值显示，核心结果并不是“比较显著但不知道谁在变”。例如在 `litspat_L_PPC` 中，`YY trained edge` 从 `0.1915` 降到 `0.1109`，drop = `0.0806`；`YY untrained non-edge` drop = `0.0293`；`KJ trained edge` 反而略升，drop = `-0.0096`；`KJ untrained non-edge` drop = `0.0221`。在 `lit_L_temporal_pole` 中，`YY trained edge` 从 `0.2191` 降到 `0.1266`，drop = `0.0925`；`YY untrained non-edge` drop = `0.0233`；`KJ trained edge` drop = `-0.0292`。因此这里支持的是训练过的 `YY` 关系边出现更强分化，而不是全部词对同步变散。

海马也已经在这一机制中出现靠前结果：`litspat_L_hippocampus` 的 `YY trained edge` drop = `0.0612`，而 `YY untrained non-edge` drop = `0.0169`、`KJ trained edge` drop = `-0.0105`；`litspat_R_hippocampus` 的 `YY trained edge` drop = `0.0646`，而 `YY untrained non-edge` drop = `0.0163`、`KJ trained edge` drop = `-0.0101`。这支持后续把 hippocampal-binding ROI family 提到更高优先级。

解释上，这一结果比原 Step 5C 更强：`YY` 的 trained edge drop 不仅存在，而且系统性强于 untrained non-edge、`KJ` trained edge 和 baseline pseudo-edge。因此当前主神经结果应从“pair similarity decrease”升级为 **learned-edge differentiation / relation-edge reorganization**。

![Learned-edge specificity](E:/python_metaphor/paper_outputs/figures_main/fig_edge_specificity.png)

源文件：
- [table_edge_specificity.tsv](E:/python_metaphor/paper_outputs/tables_main/table_edge_specificity.tsv)
- [table_edge_specificity_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_edge_specificity_full.tsv)
- [table_edge_process_group.tsv](E:/python_metaphor/paper_outputs/tables_si/table_edge_process_group.tsv)
- [table_edge_contrast_process_components.tsv](E:/python_metaphor/paper_outputs/tables_si/table_edge_contrast_process_components.tsv)
- [edge_delta_subject.tsv](E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_delta_subject.tsv)
- [edge_process_group.tsv](E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_process_group.tsv)
- [edge_contrast_process_components.tsv](E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_contrast_process_components.tsv)
- [edge_specificity_group_fdr.tsv](E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_specificity_group_fdr.tsv)
- [table_edge_specificity_top.tsv](E:/python_metaphor/paper_outputs/figures_main/table_edge_specificity_top.tsv)
- [fig_edge_specificity.png](E:/python_metaphor/paper_outputs/figures_main/fig_edge_specificity.png)

### 5.2 Word-identity stability control：关系边分化是否只是词身份崩解

single-word stability 分析检验同一词的 pre pattern 与 post pattern 是否仍保持可识别的相似性，并同时计算每个词相对其他词的 neighborhood similarity change。该分析的作用是排除一个弱解释：如果 pair similarity 下降只是因为整体信号变差、词身份崩解或噪声上升，那么 same-word pre-post stability 不应保持。

QC 通过：`word_stability_long.tsv` 共 156240 行，subject-level 汇总 2604 行，失败 cell 为 0。主表 93 行中，`mean_same_word_stability` 有 47/93 个 ROI × condition cell 通过 `q_bh_primary_family < .05`；neighborhood change 只有少数 cell 通过 FDR，且不呈现稳定的 YY 特异模式。

最强 same-word stability 结果如下：

| ROI set | ROI | Condition | Metric | Mean effect | p | q |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `literature_spatial` | `litspat_L_PPC` | YY | `mean_same_word_stability` | 0.0401 | 3.7e-05 | 0.00072 |
| `literature_spatial` | `litspat_L_precuneus` | YY | `mean_same_word_stability` | 0.0494 | 4.0e-05 | 0.00072 |
| `literature_spatial` | `litspat_R_PPC` | YY | `mean_same_word_stability` | 0.0276 | 9.3e-05 | 0.00112 |
| `literature_spatial` | `litspat_R_precuneus` | YY | `mean_same_word_stability` | 0.0445 | 1.48e-04 | 0.00133 |
| `literature` | `lit_L_precuneus` | YY | `mean_same_word_stability` | 0.0494 | 4.0e-05 | 0.00144 |

解释上，single-word identity 与 learned-edge differentiation 可以共存：词本身在 pre/post 之间仍有可检测稳定性，但训练过的 `YY` relation edge 出现更强分化。因此当前结果不宜解释为“整体表征质量下降”，而更适合解释为学习诱发的关系边重组。

![Single-word stability](E:/python_metaphor/paper_outputs/figures_main/fig_word_stability.png)

源文件：
- [table_word_stability.tsv](E:/python_metaphor/paper_outputs/tables_main/table_word_stability.tsv)
- [table_word_stability_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_word_stability_full.tsv)
- [word_stability_long.tsv](E:/python_metaphor/paper_outputs/qc/word_stability/word_stability_long.tsv)
- [word_stability_group_fdr.tsv](E:/python_metaphor/paper_outputs/qc/word_stability/word_stability_group_fdr.tsv)
- [table_word_stability_top.tsv](E:/python_metaphor/paper_outputs/figures_main/table_word_stability_top.tsv)
- [fig_word_stability.png](E:/python_metaphor/paper_outputs/figures_main/fig_word_stability.png)

## 6. Model-RSA 机制结果：神经 RDM 对应哪类理论结构

### 6.1 既有机制 Model-RSA

Model-RSA 用于检验神经 RDM 的变化与不同机制模型的对应关系。这里的 RDM 不是单纯“跑模型”，而是把不同理论解释转成 item/pair 间距离矩阵，再检验 neural geometry 是否向这些结构靠近或远离。

当前模型的理论含义如下：`M1_condition` 是条件身份模型，检验神经结构是否主要按 `YY/KJ/baseline` 分类；`M2_pair` 是词对身份模型，检验是否只是记住具体 pair；`M3_embedding` 是通用语义 embedding 模型，代表刺激在一般语义空间中的距离；`M7_binary` 与 `M7_continuous_confidence` 是记忆/信心相关模型，检验表征变化是否可由后续记忆强度解释；`M8_reverse_pair` 是反向/重组关系模型，检验学习后是否形成与原始 pair 方向不同的关系结构。当前 primary models 为 `M3_embedding` 与 `M8_reverse_pair`，其余模型作为 control/secondary。

`literature` 中，左 IFG 的 `M8_reverse_pair` 是最强 primary 结果：`t = 3.47`, `p = .00179`, `q_bh_primary_family = .0429`；在更宽的全 model family 中，`q_bh_model_family = .0643`。`main_functional` 中，`Spatial_gt_Metaphor_c02_Precuneus` 的 `M8_reverse_pair` 为 `t = 3.06`, `p = .00494`, `q_bh_primary_family = .0691`；左颞极 c01 的 `M3_embedding` 为 `t = -2.74`, `p = .0107`, `q_bh_primary_family = .0751`。`literature_spatial` 中没有 primary model 通过 FDR，最小 `q_bh_primary_family = .596`。

Delta-rho LMM 中，`main_functional` 总模型的 `M3_embedding` 相对 `M1_condition` 更负，`b = -0.0119`, `z = -5.24`, `p = 1.58e-07`；该模型当前 `converged = false`。`literature` 和 `literature_spatial` 的 Delta-rho LMM 均收敛，但无模型项达到显著；`literature_spatial` 的 `M3_embedding` 为边缘趋势，`b = -0.00308`, `p = .0598`。

![Model-RSA 机制图](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)

源文件：
- [table_model_rsa_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_model_rsa_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: main_functional](E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_group_summary_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: literature](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_group_summary_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: literature_spatial](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_group_summary_fdr.tsv)
- [delta_rho_lmm_params.tsv: main_functional](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_lmm_params.tsv)
- [delta_rho_lmm_params.tsv: literature](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature/delta_rho_lmm_params.tsv)
- [delta_rho_lmm_params.tsv: literature_spatial](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature_spatial/delta_rho_lmm_params.tsv)
- [fig_model_rsa_mechanism.png](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)

### 6.2 Semantic relation-vector Model-RSA：关系向量几何是否被学习重排

A 阶段进一步把机制问题从“哪个已有模型更好”推进到 pair-level relation geometry。A0 使用 embedding 中的 `target - cue` 向量为每个学习 pair 构建 relation-vector model RDM；A1 在 ROI 内计算 pre/post neural RDM 与这些模型 RDM 的 Spearman 相关；A2 汇总主机制图与 top 表。由于 pre/post pattern 是 cue 与 target 分离的 word-level pattern，A 阶段的 neural relation-vector 使用 `target pattern - cue pattern`，并同时检查 pair centroid 作为补充 neural RDM。

QC 完整通过：`main_functional` 为 `784/784` cells ok，`literature` 为 `1344/1344` cells ok，`literature_spatial` 为 `1344/1344` cells ok；三套 ROI 均无 failed cells。primary relation-vector 模型共有 11 个结果达到 `q_bh_primary_family < .05`。最强结果如下：

| ROI set | ROI | Condition | Model | Pre | Post | Post - Pre | t | p | q |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `literature` | `lit_L_pMTG` | KJ | `M9_relation_vector_direct` | 0.0253 | -0.0251 | -0.0504 | -5.06 | 2.58e-05 | 0.00040 |
| `literature` | `lit_R_temporal_pole` | KJ | `M9_relation_vector_abs` | 0.0282 | -0.0149 | -0.0432 | -4.97 | 3.30e-05 | 0.00040 |
| `literature_spatial` | `litspat_R_OPA` | YY | `M9_relation_vector_direct` | 0.0260 | -0.0283 | -0.0544 | -4.32 | 0.00019 | 0.00458 |
| `literature` | `lit_L_temporal_pole` | YY | `M9_relation_vector_direct` | 0.0266 | -0.0128 | -0.0395 | -4.05 | 0.00038 | 0.00669 |
| `literature` | `lit_R_pMTG` | YY | `M9_relation_vector_direct` | 0.0190 | -0.0128 | -0.0318 | -3.91 | 0.00056 | 0.00669 |
| `literature_spatial` | `litspat_L_OPA` | KJ | `M9_relation_vector_direct` | 0.0218 | -0.0139 | -0.0357 | -4.04 | 0.00040 | 0.00956 |

方向解释需要谨慎：这些显著结果大多为负向 `post - pre`，即学习后 neural relation geometry 与通用 embedding relation-vector RDM 的对应下降。因此当前更稳妥的解释不是“学习后更贴近通用语义向量关系”，而是 **relation-vector realignment / decoupling**：学习后的神经关系几何从通用语义 embedding 结构中脱耦，可能转向更任务化、经验化或情境化的关系表征。这个结果与 Step 5C 中 pair similarity 的 post-pre 下降方向一致，但提供了更具体的关系几何解释。

![Relation-vector Model-RSA 主图](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_rsa.png)

源文件：
- [relation_pair_manifest.tsv](E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv)
- [relation_model_rdms.npz](E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz)
- [relation_model_collinearity.tsv](E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_collinearity.tsv)
- [relation_vector_group_summary_fdr.tsv: main_functional](E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_main_functional/relation_vector_group_summary_fdr.tsv)
- [relation_vector_group_summary_fdr.tsv: literature](E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature/relation_vector_group_summary_fdr.tsv)
- [relation_vector_group_summary_fdr.tsv: literature_spatial](E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature_spatial/relation_vector_group_summary_fdr.tsv)
- [table_relation_vector_rsa.tsv](E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_rsa.tsv)
- [table_relation_vector_rsa_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_rsa_full.tsv)
- [table_relation_vector_rsa_top.tsv](E:/python_metaphor/paper_outputs/figures_main/table_relation_vector_rsa_top.tsv)
- [fig_relation_vector_rsa.png](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_rsa.png)

### 6.3 A3：YY-KJ relation decoupling contrast

A3 进一步检验 relation-vector realignment 是否是 `YY` 相对 `KJ` 更强。指标定义为：

```text
yy_minus_kj_decoupling = (pre - post relation fit in YY) - (pre - post relation fit in KJ)
```

因此正值表示 `YY` 的 relation-vector decoupling 强于 `KJ`；负值表示 `KJ` 更强。结果没有支持原先的 `YY > KJ` 假设。primary relation-vector 结果中有 3/62 个通过 `q_bh_primary_family < .05`，但方向全部为负，即 `KJ > YY decoupling`。

| ROI set | ROI | Model | YY - KJ decoupling | t | p | q |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `literature` | `lit_R_temporal_pole` | `M9_relation_vector_abs` | -0.0746 | -3.87 | 0.00063 | 0.0151 |
| `main_functional` | `func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort` | `M9_relation_vector_abs` | -0.0479 | -3.53 | 0.00152 | 0.0213 |
| `literature_spatial` | `litspat_L_OPA` | `M9_relation_vector_abs` | -0.0513 | -3.72 | 0.00094 | 0.0224 |

解释上，A3 说明 relation-vector decoupling 不是强 YY 特异机制；它更像一般关系学习或空间/情境关系学习也会发生的 realignment。这个结果不推翻 A++ 的 learned-edge specificity，但要求写作时把“YY 的 pair-level differentiation”和“relation-vector realignment 的条件方向”分开。

![Relation-vector condition contrast](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_condition_contrast.png)

源文件：
- [table_relation_vector_condition_contrast.tsv](E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_condition_contrast.tsv)
- [table_relation_vector_condition_contrast_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_condition_contrast_full.tsv)
- [relation_vector_condition_contrast_subject.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_subject.tsv)
- [relation_vector_condition_contrast_group_fdr.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_group_fdr.tsv)
- [fig_relation_vector_condition_contrast.png](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_condition_contrast.png)

### 6.4 A4：Step5C/A++ 与 relation-vector conjunction

A4 将 pair differentiation evidence 与 A3 的 relation-vector condition contrast 按 ROI 对齐。pair evidence 同时纳入两类来源：A++ edge specificity，以及原 Step5C 的 `Metaphor` drop 相对 `Spatial/Baseline` 的条件差。conjunction 分类显式保留方向：

- `yy_pair_and_yy_relation`：pair evidence 为正且显著，同时 relation decoupling 也为 `YY > KJ`。
- `yy_pair_kj_relation_dissociation`：pair evidence 为正且显著，但 relation decoupling 为 `KJ > YY`。
- `yy_pair_only`：pair evidence 显著，但 relation contrast 不显著。

主表包含 278 行，其中 `yy_pair_only = 263`，`yy_pair_kj_relation_dissociation = 15`；没有出现稳定的 `yy_pair_and_yy_relation`。最靠前的解离结果如下：

| Class | Pair source | ROI set | ROI | Pair contrast | Relation model | Pair effect | Pair q | Relation effect | Relation q |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `yy_pair_kj_relation_dissociation` | Step5C | `main_functional` | `func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort` | `step5c_metaphor_drop_minus_baseline_drop` | `M9_relation_vector_abs` | 0.1080 | 0.000086 | -0.0479 | 0.0213 |
| `yy_pair_kj_relation_dissociation` | edge specificity | `literature` | `lit_R_temporal_pole` | `yy_specificity_minus_kj_specificity` | `M9_relation_vector_abs` | 0.1251 | 0.000093 | -0.0746 | 0.0151 |
| `yy_pair_kj_relation_dissociation` | edge specificity | `literature` | `lit_R_temporal_pole` | `yy_trained_drop_minus_baseline_pseudo_drop` | `M9_relation_vector_abs` | 0.1169 | 0.000162 | -0.0746 | 0.0151 |
| `yy_pair_kj_relation_dissociation` | edge specificity | `literature` | `lit_R_temporal_pole` | `yy_trained_drop_minus_untrained_nonedge_drop` | `M9_relation_vector_abs` | 0.0650 | 0.000217 | -0.0746 | 0.0151 |

解释上，A4 的主结论不是“同一 ROI 同时支持 YY pair 与 YY relation-vector decoupling”，而是更有边界感的解离：YY 的 learned-edge differentiation 很强，但 relation-vector condition contrast 在若干重叠 ROI 中偏向 `KJ > YY`。因此 relation-vector 结果适合作为 relation realignment 的补充与边界，而不应写成 YY 特异的核心机制。

![Relation Step5C conjunction](E:/python_metaphor/paper_outputs/figures_main/fig_relation_step5c_conjunction.png)

源文件：
- [table_relation_step5c_conjunction.tsv](E:/python_metaphor/paper_outputs/tables_main/table_relation_step5c_conjunction.tsv)
- [table_relation_step5c_conjunction_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_relation_step5c_conjunction_full.tsv)
- [relation_step5c_conjunction.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_conjunction/relation_step5c_conjunction.tsv)
- [pair_evidence_for_conjunction.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_conjunction/pair_evidence_for_conjunction.tsv)
- [table_relation_step5c_conjunction_top.tsv](E:/python_metaphor/paper_outputs/figures_main/table_relation_step5c_conjunction_top.tsv)
- [fig_relation_step5c_conjunction.png](E:/python_metaphor/paper_outputs/figures_main/fig_relation_step5c_conjunction.png)

### 6.5 A5：network-level dissociation

A5 将 relation-vector subject metrics 聚合到理论 network 层面，检验 `condition × network` 分工。网络包括 `semantic_metaphor`、`spatial_context`、`hippocampus_core`、`main_metaphor_functional` 和 `main_spatial_functional`。主指标仍为 `YY - KJ decoupling`，正值表示 `YY > KJ`，负值表示 `KJ > YY`。LMM 输出中，`M9_relation_vector_direct` 使用 subject + ROI variance component；`M9_relation_vector_abs` 因 ROI variance component singular，降级为 subject random-intercept model，并在参数表中记录 `fit_method`。

network-level condition contrast 中，`M9_relation_vector_abs` 有多个网络呈显著负向，即 `KJ > YY`：

| Analysis | Network / contrast | Model | Mean effect | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| network condition contrast | `main_metaphor_functional` | `M9_relation_vector_abs` | -0.0328 | 0.00877 | 0.0248 |
| network condition contrast | `semantic_metaphor` | `M9_relation_vector_abs` | -0.0240 | 0.00991 | 0.0248 |
| network condition contrast | `spatial_context` | `M9_relation_vector_abs` | -0.0252 | 0.0179 | 0.0298 |
| network condition contrast | `hippocampus_core` | `M9_relation_vector_abs` | -0.0230 | 0.0686 | 0.0857 |

但 planned network-difference contrasts 没有显著结果；例如 `semantic_minus_spatial` 在 `M9_relation_vector_abs` 中 mean = 0.00120, q = 0.943；`main_metaphor_minus_main_spatial` mean = -0.0184, q = 0.731。因此 A5 不支持稳定的 semantic/metaphor vs spatial/context `condition × network` 分工。它进一步支持 A3/A4 的修订结论：relation-vector realignment 是任务相关/关系学习相关的补充线索，但不是当前数据中最强的 YY 特异机制。

![Relation-vector network dissociation](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_network_dissociation.png)

源文件：
- [table_relation_vector_network_dissociation.tsv](E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_network_dissociation.tsv)
- [table_relation_vector_network_dissociation_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_network_dissociation_full.tsv)
- [relation_vector_network_long.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_network/relation_vector_network_long.tsv)
- [relation_vector_network_lmm_params.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_network/relation_vector_network_lmm_params.tsv)
- [relation_vector_network_planned_contrast_group.tsv](E:/python_metaphor/paper_outputs/qc/relation_vector_network/relation_vector_network_planned_contrast_group.tsv)
- [fig_relation_vector_network_dissociation.png](E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_network_dissociation.png)

## 7. 脑-行为分析

### 7.1 Item-level brain-behavior

item-level 分析将 run-7 行为 trial 与同一 `subject × ROI × condition × pair × word` 的 neural `post - pre` delta similarity 连接。新版 join 已把 RSA pair 的 `word_label` 与 `partner_label` 展开为两行，因此每个 pair 的两个词都进入分析。最新长表包含 `117180` 行、`27` 名被试、`31` 个 ROI。

memory accuracy 的未校正候选主要集中在 `Spatial/KJ` 条件，方向为负。最靠前结果包括：`literature_spatial` 左 PPC（`b = -0.0866`, `p = .000568`）、右 PPC（`b = -0.1059`, `p = .00575`）、`literature` 左 IFG（`b = -0.0828`, `p = .0115`）、`main_functional` 右后海马旁回（`b = -0.0921`, `p = .0119`）、`main_functional` 左后梭状回（`b = -0.0852`, `p = .0224`）。这些结果未作为跨 ROI/条件/指标强校正后的主阳性结论。

![Item-level memory coupling](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling.png)

![Item-level RT coupling](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling_rt.png)

源文件：
- [table_itemlevel_coupling.tsv](E:/python_metaphor/paper_outputs/tables_main/table_itemlevel_coupling.tsv)
- [table_itemlevel_coupling_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_fdr.tsv)
- [table_itemlevel_coupling_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_full.tsv)
- [table_itemlevel_coupling_interaction.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_interaction.tsv)
- [table_itemlevel_coupling_rt.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_rt.tsv)
- [itemlevel_behavior_neural_long.tsv](E:/python_metaphor/paper_outputs/qc/itemlevel_behavior_neural_long.tsv)
- [fig_itemlevel_coupling.png](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling.png)
- [fig_itemlevel_coupling_rt.png](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling_rt.png)

### 7.2 S1 机制到行为预测

S1 使用被试层 Model-RSA 机制分数预测 run-7 行为表现。当前只纳入 `M3_embedding` 和 `M8_reverse_pair` 两个 primary 机制模型，按四个 inference family 校正：`literature`、`literature_spatial`、`main_functional_metaphor_gt_spatial`、`main_functional_spatial_gt_metaphor`。长表包含 `27` 名被试、`10044` 行记录。

全表没有任何效应达到 `q_within_inference_family < .05` 或 `q_within_roi_set < .05`。最小 q 出现在 `literature_spatial`：左 RSC 的 `M8_reverse_pair` 预测 `YY - KJ` correct retrieval efficiency，`Spearman r = .634`, `permutation p = .0008`, `q_within_inference_family = .115`, `q_within_primary_family = .0768`。

![S1 机制到行为预测](E:/python_metaphor/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png)

源文件：
- [table_mechanism_behavior_prediction.tsv](E:/python_metaphor/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv)
- [table_mechanism_behavior_prediction_perm.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv)
- [mechanism_behavior_prediction_long.tsv](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_prediction_long.tsv)
- [mechanism_behavior_input_scope.tsv](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_input_scope.tsv)
- [fig_mechanism_behavior_prediction.png](E:/python_metaphor/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png)

## 8. Searchlight 全脑定位

本轮 searchlight 使用 `--reuse-subject-maps`，即复用既有 subject-level maps，刷新 group-level sign-flip permutation、FWE map、peak table 和图。

pair-similarity searchlight 中，`YY` 的 post - pre pair similarity 出现 FWE 校正后显著下降。前 10 个 peak 均为负 t 值，最高峰 MNI `[28, 16, 48]`，`t = -8.66`, `p_FWE = .00050`。其他显著峰包括 `[-50, 16, 2]`、`[18, -24, 10]`、`[14, -72, -28]`、`[26, 14, 42]`、`[14, -64, 4]`、`[20, -44, -2]`、`[8, -28, 50]`、`[-24, 16, 46]`、`[-32, -18, 6]`，均 `p_FWE < .003`。`KJ` 没有 FWE 显著 peak，最强峰 `[-4, -36, 40]` 的 `p_FWE = .351`。

RD searchlight 未发现 FWE 显著 peak。`YY` 最强峰 `[-10, 12, 2]` 的 `p_FWE = .159`；`KJ` 最强峰 `[-26, -14, -26]` 的 `p_FWE = .149`；baseline 最强峰 `[-64, -52, -2]` 的 `p_FWE = .387`。

![Pair-similarity searchlight](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png)

![RD searchlight](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_delta.png)

源文件：
- [table_searchlight_pair_similarity_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv)
- [pair_similarity_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/pair_similarity_searchlight_summary.tsv)
- [searchlight_subject_pairing_manifest.tsv: pair similarity](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/searchlight_subject_pairing_manifest.tsv)
- [fig_searchlight_pair_similarity_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png)
- [table_searchlight_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_peaks.tsv)
- [rd_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/rd_searchlight/rd_searchlight_summary.tsv)
- [searchlight_subject_pairing_manifest.tsv: RD](E:/python_metaphor/paper_outputs/qc/rd_searchlight/searchlight_subject_pairing_manifest.tsv)
- [fig_searchlight_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_delta.png)

## 9. ROI 间 representational connectivity

Representational connectivity 检验 ROI 间表征 profile 的相似性。最新结果中，6 个 planned scope 的 `YY vs KJ` 均未达到显著。

| ROI pool | Pair scope | mean YY | mean KJ | t | p |
| --- | --- | ---: | ---: | ---: | ---: |
| independent | within_literature | 0.392 | 0.428 | -1.32 | .199 |
| independent | within_literature_spatial | 0.447 | 0.485 | -1.55 | .134 |
| independent | cross_independent_sets | 0.488 | 0.526 | -1.65 | .110 |
| main_functional | within_Metaphor_gt_Spatial | 0.341 | 0.271 | 1.27 | .216 |
| main_functional | within_Spatial_gt_Metaphor | 0.445 | 0.463 | -0.64 | .528 |
| main_functional | cross_main_functional_family | 0.301 | 0.309 | -0.27 | .787 |

![Representational connectivity](E:/python_metaphor/paper_outputs/figures_main/fig_repr_connectivity.png)

源文件：
- [table_repr_connectivity.tsv](E:/python_metaphor/paper_outputs/tables_main/table_repr_connectivity.tsv)
- [table_repr_connectivity_subject.tsv](E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_subject.tsv)
- [table_repr_connectivity_edges.tsv](E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_edges.tsv)
- [fig_repr_connectivity.png](E:/python_metaphor/paper_outputs/figures_main/fig_repr_connectivity.png)

## 10. gPPI 网络层结果

gPPI 使用两个左颞极 seed：`func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort` 和 `func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort`。target ROI 合并 `literature + literature_spatial`，共 24 个 target；阶段为 pre/post；条件为 `yy/kj`；deconvolution 为 `ridge`。本轮未重跑 raw gPPI，只刷新 summary export。

6 个 planned `yy_minus_kj` scope 主检验均不显著，sign-flip permutation + BH-FDR 后也均不显著，最小 `q_within_main_tests = .770`。

| Seed | Scope | mean post | mean pre | mean diff | p | permutation p | q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c01 temporal pole | within_literature | -5.287 | -0.806 | -4.480 | .421 | .825 | .825 |
| c01 temporal pole | within_literature_spatial | -1.520 | -0.979 | -0.541 | .763 | .803 | .825 |
| c01 temporal pole | cross_sets | -3.766 | 0.173 | -3.939 | .350 | .599 | .825 |
| c02 temporal pole | within_literature | -1.886 | -0.139 | -1.746 | .488 | .646 | .825 |
| c02 temporal pole | within_literature_spatial | 1.494 | -1.052 | 2.546 | .231 | .257 | .770 |
| c02 temporal pole | cross_sets | -3.380 | 0.913 | -4.293 | .269 | .184 | .770 |

本节当前无单独结果图。

源文件：
- [table_gppi_summary_main.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_summary_main.tsv)
- [table_gppi_scope_permutation.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_scope_permutation.tsv)
- [table_gppi_target_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_target_peaks.tsv)
- [table_gppi_target_level.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_target_level.tsv)
- [gppi_summary_export_manifest.json](E:/python_metaphor/paper_outputs/qc/gppi_summary_export_manifest.json)

## 11. 稳健性与 noise ceiling

Robustness suite 基于 Step 5C subject-level interaction delta。Bootstrap CI 均不跨 0。例子：`literature__all` 的 `YY vs baseline` 平均 interaction delta 为 `-0.0956`，95% CI `[-0.1219, -0.0703]`；`YY vs KJ` 为 `-0.0774`，95% CI `[-0.1032, -0.0517]`。`main_functional__all` 的 `YY vs baseline` 为 `-0.0865`，95% CI `[-0.1137, -0.0580]`；`YY vs KJ` 为 `-0.0629`，95% CI `[-0.0877, -0.0372]`。

LOSO 中，10 个 `roi_scope × contrast` 均维持同方向。Split-half 中所有 scope 的方向一致率为 `1.0`，两半均 `p < .05` 的比例为 `0.680` 到 `1.000`。Bayes factor 表目前没有 BF10 数值，因为运行环境未安装 `pingouin`。

Noise ceiling 显示 Step 5C itemwise delta 有可解释的跨被试重复性。`literature__all` 中 `YY` lower/upper mean r 为 `0.438/0.496`，`KJ` 为 `0.382/0.470`，baseline 为 `0.506/0.547`。`main_functional__all` 中 `YY` 为 `0.314/0.389`，`KJ` 为 `0.339/0.426`，baseline 为 `0.470/0.498`。

![Noise ceiling](E:/python_metaphor/paper_outputs/figures_si/fig_noise_ceiling.png)

源文件：
- [table_bootstrap_ci.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bootstrap_ci.tsv)
- [table_loso.tsv](E:/python_metaphor/paper_outputs/tables_si/table_loso.tsv)
- [table_splithalf.tsv](E:/python_metaphor/paper_outputs/tables_si/table_splithalf.tsv)
- [table_bayes_factors.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bayes_factors.tsv)
- [table_noise_ceiling.tsv](E:/python_metaphor/paper_outputs/tables_si/table_noise_ceiling.tsv)
- [fig_noise_ceiling.png](E:/python_metaphor/paper_outputs/figures_si/fig_noise_ceiling.png)

## 12. ROI-level RD / GPS 与学习动态补充

ROI-level RD/GPS 当前作为补充分析。RD 结果总体为阴性，没有提供稳定的 ROI-level 维度变化证据。GPS 中，`literature` 的左 AG 和左 pSTS 出现未校正显著，方向为 `YY` 的 GPS delta 比 `KJ` 更负：左 AG 中，`YY = -0.0309`，`KJ = -0.0100`，`t = -3.68`, `p = .0010`；左 pSTS 中，`YY = -0.0358`，`KJ = -0.0140`，`t = -2.19`, `p = .037`。当前 RD/GPS 表未提供跨 ROI 的 FDR 校正列。

![RD/GPS main_functional](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_main_functional.png)

![RD/GPS literature](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_literature.png)

学习动态结果当前作为学习阶段过程补充。已有图包括 `main_functional` 与 `literature` 两套 ROI 的 repeated-exposure 表征动态。

新增 relation-vector learning trajectory（C 阶段）检验 run3 到 run4 的 pair-pattern RDM 是否逐步贴近 relation-vector model RDM。QC 通过：3472/3472 window cells 均为 ok，failure 文件为空。primary relation-vector learning trajectory 没有通过 primary-family FDR；最接近的候选为 `literature_spatial/litspat_L_RSC/YY/M9_relation_vector_direct`，run4 - run3 = 0.0296, p = 0.0567, q = 0.537；以及 `literature_spatial/litspat_L_hippocampus/YY/M9_relation_vector_abs`，run4 - run3 = -0.0267, p = 0.0682, q = 0.537。因此 C 阶段不能支持“relation geometry 在学习过程中逐步形成”的强主张，更适合作为边界结果：pre/post 表征重组明确，但 run3/run4 pair-pattern trajectory 没有稳定捕捉同一机制。

run7 retrieval geometry 当前尚未进入结果，因为数据检查显示 `pattern_root` 只包含 pre/post/learn patterns，`lss_betas_final/lss_metadata_index_final.csv` 只包含 run 1/2/5/6 与 Pre/Post；虽然 run7 行为表存在，但暂未发现可直接用于 retrieval-stage RSA 的 run7 神经 pattern。后续若定位或生成 run7 LSS/pattern 后，再补 `retrieval_pair_similarity.py` 与 retrieval success prediction。

![Learning dynamics main_functional](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_main_functional.png)

![Learning dynamics literature](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_literature.png)

源文件：
- [table_rd_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rd_main_functional.tsv)
- [table_gps_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gps_main_functional.tsv)
- [table_rd_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rd_literature.tsv)
- [table_gps_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gps_literature.tsv)
- [fig_rd_gps_main_functional.png](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_main_functional.png)
- [fig_rd_gps_literature.png](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_literature.png)
- [table_learning_repeated_exposure_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_main/table_learning_repeated_exposure_main_functional.tsv)
- [table_learning_repeated_exposure_literature.tsv](E:/python_metaphor/paper_outputs/tables_main/table_learning_repeated_exposure_literature.tsv)
- [table_relation_learning_dynamics.tsv](E:/python_metaphor/paper_outputs/tables_main/table_relation_learning_dynamics.tsv)
- [relation_learning_group_summary_fdr.tsv](E:/python_metaphor/paper_outputs/qc/relation_learning_dynamics/relation_learning_group_summary_fdr.tsv)
- [relation_learning_dynamics_long.tsv](E:/python_metaphor/paper_outputs/qc/relation_learning_dynamics/relation_learning_dynamics_long.tsv)
- [fig_learning_dynamics_main_functional.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_main_functional.png)
- [fig_learning_dynamics_literature.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_literature.png)
- [fig_relation_learning_dynamics.png](E:/python_metaphor/paper_outputs/figures_main/fig_relation_learning_dynamics.png)

## 13. 当前结果一页概览

| 模块 | 当前结果 |
| --- | --- |
| 行为 | `YY` 最终记忆正确率高于 `KJ`，正确 trial 反应更快 |
| 学习 GLM | `Metaphor > Spatial` 定位左前颞/颞极；`Spatial > Metaphor` 定位后部视觉-顶叶-内侧颞网络 |
| ROI-level RSA | 已改为单 ROI 展示；`main_functional` 中 5/7 个 ROI 的 `Metaphor × Pre` 过 `q < .05`，7/7 个到 `q < .10`，方向均为 `Metaphor` pair similarity 下降 |
| Learned relation-edge specificity | `YY` trained edge 的 pre-post similarity drop 显著强于 untrained non-edge、`KJ` trained edge 与 baseline pseudo-edge；过程表显示主要来自 `YY trained edge` 明确下降，而不是所有 non-edge 同步下降 |
| Word-identity stability control | same-word pre-post stability 在 47/93 个 ROI × condition cell 中通过 FDR；说明 pair/edge 分化并非简单整体噪声或词身份崩解 |
| Model-RSA | 已补充各 RDM 模型理论含义；既有机制模型中 `literature` 左 IFG 的 `M8_reverse_pair` 在 primary-family 内显著；relation-vector Model-RSA 支持 relation geometry realignment/decoupling 而非简单 alignment 增强 |
| YY-KJ relation-vector contrast | 不支持 `YY > KJ` relation decoupling；3 个 primary relation-vector 结果通过 FDR，但均为负向，即 `KJ > YY` |
| Pair evidence × relation-vector conjunction | 没有稳定的 `YY pair + YY relation` 同向 conjunction；主要为 `YY pair only` 和少量 `YY pair + KJ relation` 解离 |
| Network-level relation-vector dissociation | `M9_relation_vector_abs` 在多个 network 中呈 `KJ > YY` 条件差，但 planned condition × network contrast 不显著；不支持语义网络 vs 空间网络的强分工 |
| Item-level 脑-行为 | memory accuracy 的局部候选主要集中在 `Spatial/KJ`；没有形成强 YY item-level accuracy coupling |
| S1 机制预测行为 | 无 FDR 校正后显著结果；最佳候选为 `literature_spatial` 左 RSC 的 `M8_reverse_pair` 预测 `YY - KJ` retrieval efficiency |
| Searchlight | pair-similarity searchlight 中 `YY` 有 FWE 显著下降；`KJ` 无 FWE 显著；RD searchlight 阴性 |
| Representational connectivity | 六个 planned scope 均不显著 |
| gPPI | 六个 planned `yy_minus_kj` scope 主检验均不显著 |
| Robustness / ceiling | Step 5C 主 RSA 效应 bootstrap、LOSO、split-half 支持方向稳定；noise ceiling 处于可解释范围 |
| Learning trajectory / retrieval | run3/run4 relation-vector trajectory 未通过 FDR；run7 retrieval 神经 pattern 尚未定位，当前只能使用 run7 行为，不能做 retrieval-stage RSA |
