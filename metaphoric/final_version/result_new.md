# 当前结果整理：隐喻联结学习的行为与神经表征结果

最后更新：2026-05-02

本文档用于客观汇总当前 `final_version` 的主要分析结果、对应图片和源文件。内容按研究链路排列：数据与执行状态、行为结果、材料控制、学习阶段 GLM、ROI-level RSA、Model-RSA、脑-行为分析、searchlight、连接分析、稳健性与补充分析。

## 1. 数据与执行状态

当前正式 ROI 范围为三套：`main_functional`、`literature`、`literature_spatial`。本轮已基于最新代码重跑 Step 5C RSA、RSA LMM、Model-RSA、Delta-rho LMM、item-level brain-behavior、S1 机制到行为预测、representational connectivity、robustness suite 和 noise ceiling。新增 A 阶段 relation-vector Model-RSA 已完成 A0/A1/A2：构建 pair-level semantic relation-vector model RDM，并在三套 ROI 中检验 pre/post neural geometry 与该模型的关系。两个 searchlight 使用 `--reuse-subject-maps`，复用既有 subject-level `.nii.gz`，刷新 group-level permutation/FWE、peak table、QC 和图。gPPI 本轮执行 `export_gppi_summary.py`，基于既有 raw gPPI 结果刷新 summary 表。

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

三套 ROI 的描述性均值如下。

| ROI set | Condition | Pre | Post | Post - Pre |
| --- | --- | ---: | ---: | ---: |
| `main_functional` | Baseline | 0.0885 | 0.1039 | +0.0154 |
| `main_functional` | Metaphor | 0.1725 | 0.1013 | -0.0712 |
| `main_functional` | Spatial | 0.1538 | 0.1455 | -0.0083 |
| `literature` | Baseline | 0.0726 | 0.1025 | +0.0299 |
| `literature` | Metaphor | 0.1731 | 0.1074 | -0.0657 |
| `literature` | Spatial | 0.1576 | 0.1694 | +0.0118 |
| `literature_spatial` | Baseline | 0.0802 | 0.0963 | +0.0161 |
| `literature_spatial` | Metaphor | 0.1623 | 0.1018 | -0.0605 |
| `literature_spatial` | Spatial | 0.1480 | 0.1532 | +0.0052 |

`main_functional` 中，`Metaphor × Pre` 在两个 base-contrast family 上均显著：`Spatial_gt_Metaphor` family `b = 0.0768`, `p = 2.15e-13`, planned-interaction q = `8.61e-13`；`Metaphor_gt_Spatial` family `b = 0.1109`, `p = 1.17e-10`, q = `2.34e-10`。单 ROI 层面，7/7 个主功能 ROI 的 `Metaphor × Pre` 通过 planned-interaction FDR。

`literature` 中，12/12 个 ROI 的 `Metaphor × Pre` 通过 FDR，最小 q = `6.36e-06`。`literature_spatial` 中，12/12 个 ROI 的 `Metaphor × Pre` 通过 FDR，最小 q = `1.90e-05`。`Spatial × Pre` 没有形成同等级别的跨 ROI 稳定结果。

![Step 5C RSA 主图](E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png)

![三套 ROI 结果稳健性图](E:/python_metaphor/paper_outputs/figures_main/fig_roi_set_robustness.png)

源文件：
- [rsa_itemwise_details.csv: main_functional](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature_spatial](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv)
- [table_rsa_lmm_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rsa_lmm_fdr.tsv)
- [table_step5c_condition_summary.tsv](E:/python_metaphor/paper_outputs/figures_main/table_step5c_condition_summary.tsv)
- [table_step5c_roi_delta.tsv](E:/python_metaphor/paper_outputs/figures_main/table_step5c_roi_delta.tsv)
- [fig_step5c_main.png](E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png)
- [fig_roi_set_robustness.png](E:/python_metaphor/paper_outputs/figures_main/fig_roi_set_robustness.png)

## 6. Model-RSA 机制结果

### 6.1 既有机制 Model-RSA

Model-RSA 用于检验神经 RDM 的变化与不同机制模型的对应关系。当前 primary models 为 `M3_embedding` 与 `M8_reverse_pair`，control/secondary 模型包括 `M1_condition`、`M2_pair`、`M7_binary`、`M7_continuous_confidence`。

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

### 6.2 新增 relation-vector Model-RSA（A 阶段）

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
- [fig_learning_dynamics_main_functional.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_main_functional.png)
- [fig_learning_dynamics_literature.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_literature.png)

## 13. 当前结果一页概览

| 模块 | 当前结果 |
| --- | --- |
| 行为 | `YY` 最终记忆正确率高于 `KJ`，正确 trial 反应更快 |
| 学习 GLM | `Metaphor > Spatial` 定位左前颞/颞极；`Spatial > Metaphor` 定位后部视觉-顶叶-内侧颞网络 |
| ROI-level RSA | 三套 ROI 中 `YY/Metaphor` 均表现出稳定 post-pre pair similarity 下降；`KJ` 与 baseline 不呈现同等下降 |
| Model-RSA | 既有机制模型中 `literature` 左 IFG 的 `M8_reverse_pair` 在 primary-family 内显著；新增 relation-vector Model-RSA 出现 11 个 primary-family FDR 显著结果，主要表现为 post-pre relation-vector alignment 下降，支持 relation geometry realignment/decoupling 而非简单 alignment 增强 |
| Item-level 脑-行为 | memory accuracy 的局部候选主要集中在 `Spatial/KJ`；没有形成强 YY item-level accuracy coupling |
| S1 机制预测行为 | 无 FDR 校正后显著结果；最佳候选为 `literature_spatial` 左 RSC 的 `M8_reverse_pair` 预测 `YY - KJ` retrieval efficiency |
| Searchlight | pair-similarity searchlight 中 `YY` 有 FWE 显著下降；`KJ` 无 FWE 显著；RD searchlight 阴性 |
| Representational connectivity | 六个 planned scope 均不显著 |
| gPPI | 六个 planned `yy_minus_kj` scope 主检验均不显著 |
| Robustness / ceiling | Step 5C 主 RSA 效应 bootstrap、LOSO、split-half 支持方向稳定；noise ceiling 处于可解释范围 |
