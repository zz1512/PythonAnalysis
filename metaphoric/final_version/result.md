## 0、本轮 `new_rerun_list` 重跑结果总览（2026-05-02 更新）

本轮根据 `new_rerun_list.md` 完成了 P0 主链和 RSA 下游补充链的结果刷新。需要先固定执行状态：`Step 5C RSA`、`RSA LMM`、`M7 memory strength`、`Model-RSA`、`Delta-rho LMM`、`S1 机制到行为预测`、`item-level brain-behavior`、`representational connectivity`、`robustness suite`、`noise ceiling` 均已基于最新产物重跑；两个 searchlight 使用 `--reuse-subject-maps` 复用既有 subject-level `.nii.gz`，只刷新 group-level permutation/FWE、peak table、QC 和图；gPPI 本轮只执行轻量 summary export，没有重跑 raw gPPI 模型。也就是说，当前 `result.md` 的正式口径应理解为：主 ROI/RSA/Model-RSA/脑行为链为全量更新，searchlight 与 gPPI 为基于既有底层图/模型的正式统计刷新。

### 0.1 执行日志与总体 QC

`Step 5C RSA` 三套 ROI set 均成功完成，且没有 failed cell 或 incomplete cell。`main_functional` 用时约 `30.92 min`，包含 `28` 名被试、`7` 个 ROI、`35280` 条 item-wise 行、`7840` 条 baseline item-wise 行；`literature` 用时约 `49.01 min`，包含 `28` 名被试、`12` 个 ROI、`60480` 条 item-wise 行、`13440` 条 baseline 行；`literature_spatial` 用时约 `48.37 min`，规模同为 `28 × 12 ROI`、`60480` 条 item-wise 行、`13440` 条 baseline 行。三套 ROI 的 `n_failed_cells = 0`、`n_incomplete_cells = 0`，说明 P0a 的 fail-fast/completeness 检查通过。

后续重跑日志也显示主链已经闭合：`Model-RSA` 三套 ROI 分别用时约 `10.94 min`、`18.68 min` 和 `18.73 min`；`mechanism_behavior_prediction.py` 完整运行约 `31 min`；`itemlevel_brain_behavior.py`、representational connectivity、robustness、noise ceiling、searchlight group-level 刷新和 gPPI export 均为 `exit_code = 0`。本轮有两个已处理的执行细节：第一，`build_memory_strength_table.py` 初次运行因为未设置 `METAPHOR_ROI_SET` 失败，补设后成功；第二，`delta_rho_lmm.py --family-split` 在 `literature` 上初次因 `base_contrast` 缺失失败，现已修复为从 ROI manifest 合并元数据，`literature/literature_spatial` 无 `base_contrast` 时退回到 `roi_family`，三套 ROI 均已重跑成功。

本节依据产物：
- [step5c_rsa_summary.tsv](E:/python_metaphor/paper_outputs/logs/step5c_rsa_20260501_200250/step5c_rsa_summary.tsv)
- [rerun_after_step5c logs](E:/python_metaphor/paper_outputs/logs/rerun_after_step5c_20260501_234434)
- [result_update_stats.txt](E:/python_metaphor/paper_outputs/logs/rerun_after_step5c_20260501_234434/result_update_stats.txt)

### 0.2 Step 5C ROI-level RSA 与 RSA LMM

最新 Step 5C 的核心模式仍然非常稳定：三套 ROI 中，`Metaphor/YY` 都表现为最清楚的 post 相对 pre 的 pair similarity 下降，而 `Spatial/KJ` 较弱或不稳定，`Baseline` 多为稳定或上升。描述性均值如下：

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

RSA LMM 的 FDR 结果进一步支持这一点。`main_functional` 中 planned interaction 的最小 q 为 `8.61e-13`，`Metaphor × Pre` 在两个 base-contrast family 上均显著：`Spatial_gt_Metaphor` family `b = 0.0768`, `p = 2.15e-13`；`Metaphor_gt_Spatial` family `b = 0.1109`, `p = 1.17e-10`。单 ROI 层面，7 个主功能 ROI 的 `Metaphor × Pre` 均通过 planned-interaction FDR。`literature` 的 planned interaction 中，`Metaphor × Pre` 在 12/12 个 ROI 通过 FDR，最小 q = `6.36e-06`；`literature_spatial` 同样为 12/12 个 ROI 通过 FDR，最小 q = `1.90e-05`。相比之下，`Spatial × Pre` 没有形成同等级别的跨 ROI 稳定结果。

因此，本轮 RSA 主结论不变且更干净：`YY` 的学习后 pair-level 去相似化是跨主功能 ROI、隐喻文献 ROI 与空间文献 ROI 都可重复的稳定效应；baseline 控制和空间文献 ROI 结果共同约束了“普遍重测效应”与“ROI 选择不够空间化”这两类替代解释。

本节依据产物：
- [rsa_itemwise_details.csv: main_functional](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv)
- [rsa_itemwise_details.csv: literature_spatial](E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv)
- [table_rsa_lmm_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rsa_lmm_fdr.tsv)
- [fig_step5c_main.png](E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png)
- [fig_roi_set_robustness.png](E:/python_metaphor/paper_outputs/figures_main/fig_roi_set_robustness.png)

### 0.3 Model-RSA 与 Delta-rho LMM

Model-RSA 本轮正式启用 `M3_embedding` 与 `M8_reverse_pair` 作为 primary model，同时保留 `M1_condition`、`M2_pair`、`M7_binary`、`M7_continuous_confidence` 作为 control/secondary 模型。三套 ROI set 均为 pooled/all 口径。最新 FDR 结果显示，最接近正式阳性的机制线索集中在 `M8_reverse_pair` 与 `M3_embedding`，但校正后强度有限。

`literature` 中，左 IFG 的 `M8_reverse_pair` 是最强 primary 结果：`t = 3.47`, `p = .00179`, `q_bh_primary_family = .0429`，说明文献语义控制 ROI 中 pair-based differentiation 机制有一个通过 primary-family FDR 的候选证据；但若放进全 model family，`q_bh_model_family = .0643`，因此表述应保守。`main_functional` 中最强 primary 结果包括 `Spatial_gt_Metaphor_c02_Precuneus` 的 `M8_reverse_pair`（`t = 3.06`, `p = .00494`, `q_bh_primary_family = .0691`）和左颞极 c01 的 `M3_embedding`（`t = -2.74`, `p = .0107`, `q_bh_primary_family = .0751`），均为接近但未过 `.05` 的趋势。`literature_spatial` 没有 primary model 通过 FDR，最小 `q_bh_primary_family = .596`。

Delta-rho LMM 用于概括每个模型从 pre 到 post 的 rho 变化。`main_functional` 总模型中 `M3_embedding` 相对 `M1_condition` 显著更负（`b = -0.0119`, `z = -5.24`, `p = 1.58e-07`），提示主功能 ROI 中 embedding-related alignment 从 pre 到 post 出现更明显下降；但该总模型 `converged = false`，且 `Spatial_gt_Metaphor` family split 也未完全收敛，所以这部分更适合写成补充性方向证据。`literature` 与 `literature_spatial` 的 delta-rho LMM 均收敛，但没有模型项达到显著；`literature_spatial` 的 `M3_embedding` 为边缘趋势（`b = -0.00308`, `p = .0598`）。

本节依据产物：
- [table_model_rsa_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_model_rsa_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: main_functional](E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_group_summary_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: literature](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_group_summary_fdr.tsv)
- [model_rdm_group_summary_fdr.tsv: literature_spatial](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_group_summary_fdr.tsv)
- [delta_rho_lmm_params.tsv: main_functional](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_lmm_params.tsv)
- [delta_rho_lmm_params.tsv: literature](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature/delta_rho_lmm_params.tsv)
- [delta_rho_lmm_params.tsv: literature_spatial](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature_spatial/delta_rho_lmm_params.tsv)
- [fig_model_rsa_mechanism.png](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)

### 0.4 脑-行为链：S1 与 item-level coupling

S1 `mechanism_behavior_prediction.py` 已基于最新 Model-RSA 结果重新运行。合并后的长表包含 `27` 名被试、`10044` 行记录，仅纳入 primary 机制模型 `M3_embedding` 与 `M8_reverse_pair`，并按四个 inference family 校正：`literature`、`literature_spatial`、`main_functional_metaphor_gt_spatial`、`main_functional_spatial_gt_metaphor`。全表没有任何结果达到 `q_within_inference_family < .05`，也没有达到 `q_within_roi_set < .05`。最接近的是 `literature_spatial` 左 RSC 的 `M8_reverse_pair` 预测 `YY - KJ` correct retrieval efficiency：`Spearman r = .634`, `permutation p = .0008`, `q_within_inference_family = .115`, `q_within_primary_family = .0768`。因此，S1 当前仍应写成“有探索性候选线索，但没有 FDR 校正后的被试层行为闭环”。

item-level brain-behavior 已用最新 RSA itemwise 和 refined behavior 重跑。长表包含 `117180` 行、`27` 名被试、`31` 个 ROI，说明 pair 展开 join 修复后，每条行为 trial 均接入三套 ROI 的 neural predictor。memory accuracy 的未校正候选仍主要集中在 `Spatial/KJ` 条件，方向为负：例如 `literature_spatial` 左 PPC（`b = -0.0866`, `p = .000568`）、右 PPC（`b = -0.1059`, `p = .00575`）、`literature` 左 IFG（`b = -0.0828`, `p = .0115`）以及 `main_functional` 右后海马旁回（`b = -0.0921`, `p = .0119`）。这意味着 item-level 层面没有支持“YY 中 neural differentiation 越强，memory accuracy 越高”的强命题；更稳妥的写法仍是：局部脑-行为耦合存在候选信号，但不是主机制闭环的强阳性证据。

本节依据产物：
- [table_mechanism_behavior_prediction.tsv](E:/python_metaphor/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv)
- [table_mechanism_behavior_prediction_perm.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv)
- [mechanism_behavior_prediction_long.tsv](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_prediction_long.tsv)
- [fig_mechanism_behavior_prediction.png](E:/python_metaphor/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png)
- [table_itemlevel_coupling.tsv](E:/python_metaphor/paper_outputs/tables_main/table_itemlevel_coupling.tsv)
- [table_itemlevel_coupling_fdr.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_fdr.tsv)
- [itemlevel_behavior_neural_long.tsv](E:/python_metaphor/paper_outputs/qc/itemlevel_behavior_neural_long.tsv)
- [fig_itemlevel_coupling.png](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling.png)

### 0.5 RSA 下游补充：representational connectivity、robustness、noise ceiling

Representational connectivity 本轮读取三套最新 RSA itemwise 后重新导出。六个 planned scope 的 `YY vs KJ` 均未达到显著：`independent` ROI pool 中，`within_literature` 为 `t = -1.32`, `p = .199`，`within_literature_spatial` 为 `t = -1.55`, `p = .134`，`cross_independent_sets` 为 `t = -1.65`, `p = .110`；`main_functional` 中，`within_Metaphor_gt_Spatial` 为 `t = 1.27`, `p = .216`，`within_Spatial_gt_Metaphor` 为 `t = -0.64`, `p = .528`，`cross_main_functional_family` 为 `t = -0.27`, `p = .787`。因此，B2 当前不支持稳定的 ROI 间 representational connectivity 条件差异，仍应作为阴性补充。

Robustness suite 对 Step 5C subject-level interaction delta 给出强支持。10 个 `roi_scope × contrast` 的 LOSO 均为 `28/28` 同方向；split-half 中所有 scope 的方向一致率均为 `1.0`，两半均 `p < .05` 的比例从 `0.680` 到 `1.000` 不等。bootstrap CI 也均不跨 0，例如 `literature__all` 的 `YY vs baseline` 平均 interaction delta 为 `-0.0956`，95% CI `[-0.1219, -0.0703]`；`YY vs KJ` 为 `-0.0774`，95% CI `[-0.1032, -0.0517]`。Bayes factor 表当前因 `pingouin` 未安装无法给出 BF10，因此这部分只能报告 frequentist robustness，而不应写 BF 证据。

Noise ceiling 结果显示 Step 5C itemwise delta 的跨被试可重复性处于可解释范围。以 `step5c_itemwise_delta` 为例，`literature__all` 中 `YY` 的 lower/upper mean r 为 `0.438/0.496`，`KJ` 为 `0.382/0.470`，baseline 为 `0.506/0.547`；`main_functional__all` 中 `YY` 为 `0.314/0.389`，`KJ` 为 `0.339/0.426`，baseline 为 `0.470/0.498`。这些 ceiling 不是显著性检验，但说明当前 RSA profile 并非完全不可重复，适合作为 SI 中的可靠性边界。

本节依据产物：
- [table_repr_connectivity.tsv](E:/python_metaphor/paper_outputs/tables_main/table_repr_connectivity.tsv)
- [table_repr_connectivity_subject.tsv](E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_subject.tsv)
- [fig_repr_connectivity.png](E:/python_metaphor/paper_outputs/figures_main/fig_repr_connectivity.png)
- [table_bayes_factors.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bayes_factors.tsv)
- [table_bootstrap_ci.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bootstrap_ci.tsv)
- [table_loso.tsv](E:/python_metaphor/paper_outputs/tables_si/table_loso.tsv)
- [table_splithalf.tsv](E:/python_metaphor/paper_outputs/tables_si/table_splithalf.tsv)
- [table_noise_ceiling.tsv](E:/python_metaphor/paper_outputs/tables_si/table_noise_ceiling.tsv)
- [fig_noise_ceiling.png](E:/python_metaphor/paper_outputs/figures_si/fig_noise_ceiling.png)

### 0.6 Searchlight 与 gPPI 的本轮刷新

两个 searchlight 本轮不是全量重算 subject-level `.nii.gz`，而是使用 `--reuse-subject-maps` 复用既有个体图，重新刷新 subject pairing QC、group-level sign-flip permutation、FWE map、peak table 和图。pair-similarity searchlight 的结果支持主线：`YY` 的 post - pre pair similarity 出现 voxelwise max-T FWE 显著下降，前 10 个 peak 均为负 t 值，最高峰为 MNI `[28, 16, 48]`，`t = -8.66`, `p_FWE = .00050`；其他显著峰包括 `[-50, 16, 2]`、`[18, -24, 10]`、`[14, -72, -28]`、`[26, 14, 42]` 等，均 `p_FWE < .003`。`KJ` 没有任何 FWE 显著 peak，最强峰 `[-4, -36, 40]` 的 `p_FWE = .351`。RD searchlight 为阴性：`YY` 最强峰 `[-10, 12, 2]` 的 `p_FWE = .159`，`KJ` 最强峰 `[-26, -14, -26]` 的 `p_FWE = .149`，baseline 最强峰 `[-64, -52, -2]` 的 `p_FWE = .387`。因此，S3 的正式写法应是：全脑补充证据支持 `YY` 的 pair-level differentiation，但 RD 几何指标未形成 FWE 显著定位；且目前没有直接的 `YY - KJ` 全脑交互 searchlight。

gPPI 本轮只执行 `export_gppi_summary.py` 轻量导出，raw gPPI 模型沿用既有产物。两个左颞极 seed、`literature + literature_spatial` 共 24 个 target、`yy/kj`、pre/post、ridge deconvolution 的主汇总仍为阴性。6 个 planned `yy_minus_kj` scope 主检验均不显著，且 sign-flip permutation 后再 BH-FDR 也完全阴性，最小 `q_within_main_tests = .770`。per-target 层面也没有通过 family FDR 的 target；最接近的 `yy_minus_kj` target peak 仍未显著，例如 seed2 到 `litspat_L_PPC` 的 `mean_diff = 4.90`, `p_permutation = .144`, `q = .982`。因此，S2 当前仍应写成阴性边界结果：没有证据支持稳定的左颞极 seed 网络重配置，核心证据仍来自 ROI 内表征重组。

本节依据产物：
- [table_searchlight_pair_similarity_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv)
- [fig_searchlight_pair_similarity_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png)
- [pair_similarity_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/pair_similarity_searchlight_summary.tsv)
- [searchlight_subject_pairing_manifest.tsv: pair similarity](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/searchlight_subject_pairing_manifest.tsv)
- [table_searchlight_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_peaks.tsv)
- [fig_searchlight_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_delta.png)
- [rd_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/rd_searchlight/rd_searchlight_summary.tsv)
- [searchlight_subject_pairing_manifest.tsv: RD](E:/python_metaphor/paper_outputs/qc/rd_searchlight/searchlight_subject_pairing_manifest.tsv)
- [table_gppi_summary_main.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_summary_main.tsv)
- [table_gppi_scope_permutation.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_scope_permutation.tsv)
- [table_gppi_target_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_target_peaks.tsv)
- [table_gppi_target_level.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_target_level.tsv)
- [gppi_summary_export_manifest.json](E:/python_metaphor/paper_outputs/qc/gppi_summary_export_manifest.json)

## 1、行为结果总结：

当前行为部分已采用 refined 版本结果作为正式口径。主行为终点是 run-7 的 `memory`，补充终点是正确反应 trial 上的 `log_rt_correct`。这里需要先固定一个统计口径：`memory` 是二值变量，因此本轮重跑后正式推断已经从旧版 Gaussian MixedLM 改为按被试聚类的 `binomial GEE`；`log_rt_correct` 仍作为连续变量，用 Gaussian mixed model 拟合。也就是说，行为主效应现在可以写成“trial-level binomial GEE + Wald z”，而不是线性概率混合模型。

在 27 名被试、3780 条 `YY/KJ` run-7 trial 中，`YY` 条件的平均正确率为 `0.680`，高于 `KJ` 条件的 `0.588`，原始差值为 `+0.092`。正式 GEE 模型 `memory ~ C(condition)` 显示 `YY` 相对 `KJ` 的 log-odds 显著更高，`b = 0.398`, `SE = 0.084`, `z = 4.72`, `p = 2.37e-06`，对应 odds ratio 约为 `1.49`。因此，行为主结果可以稳定写成：被试在隐喻联结条件下的最终记忆表现显著优于空间联结条件。

补充行为结果同样支持 `YY` 的行为优势。若只考察答对 trial 的反应时，则 `YY` 条件下的平均正确反应时为 `1070.43 ms`，快于 `KJ` 条件的 `1100.29 ms`，原始差值约为 `-29.86 ms`。对应的 `log_rt_correct ~ C(condition)` 混合模型显示 `YY` 的正确提取速度也显著更快，`b = -0.026`, `SE = 0.010`, `z = -2.61`, `p = .009`。这说明 `YY` 的行为优势并不只体现为“更准”，也体现为答对项目上的提取更快。

从整条证据链的角度看，行为层最稳的结论是：`YY` 在 run-7 的最终测验中表现出更优的学习结果，这种优势同时体现在正确率和正确反应提取效率上。因此，后续把神经结果与行为结果放在一起解释时，更合适的写法是：隐喻学习在组水平上同时带来了更好的行为表现和更强的学习后表征重组；但行为优势本身并不自动说明 item-level 的神经变化一定逐项预测 memory accuracy。

本节依据产物：
- [behavior_trials.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv)
- [subject_condition_summary.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_condition_summary.tsv)
- [run7_memory/behavior_lmm_params.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/lmm/run7_memory/behavior_lmm_params.tsv)
- [run7_memory/behavior_lmm_model.json](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/lmm/run7_memory/behavior_lmm_model.json)
- [run7_log_rt_correct/behavior_lmm_params.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/lmm/run7_log_rt_correct/behavior_lmm_params.tsv)
- [fig_behavior_accuracy_rt.png](E:/python_metaphor/paper_outputs/figures_main/fig_behavior_accuracy_rt.png)

### 1.1 被试内 item-level 脑-行为分析

脑-行为关联部分已经从旧的 subject-level 差值相关升级为 within-subject item-level 分析。具体做法是：先把 refined run-7 行为 trial 通过 `stimuli_template.csv` 映射到 `pair_id` 与规范化 `word_label`，再把每个行为 item 连接到同一 `subject × ROI × condition × pair_id × word_label` 下的 neural `delta_similarity_post_minus_pre = post - pre`。新版代码已经修复了旧 join 的关键问题：RSA item-wise 表中每个 pair 只输出一行，第一词在 `word_label`，第二词在 `partner_label`；现在 neural pair delta 会展开成两行，使 pair 中两个词都能进入 item-level join。

最新联表质量如下。27 名被试的 refined 行为表共提供 `3780` 条 `YY/KJ` trial；三套 ROI 全部纳入后，共形成 `117180` 条 `subject × roi × word` 级记录。其中 `main_functional` 为 `26460` 行、`7` 个 ROI；`literature` 为 `45360` 行、`12` 个 ROI；`literature_spatial` 为 `45360` 行、`12` 个 ROI。换句话说，现在每条行为 trial 都成功接入了 `7 + 12 + 12 = 31` 个 ROI 的 neural predictor，已经不再丢掉每个 pair 的第二个词。

主模型使用 trial-level `GEE binomial` 拟合 `memory ~ delta_similarity_within + delta_similarity_subject_mean`，其中 `delta_similarity_within` 表示同一被试、同一 ROI、同一条件内部的 item-level 偏离量。结果显示，item-level neural change 并没有形成一个 `YY` 特异、跨 ROI 一致的 memory accuracy 耦合。显著的 memory slope 主要集中在 `Spatial/KJ` 条件，且系数为负：这意味着 neural `post - pre` 越低，也就是更强的 similarity decrease / differentiation，越倾向于对应更好的 memory。

memory accuracy 中达到未校正 `p < .05` 的 within-subject item-level 结果如下。这里的作用是定位局部脑-行为耦合线索，而不是给出跨全部 ROI/条件/指标的强校正阳性结论；因此表述时应避免把单个 ROI 的传统显著直接升级为主结论。

| ROI set | ROI | Condition | b | p | 解释 |
| --- | --- | --- | --- | --- | --- |
| `main_functional` | 右后海马旁回 | Spatial | -0.092 | .012 | KJ 中更强 neural decrease 预测更高 memory |
| `main_functional` | 左后梭状回 | Spatial | -0.085 | .022 | 同上 |
| `literature` | 左 IFG | Spatial | -0.083 | .011 | 文献语义控制区中 KJ 耦合显著 |
| `literature` | 右 temporal pole | Spatial | -0.055 | .032 | KJ 耦合显著 |
| `literature` | 左 precuneus | Spatial | -0.084 | .045 | KJ 耦合显著 |
| `literature_spatial` | 左 PPC | Spatial | -0.087 | <.001 | 空间文献 ROI 中最强的 KJ 耦合之一 |
| `literature_spatial` | 右 PPC | Spatial | -0.106 | .006 | 空间文献 ROI 中 KJ 耦合显著 |
| `literature_spatial` | 左 precuneus | Spatial | -0.084 | .045 | 与文献 ROI 的左 precuneus 一致 |

交互模型整体上仍然支持谨慎解释。大多数 ROI 的 `condition × delta_similarity_within` 交互不显著；唯一达到传统显著的是 `literature_spatial` 的左 PPC，`C(condition)[T.Spatial] × delta_similarity_within_z: b = -0.128, p = .017`，方向表明 Spatial 条件的负向 memory coupling 强于 Metaphor 条件。这一结果进一步提示：当前 item-level accuracy 耦合不是“YY 分化越强，记忆越好”的简单命题；相反，显著 accuracy 耦合更集中在空间条件及其空间相关 ROI 中。

补充的正确 trial RT 分析呈现出更接近“提取效率”的模式。由于主神经预测量定义为 `post - pre`，正系数表示 neural delta 越偏向上升或越少下降，正确反应越慢；反过来说，更强的 neural decrease 与更快的正确提取相关。RT 模型中，`YY/Metaphor` 条件出现了更多局部未校正显著的正向耦合：`literature` 的左 AG、左 IFG、右 IFG、右 pSTS，`literature_spatial` 的左 PPC、右 PPA、双侧 hippocampus，以及 `main_functional` 的右后海马旁回均达到 `p < .05`。此外，`Spatial` 条件下 `literature_spatial` 的右 hippocampus 与右 PPA 也显著。RT 交互项没有达到 `p < .10`，因此这些 RT 信号更适合写成局部效率调制，而不是条件差异的强证据。

因此，当前脑-行为部分最稳妥的写法是：within-subject item-level 分析并没有把主结果改写成“YY 中分化越强，memory accuracy 越高”。它一方面确认脑-行为关系不是一个强而直接的 accuracy 耦合；另一方面又显示 neural differentiation 与行为并非完全无关，而是更可能在若干局部 ROI 中体现为对正确提取效率的调制。主线仍然应放在“YY 在组水平上更好学、并在神经表征上表现出更强去相似化/重组”；脑-行为耦合则作为局部、次级、需要谨慎表述的补充证据。

本节依据产物：
- [itemlevel_behavior_join_qc.tsv](E:/python_metaphor/paper_outputs/qc/itemlevel_behavior_join_qc.tsv)
- [itemlevel_coupling_manifest.json](E:/python_metaphor/paper_outputs/qc/itemlevel_coupling_manifest.json)
- [itemlevel_behavior_neural_long.tsv](E:/python_metaphor/paper_outputs/qc/itemlevel_behavior_neural_long.tsv)
- [itemlevel_neural_pair_delta.tsv](E:/python_metaphor/paper_outputs/qc/itemlevel_neural_pair_delta.tsv)
- [table_itemlevel_coupling.tsv](E:/python_metaphor/paper_outputs/tables_main/table_itemlevel_coupling.tsv)
- [table_itemlevel_coupling_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_full.tsv)
- [table_itemlevel_coupling_interaction.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_interaction.tsv)
- [table_itemlevel_coupling_rt.tsv](E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_rt.tsv)
- [fig_itemlevel_coupling.png](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling.png)
- [fig_itemlevel_coupling_rt.png](E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling_rt.png)

### 1.2 S0 刺激控制分析

材料控制部分现在只保留既有的历史材料评分结果，不再纳入本轮额外整理的刺激控制内容，也不在行为或神经模型中增加材料控制项。也就是说，这一节只作为材料合理性的背景说明，不作为后续统计模型的一部分。

历史评分表 `实验材料+匹配结果.xlsx` 提供了可理解度、难度、新颖性、熟悉度、词对关联度和词对熟悉度等人工评分。在当前正式刺激中，历史句子评分覆盖 `19/35` 个 YY cue 和 `17/35` 个 KJ cue；词对评分覆盖 `18/35` 个 YY pair、`18/35` 个 KJ pair，并完整覆盖 `20/20` 个 baseline pair。因此，这部分结果应按“已有材料评分子集”表述，而不是完整刺激全集控制。

在这个可用子集中，YY/KJ 的可理解度相近（`6.60` vs `6.53`, p = .665），难度相近（`1.67` vs `1.61`, p = .663），人工熟悉度也无显著差异（`4.54` vs `4.85`, p = .300）。词对关联度和词对熟悉度在 YY/KJ/baseline 间也无显著差异（association p = .666；pair familiarity p = .975）。历史评分中唯一清楚的差异是 YY 句子新颖性高于 KJ（`4.07` vs `2.53`, p = 6.1e-05），这与隐喻材料本身的新颖关系属性一致，可作为材料特征和解释边界报告。

因此，当前材料部分的写法应保持轻量：已有评分结果显示，YY/KJ 在可理解度、难度、熟悉度、词对关联度和词对熟悉度上没有明显差异；YY 的更高新颖性是隐喻材料的理论相关属性，而不是本轮需要进入行为或神经模型的协变量。后续行为、RSA、Model-RSA、gPPI 与 searchlight 分析均不再加入刺激材料控制项。

本节依据产物：
- [table_stimulus_control_from_materials.tsv](E:/python_metaphor/paper_outputs/tables_si/table_stimulus_control_from_materials.tsv)
- [table_stimulus_balance_materials_available.tsv](E:/python_metaphor/paper_outputs/tables_si/table_stimulus_balance_materials_available.tsv)
- [stimulus_control_materials_coverage.tsv](E:/python_metaphor/paper_outputs/qc/stimulus_control/stimulus_control_materials_coverage.tsv)
- [stimulus_control_from_materials_manifest.json](E:/python_metaphor/paper_outputs/qc/stimulus_control/stimulus_control_from_materials_manifest.json)

## 2、学习阶段全脑 GLM 结果总结（Step 3B: `main_analysis.py`）

在 28 名被试的学习阶段（run-3/4）全脑二阶分析中，我们采用了基于 5000 次置换检验的 cluster-level FWE 校正，cluster-forming threshold 设为 `p < .001`，显著性阈值为 cluster-level `FWE < .05`。结果显示，`Metaphor > Spatial` 与 `Spatial > Metaphor` 两个对比均获得了显著簇。

对于 `Metaphor > Spatial`，显著效应主要集中在左侧颞极区域。具体而言，检测到两个显著簇，峰值分别位于左侧颞极（MNI: `-52, 6, -24`，cluster size = `1592 mm³`）和左侧前颞叶/颞极邻近区域（MNI: `-38, 12, -36`，cluster size = `392 mm³`）。这一结果提示，在学习阶段，隐喻联结相较于空间联结更强地募集了左侧前颞语义相关区域。

对于 `Spatial > Metaphor`，显著效应范围更广，主要分布在后部视觉-内侧颞叶-顶叶网络。具体包括：左侧后部梭状回（MNI: `-28, -38, -16`，cluster size = `4056 mm³`）、左侧楔前叶（MNI: `-10, -54, 8`，cluster size = `3648 mm³`）、右侧楔前叶（MNI: `12, -52, 10`，cluster size = `2544 mm³`）、右侧后部海马旁回（MNI: `28, -32, -18`，cluster size = `2440 mm³`），以及左侧外侧枕叶上部（MNI: `-38, -78, 30`，cluster size = `2200 mm³`）。这一模式表明，空间联结学习相较于隐喻联结，更依赖后部视觉表征、场景/空间加工以及顶叶相关区域。

总体上，这一步的全脑 GLM 结果为后续 ROI 级表征分析提供了“空间定位”证据：隐喻条件的优势更偏向左侧前颞语义系统，而空间条件的优势则更多体现在后部视觉-内侧颞叶-顶叶网络。相应的 ROI mask 已在 `glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/` 下生成，可作为后续 RSA、RD/GPS、MVPA 和连接分析的 ROI 来源。

## 3、GLM结果与已有隐喻/空间加工研究的对照解释

从已有隐喻研究看，当前 `Metaphor > Spatial` 主要落在左侧颞极，并不构成明显异常，反而与“隐喻理解更依赖左侧语义整合/语义枢纽系统”的一类结果是相容的。已有文献和本地整理都提示，隐喻加工并不稳定地要求右半球特异参与，其激活格局会受到到到新颖性、熟悉度、任务要求和刺激语境到到的调节。对于新颖隐喻，Mashal 等（2007）更强调右侧PSTS/右IFG的参与；但 Cardillo 等（2012）和 Lai 等（2015）提示，当隐喻逐渐熟悉，或当任务并非强烈要求新颖意义重解释时，隐喻加工更常表现为以左侧语言-语义网络为主的模式。结合你当前的实验，学习阶段的 `yy` 条件已经经历了反复联结建立，因此没有出现很强的右半球优势，本身并不违背文献。

同时，你现在的 `Metaphor > Spatial` 结果比一些经典隐喻研究更“窄”，主要突出左侧颞极，而没有在 cluster-FWE 主结果中明显出现左IFG、左MTG/STS 或 AG。这一点点点不一定是错误点点，但提示当前效应更像是“前颞语义枢纽优势”，而不是完整铺开的经典隐喻网络。结合本地方案文档，原始理论预期里 IFG、AG、Precuneus 等也被视为重要候选区，因此后续最好把这一点写成成成保守解释成成：学习阶段全脑结果首先支持了左前颞语义系统参与隐喻联结学习，但更广泛的语言控制/语义整合网络是否同样参与，还需要后续 ROI 级 RSA、RD/GPS、MVPA 分析进一步补充。

对于 `Spatial > Metaphor`，你现在看到的后部梭状回、双侧楔前叶、后海马旁回和外侧枕叶网络，和空间/场景加工文献是高度一致的。无论是本地方案笔记，还是关于空间语言、场景表征和视空间联结学习的研究，都反复指向向向楔前叶/顶叶的空间表征与心理意象功能向向，以及及及海马旁回/后部梭状回的场景、地点与空间关联编码功能及及。因此，`kj > yy` 落在这一后部视觉-内侧颞叶-顶叶网络，从认知神经机制上是非常合理的，并不显示明显异常。

如果说当前结果有什么需要谨慎而不是“明显有问题”的地方，主要有三点。第一，你的主对比是   隐喻联结 vs 空间联结  ，而不是隐喻 vs 字面/无关条件，因此这里更适合被解释为“隐喻学习相对于空间学习更偏向语义系统，而空间学习更偏向视空间-场景系统”，而不是直接下结论说“这些区域就是隐喻加工专属脑区”。第二，`Spatial > Metaphor` 的效应范围明显大于 `Metaphor > Spatial`，这可能说明空间材料在视觉意象、场景表征或联结稳定性上具有更直接的后部皮层优势，也可能提示两类材料在意象性/可视化程度上并不完全等价。第三，当前全脑 GLM 结果本身更像是是是定位证据（where）是是，是否真的对应“表征重组（how）”，还要看你后面的 RSA、RD/GPS 和脑-行为关联是否能在这些 ROI 中给出同向支持。

可作为这一判断的代表性文献包括：Mashal et al. (2007) 强调调调新颖隐喻调调更容易招募右侧PSTS/右IFG，而 Cardillo et al. (2012) 与 Lai et al. (2015) 都提示，随着熟悉度上升或在控制新颖性/难度后，隐喻加工并不稳定地表现为右半球优势，反而更常回到以左侧语言-语义系统为主的模式；Rapp et al. (2004) 也报告了隐喻相对字面句更多激活左侧IFG和颞叶语义区。相对地，场景/空间加工研究一致指出海马旁回、枕叶场景区和楔前叶在空间布局、场景知觉和视空间意象中具有核心作用。因此，你当前“左前颞偏隐喻、后部视觉-内侧颞叶-顶叶偏空间”的总格局，与经典文献总体上是相容的，只是隐喻效应更偏向语义枢纽、范围也比部分新颖隐喻研究更收敛。

<br />

## 4、前后测 ROI-level RSA 结果（Step 5C: `run_rsa_optimized.py` + `rsa_lmm.py`）

### 4.1 分析方法简介

这一部分的核心目标，不是再去比较“哪个脑区激活更强”，而是检验验验学习是否改变了 ROI 内部的表征结构验验。具体做法是：首先基于前后测阶段的单 trial LSS beta，在每个 ROI 内提取每个词项目的多体素模式；随后计算项目间的 pattern similarity。这里采用的是基于 Pearson 相关的相似性指标，即先计算 correlation distance，再转换为 similarity，因此数值越高表示两项词在该 ROI 中的神经模式越相似。接着，我们从完整的相似度矩阵中抽取 item-wise 指标：对 `yy/kj` 来说，核心是“同一学习配对内部”的相似度，例如 `yyw_1` 与 `yyew_1`、`kjw_1` 与 `kjew_1`；对 baseline 来说，当前主流程使用模板中预定义的 `jx` 伪配对（如 `jx_1` 对 `jx_2`）作为控制分布，而不是把所有 baseline trial 两两混合。

在统计分析上，我们进一步采用线性混合模型检验 `similarity ~ condition * time + (1|subject) + (1|item)`。其中，`condition × time` 交互是最关键的效应，因为它直接回答：隐喻联结与空间联结在学习前后，其配对项目的神经相似性变化是否不同。与单纯的主效应相比，这一交互更能对应本研究的核心问题，即隐喻学习是否引发了不同于空间学习的表征重组。

需要特别说明的是：本节 `main_functional`、`literature` 与 `literature_spatial` 三层 ROI 结果都已经更新为为为三条件正式版本为为。因此，当前第 4 节的 RSA 叙事应以 `yy / kj / baseline(jx)` 同时进入默认主链的 Step 5C 输出为准。三套 ROI 的结果虽然来源不同，但方向高度一致：当前主故事更接近   representational differentiation（表征分化增强）  ，而不是简单的配对收敛。

### 4.2 主功能 ROI（`main_functional`）结果

`main_functional` 的 Step 5C 现在已经是是是三条件版本是是。当前主链输入来自 `rsa_itemwise_details.csv`，共 28 名被试、7 个主功能 ROI、35280 条 item-wise 记录；其中 baseline 不是额外旁支，而是与 `Metaphor / Spatial` 一起进入同一个 LMM。由于 7 个 ROI 来自两类不同的 GLM 对比家族，脚本默认按 `base_contrast` 分层拟合，因此本节的正式统计结论以 `Metaphor > Spatial` 家族（2 个 ROI）和 `Spatial > Metaphor` 家族（5 个 ROI）的模型为主，再辅以单 ROI 结果。

先看三条件的总体描述性模式，故事已经很清楚。跨 7 个主功能 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1725 下降到 Post 的 0.1013，下降幅度最大；`Spatial` 仅从 0.1538 轻度下降到 0.1455；`Baseline` 则从 0.0885 小幅上升到 0.1039。也就是说，当前数据最稳定的变化不是“学习后同 pair 更像”，而是   yy 在学习后变得更不相似  ；与此同时，baseline 没有出现同方向下降，因此这不像是一个普遍的重测/熟悉化效应。

在 ROI 家族层面，这一模式在两类主功能 ROI 中都成立。对于来源于 `Metaphor > Spatial` 的左前颞 ROI 家族，baseline 从 0.0635 上升到 0.1166，而 `Metaphor` 从 0.1699 降到 0.1121；对应的 `C(condition)[T.Metaphor]:C(time)[T.Pre]` 交互显著，b = 0.1109, SE = 0.0172, 95% CI \[0.0772, 0.1446], p < .001。这里之所以系数为正，是因为模型把 `Baseline` 与 `Post` 设为参考水平；换成更直观的话说，就是   Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化  。相比之下，同一家族中 `Spatial` 从 0.1363 轻度上升到 0.1571，`Spatial × Pre` 仅边缘显著，b = 0.0323, p = .061。

对于来源于 `Spatial > Metaphor` 的后部视觉-顶叶-内侧颞叶 ROI 家族，故事同样一致，但 baseline 更稳定：baseline 几乎不变（Pre = 0.0985, Post = 0.0987），`Metaphor` 则从 0.1735 降到 0.0970；对应的 `Metaphor × Pre` 交互高度显著，b = 0.0768, SE = 0.0105, 95% CI \[0.0563, 0.0973], p < .001。`Spatial` 在该家族中只表现为较弱下降（0.1608 → 0.1409），其 `Spatial × Pre` 仍仅为边缘显著，b = 0.0202, p = .054。换言之，无论 ROI 是从“隐喻更强激活”的左前颞系统中定义，还是从“空间更强激活”的后部视觉-顶叶-内侧颞系统中定义，最稳定的结果都不是 `kj` 或 `baseline` 的改变，而是 `yy` 的明显去相似化。

单 ROI 结果进一步说明，这一效应并不是由某一个特定脑区偶然驱动。7 个主功能 ROI 中，`Metaphor × Pre` 项在全部 7 个 ROI 内都达到显著，效应范围约为 b = 0.0604–0.1138，p = .023 到 p < .001。两个左侧颞极 ROI 的效应最强，左后梭状回、双侧楔前叶、右后海马旁回和左外侧枕叶也都呈现同方向结果。相反，`Spatial × Pre` 在单 ROI 层面均未达到显著。这说明当前主功能 ROI 的 Step 5C 结果具有很高的一致性：：：真正系统性发生前后重组的是隐喻配对，而不是空间配对或 baseline 配对。：：

因此，`main_functional` 的当前结果故事可以概括为：学习后最显著的神经表征变化，并不是 pair similarity 的普遍升高，而是 `yy` 在多个主功能 ROI 中出现了稳定的 similarity 下降；`baseline` 基本稳定，部分 ROI 甚至轻度上升；`kj` 变化较弱且不稳定。这个模式对论文主线有两个直接含义。第一，它用 baseline 封堵了“纯重测/熟悉化”解释，使得当前的 `yy` 效应更像像像学习特异性的 representational differentiation像像。第二，它为后续 Model-RSA 提供了明确约束：Step 5D 不应再默认押注“学习后同 pair 更收敛”，而应重点检验当前这种 post 阶段更强去相似化，究竟更符合 differentiation 路径，还是某种在预存语义底盘之上发生的重加权重组。

### 4.3 论文 ROI（`literature`）结果

在文献 ROI 集（`literature`）上，Step 5C 同样已经是三条件版本。当前输入包含 28 名被试、12 个理论 ROI、60480 条 item-wise 记录。由于这套 ROI 并不来自同一组 GLM 对比家族，脚本没有再做 `base_contrast` 分层，而是直接按 ROI 分别拟合 LMM。因此，`literature` 层的重点不在“一个 pooled 总模型”，而在于：：：理论 ROI 是否在方向上表现出高度一致的重复支持：：。

先看总体描述性模式，文献 ROI 与主功能 ROI 的方向几乎完全一致，而且隐喻效应更突出。跨 12 个文献 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1731 下降到 Post 的 0.1074；`Spatial` 则从 0.1576 上升到 0.1694；`Baseline` 从 0.0726 上升到 0.1025。也就是说，在文献 ROI 中不仅 `yy` 出现了清晰的 post 下降，而且 `baseline` 与 `kj` 都没有呈现同方向下降，反而整体略有上升。这个模式比主功能 ROI 更强地表明：当前观察到的 `yy` 去相似化并不是一个普遍的前后测趋势，而是更接近隐喻学习特异性的表征变化。

在单 ROI 层面，这种一致性非常高。12 个文献 ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 12 个 ROI 内都达到显著，效应范围约为 b = 0.0635–0.1212，p = .0066 到 p < .001。具体而言，左侧 AG（b = 0.1042, p < .001）、IFG（b = 0.1027, p < .001）、pMTG（b = 0.0861, p = .001）、pSTS（b = 0.0947, p < .001）、precuneus（b = 0.0635, p = .007）和 temporal pole（b = 0.1212, p < .001）全部显著；右侧 AG（b = 0.1095, p < .001）、IFG（b = 0.1068, p < .001）、pMTG（b = 0.0984, p < .001）、pSTS（b = 0.0706, p = .002）、precuneus（b = 0.0728, p = .003）和 temporal pole（b = 0.1169, p < .001）也全部显著。按参考组解释，这些正向交互都意味着：：：Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化：：。

相比之下，`Spatial × Pre` 在文献 ROI 中并不稳定。12 个 ROI 里没有任何一个 ROI 在传统阈值下表现出稳健的 `Spatial × Pre` 效应；最接近显著的是左 IFG（b = 0.0396, p = .050）以及左/右 AG（p ≈ .084）等少数 ROI，但整体上都不足以支持一个系统性的 `kj` 前后重组模式。换句话说，文献 ROI 并没有把主结果改写成“隐喻和空间都在同等程度上变化”，而是进一步强化了这样一个结论：：：真正稳定、广泛发生的，是隐喻配对的去相似化，而不是空间配对或 baseline 配对。：：

### 4.4 空间文献 ROI（`literature_spatial`）结果

为了进一步检验“`kj` 的空间联结是否会在经典空间网络中表现出更明显的前后测去相似化”，我们又构建并分析了专门面向空间/场景/导航文献的 `literature_spatial` ROI 集。该分析包含 28 名被试、12 个空间理论 ROI、60480 条 item-wise 记录；ROI 包括双侧 OPA、PPA、PPC、precuneus、RSC 和 hippocampus。与 `literature` 一样，这套 ROI 并非来自同一组 GLM 对比家族，因此正式统计以单 ROI LMM 为主，而不是 pooled 总模型。

先看总体描述性模式，结果并没有支持“空间 ROI 中 `kj` 会显著下降”的预期。跨 12 个空间 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1623 下降到 Post 的 0.1018，下降幅度依然最大；`Spatial` 则从 0.1480 轻度上升到 0.1532；`Baseline` 从 0.0802 上升到 0.0963。也就是说，即使把 ROI 限定在最偏向空间认知的网络中，最稳定的前后测变化仍然不是 `kj` 的下降，而是 `yy` 的显著去相似化。

单 ROI 层面的结果进一步强化了这一点。12 个空间 ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 12 个 ROI 内都达到显著，效应范围约为 b = 0.0635–0.0956，p = .0066 到 p < .001。具体而言，左侧 OPA（b = 0.0751, p = .004）、PPA（b = 0.0734, p = .002）、PPC（b = 0.0715, p = .001）、RSC（b = 0.0662, p = .005）、hippocampus（b = 0.0693, p < .001）和 precuneus（b = 0.0635, p = .007）均显著；右侧 OPA（b = 0.0937, p < .001）、PPA（b = 0.0823, p = .001）、PPC（b = 0.0766, p < .001）、RSC（b = 0.0801, p < .001）、hippocampus（b = 0.0956, p < .001）和 precuneus（b = 0.0728, p = .003）也全部显著。按参考组解释，这些正向交互都意味着：：：Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化：：。

相比之下，`Spatial × Pre` 在 12 个空间 ROI 中没有一个达到显著。尽管个别 ROI 的原始均值出现了轻微下降，例如左 OPA（0.1758 → 0.1699）、左 precuneus（0.1813 → 0.1575）或右 precuneus（0.1610 → 0.1455），但这些变化在混合模型中都不稳定，`C(condition)[T.Spatial]:C(time)[T.Pre]` 的 p 值全部大于 .16。与此同时，也有若干经典空间 ROI 反而呈现 `kj` 的轻度上升，例如双侧 PPA、右 OPA、双侧 hippocampus 和右 RSC。因此，这套专门为空间认知构建的 ROI 集并没有把故事改写成“空间联结在空间网络里显著去相似化”，而是再次重复了与前两套 ROI 一致的结论：：：最稳定的学习后重组依然发生在：： ：：`yy`，而不是：： ：：`kj`。：：

从解释上看，这一结果的意义非常关键。它说明当前 `kj` 效应偏弱，并不能简单归因为“之前选的 ROI 不够空间化”。即便把分析范围明确收缩到 OPA/PPA/RSC/hippocampus/precuneus/PPC 这样的经典空间网络，`kj` 仍未表现出跨 ROI、一致、可靠的 pre→post similarity 下降。相反，`yy` 的去相似化却在这些空间 ROI 中依旧稳健存在。这意味着，Step 5C 当前支持的不是“不同类型材料分别在各自优势网络内对称重组”，而更接近一种种种对隐喻学习更敏感、跨网络可见的表征几何重塑种种。

### 4.5 Trial / voxel 稳健性检查

为了进一步排除“`yy` 的 pre→post 去相似化只是由 trial 数或 ROI 体素数不平衡造成”的解释，我们又对三层 ROI 结果做了专门的 trial/voxel 稳健性检查。该检查分别基于 `pattern_root/*_metadata.tsv`、`rsa_itemwise_details.csv` 和 `rsa_summary_stats.csv` 统计三类信息：第一，前后测阶段每位被试在 `yy / kj / baseline` 下的原始 trial 数是否完全匹配；第二，进入 item-wise RSA 之后，每个 `subject × ROI × stage × condition` 的 downstream pair 数是否仍然完全匹配；第三，每个 ROI 在 pre 与 post 阶段用于 RSA 的 `n_voxels` 是否存在系统漂移。

结果显示，这三类平衡性在三套 ROI 中都非常干净，而且不是“均值上差不多”，而是逐被试逐 ROI 的的的完全匹配的的。以原始 trial 数为例，三层 ROI 的 `trial_metadata_stage_tests.tsv` 都显示：所有被试在 `yy` 条件下均为 `Pre = 70, Post = 70`，在 `kj` 条件下均为 `Pre = 70, Post = 70`，在 `baseline` 条件下均为 `Pre = 40, Post = 40`，且 `exact_match_all_subjects = True`。在 downstream item-wise 层，所有 ROI 都表现为 `Metaphor = 35 vs 35`、`Spatial = 35 vs 35`、`Baseline = 20 vs 20`，同样全部 `exact_match_all_subjects = True`。这意味着当前 Step 5C 的 `yy` 去相似化并不是由 pre/post 条件下进入分析的项目数不平衡造成的。

体素数稳健性检查也给出了同样清晰的结论。`main_functional`、`literature` 与 `literature_spatial` 三层 ROI 的 `voxel_counts_stage_tests.tsv` 均显示：每个 ROI 的 `n_voxels` 在 pre 与 post 阶段完全一致，`post_minus_pre_mean = 0`，`ttest_p = 1.0`，且 `exact_match_all_subjects = True`。例如，主功能 ROI 中左 Temporal Pole 两个 ROI 分别稳定为 `199` 和 `49` 个体素，左后梭状回为 `507`，双侧 precuneus 为 `456` 与 `318`；文献 ROI 中左/右 IFG 分别稳定为 `1336` 和 `1151` 个体素；空间 ROI 中双侧 OPA 稳定为 `4955` 和 `4824` 个体素，双侧 hippocampus 稳定为 `932` 和 `946` 个体素。换句话说，当前主结果同样不能被解释为“post 阶段进入 RSA 的体素数更少或更多，从而机械地改写了 similarity”。

因此，这一步稳健性分析对当前主线提供了非常直接的支持：`yy` 的 pre→post 去相似化并不是由原始 trial 数失衡、downstream pair 数失衡，或 `n_voxels` 漂移造成的。结合前面的 baseline 控制、QC/reliability 结果和三层 ROI 的重复支持，当前更稳妥的判断是：：：`yy`：： ：：的去相似化更可能反映真实的学习后表征重组，而不是由设计平衡性或 ROI 采样偏差驱动的假象。：：

### 4.6 `atlas_robustness` 结果

为了进一步确认当前主结论并不依赖于现有 ROI 的定义方式，我们又在一套更“粗粒度”、更接近标准 atlas parcel 的 ROI 集上重复了 Step 5C。`atlas_robustness` 当前包含 9 个 atlas ROI：左 AG、左 IFGop、左 IFGtri、左 fusiform、左 pMTG、左 precuneus、左 temporal pole、右 parahippocampal 以及右 precuneus。与前三套 ROI 一样，这一层同样使用包含 `Metaphor / Spatial / Baseline` 的默认三条件 item-wise LMM 主链，因此这里要回答的问题非常直接：：：如果把 ROI 改写成 atlas 级 parcel，`yy`：： ：：的学习后去相似化是否仍然存在。：：

答案是肯定的。跨 9 个 atlas ROI 的描述性均值显示：`Metaphor` 的 pair similarity 从 Pre 的 `0.1725` 下降到 Post 的 `0.1072`；`Spatial` 则从 `0.1593` 轻微上升到 `0.1619`；`Baseline` 从 `0.0777` 上升到 `0.1039`。这一方向与前面的 `main_functional`、`literature` 和 `literature_spatial` 基本完全一致：最清晰的前后变化依旧是 `yy` 的下降，而不是 `kj` 或 baseline 的同步下降。换句话说，把 ROI 换成 atlas parcel 并没有把结果改写成“所有条件都一起下降”，而是继续保留了隐喻学习特异性的去相似化模式。

单 ROI 层面的证据同样稳定。9 个 atlas ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 9 个 ROI 内都达到显著，效应范围约为 `b = 0.0586–0.1107`，`p = .0158` 到 `p < .001`。具体而言，左 AG（`b = 0.0972, p < .001`）、左 IFGop（`b = 0.1030, p < .001`）、左 IFGtri（`b = 0.0977, p < .001`）、左 fusiform（`b = 0.1107, p < .001`）、左 pMTG（`b = 0.0951, p < .001`）、左 precuneus（`b = 0.0770, p = .002`）、左 temporal pole（`b = 0.1078, p < .001`）、右 parahippocampal（`b = 0.0762, p = .001`）和右 precuneus（`b = 0.0586, p = .016`）全部显著。这意味着，即便采用 atlas 级 ROI，`Metaphor` 的 pre→post 下降仍然系统性大于 baseline 的 pre→post 变化。

相比之下，`Spatial × Pre` 只有 3 个 atlas ROI 达到显著，分别是左 AG（`b = 0.0647, p = .010`）、左 IFGop（`b = 0.0399, p = .046`）和左 IFGtri（`b = 0.0506, p = .011`）；其余 6 个 atlas ROI 全部不显著（`p = .33–.95`）。因此，atlas 级重复分析并没有把故事改写成“`kj` 其实和 `yy` 一样稳健，只是之前 ROI 选得不对”。更准确的结论是：：：`yy`：： ：：的去相似化在 atlas 粒度下依旧稳健，而：： ：：`kj`：： ：：只在少数左侧顶额/额下回 parcel 中出现有限信号，缺乏跨 ROI 的一致性。：：

从稳健性角度看，这一层结果非常关键。它表明当前主现象并不是某一套功能性 ROI 或理论 ROI 的偶然产物；即便改用更标准、边界更宽的 atlas parcel，`yy` 的学习后去相似化仍然重复出现。因此，ROI 定义方式的改变不会推翻 Step 5C 的核心判断，反而进一步支持：：：隐喻学习相关的表征重组是一个跨 ROI 定义方案都可观察到的稳定现象。：：

### 4.7 四层 ROI 的整合结果故事

把 `main_functional`、`literature`、`literature_spatial` 和 `atlas_robustness` 四层 ROI 放在一起看，Step 5C 现在已经形成了一条相当完整的结果链。第一，在四套 ROI 中，`yy` 都表现出最稳定的 pre→post similarity 下降。主功能 ROI 上，这种模式出现在左前颞语义家族和后部视觉-顶叶-内侧颞家族两侧；隐喻文献 ROI 上，它进一步扩展到双侧 AG、IFG、pMTG、pSTS、precuneus 和 temporal pole；空间文献 ROI 上，它又重复出现在双侧 OPA、PPA、PPC、RSC、hippocampus 和 precuneus；atlas parcel 层上，它还在左 AG、左 IFG、左 fusiform、左 pMTG、左 temporal pole、双侧 precuneus 和右 parahippocampal 中稳定重现。这说明这一效应不仅不依赖某一种理论 ROI 选择，甚至在更粗粒度的 atlas 定义下也依然保留。第二，`baseline` 在已有稳健性链条中始终没有出现与 `yy` 相同方向的下降，反而多为稳定或轻度上升，因此当前结果很难被解释为简单的重测噪声、普遍熟悉化或整体信号衰减。第三，`kj` 的变化在四套 ROI 中都明显弱于 `yy`，且缺乏跨 ROI 的一致显著性；即便在专门为空间认知整理出来的 ROI 集中，`kj` 也没有呈现稳定的 pre→post 去相似化，而 atlas 层也只有少数左侧顶额 parcel 出现有限信号。这说明本研究中真正主导 Step 5C 结果的不是“所有学习材料都在各自优势网络内对称重组”，而是是是隐喻学习诱发了更强、更跨网络、且跨 ROI 定义方案都可见的表征几何重塑是是。

因此，第 4 节目前可以支持一个清晰而聚合的结论：无论是数据驱动得到的主功能 ROI、依据隐喻文献与空间认知文献分别预先定义的两套理论 ROI，还是更标准 atlas parcel 方案，结果都指向同一方向，即隐喻配对项目在学习后并没有普遍表现为“更相似”，反而更明显地表现为神经模式相似性的下降。这说明本研究中的隐喻学习更可能对应于于于representational differentiation / reorganization于于，而不是简单的配对聚合。换言之，学习并非只是让两个词在表征空间里机械地“靠近”，而更可能是在预存语义网络的基础上，把两者嵌入到一个更精细、区分度更高的新关系结构中；而 `kj` 在经典空间网络及 atlas 稳健性层中都缺乏稳定去相似化，也进一步表明这种重组并不是一个由 ROI 选择偏差造成的表面现象。

这也为后续 Model-RSA 提供了非常明确的方向性约束。到 Step 5D 时，核心问题已经不应再表述为“学习后同 pair 会不会更像”，而应转为：“这种在 `yy` 中最明显、且被 baseline 控制支持的 post 去相似化，到底更接近 differentiation 路径，还是某种建立在原有语义底盘之上的重加权重组？” 从这个意义上说，Step 5C 不只是一个前后测现象描述，而已经为整篇论文的机制解释部分规定了方向。

## 5、Model-RSA 主流程机制解释（Step 5D: `model_rdm_comparison.py`）

这一部分对应 Step 5D 的 pooled Model-RSA 结果。当前 `main_functional`、`literature` 与 `literature_spatial` 三套 ROI 都应纳入前期结果整理。每个 `subject × ROI × time` 单元内，脚本把 `yy / kj / baseline` 三类 trial 合并构造 pooled neural RDM；结果表中的 `condition_group=all` 即表示这一 pooled 单元。统计上，`model_rdm_group_summary.tsv` 使用配对 t 检验比较 `post` 与 `pre` 的模型相关系数，其中 `mean_a = post`，`mean_b = pre`。因此，若某模型出现 `post < pre`，表示学习后该模型对神经几何结构的解释力减弱；若 `post > pre`，则表示解释力增强。

当前正式报告的模型包括：`M1_condition`（条件类别）、`M2_pair`（同 pair 接近）、`M3_embedding`（BERT 预训练语义距离）、`M7_binary`（低分辨率二值记忆强度）、`M7_continuous_confidence`（连续记忆强度：在答对项内部结合 RT 派生速度分量）、以及 `M8_reverse_pair`（同 pair 分化探针）。其中，`M2_pair` 与 `M8_reverse_pair` 是镜像式互补编码，解释时应把二者视为同一条“pair 聚合 vs pair 分化”证据链的两个方向，而不是两个相互独立的机制。

### 5.1 主功能 ROI（`main_functional`）结果

`main_functional` 层共 7 个 ROI。`M1_condition`、`M2_pair`、`M3_embedding` 与 `M8_reverse_pair` 均在 `392` 个 `subject × ROI × time` 单元中全部可估；`M7_binary` 与 `M7_continuous_confidence` 因 `sub-12` 缺少记忆强度表，均在 `378` 个单元中有效。总体上，这一层仍然给出两条最清晰的机制信号。

第一，在左侧楔前叶 ROI（`func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort`）中，`M2_pair` 从 pre 的 `0.0060` 下降到 post 的 `0.0010`，`t(27) = -3.06, p = .0049, dz = -0.58`；与之互补的 `M8_reverse_pair` 从 pre 的 `-0.0060` 上升到 post 的 `-0.0010`，`t(27) = 3.06, p = .0049, dz = 0.58`。这说明学习后该 ROI 的表征结构明显偏离了“同 pair 更接近”的组织方式，更符合 pair-based differentiation / reorganization，而不是简单的 pair convergence。

第二，在左侧颞极 ROI（`func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort`）中，`M3_embedding` 从 pre 的 `0.0132` 下降到 post 的 `-0.0061`，`t(27) = -2.74, p = .0107, dz = -0.52`，提示学习后这里的神经几何不再主要按预训练语义距离组织，而是在原有语义底盘上发生了额外重组。主功能 ROI 中的 M7 结果没有因连续化而翻转：`M7_binary` 与 `M7_continuous_confidence` 在 7 个 ROI 中均未达到显著。

### 5.2 论文 ROI（`literature`）结果

`literature` 层共 12 个理论 ROI。`M1`、`M2`、`M3` 与 `M8` 均在 `672` 个 pooled 单元中全部可估；两条 M7 在 `648` 个单元中有效。与 `main_functional` 相比，这一层的 Model-RSA 故事更集中在经典语义控制节点和后部语义节点。

最明确的 pair-based 信号出现在左 IFG：`M2_pair` 从 pre 的 `0.0098` 下降到 post 的 `0.0033`，`t(27) = -3.47, p = .0018, dz = -0.65`；镜像模型 `M8_reverse_pair` 从 pre 的 `-0.0098` 上升到 post 的 `-0.0033`，`t(27) = 3.47, p = .0018, dz = 0.65`。这与主功能 ROI 左 precuneus 的方向完全一致，说明在一个经典语义控制节点中，学习后神经几何同样脱离了“同 pair 更相似”的组织方式，更接近 pair-based differentiation。

连续 M7 在文献 ROI 中提供了一处局部阳性结果：右 pMTG 的 `M7_continuous_confidence` 从 pre 的 `0.0029` 下降到 post 的 `-0.0069`，`t(26) = -2.20, p = .037, dz = -0.42`。对应的二值版 `M7_binary` 仅为趋势（`p = .104`），说明把正确项内部的 RT 差异纳入连续记忆强度之后，M7 的灵敏度确实有所提升。除此之外，双侧 AG 的连续 M7 也达到边缘水平（`p = .078–.079`）。因此，M7 更适合被解读为局部后部语义节点上的补充信号，而不是主导全局重组的机制。

### 5.3 空间文献 ROI（`literature_spatial`）结果

`literature_spatial` 层共 12 个空间文献 ROI，最新补跑后 `model_rdm_subject_metrics.tsv` 共 `4032` 行，`model_rdm_group_summary.tsv` 共 `72` 个 ROI × model 汇总单元。与 `main_functional` 和 `literature` 不同，这一层没有任何 ROI-level Model-RSA 结果达到 `p < .05`，因此它目前更像是空间文献 ROI 的稳健性/边界检查，而不是新的强主结果来源。

不过，`literature_spatial` 中仍有若干方向上值得记录的趋势。左 hippocampus 的 `M7_continuous_confidence` 从 pre 的 `0.0022` 下降到 post 的 `-0.0060`，`t(26) = -1.85, p = .076, dz = -0.36`；同一 ROI 的 `M3_embedding` 也呈下降趋势，`t(27) = -1.81, p = .081, dz = -0.34`。左 RSC 的 `M7_binary` 则从 pre 的 `-0.0060` 上升到 post 的 `0.0035`，`t(26) = 1.80, p = .083, dz = 0.35`。右 hippocampus 的 `M2_pair` / `M8_reverse_pair` 也呈镜像趋势（`p = .093`）。这些结果说明空间文献 ROI 中存在一些与 hippocampus/RSC 相关的弱信号，但尚不足以构成可强报告的机制结论。

因此，`literature_spatial` 的当前写法应保守：它没有推翻 `main_functional` 与 `literature` 得出的 pair differentiation / semantic reweighting 主线；相反，它提示空间导航/场景记忆相关 ROI 中的 Model-RSA 效应较弱，最多在 hippocampus、RSC 等区域呈趋势性补充。

本节依据产物：
- [model_rdm_results_main_functional/model_rdm_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_subject_metrics.tsv)
- [model_rdm_results_main_functional/model_rdm_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_group_summary.tsv)
- [model_rdm_results_literature/model_rdm_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_subject_metrics.tsv)
- [model_rdm_results_literature/model_rdm_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_group_summary.tsv)
- [model_rdm_results_literature_spatial/model_rdm_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_subject_metrics.tsv)
- [model_rdm_results_literature_spatial/model_rdm_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_group_summary.tsv)
- [table_model_rsa_mechanism.tsv](E:/python_metaphor/paper_outputs/figures_main/table_model_rsa_mechanism.tsv)
- [fig_model_rsa_mechanism.png](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)

### 5.4 跨 ROI 的 Δρ LMM（`delta_rho_lmm.py`）

为了进一步检验“哪一种模型的前后变化最稳定”，我们在 `model_rdm_subject_metrics.tsv` 基础上，对每个 `subject × ROI × model` 先计算 `Δρ = ρ_post − ρ_pre`，再在 ROI 层面拟合 `delta_rho ~ C(model) + (1|subject) + (1|roi)`。这里 `M1_condition` 为参考模型，因此截距表示 `M1` 的平均 `Δρ`，其他系数表示对应模型相对 `M1` 的额外变化量。若系数为负，表示该模型比 `M1` 更倾向于在 post 阶段下降；若系数为正，则表示它比 `M1` 更倾向于上升。

三套 ROI 的 Δρ LMM 汇总如下：

| ROI set | n obs | n ROI | 关键结果 |
| --- | ---: | ---: | --- |
| `main_functional` | 1162 | 7 | `M3_embedding` 相对 `M1` 更负，`b = -0.0119`, `SE = 0.0023`, `z = -5.24`, `p = 1.58e-07`；`M2_pair` 边缘更负，`p = .098`；两条 M7 均不显著。需要注意该 pooled MixedLM 当前 `converged = false`，所以这里应作为方向性汇总/敏感性结果，而不是强裁决。 |
| `literature` | 1992 | 12 | `M2_pair` 相对 `M1` 边缘更负，`b = -0.0029`, `SE = 0.0016`, `z = -1.78`, `p = .076`；`M3` 与两条 M7 均不显著。 |
| `literature_spatial` | 1992 | 12 | `M3_embedding` 相对 `M1` 呈边缘更负趋势，`b = -0.0031`, `SE = 0.0016`, `z = -1.88`, `p = .060`；`M2/M8` 与两条 M7 均不显著。 |

从原始 `Δρ` 均值看，`main_functional` 的 `M3_embedding` 下降最明显（均值 `-0.0104`），这与左颞极和两个 family split 中的 M3 下降相互印证。`literature` 中 `M2_pair` 平均为 `-0.0021`、`M8_reverse_pair` 平均为 `+0.0021`，与左 IFG 的 ROI 级显著 M2/M8 镜像结果一致。`literature_spatial` 的整体效应更弱，主要表现为 `M3_embedding` 小幅下降趋势（均值 `-0.0031`）。因此，Δρ LMM 更适合支持一个方向性排序：主功能 ROI 中语义几何重加权最明显；文献 ROI 中 pair differentiation 最清晰；空间文献 ROI 目前提供的是弱趋势性边界证据。

需要注意的是，`main_functional` 可以按 `base_contrast` 做 family split，并且两个 family 都重复了 `M3_embedding` 更负的方向；`literature` 与 `literature_spatial` 没有同样的 `base_contrast` 字段，因此本轮采用 pooled Δρ LMM 作为汇总口径。所有新 JSON 产物已改为严格 JSON，不再包含裸 `NaN`。

本节依据产物：
- [main_functional/delta_rho_lmm_params.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_lmm_params.tsv)
- [main_functional/delta_rho_long.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_long.tsv)
- [main_functional/delta_rho_family_split_params.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_family_split_params.tsv)
- [literature/delta_rho_lmm_params.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature/delta_rho_lmm_params.tsv)
- [literature/delta_rho_long.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature/delta_rho_long.tsv)
- [literature_spatial/delta_rho_lmm_params.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature_spatial/delta_rho_lmm_params.tsv)
- [literature_spatial/delta_rho_long.tsv](E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/literature_spatial/delta_rho_long.tsv)

### 5.5 三套 ROI 的整合结论

把 ROI 级 Model-RSA、总体 Δρ LMM 与新增 `literature_spatial` 合在一起看，Step 5D 当前给出了一条更清楚、也更克制的机制约束。第一，pair-based differentiation 仍然是关键主线之一：主功能 ROI 的左 precuneus 与文献 ROI 的左 IFG 都表现为 `M2_pair` 下降、`M8_reverse_pair` 上升，说明学习后表征偏离“同 pair 聚合”，而不是简单变得更相似。

第二，semantic reweighting 是另一条稳定主线。主功能 ROI 中左颞极的 `M3_embedding` 显著下降，跨 ROI Δρ LMM 也显示 `main_functional` 的 `M3` 相对 `M1` 显著更负；`literature_spatial` 虽无显著 ROI-level 结果，但总体 Δρ 中也出现 `M3` 更负的趋势。这说明学习后的神经几何不再完全受预训练语义距离支配，而是在既有语义底盘之上引入了新的任务关系约束。

第三，M7 的连续化升级没有推翻主线，但使记忆强度几何从“整体无信号”推进到“局部可见”。`M7_continuous_confidence` 在文献 ROI 的右 pMTG 达到显著，在空间文献 ROI 的左 hippocampus 达到趋势；然而它在三套 ROI 的 Δρ LMM 中都没有形成跨 ROI 统一效应。因此，M7 当前最适合写成局部、次级、值得继续追踪的调制因素，而不是核心机制。

因此，当前 Step 5D 的总故事可以写成：隐喻学习引发的不是简单 pair convergence，而是沿着 pair-based differentiation 与 semantic reweighting 共同展开的表征重组；连续记忆强度可能在少数后部语义和海马相关节点上提供附加调制，但尚未成为跨 ROI 的主导解释。

## 6、学习阶段 repeated-exposure 表征动态（B1: `learning_dynamics.py`）
这一节回答的问题是：学习阶段本身有没有留下“同一词对被反复学习后，神经表征更稳定/更可分”的痕迹。它不是前后测 RSA 的主终点，而是用学习阶段 run 内/跨 run 的 trial pattern 做补充，帮助解释 `YY` 的行为优势是否伴随学习过程中的表征变化。

当前最重要的结论是：B1 的 primary 指标 `pair_discrimination` 在 family-level 上不强，不能单独作为主结果写成“学习阶段 pair discrimination 已显著增强”；但 supplementary 指标 `same_sentence_cross_run_similarity` 给出了较清楚的支持：`YY` 条件在若干语义/隐喻相关 ROI 中，比 `KJ` 表现出更高的跨 run 同句稳定性。

在 `literature` ROI 总体层面，`pair_discrimination` 的 `YY` 均值为 0.0058，`KJ` 为 0.0014，差异不显著，t = 1.13, p = .270, dz = 0.21。相反，`same_sentence_cross_run_similarity` 显示 `YY` 高于 `KJ`：`YY` = 0.0544，`KJ` = 0.0447，t = 2.27, p = .031, dz = 0.43。这说明如果把学习阶段看作“同一句/同一材料在不同 run 中是否形成更一致的表征”，隐喻学习条件有更稳定的学习期表征。

在 `main_functional` ROI family 中，`Metaphor_gt_Spatial` family 的 `same_sentence_cross_run_similarity` 也支持这一点：`YY` = 0.0554，`KJ` = 0.0326，t = 2.49, p = .019, dz = 0.47。`Spatial_gt_Metaphor` family 则没有明显条件差异：`YY` = 0.0767，`KJ` = 0.0679，p = .361。这一模式与主结果方向一致：更像是隐喻相关网络在学习阶段对 `YY` 更敏感，而不是所有 ROI 都出现泛化式增强。

ROI-level 上，可作为补充证据的结果主要集中在 angular gyrus 和 temporal pole。`literature` 中，`lit_L_AG` 的 same-sentence similarity 为 `YY` = 0.0937、`KJ` = 0.0701，t = 2.47, p = .020；`lit_R_AG` 为 `YY` = 0.0512、`KJ` = 0.0229，t = 2.33, p = .028。`lit_R_AG` 的 primary pair discrimination 也达到显著：`YY` = 0.0221、`KJ` = -0.0068，t = 2.10, p = .045。`main_functional` 中，左 temporal pole 的第二个簇在 same-sentence similarity 上显著：`YY` = 0.0609、`KJ` = 0.0304，t = 2.49, p = .019。

因此，B1 最适合写成前期补充结果：学习阶段数据并不强力支持“pair discrimination 已经全面增强”，但支持“隐喻学习在语义/隐喻相关 ROI 中形成了更稳定的跨 run 表征”。这个结论可以作为后续前后测 RSA 与 Model-RSA 的学习过程背景，而不是替代 Step 5C 的主证据。

本节依据产物：[table_learning_repeated_exposure_literature.tsv](E:/python_metaphor/paper_outputs/tables_main/table_learning_repeated_exposure_literature.tsv)、[table_learning_repeated_exposure_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_main/table_learning_repeated_exposure_main_functional.tsv)、[table_learning_repeated_exposure_family_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_learning_repeated_exposure_family_literature.tsv)、[table_learning_repeated_exposure_family_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_si/table_learning_repeated_exposure_family_main_functional.tsv)、[fig_learning_dynamics_literature.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_literature.png)、[fig_learning_dynamics_main_functional.png](E:/python_metaphor/paper_outputs/figures_main/fig_learning_dynamics_main_functional.png)。

## 7、ROI 间 representational connectivity（B2: `representational_connectivity.py`）
B2 看的不是单个 ROI 内 pair similarity 怎么变，而是不同 ROI 之间的 item-level representational profile 是否更同步。简单说，如果两个 ROI 对同一批 item 的表征变化模式相似，那么它们的 representational connectivity 更高。

当前主结果比较克制：在预先定义的 pooled scopes 中，没有发现 `YY` 显著高于 `KJ` 的 ROI 间表征连接增强。`independent` ROI 池中，跨 `literature` 与 `literature_spatial` 的连接为 `YY` = 0.488、`KJ` = 0.526，t = -1.65, p = .110；`within_literature` 为 `YY` = 0.392、`KJ` = 0.428，p = .199；`within_literature_spatial` 为 `YY` = 0.447、`KJ` = 0.485，p = .134。也就是说，按独立文献 ROI 池来看，`KJ` 甚至数值上更高，但没有达到显著。

在 `main_functional` ROI 池中也没有稳定条件差异。跨两个 family 的连接为 `YY` = 0.301、`KJ` = 0.309，p = .787；`within_Metaphor_gt_Spatial` 为 `YY` = 0.341、`KJ` = 0.271，p = .216；`within_Spatial_gt_Metaphor` 为 `YY` = 0.445、`KJ` = 0.463，p = .528。整体上，B2 不能支持“隐喻学习增强了 ROI 间整体表征耦合”这一强主张。

edge-level 探索性结果中有若干单边显著边，但方向并不统一。例如 `lit_L_pSTS <-> lit_R_IFG`、`lit_L_AG <-> lit_L_pSTS`、`lit_L_AG <-> lit_R_IFG` 等边表现为 `KJ` 更强；而 `lit_L_pMTG <-> lit_R_pMTG` 表现为 `YY` 更强。这些结果可以放在补充材料中作为网络层面的探索观察，但不适合作为主文中心论点。

因此，B2 的论文写法应当是一个边界条件：当前主结论主要来自 ROI 内 representational reorganization，而不是 ROI 间 connectivity 的整体增强。这个阴性结果反而能帮助收窄解释范围，说明 `YY` 效应不是简单的“全网络同步增强”。

本节依据产物：[table_repr_connectivity.tsv](E:/python_metaphor/paper_outputs/tables_main/table_repr_connectivity.tsv)、[table_repr_connectivity_subject.tsv](E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_subject.tsv)、[table_repr_connectivity_edges.tsv](E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_edges.tsv)、[repr_connectivity_summary_group.tsv](E:/python_metaphor/paper_outputs/qc/repr_connectivity_summary_group.tsv)、[repr_connectivity_edge_group.tsv](E:/python_metaphor/paper_outputs/qc/repr_connectivity_edge_group.tsv)、[fig_repr_connectivity.png](E:/python_metaphor/paper_outputs/figures_main/fig_repr_connectivity.png)。

## 8、稳健性检验（B4: `robustness_suite.py`）
B4 是围绕 Step 5C 主终点做的稳健性检查。这里的核心效应定义为 interaction delta：`yy_vs_baseline = (yy_post - yy_pre) - (baseline_post - baseline_pre)`，`yy_vs_kj = (yy_post - yy_pre) - (kj_post - kj_pre)`。因为 Step 5C 的主现象是 `YY` 从 pre 到 post 的 similarity 下降更强，所以 interaction delta 越负，越支持“`YY` 的学习后去相似化/表征重组更强”。

bootstrap 结果显示，所有核心 ROI scope 的 95% CI 都稳定低于 0。`literature__all` 中，`yy_vs_baseline` 均值为 -0.0956，95% CI [-0.1219, -0.0703]；`yy_vs_kj` 均值为 -0.0774，95% CI [-0.1032, -0.0517]。`literature_spatial__all` 中，`yy_vs_baseline` 为 -0.0767，95% CI [-0.1036, -0.0502]；`yy_vs_kj` 为 -0.0658，95% CI [-0.0878, -0.0438]。

`main_functional` 的稳健性同样较强。`Metaphor_gt_Spatial` family 中，`yy_vs_baseline` 均值为 -0.1109，95% CI [-0.1423, -0.0798]；`yy_vs_kj` 为 -0.0786，95% CI [-0.1105, -0.0448]。`Spatial_gt_Metaphor` family 中，`yy_vs_baseline` 为 -0.0768，95% CI [-0.1085, -0.0427]；`yy_vs_kj` 为 -0.0566，95% CI [-0.0810, -0.0317]。把 `main_functional` 全部 ROI 合并后，`yy_vs_baseline` 为 -0.0865，95% CI [-0.1137, -0.0580]；`yy_vs_kj` 为 -0.0629，95% CI [-0.0877, -0.0372]。

LOSO（leave-one-subject-out）结果显示，任意去掉一个被试后，所有 scope 的效应仍保持负向，且所有 leave-one-out 版本均 p < .05。最大 p 值仍很小：`literature__all` 的 `yy_vs_kj` 最大 p = 1.22e-05，`literature_spatial__all` 的 `yy_vs_kj` 最大 p = 1.21e-05，`main_functional__all` 的 `yy_vs_kj` 最大 p = 1.50e-04。这说明主效应不是由某一个极端被试单独驱动的。

split-half replication 也支持效应方向稳定。200 次随机半样本拆分中，所有 scope/contrast 的两个半样本效应方向均一致；两个半样本都达到 p < .05 的比例在 `literature__all` 为 100%，在 `literature_spatial__all` 为 98.5%（`yy_vs_baseline`）和 99.0%（`yy_vs_kj`），在 `main_functional__all` 为 99.5%（`yy_vs_baseline`）和 85.5%（`yy_vs_kj`）。较弱的是 `main_functional__Spatial_gt_Metaphor`，但也有 68.0%（`yy_vs_baseline`）和 72.0%（`yy_vs_kj`）的拆分同时显著。

需要注意：`table_bayes_factors.tsv` 当前没有真正的 BF10 数值，因为运行环境未安装 `pingouin`，表中明确标记为 unavailable。因此现阶段不要在论文中报告 Bayes Factor，只报告 bootstrap、LOSO 和 split-half。另一个小点是，当前已修复 `robustness_suite.py` 中 LOSO `same_direction_as_full` 按“是否大于 0”写死的问题；但现有 `table_loso.tsv` 是旧脚本产物，下一次重跑 robustness 后该字段才会更新。因此本节仍采用实际均值范围和 p 值来解释。

本节依据产物：[table_bayes_factors.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bayes_factors.tsv)、[table_bootstrap_ci.tsv](E:/python_metaphor/paper_outputs/tables_si/table_bootstrap_ci.tsv)、[table_loso.tsv](E:/python_metaphor/paper_outputs/tables_si/table_loso.tsv)、[table_splithalf.tsv](E:/python_metaphor/paper_outputs/tables_si/table_splithalf.tsv)、[robustness_subject_interaction.tsv](E:/python_metaphor/paper_outputs/qc/robustness_subject_interaction.tsv)、[robustness_meta.json](E:/python_metaphor/paper_outputs/qc/robustness_meta.json)。

## 9、Noise ceiling（B5: `noise_ceiling.py`）
B5 的作用是判断当前 RSA / Model-RSA 数据有没有足够的被试间一致性。如果 noise ceiling 很低，那么任何模型解释都会被测量噪声限制；如果 ceiling 达到中等水平，就说明这些神经表征矩阵虽然不可能非常完美，但有足够可靠性支撑主分析。

Step 5C itemwise delta 的 ceiling 总体处于中等范围。`literature__all` 中，`baseline` 的 lower/upper mean r 为 0.506/0.547，`KJ` 为 0.382/0.470，`YY` 为 0.438/0.496。`literature_spatial__all` 中，`baseline` 为 0.467/0.514，`KJ` 为 0.377/0.450，`YY` 为 0.368/0.437。`main_functional__all` 中，`baseline` 为 0.470/0.498，`KJ` 为 0.339/0.426，`YY` 为 0.314/0.389。

分 main_functional family 看，`Metaphor_gt_Spatial` family 的 `YY` ceiling 为 0.274/0.356，`KJ` 为 0.312/0.390，`baseline` 为 0.410/0.454；`Spatial_gt_Metaphor` family 的 `YY` ceiling 为 0.251/0.340，`KJ` 为 0.292/0.389，`baseline` 为 0.394/0.446。也就是说，`YY` 条件在主功能 ROI family 中的可靠性低于 baseline，但并没有低到无法分析；这更适合写成“测量噪声限制了效应上界，需要克制解释效应大小”，而不是“结果不可靠”。

Model-RSA 神经 RDM 的 ceiling 也支持谨慎解释。`main_functional` 的 pre 阶段平均 lower/upper mean r 为 0.262/0.353，post 为 0.238/0.336；`literature` 的 pre 为 0.319/0.402，post 为 0.285/0.372。post 阶段略低于 pre，说明学习后表征结构可能更分散或更受噪声影响，这与主结果中 post-learning reorganization 的解释并不冲突，但提醒我们不要把 Model-RSA 的局部显著结果夸大为全局机制定论。

因此，B5 可以作为 reliability / measurement-boundary 结果写入补充材料：当前数据有中等程度的可重复结构，足以支持 Step 5C 和 Model-RSA 的方向性解释；同时 ceiling 限制提醒我们，机制结论应写成“constrains / suggests”，而不是“fully explains”。

本节依据产物：[table_noise_ceiling.tsv](E:/python_metaphor/paper_outputs/tables_si/table_noise_ceiling.tsv)、[noise_ceiling_step5c_subject.tsv](E:/python_metaphor/paper_outputs/qc/noise_ceiling_step5c_subject.tsv)、[noise_ceiling_model_rsa_subject.tsv](E:/python_metaphor/paper_outputs/qc/noise_ceiling_model_rsa_subject.tsv)、[noise_ceiling_meta.json](E:/python_metaphor/paper_outputs/qc/noise_ceiling_meta.json)、[fig_noise_ceiling.png](E:/python_metaphor/paper_outputs/figures_si/fig_noise_ceiling.png)。

## 10、ROI-level RD / GPS 扩展结果（C7: `rd_analysis.py` + `gps_analysis.py`）
这一节对应 ROI-level representational dimensionality（RD）和 global pattern similarity（GPS）分析。它们和 Step 5C 的 pair-wise RSA 不是同一个问题：Step 5C 问“同一学习 pair 的两个词是否变得更像/更不像”，RD 问“一个 ROI 内的 trial pattern 是否占据更高维、更分散的表征空间”，GPS 问“同一条件内的 trial pattern 是否整体更紧凑、更一致”。因此，RD/GPS 更适合作为表征几何的扩展描述，而不是替代 Step 5C 的主终点。

当前已跑出的 ROI_SET 包括 `main_functional` 和 `literature`。两套分析均包含 28 名被试；`main_functional` 覆盖 7 个 ROI，`literature` 覆盖 12 个 ROI。RD 当前采用正式推荐口径 `participation_ratio`，即基于 voxel covariance eigenspectrum 的软维度；GPS 则基于同一条件 trial-by-trial pattern correlation 的 Fisher-Z 平均相似性。当前没有看到 `literature_spatial` 的 RD/GPS 产物，因此本节暂不把空间文献 ROI 写成已完成结果。

RD 的结果总体是阴性的。`main_functional` 的 7 个 ROI 中，没有任何 ROI 的 `YY` 与 `KJ` 的 RD delta 差异达到 p < .05；`literature` 的 12 个 ROI 同样没有显著差异。数值上，`main_functional` 中最接近趋势的是右后海马旁回：`YY` RD delta = 0.2105，`KJ` = -0.4981，t = 1.67, p = .106, dz = 0.32；后部梭状回为 `YY` = -0.1071，`KJ` = -0.9986，t = 1.54, p = .136。`literature` 中最小 p 值也只有右 AG：`YY` = -0.0691，`KJ` = -0.5256，t = 0.95, p = .349。也就是说，目前没有证据表明隐喻学习相对空间学习带来了稳定的 ROI-level 表征维度升高或降低。

GPS 的结果更有信息量。`main_functional` 中没有 ROI 达到 p < .05，但两个左颞极 ROI 呈现 `YY` GPS 下降更强的方向，其中第二个 temporal pole 簇为 `YY` = -0.0362，`KJ` = -0.0199，t = -1.75, p = .092, dz = -0.33；第一个 temporal pole 簇为 `YY` = -0.0113，`KJ` = 0.0032，p = .165。这个方向与主线中的“隐喻学习后表征更分化/更不紧凑”相容，但在主功能 ROI 层面还不足以作为显著结果报告。

`literature` 的 GPS 则出现两个未校正显著 ROI。左 AG 中，`YY` GPS delta = -0.0309，`KJ` = -0.0100，t = -3.68, p = .0010, dz = -0.70；左 pSTS 中，`YY` = -0.0358，`KJ` = -0.0140，t = -2.19, p = .037, dz = -0.41。因为 GPS 越高表示同条件 trial pattern 越紧凑、越一致，所以这里的负向 `YY - KJ` delta 表示：从 pre 到 post，隐喻条件在左 AG 和左 pSTS 中出现了比空间条件更强的全局去紧凑化。这和 Step 5C 的 pair similarity 下降并不是完全同一个指标，但方向上共同支持“学习后不是简单 convergence，而是表征空间变得更分化/更重组”。

需要克制的是，GPS 显著结果目前集中在 `literature` 的少数语义/整合 ROI，而且当前 RD/GPS 表没有提供跨 ROI 的 FDR 校正列；RD 也没有给出维度层面的显著条件差异。因此，RD/GPS 最适合写成探索性补充机制证据：Step 5C 证明 pair-level similarity 在 `YY` 中更明显下降；GPS 进一步提示，在左 AG 与左 pSTS 这类文献语义整合节点中，`YY` 的同条件整体 pattern 也变得更不紧凑；但 RD 尚未支持“表征维度整体升高”这一更强说法。

本节依据产物：
- [table_rd_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rd_main_functional.tsv)
- [table_rd_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_rd_literature.tsv)
- [rd_results_main_functional/rd_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/rd_results_main_functional/rd_group_summary.tsv)
- [rd_results_literature/rd_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/rd_results_literature/rd_group_summary.tsv)
- [rd_results_main_functional/rd_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/rd_results_main_functional/rd_subject_metrics.tsv)
- [rd_results_literature/rd_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/rd_results_literature/rd_subject_metrics.tsv)
- [table_gps_main_functional.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gps_main_functional.tsv)
- [table_gps_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gps_literature.tsv)
- [gps_results_main_functional/gps_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/gps_results_main_functional/gps_group_summary.tsv)
- [gps_results_literature/gps_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/gps_results_literature/gps_group_summary.tsv)
- [gps_results_main_functional/gps_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/gps_results_main_functional/gps_subject_metrics.tsv)
- [gps_results_literature/gps_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/gps_results_literature/gps_subject_metrics.tsv)
- [fig_rd_gps_main_functional.png](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_main_functional.png)
- [fig_rd_gps_literature.png](E:/python_metaphor/paper_outputs/figures_si/fig_rd_gps_literature.png)

## 11、S1 机制 → 行为 / 记忆预测（`mechanism_behavior_prediction.py`）
这一节对应当前方案中的 S1：把 Step 5D 的被试层 Model-RSA 机制分数和 run-7 最终学习收益连起来，问“机制分数的个体差异是否能解释最终记忆 / 提取表现的个体差异”。它和前面的 item-level 脑-行为分析不同：item-level 分析的样本单位是 trial / word，问局部神经变化是否预测单个项目是否记住；S1 的样本单位是被试，问某个 ROI 中某种机制分数更强的被试，是否整体学得更好。因此，S1 不能当因果证据，但可以作为“机制是否具有行为意义”的被试层闭环检验。

本轮默认主分析只纳入 `M8_reverse_pair` 与 `M3_embedding` 两个主机制模型，未纳入 `M7_continuous_confidence`。脚本自动读取三套 ROI set：`main_functional`、`literature`、`literature_spatial`，合计 31 个 ROI；与 refined run-7 行为表合并后共有 27 名被试、10044 行长表记录。P0a 修复后，S1 不再把 pooled/all 的同一个机制分数复制成 `YY` 和 `KJ` 两份 condition-specific 证据；当前只保留两个合理口径：`all` 表示 pooled 机制分数预测总体 run-7 表现，`yy_minus_kj` 表示 pooled 机制分数预测 `YY - KJ` 行为差异。每个效应给出标准化 beta、bootstrap CI、Spearman r 与 5000 次 permutation p。为避免把不同理论层级的 ROI 混在同一个过大的校正家族中，本轮已经把推断层级拆成四个 inference family：`literature`、`literature_spatial`、`main_functional_metaphor_gt_spatial` 与 `main_functional_spatial_gt_metaphor`；正式判断优先看各 family 内的 BH-FDR，即 `q_within_inference_family`，并辅以主终点校正列 `q_within_primary_family`。

校正后的结果仍然是阴性：全表没有任何效应达到 `q_within_inference_family < .05` 或 `q_within_roi_set < .05`。各 family 的最小 `q_within_inference_family` 分别为：`literature_spatial` = .115，`main_functional_spatial_gt_metaphor` = .192，`main_functional_metaphor_gt_spatial` = .245，`literature` = .662。若只看预设主终点，最佳候选是 `literature_spatial` 左 RSC 的 `M8_reverse_pair` 预测 `run7_correct_retrieval_efficiency` 的 `YY - KJ` 差值：β = 0.408，Spearman r = 0.634，permutation p = .0008；但 family-level q = .115，primary-family q = .077，仍不能作为正式阳性发现。

最重要的结论是：S1 不能再写成“隐喻机制分数越强，YY 条件最终行为越好”的 condition-specific 证据，因为当前输入的 Model-RSA 机制分数本身是 pooled/all 的被试层分数。修复后，问题被改写得更严谨：这个总体机制分数能否预测总体行为，或预测 `YY - KJ` 的行为优势。答案是：没有 FDR 校正后的稳定阳性；但空间文献 ROI 中的 RSC/OPA/PPA 等区域出现了一些与 `YY - KJ` retrieval efficiency 或 memory accuracy 相关的未校正候选线索。

未校正候选结果的方向更像“差异预测”而不是“单条件预测”。例如 `literature_spatial` 左 RSC 的 `M8_reverse_pair` 对 `YY - KJ` retrieval efficiency 为正向预测，表示该 ROI 中 pair-differentiation 指标越强的被试，越倾向于表现出更大的 `YY` 相对 `KJ` 提取效率优势。右 OPA 的 `M8_reverse_pair` 对 `YY - KJ` memory accuracy 也有未校正候选信号，β = 0.448，Spearman r = 0.526，permutation p = .0064，但 q = .245。总体行为口径 `all` 中反而常见负向关系，例如左 RSC 的 `M8_reverse_pair` 预测总体 memory accuracy β = -0.557，p = .0100，说明机制-行为映射并不是简单“机制越强，整体越好”。

辅助 speed 指标中仍有若干未校正候选，例如 `literature_spatial` 左 precuneus 的 `M3_embedding` 预测 `YY - KJ` correct-trial speed，β = 0.464，Spearman r = 0.532，permutation p = .0066；以及 `main_functional_metaphor_gt_spatial` 两个左颞极 ROI 的 `M3_embedding` 预测 `YY - KJ` speed，permutation p 约 .016-.020。但 speed 是补充指标，不是 S1 默认主终点，而且这些结果没有通过 family-level FDR。因此，它们最多作为探索性线索，不能作为主文机制闭环的强证据。

对应图片 `fig_mechanism_behavior_prediction.png` 的定位也需要固定：这张图不是“显著结果图”，而是把未校正 permutation p 最小的候选关系画成被试散点图。图中每个点是一名被试，横轴是 aligned mechanism score，纵轴是 run-7 行为表现；标题中的 `p_perm` 是单个候选关系的置换 p 值。因为这些候选关系在 family-level BH-FDR 后均未显著，所以图片只能用来说明“最像有信号的探索性模式主要集中在 `M8_reverse_pair`、空间文献 ROI 的 L_RSC/OPA/PPA，以及 `YY - KJ` 行为差异”，不能用作正式显著性证据。

所以 S1 当前最稳妥的写法是边界结果：被试层 pooled 机制分数没有稳健支持“pair-based differentiation / semantic reweighting 越强，最终学习收益越好”的简单线性命题。它并不推翻组水平 RSA / Model-RSA 主线，因为组水平表征重组和被试间行为预测是不同层级的问题；但它提示我们不能把机制分数直接包装成强行为预测器。主文可以把 S1 写成补充约束：表征机制在组水平上稳定存在，但其被试层行为映射目前只在 `YY - KJ` 差异、空间文献 ROI、未校正候选层面出现线索，尚不足以构成强个体差异闭环。

本节依据产物：
- [table_mechanism_behavior_prediction.tsv](E:/python_metaphor/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv)
- [table_mechanism_behavior_prediction_perm.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv)
- [table_mechanism_behavior_prediction_literature.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_literature.tsv)
- [table_mechanism_behavior_prediction_literature_spatial.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_literature_spatial.tsv)
- [table_mechanism_behavior_prediction_main_functional_metaphor_gt_spatial.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_main_functional_metaphor_gt_spatial.tsv)
- [table_mechanism_behavior_prediction_main_functional_spatial_gt_metaphor.tsv](E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_main_functional_spatial_gt_metaphor.tsv)
- [mechanism_behavior_prediction_long.tsv](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_prediction_long.tsv)
- [mechanism_behavior_input_scope.tsv](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_input_scope.tsv)
- [mechanism_behavior_prediction_manifest.json](E:/python_metaphor/paper_outputs/qc/mechanism_behavior_prediction_manifest.json)
- [fig_mechanism_behavior_prediction.png](E:/python_metaphor/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png)
- [model_rdm_subject_metrics.tsv: main_functional](E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_subject_metrics.tsv)
- [model_rdm_subject_metrics.tsv: literature](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_subject_metrics.tsv)
- [model_rdm_subject_metrics.tsv: literature_spatial](E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_subject_metrics.tsv)
- [subject_summary_wide.tsv](E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv)

## 12、S2 gPPI 网络层重配置（`gPPI_analysis.py` + `export_gppi_summary.py`）
这一节对应当前方案中的 S2：用 gPPI 检验左颞极 seed 与文献 ROI / 空间文献 ROI 的连接调制是否从 pre 到 post 发生变化。它和 B2 的 representational connectivity 不同：B2 是表征 profile 的 ROI 间相似性；gPPI 是传统功能连接框架下的 `seed × psychological condition` 调制项。当前 gPPI 的作用是补充网络层证据，但不能写成因果连接或连接驱动表征变化。

本轮正式 gPPI 使用两个 learning GLM 定位出的左颞极 seed：`func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort` 与 `func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort`。target ROI 合并为 `literature + literature_spatial`，共 24 个 target；阶段为 pre/post，即 run1/2 vs run5/6；条件为 `yy` 与 `kj`；deconvolution 使用 `ridge`。两个 seed 均成功完成，均为 28 名被试、每个 seed `8064` 行 subject/run/target/condition 级 gPPI beta，且 `n_errors = 0`。

主表只保留最关键的 `yy_minus_kj` 在 post vs pre 的变化，并按三个 scope 汇总：`within_literature`、`within_literature_spatial` 和 `cross_sets`。结果比较清楚：6 个主检验均未达到显著。第一个左颞极 seed 中，`within_literature` 的 `yy_minus_kj` mean_diff = -4.480，t = -0.82，p = .421，dz = -0.15；`within_literature_spatial` mean_diff = -0.541，p = .763；`cross_sets` mean_diff = -3.939，p = .350。第二个左颞极 seed 中，`within_literature` mean_diff = -1.746，p = .488；`within_literature_spatial` mean_diff = 2.546，p = .231；`cross_sets` mean_diff = -4.293，p = .269。也就是说，当前没有证据表明 `YY` 相对 `KJ` 的 seed-target 连接调制在 post 阶段相对 pre 阶段发生了稳定增强或减弱。

轻量导出层已统一改为“先置换、再 BH-FDR”的 ROI/target 级口径。对 6 个 planned scope 主检验，脚本根据 subject-level scope mean 重新计算被试内 sign-flip permutation p，并在 6 个 `yy_minus_kj` 主检验内给出 `q_within_main_tests`；结果仍然完全阴性，最小 q = .770。第一个 seed 的三个主检验 permutation p 分别为 .825、.803、.599；第二个 seed 分别为 .646、.257、.184。也就是说，主表的阴性不是参数 t 检验造成的。

为避免 scope mean 把少数 target 的局部效应平均稀释，本轮进一步加入了 per-target 分层校正：先对每条 `seed × target ROI × condition` 的 post-pre 差值做被试内 sign-flip permutation，得到 `p_permutation`；再在每个 `seed × target_scope × condition` family 内对 permutation p 做 BH-FDR，得到 `q_within_seed_scope_condition`。结果显示，scope mean 的阴性并不是因为存在一个被平均掩盖的稳健 target 主效应。SI full table 中原本唯一参数检验达到 p < .05 的边，是第二个左颞极 seed 到 `litspat_R_PPC` 的 `yy` 单条件 PPI 从 pre 到 post 上升，mean_diff = 2.456，t = 2.25，参数 p = .033；置换后 `p_permutation = .034`，但分层 BH-FDR 后 `q_within_seed_scope_condition = .406`，不再达到显著。所有 per-target 结果均未通过 FDR；当前最小 q 约为 .242。主 contrast `yy_minus_kj` 的 target peak 也均未显著：两个 seed 在 `within_literature` 的最小 q 分别为 .893 和 .603，在 `within_literature_spatial` 分别为 .873 和 .982。

因此，gPPI 当前最稳妥的结论是阴性边界结果：它没有支持“隐喻学习伴随稳定的左颞极 seed 网络重配置”这一强说法。主文如果保留 S2，建议写成：在当前 pilot gPPI 设定下，网络连接调制没有形成可报告的主效应；因此，本文的核心证据仍来自 ROI 内 representational reorganization，而不是传统 seed-based connectivity 的稳定改变。这个结果有价值，因为它帮我们避免把 Step 5C / Model-RSA 的表征重组过度解释成连接层面的全局再配置。

本节依据产物：
- [table_gppi_summary_main.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_summary_main.tsv)
- [table_gppi_target_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_gppi_target_peaks.tsv)
- [table_gppi_summary_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_summary_full.tsv)
- [table_gppi_target_level.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_target_level.tsv)
- [table_gppi_target_peaks_full.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_target_peaks_full.tsv)
- [table_gppi_scope_permutation.tsv](E:/python_metaphor/paper_outputs/tables_si/table_gppi_scope_permutation.tsv)
- [gppi_summary_export_manifest.json](E:/python_metaphor/paper_outputs/qc/gppi_summary_export_manifest.json)
- [gppi_subject_metrics_all_seeds.tsv](E:/python_metaphor/paper_outputs/qc/gppi_combined_literatureplus_literature_spatial/gppi_subject_metrics_all_seeds.tsv)
- [seed1 gppi_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_subject_metrics.tsv)
- [seed1 gppi_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_group_summary.tsv)
- [seed1 gppi_meta.json](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_meta.json)
- [seed2 gppi_subject_metrics.tsv](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_subject_metrics.tsv)
- [seed2 gppi_group_summary.tsv](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_group_summary.tsv)
- [seed2 gppi_meta.json](E:/python_metaphor/paper_outputs/qc/gppi_results_func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort_literatureplus_literature_spatial/gppi_meta.json)

## 13、S3 Searchlight 全脑定位（`pair_similarity_searchlight.py` + `rd_searchlight.py`）
这一节对应当前方案中的 S3。这里有两个 searchlight：主版是 pair-similarity searchlight，直接对齐 Step 5C 的核心问题，问“全脑哪些位置出现 pair-level similarity 的 post - pre 改变”；补充版是 RD searchlight，问“全脑哪些位置出现局部表征维度的 post - pre 改变”。因此，pair-similarity searchlight 更贴主线，RD searchlight 是几何补充。

本轮已用修正后的 group-level FWE 逻辑重跑完成，且 `reuse_subject_maps = true`，也就是说没有重算最慢的 subject-level `.nii.gz`，而是复用个体图后重新计算组水平 sign-flip permutation、FWE map、peak table 和 figure。主版 pair-similarity searchlight 的 `YY` 与 `KJ` 均包含 28 名被试、约 207795 个有效 searchlight voxel、2000 次 permutation，backend 为 GPU。`baseline` 仍然没有进入 pair-similarity 组比较，因为 baseline metadata 没有可用的 pair 结构；QC 中 `pre_baseline = 0`、`post_baseline = 0`，56 个 baseline cell 被跳过，这是设计上合理的。

主版结果非常清楚：`YY` 的 post - pre pair similarity 出现了全脑 FWE 校正后显著下降，而 `KJ` 没有。`YY` peak table 的前 10 个峰均为负 t 值，说明 post 相对 pre 的 pair similarity 下降；最高峰位于 MNI `[28, 16, 48]`，t = -8.66，p_FWE = .00050。其他显著峰包括 `[-50, 16, 2]`，t = -8.51，p_FWE = .0010；`[18, -24, 10]`，t = -8.25，p_FWE = .0010；`[14, -72, -28]`，t = -8.10，p_FWE = .0015；`[26, 14, 42]`，t = -8.04，p_FWE = .0015；以及 `[14, -64, 4]`、`[20, -44, -2]`、`[8, -28, 50]`、`[-24, 16, 46]`、`[-32, -18, 6]` 等峰，均达到 p_FWE < .003。因为当前 peak 表还没有 atlas label，主文中应先以 MNI 坐标报告，后续如需写脑区名，建议单独跑 atlas annotation。

`KJ` pair-similarity searchlight 则没有任何 FWE 显著 peak。其最强峰为 MNI `[-4, -36, 40]`，t = -5.32，但 p_FWE = .351；第二峰 `[6, 2, -12]` 为 t = 5.21，p_FWE = .415；其余峰的 p_FWE 更高。因此，S3 主版与 ROI-level Step 5C 的核心故事是同向的：隐喻学习后出现更强、更广泛的 pair-level 去相似化 / differentiation，而空间条件没有形成可报告的全脑校正后效应。

补充版 RD searchlight 也已重跑完成，`YY`、`KJ`、`baseline` 三个条件均为 28 名被试，约 207789-207793 个有效 voxel，2000 次 GPU permutation，且没有 skipped cell。与 pair-similarity searchlight 不同，RD searchlight 没有任何条件达到 FWE 显著：`YY` 最强峰为 `[-10, 12, 2]`，t = 6.15，p_FWE = .159；`KJ` 最强峰为 `[-26, -14, -26]`，t = 6.30，p_FWE = .149；`baseline` 最强峰为 `[-64, -52, -2]`，t = -5.79，p_FWE = .387。也就是说，当前全脑层面的显著定位并不是“局部表征维度 RD 改变”，而是更贴近主线的 pair-level similarity 改变。

因此，S3 现在可以写成一个有力但边界清楚的全脑补充证据：pair-similarity searchlight 在全脑范围内验证了 `YY` 的学习后 pair-level similarity 下降，并且这种下降在 voxelwise max-T FWE 校正后仍然显著；`KJ` 没有同等级别的全脑效应；RD searchlight 作为几何补充为阴性。这个模式支持“隐喻学习诱发的表征重组主要体现为 pair-level differentiation，而不是简单的局部维度升高”。需要克制的是，S3 目前只做了单条件 post - pre 的 searchlight，没有直接做 `YY - KJ` 的全脑交互图，因此主文表述应写成“YY 出现 FWE 显著、KJ 未出现”的条件内定位证据，而不是声称已经在全脑 searchlight 上直接证明了 `YY > KJ` 交互。

本节依据产物：
- [table_searchlight_pair_similarity_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv)
- [fig_searchlight_pair_similarity_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png)
- [pair_similarity_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/pair_similarity_searchlight_summary.tsv)
- [pair_similarity_searchlight_summary.json](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/pair_similarity_searchlight_summary.json)
- [pair_similarity_searchlight_map_qc.tsv](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/pair_similarity_searchlight_map_qc.tsv)
- [yy pair-similarity group maps](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/group_yy_post_vs_pre)
- [kj pair-similarity group maps](E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/group_kj_post_vs_pre)
- [table_searchlight_peaks.tsv](E:/python_metaphor/paper_outputs/tables_main/table_searchlight_peaks.tsv)
- [fig_searchlight_delta.png](E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_delta.png)
- [rd_searchlight_summary.tsv](E:/python_metaphor/paper_outputs/qc/rd_searchlight/rd_searchlight_summary.tsv)
- [rd_searchlight_summary.json](E:/python_metaphor/paper_outputs/qc/rd_searchlight/rd_searchlight_summary.json)
- [rd_searchlight_map_qc.tsv](E:/python_metaphor/paper_outputs/qc/rd_searchlight/rd_searchlight_map_qc.tsv)
- [yy RD group maps](E:/python_metaphor/paper_outputs/qc/rd_searchlight/group_yy_post_vs_pre)
- [kj RD group maps](E:/python_metaphor/paper_outputs/qc/rd_searchlight/group_kj_post_vs_pre)
- [baseline RD group maps](E:/python_metaphor/paper_outputs/qc/rd_searchlight/group_baseline_post_vs_pre)

## 14、主文固定口径（当前版本）

这一节不是新增结果，而是把当前最稳妥、最适合放进主文的写法固定下来，避免后续表述来回摇摆。

### 14.1 核心主张

主文的 primary claim 建议固定为：

- 相比空间联结学习，隐喻联结学习诱发了更强的学习后表征重组。
- 这种重组在当前数据中主要表现为 `yy` 的 pair similarity 从 pre 到 post 更明显下降。
- 最新 pair-similarity searchlight 进一步显示，`yy` 的 post-to-pre similarity 下降在全脑 voxelwise FWE 校正后仍可定位，而 `kj` 没有同等级别的全脑效应。
- 该模式更支持   representational differentiation / reorganization  ，而不是简单的 pair convergence。
- 在机制层面，当前结果更支持：
  - pair-based differentiation
  - semantic reweighting

### 14.2 结果段固定写法

主结果段建议优先这样写：

> 行为结果显示，被试对 `yy` 条件的学习表现优于 `kj`。在三条件 item-wise RSA 中，`yy` 在多个 ROI 集内均表现出比 `kj` 和 `baseline` 更明显的 pre-to-post similarity 下降，说明隐喻学习后的神经表征变化并未表现为简单的配对聚合，而更接近表征分化增强。进一步的 Model-RSA 显示，这种变化在左 Precuneous 与左 IFG 更符合 pair-based differentiation，在左 temporal pole 更符合 semantic reweighting。全脑 pair-similarity searchlight 也在 `yy` 条件中定位到 FWE 校正后显著的 post-to-pre similarity 下降，而 `kj` 没有对应的全脑显著效应。整体而言，隐喻学习引发的更可能是学习后表征几何的重组，而不是 pair members 的简单收敛。

### 14.3 讨论段固定写法

讨论段建议优先这样写：

> 当前结果表明，隐喻学习相较空间联结学习，更可能诱发学习后神经表征的重组。重要的是，这种变化并未表现为同 pair 项目在 post 阶段变得更相似，而是表现为 `yy` 的 pair similarity 更明显下降，并在部分 ROI 中伴随 pair-based 模型减弱与预训练语义模型减弱。这提示隐喻学习可能不是把两个词条简单压缩为一个更接近的联合表征，而是在原有语义结构之上形成了更精细、区分度更高的新关系几何。与此同时，即使把分析范围进一步限制到经典空间网络 ROI，`kj` 也没有表现出跨 ROI、一致、可靠的 pre/post 去相似化；因此，当前 `yy` 效应并不能简单归因为 ROI 选择偏向语义网络，而更像是一种对隐喻学习更敏感、跨网络可见的表征重组。连续记忆强度模型进一步提示，记忆几何并非全局主导机制，但在右 pMTG 等局部后部语义节点上可能提供次级调制。因而，本研究更支持隐喻学习体现为 representational differentiation / reorganization，而不是简单的 pair convergence。

### 14.4 必须保留的克制表述

当前主文应明确保留以下边界：

- 不把结果写成“已经证明隐喻学习失败所以没有收敛”
- 不把结果写成“已经完全排除所有质量解释”
- 不把 `main_functional` 层总体或 family-split 的 `delta_rho_lmm` 写成强裁决，因为这些 MixedLM 目前仍有 `converged = false`
- 不把 `literature` 层 ROI-split 写成强裁决，因为虽然总体 MixedLM 已收敛，但个别 ROI split 仍可能出现 singular fit（如 `lit_R_AG`）
- 不把 `M7_binary` 或 `M7_continuous_confidence` 写成当前主导机制；更准确的说法是：连续 M7 在右 pMTG 提供了局部、次级支持，但尚未形成跨 ROI 的统一主结论
- 不把空间 ROI 的阴性结果写成“空间学习没有任何神经变化”；更准确的说法是：`kj` 缺乏跨 ROI、一致、可靠的去相似化模式
- 不把 item-level 脑-行为、B1 学习期、RD/GPS 的未校正局部 p 值写成跨 ROI 校正后的主发现；这些结果适合做机制线索或补充证据
- 不把 S1 机制-行为预测写成强阳性闭环；当前被试层机制分数没有稳健支持“机制越强，最终行为越好”的简单线性命题
- 不把 S3 searchlight 写成已经证明全脑 `YY > KJ` 交互；当前最准确的表述是 `yy` 条件内 post-to-pre 下降 FWE 显著，而 `kj` 条件内未显著

QC / reliability 建议固定写法为：

> voxel split-half reliability 没有显示 post 阶段整体 reliability 崩坏，因此当前结果不支持“`yy` 的下降只是整体测量不稳定”这一解释；不过，motion/QC 指标在 post 阶段确实略差，且与 `yy` 的变化存在相关，因此更稳妥的表述是：质量因素可能影响效应幅度，但不足以单独解释主结果方向。

### 14.5 不再建议使用的表述

后续主文里尽量避免以下说法：

- “学习后同 pair 会不会更像”
- “隐喻学习后发生了更强的 convergence”
- “结果说明隐喻没有真正学会”
- “记忆更好但神经相似性下降是矛盾的”

更推荐替换为：

- “学习后表征重组是否更强”
- “是否出现 pair-based differentiation”
- “是否存在 semantic reweighting”
- “行为收益与表征分化并不矛盾，二者可能共同反映更精细的关系编码”

## 15、主文图题与图注（当前版本）

这一节固定当前主文 5 张核心图的标题与图注写法，对应的图片文件位于 `E:\python_metaphor\figures_main_story`。

### Figure 1



图题

：Behavioral advantage for metaphor learning over spatial learning



图注

：被试在 `YY` 条件下表现出更优的学习收益。左图显示 run-7 记忆正确率，`YY` 的平均正确率高于 `KJ`；右图显示仅在答对 trial 上计算的反应时，`YY` 在正确提取时略快于 `KJ`。两幅面板均采用 violin-first 展示：小提琴轮廓表示被试分布，浅灰连线表示同一被试在两条件之间的配对变化，彩色点表示单被试取值，白色箱体表示分位区间，黑点和误差线表示组均值及其标准差（SD）。该图对应 refined behavior analysis 的两个正式终点：`memory` 与 `log_rt_correct`。

![Figure 1](/E:/python_metaphor/figures_main_story/fig_behavior_accuracy_rt.png)

### Figure 2



图题

：Step 5C item-wise RSA reveals stronger post-learning de-similarity for metaphor pairs



图注

：左面板给出主功能 ROI 层面的 pooled Step 5C similarity 模式，采用 split-violin 展示 `Pre` 与 `Post` 在三类条件中的分布形态，显示 `Metaphor` 条件从 pre 到 post 的 pair similarity 下降最明显，而 `Spatial` 与 `Baseline` 变化较弱。右面板显示各主功能 ROI 中 `Post - Pre` 的 delta similarity；负值表示学习后 pair similarity 下降，且 `Metaphor` 在全部主功能 ROI 中都呈现更强的负向变化。点位表示组均值，误差线表示 ROI 内跨被试标准差（SD）。该图用于呈现当前最核心的现象结果：隐喻学习后的神经表征变化更接近去相似化，而不是 pair convergence。

![Figure 2](/E:/python_metaphor/figures_main_story/fig_step5c_main.png)

### Figure 3



图题

：Model-RSA constrains the mechanism to pair-based differentiation and semantic reweighting



图注

：图中颜色表示各 ROI 内模型相关系数的 `Post - Pre` 变化量（`Δρ`），蓝色表示 post 相对 pre 下降，红色表示 post 相对 pre 上升；星号标记 ROI 级配对比较达到显著的结果。当前机制图使用最新 `table_model_rsa_mechanism.tsv` 绘制，覆盖 `main_functional` 与 `literature` 两套主报告 ROI；`literature_spatial` 作为空间文献稳健性层在正文第 5.3 节单独汇总。最关键的模式是：左侧 precuneus / IFG 的 `M2_pair` 下降并伴随 `M8_reverse_pair` 上升，说明学习后表征偏离同 pair 聚合；左侧 temporal pole 的 `M3_embedding` 下降，则提示学习后表征不再主要按预训练语义距离组织。整体上，该图支持的不是 simple pair convergence，而是 `pair-based differentiation + semantic reweighting`。

依据产物：[table_model_rsa_mechanism.tsv](E:/python_metaphor/paper_outputs/figures_main/table_model_rsa_mechanism.tsv)，[fig_model_rsa_mechanism.png](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)，[fig_model_rsa_mechanism.pdf](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.pdf)。

![Figure 3](E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png)

### Figure 4



图题

：The Step 5C metaphor effect is robust across ROI-definition schemes



图注

：该图汇总四套 ROI 定义方案下的 `Post - Pre` delta similarity，包括 `main_functional`、`literature`、`literature_spatial` 与 `atlas_robustness`。三个面板分别对应 `Metaphor`、`Spatial` 与 `Baseline`；每个面板均采用 violin-first 展示，小提琴表示被试分布，半透明小点表示被试层面的 ROI-set 平均 delta similarity，白色箱体表示分位区间，黑点和误差线表示组均值及其标准差（SD）。在四套 ROI 中，`Metaphor` 都稳定表现为更负的 delta similarity，而 `Spatial` 与 `Baseline` 缺乏同等强度、跨 ROI 一致的下降。该图用于回答稳健性问题：当前主结论并不依赖某一种 ROI 选择方式，同时保留了被试间分布信息，而不仅仅是均值不确定性。

![Figure 4](/E:/python_metaphor/figures_main_story/fig_roi_set_robustness.png)

### Figure 5



图题

：Activation strength and representational reorganization are dissociable



图注

：该图不再使用散点关系，而是直接在同一 ROI 内比较两类指标：灰蓝色点表示学习期激活强度，红色点表示 `YY` 的表征重组强度（定义为 `Pre - Post`，数值越大表示学习后去相似化越强），两点之间的连线表示同一 ROI 在两种指标上的相对位置差异。为了便于跨指标比较，横轴使用各指标内部标准化后的 z 值。图中可见，某些 ROI 在激活上较高，但在表征重组上并不等幅增强；另一些 ROI 则表现出更强的表征重组而并无同等强的激活增益。该图用于更直观地支持“学习期激活增强”与“学习后表征重塑增强”并非同一件事。

![Figure 5](/E:/python_metaphor/figures_main_story/fig_activation_representation_dissociation.png)
