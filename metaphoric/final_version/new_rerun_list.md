# 全链路重跑清单与 relation-vector A+ 执行计划

最后更新：2026-05-03

本清单用于在 P0a 代码修复后，重新生成与当前代码逻辑一致的论文结果产物；同时补充 2026-05-02 后新增的 relation-vector A/A+ 机制路线、2026-05-03 老师建议扩展出的 A++ learned-edge specificity、baseline pseudo-edge、single-word stability、run7 retrieval 与 learning driver 分析，并把师兄旧探索中可参考的 cross-stage similarity/RDM 与 left-HPC network 结果转成新版补充链。命令假设从项目外层目录 `E:\PythonAnalysis` 执行；如果你已经在 `metaphoric/final_version` 内，请把命令中的 `metaphoric/final_version/` 前缀去掉。

## 0. 通用环境

```powershell
cd E:\PythonAnalysis
$py = "C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe"
$env:PYTHON_METAPHOR_ROOT = "E:/python_metaphor"
$embedding = "E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv"
$memoryDir = "E:/python_metaphor/paper_outputs/qc/memory_strength"
```

注意：
- 现在依赖 `rsa_config.py` 的 RSA/ROI-level 脚本必须显式设置 `METAPHOR_ROI_SET`，否则会直接报错。
- 如果前面的上游结果重跑了，下游依赖这些结果的脚本也必须重跑。
- `mechanism_behavior_prediction.py` 虽然已在 P0a 修复后跑过一次，但如果后续重新跑了 Model-RSA，它也必须再跑。

## 1. P0 主链：必须重跑

### 1.1 Behavior refined

原因：`itemlevel_brain_behavior.py` 和 S1 都依赖 refined behavior 输出。

```powershell
& $py metaphoric/final_version/behavior_analysis/behavior_refined.py
& $py metaphoric/final_version/figures/plot_behavior_results.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv`
- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_behavior_accuracy_rt.png`

### 1.2 Step 5C RSA

原因：`run_rsa_optimized.py` 已加入 completeness/fail-fast；下游的 item-level、representational connectivity、robustness、noise ceiling 都依赖新的 `rsa_itemwise_details.csv`。

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
}
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_<roi_set>/rsa_completeness_manifest.tsv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_<roi_set>/rsa_failed_cells.tsv`

### 1.3 RSA LMM 与 Step 5C 图

原因：RSA itemwise 更新后，LMM 参数表和主图都需要同步。

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/rsa_lmm.py
}

$env:METAPHOR_ROI_SET = "main_functional"
& $py metaphoric/final_version/figures/plot_step5c_rsa_main.py
& $py metaphoric/final_version/figures/plot_roi_set_robustness.py
```

新增 FDR 产物：
- `E:/python_metaphor/paper_outputs/tables_si/table_rsa_lmm_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_<roi_set>/lmm_<roi_set>/rsa_lmm_fdr.tsv`

### 1.4 M7 记忆强度表

原因：Model-RSA 的 M7 模型依赖这批 subject-level memory strength 文件。

```powershell
& $py metaphoric/final_version/rsa_analysis/build_memory_strength_table.py --score-modes binary continuous_confidence
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/memory_strength/memory_strength_binary_sub-XX.tsv`
- `E:/python_metaphor/paper_outputs/qc/memory_strength/memory_strength_continuous_confidence_sub-XX.tsv`
- `E:/python_metaphor/paper_outputs/qc/memory_strength/memory_strength_manifest.tsv`

### 1.5 Model-RSA

原因：S1、Delta-rho LMM、机制图都依赖 `model_rdm_subject_metrics.tsv`。三套 ROI 都要跑。

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/model_rdm_comparison.py `
    --embedding-file $embedding `
    --memory-strength-dir $memoryDir `
    --analysis-mode pooled `
    --enabled-models M1_condition M2_pair M3_embedding M7_binary M7_continuous_confidence M8_reverse_pair `
    --primary-models M3_embedding M8_reverse_pair
}

& $py metaphoric/final_version/figures/plot_model_rsa_mechanism.py --condition-group all
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature/model_rdm_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_literature_spatial/model_rdm_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_model_rsa_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_<roi_set>/model_rdm_group_summary_fdr.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png`

### 1.6 Delta-rho LMM

原因：Model-RSA 更新后，跨 ROI 机制汇总必须同步。

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/delta_rho_lmm.py --family-split
}
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/<roi_set>/delta_rho_lmm_params.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/delta_rho_lmm/<roi_set>/delta_rho_long.tsv`

### 1.7 S1 机制到行为预测

原因：P0a 已修复 pooled/all 机制分数被错误展开为 condition-specific 的问题。若前面 Model-RSA 重跑，本脚本也必须重跑。

```powershell
& $py metaphoric/final_version/brain_behavior/mechanism_behavior_prediction.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv`
- `E:/python_metaphor/paper_outputs/qc/mechanism_behavior_prediction_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/mechanism_behavior_input_scope.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png`

### 1.8 item-level brain-behavior

原因：依赖新的 behavior refined 和 RSA itemwise。

```powershell
& $py metaphoric/final_version/brain_behavior/itemlevel_brain_behavior.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_itemlevel_coupling.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_full.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_interaction.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_rt.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_full_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_itemlevel_coupling_interaction_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/itemlevel_behavior_neural_long.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling.png`
- `E:/python_metaphor/paper_outputs/figures_main/fig_itemlevel_coupling_rt.png`

## 2. RSA 下游补充链：RSA 重跑后必须刷新

### 2.1 Representational connectivity

原因：依赖三套 ROI 的最新 `rsa_itemwise_details.csv`。

```powershell
& $py metaphoric/final_version/representation_analysis/representational_connectivity.py
& $py metaphoric/final_version/figures/plot_repr_connectivity_final.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_repr_connectivity.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_subject.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_repr_connectivity_edges.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_repr_connectivity.png`

### 2.2 Robustness suite

原因：依赖最新 Step 5C subject-level interaction delta。

```powershell
& $py metaphoric/final_version/rsa_analysis/robustness_suite.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_si/table_bayes_factors.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_bootstrap_ci.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_loso.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_splithalf.tsv`

### 2.3 Noise ceiling

原因：依赖最新 RSA itemwise；如果 Model-RSA audit 文件更新，也会同步更新 Model-RSA ceiling。

```powershell
& $py metaphoric/final_version/rsa_analysis/noise_ceiling.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_si/table_noise_ceiling.tsv`
- `E:/python_metaphor/paper_outputs/figures_si/fig_noise_ceiling.png`

## 3. Relation-vector 机制链：A/A+ 当前新增主线

这一部分是 2026-05-02 后新增的机制路线。A0-A2 已经跑通，用于回答 relation-vector geometry 是否发生 pre/post realignment；A3-A5 是下一步深挖，用于回答 YY 相对 KJ 是否有更强的 relation decoupling，以及这些结果是否与 Step 5C pair differentiation 汇合。

### 3.1 A0 构建 relation-vector model RDM

```powershell
& $py metaphoric/final_version/rsa_analysis/build_relation_vectors.py `
  --stimuli-template E:/python_metaphor/stimuli_template.csv `
  --embedding-file $embedding `
  --output-dir E:/python_metaphor/paper_outputs/qc/relation_vectors `
  --conditions yy kj
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_embedding_coverage.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz`
- `E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_collinearity.tsv`

验收：
- embedding coverage 全部为 `ok`。
- `yy` 和 `kj` 各 35 个 pair。
- 注意 `M9_relation_vector_direct`、`M9_relation_vector_reverse`、`M9_relation_vector_direction_only` 可能共线；主文解释不要过度强调 directionality。

### 3.2 A1 ROI-level relation-vector Model-RSA

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  & $py metaphoric/final_version/rsa_analysis/relation_vector_rsa.py `
    --pattern-root E:/python_metaphor/pattern_root `
    --roi-set $roi `
    --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
    --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
    --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
    --output-dir "E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_$roi" `
    --conditions yy kj `
    --neural-rdm-types relation_vector pair_centroid
}
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_group_summary_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_cell_qc.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_rsa.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_rsa_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_subject_metrics.tsv`

验收：
- 三套 ROI set 的 `relation_vector_cell_qc.tsv` 全部为 `ok`。
- 主文先看 `neural_rdm_type = relation_vector` 且 model 为 `M9_relation_vector_direct` 或 `M9_relation_vector_abs` 的 primary-family FDR。
- 若显著结果主要为负向 `post - pre`，解释为 `relation-vector realignment / decoupling`，不要写成 alignment 增强。

### 3.3 A2 relation-vector 主机制图

```powershell
& $py metaphoric/final_version/figures/plot_relation_vector_rsa.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

关键产物：
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_rsa.png`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_rsa.pdf`
- `E:/python_metaphor/paper_outputs/figures_main/table_relation_vector_rsa_top.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_rsa_full.tsv`

### 3.4 B1/B2 relation-vector 脑-行为桥接

当前 B1 为 subject-level 版本，不含 pair-level relation fit。

```powershell
& $py metaphoric/final_version/brain_behavior/relation_behavior_prediction.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_main_functional `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature_spatial `
  --behavior-summary E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_relation_behavior_prediction.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/relation_behavior/relation_behavior_subject_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_behavior/relation_behavior_join_qc.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_behavior_prediction.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_relation_behavior_prediction_fdr.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_behavior_prediction.png`

验收：
- subject-level merge 至少 26 名被试。
- 若 primary-family q 未达 .10，只作为候选行为桥接，不作为主机制。

### 3.5 C1/C2 relation-vector learning trajectory

当前 C1 用 learning-stage pair/item patterns 检验 run4 - run3 的 pair-pattern RDM alignment，不是 cue/target word-level relation vector。

```powershell
& $py metaphoric/final_version/temporal_dynamics/relation_learning_dynamics.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
  --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_relation_learning_dynamics.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

关键产物：
- `E:/python_metaphor/paper_outputs/qc/relation_learning_dynamics/relation_learning_dynamics_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_learning_dynamics/relation_learning_group_summary_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_learning_dynamics.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_relation_learning_dynamics_subject.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_learning_dynamics.png`

验收：
- `relation_learning_window_qc.tsv` 全部 ok。
- 若 primary-family q 不显著，C 阶段降级为 SI/边界结果。

### 3.6 A3 下一步：YY-KJ relation decoupling contrast

计划新增脚本：

```text
rsa_analysis/relation_vector_condition_contrast.py
figures/plot_relation_vector_condition_contrast.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/relation_vector_condition_contrast.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_main_functional `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature_spatial `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_relation_vector_condition_contrast.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_subject.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_condition_contrast.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_condition_contrast.png`

核心指标：

```text
relation_decoupling = pre_alignment - post_alignment
YY_minus_KJ_decoupling = decoupling_YY - decoupling_KJ
```

验收：
- primary models: `M9_relation_vector_direct`, `M9_relation_vector_abs`。
- 若 `YY_minus_KJ_decoupling > 0` 且 q < .05，说明该 ROI 的 relation realignment 更偏隐喻学习。

### 3.7 A4 下一步：A × Step5C conjunction

计划新增脚本：

```text
rsa_analysis/relation_step5c_conjunction.py
figures/plot_relation_step5c_conjunction.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/relation_step5c_conjunction.py `
  --relation-contrast E:/python_metaphor/paper_outputs/qc/relation_vector_contrast/relation_vector_condition_contrast_subject.tsv `
  --step5c-summary E:/python_metaphor/paper_outputs/figures_main/table_step5c_roi_delta.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_relation_step5c_conjunction.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/relation_vector_conjunction/relation_step5c_conjunction.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_step5c_conjunction.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_step5c_conjunction.png`

验收：
- 找到同时满足 `YY pair differentiation > KJ` 和 `YY relation decoupling > KJ` 的 ROI。
- 若共同 ROI 集中在 temporal pole、pMTG、IFG、AG/pSTS 或空间上下文网络，可作为主机制候选。

### 3.8 A5 下一步：network-level dissociation

计划新增脚本：

```text
rsa_analysis/relation_vector_network_lmm.py
figures/plot_relation_vector_network_dissociation.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/relation_vector_network_lmm.py `
  --relation-subject-metrics E:/python_metaphor/paper_outputs/tables_si/table_relation_vector_subject_metrics.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_relation_vector_network_dissociation.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/relation_vector_network/relation_vector_network_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/relation_vector_network/relation_vector_network_lmm_params.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_network_dissociation.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_network_dissociation.png`

验收：
- 重点看 `condition × network`。
- 若 `YY` 在 semantic/metaphor network 更强，`KJ` 在 spatial/context network 更强，可作为 A+ 最强解释。

### 3.9 P0 下一步：edge specificity + baseline pseudo-edge

目的：验证 pair similarity 下降是否特异于学习过的关系边，而不是所有词之间都同步下降。该分析同时把 baseline 按固定 `jx1-jx2` 方式构建 pseudo-pair，作为未学习关系边的关键对照。

计划新增脚本：

```text
rsa_analysis/edge_specificity_rsa.py
figures/plot_edge_specificity.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/edge_specificity_rsa.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --stimuli-template E:/python_metaphor/stimuli_template.csv `
  --baseline-pair-mode fixed_jx `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --conditions yy kj baseline

& $py metaphoric/final_version/figures/plot_edge_specificity.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_similarity_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_specificity_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_construction_qc.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_edge_specificity.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_edge_specificity.png`

验收：
- `trained_edge`、`untrained_nonedge`、`baseline_pseudo_edge` 数量与 metadata 匹配，且每名被试覆盖完整。
- 最关键对比是 `YY trained_edge drop > YY untrained_nonedge drop` 与 `YY trained_edge drop > baseline_pseudo_edge drop`。
- 如果 untrained non-edge 不下降或上升，而 trained edge 下降，可把主线写为 learned relation edge differentiation。

### 3.10 P0 下一步：single-word stability

目的：确认 pair similarity 下降不是单词表征整体变差，而是词身份稳定背景下的关系边重组。

计划新增脚本：

```text
rsa_analysis/word_stability_analysis.py
figures/plot_word_stability.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/word_stability_analysis.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --stimuli-template E:/python_metaphor/stimuli_template.csv `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --conditions yy kj baseline

& $py metaphoric/final_version/figures/plot_word_stability.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/word_stability/word_stability_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/word_stability/word_neighborhood_change.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_word_stability.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_word_stability.png`

验收：
- same-word stability 至少可在多数 ROI 中可靠估计。
- 若 same-word stability 保持稳定，同时 trained pair similarity 下降，优先解释为 relation reorganization，而不是 global degradation。

### 3.11 P1 下一步：run7 retrieval geometry 与回忆成功预测

目的：把 run7 回忆阶段作为第三时间点，检验 pre/post 的表征重组是否延续到 retrieval，并进一步预测 trial-level recall success。

计划新增脚本：

```text
rsa_analysis/retrieval_pair_similarity.py
brain_behavior/retrieval_success_prediction.py
figures/plot_retrieval_geometry.py
figures/plot_retrieval_prediction.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/retrieval_pair_similarity.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --conditions yy kj

& $py metaphoric/final_version/brain_behavior/retrieval_success_prediction.py `
  --retrieval-geometry E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_similarity_long.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_retrieval_geometry.py `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_retrieval_prediction.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_similarity_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_trajectory_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/retrieval_prediction/retrieval_prediction_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/retrieval_prediction/retrieval_prediction_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_retrieval_geometry.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_retrieval_prediction.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_retrieval_geometry.png`
- `E:/python_metaphor/paper_outputs/figures_main/fig_retrieval_prediction.png`

验收：
- run7 metadata 能区分 cue/target、condition 与 recall success。
- 优先检验 `pre -> post -> retrieval` trajectory、success vs failure 的 retrieval similarity、post-to-retrieval stability。
- 若预测不显著，该分析仍可作为 retrieval-stage descriptive evidence，不作为主机制。

### 3.12 P1 下一步：learning driver 与中介/调节筛选

目的：回答“学习阶段哪些机制让词相似性变差”。执行上先做 driver screening，只有 driver path 成立时才进入中介/调节。

计划新增脚本：

```text
temporal_dynamics/learning_driver_analysis.py
figures/plot_learning_driver_analysis.py
```

计划命令：

```powershell
& $py metaphoric/final_version/temporal_dynamics/learning_driver_analysis.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --edge-specificity E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_similarity_long.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_learning_driver_analysis.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/learning_driver/learning_driver_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_driver/learning_driver_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_driver/learning_mediation_screen.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_learning_driver.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_learning_driver.png`

验收：
- 先看 `learning_metric -> post_pre_pair_differentiation`。
- 只有 driver path 与 memory path 方向一致时，才报告 mediation/moderation。
- 若不成立，保留为探索性过程分析，不影响 A++ 主线。

### 3.13 P2 下一步：run3/run4 激活特异性与 ROI 覆盖审计

目的：确认隐喻句子激活是否可作为学习阶段 driver，并检查当前 metaphor/spatial ROI 是否覆盖元分析网络中的关键脑区。

计划新增脚本：

```text
activation_analysis/learning_activation_specificity.py
roi_analysis/roi_coverage_audit.py
```

计划命令：

```powershell
& $py metaphoric/final_version/activation_analysis/learning_activation_specificity.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/roi_analysis/roi_coverage_audit.py `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/activation_specificity/learning_activation_specificity.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_learning_activation_specificity.tsv`
- `E:/python_metaphor/paper_outputs/qc/roi_coverage/roi_coverage_audit.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_roi_coverage_audit.tsv`

验收：
- run3/run4 激活只作为 driver 候选，不单独升级为主线。
- ROI 审计只提出理论必要的 exploratory ROI，不盲目扩张 ROI family。

### 3.14 P1 下一步：师兄 cross_similarity -> cross-stage reinstatement

目的：把师兄旧 `cross_similarity` 思路转成新版 `same_pair - different_pair` 跨阶段复现指标，检验学习期形成的 pair/句子表征是否延续到 post 或 retrieval。旧代码线索包括 `H:/metaphor/cross_similarity/new_measure_dsm_corr.m`、`H:/metaphor/LSS/words_with_juzi/run3_with_run56` 和 `H:/metaphor/LSS/words_with_juzi/run3_with_run7`。

计划新增脚本：

```text
temporal_dynamics/cross_stage_reinstatement.py
figures/plot_cross_stage_reinstatement.py
```

计划命令：

```powershell
& $py metaphoric/final_version/temporal_dynamics/cross_stage_reinstatement.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --stage-pairs learn3-learn4 learn-post post-retrieval learn-retrieval `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --conditions yy kj

& $py metaphoric/final_version/figures/plot_cross_stage_reinstatement.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/cross_stage_reinstatement/cross_stage_reinstatement_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/cross_stage_reinstatement/cross_stage_reinstatement_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/cross_stage_reinstatement/cross_stage_reinstatement_qc.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_cross_stage_reinstatement.tsv`
- `E:/python_metaphor/paper_outputs/figures_si/fig_cross_stage_reinstatement.png`

核心指标：

```text
same_pair_cross_stage_similarity = sim(stage_a_i, stage_b_i)
different_pair_cross_stage_similarity = mean(sim(stage_a_i, stage_b_j), j != i)
cross_stage_reinstatement = same_pair - different_pair
```

验收：
- 每个 stage pair 都必须输出 subject × ROI × condition 的 coverage QC。
- 主看 `learn -> post` 与 `post -> retrieval`，run3 -> run4 只作为 learning-stage stability。
- 若 YY 的 cross-stage reinstatement 高于 KJ，且与 Step5C/A++ 方向一致，可作为过程证据；若不显著，降级为 SI 边界结果。

### 3.15 P1 下一步：师兄 cross_RDM -> cross-stage RDM stability

目的：把师兄旧 `cross_RDM` 思路转成阶段间整体 RDM stability / realignment，用于补充 relation-vector decoupling 的解释。旧代码线索包括 `H:/metaphor/cross_RDM/new_measure_dsm_corr.m` 和 `H:/metaphor/cross_RDM/rca_crossstage.m`。

计划新增脚本：

```text
rsa_analysis/cross_stage_rdm_stability.py
figures/plot_cross_stage_rdm_stability.py
```

计划命令：

```powershell
& $py metaphoric/final_version/rsa_analysis/cross_stage_rdm_stability.py `
  --pattern-root E:/python_metaphor/pattern_root `
  --roi-sets main_functional literature literature_spatial `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --stage-pairs pre-post learn3-learn4 learn-post post-retrieval `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --conditions yy kj

& $py metaphoric/final_version/figures/plot_cross_stage_rdm_stability.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/cross_stage_rdm/cross_stage_rdm_subject.tsv`
- `E:/python_metaphor/paper_outputs/qc/cross_stage_rdm/cross_stage_rdm_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/cross_stage_rdm/cross_stage_rdm_qc.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_cross_stage_rdm_stability.tsv`
- `E:/python_metaphor/paper_outputs/figures_si/fig_cross_stage_rdm_stability.png`

核心指标：

```text
rdm_stability = corr(vectorize(RDM_stage_a), vectorize(RDM_stage_b))
rdm_realignment = rdm_stability_post_related - rdm_stability_pre_related
```

验收：
- 只作为 SI 或机制补充，除非 `learn -> post` 在独立 ROI set 中出现稳定 FDR 结果。
- 若 global RDM stability 与 relation-vector decoupling 方向不同，不视为冲突；二者分别反映整体结构延续与局部关系向量重排。

### 3.16 P1 下一步：hippocampal-binding ROI family 与师兄 left-HPC 网络结果的 legacy confirmation

目的：先把当前已经存在的 hippocampus ROI 从 `literature_spatial` 中单独提升为 hippocampal-binding family，再把师兄旧 whole-brain gPPI / beta-series connectivity 中最有解释力的 left-HPC -> PCC/Precuneus 线索，用当前 `final_version` 的 subject-level QC、planned target family、permutation/FDR 口径重新确认。旧结果只作为 hypothesis generator，不直接进入主文。

当前 ROI 状态：
- 已有：`litspat_L_hippocampus`
- 已有：`litspat_R_hippocampus`
- 已有相关上下文节点：PPA/PHG、RSC/PCC、Precuneus
- 因此优先级是先单独汇总 hippocampal-binding family，不是马上新增 mask。

旧结果线索：
- `H:/metaphor/gppi/lHPC_seed/scripts/run3.m`
- `H:/metaphor/gppi/output_run3/second_lhip/yy-kj/cluster_fwe_001_peaks.csv`
- `H:/metaphor/gppi/output_run4/second_lhip/kj-yy/cluster_FWE_peaks.csv`
- `H:/metaphor/BSC/seed_to_voxel_correlation beta_run34.ipynb`
- `H:/metaphor/BSC/F_run3/fwe_data_peaks.csv`

计划新增或扩展脚本：

```text
roi_analysis/hippocampal_binding_summary.py
connectivity_analysis/legacy_lhip_gppi_confirmation.py
figures/plot_hippocampal_binding_summary.py
figures/plot_legacy_lhip_gppi.py
```

计划命令：

```powershell
& $py metaphoric/final_version/roi_analysis/hippocampal_binding_summary.py `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --include-analyses step5c edge_specificity word_stability relation_vector retrieval_geometry `
  --hippocampal-rois litspat_L_hippocampus litspat_R_hippocampus `
  --context-roi-keywords PPA PHG RSC PCC Precuneus

& $py metaphoric/final_version/connectivity_analysis/legacy_lhip_gppi_confirmation.py `
  --seed-roi-names litspat_L_hippocampus `
  --target-roi-sets literature literature_spatial `
  --planned-target-families pcc_precuneus ag_mtg ifg_mfg hippocampal_context `
  --stage learn `
  --conditions yy kj `
  --contrast yy-kj `
  --deconvolution ridge `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_hippocampal_binding_summary.py `
  --paper-output-root E:/python_metaphor/paper_outputs

& $py metaphoric/final_version/figures/plot_legacy_lhip_gppi.py `
  --paper-output-root E:/python_metaphor/paper_outputs
```

计划产物：
- `E:/python_metaphor/paper_outputs/qc/hippocampal_binding/hippocampal_binding_summary.tsv`
- `E:/python_metaphor/paper_outputs/qc/hippocampal_binding/hippocampal_binding_subject.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_hippocampal_binding_summary.tsv`
- `E:/python_metaphor/paper_outputs/figures_si/fig_hippocampal_binding_summary.png`
- `E:/python_metaphor/paper_outputs/qc/legacy_gppi/legacy_gppi_subject.tsv`
- `E:/python_metaphor/paper_outputs/qc/legacy_gppi/legacy_gppi_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/qc/legacy_gppi/legacy_gppi_wholebrain_peaks.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_legacy_gppi_lhip.tsv`
- `E:/python_metaphor/paper_outputs/figures_si/fig_legacy_gppi_lhip.png`

验收：
- hippocampal-binding summary 必须先汇总现有 hippocampus ROI 在 Step5C/A++/relation-vector/retrieval 中的方向与校正结果。
- 如果 hippocampus 的 learned-edge 或 retrieval reinstatement 结果方向稳定，它应上升为机制性 SI 结果，而不是只作为 gPPI seed。
- primary planned target 是 PCC/Precuneus；AG/MTG、IFG/MFG、hippocampal-context network 为补充 family。
- 若 left-HPC -> PCC/Precuneus 的 `YY > KJ` 通过 planned family，可写为 SI 网络支持。
- 若只出现未校正趋势，写作 legacy exploratory clue。
- 不复现则不影响主线，并保留当前 `final_version` 的 gPPI 阴性结论。

## 4. Searchlight 链

P0a 已修复两个 searchlight 的 pre/post subject pairing。若已存在 subject-level `.nii.gz` maps，优先使用 `--reuse-subject-maps`，这样只重跑 group-level 统计和 QC manifest，不重新计算最慢的 subject maps。

### 4.1 Pair-similarity searchlight 主版

```powershell
& $py metaphoric/final_version/representation_analysis/pair_similarity_searchlight.py `
  --reuse-subject-maps `
  --permutation-backend auto
```

如果需要完全重算 subject-level maps，去掉 `--reuse-subject-maps`：

```powershell
& $py metaphoric/final_version/representation_analysis/pair_similarity_searchlight.py `
  --permutation-backend auto
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png`
- `E:/python_metaphor/paper_outputs/qc/pair_similarity_searchlight/searchlight_subject_pairing_manifest.tsv`

### 4.2 RD searchlight 探索版

```powershell
& $py metaphoric/final_version/representation_analysis/rd_searchlight.py `
  --reuse-subject-maps `
  --permutation-backend auto
```

如果需要完全重算 subject-level maps，去掉 `--reuse-subject-maps`：

```powershell
& $py metaphoric/final_version/representation_analysis/rd_searchlight.py `
  --permutation-backend auto
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_searchlight_peaks.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_searchlight_delta.png`
- `E:/python_metaphor/paper_outputs/qc/rd_searchlight/searchlight_subject_pairing_manifest.tsv`

## 5. gPPI 链

当前 `final_version` 的主 gPPI 链是 ROI-to-ROI planned summary；师兄旧结果更接近 whole-brain left-HPC seed exploratory gPPI。两者不能混为同一结论。主链仍按 temporal-pole seed 跑；师兄结果只通过 3.16 的 legacy confirmation 做 SI 级验证。

如果 raw gPPI 已经跑完，只需要轻量导出汇总：

```powershell
& $py metaphoric/final_version/connectivity_analysis/export_gppi_summary.py
```

如果需要从 raw gPPI 重新开始：

```powershell
& $py metaphoric/final_version/connectivity_analysis/gPPI_analysis.py `
  --seed-roi-names `
    func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort `
    func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort `
  --target-roi-sets literature literature_spatial `
  --stage prepost `
  --conditions yy kj `
  --deconvolution ridge

& $py metaphoric/final_version/connectivity_analysis/export_gppi_summary.py
```

关键产物：
- `E:/python_metaphor/paper_outputs/tables_main/table_gppi_summary_main.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_gppi_target_peaks.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_gppi_summary_full.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_gppi_target_level.tsv`

验收：
- 若主 gPPI 仍阴性，不影响 A++ / relation-vector 主线。
- legacy left-HPC 结果只在 3.16 中按 planned target family 单独判断，避免把旧探索和当前 planned gPPI 混在同一张主表里。

## 6. 可选补充链

这些脚本没有被本轮 P0a 直接改动，但如果你希望 `result.md` 中所有补充模块都和最新 RSA / behavior 主链完全同步，可以在主链完成后运行。

### 6.1 学习期 pattern

仅当 `E:/python_metaphor/pattern_root/sub-XX/learn_yy.nii.gz` 或 `learn_kj.nii.gz` 缺失时运行。

```powershell
& $py metaphoric/final_version/representation_analysis/stack_patterns.py
```

### 6.2 Learning dynamics

```powershell
& $py metaphoric/final_version/temporal_dynamics/learning_dynamics.py `
  E:/python_metaphor/pattern_root `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --roi-set literature

& $py metaphoric/final_version/temporal_dynamics/learning_dynamics.py `
  E:/python_metaphor/pattern_root `
  --paper-output-root E:/python_metaphor/paper_outputs `
  --roi-set main_functional

$env:METAPHOR_ROI_SET = "literature"
& $py metaphoric/final_version/figures/plot_learning_dynamics_final.py

$env:METAPHOR_ROI_SET = "main_functional"
& $py metaphoric/final_version/figures/plot_learning_dynamics_final.py
```

## 7. 最后文档更新

所有重跑完成后，再更新结果文档：

```powershell
# 这一步不是脚本命令，而是提醒：
# 1. 检查各 tables_main / tables_si / qc 产物的 LastWriteTime。
# 2. 重新把关键结果写入 metaphoric/final_version/result.md。
# 3. 对照 new_todo.md，把已完成项打勾。
```

最小必须完成顺序：

1. `behavior_refined.py`
2. 三套 `run_rsa_optimized.py`
3. 三套 `rsa_lmm.py` 与 Step 5C 图
4. `build_memory_strength_table.py`
5. 三套 `model_rdm_comparison.py`
6. 三套 `delta_rho_lmm.py`
7. `mechanism_behavior_prediction.py`
8. `itemlevel_brain_behavior.py`
9. `representational_connectivity.py`
10. `robustness_suite.py`
11. `noise_ceiling.py`
12. A0-A2 relation-vector 机制链，确认 realignment / decoupling 口径
13. P0 A++ edge specificity：trained edge、untrained non-edge、baseline pseudo-edge
14. P0 A++ single-word stability：same-word stability 与 word-neighborhood change
15. A3-A5 relation-vector 深挖链：condition contrast、conjunction、network dissociation
16. P1 run7 retrieval geometry 与 retrieval success prediction
17. P1 learning driver screening；只有路径成立时再做 mediation/moderation
18. P1 legacy-informed cross-stage reinstatement：师兄 `cross_similarity` 的新版 same-vs-different pair 版本
19. P1 legacy-informed cross-stage RDM stability：师兄 `cross_RDM` 的新版 RDM continuity / realignment 版本
20. P1 hippocampal-binding family summary：先单独汇总现有 L/R hippocampus ROI
21. P1 legacy left-HPC gPPI confirmation：只验证 planned PCC/Precuneus 等 target family
22. P2 run3/run4 activation specificity 与 ROI coverage audit
23. B1/B2 relation-vector 脑-行为桥接，作为补充而不是当前阻断项
24. C1/C2 relation-vector learning trajectory，作为补充而不是当前阻断项
25. 两个 searchlight 使用 `--reuse-subject-maps` 轻量重跑
26. `export_gppi_summary.py`
27. 更新 `result.md` / `result_new.md`
