# P0a 修复后的全链路重跑清单

最后更新：2026-05-01

本清单用于在 P0a 代码修复后，重新生成与当前代码逻辑一致的论文结果产物。命令假设从项目外层目录 `E:\PythonAnalysis` 执行；如果你已经在 `metaphoric/final_version` 内，请把命令中的 `metaphoric/final_version/` 前缀去掉。

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

## 3. Searchlight 链

P0a 已修复两个 searchlight 的 pre/post subject pairing。若已存在 subject-level `.nii.gz` maps，优先使用 `--reuse-subject-maps`，这样只重跑 group-level 统计和 QC manifest，不重新计算最慢的 subject maps。

### 3.1 Pair-similarity searchlight 主版

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

### 3.2 RD searchlight 探索版

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

## 4. gPPI 链

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

## 5. 可选补充链

这些脚本没有被本轮 P0a 直接改动，但如果你希望 `result.md` 中所有补充模块都和最新 RSA / behavior 主链完全同步，可以在主链完成后运行。

### 5.1 学习期 pattern

仅当 `E:/python_metaphor/pattern_root/sub-XX/learn_yy.nii.gz` 或 `learn_kj.nii.gz` 缺失时运行。

```powershell
& $py metaphoric/final_version/representation_analysis/stack_patterns.py
```

### 5.2 Learning dynamics

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

## 6. 最后文档更新

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
12. 两个 searchlight 使用 `--reuse-subject-maps` 轻量重跑
13. `export_gppi_summary.py`
14. 更新 `result.md`
