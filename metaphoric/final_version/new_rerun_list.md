# Meta-ROI 主线重跑清单

最后更新：2026-05-03

本清单按新的论文故事重排：主文只围绕 `meta_metaphor` 与 `meta_spatial`，旧 `literature` / `literature_spatial` 作为文献 ROI sensitivity，学习阶段 GLM ROI `main_functional` 只作为本数据参考。当前不再写“三层 ROI 主结果”，也不把师兄旧探索直接混入主 family。

核心故事顺序：

```text
外部 meta ROI 定义
-> 行为结果
-> Step5C pair differentiation
-> learned-edge specificity
-> single-word stability control
-> relation-vector / network 机制补充
-> brain-behavior 与 robustness
-> 师兄 legacy 方法作为 SI/sensitivity
```

## 0. 环境与 ROI Family

```powershell
cd E:\PythonAnalysis
$py = "C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe"
$env:PYTHON_METAPHOR_ROOT = "E:/python_metaphor"
$paper = "E:/python_metaphor/paper_outputs"
$manifest = "E:/python_metaphor/roi_library/manifest.tsv"
$patternRoot = "E:/python_metaphor/pattern_root"
$stimuli = "E:/python_metaphor/stimuli_template.csv"
$embedding = "E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv"
$memoryDir = "E:/python_metaphor/paper_outputs/qc/memory_strength"

$primaryRois = @("meta_metaphor", "meta_spatial")
$literatureSensitivityRois = @("literature", "literature_spatial")
$glmReferenceRois = @("main_functional")
```

注意：

- 主文 FDR family 只用 `$primaryRois`。
- `meta_spatial` 内的 `meta_L_hippocampus` / `meta_R_hippocampus` 是 hippocampal-binding planned subfamily，不另起第三套主 ROI。
- `literature` / `literature_spatial` 用来回答“换成旧文献/atlas ROI 是否同向”。
- `main_functional` 用来回答“学习阶段 GLM 定位 ROI 是否参考性支持”，不能进入主结论。

## 1. P0: ROI Library 与 Mask QC

目的：先确认新版 mask 是真正可追踪的外部先验，而不是本数据 GLM ROI。

输入：

```text
E:/python_metaphor/roi_library/meta_sources/meta_roi_peaks.tsv
metaphoric/final_version/roi_management/meta_roi_peaks_curated.tsv
metaphoric/final_version/roi_management/meta_roi_decision_review.tsv
metaphoric/final_version/roi_management/legacy_mask_review.md
```

当前 planned masks：

- `meta_metaphor`: L/R IFG, L/R temporal pole/ATL, L/R AG, L/R pMTG-pSTS。
- `meta_spatial`: L/R hippocampus, L/R PPA/PHG, L/R RSC/PCC, L/R PPC/SPL, L/R precuneus。
- backup/SI: OPA/LOC, dmPFC/MFG, 师兄 aHPC/pHPC 或 rostral/caudal hippocampus。

命令：

```powershell
& $py metaphoric/final_version/roi_management/build_meta_roi_library.py `
  --peaks E:/python_metaphor/roi_library/meta_sources/meta_roi_peaks.tsv `
  --reference-image E:/python_metaphor/pattern_root/sub-01/pre_yy.nii.gz `
  --output-dir E:/python_metaphor/roi_library `
  --manifest $manifest `
  --roi-sets meta_metaphor meta_spatial
```

验收：

- `E:/python_metaphor/roi_library/meta_sources/meta_roi_build_manifest.json` 中 `n_roi_masks = 18`。
- `E:/python_metaphor/roi_library/meta_sources/meta_roi_manifest_rows.tsv` 每个 mask 非空。
- 当前 QC 应为：每个 6 mm sphere 约 `123` voxels，shape `91x109x91`，同一 ROI set 内无 overlap。

## 2. P1: 行为主结果

目的：所有脑-行为与图示依赖 refined behavior；行为图也必须带显著性标识。

```powershell
& $py metaphoric/final_version/behavior_analysis/behavior_refined.py
& $py metaphoric/final_version/figures/plot_behavior_results.py
```

产物：

- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv`
- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_behavior_accuracy_rt.png`

## 3. P2: Step5C Pair Differentiation 主效应

目的：在独立 `meta_metaphor` / `meta_spatial` 中逐 ROI 检验 `YY/Metaphor` pre-post pair similarity drop。不能把同类 ROI 合并成跨 ROI 主效应。

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
  & $py metaphoric/final_version/rsa_analysis/rsa_lmm.py
}

$env:METAPHOR_ROI_SET = "meta_metaphor"
& $py metaphoric/final_version/figures/plot_step5c_rsa_main.py
& $py metaphoric/final_version/figures/plot_roi_set_robustness.py
```

产物：

- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_meta_metaphor/rsa_itemwise_details.csv`
- `E:/python_metaphor/paper_outputs/qc/rsa_results_optimized_meta_spatial/rsa_itemwise_details.csv`
- `E:/python_metaphor/paper_outputs/tables_si/table_rsa_lmm_fdr.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_step5c_main.png`

验收：

- `rsa_completeness_manifest.tsv` 无关键缺失。
- `table_step5c_roi_condition_group.tsv` 逐 ROI 给出 condition × time 过程数据。
- 主文不写 pooled ROI-level result，只写每个 ROI 或 planned subfamily。

## 4. P3: Learned-Edge Specificity 主机制

目的：证明 pair similarity drop 是 learned relation edge 的分化，而不是所有词都一起下降。

```powershell
& $py metaphoric/final_version/rsa_analysis/edge_specificity_rsa.py `
  --pattern-root $patternRoot `
  --stimuli-template $stimuli `
  --roi-manifest $manifest `
  --roi-sets meta_metaphor meta_spatial `
  --conditions yy kj baseline `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_edge_specificity.py `
  --paper-output-root $paper
```

产物：

- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_similarity_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_process_group.tsv`
- `E:/python_metaphor/paper_outputs/qc/edge_specificity/edge_contrast_process_components.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_edge_specificity.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_edge_specificity.png`

验收：

- 图和表都必须显示 trained edge、untrained non-edge、baseline pseudo-edge 的本身过程值。
- 关键对比：`YY trained_edge drop > YY untrained_nonedge drop`，以及 `YY trained_edge drop > baseline pseudo-edge drop`。

## 5. P4: Single-Word Stability Control

目的：排除“神经表征整体变差/噪声变大”的弱解释。理想结果是 single-word identity 稳定，同时 trained pair edge 分化。

```powershell
& $py metaphoric/final_version/rsa_analysis/word_stability_analysis.py `
  --pattern-root $patternRoot `
  --stimuli-template $stimuli `
  --roi-manifest $manifest `
  --roi-sets meta_metaphor meta_spatial `
  --conditions yy kj baseline `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_word_stability.py `
  --paper-output-root $paper
```

产物：

- `E:/python_metaphor/paper_outputs/qc/word_stability/word_stability_long.tsv`
- `E:/python_metaphor/paper_outputs/qc/word_stability/word_stability_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_word_stability.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_word_stability.png`

## 6. P5: Model-RSA 与 Relation-Vector 机制补充

### 6.1 旧 Model-RSA

目的：报告已用哪些 RDM 模型，以及这些模型的理论意义；作为 relation-vector 前的机制基线。

```powershell
& $py metaphoric/final_version/rsa_analysis/build_memory_strength_table.py `
  --score-modes binary continuous_confidence

foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/model_rdm_comparison.py `
    --embedding-file $embedding `
    --memory-strength-dir $memoryDir `
    --analysis-mode pooled `
    --enabled-models M1_condition M2_pair M3_embedding M7_binary M7_continuous_confidence M8_reverse_pair `
    --primary-models M3_embedding M8_reverse_pair

  & $py metaphoric/final_version/rsa_analysis/delta_rho_lmm.py --family-split
}

& $py metaphoric/final_version/figures/plot_model_rsa_mechanism.py --condition-group all
```

产物：

- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_meta_metaphor/model_rdm_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/model_rdm_results_meta_spatial/model_rdm_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_model_rsa_fdr.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_model_rsa_mechanism.png`

### 6.2 Relation-Vector A0-A2

目的：检验神经 RDM 是否与 cue-to-target relation vector 对齐或发生 realignment / decoupling。

```powershell
& $py metaphoric/final_version/rsa_analysis/build_relation_vectors.py `
  --stimuli-template $stimuli `
  --embedding-file $embedding `
  --output-dir E:/python_metaphor/paper_outputs/qc/relation_vectors `
  --conditions yy kj

foreach ($roi in $primaryRois) {
  & $py metaphoric/final_version/rsa_analysis/relation_vector_rsa.py `
    --pattern-root $patternRoot `
    --roi-set $roi `
    --roi-manifest $manifest `
    --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
    --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
    --output-dir "E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_$roi" `
    --conditions yy kj `
    --neural-rdm-types relation_vector pair_centroid
}

& $py metaphoric/final_version/figures/plot_relation_vector_rsa.py `
  --paper-output-root $paper
```

产物：

- `E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz`
- `E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_subject_metrics.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_rsa.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_vector_rsa.png`

### 6.3 Relation-Vector A3-A5 深挖

目的：检验 YY-KJ condition contrast、Step5C/edge 与 relation-vector 的 conjunction，以及 `meta_metaphor` vs `meta_spatial` 的网络分工。

```powershell
& $py metaphoric/final_version/rsa_analysis/relation_vector_condition_contrast.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_meta_metaphor `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_meta_spatial `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_relation_vector_condition_contrast.py `
  --paper-output-root $paper

& $py metaphoric/final_version/rsa_analysis/relation_step5c_conjunction.py `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_relation_step5c_conjunction.py `
  --paper-output-root $paper

& $py metaphoric/final_version/rsa_analysis/relation_vector_network_lmm.py `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_relation_vector_network_dissociation.py `
  --paper-output-root $paper
```

产物：

- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_condition_contrast.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_step5c_conjunction.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_vector_network_dissociation.tsv`
- 对应三张 `figures_main/fig_relation_*.png`

解释规则：

- 若 A3/A5 不支持 YY-specific relation decoupling，不强写主机制。
- 当前最稳主线仍是 Step5C + learned-edge specificity；relation-vector 作为机制边界与补充。

## 7. P6: Brain-Behavior Bridge

目的：在主神经结果稳定后，再检验是否预测最终记忆或 YY-KJ 行为差异。

```powershell
& $py metaphoric/final_version/brain_behavior/mechanism_behavior_prediction.py

& $py metaphoric/final_version/brain_behavior/itemlevel_brain_behavior.py

& $py metaphoric/final_version/brain_behavior/relation_behavior_prediction.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_meta_metaphor `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_meta_spatial `
  --behavior-summary E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_relation_behavior_prediction.py `
  --paper-output-root $paper
```

产物：

- `E:/python_metaphor/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_itemlevel_coupling.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_relation_behavior_prediction.tsv`
- `E:/python_metaphor/paper_outputs/figures_main/fig_relation_behavior_prediction.png`

解释规则：

- 脑-行为若未通过校正，只写作 boundary / exploratory bridge。
- 不让脑-行为阴性削弱 learned-edge 主机制。

## 8. P7: Robustness 与 RSA 下游同步

目的：所有依赖 RSA itemwise 或 subject-level 指标的补充结果都要同步刷新。

```powershell
& $py metaphoric/final_version/representation_analysis/representational_connectivity.py
& $py metaphoric/final_version/figures/plot_repr_connectivity_final.py

& $py metaphoric/final_version/rsa_analysis/robustness_suite.py

& $py metaphoric/final_version/rsa_analysis/noise_ceiling.py
& $py metaphoric/final_version/figures/plot_noise_ceiling.py
```

产物：

- `E:/python_metaphor/paper_outputs/tables_main/table_repr_connectivity.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_bootstrap_ci.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_loso.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_splithalf.tsv`
- `E:/python_metaphor/paper_outputs/tables_si/table_noise_ceiling.tsv`

## 9. P8: Learning Dynamics 与 Searchlight

这些结果不作为当前最小主线阻断项，但更新 meta ROI 后建议轻量刷新。

### 9.1 Learning Dynamics

```powershell
foreach ($roi in $primaryRois) {
  & $py metaphoric/final_version/temporal_dynamics/learning_dynamics.py `
    E:/python_metaphor/pattern_root `
    --paper-output-root $paper `
    --roi-set $roi

  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/figures/plot_learning_dynamics_final.py
}

& $py metaphoric/final_version/temporal_dynamics/relation_learning_dynamics.py `
  --pattern-root $patternRoot `
  --roi-sets meta_metaphor meta_spatial `
  --roi-manifest $manifest `
  --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
  --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
  --paper-output-root $paper

& $py metaphoric/final_version/figures/plot_relation_learning_dynamics.py `
  --paper-output-root $paper
```

### 9.2 Searchlight

若 subject-level maps 已存在，优先复用，只刷新 group-level 与图表。

```powershell
& $py metaphoric/final_version/representation_analysis/pair_similarity_searchlight.py `
  --reuse-subject-maps `
  --permutation-backend auto

& $py metaphoric/final_version/representation_analysis/rd_searchlight.py `
  --reuse-subject-maps `
  --permutation-backend auto
```

产物：

- `E:/python_metaphor/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_searchlight_peaks.tsv`

## 10. P9: gPPI 与师兄 Legacy Network

当前仓库已有 `gPPI_analysis.py` 和 `export_gppi_summary.py`，但尚未实现专门的 `legacy_lhip_gppi_confirmation.py`。因此当前执行策略是：先导出现有 gPPI summary；师兄 left-HPC -> PCC/Precuneus 作为结果解释和后续脚本 TODO。

```powershell
& $py metaphoric/final_version/connectivity_analysis/export_gppi_summary.py
```

如果需要从 raw gPPI 重新开始，先确认 seed 名称已经存在于新版 manifest。旧 GLM temporal-pole seed 只适合作 reference，不适合作 meta 主 seed：

```powershell
& $py metaphoric/final_version/connectivity_analysis/gPPI_analysis.py `
  --seed-roi-names meta_L_temporal_pole meta_R_temporal_pole `
  --target-roi-sets meta_metaphor meta_spatial `
  --stage prepost `
  --conditions yy kj `
  --deconvolution ridge

& $py metaphoric/final_version/connectivity_analysis/export_gppi_summary.py
```

解释规则：

- 主 gPPI 阴性不影响 Step5C / learned-edge 主线。
- 师兄 left-HPC -> PCC/Precuneus 需要后续 planned legacy-confirmation 脚本，不能直接把旧 whole-brain peak 当新版主证据。

## 11. P10: Sensitivity / Reference ROI

目的：保留旧文献 ROI 和学习阶段 GLM ROI 的参考价值，但不进入主文主 family。

### 11.1 Literature Sensitivity

```powershell
foreach ($roi in $literatureSensitivityRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
  & $py metaphoric/final_version/rsa_analysis/rsa_lmm.py

  & $py metaphoric/final_version/rsa_analysis/model_rdm_comparison.py `
    --embedding-file $embedding `
    --memory-strength-dir $memoryDir `
    --analysis-mode pooled `
    --enabled-models M1_condition M2_pair M3_embedding M7_binary M7_continuous_confidence M8_reverse_pair `
    --primary-models M3_embedding M8_reverse_pair
}
```

### 11.2 Main Functional Reference

```powershell
foreach ($roi in $glmReferenceRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
  & $py metaphoric/final_version/rsa_analysis/rsa_lmm.py
}
```

解释规则：

- `literature/literature_spatial` 用于“旧文献/atlas ROI 是否同向”。
- `main_functional` 用于“本数据学习阶段定位 ROI 是否参考性支持”。
- 这两类不与 `meta_metaphor/meta_spatial` 混在同一主 FDR family。

## 12. 目前不要写成可执行主链的 TODO

这些分析有价值，但当前仓库中尚未发现对应脚本，先不要放进必须重跑顺序：

- `rsa_analysis/retrieval_pair_similarity.py`
- `brain_behavior/retrieval_success_prediction.py`
- `temporal_dynamics/cross_stage_reinstatement.py`
- `rsa_analysis/cross_stage_rdm_stability.py`
- `roi_analysis/hippocampal_binding_summary.py`
- `connectivity_analysis/legacy_lhip_gppi_confirmation.py`
- `activation_analysis/learning_activation_specificity.py`
- `roi_analysis/roi_coverage_audit.py`
- `figures/plot_hippocampal_binding_summary.py`
- `figures/plot_legacy_lhip_gppi.py`

如果后续要补，优先级是：

1. hippocampal-binding summary：汇总 `meta_spatial` 中 L/R hippocampus 的 Step5C、edge、word、relation-vector。
2. legacy left-HPC gPPI confirmation：planned target 只看 PCC/Precuneus、AG/MTG、IFG/MFG、hippocampal-context。
3. cross-stage reinstatement / cross-stage RDM：吸收师兄 cross_similarity / cross_RDM，但必须用当前 QC/FDR 口径重写。
4. retrieval geometry：只有找到或生成 run7 neural patterns 后再做。

## 13. 最小必须完成顺序

1. P0 meta ROI library 与 mask QC。
2. P1 behavior refined 与行为图。
3. P2 Step5C RSA + RSA LMM + Step5C 主图。
4. P3 learned-edge specificity + 过程数据图。
5. P4 single-word stability。
6. P5.1 Model-RSA。
7. P5.2-P5.3 relation-vector A0-A5。
8. P6 brain-behavior bridge。
9. P7 robustness / noise ceiling / representational connectivity。
10. P8 learning dynamics 与 searchlight。
11. P9 gPPI summary。
12. P10 literature / main_functional sensitivity。
13. 更新 `result_new.md`，按 meta ROI 主线重写结果与图表引用。
