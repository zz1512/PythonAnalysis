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


---

## P11: Learning-phase RSA split profiles (word-level + learning sentence-level) (extend-learning-rsa-trajectory)

> Spec id: `extend-learning-rsa-trajectory`
> **注意：本章节仅记录命令，不在本机执行。**用户本机无原始数据，所有命令需在有 `${PYTHON_METAPHOR_ROOT}/pattern_root/` 与 `roi_library/manifest.tsv` 的机器上顺序跑。

原因：
- pre / post / retrieval 已有 word-level RSA 指标，learning (run3/run4) 则是 sentence-level 条件几何；本章节补齐 learning 阶段的 condition-level RDM (Primary)、constituent-to-sentence reinstatement (Secondary)、within-item reliability (Supplementary)。
- 与老师沟通后，**不再**把 `pre / run3 / run4 / post / retrieval` 串成一条主文 trajectory，因为 `pre/post/retrieval` 是两个词的比较，而 `run3/run4` 是句子的比较，跨格式直接连线容易被审稿人质疑。
- 主文改为两条线/两个模块：`pre → post → retrieval` 的 **word-level profile**，以及 `run3 → run4` 的 **learning sentence-level profile**。旧的 5 点总表仅保留为兼容输出 / SI 参考，不作主文强解释。
- 所有新脚本都复用 `stack_patterns.py` 已产出的 `pattern_root/sub-XX/{phase}_{yy,kj}.nii.gz + *_metadata.tsv`（含 `run / word_label / pair_id`）；**无需**再跑 LSS 与 stacking。

ROI family 口径（沿用第 0 节定义）：
- 主 family 仍是 `$primaryRois = @("meta_metaphor", "meta_spatial")`，主文 FDR 与主图都基于这两组。
- `$literatureSensitivityRois = @("literature", "literature_spatial")` 作为文献 sensitivity，进入 SI。
- `$glmReferenceRois = @("main_functional")` 仅作学习阶段 GLM 参考，不入主 FDR family，不出主图，仅生成表格。

依赖（输入）：
- `${PYTHON_METAPHOR_ROOT}/pattern_root/sub-XX/learn_yy.nii.gz`、`learn_kj.nii.gz` + `learn_{yy,kj}_metadata.tsv`（含 `run ∈ {3,4}`、`word_label`、`pair_id`）
- `${PYTHON_METAPHOR_ROOT}/pattern_root/sub-XX/{pre,post,retrieval}_{yy,kj}.nii.gz` + 对应 metadata
- `$manifest`（`include_in_rsa=true` 的 ROI manifest）

推荐运行顺序与依赖：
- 先跑 1）Primary condition-level RDM（同时生成兼容用的 5 点总表，以及主文需要的 `word-level` / `learning` 两套拆分表与组水平统计）。
- 2）与 3）可与 1）并行（读取相同 pattern，互不写入同一目录）。
- 最后跑 4）、5）绘图（依赖前三步产物）。

### P11.1 Primary: Learning condition-level RDM + split profiles

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/learning_condition_rdm.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets meta_metaphor meta_spatial `
    --paper-output-root $paper
}
```

默认产物（`{roi_tag}` 由 `METAPHOR_ROI_SET` 决定；主 family 会得到 `meta_metaphor` 与 `meta_spatial` 两套）：

- `$paper/qc/learning_condition_rdm_{roi_tag}/learning_condition_rdm_subject.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/word_level_profile_{roi_tag}.csv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/word_level_profile_group_one_sample.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/word_level_profile_group_pairwise.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/learning_profile_{roi_tag}.csv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/learning_profile_group_one_sample.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/learning_profile_group_pairwise.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_{roi_tag}.csv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_one_sample.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_pairwise.tsv`
- `$paper/qc/learning_condition_rdm_{roi_tag}/learning_condition_rdm_qc.tsv`
- `$paper/tables_si/table_word_level_profile_{roi_tag}.tsv`
- `$paper/tables_si/table_learning_profile_{roi_tag}.tsv`
- `$paper/tables_si/table_four_phase_trajectory_{roi_tag}.tsv`

### P11.2 Secondary: Cross-phase constituent-to-sentence reinstatement

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/cross_phase_reinstatement.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets meta_metaphor meta_spatial `
    --paper-output-root $paper
}
```

默认产物：

- `$paper/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_item.tsv`
- `$paper/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_subject.tsv`
- `$paper/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_group_one_sample.tsv`
- `$paper/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_group_run_contrast.tsv`
- `$paper/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_qc.tsv`
- `$paper/tables_si/table_cross_phase_reinstatement_{roi_tag}.tsv`

### P11.3 Supplementary: Learning within-item reliability (run3 ↔ run4)

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/learning_within_item_reliability.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets meta_metaphor meta_spatial `
    --paper-output-root $paper
}
```

默认产物：

- `$paper/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_item.tsv`
- `$paper/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_subject.tsv`
- `$paper/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_group_one_sample.tsv`
- `$paper/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_group_yy_vs_kj.tsv`
- `$paper/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_qc.tsv`
- `$paper/tables_si/table_learning_within_item_reliability_{roi_tag}.tsv`

### P11.4 Figure: Split significance profiles (主图, 仅 primary family)

依赖：P11.1 在 `$primaryRois` 下产出的：
- `word_level_profile_{roi_tag}.csv` + `word_level_profile_group_one_sample.tsv` + `word_level_profile_group_pairwise.tsv`
- `learning_profile_{roi_tag}.csv` + `learning_profile_group_one_sample.tsv` + `learning_profile_group_pairwise.tsv`

作图口径：
- **不要求折线图**，默认使用更稳妥的“柱状图 + 被试散点 + phase 间显著性括号”。
- `word-level` 图展示 `pre / post / retrieval` 三个 phase，并重点标注 `post vs pre`、`retrieval vs post`、`retrieval vs pre` 是否显著。
- `learning` 图展示 `run3 / run4` 两个 phase，并重点标注 `run4 vs run3` 是否显著。
- 旧的 `four_phase_trajectory_{roi_tag}.csv` 若存在，会作为兼容输入被脚本内部切分；但**不再**作为主文 trajectory 叙述依据。

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/figures/plot_four_phase_trajectory.py `
    --paper-output-root $paper
}
```

默认产物：

- `$paper/figures_main/fig_word_level_profile_{roi_tag}.png`
- `$paper/figures_main/fig_word_level_profile_{roi_tag}.pdf`
- `$paper/figures_main/fig_learning_profile_{roi_tag}.png`
- `$paper/figures_main/fig_learning_profile_{roi_tag}.pdf`

### P11.5 Figure: Reinstatement + within-item reliability (SI 附图, 仅 primary family)

依赖：P11.2 与 P11.3 在 `$primaryRois` 下的 subject 与 group 产物。

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/figures/plot_learning_reinstatement.py `
    --paper-output-root $paper
}
```

默认产物：

- `$paper/figures_si/fig_learning_reinstatement_{roi_tag}.png`
- `$paper/figures_si/fig_learning_reinstatement_{roi_tag}.pdf`

### P11.6 Literature sensitivity (SI 表, 不进主图)

仅在需要回答"换成旧文献/atlas ROI 是否同向"时运行；**不生成主图**、**不与 primary 放在同一 FDR family**。

```powershell
foreach ($roi in $literatureSensitivityRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/learning_condition_rdm.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets literature literature_spatial `
    --paper-output-root $paper

  & $py metaphoric/final_version/rsa_analysis/cross_phase_reinstatement.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets literature literature_spatial `
    --paper-output-root $paper

  & $py metaphoric/final_version/rsa_analysis/learning_within_item_reliability.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets literature literature_spatial `
    --paper-output-root $paper
}
```

产物落在同样的 `$paper/qc/<analysis>_{roi_tag}/` 和 `$paper/tables_si/` 下，`{roi_tag}` 为 `literature` / `literature_spatial`。

### P11.7 Main-functional GLM reference (仅表, 不进主 family)

仅用于报告"本数据学习阶段 GLM ROI 是否参考性支持"。建议只跑 Primary condition-level RDM 即可，不跑图脚本。

```powershell
foreach ($roi in $glmReferenceRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/rsa_analysis/learning_condition_rdm.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets main_functional `
    --paper-output-root $paper
}
```

产物 `{roi_tag} = main_functional`，写入 `$paper/qc/learning_condition_rdm_main_functional/` 与 `$paper/tables_si/table_word_level_profile_main_functional.tsv` / `$paper/tables_si/table_learning_profile_main_functional.tsv` / `$paper/tables_si/table_four_phase_trajectory_main_functional.tsv`；**不得**进入主文 FDR family 或主图。

### 依赖关系一览

- P11.1 是 P11.4 的前置。
- P11.2 与 P11.3 互不依赖，可与 P11.1 并行。
- P11.5 依赖 P11.2 + P11.3。
- P11.4 主文图现在拆成两个模块：`word-level profile` 与 `learning sentence-level profile`；不再输出单张五点 trajectory 主图。
- P11.6 / P11.7 只产表，不生成 `figures_main/`；**不得**与 `$primaryRois` 合并到同一 FDR family。
- 本章节所有脚本**不会**修改 `stack_patterns.py` 与 LSS 产物，也**不会**改写 pre/post/retrieval 已有 RSA primary 结果。

## P12 Metaphor × Learning × Memory 桥接（extend-metaphor-learning-memory-bridge）

> **重要**：本机仅记录命令，不在本机执行。以下脚本必须在已完成 P1–P11 的机器上运行。`$py / $paper / $patternRoot / $manifest / $primaryRois / $literatureSensitivityRois / $glmReferenceRois` 沿用第 0 节定义。

本章节的主线：**隐喻 × 学习 × 记忆**。在 primary ROI family (`meta_metaphor + meta_spatial`) 维持不变的前提下：
- **A1**：加入 Language / ToM network 作为 SI sensitivity（不进主 FDR family）。
- **B1/B2**：**必做** — item-level ERS 与 hippocampal-binding subfamily 汇总。
- **C1/C2**：**gated** — 只有当上游 gate 通过才跑。

### P12.A1.1 构建 Language / ToM network 新 ROI mask

先用既有 `build_meta_roi_library.py` 把 Language / ToM network 的 peak TSV 编译成 6mm 球 mask，并更新 `manifest.tsv`。

```powershell
& $py metaphoric/final_version/roi_management/build_meta_roi_library.py `
  --peaks metaphoric/final_version/roi_management/language_network_peaks.tsv `
          metaphoric/final_version/roi_management/tom_network_peaks.tsv `
  --roi-library-root "$env:PYTHON_METAPHOR_ROOT/roi_library" `
  --reference-image "$patternRoot/sub-01/pre_yy.nii.gz" `
  --append-manifest
```

> 如果 `build_meta_roi_library.py` 的 CLI 与上述有差异，按实际参数调整；目标等价：为 `roi_set ∈ {language_network, tom_network}` 的每条 peak 生成 6mm 球 NIfTI，并追加到 `$manifest` 中（`include_in_rsa=true`）。

### P12.A1.2 Overlap QC（Language/ToM × Primary meta ROI）

```powershell
& $py metaphoric/final_version/roi_management/network_overlap_qc.py `
  --primary-peaks metaphoric/final_version/roi_management/meta_roi_peaks_curated.tsv `
  --network-peaks metaphoric/final_version/roi_management/language_network_peaks.tsv `
                  metaphoric/final_version/roi_management/tom_network_peaks.tsv `
  --reference-image "$patternRoot/sub-01/pre_yy.nii.gz" `
  --output "$paper/qc/roi_library/network_overlap_qc.tsv"
```

输出 `$paper/qc/roi_library/network_overlap_qc.tsv` + `.json`，含每对 peak 的 Euclidean 距离与 6mm 球 IoU。

### P12.A1.3 在 Language / ToM ROI 上重跑 4 套主 RSA 脚本

ROI 换了，RSA 必须重算。对 `{language_network, tom_network}` 各自驱动 `METAPHOR_ROI_SET` 跑 4 个主脚本：

```powershell
$networkRois = @("language_network", "tom_network")

foreach ($roi in $networkRois) {
  $env:METAPHOR_ROI_SET = $roi

  & $py metaphoric/final_version/rsa_analysis/step5c_rsa.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets $roi `
    --paper-output-root $paper

  & $py metaphoric/final_version/rsa_analysis/edge_specificity_rsa.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets $roi `
    --paper-output-root $paper

  & $py metaphoric/final_version/rsa_analysis/learning_condition_rdm.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets $roi `
    --paper-output-root $paper

  & $py metaphoric/final_version/rsa_analysis/retrieval_pair_similarity.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets $roi `
    --paper-output-root $paper
}
```

产物落在 `$paper/qc/<analysis>_{roi_tag}/`，`{roi_tag} ∈ {language_network, tom_network}`；**不进入**主 FDR family。

### P12.A1.4 Language / ToM sensitivity 汇总（仅汇总，不重算 RSA）

```powershell
& $py metaphoric/final_version/rsa_analysis/language_tom_network_sensitivity.py `
  --paper-output-root $paper `
  --roi-sets language_network tom_network
```

产物：
- `$paper/qc/language_tom_network_sensitivity_{roi_tag}/step5c_summary.tsv / edge_summary.tsv / word_level_profile_summary.tsv / learning_profile_summary.tsv / trajectory_compatibility_summary.tsv / retrieval_summary.tsv / missing_sources.tsv / sensitivity_manifest.json`
- `$paper/tables_si/table_language_tom_network_sensitivity_{roi_tag}.tsv`

所有行带 `family="SI_sensitivity"`、`in_main_fdr_family=false`。

### P12.B1 Encoding-Retrieval Similarity（必做，primary ROI family）

```powershell
foreach ($roi in $primaryRois) {
  $env:METAPHOR_ROI_SET = $roi
  & $py metaphoric/final_version/brain_behavior/encoding_retrieval_similarity.py `
    --pattern-root $patternRoot `
    --roi-manifest $manifest `
    --roi-sets meta_metaphor meta_spatial `
    --paper-output-root $paper
}
```

产物：
- `$paper/qc/encoding_retrieval_similarity_{roi_tag}/ers_item.tsv`
- `$paper/qc/encoding_retrieval_similarity_{roi_tag}/ers_group_one_sample.tsv`（BH-FDR within ROI set × variant × analysis_type）
- `$paper/qc/encoding_retrieval_similarity_{roi_tag}/ers_itemlevel_gee.tsv`（GEE: memory/log_rt_correct ~ ERS × condition）
- `$paper/qc/encoding_retrieval_similarity_{roi_tag}/ers_qc.tsv / ers_failures.tsv / ers_manifest.json`
- `$paper/tables_main/table_encoding_retrieval_similarity_{roi_tag}.tsv`

Variants：`run3_to_retrieval`、`run4_to_retrieval`、`max_run_to_retrieval`。该表同时是 C1 的 Gate G1 依据。

### P12.B2 Hippocampal-binding subfamily 汇总（必做，仅汇总）

```powershell
& $py metaphoric/final_version/rsa_analysis/hippocampal_binding_summary.py `
  --roi-manifest $manifest `
  --roi-sets meta_metaphor meta_spatial `
  --paper-output-root $paper
```

产物：
- `$paper/qc/hippocampal_binding_summary_{roi_tag}/subfamily_rois.tsv`
- `$paper/qc/hippocampal_binding_summary_{roi_tag}/{step5c,edge,word_level_profile,learning_profile,trajectory_compatibility,reinstatement,reliability,retrieval,ers}_subfamily_fdr.tsv`（subfamily 内重做 BH-FDR）
- `$paper/qc/hippocampal_binding_summary_meta_hipp_subfamily/combined_subfamily_fdr.tsv`
- `$paper/tables_main/table_hippocampal_binding_summary_meta_hipp_subfamily.tsv`

subfamily 定义：`theory_role ∈ {hippocampal_binding, scene_context_binding}` 或 ROI 名命中 `(?i)hippocamp|PPA|PHG`。
依赖提醒：B2 依赖 Step5C + edge + retrieval + 本轮 split profiles + reinstatement + reliability + ERS（若已有）结果文件都已存在；脚本本身不重算任何 RSA。

### P12.C1 Memory-strength parametric RSA（gated by G1）

**Gate G1**：`ers_group_one_sample.tsv` 中存在至少一行 `analysis_type=ers_one_sample_gt_zero`、`variant ∈ {run3_to_retrieval, run4_to_retrieval}`、`roi_set ∈ {meta_metaphor, meta_spatial}`、`q_bh_within_family < 0.1`。若未通过，脚本只写 `gate_status.tsv` 并退出；若要强制跑，加 `--force-run`（不建议）。

```powershell
& $py metaphoric/final_version/brain_behavior/memory_strength_parametric.py `
  --paper-output-root $paper `
  --roi-sets meta_metaphor meta_spatial
```

产物（gate 通过时）：
- `$paper/qc/memory_strength_parametric_meta_mem_strength/gate_status.tsv`
- `$paper/qc/memory_strength_parametric_meta_mem_strength/memory_weighted_ers_group.tsv`（subject-level Δ = remembered − forgot 的组水平 one-sample t + BH-FDR）
- `$paper/qc/memory_strength_parametric_meta_mem_strength/memory_strength_parametric.tsv`（item-level MixedLM：memory / log_rt_correct ~ ERS_z × condition + 1|subject）
- `$paper/tables_main/table_memory_strength_parametric_meta_mem_strength.tsv`

### P12.C2 HPC ↔ vmPFC post-encoding connectivity（gated by G2）

**Gate G2**：`$paper/qc/gppi_results_*/gppi_group_summary.tsv` 中存在 HPC seed（`(?i)hippocamp|HPC`）→ vmPFC target（`(?i)vmPFC|ventromedial|vMPFC|MMPFC`）的行 `p < 0.1`。若未通过，脚本只写 `gate_status.tsv` 并退出；若要强制跑，加 `--force-run`（不建议）。

脚本需要本机的 BOLD 清单（pre/post-encoding TR 窗口）与 HPC / vmPFC mask：

```powershell
& $py metaphoric/final_version/connectivity_analysis/hpc_vmpfc_post_encoding_ppi.py `
  --paper-output-root $paper `
  --bold-manifest "$env:PYTHON_METAPHOR_ROOT/qc/post_encoding_bold_manifest.tsv" `
  --hpc-masks    "$env:PYTHON_METAPHOR_ROOT/roi_library/meta_L_hippocampus.nii.gz" `
                 "$env:PYTHON_METAPHOR_ROOT/roi_library/meta_R_hippocampus.nii.gz" `
  --vmpfc-masks  "$env:PYTHON_METAPHOR_ROOT/roi_library/tom_VMPFC.nii.gz" `
                 "$env:PYTHON_METAPHOR_ROOT/roi_library/tom_MMPFC.nii.gz"
```

`--bold-manifest` 必须含列：`subject, run_label, bold_path, confounds_path, pre_encoding_tr_start, pre_encoding_tr_end, post_encoding_tr_start, post_encoding_tr_end, tr`；本机上请先整理一份。

产物（gate 通过 + 参数齐全时）：
- `$paper/qc/hpc_vmpfc_post_encoding_ppi_meta_hpc_vmpfc_ppi/gate_status.tsv`
- `$paper/qc/hpc_vmpfc_post_encoding_ppi_meta_hpc_vmpfc_ppi/hpc_vmpfc_subject_delta.tsv`
- `$paper/qc/hpc_vmpfc_post_encoding_ppi_meta_hpc_vmpfc_ppi/hpc_vmpfc_group_delta.tsv`（one-sample t on Δz = z_post − z_pre + BH-FDR within seed × target）
- `$paper/tables_main/table_hpc_vmpfc_post_encoding_ppi_meta_hpc_vmpfc_ppi.tsv`

### P12 依赖关系一览

- P12.A1.1 → P12.A1.2、P12.A1.3 → P12.A1.4（A1 内部严格串行：先建 mask 并更新 manifest，再重跑 RSA，最后才能汇总）。
- P12.B1 与 P12.B2 可并行（均只读 primary ROI 主 RSA 产物 + pattern_root）。
- P12.C1 依赖 P12.B1 产出的 `ers_group_one_sample.tsv`（Gate G1）。
- P12.C2 依赖已有的 `$paper/qc/gppi_results_*/gppi_group_summary.tsv`（Gate G2），与 B1/B2 互相独立。
- A1 的 SI sensitivity 与 B1/B2/C1/C2 之间**不共享 BH-FDR family**；论文 writeup 里分两处报告。
- 所有 P12 脚本**不修改** `stack_patterns.py` / LSS 产物 / P1–P11 已有 RSA 结果；仅新增 `$paper/qc/<新目录>/` 与 `$paper/tables_main/tables_si/` 下的新表。

