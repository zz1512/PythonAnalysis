# 最短重跑顺序（rerun_list）

最后更新：2026-04-28

本清单按**真实依赖顺序**整理，只保留这次本地代码修改后必须或建议重跑的最短链条。

约定：
- `BASE_DIR` = `rsa_config.BASE_DIR` = `${PYTHON_METAPHOR_ROOT}`
- 默认目标：所有产物都落到 `paper_outputs/`
- 若旧目录仍有历史结果，不建议混用；优先重跑到新默认路径

## 最短顺序总览

从上到下依次执行：

1. `behavior_refined.py`
2. `run_rsa_optimized.py`
3. `build_memory_strength_table.py`
4. `model_rdm_comparison.py`
5. `delta_rho_lmm.py`
6. `itemlevel_brain_behavior.py`
7. `stack_patterns.py`（如果学习期 pattern 尚未存在）
8. `learning_dynamics.py`
9. `representational_connectivity.py`
10. `rd_searchlight.py`
11. `robustness_suite.py`

其中：
- **1-6** 是这次代码修复后的主链，优先级最高
- **7-9** 是 B 层新增分析链
- **10** 是 B3 的“where”补充链
- **11** 是 B4 的稳健性三件套
- `rsa_lmm.py`、`trial_voxel_robustness.py` 仅在你需要同步 QC / LMM 表时再跑

---

## 1) Behavior

原因：
- A5 后 behavior 默认输出目录已迁移到 `paper_outputs/qc/behavior_results/`
- `itemlevel_brain_behavior.py` 默认输入现在也依赖这条新路径

运行：

```bash
python metaphoric/final_version/behavior_analysis/behavior_refined.py
```

产物：
- `${BASE_DIR}/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv`
- `${BASE_DIR}/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv`

---

## 2) Step 5C RSA

原因：
- A5 后 RSA 默认输出目录已迁移到 `paper_outputs/qc/rsa_results_optimized_{ROI_SET}/`
- 下游脚本依赖：
  - `itemlevel_brain_behavior.py` 需要 `main_functional` + `literature`
  - `representational_connectivity.py` 需要 `main_functional` + `literature` + `literature_spatial`

运行：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
```

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
```

```bash
export METAPHOR_ROI_SET=literature_spatial
python metaphoric/final_version/rsa_analysis/run_rsa_optimized.py
```

默认关键产物：
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv`

可选：
- 如果你要同步主文里的 RSA LMM 参数表，再跑：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/rsa_lmm.py
```

- 如果你要同步 trial/voxel 平衡性与漂移 QC，再跑：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/trial_voxel_robustness.py
```

---

## 3) M7 记忆强度表

原因：
- `continuous_confidence` 的 RT 归一化逻辑已改为“只在 correct trial 内归一化”

运行：

```bash
python metaphoric/final_version/rsa_analysis/build_memory_strength_table.py --score-modes binary continuous_confidence
```

产物：
- `${BASE_DIR}/paper_outputs/qc/memory_strength/memory_strength_binary_sub-XX.tsv`
- `${BASE_DIR}/paper_outputs/qc/memory_strength/memory_strength_continuous_confidence_sub-XX.tsv`
- `${BASE_DIR}/paper_outputs/qc/memory_strength/memory_strength_manifest.tsv`

---

## 4) Model-RSA

原因：
- M7 输入变化会改写 `M7_continuous_confidence` 结果与所有下游汇总

运行：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/model_rdm_comparison.py --embedding-file <你的stimulus_embeddings_bert.tsv路径> --memory-strength-dir ${BASE_DIR}/paper_outputs/qc/memory_strength
```

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/rsa_analysis/model_rdm_comparison.py --embedding-file <你的stimulus_embeddings_bert.tsv路径> --memory-strength-dir ${BASE_DIR}/paper_outputs/qc/memory_strength
```

默认关键产物：
- `${BASE_DIR}/paper_outputs/qc/model_rdm_results_main_functional/model_rdm_subject_metrics.tsv`
- `${BASE_DIR}/paper_outputs/qc/model_rdm_results_literature/model_rdm_subject_metrics.tsv`

---

## 5) Δρ LMM

原因：
- 参数表补齐了 `SE/z/p/CI`
- 单 ROI split 会自动 fallback
- Model-RSA 指标表已更新

运行：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/delta_rho_lmm.py --family-split
```

如果你也报告 `literature`：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/rsa_analysis/delta_rho_lmm.py --family-split
```

可选稳健性版：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/rsa_analysis/delta_rho_lmm.py --family-split --condition-col condition_group
```

产物：
- `${BASE_DIR}/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_lmm_params.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/delta_rho_lmm/main_functional/delta_rho_family_split_params.tsv`

---

## 6) item-level 脑-行为

原因：
- join 键已从 `subject+condition+pair_id` 改为 `+word_label`
- 新增 RT 主图
- 默认输入路径已切到 `paper_outputs/qc/...`

运行：

```bash
python metaphoric/final_version/brain_behavior/itemlevel_brain_behavior.py
```

产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_itemlevel_coupling.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_itemlevel_coupling.png`
- `${BASE_DIR}/paper_outputs/figures_main/fig_itemlevel_coupling_rt.png`
- `${BASE_DIR}/paper_outputs/qc/itemlevel_behavior_neural_long.tsv`

---

## 7) 学习期 pattern（仅 B1 需要）

先检查以下文件是否已存在：
- `${BASE_DIR}/pattern_root/sub-XX/learn_yy.nii.gz`
- `${BASE_DIR}/pattern_root/sub-XX/learn_kj.nii.gz`
- `${BASE_DIR}/pattern_root/sub-XX/learn_yy_metadata.tsv`
- `${BASE_DIR}/pattern_root/sub-XX/learn_kj_metadata.tsv`

如果已经存在，直接跳到第 8 步。

如果不存在，运行：

```bash
python metaphoric/final_version/representation_analysis/stack_patterns.py
```

---

## 8) B1 学习阶段 repeated-exposure

原因：
- B1 已重定义为：
  - 主分析 A2：`pair_discrimination = same_pair_cross_run_similarity - different_pair_cross_run_similarity`
  - 补充分析 A1：`same_sentence_cross_run_similarity`

主分析（`literature`）：

```bash
python metaphoric/final_version/temporal_dynamics/learning_dynamics.py \
  ${BASE_DIR}/pattern_root \
  --paper-output-root ${BASE_DIR}/paper_outputs \
  --roi-set literature
```

补充分析（`main_functional` family split）：

```bash
python metaphoric/final_version/temporal_dynamics/learning_dynamics.py \
  ${BASE_DIR}/pattern_root \
  --paper-output-root ${BASE_DIR}/paper_outputs \
  --roi-set main_functional
```

产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_learning_repeated_exposure.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_learning_repeated_exposure_subject.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_learning_repeated_exposure_family.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_learning_dynamics.png`

---

## 9) B2 representational connectivity

原因：
- B2 已重构为 **all-by-all ROI pair representational connectivity**
- 主分析：`main_functional`（family-aware）
- 确认性分析：`literature + literature_spatial`

运行：

```bash
python metaphoric/final_version/representation_analysis/representational_connectivity.py
```

默认会自动读取：
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv`

产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_repr_connectivity.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_repr_connectivity_subject.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_repr_connectivity_edges.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_repr_connectivity.png`
- `${BASE_DIR}/paper_outputs/qc/repr_connectivity_summary_group.tsv`
- `${BASE_DIR}/paper_outputs/qc/repr_connectivity_edge_group.tsv`

---

## 10) B3 Searchlight RD

原因：
- B3 现已补齐：
  - `yy / kj / baseline`
  - group-level sign-flip permutation
  - `table_searchlight_peaks.tsv`
  - `fig_searchlight_delta.png`
- 默认输出已统一到 `paper_outputs/`

运行：

```bash
python metaphoric/final_version/representation_analysis/rd_searchlight.py \
  --permutation-backend auto
```

默认会读取：
- `${BASE_DIR}/pattern_root/sub-XX/{pre,post}_{yy,kj,baseline}.nii.gz`
- `${BASE_DIR}/lss_betas_final/sub-XX/mask.nii.gz`（若 `mask.nii` 不存在会自动尝试 `.nii.gz`）

默认产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_searchlight_peaks.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_searchlight_delta.png`
- `${BASE_DIR}/paper_outputs/qc/rd_searchlight/group_yy_post_vs_pre/...`
- `${BASE_DIR}/paper_outputs/qc/rd_searchlight/group_kj_post_vs_pre/...`
- `${BASE_DIR}/paper_outputs/qc/rd_searchlight/group_baseline_post_vs_pre/...`

---

## 11) B4 稳健性三件套

原因：
- B4 已统一到一个脚本里，围绕 Step 5C 的 subject-level interaction delta 输出：
  - Bayes Factor
  - bootstrap CI
  - LOSO
  - split-half replication

运行：

```bash
python metaphoric/final_version/rsa_analysis/robustness_suite.py
```

默认会读取：
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv`

默认产物：
- `${BASE_DIR}/paper_outputs/tables_si/table_bayes_factors.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_bootstrap_ci.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_loso.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_splithalf.tsv`
- `${BASE_DIR}/paper_outputs/qc/robustness_subject_interaction.tsv`
