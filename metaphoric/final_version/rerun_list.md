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
6. `mechanism_behavior_prediction.py`
7. `itemlevel_brain_behavior.py`
8. `stack_patterns.py`（如果学习期 pattern 尚未存在）
9. `learning_dynamics.py`
10. `representational_connectivity.py`
11. `export_gppi_summary.py`
12. `pair_similarity_searchlight.py`
13. `rd_searchlight.py`（仅补充探索版）
14. `robustness_suite.py`
15. `noise_ceiling.py`

其中：
- **1-7** 是这次代码修复后的主链，优先级最高
- **8-10** 是 B 层新增分析链
- **11** 是 S2 的轻量导出层
- **12-13** 是 S3 的“where”补充链（主版 + 探索版）
- **14** 是 B4 的稳健性三件套
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

出图：

```bash
python metaphoric/final_version/figures/plot_behavior_results.py
```

产物：
- `${BASE_DIR}/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv`
- `${BASE_DIR}/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_behavior_accuracy_rt.png`

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

Step 5C 主图：

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/figures/plot_step5c_rsa_main.py
```

四层 ROI robustness 图：

```bash
python metaphoric/final_version/figures/plot_roi_set_robustness.py
```

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

机制热图：

```bash
python metaphoric/final_version/figures/plot_model_rsa_mechanism.py \
  --condition-group all
```

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

## 7) S1 机制 → 行为 / 记忆预测

原因：
- S1 主分析现已固定为**被试级机制分数预测 run-7 最终学习收益**
- 默认主模型只保留：`M3_embedding + M8_reverse_pair`
- `M7_continuous_confidence` 仅在你需要补充局部记忆强度结果时，才通过 `--include-m7` 加入
- 已明确移除：
  - `M2_pair`（避免与 `M8_reverse_pair` 的镜像编码重复记账）
  - `run3/4 learning accuracy`（learning-phase 在线表现，不作为 S1 主终点）

默认主分析运行：

```bash
python metaphoric/final_version/brain_behavior/mechanism_behavior_prediction.py
```

如需把 `M7_continuous_confidence` 作为补充项一起跑：

```bash
python metaphoric/final_version/brain_behavior/mechanism_behavior_prediction.py \
  --include-m7
```

默认关键产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_mechanism_behavior_prediction.png`
- `${BASE_DIR}/paper_outputs/qc/mechanism_behavior_prediction_long.tsv`

---

## 8) 学习期 pattern（仅 B1 需要）

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

## 9) B1 学习阶段 repeated-exposure

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
- `${BASE_DIR}/paper_outputs/tables_main/table_learning_repeated_exposure_<roi_set>.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_learning_repeated_exposure_subject_<roi_set>.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_learning_repeated_exposure_family_<roi_set>.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_learning_dynamics_<roi_set>.png`

注意：该脚本会按 `--roi-set` 自动加后缀，避免 `literature` 与 `main_functional` 两次运行互相覆盖。
例如：
- `${BASE_DIR}/paper_outputs/tables_main/table_learning_repeated_exposure_literature.tsv`
- `${BASE_DIR}/paper_outputs/tables_main/table_learning_repeated_exposure_main_functional.tsv`

统一重画主图：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/figures/plot_learning_dynamics_final.py
```

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/figures/plot_learning_dynamics_final.py
```

---

## 10) B2 representational connectivity

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

统一重画主图：

```bash
python metaphoric/final_version/figures/plot_repr_connectivity_final.py
```

---

## 11) S2 gPPI 导出汇总

原因：
- `gPPI_analysis.py` 已负责重计算与 seed 级 SI 表输出
- 这个脚本只做**不影响现有运行**的轻量收口，把现有 seed 级表整合成论文正式总表

运行：

```bash
python metaphoric/final_version/connectivity_analysis/export_gppi_summary.py
```

默认产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_gppi_summary_main.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_gppi_summary_full.tsv`
- `${BASE_DIR}/paper_outputs/qc/gppi_summary_export_manifest.json`

---

## 12) S3 Searchlight 主版：Pair Similarity

原因：
- 这是当前 `todo` 里正式的 S3 主版
- 它和 Step 5C 主结果同构，直接回答“哪里出现了 pair-level representational reorganization”
- 默认产物沿用主文路径：
  - `table_searchlight_pair_similarity_peaks.tsv`
  - `fig_searchlight_pair_similarity_delta.png`

运行：

```bash
python metaphoric/final_version/representation_analysis/pair_similarity_searchlight.py \
  --permutation-backend auto
```

时间紧张时，只跑 `yy` 最小版：

```bash
python metaphoric/final_version/representation_analysis/pair_similarity_searchlight.py \
  --conditions yy \
  --permutation-backend auto
```

默认会读取：
- `${BASE_DIR}/pattern_root/sub-XX/{pre,post}_{yy,kj,baseline}.nii.gz`
- `${BASE_DIR}/pattern_root/sub-XX/{pre,post}_{yy,kj,baseline}_metadata.tsv`
- `${BASE_DIR}/lss_betas_final/sub-XX/mask.nii.gz`（若 `mask.nii` 不存在会自动尝试 `.nii.gz`）

默认产物：
- `${BASE_DIR}/paper_outputs/tables_main/table_searchlight_pair_similarity_peaks.tsv`
- `${BASE_DIR}/paper_outputs/figures_main/fig_searchlight_pair_similarity_delta.png`
- `${BASE_DIR}/paper_outputs/qc/pair_similarity_searchlight/group_yy_post_vs_pre/...`
- `${BASE_DIR}/paper_outputs/qc/pair_similarity_searchlight/group_kj_post_vs_pre/...`
- `${BASE_DIR}/paper_outputs/qc/pair_similarity_searchlight/group_baseline_post_vs_pre/...`

---

## 13) S3 补充探索版：RD Searchlight

原因：
- RD searchlight 回答的是局部表征维度变化，不是 pair-level relation change
- 适合写成 supplementary / exploratory 几何描述，不建议与 S3 主版并列主打

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

## 14) B4 稳健性三件套

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

---

## 15) B5 Noise ceiling

原因：
- 为 Step 5C 与 Model-RSA 提供可汇报的上限参考（supporting information）
- 采用 LOSO lower/upper：`corr(subject, mean_others)` / `corr(subject, mean_all)`

运行：

```bash
python metaphoric/final_version/rsa_analysis/noise_ceiling.py
```

默认会读取：
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/rsa_results_optimized_atlas_robustness/rsa_itemwise_details.csv`
- `${BASE_DIR}/paper_outputs/qc/model_rdm_results_*/model_rdm_audit/*_rdms.npz`（若存在）

实现说明：
- Step 5C noise ceiling 默认覆盖四层 ROI
- Model-RSA noise ceiling 会使用 audit 中的 `metadata + pair_i/pair_j` 重建共享 item-pair 特征后再做跨被试对齐，不再直接假设各被试 `neural_rdm` 向量索引天然一致

默认产物：
- `${BASE_DIR}/paper_outputs/tables_si/table_noise_ceiling.tsv`
- `${BASE_DIR}/paper_outputs/qc/noise_ceiling_step5c_subject.tsv`
- `${BASE_DIR}/paper_outputs/qc/noise_ceiling_model_rsa_subject.tsv`
- `${BASE_DIR}/paper_outputs/qc/noise_ceiling_meta.json`

出图（SI）：

```bash
python metaphoric/final_version/figures/plot_noise_ceiling.py \
  --model-condition all
```

---

## C7) ROI-level RD / GPS（推荐先做）

原因：
- 用几何指标补充 Step 5C 的解释边界（更像压缩/分化/重组哪种）
- 成本低、可作为 SI/扩展模块

RD（每个 ROI、每个 subject 的表征维度）：
- 默认口径已改为 `participation_ratio`：基于 voxel covariance eigenspectrum 的软维度
- `covariance` 与旧 `rdm` 仅建议作为敏感性分析，不再作为主口径

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/representation_analysis/rd_analysis.py \
  --rd-mode participation_ratio
```

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/representation_analysis/rd_analysis.py \
  --rd-mode participation_ratio
```

敏感性分析示例：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/representation_analysis/rd_analysis.py \
  --rd-mode covariance
```

默认产物（按 ROI_SET 自动加后缀，避免覆盖）：
- `${BASE_DIR}/paper_outputs/tables_si/table_rd_<roi_set>.tsv`
- `${BASE_DIR}/paper_outputs/qc/rd_results_<roi_set>/rd_group_summary.tsv`
- `${BASE_DIR}/paper_outputs/qc/rd_results_<roi_set>/rd_subject_metrics.tsv`

GPS（同条件 trials 的全局紧凑度）：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/representation_analysis/gps_analysis.py
```

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/representation_analysis/gps_analysis.py
```

默认产物：
- `${BASE_DIR}/paper_outputs/tables_si/table_gps_<roi_set>.tsv`
- `${BASE_DIR}/paper_outputs/qc/gps_results_<roi_set>/gps_group_summary.tsv`
- `${BASE_DIR}/paper_outputs/qc/gps_results_<roi_set>/gps_subject_metrics.tsv`

统一重画 RD / GPS 图（建议在同一 `roi_set` 的 RD 与 GPS 都跑完后执行）：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/figures/plot_rd_gps_summary.py
```

```bash
export METAPHOR_ROI_SET=main_functional
python metaphoric/final_version/figures/plot_rd_gps_summary.py
```

---

## C3) gPPI（pilot，少 seed + 受控 targets）

原因：
- 补网络层机制证据（**pre/post：run1/2 vs run5/6** 的连接调制变化），对齐主线 Step 5C 的 `condition × time`
- 建议先做 pilot，再决定是否扩展到多 seed / 全连接（多重比较负担）

运行示例（推荐：自动从 `roi_library/manifest.tsv` 取 mask；脚本默认 `--stage prepost`，自动用 runs 1/2/5/6）：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/connectivity_analysis/gPPI_analysis.py \
  --seed-roi-names \
    func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort \
    func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort \
  --target-roi-sets literature literature_spatial \
  --stage prepost \
  --conditions yy kj \
  --deconvolution ridge
```

说明：
- `seed-roi-names` 这两个就是你 learning(run3/4) 的 `Metaphor > Spatial` 左颞极两个显著簇 ROI
- 这两个 seed 负责提供 learning-phase 的 where 定位；gPPI 本体仍在 pre/post（run1/2 vs run5/6）上检验连接调制变化
- `target-roi-sets` 同时包含 `literature` 与 `literature_spatial`，脚本会在汇总里输出：
  - `within_literature`
  - `within_literature_spatial`
  - `cross_sets`（定义：within_literature 的 post-pre 变化减去 within_literature_spatial 的 post-pre 变化）
- 若一次运行多个 seed 且手动传入 `output_dir`，脚本会自动按 `seed_tag/` 分子目录，避免结果互相覆盖

默认产物：
- `${BASE_DIR}/paper_outputs/qc/gppi_results_<seed>_<roi_set>/gppi_subject_metrics.tsv`
- `${BASE_DIR}/paper_outputs/qc/gppi_results_<seed>_<roi_set>/gppi_group_summary.tsv`
- `${BASE_DIR}/paper_outputs/tables_si/table_gppi_summary_<seed>_<roi_set>.tsv`
 - `${BASE_DIR}/paper_outputs/qc/gppi_combined_<roi_set>/gppi_subject_metrics_all_seeds.tsv`（合并版，便于二次汇总）

推荐写法（更接近标准 gPPI / SPM 路线）：

```bash
export METAPHOR_ROI_SET=literature
python metaphoric/final_version/connectivity_analysis/gPPI_analysis.py \
  <seed_mask.nii.gz> \
  <target_roi_dir> \
  --conditions yy kj \
  --stage prepost \
  --deconvolution ridge
```
