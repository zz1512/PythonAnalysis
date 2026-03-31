# emo_final

`emo_final` 目前包含三条彼此衔接、但可以独立运行的分析分支：

1. 脑和年龄模型分析
2. 脑和行为分析
3. 脑和年龄调制的行为分析

整体目录约定：

- `by_stimulus/<stimulus_type>/`：trial-level 结果
- `by_emotion/<stimulus_type>/`：emotion-level 结果
- `behavior/`：行为分支脚本与配置
- `data_check/`：校验脚本
- `figures/`：热图、脑图、轨迹图等可视化结果

常用结果根目录示例：

```bash
/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```

---

## 一、脑和年龄模型分析

这一部分是原始 `emo_final` 的主干流程：先构造脑表征矩阵，再计算脑的被试 × 被试相似矩阵，最后和三种年龄模型做关联与置换检验。

### 1. 主要脚本

- `lss_main.py`：可选，生成 LSS 单 trial beta 和 aligned index
- `build_roi_repr_matrix.py`：构造 trial-level ROI 表征矩阵
- `build_roi_emotion_repr_matrix.py`：从 trial-level pattern 聚合到 emotion-level ROI 表征矩阵
- `calc_roi_isc_by_age.py`：计算脑的被试 × 被试相似矩阵，并按年龄排序
- `joint_analysis_roi_isc_dev_models.py`：将脑 ISC 与三种年龄模型做关联和置换检验
- `plot_roi_isc_dev_models_perm_sig.py`：绘制显著 ROI 热图
- `plot_brain_surface_vol.py`：把显著 ROI 映射回脑表面/皮下
- `plot_roi_isc_age_trajectory.py`：画被试对平均年龄 vs ROI ISC 的轨迹图

### 2. 推荐配置

- `config_trial.json`：trial-level 脑和年龄模型分析
- `config_emotion.json`：emotion-level 脑和年龄模型分析

一键运行：

```bash
python run_pipeline.py --config config_trial.json
```

```bash
python run_pipeline.py --config config_emotion.json
```

### 3. 核心步骤

#### Step 0，可选：LSS

```bash
python lss_main.py --write-mode sync
```

关键点：

- 会生成 `lss_index_*_aligned.csv`
- `stimulus_content` 约定为 `Task_Condition_Emotion`
- 后续脑和行为分析都依赖这套对齐键

#### Step 1：trial-level ROI 表征矩阵

```bash
python build_roi_repr_matrix.py \
  --lss-root /public/home/dingrui/fmri_analysis/zz_analysis/lss_results \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --roi-set 232 \
  --rsm-method spearman \
  --threshold-ratio 0.8
```

主要输出：

- `roi_repr_matrix_232.npz`
- `roi_repr_matrix_232_subjects.csv`
- `roi_repr_matrix_232_rois.csv`
- `roi_repr_matrix_232_meta.csv`
- `stimulus_order.csv`
- `roi_beta_patterns_232.npz`

#### Step 1.5：emotion-level ROI 表征矩阵

```bash
python build_roi_emotion_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --roi-set 232 \
  --pattern-prefix roi_beta_patterns_232 \
  --out-stimulus-dir-name by_emotion \
  --rsm-method spearman
```

主要输出：

- `roi_repr_matrix_232_emotion4.npz`
- `roi_repr_matrix_232_emotion4_subjects.csv`
- `roi_repr_matrix_232_emotion4_rois.csv`
- `roi_repr_matrix_232_emotion4_meta.csv`
- `emotion_order.csv`
- `stimulus_to_emotion.csv`

#### Step 2：脑 ISC

trial-level：

```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix roi_repr_matrix_232 \
  --isc-method mahalanobis
```

emotion-level：

```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix roi_repr_matrix_232_emotion4 \
  --isc-method mahalanobis
```

主要输出：

- `roi_isc_mahalanobis_by_age.npy`
- `roi_isc_mahalanobis_by_age_subjects_sorted.csv`
- `roi_isc_mahalanobis_by_age_rois.csv`
- `roi_isc_mahalanobis_by_age_meta.csv`

#### Step 3：脑 ISC 和年龄模型关联

trial-level：

```bash
python joint_analysis_roi_isc_dev_models.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --isc-method mahalanobis \
  --assoc-method spearman \
  --correction-mode fdr_only \
  --n-perm 5000 \
  --seed 42
```

emotion-level：

```bash
python joint_analysis_roi_isc_dev_models.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --isc-method mahalanobis \
  --assoc-method spearman \
  --correction-mode fdr_only \
  --n-perm 5000 \
  --seed 42
```

主要输出：

- `roi_isc_dev_models_perm_fwer.csv`
- `roi_isc_dev_models_perm_fwer_subjects_sorted.csv`
- `roi_isc_dev_models_perm_fwer_rois.csv`

### 4. 可视化

显著 ROI 热图：

```bash
python plot_roi_isc_dev_models_perm_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --alpha 0.05 \
  --sig-method fwer
```

单个 `stimulus_type + model` 的脑图：

```bash
python plot_brain_surface_vol.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --stimulus-type Passive_Emo \
  --model M_conv \
  --sig-method fwer \
  --alpha 0.05 \
  --no-preview
```

单个 `stimulus_type + model` 的年龄轨迹图：

```bash
python plot_roi_isc_age_trajectory.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --stimulus-type Passive_Emo \
  --model M_conv \
  --isc-method mahalanobis \
  --method fdr_model_wise \
  --top-k 5
```

---

## 二、脑和行为分析

这一部分在脑分支的基础上，构造行为的 trial-level 和 emotion-level 被试 × 被试矩阵，再与脑 ISC 做相关和置换检验。

### 1. 主要脚本

- `behavior/behavior_data.py`：读取、清洗、导出行为数据
- `behavior/build_behavior_trial_repr_matrix.py`：构造 trial-level 行为表征矩阵
- `behavior/build_behavior_emotion_repr_matrix.py`：从 trial-level pattern 聚合到 emotion-level 行为表征矩阵
- `behavior/calc_behavior_isc_by_age.py`：计算行为的被试 × 被试相似矩阵
- `behavior/joint_analysis_roi_isc_behavior.py`：计算脑 ISC 与行为 ISC 的关联，并做置换检验
- `plot_roi_isc_behavior_results.py`：绘制脑-行为结果热图和 top ROI 图

### 2. 当前实现要点

- 行为 trial/emotion 分支的刺激顺序，直接读取脑分支生成的 `stimulus_order.csv`
- 行为和脑的正式关联阶段，会自动取共同被试
- `data_4_hddm_ER.csv` 不再按 `sub_list_valid_4_ER.txt` 过滤被试
- 行为 ISC 可按 `missing_fraction_repr` 过滤缺失过高的被试
- 当前 `calc_behavior_isc_by_age.py` 默认 `--max-missing-fraction 0.2`
- 行为 distance-based ISC 不再错误使用 `fisher_z`
- 脑-行为 joint 默认支持双尾检验，当前推荐 `--tail two_sided`

### 3. 行为数据导出

```bash
python behavior/behavior_data.py \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior \
  --save-feature-table
```

主要输出：

- `behavior/data_4_hddm_ER.csv`
- `behavior/data_4_hddm_ER_subject_audit.csv`
- `behavior/data_4_behavior_feature_table.csv`

### 4. trial-level 脑和行为分析

先确保脑 trial 分支已经完成 `repr_trial` 和 `isc_trial`。

#### 4.1 构造行为 trial 表征矩阵

```bash
python behavior/build_behavior_trial_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --brain-repr-prefix roi_repr_matrix_232 \
  --feature-cols emot_rating \
  --agg-func mean \
  --diff-method euclidean
```

主要输出：

- `behavior_subject_stimulus_scores.csv`
- `behavior_subject_coverage.csv`
- `behavior_patterns_trial.npz`
- `behavior_diff_matrix_trial.npz`
- `behavior_repr_matrix_trial.npz`

#### 4.2 计算行为 ISC

```bash
python behavior/calc_behavior_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix behavior_repr_matrix_trial \
  --isc-method mahalanobis \
  --max-missing-fraction 0.2
```

主要输出：

- `behavior_isc_mahalanobis_by_age.npy`
- `behavior_isc_mahalanobis_by_age_subjects_sorted.csv`
- `behavior_isc_mahalanobis_by_age_meta.csv`
- `behavior_isc_mahalanobis_by_age_subject_filter.csv`

#### 4.3 脑-行为关联与置换检验

```bash
python behavior/joint_analysis_roi_isc_behavior.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --brain-repr-prefix roi_repr_matrix_232 \
  --brain-isc-method mahalanobis \
  --behavior-repr-prefix behavior_repr_matrix_trial \
  --behavior-isc-method mahalanobis \
  --assoc-method spearman \
  --tail two_sided \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

主要输出：

- `roi_isc_behavior_perm_fwer.csv`
- `roi_isc_behavior_perm_fwer_subjects_sorted.csv`
- `roi_isc_behavior_perm_fwer_rois.csv`
- `roi_isc_behavior_perm_fwer_summary.csv`

### 5. emotion-level 脑和行为分析

先确保脑 emotion 分支已经完成 `repr_emotion` 和 `isc_emotion`，并且 trial 行为分支已经跑完。

#### 5.1 构造行为 emotion 表征矩阵

```bash
python behavior/build_behavior_emotion_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --pattern-prefix behavior_patterns_trial \
  --out-stimulus-dir-name by_emotion \
  --diff-method euclidean
```

主要输出：

- `behavior_subject_emotion_scores.csv`
- `behavior_patterns_emotion4.npz`
- `behavior_diff_matrix_emotion4.npz`
- `behavior_repr_matrix_emotion4.npz`

#### 5.2 计算 emotion 行为 ISC

```bash
python behavior/calc_behavior_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix behavior_repr_matrix_emotion4 \
  --isc-method mahalanobis \
  --max-missing-fraction 0.2
```

#### 5.3 emotion 脑-行为关联与置换检验

```bash
python behavior/joint_analysis_roi_isc_behavior.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --brain-repr-prefix roi_repr_matrix_232_emotion4 \
  --brain-isc-method mahalanobis \
  --behavior-repr-prefix behavior_repr_matrix_emotion4 \
  --behavior-isc-method mahalanobis \
  --assoc-method spearman \
  --tail two_sided \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

### 6. 一键配置

trial：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json
```

emotion：

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json
```

如果脑分支已经跑完，只重跑行为与脑-行为步骤：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json --steps beh_repr_trial,beh_isc_trial,brain_beh_trial
```

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json --steps beh_repr_trial,beh_repr_emotion,beh_isc_emotion,brain_beh_emotion
```

### 7. 结果可视化

trial：

```bash
python plot_roi_isc_behavior_results.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --p-col p_fdr_bh_global \
  --alpha 0.05 \
  --top-k 10
```

emotion：

```bash
python plot_roi_isc_behavior_results.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --p-col p_fdr_bh_global \
  --alpha 0.05 \
  --top-k 10
```

默认输出目录：

- `figures/by_stimulus_brain_behavior/`
- `figures/by_emotion_brain_behavior/`

主要图和表：

- `brain_behavior_sig_results_<branch>_<p_col>_a<alpha>.csv`
- `brain_behavior_heatmap_<branch>_<p_col>_a<alpha>_all.png`
- `brain_behavior_top_roi_<branch>_<stimulus_type>_<source>_<p_col>_a<alpha>.png`

---

## 三、脑和年龄调制的行为分析

这一部分在已经得到 `Brain_ISC` 和 `Behavior_ISC` 的前提下，检验三种年龄模型是否会调制脑-行为关系。

### 1. 核心思想

对每个 ROI、每个年龄模型，建立如下回归：

```text
Brain_ISC = β1 * Behavior_ISC + β2 * Age_Model + β3 * (Behavior_ISC * Age_Model) + ε
```

其中：

- `β1`：行为主效应
- `β2`：年龄模型主效应
- `β3`：年龄调制脑-行为关系的交互效应

当前实现特点：

- 三种年龄模型 `M_nn / M_conv / M_div` 分别计算
- 三种年龄模型分别做 permutation 校正
- 正式推断以 permutation interaction 结果为主
- 不再输出缺乏统计解释意义的参数法 `p_behavior / p_age / p_interaction`

### 2. 主要脚本

- `behavior/joint_analysis_roi_isc_behavior_age_regression.py`
- `plot_roi_isc_behavior_age_regression_results.py`

### 3. 年龄调制分析命令

trial：

```bash
python behavior/joint_analysis_roi_isc_behavior_age_regression.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --brain-isc-prefix roi_isc_mahalanobis_by_age \
  --behavior-isc-prefix behavior_isc_mahalanobis_by_age \
  --rank-transform \
  --tail positive \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

emotion：

```bash
python behavior/joint_analysis_roi_isc_behavior_age_regression.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --brain-isc-prefix roi_isc_mahalanobis_by_age \
  --behavior-isc-prefix behavior_isc_mahalanobis_by_age \
  --rank-transform \
  --tail positive \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

推荐重点查看的列：

- `beta_interaction`
- `t_interaction`
- `p_perm_interaction`
- `p_fwer_interaction_model_wise`
- `p_fdr_perm_interaction_model_wise`

主要输出：

- `roi_isc_behavior_age_regression.csv`
- `roi_isc_behavior_age_regression_subjects_sorted.csv`
- `roi_isc_behavior_age_regression_summary.csv`

其中：

- `roi_isc_behavior_age_regression.csv`：每个 `stimulus_type` 下，每个年龄模型、每个 ROI 的回归结果
- `roi_isc_behavior_age_regression_subjects_sorted.csv`：实际进入年龄调制分析的共同被试和年龄
- `roi_isc_behavior_age_regression_summary.csv`：各 `stimulus_type × model` 下显著交互项 ROI 数量汇总

### 4. 参数说明

- `--rank-transform`：先对脑、行为、年龄模型向量做 rank + zscore，再进入回归。推荐开启，可以减小极端值影响，使分析更接近 Spearman 风格的稳健比较。
- `--tail two_sided`：推荐使用双尾 permutation 检验。
- `--correction-mode perm_fwer_fdr`：输出 permutation raw p、FWER 和 FDR。

### 5. 年龄调制结果可视化

trial：

```bash
python plot_roi_isc_behavior_age_regression_results.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --effect-col beta_interaction \
  --p-col p_fdr_perm_interaction_model_wise \
  --alpha 0.05 \
  --top-k 10
```

emotion：

```bash
python plot_roi_isc_behavior_age_regression_results.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --effect-col beta_interaction \
  --p-col p_fdr_perm_interaction_model_wise \
  --alpha 0.05 \
  --top-k 10
```

默认输出目录：

- `figures/by_stimulus_brain_behavior_age_regression/`
- `figures/by_emotion_brain_behavior_age_regression/`

主要图和表：

- `brain_behavior_age_regression_sig_results_<branch>_<effect_col>_<p_col>_a<alpha>.csv`
- `brain_behavior_age_regression_heatmap_<branch>_<model>_<effect_col>_<p_col>_a<alpha>_all.png`
- `brain_behavior_age_regression_top_roi_<branch>_<stimulus_type>_<model>_<source>_<effect_col>_<p_col>_a<alpha>.png`

---

## 附：校验脚本

常用校验脚本位于 `data_check/`：

- `check_stage0_lss.py`：检查 LSS 输出
- `check_stage1_repr_matrix.py`：检查脑表征矩阵输出
- `check_stage2_isc.py`：检查脑 ISC 输出
- `check_stage3_perm_results.py`：检查脑和年龄模型结果输出
- `check_stage4_figures.py`：检查脑分支图件输出
- `check_behavior_branch.py`：检查行为 trial/emotion 分支、行为 ISC、脑-行为 joint 的产物与被试覆盖
- `check_data_4_hddm_er_scores.py`：比较 `data_4_hddm_ER.csv` 和 `data_4_hddm_ER_base.csv` 的被试得分顺序是否一致
- `check_end_to_end_consistency.py`：做整体链路一致性检查

示例：

```bash
python data_check/check_behavior_branch.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```
