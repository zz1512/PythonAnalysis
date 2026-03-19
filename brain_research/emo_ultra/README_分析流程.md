# emo_ultra 分析流程

## 目标
在 `emo`（偏保守）与 `emo_new`（偏敏感）之间，提供一个折中的新流程：
- 先在每个 ROI 内构建刺激×刺激表征矩阵（RSM）
- 再做被试间相似性（ISC）
- 再做发育模型置换检验（模型内 FWER 或 FDR-BH，默认正向单尾）

## 输出目录结构（约定）
`--out-dir/--matrix-dir` 指向同一个根目录（默认是共享存放结果的目录），主要包含：
- `by_stimulus/<stimulus_type>/`：每个刺激类型一个子目录
- `figures/`：Step 4 绘图输出目录

每个 `by_stimulus/<stimulus_type>/` 中的主要产物：
- Step 1：`roi_repr_matrix_232.npz` + `roi_repr_matrix_232_{subjects,rois,meta}.csv` + `stimulus_order*.csv`
- Step 2：`roi_isc_spearman_by_age.npy` + `roi_isc_spearman_by_age_{subjects_sorted,rois}.csv`
- Step 3：`roi_isc_dev_models_perm_fwer.csv` + `roi_isc_dev_models_perm_fwer_{subjects_sorted,rois}.csv`

## 运行顺序（推荐）
1. Step 1：构建 ROI 表征矩阵（RSM）`build_roi_repr_matrix_232.py`
2. Step 2：RSM → ROI-ISC（按年龄排序）`calc_roi_isc_by_age.py`
3. Step 3：ROI-ISC × 发育模型置换检验 `joint_analysis_roi_isc_dev_models_perm_fwer.py`
4. Step 4：显著结果可视化（FWER 或 FDR）`plot_roi_isc_dev_models_perm_fwer_sig.py` / `plot_roi_isc_dev_models_perm_fdr_sig.py`

## Step 1：build_roi_repr_matrix_232.py
用途：对每个 `stimulus_type`，在每个 ROI 内构建被试的刺激×刺激表征矩阵（RSM）。

关键实现点：
- 同时读取 `lss_index_*_aligned.csv`（surface_L/surface_R/volume），合并后按条件过滤
- 被试必须同时具备 `EMO` 与 `SOC` 两任务
- 232 ROI = Schaefer200（surface L/R）+ Tian32（subcortex volume）
- 默认启用 cocktail blank：对 ROI 内特征做跨刺激去均值（减少单变量均值驱动）
- volume beta 若空间不一致，会重采样到 Tian atlas 空间（continuous 插值）

运行示例：
```bash
python build_roi_repr_matrix_232.py \
  --lss-root /public/home/dingrui/fmri_analysis/zz_analysis/lss_results \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --threshold-ratio 0.8
```

可选参数：
- `--stim-type-col`：索引中作为刺激类型的列名（默认 `raw_emotion`；若索引含 `label`，会优先按 `label` 前缀匹配 trial_type）
- `--stim-id-col`：索引中作为刺激 ID 的列名（默认 `stimulus_content`）
- `--no-cocktail-blank`：关闭 cocktail blank（默认启用）

## Step 2：calc_roi_isc_by_age.py
用途：读取 Step 1 的 RSM（subject×stim×stim），对每个 ROI：
1) 将每名被试的 RSM 取上三角展开为向量；2) 计算被试×被试 Spearman ISC；3) 按年龄排序输出。

运行示例：
```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv
```

说明：
- `--subject-info` 支持 `.csv/.tsv/.xlsx`，列名兼容 `sub_id/age` 或 `被试编号/采集年龄`
- 缺失年龄会被排到末尾（年龄=999）

## Step 3：joint_analysis_roi_isc_dev_models_perm_fwer.py
用途：对每个 `stimulus_type`，把 ROI-ISC 的上三角向量与 3 个发育模型进行置换检验，并做 model-wise maxT FWER 校正（正向单尾）。

发育模型（按 subject pair 定义）：
- `M_nn = amax - |ai-aj|`（年龄越近越相似）
- `M_conv = min(ai,aj)`（越成熟越相似的一种编码）
- `M_div = amax - 0.5*(ai+aj)`（平均年龄越小越大的一种编码）

运行示例：
```bash
python joint_analysis_roi_isc_dev_models_perm_fwer.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --n-perm 5000 \
  --seed 42
```

可选参数：
- `--no-normalize-models`：关闭对模型向量的 zscore（默认开启）
- `--no-fisher-z`：关闭 Fisher Z（atanh）变换（默认开启；开启时会先对 ISC r 值做 Fisher Z 再进入 zscore 与置换统计）

输出：
- `by_stimulus/<stimulus_type>/roi_isc_dev_models_perm_fwer.csv`
- 同目录下的 `roi_isc_dev_models_perm_fwer_subjects_sorted.csv` / `roi_isc_dev_models_perm_fwer_rois.csv`

## Step 4：显著结果可视化

### 4.1 FWER 版本：plot_roi_isc_dev_models_perm_fwer_sig.py
用途：汇总所有 stimulus_type 的显著 ROI（默认正效应 + FWER）并绘制热图。

运行示例：
```bash
python plot_roi_isc_dev_models_perm_fwer_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --alpha 0.05
```

### 4.2 FDR 版本：plot_roi_isc_dev_models_perm_fdr_sig.py
用途：基于 Step3 的置换原始 p（`p_perm_one_tailed`）计算 BH-FDR，并绘制显著结果热图。

运行示例：
```bash
python plot_roi_isc_dev_models_perm_fdr_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --alpha 0.05 \
  --fdr-mode model_wise
```

可选参数：
- `--fdr-mode`：`model_wise`（默认，每个模型内做 BH）或 `global`（该 stimulus_type 下全检验一起做 BH）
- `--positive-only`：仅保留 `r_obs > 0`

## Step 5：脑图映射（Surface + Volume）

### 5.1 显著 ROI 映射回大脑：plot_brain_surface_vol.py
用途：把 Step3 的显著 ROI（以 `r_obs` 为值，不显著置 0）映射回：
- Schaefer surface（L/R）导出 `.func.gii`
- Tian subcortex volume 导出 `.nii.gz`

运行示例（FWER 映射）：
```bash
python plot_brain_surface_vol.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --sig-method fwer \
  --alpha 0.05 \
  --no-preview
```

运行示例（FDR 映射：默认每个模型内 BH）：
```bash
python plot_brain_surface_vol.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --sig-method fdr_model_wise \
  --alpha 0.05 \
  --no-preview
```

可选参数：
- `--sig-method`：`fwer` / `fdr_model_wise` / `fdr_global` / `raw_p`
- `--sig-col`：当 `--sig-method raw_p` 时使用（`p_fwer_model_wise` 或 `p_perm_one_tailed`）
- `--no-preview`：仅导出影像文件，跳过 Nilearn 在线预览（推荐在服务器/离线环境使用）

输出目录：
- `<matrix-dir>/figures/brain_maps/`

### 5.2 Pair-wise 轨迹图：plot_roi_isc_age_trajectory.py
用途：选择显著 Top ROI，绘制“被试对平均年龄 vs ISC”的密度/散点图，并叠加拟合曲线（poly/lowess）。

运行示例：
```bash
python plot_roi_isc_age_trajectory.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --method fdr_model_wise \
  --alpha 0.05 \
  --top-k 5 \
  --plot-mode hexbin \
  --fit poly2
```

可选参数：
- `--plot-mode`：`hexbin`（默认，解决过度绘制）或 `scatter`
- `--fit`：`linear`/`poly2`/`poly3`/`lowess`（LOWESS 会自动对拟合数据点降采样）
- `--max-points`：绘图最大点数（默认 40000）
- `--fit-max-points`：LOWESS 拟合最大点数（默认 5000）

## data_check
- `data_check/check_stage1_repr_matrix.py`
- `data_check/check_stage2_isc.py`
- `data_check/check_stage3_perm_results.py`

> 建议每一步后都运行对应 `data_check` 脚本，确认维度、被试排序与显著性统计数量。
