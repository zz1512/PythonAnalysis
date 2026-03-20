# emo\_ultra 分析流程

## 目标

在 `emo`（偏保守）与 `emo_new`（偏敏感）之间，提供一个折中的新流程：

- 先在每个 ROI 内构建刺激×刺激表征矩阵（RSM）
- 再做被试间相似性（ISC）
- 再做发育模型置换检验（模型内 FWER 或 FDR-BH，默认正向单尾）

## 输出目录结构（约定）

`--out-dir/--matrix-dir` 指向同一个根目录（默认是共享存放结果的目录），主要包含：

- `by_stimulus/<stimulus_type>/`：每个刺激类型一个子目录
- `figures/`：绘图与脑图导出目录

每个 `by_stimulus/<stimulus_type>/` 中的主要产物：

- Step 1：`roi_repr_matrix_{200|232}.npz` + `roi_repr_matrix_{200|232}_{subjects,rois,meta}.csv` + `stimulus_order*.csv`
- Step 2：`roi_isc_{spearman|pearson}_by_age.npy` + `roi_isc_{...}_by_age_{subjects_sorted,rois,meta}.csv`
- Step 3：`roi_isc_dev_models_perm_fwer.csv`（含 FWER 与预计算 BH-FDR 列）+ `roi_isc_dev_models_perm_fwer_{subjects_sorted,rois}.csv`

## 运行顺序（推荐）

1. Step 1：构建 ROI 表征矩阵（RSM）`build_roi_repr_matrix.py`
2. Step 1.5（可选）：刺激矩阵 → 刺激情绪矩阵（4×4）`build_roi_emotion_repr_matrix.py`
3. Step 2：RSM → ROI-ISC（按年龄排序）`calc_roi_isc_by_age.py`
4. Step 3：ROI-ISC × 发育模型置换检验 `joint_analysis_roi_isc_dev_models.py`
5. Step 4：显著结果可视化热图（FWER 或 FDR）`plot_roi_isc_dev_models_perm_sig.py`
6. Step 5：显著结果映射到脑图（Surface + Volume）`plot_brain_surface_vol.py`
7. Step 6：显著 ROI 的 pair-wise 年龄轨迹图（可选）`plot_roi_isc_age_trajectory.py`


## 关键约定与输入要求

### Step 1 输入（LSS index + beta）

- `--lss-root` 下需要存在 `lss_index_*_aligned.csv`（通常对应 surface\_L / surface\_R / volume 的索引）。
- 每个索引文件至少需要包含列：`subject/task/file/<stim_type_col>/<stim_id_col>`。
- `subject` 会做字符串清洗；`task` 会转为大写并要求每名被试同时具备 `EMO` 与 `SOC` 两任务（用于被试筛选）。
- `stimulus_type` 的确定逻辑：
  - 如果索引中包含 `label` 列，会用 `label` 的前缀去匹配 `trial_type`，匹配到的前缀作为 `stimulus_type`；
  - 否则用 `--stim-type-col` 指定的列（默认 `raw_emotion`）并过滤到 `trial_type` 列表内。
- `trial_type` 列表来自：`<lss-root>/lss_audit_surface_L.csv` 中的 `trial_type` 列。
- beta 文件路径解析：
  - 若 `file` 列是绝对路径，直接使用；
  - 否则按 `<lss-root>/<task>/<subject>/<file>` 或 `<lss-root>/<task>/<subject>/run-<run>/<file>` 尝试拼接。

### 年龄表约定（Step 2/3）

- `--subject-info` 支持 `.csv/.tsv/.xlsx`：
  - 英文列名：`sub_id/age`
  - 中文列名：`被试编号/采集年龄`
- 被试编号会统一成 `sub-XXXX` 格式。
- 年龄缺失会被写成 `999` 并排到末尾。分析假定年龄表完整且无缺失；如确实存在缺失，建议先补齐或在预处理时剔除缺失被试，避免影响模型含义与统计解释。

### 统计检验约定（Step 3/4/5）

- Step 3 置换检验使用正向单尾（对更大的正相关更敏感），并做 model-wise maxT FWER 校正；同时在结果表中预计算 BH-FDR（model-wise 与 global 两种列）。
- Step 4 的 FWER 热图默认只汇总正效应显著 ROI；FDR 热图可通过 `--positive-only` 控制是否仅保留 `r_obs > 0`。

## Step 1：build\_roi\_repr\_matrix.py

用途：对每个 `stimulus_type`，在每个 ROI 内构建被试的刺激×刺激表征矩阵（RSM）。

关键实现点：

- 同时读取 `lss_index_*_aligned.csv`（surface\_L/surface\_R/volume），合并后按条件过滤
- 被试必须同时具备 `EMO` 与 `SOC` 两任务
- 232 ROI = Schaefer200（surface L/R）+ Tian32（subcortex volume）
- 默认启用 cocktail blank：对 ROI 内特征做跨刺激去均值（减少单变量均值驱动）
- volume beta 若空间不一致，会重采样到 Tian atlas 空间（continuous 插值）
- stimulus 覆盖率筛选：每个 `stimulus_type` 内仅保留在足够多被试中出现的 stimulus（由 `--threshold-ratio` 控制），并进一步筛选出对保留下来的 stimuli “全覆盖”的被试集合进入该 `stimulus_type` 的输出。

运行示例：

```bash
python build_roi_repr_matrix.py \
  --lss-root /public/home/dingrui/fmri_analysis/zz_analysis/lss_results \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --threshold-ratio 0.8
```

可选参数：

- `--stim-type-col`：索引中作为刺激类型的列名（默认 `raw_emotion`；若索引含 `label`，会优先按 `label` 前缀匹配 trial\_type）
- `--stim-id-col`：索引中作为刺激 ID 的列名（默认 `stimulus_content`）
- `--no-cocktail-blank`：关闭 cocktail blank（默认启用）
- `--roi-set`：选择 ROI 集合（`232`=surface+volume；`200`=仅 surface；默认 `200`）
- `--rsm-method`：RSM 相关方法（`pearson`/`spearman`；默认 `pearson`）


## Step 1.5（可选）：build_roi_emotion_repr_matrix.py

用途：读取 Step1 的 `subject×stimulus×stimulus` ROI 矩阵，并按刺激文件名中的关键词聚合为 `subject×emotion×emotion`。

默认情绪顺序：`Anger, Disgust, Fear, Sad`，输出 4×4 矩阵。

运行示例：

```bash
python build_roi_emotion_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --repr-prefix roi_repr_matrix_232 \
  --out-prefix roi_repr_matrix_232_emotion4
```

输出（每个 `by_stimulus/<stimulus_type>/`）：

- `roi_repr_matrix_232_emotion4.npz`（ROI: `n_sub×4×4`）
- `roi_repr_matrix_232_emotion4_{subjects,rois,meta}.csv`
- `emotion_order.csv`、`stimulus_to_emotion.csv`、`emotion_stimulus_counts.csv`

## Step 2：calc\_roi\_isc\_by\_age.py

用途：读取 Step 1 的 RSM（subject×stim×stim），对每个 ROI：

1. 将每名被试的 RSM 取上三角展开为向量；2) 计算被试×被试 ISC（Pearson/Spearman）；3) 按年龄排序输出。

运行示例：

```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv
```

说明：

- `--subject-info` 支持 `.csv/.tsv/.xlsx`，列名兼容 `sub_id/age` 或 `被试编号/采集年龄`
- 缺失年龄会被排到末尾（年龄=999；默认假定不会发生缺失）
- 默认输出前缀为 `roi_isc_{isc_method}_by_age`，可用 `--isc-prefix` 自定义
- `--isc-method ` （`pearson`/`spearman`；默认 `spearman`）只影响 Step2 被试×被试 ISC 的计算方式；不会改变 Step1 的 `--rsm-method`，也不会改变 Step3 的 `--assoc-method`
- `--stimulus-dir-name`：指定条件目录名（默认 `by_stimulus`），便于兼容其他分组目录
- 若使用 Step1.5 产物，请将 `--repr-prefix` 设为对应情绪矩阵前缀（如 `roi_repr_matrix_232_emotion4`）
- 若某个 `stimulus_type` 目录缺少 `--repr-prefix` 对应的 `{repr_prefix}.npz / {repr_prefix}_subjects.csv / {repr_prefix}_rois.csv`，Step2 会自动打印 `[SKIP]` 并跳过该目录。

## Step 3：joint\_analysis\_roi\_isc\_dev\_models.py

用途：对每个 `stimulus_type`，把 ROI-ISC 的上三角向量与 3 个发育模型进行置换检验，并做 model-wise maxT FWER 校正（正向单尾）。

发育模型（按 subject pair 定义）：

- `M_nn = amax - |ai-aj|`（年龄越近越相似）
- `M_conv = min(ai,aj)`（越成熟越相似的一种编码）
- `M_div = amax - 0.5*(ai+aj)`（平均年龄越小越大的一种编码）

运行示例：

```bash
python joint_analysis_roi_isc_dev_models.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --n-perm 5000 \
  --seed 42
```

可选参数：

- `--isc-method` / `--isc-prefix`：指定读取 Step2 的哪一种 ISC 产物；不指定时会在每个 stimulus\_type 目录下自动识别 `roi_isc_*_by_age`（若存在多套会要求显式指定）
- `--repr-prefix`：可选；用于按 Step1/1.5 的矩阵前缀筛选可分析的 `stimulus_type`。若目录缺少该前缀对应三件套文件，则自动 `[SKIP]`。
- `--stimulus-dir-name`：指定条件目录名（默认 `by_stimulus`）
- `--assoc-method`：ROI-ISC 与发育模型的关联方式（默认 pearson；可选 spearman）
- 性能说明：当 `--assoc-method spearman` 时，会对 ROI-ISC 与模型向量做预排秩并使用矩阵运算计算关联，避免在置换循环中对每个 ROI 反复排序，适合大 `--n-perm` 场景。
- `--no-normalize-models`：关闭对模型向量的 zscore（默认开启）
- `--no-fisher-z`：关闭 Fisher Z（atanh）变换（默认开启；开启时会先对 ISC r 值做 Fisher Z 再进入 zscore 与置换统计）

输出：

- `by_stimulus/<stimulus_type>/roi_isc_dev_models_perm_fwer.csv`
- 同目录下的 `roi_isc_dev_models_perm_fwer_subjects_sorted.csv` / `roi_isc_dev_models_perm_fwer_rois.csv`

## Step 4：显著结果可视化

### 4.1 FWER 版本：plot\_roi\_isc\_dev\_models\_perm\_sig.py

用途：汇总所有 stimulus\_type 的显著 ROI（默认正效应 + FWER）并绘制热图。

运行示例：

```bash
python plot_roi_isc_dev_models_perm_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --alpha 0.05
```

### 4.2 FDR 版本：plot\_roi\_isc\_dev\_models\_perm\_sig.py

用途：基于 Step3 已写入的 BH-FDR 列（`p_fdr_bh_model_wise` / `p_fdr_bh_global`）绘制显著结果热图。

运行示例：

```bash
python plot_roi_isc_dev_models_perm_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra \
  --alpha 0.05 \
  --sig-method fdr_model_wise
```

可选参数：

- `--fdr-mode`：`model_wise`（默认，每个模型内做 BH）或 `global`（该 stimulus\_type 下全检验一起做 BH）
- `--positive-only`：仅保留 `r_obs > 0`
- `--stimulus-dir-name`：指定条件目录名（默认 `by_stimulus`）
- `--repr-prefix`：可选；若提供，仅汇总存在该前缀矩阵三件套的 `stimulus_type`，其余目录自动 `[SKIP]`。

## Step 5：脑图映射（Surface + Volume）

### 5.1 显著 ROI 映射回大脑：plot\_brain\_surface\_vol.py

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
- `--sig-col`：当 `--sig-method raw_p` 时使用（`p_perm_one_tailed`）
- `--no-preview`：仅导出影像文件，跳过 Nilearn 在线预览（推荐在服务器/离线环境使用）
- `--stimulus-dir-name`：指定条件目录名（默认 `by_stimulus`）
- `--repr-prefix`：可选；若指定且当前 `--stimulus-type` 目录缺少对应矩阵三件套，则直接 `[SKIP]` 返回。

输出目录：

- `<matrix-dir>/figures/brain_maps/`

## Step 6：Pair-wise 轨迹图：plot\_roi\_isc\_age\_trajectory.py

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
- `--stimulus-dir-name`：指定条件目录名（默认 `by_stimulus`）
- `--repr-prefix`：可选；若指定且当前 `--stimulus-type` 目录缺少对应矩阵三件套，则直接 `[SKIP]` 返回。
- `--max-points`：绘图最大点数（默认 40000）
- `--fit-max-points`：LOWESS 拟合最大点数（默认 5000）

## data\_check

- `data_check/check_stage1_repr_matrix.py`
- `data_check/check_stage2_isc.py`
- `data_check/check_stage3_perm_results.py`

> 建议每一步后都运行对应 `data_check` 脚本，确认维度、被试排序与显著性统计数量。
