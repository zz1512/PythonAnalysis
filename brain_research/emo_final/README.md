# emo\_final 分析流程（仅 Trial + Emotion）

该目录提供一套“ROI 表征（RSM）→ 被试间相似性（ISC）→ 发育模型置换检验 → 可视化”的端到端流程。

本版 `emo_final` 仅保留两条分析分支：

- Trial-level：按 `stimulus_type` 输出 trial×trial 的 ROI-RSM（默认输出在 `by_stimulus/`）
- Emotion-level（可选）：把 trial-level ROI pattern 聚合为 emotion-level pattern，再重算 emotion×emotion 的 ROI-RSM（默认输出在 `by_emotion/`）

默认参数（可通过命令行或 config 覆盖）：

- ROI set：`232`（Schaefer200 surface L/R + Tian subcortex volume）
- RSM：`--rsm-method spearman`
- ISC：`--isc-method mahalanobis`
- Permutation association：`--assoc-method spearman`

***

## 输出目录结构（约定）

所有下游脚本的 `--out-dir/--matrix-dir` 指向同一个根目录（建议固定一个结果根目录），其结构约定如下：

- `by_stimulus/<stimulus_type>/`：Trial-level 结果（每个 stimulus\_type 一个目录）
- `by_emotion/<stimulus_type>/`：Emotion-level 结果（每个 stimulus\_type 一个目录，和 by\_stimulus 并列，不覆盖）
- `figures/`：热图、脑图、轨迹图输出

Trial-level（`by_stimulus/<stimulus_type>/`）的主要产物：

- Step 1：`roi_repr_matrix_{200|232}.npz` + `roi_repr_matrix_{200|232}_{subjects,rois,meta}.csv` + `stimulus_order.csv`
- Step 1（可选，但强烈建议保留）：`roi_beta_patterns_{200|232}.npz` + `roi_beta_patterns_*_{subjects,rois,roi_feat_sizes,meta}.csv`
- Step 2：`roi_isc_<method>_by_age.npy` + `roi_isc_<...>_by_age_{subjects_sorted,rois,meta}.csv`
- Step 3：`roi_isc_dev_models_perm_fwer.csv`（含 FWER + 预计算 BH-FDR）+ `roi_isc_dev_models_perm_fwer_{subjects_sorted,rois}.csv`

Emotion-level（`by_emotion/<stimulus_type>/`）的主要产物：

- `roi_repr_matrix_{roi_set}_emotion4.npz`（ROI: `n_sub×4×4`）
- `roi_repr_matrix_{roi_set}_emotion4_{subjects,rois,meta}.csv`
- `emotion_order.csv`、`stimulus_to_emotion.csv`、`emotion_stimulus_counts.csv`
- （可选）`roi_repr_matrix_{roi_set}_emotion4_patterns.npz`（emotion-level patterns）

***

## 快速开始（推荐：配置驱动一键跑）

入口脚本：[run\_pipeline.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/run_pipeline.py)\
示例配置：[example\_config.json](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/example_config.json)

1. 复制一份配置并按需修改路径：

- `repr_trial.lss_root`：LSS 输出目录（包含 `lss_index_*_aligned.csv`）
- `repr_trial.out_dir`：ROI 结果根目录（by\_stimulus/by\_emotion/figures 会写在这里）
- `isc_*.subject_info`：年龄表

1. 执行（默认按 config 的 steps 顺序）：

```bash
python run_pipeline.py --config example_config.json
```

1. 只跑某几个步骤（逗号分隔）：

```bash
python run_pipeline.py --config example_config.json --steps repr_trial,isc_trial,perm_trial
```

### steps 说明（emo\_final）

- `lss`：可选，跑 LSS 生成 betas + aligned index
- `repr_trial`：构建 trial-level ROI-RSM（并默认保存 ROI patterns，供 emotion 聚合使用）
- `repr_emotion`：从 ROI patterns 聚合到 emotion-level，再重算 emotion-level ROI-RSM
- `isc_trial` / `isc_emotion`：对 trial/emotion 两条分支分别计算 ROI-ISC（按年龄排序）
- `perm_trial` / `perm_emotion`：对 trial/emotion 两条分支分别做发育模型置换检验（FWER + FDR）

***

## 手动逐步教程（当你希望单步调参/调试）

### Step 0（可选）：LSS（单试次 GLM）

脚本：[lss\_main.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/lss_main.py)\
配置：[glm\_config.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/glm_config.py)（包含 BIDS 路径规则、输出根目录）

关键点：

- LSS 会输出 `lss_index_<scenario>_aligned.csv`（含 `stimulus_content/raw_emotion/...`），供 Step1 对齐
- `stimulus_content` 的约定：`Task_Condition_Emotion`，用于跨被试 stimulus 顺序对齐

运行示例（建议先 sync 扫描已有输出，再决定是否重跑）：

```bash
python lss_main.py --write-mode sync
```

***

## Step 1：Trial-level ROI-RSM（build\_roi\_repr\_matrix.py）

脚本：[build\_roi\_repr\_matrix.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/build_roi_repr_matrix.py)

用途：对每个 `stimulus_type`，在每个 ROI 内构建被试的 `stimulus×stimulus` 表征矩阵（RSM）。

关键实现点（会影响输出）：

- 被试必须同时具备 `EMO` 与 `SOC` 两任务
- `stimulus_type`：优先由 aligned index 的 `label` 前缀匹配 audit 中的 trial\_type；否则使用 `raw_emotion`（或 `--stim-type-col`）
- 覆盖率筛选：按 `--threshold-ratio` 保留足够多被试中出现的 stimulus，并仅保留对这些 stimulus 全覆盖的被试（≥3 才输出）
- 默认开启 cocktail blank（跨 stimulus 去均值）
- 默认 `--rsm-method spearman`
- 默认 `save_pattern=true`：会额外保存 `roi_beta_patterns_*`（这是 emotion 聚合的输入）

运行示例：

```bash
python build_roi_repr_matrix.py \
  --lss-root /public/home/dingrui/fmri_analysis/zz_analysis/lss_results \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --roi-set 232 \
  --rsm-method spearman \
  --threshold-ratio 0.8
```

***

## Step 1.5（可选）：Emotion-level ROI-RSM（build\_roi\_emotion\_repr\_matrix.py）

脚本：[build\_roi\_emotion\_repr\_matrix.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/build_roi_emotion_repr_matrix.py)

用途：从 Step1 额外保存的 `roi_beta_patterns_{roi_set}.npz` 出发：

1. 在每个 ROI 内把 trial-level stimulus patterns 按情绪关键词聚合为 emotion-level patterns（默认 4 类：Anger/Disgust/Fear/Sad）
2. 再对 emotion-level patterns 重新计算 emotion×emotion RSM
3. 输出写入 `by_emotion/<stimulus_type>/`（不会覆盖 by\_stimulus 的 trial-level 结果）

运行示例：

```bash
python build_roi_emotion_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --roi-set 232 \
  --out-stimulus-dir-name by_emotion \
  --rsm-method spearman
```

说明：

- 若某个 `stimulus_type` 无法凑齐 4 类情绪（比如缺 Fear），该目录会被 `[SKIP]` 跳过（避免输出“不完整的 4×4”）

***

## Step 2：ROI-ISC（calc\_roi\_isc\_by\_age.py）

脚本：[calc\_roi\_isc\_by\_age.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/calc_roi_isc_by_age.py)

用途：对每个 ROI，把每名被试的 RSM 取上三角展平成向量，计算被试×被试相似性（ISC），并按年龄排序输出。

年龄表约定：

- 支持 `.csv/.tsv/.xlsx`
- 列名支持：`sub_id/age` 或 `被试编号/采集年龄`
- 年龄缺失会变成 `999` 并排到末尾（建议在正式分析前清理缺失）

Trial-level ISC 运行示例：

```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --isc-method mahalanobis
```

Emotion-level ISC 运行示例：

```bash
python calc_roi_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --repr-prefix roi_repr_matrix_232_emotion4 \
  --isc-method mahalanobis
```

***

## Step 3：置换检验（joint\_analysis\_roi\_isc\_dev\_models.py）

脚本：[joint\_analysis\_roi\_isc\_dev\_models.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/joint_analysis_roi_isc_dev_models.py)

用途：对每个 `stimulus_type`，把 ROI-ISC 的上三角向量与 3 个发育模型向量做关联。支持两种统计模式（`--correction-mode`）：

- `fdr_only`（默认）：仅做参数法单尾 p（基于相关系数近似）+ BH-FDR（不做置换，不输出可用的 FWER）
- `perm_fwer_fdr`：置换检验（正向单尾）+ model-wise maxT FWER + 预计算 BH-FDR

输出：

- raw permutation p：`p_perm_one_tailed`
- model-wise maxT FWER：`p_fwer_model_wise`
- 预计算 BH-FDR：`p_fdr_bh_model_wise / p_fdr_bh_global`

Trial-level 运行示例：

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

Emotion-level 运行示例：

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

置换检验示例（启用 FWER + FDR）：

```bash
python joint_analysis_roi_isc_dev_models.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --isc-method mahalanobis \
  --assoc-method spearman \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

***

## Step 4-6：可视化与脑图导出

- Step4：显著热图汇总：[plot\_roi\_isc\_dev\_models\_perm\_sig.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/plot_roi_isc_dev_models_perm_sig.py)
- Step5：显著 ROI 映射回脑：[plot\_brain\_surface\_vol.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/plot_brain_surface_vol.py)
- Step6：显著 ROI 的 pair-wise 轨迹图：[plot\_roi\_isc\_age\_trajectory.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/plot_roi_isc_age_trajectory.py)

建议：用 `--stimulus-dir-name by_stimulus` 或 `by_emotion` 显式指定你要画哪条分支的结果。

### Step 4：显著热图（plot\_roi\_isc\_dev\_models\_perm\_sig.py）

用途：汇总每个 `stimulus_type` 的显著 ROI×model（`M_nn/M_conv/M_div`）效应，并输出热图 + 显著结果表。

Trial-level 示例（对应 `config_trial.json -> plot_sig_trial`）：

```bash
python plot_roi_isc_dev_models_perm_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --sig-method fwer \
  --alpha 0.05 \
  --dpi 300
```

Emotion-level 示例（对应 `config_emotion.json -> plot_sig_emotion`）：

```bash
python plot_roi_isc_dev_models_perm_sig.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --repr-prefix roi_repr_matrix_232_emotion4 \
  --sig-method fwer \
  --alpha 0.05 \
  --dpi 300
```

补充：

- 若你更偏向 FDR，可改为 `--sig-method fdr_model_wise`（与配置中的 `fdr_mode=model_wise` 对齐）
- `--positive-only` 可仅保留 `r_obs > 0` 的显著 ROI（与 Step3 后处理逻辑一致）

### Step 5：显著 ROI 脑图导出（plot\_brain\_surface\_vol.py）

用途：针对“单个 stimulus\_type + 单个 model”，将显著 ROI 的 `r_obs` 回填到表面（L/R）+ 体积（V）并导出：

- `*.func.gii`（左/右皮层）
- `*.nii.gz`（皮层下体积）
- （可选）`*_brain_map.png` 预览图

Trial-level 示例（对应 `config_trial.json -> plot_brain_trial`）：

```bash
python plot_brain_surface_vol.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --sig-method fwer \
  --sig-col p_perm_one_tailed \
  --alpha 0.05 \
  --no-preview
```

Emotion-level 示例（对应 `config_emotion.json -> plot_brain_emotion`）：

```bash
python plot_brain_surface_vol.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --repr-prefix roi_repr_matrix_232_emotion4 \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --sig-method fwer \
  --sig-col p_perm_one_tailed \
  --alpha 0.05 \
  --no-preview
```

补充：

- 若在离线环境，建议保留 `--no-preview`（避免 nilearn 下载模板失败）
- 若想看某个 model 的原始置换 p 阈值图，可用 `--sig-method raw_p --sig-col p_perm_one_tailed`

### Step 6：pair-wise 年龄轨迹图（plot\_roi\_isc\_age\_trajectory.py）

用途：针对“单个 stimulus\_type + 单个 model”，筛选显著 Top-K ROI，绘制被试对平均年龄 vs ISC 的散点/hexbin 及拟合曲线。

Trial-level 示例（对应 `config_trial.json -> plot_traj_trial`）：

```bash
python plot_roi_isc_age_trajectory.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --repr-prefix roi_repr_matrix_232 \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --isc-method mahalanobis \
  --alpha 0.05 \
  --method fdr_model_wise \
  --top-k 5 \
  --fit lowess \
  --plot-mode hexbin \
  --max-points 40000 \
  --fit-max-points 5000 \
  --hexbin-gridsize 55 \
  --seed 42 \
  --dpi 300
```

Emotion-level 示例（对应 `config_emotion.json -> plot_traj_emotion`）：

```bash
python plot_roi_isc_age_trajectory.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --repr-prefix roi_repr_matrix_232_emotion4 \
  --stimulus-type <stimulus_type> \
  --model M_conv \
  --isc-method mahalanobis \
  --alpha 0.05 \
  --method fdr_model_wise \
  --top-k 5 \
  --fit poly2  \
  --plot-mode hexbin \
  --max-points 40000 \
  --fit-max-points 5000 \
  --hexbin-gridsize 55 \
  --seed 42 \
  --dpi 300
```

补充：

- 若想画更平滑趋势可改 `--fit lowess`（或加 `--lowess`）
- 若希望只看正相关 ROI，可加 `--positive-only`
- 该脚本会自动从 Step2 输出中匹配 `roi_isc_<isc-method>_by_age` 前缀；若同目录有多个 ISC 前缀，请显式传 `--isc-prefix`

***

## data\_check（强烈建议每步都跑）

- [check\_stage1\_repr\_matrix.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_stage1_repr_matrix.py)
- [check\_stage2\_isc.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_stage2_isc.py)
- [check\_stage3\_perm\_results.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_stage3_perm_results.py)

补充的检查脚本（对应当前 emo\_final 主流程）：

- [check\_stage0\_lss.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_stage0_lss.py)：检查 LSS aligned index/audit 与 beta 文件存在性（可选）
- [check\_end\_to\_end\_consistency.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_end_to_end_consistency.py)：检查 repr/pattern/isc/perm 的 subjects/rois/prefix 对齐
- [check\_stage4\_figures.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/data_check/check_stage4_figures.py)：检查热图/脑图/轨迹图输出文件数量

推荐顺序（与 pipeline stages 对齐）：

1. Stage0：LSS 输出检查（如果你跑了 lss\_main）：

```bash
python data_check/check_stage0_lss.py --lss-root /public/home/dingrui/fmri_analysis/zz_analysis/lss_results --check-files
```

1. Stage1：trial/emotion repr 输出检查：

```bash
python data_check/check_stage1_repr_matrix.py --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```

1. Stage2：ISC 输出检查（默认 mahalanobis）：

```bash
python data_check/check_stage2_isc.py --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```

1. Stage3：置换检验输出检查（可选同时核验 figures）：

```bash
python data_check/check_stage3_perm_results.py --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final --check-figures
```

1. End-to-end：跨阶段一致性检查（建议在正式统计前跑一次）：

```bash
python data_check/check_end_to_end_consistency.py --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```

1. Stage4-6：仅绘图产物检查（当你单独重跑绘图或想快速点检时）：

```bash
python data_check/check_stage4_figures.py --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final
```

说明：

- 默认同时检查 trial（`by_stimulus`）与 emotion（`by_emotion`）两条分支
- 若你自定义了 prefix/目录名，可通过各脚本参数覆盖默认值（`--trial-dir-name/--emotion-dir-name/--*-prefix`）

***

## 推荐两套“研究流程模板”（对应 Trial / Emotion 两分支）

### 流程 A：Trial-level（保全分辨率）

```bash
python build_roi_repr_matrix.py --roi-set 232 --rsm-method spearman
python calc_roi_isc_by_age.py --stimulus-dir-name by_stimulus --repr-prefix roi_repr_matrix_232 --isc-method mahalanobis
python joint_analysis_roi_isc_dev_models.py --stimulus-dir-name by_stimulus --isc-method mahalanobis --assoc-method spearman
python plot_roi_isc_dev_models_perm_sig.py --sig-method fwer --repr-prefix roi_repr_matrix_232 --stimulus-dir-name by_stimulus

# Optional: export brain maps for one stimulus_type and one model
python plot_brain_surface_vol.py --stimulus-dir-name by_stimulus --stimulus-type <stimulus_type> --model M_conv --sig-method fwer --alpha 0.05 --no-preview

# Optional: pairwise age trajectory for one stimulus_type and one model
python plot_roi_isc_age_trajectory.py --stimulus-dir-name by_stimulus --stimulus-type <stimulus_type> --model M_conv --isc-method mahalanobis --method fdr_model_wise --alpha 0.05 --top-k 5
```

对应 run\_pipeline 的一键模板（Trial-level）：

使用已提供的 [config\_trial.json](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/config_trial.json)。只需要按你的环境修改路径字段（如 `lss_root/out_dir/subject_info`），然后直接运行即可。

该配置会依次执行：

- `repr_trial` → `isc_trial` → `perm_trial`
- `plot_sig_trial`（显著热图，输出到 `figures/by_stimulus/`）
- `plot_brain_trial`（脑图映射，输出到 `figures/by_stimulus_brain_maps/`）
- `plot_traj_trial`（年龄轨迹，输出到 `figures/by_stimulus_pair_age_traj/`）

```bash
python run_pipeline.py --config config_trial.json
```

### 流程 B：Emotion-level（4×4 聚合降噪）

```bash
python build_roi_repr_matrix.py --roi-set 232 --rsm-method spearman
python build_roi_emotion_repr_matrix.py --roi-set 232
python calc_roi_isc_by_age.py --stimulus-dir-name by_emotion --repr-prefix roi_repr_matrix_232_emotion4 --isc-method mahalanobis
python joint_analysis_roi_isc_dev_models.py --stimulus-dir-name by_emotion --isc-method mahalanobis --assoc-method spearman
python plot_roi_isc_dev_models_perm_sig.py --sig-method fwer --repr-prefix roi_repr_matrix_232_emotion4 --stimulus-dir-name by_emotion

# Optional: export brain maps for one stimulus_type and one model (emotion branch)
python plot_brain_surface_vol.py --stimulus-dir-name by_emotion --stimulus-type <stimulus_type> --model M_conv --sig-method fwer --alpha 0.05 --no-preview

# Optional: pairwise age trajectory for one stimulus_type and one model (emotion branch)
python plot_roi_isc_age_trajectory.py --stimulus-dir-name by_emotion --stimulus-type <stimulus_type> --model M_conv --isc-method mahalanobis --method fdr_model_wise --alpha 0.05 --top-k 5
```

对应 run\_pipeline 的一键模板（Emotion-level）：

使用已提供的 [config\_emotion.json](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/config_emotion.json)。只需要按你的环境修改路径字段（如 `lss_root/out_dir/matrix_dir/subject_info`），然后直接运行即可。

该配置会依次执行：

- `repr_trial` → `repr_emotion` → `isc_emotion` → `perm_emotion`
- `plot_sig_emotion`（显著热图，输出到 `figures/by_emotion/`）
- `plot_brain_emotion`（脑图映射，输出到 `figures/by_emotion_brain_maps/`）
- `plot_traj_emotion`（年龄轨迹，输出到 `figures/by_emotion_pair_age_traj/`）

```bash
python run_pipeline.py --config config_emotion.json
```

***

## 行为分支补充说明（trial / emotion）

行为相关脚本统一收纳到 `behavior/` 子目录中：

- `behavior/behavior_data.py`：行为数据读取、清洗、对齐的公共逻辑
- `behavior/build_behavior_trial_repr_matrix.py`：构造 trial 级行为 `stimulus x stimulus` 表征矩阵
- `behavior/build_behavior_emotion_repr_matrix.py`：从 trial 级行为 pattern 聚合到 emotion 级行为矩阵
- `behavior/calc_behavior_isc_by_age.py`：计算行为分支的被试 x 被试相似性矩阵
- `behavior/joint_analysis_roi_isc_behavior.py`：计算脑 ISC 与行为 ISC 的关联，并做置换检验

行为结果导出现在统一由 [behavior/behavior_data.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/brain_research/emo_final/behavior/behavior_data.py) 负责，默认输出到与脑结果同一总根目录下的：

- `/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior/data_4_hddm_ER.csv`

### 行为 trial 分支手动命令

先导出 ER 行为表：

```bash
python behavior/behavior_data.py \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior
```

再构造 trial 级行为矩阵、行为 ISC、以及脑-行为检验：

```bash
python behavior/build_behavior_trial_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --brain-repr-prefix roi_repr_matrix_232 \
  --feature-cols emot_rating \
  --agg-func mean \
  --diff-method euclidean
```

```bash
python behavior/calc_behavior_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix behavior_repr_matrix_trial \
  --isc-method mahalanobis
```

```bash
python behavior/joint_analysis_roi_isc_behavior.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --brain-repr-prefix roi_repr_matrix_232 \
  --brain-isc-method mahalanobis \
  --behavior-repr-prefix behavior_repr_matrix_trial \
  --behavior-isc-method mahalanobis \
  --assoc-method spearman \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

Trial 分支的新增产物包括：

- `by_stimulus/<stimulus_type>/behavior_subject_stimulus_scores.csv`
- `by_stimulus/<stimulus_type>/behavior_patterns_trial.npz`
- `by_stimulus/<stimulus_type>/behavior_diff_matrix_trial.npz`
- `by_stimulus/<stimulus_type>/behavior_repr_matrix_trial.npz`
- `by_stimulus/<stimulus_type>/behavior_isc_<method>_by_age.npy`
- `by_stimulus/<stimulus_type>/roi_isc_behavior_perm_fwer.csv`

### 行为 emotion 分支手动命令

在完成 trial 行为 pattern 后，继续聚合到 emotion 级：

```bash
python behavior/build_behavior_emotion_repr_matrix.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --pattern-prefix behavior_patterns_trial \
  --out-stimulus-dir-name by_emotion \
  --diff-method euclidean
```

```bash
python behavior/calc_behavior_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix behavior_repr_matrix_emotion4 \
  --isc-method mahalanobis
```

```bash
python behavior/joint_analysis_roi_isc_behavior.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_emotion \
  --brain-repr-prefix roi_repr_matrix_232_emotion4 \
  --brain-isc-method mahalanobis \
  --behavior-repr-prefix behavior_repr_matrix_emotion4 \
  --behavior-isc-method mahalanobis \
  --assoc-method spearman \
  --correction-mode perm_fwer_fdr \
  --n-perm 5000 \
  --seed 42
```

Emotion 分支的新增产物包括：

- `by_emotion/<stimulus_type>/behavior_subject_emotion_scores.csv`
- `by_emotion/<stimulus_type>/behavior_patterns_emotion4.npz`
- `by_emotion/<stimulus_type>/behavior_diff_matrix_emotion4.npz`
- `by_emotion/<stimulus_type>/behavior_repr_matrix_emotion4.npz`
- `by_emotion/<stimulus_type>/behavior_isc_<method>_by_age.npy`
- `by_emotion/<stimulus_type>/roi_isc_behavior_perm_fwer.csv`

### 一键配置运行

如果你希望从脑分支与行为分支一起顺序跑完，可以直接使用新配置：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json
```

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json
```

如果脑分支结果已经准备好，也可以只跑行为与脑-行为步骤：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json --steps beh_repr_trial,beh_isc_trial,brain_beh_trial
```

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json --steps beh_repr_trial,beh_repr_emotion,beh_isc_emotion,brain_beh_emotion
```

***

## 行为分支更新说明（2026-03）

### 1. 行为数据导出

行为结果现在统一由 `behavior/behavior_data.py` 导出，常用命令：

```bash
python behavior/behavior_data.py \
  --out-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior \
  --save-feature-table
```

默认输出包括：

- `behavior/data_4_hddm_ER.csv`
- `behavior/data_4_hddm_ER_subject_audit.csv`
- `behavior/data_4_behavior_feature_table.csv`（启用 `--save-feature-table` 时）

说明：

- `data_4_hddm_ER.csv` 已不再按 `sub_list_valid_4_ER.txt` 过滤被试
- 只要被试存在可用的 `valid_emot` trial，就会进入 ER 导出结果

### 2. 行为 ISC 缺失过滤

`behavior/calc_behavior_isc_by_age.py` 现在支持：

```bash
--max-missing-fraction
```

这个参数用于过滤行为表征矩阵缺失比例过高的被试。

例如：

```bash
python behavior/calc_behavior_isc_by_age.py \
  --matrix-dir /public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final \
  --stimulus-dir-name by_stimulus \
  --subject-info /public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv \
  --repr-prefix behavior_repr_matrix_trial \
  --isc-method mahalanobis \
  --max-missing-fraction 0.2
```

含义：

- `0.2` 表示只保留 `missing_fraction_repr < 0.2` 的被试
- 过滤发生在行为 ISC 计算阶段
- 后续脑-行为 joint 会自动取脑和行为的共同被试

新增输出：

- `behavior_isc_<method>_by_age_subject_filter.csv`
  记录每个被试的
  - `missing_fraction_repr`
  - `keep_for_behavior_isc`
- `behavior_isc_<method>_by_age_meta.csv`
  现在还会记录
  - `max_missing_fraction`
  - `n_subjects_before_filter`
  - `n_subjects_after_filter`
  - `n_subjects_excluded_missing`

### 3. pipeline 默认配置

两份行为配置已经默认加入缺失过滤阈值：

- `behavior/config_behavior_trial.json`
  - `beh_isc_trial.max_missing_fraction = 0.2`
- `behavior/config_behavior_emotion.json`
  - `beh_isc_emotion.max_missing_fraction = 0.2`

直接运行即可生效：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json
```

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json
```

如果脑分支已经跑完，只重跑行为与脑-行为关联：

```bash
python run_pipeline.py --config behavior/config_behavior_trial.json --steps beh_repr_trial,beh_isc_trial,brain_beh_trial
```

```bash
python run_pipeline.py --config behavior/config_behavior_emotion.json --steps beh_repr_trial,beh_repr_emotion,beh_isc_emotion,brain_beh_emotion
```
