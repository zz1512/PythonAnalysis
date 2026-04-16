# metaphoric/final_version

> `README.md` = 怎么跑（命令与产出对账）；`readme_detail.md` = 为什么这么跑 / 每步怎么解读与 QC  
> 教学版详解：[`readme_detail.md`](./readme_detail.md)

这版 `final_version` 目标是“能跑通、能复现、能讲清楚主线故事”。为避免分析过多导致叙事混乱，下面把流程重构为：

- 核心主线（必跑）：支持论文主结论
- 控制验证（推荐）：排除重测/重复暴露等替代解释
- 机制扩展（可选）：PNAS 加分项，按需添加

## 根目录与环境变量

默认数据根目录为 `E:/python_metaphor`，但建议统一通过环境变量指定，保证跨平台可复现：

- Windows PowerShell：
  - `$env:PYTHON_METAPHOR_ROOT="E:/python_metaphor"`
- macOS/Linux：
  - `export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor`

可选环境变量：
- `METAPHOR_DATA_EVENTS`：行为/事件文件根目录（默认 `${PYTHON_METAPHOR_ROOT}/data_events`）
- `METAPHOR_BEHAVIOR_RESULTS`：行为输出目录（默认 `${PYTHON_METAPHOR_ROOT}/behavior_results`）

## 输入与输出（相对于 `${PYTHON_METAPHOR_ROOT}`）

输入：
- `data_events/`：events 与行为数据（含 run-1..7 的 `*_events.tsv`）
- `Pro_proc_data/`：预处理后的 BOLD + confounds
- `stimuli_template.csv`：前后测 RSA 的刺激配对模板

中间产物（脚本会生成）：
- `lss_betas_final/`：LSS 的单 trial beta + trial metadata
- `glm_analysis_fwe_final/`：学习阶段 GLM 结果
- `roi_masks_final/`：ROI masks（若使用 `create_optimized_roi_masks.py`）
- `pattern_root/`：由 `stack_patterns.py` 生成的 4D patterns（你也可自定义路径）

输出（常见）：`rd_results/`、`gps_results/`、`rsa_results_optimized/`、`rsa_learning_results/`、`mvpa_*_results/`、`brain_behavior_results/` 等。

## 输出文件名约定（必须遵守）

### 1) Pattern Root（由 `stack_patterns.py` 生成）

目录结构：
- `${PATTERN_ROOT}/sub-01/`
- `${PATTERN_ROOT}/sub-02/`
- ...

每个 subject 文件夹内的核心文件（4D NIfTI，最后一维是 trials）：
- `pre_yy.nii.gz`, `pre_kj.nii.gz`
- `post_yy.nii.gz`, `post_kj.nii.gz`
- `learn_yy.nii.gz`, `learn_kj.nii.gz`（可选：做学习阶段动态/泛化才需要）

每个 4D 文件都有对应 metadata：
- `pre_yy_metadata.tsv`（至少包含 `beta_path`、`run`、`trial_id`）

命名规则说明：
- `stack_patterns.py` 会把 `yyw/yyew -> yy`、`kjw/kjew -> kj`，避免生成下游找不到的 `pre_yyw.nii.gz`。
- 默认排除 `fake`（可通过 `--exclude-conditions` 改）。

### 2) ROI 目录

ROI mask 目录通常为：
- `${PYTHON_METAPHOR_ROOT}/roi_masks_final/`（推荐）

或直接使用 GLM 二阶目录中的 `*_roi_mask.nii.gz`。

### 3) Searchlight 的 subject mask

`rd_searchlight.py` 默认读取 `${SUBJECT_MASK_ROOT}/sub-xx/mask.nii`，建议显式指定：
- `--mask-filename mask.nii.gz`

## 故事线（重排版）

### A. 核心主线（必跑，支撑主结论）
目标：回答“隐喻联结是否引起比空间联结更强的神经表征重组？”

1. 行为（Run7）：`YY vs KJ` 记忆/反应时差异
2. 学习阶段 GLM（Run3-4）：定位 `yy > kj` 的关键脑区（给“在哪里”）
3. 前后测 LSS（Run1-2/5-6）→ patterns：得到 `pre/post_yy/kj` 的 4D patterns（后续一切输入）
4. 前后测 RSA（item-wise）+ LMM：核心证据（表征结构变化）
5. RD/GPS（ROI-level）：互补证据（几何压缩/紧凑度变化）
6. 脑-行为关联：跨层次一致性与解释力

### B. 控制验证（推荐）
目标：排除“只是重复暴露效应”的替代解释。

- baseline 控制：在 patterns 中加入 baseline，计算 `Δbaseline` 并验证 `Δyy/Δkj` 明显大于 `Δbaseline`。

### C. 机制扩展（可选加分项）
- learning dynamics：学习阶段按时间窗追踪 RD/GPS/（可选）MVPA 趋势
- gPPI：学习阶段 seed×(yy-kj) 的调节连接
- effective connectivity：简化方向性信息流（beta-series 回归）
- parametric modulation：熟悉度/暴露次数等连续变量调制
- lateralization：偏侧化（LI）指标
- mediation：中介路径（把“并列证据”变成“机制假设”）
- BF10 / bootstrap CI：稳健性呈现

## 脚本与功能（索引）

## 脚本输出速查（主线常用）

下面列的是“跑完之后你应该在磁盘上看到什么”，用于快速核对（不是完整列表，以脚本实际输出为准）：

| 脚本 | 主要输出 |
|------|----------|
| `glm_analysis/run_lss.py` | `${PYTHON_METAPHOR_ROOT}/lss_betas_final/sub-xx/run-*/` + `lss_metadata_index_final.csv` |
| `representation_analysis/stack_patterns.py` | `${PATTERN_ROOT}/sub-xx/{pre,post,learn}_{yy,kj}.nii.gz` + `*_metadata.tsv` |
| `rsa_analysis/run_rsa_optimized.py` | `rsa_itemwise_details.csv` + `rsa_summary_stats.csv` |
| `rsa_analysis/rsa_lmm.py` | `rsa_lmm_summary.txt` + 参数/系数表（tsv/csv） |
| `representation_analysis/rd_analysis.py` | `rd_subject_metrics.tsv` + `rd_group_summary.tsv` |
| `representation_analysis/gps_analysis.py` | `gps_subject_metrics.tsv` + `gps_group_summary.tsv` |
| `brain_behavior/brain_behavior_correlation.py` | `brain_behavior_merged.tsv` + 相关矩阵/图表 |

### 1) 行为
- `behavior_analysis/addMemory.py`
- `behavior_analysis/getMemoryDetail.py`
- `behavior_analysis/getAccuracyScore.py`
- `behavior_analysis/getTimeActionScore.py`
- `behavior_analysis/behavior_lmm.py`

### 2) GLM / LSS
- `glm_analysis/main_analysis.py`（学习阶段 GLM）
- `glm_analysis/run_lss.py`（前后测 LSS）
- `glm_analysis/check_lss_comprehensive.py`（LSS 质量检查）
- `fmri_lss/fmri_lss_enhance.py`（学习阶段 LSS）
- `glm_analysis/parametric_modulation.py`（可选：参数调制）

### 3) MVPA
- `fmri_mvpa/fmri_mvpa_roi/fmri_mvpa_roi_pipeline_merged.py`
- `fmri_mvpa/fmri_mvpa_searchlight/fmri_mvpa_searchlight_pipeline_multi.py`
- `fmri_mvpa/fmri_mvpa_roi/mvpa_pre_post_comparison.py`
- `fmri_mvpa/fmri_mvpa_roi/cross_phase_decoding.py`

### 4) RSA
- `rsa_analysis/check_data_integrity.py`
- `rsa_analysis/run_rsa_optimized.py`
- `rsa_analysis/rsa_lmm.py`
- `rsa_analysis/cross_roi_rsa.py`
- `rsa_analysis/model_rdm_comparison.py`
- `rsa_analysis/plot_rsa_results.py`

### 5) 表征几何（RD/GPS）
- `representation_analysis/stack_patterns.py`
- `representation_analysis/rd_analysis.py`
- `representation_analysis/gps_analysis.py`
- `representation_analysis/rd_searchlight.py`
- `representation_analysis/seed_connectivity.py`
- `representation_analysis/rd_robustness.py`

### 6) 脑-行为
- `brain_behavior/brain_behavior_correlation.py`
- `brain_behavior/mediation_analysis.py`（可选）

### 7) 连接与偏侧化（可选）
- `connectivity_analysis/gPPI_analysis.py`
- `connectivity_analysis/effective_connectivity.py`
- `lateralization/lateralization_analysis.py`

### 8) 统计工具（可选）
- `utils/bayesian_stats.py`
- `utils/bootstrap_ci.py`

## 推荐执行顺序（带命令示例）

说明：示例默认在仓库根目录执行；路径里尽量使用 `${PYTHON_METAPHOR_ROOT}`。

### 每步产出检查点（跑完用来对账）

如果你只想确认“我是不是跑对了”，优先看每一步的这些产出是否出现：

1. Step 1 行为：输出 CSV/TSV 统计结果（准确率、RT）与控制台统计摘要
2. Step 2 前后测 LSS：`${PYTHON_METAPHOR_ROOT}/lss_betas_final/sub-xx/run-1/` 等目录出现大量单 trial beta + `lss_metadata_index_final.csv`
3. Step 3 学习 LSS + GLM：`lss_betas_final/sub-xx/run-3/` 和 `glm_analysis_fwe_final/` 下出现学习阶段对比与二阶结果
4. Step 4 patterns：`${PYTHON_METAPHOR_ROOT}/pattern_root/sub-xx/pre_yy.nii.gz`、`post_kj.nii.gz` 等 4D 文件出现
5. Step 5 RSA：生成 `rsa_itemwise_details.csv`、`rsa_summary_stats.csv`、以及 LMM 结果文本/表格
6. Step 6 RD/GPS：生成 ROI-level 的 `rd_results/*.tsv`、`gps_results/*.tsv`
7. Step 7 脑-行为：生成合并后的指标表与相关/回归输出
8. Step 8（可选）：learning dynamics / gPPI / effective connectivity / mediation / BF10 / bootstrap CI 等补充结果

### Step 0：设置根目录（建议每次都做）

Windows PowerShell：
```powershell
$env:PYTHON_METAPHOR_ROOT="E:/python_metaphor"
```

macOS/Linux：
```bash
export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor
```

### Step 1（必跑）：行为（Run7）
```bash
cd metaphoric/final_version/behavior_analysis
python addMemory.py
python getMemoryDetail.py
python getAccuracyScore.py
python getTimeActionScore.py
```

产出（默认相对于 `${PYTHON_METAPHOR_ROOT}`，具体文件名以脚本内常量为准）：
- `data_events/**/sub-xx_run-7_events.tsv` 被补充字段（如 `memory`、`action_time` 等）
- `behavior_results/` 下的行为统计汇总（如果脚本写文件）或终端输出的 t 检验摘要

### Step 2（必跑）：前后测 LSS（Run1-2, 5-6）+ 质量检查
```bash
cd ../glm_analysis
python run_lss.py
python check_lss_comprehensive.py
```

产出：
- `lss_betas_final/sub-xx/run-1/` ... `run-6/`：单 trial 的 beta maps（NIfTI）
- `lss_betas_final/lss_metadata_index_final.csv`：索引表（RSA/stack_patterns 会用）
- 质量检查报告：`check_lss_comprehensive.py` 的日志输出（以及它写出的任何 `*_report`/`*.json` 文件）

### Step 3（必跑）：学习阶段 LSS（Run3-4）+ 学习阶段 GLM（Run3-4）
```bash
cd ../fmri_lss
python fmri_lss_enhance.py

cd ../glm_analysis
python main_analysis.py
```

产出：
- `lss_betas_final/sub-xx/run-3/`、`run-4/`：学习阶段单 trial beta maps
- `glm_analysis_fwe_final/`：学习阶段 GLM 一阶/二阶结果（含 cluster-FWE）

（可选）生成 ROI masks：
```bash
cd ../fmri_mvpa/fmri_mvpa_roi
python create_optimized_roi_masks.py
```

产出（推荐放到统一目录）：
- `roi_masks_final/*.nii.gz`：解剖/网络/功能簇 ROI masks（供 MVPA/RSA/RD/GPS 使用）

### Step 4（必跑）：模式堆叠（生成 pattern_root）
```bash
cd ../../representation_analysis
python stack_patterns.py ${PYTHON_METAPHOR_ROOT}/lss_betas_final/lss_metadata_index_final.csv ${PYTHON_METAPHOR_ROOT}/pattern_root
```

产出（强依赖命名；下游脚本默认找这些文件）：
- `pattern_root/sub-xx/pre_yy.nii.gz`、`pre_kj.nii.gz`
- `pattern_root/sub-xx/post_yy.nii.gz`、`post_kj.nii.gz`
- `pattern_root/sub-xx/*_metadata.tsv`：每个 4D patterns 的 trial 元数据

### Step 5（必跑）：前后测 RSA（核心证据）+ LMM
```bash
cd ../rsa_analysis
python check_data_integrity.py
python run_rsa_optimized.py
python rsa_lmm.py INPUT.tsv OUT_DIR
```

产出（核心论文证据）：
- `rsa_results_optimized/`（或脚本配置的输出目录）：
  - `rsa_itemwise_details.csv`：item-wise 数据（LMM 输入）
  - `rsa_summary_stats.csv`：汇总统计
- `OUT_DIR/`：
  - `rsa_lmm_summary.txt` / `rsa_lmm_params.tsv`（具体以脚本输出为准）

（可选）跨 ROI 与模型 RDM：
```bash
python cross_roi_rsa.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR OUT_DIR
python model_rdm_comparison.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR OUT_DIR --numeric-model-cols novelty familiarity
python plot_rsa_results.py
```

产出：
- `OUT_DIR/cross_roi_rsa_*.tsv`（跨 ROI RDM 一致性）
- `OUT_DIR/model_rdm_*.tsv`（模型 RDM 比较结果）
- `plot_rsa_results.py` 输出的图表文件（png/svg/html，视脚本配置）

### Step 6（必跑）：RD / GPS（互补证据）
```bash
cd ../representation_analysis
python rd_analysis.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR ${PYTHON_METAPHOR_ROOT}/rd_results
python gps_analysis.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR ${PYTHON_METAPHOR_ROOT}/gps_results
```

产出：
- `rd_results/rd_subject_metrics.tsv`、`rd_results/rd_group_summary.tsv`（文件名以脚本为准）
- `gps_results/gps_subject_metrics.tsv`、`gps_results/gps_group_summary.tsv`（文件名以脚本为准）

### Step 7（必跑）：脑-行为关联
```bash
cd ../brain_behavior
python brain_behavior_correlation.py ${PYTHON_METAPHOR_ROOT}/brain_behavior_results METRIC1.tsv METRIC2.tsv METRIC3.tsv
```

产出：
- `brain_behavior_results/brain_behavior_merged.tsv`（合并后的 subject-level 指标表）
- `brain_behavior_results/brain_behavior_correlations.tsv`（两两相关结果，含 p 值）
- `brain_behavior_results/brain_behavior_summary.json`（元信息）

### Step 8（可选加分项）：机制与稳健性

学习阶段时间窗动态（建议先跑）：
```bash
cd ../temporal_dynamics
python learning_dynamics.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR OUT_DIR --window-size 20 --compute-mvpa
```

产出：
- `OUT_DIR/learning_dynamics_windows.tsv`（窗口级指标）
- `OUT_DIR/learning_dynamics_group_summary.tsv`（斜率/趋势的组水平结果）

gPPI（ROI-to-ROI，学习阶段）：
```bash
cd ../connectivity_analysis
python gPPI_analysis.py SEED_MASK.nii.gz ROI_DIR OUT_DIR
```

产出：
- `OUT_DIR/gppi_subject_metrics.tsv`、`OUT_DIR/gppi_group_summary.tsv`

有效连接（简化版）：
```bash
python effective_connectivity.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR OUT_DIR --time learn --condition yy
```

产出：
- `OUT_DIR/effective_connectivity_edges.tsv`（边：from→to，subject-level）
- `OUT_DIR/effective_connectivity_group_summary.tsv`（组水平 one-sample t）

偏侧化：
```bash
cd ../lateralization
python lateralization_analysis.py ${PYTHON_METAPHOR_ROOT}/pattern_root ROI_DIR OUT_DIR
```

产出：
- `OUT_DIR/lateralization_subject_metrics.tsv`、`OUT_DIR/lateralization_group_summary.tsv`

中介分析（subject-level 指标表）：
```bash
cd ../brain_behavior
python mediation_analysis.py ${PYTHON_METAPHOR_ROOT}/brain_behavior_results/brain_behavior_merged.tsv OUT_DIR --x-col X --m-col M --y-col Y
```

产出：
- `OUT_DIR/mediation_summary.json`、`OUT_DIR/mediation_summary.tsv`

BF10 与 bootstrap CI：
```bash
cd ../utils
python bayesian_stats.py INPUT.tsv OUT_DIR --mode paired --a-col post --b-col pre --group-col roi
python bootstrap_ci.py INPUT.tsv OUT_DIR --value-col value --group-cols roi time condition
```

产出：
- `OUT_DIR/bayesfactor_ttest.tsv`、`OUT_DIR/bayesfactor_meta.json`
- `OUT_DIR/bootstrap_mean_ci.tsv`、`OUT_DIR/bootstrap_mean_ci_meta.json`

## 使用约定（最小版）

1. 只要 `pattern_root/sub-xx/pre_yy.nii.gz` 等命名齐了，下游多数脚本无需额外配置即可跑通。
2. 默认主线只用 `yy/kj`；baseline 控制验证属于可选扩展，先把主线跑通再加。
3. 论文叙事优先“少而强”：先用 RSA（核心）+ RD/GPS（互补）+ GLM（定位）三件套讲清楚，再考虑 MVPA/连接/中介等加分项。
