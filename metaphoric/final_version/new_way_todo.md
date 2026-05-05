# new_way_todo.md

最后更新：2026-05-02

目标：把 `new_way.md` 中的 Relation-vector Geometry 路线转成具体可执行任务。本文档聚焦三件事：

- 写哪些脚本。
- 产出哪些结果文件、图和 QC。
- 如何验收结果，并决定进入主文、SI，还是降级为探索。

本路线不替代 `new_todo.md` 与 `new_rerun_list.md`。`new_todo.md` 解决既有主线可信度；本文档解决 PNAS 冲刺所需的新机制分析。

## 0. 总体执行原则

核心科学问题：

> 隐喻联结学习是否让人脑学习到 cue-to-target 的方向性关系几何，而不是简单让两个词变得更相似？

核心分析包：

| 阶段 | 脚本 | 作用 | 优先级 |
| --- | --- | --- | --- |
| A0 | `rsa_analysis/build_relation_vectors.py` | 构建 pair-level 关系向量与模型 RDM | P0 |
| A1 | `rsa_analysis/relation_vector_rsa.py` | ROI-level relation-vector Model-RSA | P0 |
| A2 | `figures/plot_relation_vector_rsa.py` | 主机制图与汇总图 | P0 |
| B1 | `brain_behavior/relation_behavior_prediction.py` | relation geometry 预测行为 | P1 |
| B2 | `figures/plot_relation_behavior_prediction.py` | 脑-行为图 | P1 |
| C1 | `temporal_dynamics/relation_learning_dynamics.py` | 学习阶段 relation trajectory | P2 |
| C2 | `figures/plot_relation_learning_dynamics.py` | 学习轨迹图 | P2 |
| D1 | `figures/update_relation_summary_tables.py` | 汇总主表、SI 表与 result_new 统计摘录 | P1 |

最小可行版本：

- 必须完成 A0 + A1 + A2。
- 若 A1 有方向性或校正后结果，再推进 B1/B2。
- C1/C2 是 PNAS 增强项，不应阻塞 A/B 的判断。

## 1. 通用环境与输入检查

### 1.1 环境设置

从 `E:\PythonAnalysis` 执行：

```powershell
cd E:\PythonAnalysis
$py = "C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe"
$env:PYTHON_METAPHOR_ROOT = "E:/python_metaphor"
$embedding = "E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv"
$patternRoot = "E:/python_metaphor/pattern_root"
$paperRoot = "E:/python_metaphor/paper_outputs"
```

### 1.2 必须存在的上游文件

验收命令：

```powershell
Test-Path $embedding
Test-Path "$patternRoot/sub-01/pre_yy.nii.gz"
Test-Path "$patternRoot/sub-01/pre_yy_metadata.tsv"
Test-Path "$paperRoot/qc/behavior_results/refined/behavior_trials.tsv"
Test-Path "$paperRoot/qc/behavior_results/refined/subject_summary_wide.tsv"
```

必须存在：

- `E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv`
- `E:/python_metaphor/pattern_root/sub-xx/{pre,post}_{yy,kj}.nii.gz`
- `E:/python_metaphor/pattern_root/sub-xx/{pre,post}_{yy,kj}_metadata.tsv`
- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv`
- `E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv`
- `E:/python_metaphor/roi_library/manifest.tsv`

### 1.3 输入覆盖 QC

新增 QC 表：

```text
paper_outputs/qc/relation_vectors/relation_embedding_coverage.tsv
paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv
```

验收标准：

- YY/KJ 正式 pair 的 word labels 在 embedding 表中覆盖率必须为 `100%`。
- 每个 condition 的 pair 数与 `stimuli_template.csv` 一致。
- 每个 pair 必须有且只有 2 个词。
- 若有缺失，先修 embedding 或 word label mapping，不进入 A1。

## 2. A0：构建关系向量与模型 RDM

### 2.1 新增脚本

路径：

```text
rsa_analysis/build_relation_vectors.py
```

职责：

- 读取 `stimuli_template.csv` 与 embedding 表。
- 按 `condition × pair_id` 构建 cue-target pair。
- 对每个 pair 计算关系向量。
- 生成 pair-level manifest、model RDM `.npz` 和长表。

### 2.2 输入

默认输入：

```text
E:/python_metaphor/stimuli_template.csv
E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv
```

命令参数：

```text
--stimuli-template
--embedding-file
--output-dir
--conditions yy kj
--word-col word_label
--pair-col pair_id
--cue-role first
```

说明：

- `--cue-role first` 表示同一 pair 内排序后的第一词为 cue，第二词为 target。
- 如果 metadata 中已有更明确的 cue/target 列，后续可改成 `--cue-col` / `--target-col`。

### 2.3 模型

必须输出：

- `M9_relation_vector_direct`：`target - cue`
- `M9_relation_vector_reverse`：`cue - target`
- `M9_relation_vector_abs`：`abs(target - cue)`
- `M9_relation_vector_length`：关系向量长度差异
- `M3_embedding_pair_distance`：pair 内普通语义距离或 pair centroids 距离

可选输出：

- `M9_relation_vector_direction_only`：单位向量方向，去掉长度。
- `M9_relation_vector_centroid_control`：pair centroid RDM，用于控制 pair 位置。

### 2.4 输出

```text
paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv
paper_outputs/qc/relation_vectors/relation_embedding_coverage.tsv
paper_outputs/qc/relation_vectors/relation_model_rdms.npz
paper_outputs/qc/relation_vectors/relation_model_rdm_manifest.tsv
paper_outputs/qc/relation_vectors/relation_model_collinearity.tsv
paper_outputs/qc/relation_vectors/relation_vectors_manifest.json
```

### 2.5 验收标准

- `relation_pair_manifest.tsv` 中每个 `condition × pair_id` 只有一行。
- `relation_model_rdms.npz` 中每个模型都是方阵，维度等于 pair 数。
- RDM 对角线为 0，矩阵对称，无 NaN。
- `relation_model_collinearity.tsv` 中标注 `abs_r > .70` 的模型对，后续解释时必须说明。
- direct 与 reverse 如果数值完全等价，不能声称方向性；需要改用 direction-sensitive neural vector 分析或把 reverse 降级为 sanity check。

### 2.6 运行命令

```powershell
& $py metaphoric/final_version/rsa_analysis/build_relation_vectors.py `
  --stimuli-template E:/python_metaphor/stimuli_template.csv `
  --embedding-file $embedding `
  --output-dir E:/python_metaphor/paper_outputs/qc/relation_vectors `
  --conditions yy kj
```

## 3. A1：Relation-vector ROI-level Model-RSA

### 3.1 新增脚本

路径：

```text
rsa_analysis/relation_vector_rsa.py
```

职责：

- 读取 pre/post patterns 与 ROI masks。
- 对每个 `subject × roi × condition × time` 计算 pair-level neural RDM。
- 与 A0 的 relation model RDM 做 Spearman 相关。
- 输出 subject-level rho、post-pre delta、group summary、FDR。

### 3.2 输入

默认输入：

```text
E:/python_metaphor/pattern_root
E:/python_metaphor/roi_library/manifest.tsv
E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz
E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv
```

命令参数：

```text
--pattern-root
--roi-set
--roi-manifest
--relation-rdm-npz
--relation-pair-manifest
--output-dir
--conditions yy kj
--models M9_relation_vector_direct M9_relation_vector_abs M9_relation_vector_length M3_embedding_pair_distance
--partial-correlation
--audit-rdms
```

### 3.3 核心算法

每个 `subject × roi × condition × time`：

1. 读取 `{time}_{condition}.nii.gz` 和 metadata。
2. ROI mask 提取 trial × voxel pattern。
3. 按 metadata 的 `pair_id` 把每个 pair 的两个词整理成 pair-level 表征。
4. 构建 neural RDM。
5. 与同 condition 的 model RDM 对齐后计算 Spearman rho。
6. 输出：

```text
rho_pre
rho_post
delta_rho = rho_post - rho_pre
```

### 3.4 Pair-level neural RDM 的两个版本

必须实现两个版本，并在主分析中选择一个，另一个作为 sensitivity：

#### 版本 1：Pair centroid RDM

```text
pair_pattern = mean(pattern_cue, pattern_target)
neural_RDM = pdist(pair_pattern, correlation)
```

优点：稳定、易解释。  
缺点：方向性较弱。

#### 版本 2：Neural relation vector RDM

```text
neural_relation_vector = pattern_target - pattern_cue
neural_RDM = pdist(neural_relation_vector, correlation)
```

优点：最贴合 relation-vector 理论。  
缺点：对 cue/target 顺序和噪声更敏感。

推荐：

- 主分析先用 `neural_relation_vector`。
- `pair_centroid` 作为控制，避免审稿人认为结果只是 pair 位置差异。

### 3.5 输出

ROI-set 级输出：

```text
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_subject_metrics.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_group_summary.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_group_summary_fdr.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_cell_qc.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_model_failures.tsv
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_manifest.json
```

全局论文输出：

```text
paper_outputs/tables_main/table_relation_vector_rsa.tsv
paper_outputs/tables_si/table_relation_vector_rsa_fdr.tsv
paper_outputs/tables_si/table_relation_vector_subject_metrics.tsv
```

可选 audit：

```text
paper_outputs/qc/relation_vector_rsa_<roi_set>/rdm_audit/sub-xx/<roi>_<time>_<condition>_rdm.npz
```

### 3.6 FDR family

必须写入以下列：

```text
model_role
q_bh_model_family
q_bh_primary_family
q_bh_model_role_family
```

模型角色：

- primary：`M9_relation_vector_direct`, `M9_relation_vector_abs`
- control：`M9_relation_vector_length`, `M3_embedding_pair_distance`
- secondary：reverse / direction-only / centroid-control 等

FDR family 建议：

- 同一 `roi_set × condition × neural_rdm_type × model_role` 内校正。
- 主文只使用 primary family。

### 3.7 运行命令

```powershell
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  & $py metaphoric/final_version/rsa_analysis/relation_vector_rsa.py `
    --pattern-root $patternRoot `
    --roi-set $roi `
    --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
    --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
    --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
    --output-dir "E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_$roi" `
    --conditions yy kj `
    --neural-rdm-types relation_vector pair_centroid
}
```

### 3.8 验收标准

数据完整性：

- `relation_vector_cell_qc.tsv` 中 failed cells 为 0，或明确列出失败原因。
- 每个 ROI set 的 subjects 与 Step 5C RSA 一致。
- 每个 `subject × roi × condition × time × neural_rdm_type` 都有相同 pair 数。

结果完整性：

- `relation_vector_subject_metrics.tsv` 包含 pre/post/delta。
- `relation_vector_group_summary_fdr.tsv` 包含所有 primary/control 模型。
- `table_relation_vector_rsa.tsv` 能直接用于主文表。

科学验收：

- 强结果：`M9_relation_vector_direct` 在至少一套 ROI set 通过 primary-family FDR，且方向为 post alignment 增强。
- 中结果：多个语义 ROI 未校正显著或 q < .10，可作为机制线索。
- 弱结果：无方向性或只有 control 模型显著，relation-vector 主机制不成立。

## 4. A2：Relation-vector 图表

### 4.1 新增脚本

路径：

```text
figures/plot_relation_vector_rsa.py
```

职责：

- 读取三套 ROI 的 `relation_vector_group_summary_fdr.tsv`。
- 输出主机制图。
- 生成主文表和简短结果摘要。

### 4.2 输出

```text
paper_outputs/figures_main/fig_relation_vector_rsa.png
paper_outputs/figures_main/table_relation_vector_rsa_top.tsv
paper_outputs/tables_main/table_relation_vector_rsa.tsv
paper_outputs/tables_si/table_relation_vector_rsa_full.tsv
```

### 4.3 图形内容

推荐三栏：

1. 关系向量模型示意：`cue -> target`，以及 direct / abs / length controls。
2. 三套 ROI set 的 primary model delta rho forest plot。
3. 关键 ROI 的 pre/post relation alignment 点图或雨云图。

### 4.4 验收标准

- 图中明确区分 `yy` 与 `kj`。
- 主模型和 control 模型颜色不同。
- 图注写清楚 neural RDM 类型：`target - cue neural relation vector` 或 `pair centroid`。
- 表中必须有 q 值列，不能只显示 p 值。

## 5. B1：Relation geometry 预测行为

### 5.1 新增脚本

路径：

```text
brain_behavior/relation_behavior_prediction.py
```

职责：

- 把 A1 的 subject-level 或 item/pair-level relation metrics 与 run7 行为连接。
- 做 subject-level 和 item-level 两套分析。
- 输出统计表、FDR、QC 和绘图输入。

### 5.2 输入

```text
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_subject_metrics.tsv
paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv
paper_outputs/qc/behavior_results/refined/behavior_trials.tsv
stimuli_template.csv
```

可选：

```text
paper_outputs/qc/relation_vector_rsa_<roi_set>/relation_vector_pair_metrics.tsv
```

如果 A1 暂不输出 pair-level metrics，B1 第一版只做 subject-level。

### 5.3 Subject-level 分析

构建：

```text
relation_alignment_pre
relation_alignment_post
relation_alignment_delta
yy_minus_kj_relation_alignment
```

行为：

```text
run7_memory_accuracy
run7_correct_retrieval_efficiency
run7_correct_retrieval_speed
yy_minus_kj_<behavior>
```

模型：

```text
yy_minus_kj_behavior ~ yy_minus_kj_relation_alignment
```

统计：

- Spearman r。
- permutation p。
- bootstrap CI。
- FDR within inference family。

### 5.4 Item-level 分析

若 A1 能输出 pair-level relation fit：

```text
memory ~ relation_fit_pre + relation_fit_post + relation_fit_delta + condition
```

或 GEE：

```text
memory ~ relation_fit_post_z * C(condition) + relation_fit_pre_z
groups = subject
family = binomial
```

RT：

```text
log_rt_correct ~ relation_fit_post_z * C(condition) + relation_fit_pre_z
groups = subject
family = Gaussian
```

### 5.5 输出

```text
paper_outputs/qc/relation_behavior/relation_behavior_subject_long.tsv
paper_outputs/qc/relation_behavior/relation_behavior_item_long.tsv
paper_outputs/qc/relation_behavior/relation_behavior_join_qc.tsv
paper_outputs/tables_main/table_relation_behavior_prediction.tsv
paper_outputs/tables_si/table_relation_behavior_prediction_fdr.tsv
paper_outputs/tables_si/table_relation_behavior_itemlevel.tsv
paper_outputs/qc/relation_behavior/relation_behavior_manifest.json
```

### 5.6 运行命令

```powershell
& $py metaphoric/final_version/brain_behavior/relation_behavior_prediction.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_main_functional `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature_spatial `
  --behavior-summary E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --paper-output-root $paperRoot
```

### 5.7 验收标准

QC：

- subject-level merge 后 `n_subjects >= 26`。
- item-level merge 后 subject、condition、pair 数与现有 item-level brain-behavior 接近。
- `relation_behavior_join_qc.tsv` 没有明显 condition 偏差。

科学验收：

- 强结果：primary relation metric 预测 `YY-KJ` memory 或 efficiency，q < .05。
- 中结果：语义 ROI 未校正显著且方向一致，可作为机制线索。
- 阴性：不写“关系几何解释行为”，只写神经机制结果。

## 6. B2：脑-行为图

### 6.1 新增脚本

路径：

```text
figures/plot_relation_behavior_prediction.py
```

### 6.2 输出

```text
paper_outputs/figures_main/fig_relation_behavior_prediction.png
paper_outputs/tables_main/table_relation_behavior_top.tsv
```

### 6.3 图形内容

推荐：

- Top 4 或 Top 6 relation-behavior associations。
- 每个 panel 显示 subject scatter + robust regression line。
- 颜色区分 ROI set 或 condition。
- 标注 permutation p 与 q。

### 6.4 验收标准

- 不展示未校正 p 很小但 q 很差的一堆散点作为主文。
- 若无 FDR 后结果，图降级为 SI 或不生成主文图。

## 7. C1：Learning-stage relation trajectory

### 7.1 新增脚本

路径：

```text
temporal_dynamics/relation_learning_dynamics.py
```

职责：

- 读取 learning-stage patterns。
- 按 exposure / run / early-late window 计算 relation alignment。
- 检验 trajectory 是否随学习增强，是否预测 post relation alignment 或 run7 behavior。

### 7.2 输入检查

先运行：

```powershell
Get-ChildItem E:/python_metaphor/pattern_root/sub-01 -Filter "learn_*"
```

若存在：

```text
learn_yy.nii.gz
learn_kj.nii.gz
learn_yy_metadata.tsv
learn_kj_metadata.tsv
```

则进入 C1。

若不存在：

- C1 暂缓。
- 不影响 A/B。

### 7.3 时间窗定义

优先顺序：

1. metadata 有 `exposure_index`：按 exposure 做 trajectory。
2. metadata 有 `run`：run3 vs run4。
3. metadata 有 trial order：early half vs late half。
4. 都没有：C1 不做。

### 7.4 输出

```text
paper_outputs/qc/relation_learning_dynamics/relation_learning_dynamics_long.tsv
paper_outputs/qc/relation_learning_dynamics/relation_learning_window_qc.tsv
paper_outputs/tables_main/table_relation_learning_dynamics.tsv
paper_outputs/tables_si/table_relation_learning_dynamics_subject.tsv
paper_outputs/qc/relation_learning_dynamics/relation_learning_manifest.json
```

### 7.5 运行命令

```powershell
& $py metaphoric/final_version/temporal_dynamics/relation_learning_dynamics.py `
  --pattern-root $patternRoot `
  --roi-sets main_functional literature `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
  --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
  --paper-output-root $paperRoot
```

### 7.6 验收标准

- 每个 window 至少有足够 pair 数，建议 `n_pairs >= 10`。
- 每个 subject 至少两个 window。
- 主模型优先检验 `YY` relation alignment slope。
- 若 trajectory 预测行为，必须单独列为 planned exploratory，避免过度因果化。

## 8. C2：学习轨迹图

### 8.1 新增脚本

路径：

```text
figures/plot_relation_learning_dynamics.py
```

### 8.2 输出

```text
paper_outputs/figures_main/fig_relation_learning_dynamics.png
paper_outputs/tables_main/table_relation_learning_top.tsv
```

### 8.3 图形内容

- YY/KJ early-late 或 exposure trajectory。
- 关键 ROI / ROI family 的 relation alignment 曲线。
- 可选：trajectory slope 与 Run7 memory scatter。

### 8.4 验收标准

- 若学习阶段 metadata 不足，不强行作图。
- 若只有趋势，作为 SI 过程图。

## 9. D1：汇总与写作更新

### 9.1 新增脚本

路径：

```text
figures/update_relation_summary_tables.py
```

职责：

- 从 A/B/C 输出中提取 top results。
- 生成一页结果概览所需表格。
- 不自动改 `result_new.md`，只输出可复制的 markdown 摘要，避免误写结论。

### 9.2 输出

```text
paper_outputs/qc/relation_summary/relation_result_update_stats.txt
paper_outputs/qc/relation_summary/relation_result_markdown_snippets.md
paper_outputs/tables_main/table_relation_summary_onepage.tsv
```

### 9.3 验收标准

- 摘要中必须区分：
  - confirmed
  - exploratory
  - negative
- 每条结果必须带：
  - ROI set
  - model
  - condition
  - neural RDM type
  - subject 数
  - correction family
  - q value

## 10. 全链路推荐运行顺序

```powershell
# 0. 环境
cd E:\PythonAnalysis
$py = "C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe"
$env:PYTHON_METAPHOR_ROOT = "E:/python_metaphor"
$embedding = "E:/python_metaphor/stimulus_embeddings/stimulus_embeddings_bert.tsv"
$patternRoot = "E:/python_metaphor/pattern_root"
$paperRoot = "E:/python_metaphor/paper_outputs"

# 1. A0: relation vectors
& $py metaphoric/final_version/rsa_analysis/build_relation_vectors.py `
  --stimuli-template E:/python_metaphor/stimuli_template.csv `
  --embedding-file $embedding `
  --output-dir E:/python_metaphor/paper_outputs/qc/relation_vectors `
  --conditions yy kj

# 2. A1: relation-vector RSA
foreach ($roi in @("main_functional", "literature", "literature_spatial")) {
  & $py metaphoric/final_version/rsa_analysis/relation_vector_rsa.py `
    --pattern-root $patternRoot `
    --roi-set $roi `
    --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
    --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
    --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
    --output-dir "E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_$roi" `
    --conditions yy kj `
    --neural-rdm-types relation_vector pair_centroid
}

# 3. A2: relation-vector figures
& $py metaphoric/final_version/figures/plot_relation_vector_rsa.py `
  --paper-output-root $paperRoot

# 4. B1/B2: behavior bridge
& $py metaphoric/final_version/brain_behavior/relation_behavior_prediction.py `
  --relation-dirs `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_main_functional `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature `
    E:/python_metaphor/paper_outputs/qc/relation_vector_rsa_literature_spatial `
  --behavior-summary E:/python_metaphor/paper_outputs/qc/behavior_results/refined/subject_summary_wide.tsv `
  --behavior-trials E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv `
  --paper-output-root $paperRoot

& $py metaphoric/final_version/figures/plot_relation_behavior_prediction.py `
  --paper-output-root $paperRoot

# 5. C1/C2: learning trajectory, only if learning patterns exist
& $py metaphoric/final_version/temporal_dynamics/relation_learning_dynamics.py `
  --pattern-root $patternRoot `
  --roi-sets main_functional literature `
  --roi-manifest E:/python_metaphor/roi_library/manifest.tsv `
  --relation-rdm-npz E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_model_rdms.npz `
  --relation-pair-manifest E:/python_metaphor/paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv `
  --paper-output-root $paperRoot

& $py metaphoric/final_version/figures/plot_relation_learning_dynamics.py `
  --paper-output-root $paperRoot

# 6. D1: summary snippets
& $py metaphoric/final_version/figures/update_relation_summary_tables.py `
  --paper-output-root $paperRoot
```

## 11. 分阶段验收门槛

### Gate A：是否值得继续做 B/C？

A1 完成后判断：

- Go：`M9_relation_vector_direct` 或 `M9_relation_vector_abs` 在至少一套 ROI set 中有 q < .05，或多个理论 ROI q < .10 且方向一致。
- Conditional Go：只有未校正 p < .05，但集中在 IFG/ATL/AG/pMTG，可做 B，但写作保持探索。
- Stop：primary 模型无方向，control 模型更强，或结果完全随机。转向 schema / SSDD 模型。

### Gate B：是否能写脑-行为桥接？

B1 完成后判断：

- 主文：relation metric 预测 `YY-KJ` memory / efficiency，q < .05。
- SI：未校正显著或 q < .10。
- 不写：无方向或由单个离群点驱动。

### Gate C：是否能写学习过程？

C1 完成后判断：

- 主文：YY relation alignment 随学习增强，且与 post relation alignment 或 behavior 有联系。
- SI：只有趋势或只在单 ROI。
- 暂缓：learning metadata 不足或 pair 数不足。

## 12. 论文结果解释模板

### A 强 + B 强 + C 中/强

PNAS 冲刺主张：

> 隐喻联结学习通过形成方向性关系几何来重塑语义记忆，该关系几何在学习过程中逐步形成，并解释个体最终记忆表现。

### A 强 + B 弱

保守主张：

> 隐喻学习诱发稳定的表征分化，并伴随方向性关系几何的神经机制证据；行为桥接仍是边界条件。

### A 弱

回退主张：

> 当前主效应仍支持 representational differentiation，但 relation-vector 模型不足以解释该效应。下一步需要 schema 或语义维度模型。

## 13. 写作注意事项

- 不写“pair 学习后更像”。
- 不把 `relation alignment` 写成因果机制，除非 learning trajectory 和行为预测都支持。
- 不把未校正的单 ROI 结果写成主机制。
- 若 direct 与 reverse 模型等价，不写方向性，只写 relation geometry。
- 若 pair centroid 比 neural relation vector 更强，解释为 pair-level semantic location，而不是 direction coding。
- 若 KJ 也有强 relation-vector 结果，应改写为“关系学习通用机制”，再讨论 YY 的语义网络特异性。

## 14. 给老师汇报的执行版摘要

可以用下面这段作为汇报开头：

> 当前 Step 5C 已经证明隐喻学习引发稳定表征重组，但如果目标是 PNAS，我们需要把“similarity 下降”升级为可解释的计算机制。下一步我建议新增 relation-vector geometry 分析：用 embedding 中的 target-cue 向量定义每个 pair 的关系方向，检验人脑学习后是否编码这种关系几何。执行上先做 A0/A1/A2 三个脚本，若 relation-vector Model-RSA 成立，再做行为预测和学习阶段轨迹。这样可以把论文从“隐喻学习改变表征相似性”推进到“创造性关系学习重塑语义记忆的神经几何”。

## 15. 2026-05-02 执行记录：A/B 阶段

### 15.1 已新增或修订脚本

- `rsa_analysis/build_relation_vectors.py`：已完成 A0，生成 condition-specific relation-vector model RDM。
- `rsa_analysis/relation_vector_rsa.py`：已完成 A1，并做性能修补；现在每个 subject/condition/time 的 4D pattern 只读取一次，再套不同 ROI mask。
- `figures/plot_relation_vector_rsa.py`：已完成 A2，生成 relation-vector 主机制图、top 表与 full SI 表。
- `brain_behavior/relation_behavior_prediction.py`：已完成 B1 subject-level 版本；同时检验 `relation_pre`、`relation_post`、`relation_delta`、`relation_decoupling`，避免只围绕 post-pre 下降解释。
- `figures/plot_relation_behavior_prediction.py`：已完成 B2 候选脑-行为展示图。

### 15.2 已生成核心输出

- `paper_outputs/qc/relation_vectors/relation_model_rdms.npz`
- `paper_outputs/qc/relation_vectors/relation_pair_manifest.tsv`
- `paper_outputs/qc/relation_vectors/relation_model_collinearity.tsv`
- `paper_outputs/qc/relation_vector_rsa_main_functional/relation_vector_group_summary_fdr.tsv`
- `paper_outputs/qc/relation_vector_rsa_literature/relation_vector_group_summary_fdr.tsv`
- `paper_outputs/qc/relation_vector_rsa_literature_spatial/relation_vector_group_summary_fdr.tsv`
- `paper_outputs/tables_main/table_relation_vector_rsa.tsv`
- `paper_outputs/tables_si/table_relation_vector_rsa_full.tsv`
- `paper_outputs/figures_main/fig_relation_vector_rsa.png`
- `paper_outputs/qc/relation_behavior/relation_behavior_subject_long.tsv`
- `paper_outputs/tables_main/table_relation_behavior_prediction.tsv`
- `paper_outputs/tables_si/table_relation_behavior_prediction_fdr.tsv`
- `paper_outputs/figures_main/fig_relation_behavior_prediction.png`

### 15.3 A1/A2 科学验收结论

- QC 通过：三套 ROI set 均无 failed cell；`main_functional` 784/784 ok，`literature` 1344/1344 ok，`literature_spatial` 1344/1344 ok。
- relation-vector primary 模型出现多处 primary-family FDR 显著结果：
  - `literature/lit_L_pMTG/KJ/M9_relation_vector_direct`：post-pre = -0.0504，p = 2.58e-05，q = 0.00040。
  - `literature/lit_R_temporal_pole/KJ/M9_relation_vector_abs`：post-pre = -0.0432，p = 3.30e-05，q = 0.00040。
  - `literature_spatial/litspat_R_OPA/YY/M9_relation_vector_direct`：post-pre = -0.0544，p = 0.00019，q = 0.00458。
  - `literature/lit_L_temporal_pole/YY/M9_relation_vector_direct`：post-pre = -0.0395，p = 0.00038，q = 0.00669。
- 方向解释需要调整：最强效应多数是负向 post-pre，不应写成“学习后更贴近通用 embedding relation-vector”。更稳妥的机制表述是：学习后相关 ROI 的 neural relation geometry 从通用语义 embedding 的关系结构中脱耦，可能形成更任务化、经验化的关系表征。

### 15.4 B1/B2 脑-行为验收结论

- subject-level merge 通过但行为表只有 27 名可用被试；relation 指标为 28 名，被试合并后为 27 名。
- primary brain-behavior 没有严格通过 q < .10；最接近结果为：
  - `literature_spatial/litspat_R_OPA/YY/M9_relation_vector_direct/relation_post -> run7_memory_accuracy`：Spearman rho = 0.622，p = 0.000529，primary-family q = 0.102。
- 因此 B 阶段目前定位为候选行为桥接线索，不应作为主结论；主结论仍应放在 relation-vector Model-RSA 的 neural mechanism。

### 15.5 下一轮建议

- 优先推进 C1/C2：检查 relation-vector effect 是否沿学习阶段或 run-level 呈现轨迹，这比继续扩展 subject-level 脑-行为更可能增强 PNAS 叙事。
- 若要加强 B 阶段，应新增 pair-level relation fit 输出，再做 item-level memory/RT 的 mixed model 或 GEE；仅 subject-level n=27 的相关分析很难承载强主张。
- 写作上把机制命名为 `relation-vector realignment/decoupling`，并明确它不是简单的 similarity decrease，而是从 item-pair similarity 下降推进到 relation geometry 的重组。

## 16. 2026-05-02 执行记录：C 阶段

### 16.1 已新增脚本

- `temporal_dynamics/relation_learning_dynamics.py`：完成 C1。由于 learning-stage patterns 是 pair/item-level 图像，不是 cue/target 分离的 word-pattern，该脚本检验 run3 与 run4 的 pair-pattern RDM 是否贴近 relation-vector model RDM，并统计 run4 - run3。
- `figures/plot_relation_learning_dynamics.py`：完成 C2。输出 learning trajectory forest plot、run3/run4 subject-level alignment 图和 top 表。

### 16.2 已生成核心输出

- `paper_outputs/qc/relation_learning_dynamics/relation_learning_dynamics_long.tsv`
- `paper_outputs/qc/relation_learning_dynamics/relation_learning_window_qc.tsv`
- `paper_outputs/qc/relation_learning_dynamics/relation_learning_group_summary_fdr.tsv`
- `paper_outputs/qc/relation_learning_dynamics/relation_learning_failures.tsv`
- `paper_outputs/qc/relation_learning_dynamics/relation_learning_manifest.json`
- `paper_outputs/tables_main/table_relation_learning_dynamics.tsv`
- `paper_outputs/tables_si/table_relation_learning_dynamics_subject.tsv`
- `paper_outputs/figures_main/fig_relation_learning_dynamics.png`
- `paper_outputs/figures_main/table_relation_learning_top.tsv`

### 16.3 C1/C2 验收结论

- QC 通过：3472/3472 window cells 为 ok，failure 文件为空。
- 输出规模：17360 条 subject/run/model metric rows，310 条 group summary rows。
- primary relation-vector learning trajectory 没有通过 primary-family FDR；当前最佳候选为：
  - `literature_spatial/litspat_L_RSC/YY/M9_relation_vector_direct`：run4-run3 = 0.0296，p = 0.0567，q = 0.537。
  - `literature_spatial/litspat_L_hippocampus/YY/M9_relation_vector_abs`：run4-run3 = -0.0267，p = 0.0682，q = 0.537。
- 因此 C 阶段不能支持“relation geometry 在学习过程中逐步形成”的强主张。它更适合作为边界结果：强 neural mechanism 出现在 pre/post 表征重组中，但 learning run3/run4 的 pair-pattern RDM 没有稳定捕捉到同一轨迹。

### 16.4 写作与下一步判断

- 不建议把 C 阶段放入主文核心链条；可放 SI，说明 learning-stage pattern 的 run3/run4 trajectory 未提供额外支持。
- PNAS 冲刺主线应继续围绕 A 阶段的 relation-vector realignment/decoupling，以及 Step 5C 原有稳健 RSA 效应。
- 若必须增强时间过程证据，下一步不是继续在当前 run3/run4 pair-pattern 上挖，而是检查是否能从 trial-level 或 exposure-order 构造更细粒度的 item-level relation fit；否则容易变成弱结果堆叠。

## 17. 2026-05-04 执行记录：meta ROI run7 retrieval 阶段

### 17.1 数据状态

- run7 LSS 已生成独立索引：`E:/python_metaphor/lss_betas_final/lss_metadata_index_run7.csv`。
- 已用 `representation_analysis/stack_patterns.py` 生成 `pattern_root/sub-xx/retrieval_yy.nii.gz` 与 `retrieval_kj.nii.gz`。
- run7 神经 pattern 可用被试为 26 名：`sub-01` 至 `sub-28` 中缺 `sub-12`、`sub-23`；其中 `sub-12` 本身不在行为主汇总中，真正行为有而神经缺的是 `sub-23`。
- 后续 run7 neural-behavior 分析均使用 neural × behavior 交集，即 26 名被试。

### 17.2 新增脚本

- `rsa_analysis/retrieval_pair_similarity.py`：在 `meta_metaphor` / `meta_spatial` 中计算 run7 retrieval pair similarity、pre/post/retrieval trajectory，以及 post-to-retrieval same-vs-different pair/word reinstatement。
- `brain_behavior/retrieval_success_prediction.py`：把 retrieval geometry item/pair 指标与 run7 memory / correct-trial RT 做 item-level GEE，显式拆分 within-subject 与 between-subject 成分。

### 17.3 核心输出

- `paper_outputs/qc/retrieval_geometry/retrieval_item_metrics.tsv`
- `paper_outputs/qc/retrieval_geometry/retrieval_pair_metrics.tsv`
- `paper_outputs/qc/retrieval_geometry/retrieval_subject_metrics.tsv`
- `paper_outputs/qc/retrieval_geometry/retrieval_geometry_group_fdr.tsv`
- `paper_outputs/tables_si/table_retrieval_geometry.tsv`
- `paper_outputs/qc/retrieval_success_prediction/retrieval_success_item_long.tsv`
- `paper_outputs/qc/retrieval_success_prediction/retrieval_success_group_fdr.tsv`
- `paper_outputs/tables_main/table_retrieval_success_prediction.tsv`
- `paper_outputs/tables_si/table_retrieval_success_prediction_fdr.tsv`

### 17.4 QC

- retrieval geometry：26 名被试，0 failure；item rows = 65,520，pair rows = 32,760，subject metric rows = 9,360。
- retrieval success prediction：long rows = 720,720，model rows = 792。
- intersection QC：behavior summary 有 27 名被试；run7 neural metrics 有 26 名；缺失 neural 的行为被试为 `sub-23`。

### 17.5 科学验收结论

run7 retrieval geometry 提供了新的 P1 补充证据：YY 相对 KJ 的 `retrieval - post pair similarity` 在若干 meta ROI 中显著为正，提示 post 阶段分化后的 pair structure 在最终 retrieval 阶段有相对更强的 rebound/reinstatement。

最强 planned 结果：

| ROI set | ROI | Metric | YY-KJ effect | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| `meta_spatial` | `meta_R_PPA_PHG` | retrieval - post pair similarity | 0.083 | 4.76e-07 | 1.91e-05 |
| `meta_spatial` | `meta_R_PPC_SPL` | retrieval - post pair similarity | 0.085 | 4.21e-05 | 0.000842 |
| `meta_metaphor` | `meta_R_temporal_pole` | retrieval - post pair similarity | 0.104 | 0.000291 | 0.00466 |
| `meta_metaphor` | `meta_R_IFG` | retrieval - post pair similarity | 0.084 | 0.000185 | 0.00466 |
| `meta_spatial` | `meta_R_hippocampus` | retrieval - post pair similarity | 0.048 | 0.00310 | 0.0413 |

post-to-retrieval same-vs-different pair reinstatement 的校正后证据较弱：`meta_R_hippocampus` YY p = 0.0207, primary q = 0.061；`meta_L_hippocampus` YY p = 0.0431, primary q = 0.119。可作为 hippocampal retrieval reinstatement 趋势，不作为主结论。

run7 retrieval success prediction 有局部桥接，但方向不够单一：

- `meta_L_pMTG_pSTS`, KJ, retrieval pair similarity -> memory: beta = 0.110, p = 0.000755, q_primary = 0.0362。
- `meta_R_AG`, YY, retrieval pair similarity -> memory: beta = 0.159, p = 0.00217, q_primary = 0.0521。
- `meta_L_PPA_PHG`, YY, retrieval pair similarity -> log RT correct: beta = 0.0164, p = 7.52e-05, q_primary = 0.00451。

RT 模型中 beta 为正表示反应更慢，因此不能写成“retrieval geometry 提高提取效率”。run7 行为桥接应定位为探索性。

### 17.6 写作建议

run7 结果适合放在主机制后的补充段落：

> learned-edge differentiation 在 post 阶段表现为 trained-pair similarity 下降；到最终 retrieval 阶段，YY 相对 KJ 在右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG 和右海马出现 pair-structure rebound。这说明学习后的关系边并非简单消失，而是在最终提取阶段以任务相关方式重新出现。

但不建议写成强行为机制：

> retrieval geometry 与 memory/RT 有局部关联，但尚不足以支持“更强 reinstatement 直接导致更好或更快记忆”的主结论。

## 18. 2026-05-04 执行记录：run7 MVPA 分类检验

### 18.1 新增脚本

- `fmri_mvpa/fmri_mvpa_roi/run7_meta_roi_decoding.py`

脚本使用 `meta_metaphor` / `meta_spatial` 中 5 个 run7 rebound ROI：

- `meta_R_PPA_PHG`
- `meta_R_PPC_SPL`
- `meta_R_temporal_pole`
- `meta_R_IFG`
- `meta_R_hippocampus`

### 18.2 分析方案

1. 主方案：pair-held-out YY/KJ decoding。每个 fold 留出同序号的一个 YY pair 和一个 KJ pair，训练其余 pair，检验 balanced accuracy 是否高于 0.5。
2. 用户设想方案：cross-role generalization，分别检验 `w -> ew` 与 `ew -> w`。
3. 诊断方案：`w-only` 与 `ew-only` 内部 pair-held-out decoding，用于判断 cross-role 失败是否因为分类信息依赖词角色。
4. 行为桥接：pair-held-out classifier evidence 预测 run7 memory 与 correct-trial log RT。

代码 review 中发现原始 `pair_id` 不能直接用于 YY/KJ 成对留出，因为 YY 的 `pair_id` 为 1-35，KJ 为 36-70；如果按单个 `pair_id` 留出，test fold 只有一个类别，主分析会被跳过。因此脚本已新增 `condition_pair_index`，按条件内排序后的 pair 序号成对留出。

### 18.3 输出与 QC

- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_manifest.json`
- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_subject_metrics.tsv`
- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_fold_metrics.tsv`
- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_trial_evidence.tsv`
- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_group_fdr.tsv`
- `paper_outputs/qc/run7_mvpa_decoding/run7_mvpa_evidence_behavior_fdr.tsv`
- `paper_outputs/tables_main/table_run7_mvpa_decoding.tsv`
- `paper_outputs/tables_main/table_run7_mvpa_evidence_behavior.tsv`

QC：

- 被试：26 名。
- failures：0。
- subject metric rows：650。
- trial evidence rows：18,200。
- 每个 subject x ROI：140 trials；YY/KJ 各 70 trials；`yyw`, `yyew`, `kjw`, `kjew` 各 35 trials。

### 18.4 验收结论

主 pair-held-out decoding 支持 run7 中存在 YY/KJ 可解码信息：

| ROI | balanced accuracy | p | q_primary |
| --- | ---: | ---: | ---: |
| `meta_R_hippocampus` | 0.565 | 2.98e-06 | 3.04e-05 |
| `meta_R_temporal_pole` | 0.553 | 4.05e-06 | 3.04e-05 |
| `meta_R_PPA_PHG` | 0.564 | 1.72e-05 | 8.58e-05 |
| `meta_R_PPC_SPL` | 0.547 | 0.000190 | 0.000711 |
| `meta_R_IFG` | 0.513 | 0.187 | 0.357 |

cross-role generalization 不成立，所有 ROI 均未通过 FDR。最接近的是 `meta_R_PPA_PHG`：

- `w -> ew`: balanced accuracy = 0.518, p = 0.0671, q_primary = 0.201。
- `ew -> w`: balanced accuracy = 0.520, p = 0.126, q_primary = 0.314。

诊断性 role-specific decoding 显示分类信息在 `w` 和 `ew` 内部均存在，尤其右 PPA/PHG、右海马、右颞极、右 PPC/SPL 均显著。这说明 MVPA 结果支持的是 role-dependent YY/KJ discriminative geometry，而不是角色不变的抽象 YY/KJ code。

classifier evidence 的行为桥接未通过 FDR：

- memory 最小 q = 0.437。
- correct-trial log RT 最小 q = 0.750。

因此当前应停止继续扩展“classifier evidence 预测记忆/RT”的链条；MVPA 可作为 run7 retrieval-stage 补充证据，但不作为行为机制主证据。

## 19. 2026-05-04 执行记录：learning-stage MVPA 分类检验

### 19.1 新增脚本

- `fmri_mvpa/fmri_mvpa_roi/learning_meta_roi_decoding.py`

使用与 run7 MVPA 相同的 5 个计划 ROI：

- `meta_R_PPA_PHG`
- `meta_R_PPC_SPL`
- `meta_R_temporal_pole`
- `meta_R_IFG`
- `meta_R_hippocampus`

### 19.2 分析方案

学习阶段 pattern 是 pair/item-level pattern，不包含 run7 的 `w/ew` 单词角色结构。因此本轮不做 cross-role decoding，而做：

1. `run3_pair_heldout`：run3 内部 YY/KJ pair-held-out decoding。
2. `run4_pair_heldout`：run4 内部 YY/KJ pair-held-out decoding。
3. `cross_run_3_to_4` 与 `cross_run_4_to_3`：检验分类边界是否跨学习 run 泛化。
4. `run4_minus_run3_pair_heldout`：检验分类准确率是否从 run3 到 run4 增强。

### 19.3 输出与 QC

- `paper_outputs/qc/learning_mvpa_decoding/learning_mvpa_manifest.json`
- `paper_outputs/qc/learning_mvpa_decoding/learning_mvpa_subject_metrics.tsv`
- `paper_outputs/qc/learning_mvpa_decoding/learning_mvpa_fold_metrics.tsv`
- `paper_outputs/qc/learning_mvpa_decoding/learning_mvpa_group_fdr.tsv`
- `paper_outputs/qc/learning_mvpa_decoding/learning_mvpa_run4_minus_run3_fdr.tsv`
- `paper_outputs/tables_main/table_learning_mvpa_decoding.tsv`
- `paper_outputs/tables_main/table_learning_mvpa_run4_minus_run3.tsv`

QC：

- 被试：28 名。
- failures：0。
- subject metric rows：560。
- fold rows：8,400。
- 每个 subject x ROI：140 learning trials；run3/run4 各 70 trials；YY/KJ 各 70 trials。
- run3/run4 中可成对留出的 YY/KJ common `pair_id` 均为 30 个。

### 19.4 验收结论

within-run pair-held-out decoding 很强，run3 和 run4 中 5 个 ROI 均显著高于 chance。

run3：

| ROI | balanced accuracy | p | q_primary |
| --- | ---: | ---: | ---: |
| `meta_R_PPA_PHG` | 0.591 | 1.81e-06 | 3.63e-05 |
| `meta_R_hippocampus` | 0.567 | 1.72e-05 | 0.000172 |
| `meta_R_temporal_pole` | 0.563 | 0.000170 | 0.000679 |
| `meta_R_PPC_SPL` | 0.545 | 0.00403 | 0.00896 |
| `meta_R_IFG` | 0.532 | 0.00770 | 0.0154 |

run4：

| ROI | balanced accuracy | p | q_primary |
| --- | ---: | ---: | ---: |
| `meta_R_temporal_pole` | 0.571 | 5.74e-05 | 0.000382 |
| `meta_R_PPA_PHG` | 0.575 | 8.89e-05 | 0.000444 |
| `meta_R_PPC_SPL` | 0.573 | 0.000600 | 0.00165 |
| `meta_R_IFG` | 0.568 | 0.000591 | 0.00165 |
| `meta_R_hippocampus` | 0.570 | 0.000661 | 0.00165 |

但两个增强型结论不成立：

- `run4 - run3` 没有任何 ROI 通过 FDR；最接近 `meta_R_IFG`，delta = 0.036, p = 0.088, q = 0.441。
- cross-run generalization 没有任何 ROI 通过 FDR；最接近 `meta_R_PPA_PHG` 的 `run3 -> run4`，accuracy = 0.515, p = 0.0404, q_primary = 0.0735。

写作建议：learning-stage MVPA 可以作为 run7 MVPA 的前置补充，说明这几个 ROI 在学习阶段已经有 YY/KJ 条件区分信息；但不能写成“分类边界随学习稳定增强”或“跨 run 稳定泛化”。主故事仍应放在 post 的 edge-specific RSA 和 run7 的 retrieval-stage rebound。

## 20. 2026-05-05 执行记录：pre/post YY-KJ 二分类 MVPA

### 20.1 新增脚本

- `fmri_mvpa/fmri_mvpa_roi/prepost_meta_roi_binary_decoding.py`

### 20.2 数据与分析方案

只保留 YY 和 KJ，围绕“pre/post 完整词集内是否可分类”来做。metadata 显示真实 run 编码为：

- pre：run1/2。
- post：run5/6。
- run3/4 是 learning 阶段，不是 post。

pre 的 run1+run2、post 的 run5+run6 合起来才覆盖完整 YY/KJ 词集，因此主报告口径使用 phase-combined pair-held-out：

- pre：合并 run1+run2，每折同时留出一个 YY pair 和一个 KJ pair。
- post：合并 run5+run6，每折同时留出一个 YY pair 和一个 KJ pair。
- chance = 0.5。
- 每个 subject x ROI x phase 有 35 个 fold。

### 20.3 输出与 QC

- `paper_outputs/qc/prepost_binary_mvpa_decoding/prepost_binary_mvpa_manifest.json`
- `paper_outputs/qc/prepost_binary_mvpa_decoding/prepost_binary_mvpa_subject_metrics.tsv`
- `paper_outputs/qc/prepost_binary_mvpa_decoding/prepost_binary_mvpa_direction_metrics.tsv`
- `paper_outputs/qc/prepost_binary_mvpa_decoding/prepost_binary_mvpa_group_fdr.tsv`
- `paper_outputs/qc/prepost_binary_mvpa_decoding/prepost_binary_mvpa_post_minus_pre_fdr.tsv`
- `paper_outputs/tables_main/table_prepost_binary_mvpa_decoding.tsv`
- `paper_outputs/tables_main/table_prepost_binary_mvpa_post_minus_pre.tsv`

QC：

- 被试：28 名。
- failures：0。
- subject metric rows：840。
- direction/fold metric rows：10,360。
- 每个 subject x ROI：280 trials；pre = 140，post = 140；每个阶段 YY = 70，KJ = 70。
- 每个 phase 有 35 个 common condition-pair indices。

### 20.4 验收结论

pre 阶段整体较弱，只有右颞极未校正显著，校正后不稳：

| ROI | pre combined pair-held-out accuracy | p | q_primary |
| --- | ---: | ---: | ---: |
| `meta_R_temporal_pole` | 0.520 | 0.0393 | 0.0983 |
| `meta_R_PPA_PHG` | 0.514 | 0.244 | 0.443 |
| `meta_R_PPC_SPL` | 0.511 | 0.307 | 0.453 |
| `meta_R_hippocampus` | 0.510 | 0.359 | 0.453 |
| `meta_R_IFG` | 0.504 | 0.571 | 0.635 |

post 阶段出现更清楚的正向分类，右颞极通过 primary-family FDR，其他 ROI 多数为未校正显著或趋势：

| ROI | post combined pair-held-out accuracy | p | q_primary |
| --- | ---: | ---: | ---: |
| `meta_R_temporal_pole` | 0.539 | 0.00254 | 0.0354 |
| `meta_R_hippocampus` | 0.525 | 0.0162 | 0.0983 |
| `meta_R_PPC_SPL` | 0.516 | 0.0306 | 0.0983 |
| `meta_R_IFG` | 0.520 | 0.0377 | 0.0983 |
| `meta_R_PPA_PHG` | 0.520 | 0.0453 | 0.101 |

写作建议：这部分不再要求 post-pre 增强显著。它的作用是把 MVPA 动态串起来：pre 弱，learning 强，post 部分保留，run7 retrieval 再次稳健。post 的右颞极结果说明静态词项阶段已有一部分 YY/KJ condition information；主机制证据仍由 post edge-specific RSA 和 run7 retrieval rebound 承担。
