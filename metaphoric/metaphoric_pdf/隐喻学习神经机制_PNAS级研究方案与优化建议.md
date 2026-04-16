# 隐喻学习的多层次神经表征重组：面向PNAS的完整研究方案

> **选定思路**：研究思路二——从激活到模式到几何结构的多层次表征重组
>
> 本文档是最终研究执行方案。每步分析均标注：已有脚本路径 / 可移植的gjxx脚本 / 需新建的脚本。

***

## 论文核心叙事

### 建议标题

> **Multi-level neural reorganization during metaphor concept learning: From univariate activation to representational geometry**

### 一句话故事

隐喻联结学习不仅改变了脑区的激活强度，更深层地重组了多变量编码模式和表征几何结构；这种多层次重组在空间联结学习中显著减弱，而在不参与学习的基线条件中不存在——证明是学习驱动的而非重复暴露效应。这种重组跨层次地预测了行为学习效果。

### 叙事逻辑链

```
行为差异 → L1激活定位 → L2多变量解码 → L3表征几何
    ↓             ↓              ↓             ↓
记忆优势    GLM条件效应    MVPA可解码性    RSA/RD/GPS重组
    └─────────────── 脑-行为关联 ──────────────┘
                         ↑
              baseline控制：排除重测效应
```

***

## 实验设计

| Run     | 阶段              | 分析用途                        |
| ------- | --------------- | --------------------------- |
| Run 1-2 | 前测 (Pre-test)   | RSA/RD/GPS 基线 + 前测MVPA      |
| Run 3-4 | 学习阶段 (Learning) | GLM / LSS / MVPA / PPI / RSA |
| Run 5-6 | 后测 (Post-test)  | RSA/RD/GPS 后测 + 后测MVPA      |
| Run 7   | 记忆测试            | 行为数据                        |

- 被试：28人（sub-01至sub-28），被试排除标准需统一（见下方问题清单）
- **条件：3个水平**——yy（隐喻联结）、kj（空间联结）、baseline（不联结基线）
- **材料**：40隐喻词对(80词) + 40空间词对(80词) + 20基线词对(40词) + 干扰假词(80词)
- 核心对比：3（条件）× 2（时间：pre vs post）被试内设计
- **注意**：baseline条件**不参与学习（Run3-4）和回忆（Run7）**，仅出现在前后测（Run1-2, 5-6）中，是关键的"无学习"控制条件

### 各Run的详细条件构成

| Run | 阶段 | 材料构成 | 任务 | 条件 |
|-----|------|---------|------|------|
| Run 1-2 | 前测 | 240词 = 80yy + 80kj + 40baseline + 40假词 | 真假词判断 | yy/kj/baseline/fake |
| Run 3 | 第一遍学习 | 80句 = 40隐喻句 + 40空间句 | 理解判断(5s) | yy/kj |
| Run 4 | 第二遍学习 | 80句 = 同Run3内容不同顺序 | 喜好判断(5s) | yy/kj |
| Run 5-6 | 后测 | 240词 = 80yy + 80kj + 40baseline + 40新假词 | 真假词判断 | yy/kj/baseline/fake |
| Run 7 | 回忆 | 160词 = 80yy线索 + 80kj线索 | 回忆判断 | yy/kj |
| Stage6 | 行为后测 | 问卷 | 回忆填写+评分 | 非fMRI |

***

## ROI体系

| ROI名称 | 来源 | 分析用途 |
|---------|------|---------|
| GLM功能簇 | glm_analysis 二阶 Cluster-FWE | RSA、RD、GPS |
| L-IFG / R-IFG | Harvard-Oxford Atlas | MVPA、RD、种子连接性 |
| L-AG / R-AG | Harvard-Oxford Atlas | MVPA、RD、跨区RSA |
| Precuneus | Harvard-Oxford Atlas | MVPA、RD |
| PCC | Harvard-Oxford Atlas | MVPA |
| L-MFG / R-MFG | Harvard-Oxford Atlas | MVPA |
| DMN网络 | Yeo 7-network Atlas | 网络级RD |
| Language网络 | Yeo 7-network Atlas | 网络级RD |
| mPFC | 球形ROI [0, 52, -6]，参考Subramaniam et al. (2013) | 效价调节、中介分析 |
| L-STS / R-STS | Harvard-Oxford Atlas | 偏侧化分析 |
| dACC / AI | Harvard-Oxford Atlas | 突显网络、有效连接 |

***

# ⚠️ 实验流程审查：关键问题与修正方案

> 以下问题来自实验流程文档与当前方案/代码的系统交叉审查。**问题A和B为严重级别，影响研究结论的可靠性，必须在运行分析前修正。**

## 问题A（严重）：baseline基线条件完全遗漏

### 问题描述

实验设计是 **3条件**（隐喻联结yy / 空间联结kj / 不联结baseline），但当前方案和几乎全部代码（15/19个文件）都只假设了2个条件（yy/kj），**完全遗漏了baseline条件**。

baseline是实验设计中的关键控制条件——它不参与学习，因此前后测中的表征变化应为零或最小值。如果没有baseline作为对照，无法区分yy/kj的表征变化是"学习效应"还是"重复测量效应（如重复抑制、熟悉度变化等）"。

### 影响范围

| 文件 | 当前状态 | 需要修正 |
|------|---------|---------|
| `common/final_utils.py` → `difference_in_differences()` | 只支持yy vs kj | 需扩展为3条件，或增加baseline对比 |
| `representation_analysis/stack_patterns.py` → `CONDITION_MAP` | 只映射yy/kj | 需添加baseline映射 |
| `representation_analysis/gps_analysis.py` | 硬编码 `for condition in ["yy", "kj"]` | 需添加baseline |
| `representation_analysis/rd_analysis.py` | 同上 | 需添加baseline |
| `rsa_analysis/cross_roi_rsa.py` | 同上 | 需添加baseline |
| `rsa_analysis/model_rdm_comparison.py` | 同上 | 需添加baseline |
| `rsa_analysis/rsa_config.py` | 只有2个对比的ROI | 需考虑是否需要yy-baseline, kj-baseline对比的ROI |
| `fmri_rsa/fmri_rsa_pipline.py` | `self.conditions = ['yy', 'kj']` | 需添加baseline（仅前后测） |
| `fmri_mvpa/cross_phase_decoding.py` | 只加载yy/kj | 可选添加baseline |
| `fmri_mvpa/mvpa_pre_post_comparison.py` | 只分类yy vs kj | 可选增加3-class分类 |

### 修正方案

**核心原则**：baseline在不同分析阶段的角色不同

- **前后测（Run1-2, 5-6）**：baseline是完整的第3个条件，必须纳入RSA/RD/GPS分析
- **学习阶段（Run3-4）**：baseline不存在，2条件设计正确
- **回忆（Run7）**：baseline不存在，2条件设计正确

**分析策略**：

1. **主分析**：保持 yy vs kj 的直接对比（论文的核心故事）
2. **关键控制分析**：增加 baseline 的前后测变化作为"无学习"对照
   - `Δ_yy = post_yy - pre_yy`（隐喻学习效应）
   - `Δ_kj = post_kj - pre_kj`（空间学习效应）
   - `Δ_baseline = post_baseline - pre_baseline`（重测效应/零假设）
   - 核心检验：`Δ_yy > Δ_baseline` 且 `Δ_kj > Δ_baseline`（学习效应超过重测）
   - 进一步检验：`(Δ_yy - Δ_baseline) > (Δ_kj - Δ_baseline)`（隐喻学习特异性超过空间学习）

**具体修改清单**：

1. `final_utils.py`：扩展 `difference_in_differences()` 支持3条件，增加 `three_condition_anova()` 函数
2. `stack_patterns.py`：`CONDITION_MAP` 增加 `{"baseline": "baseline", "bl": "baseline", ...}`
3. 所有前后测分析脚本（gps, rd, cross_roi_rsa, model_rdm等）：条件循环从 `["yy", "kj"]` 改为 `["yy", "kj", "baseline"]`
4. 统计方案：从2×2 ANOVA升级为 **3（条件）× 2（时间）重复测量ANOVA**，配合事后两两比较

## 问题B（严重）：前后测GLM分析缺失

### 问题描述

当前GLM分析（`glm_config.py`的`RUNS=[3,4]`）**只分析学习阶段**，完全未覆盖前测（Run1-2）和后测（Run5-6）。

缺少前后测GLM意味着：
1. 无法检验学习前后**激活水平**的变化（只有表征层面的RSA/RD/GPS，缺少Layer 1的纵向证据）
2. 无法生成前后测阶段的condition对比map
3. 无法做 GLM层面的 post > pre 交互效应

### 修正方案

**新增前后测GLM分析**：

1. 在`glm_config.py`中新增前后测配置：
   - `PRE_POST_RUNS = [1, 2, 5, 6]`
   - 前后测对比增加：`"yy_gt_baseline"`, `"kj_gt_baseline"`, `"learning_gt_baseline": "(yy + kj)/2 - baseline"`
2. 二阶分析增加：
   - 前后测配对t检验：`post_yy > pre_yy`, `post_kj > pre_kj`, `post_baseline > pre_baseline`
   - 交互效应：`(post_yy - pre_yy) - (post_baseline - pre_baseline)`
3. 🔴 需修改 `glm_config.py` 和 `main_analysis.py` 支持多阶段GLM

## 问题C（中等）：被试排除标准不统一

### 问题描述

不同文件的被试列表不一致：
- `glm_config.py`：28人（sub-01~28）
- `rsa_config.py`：28人
- `fmri_rsa_pipline.py`：26人（排除sub-14, sub-24）
- 行为分析暗示sub-12无数据

### 修正方案

1. 在 `final_utils.py` 或新建 `common/subjects.py` 中定义**唯一的被试排除列表**
2. 排除标准文档化：说明每个被试的排除原因（数据缺失/头动过大/行为异常等）
3. 所有脚本从统一配置读取被试列表

## 问题D（中等）：前后测行为分析缺失

### 问题描述

`getAccuracyScore.py` 只分析Run7回忆数据。但前测（Run1-2）和后测（Run5-6）也有**真假词判断的正确率和反应时**行为数据，这些数据完全未被分析。

### 修正方案

1. 🔴 需新建 `behavior_analysis/prepost_behavior.py`
   - 分析前后测的真假词判断正确率和反应时
   - **3条件（yy/kj/baseline）× 2时间（pre/post）** 重复测量ANOVA
   - 排除假词trials
   - 关键检验：学习后yy和kj的反应时是否加快（语义启动效应），baseline是否无变化

## 问题E（中等）：前后测的假词trials处理策略不明确

### 问题描述

前后测各有40个假词，这些假词在events文件中是独立的trial_type。当前LSS代码（`run_lss.py`）会为**所有trial_type**生成beta map（包括假词）。后续RSA/RD/GPS分析是否需要排除假词beta？

### 修正方案

1. 在 `stack_patterns.py` 中添加 `EXCLUDE_CONDITIONS = ["fake", "jia", ...]`（假词相关标签）
2. 确保LSS生成所有trial的beta（保持完整性），但下游分析只使用真词条件的beta
3. 在文档中明确记录假词的处理策略

## 问题F（低）：Run3和Run4的任务差异未被利用

### 问题描述

Run3（理解判断）和Run4（喜好判断）虽然刺激内容相同，但**任务要求不同**。当前方案将Run3和Run4合并为"学习阶段"统一分析，未利用两个run之间的任务差异。

### 修正方案

作为补充分析，可以比较Run3 vs Run4的：
- MVPA分类准确率（理解任务 vs 喜好任务下的条件可区分性）
- RSA表征距离（两次学习是否产生了渐进的表征变化）
- 这与"补充E. 学习阶段时间窗分析"自然整合

## 问题G（低）：Stage6行为后测数据未纳入分析

### 问题描述

Stage6收集了被试的回忆填写、回忆程度评分（1-7级），以及是否在后测中自发回忆了对应词。这些数据可以作为个体差异变量：
- 回忆程度评分 → 可作为RSA模型RDM中的"记忆强度模型"
- 后测中是否自发回忆 → 可按trial筛选（回忆 vs 未回忆）做条件分析

### 修正方案

1. 在 `model_rdm_comparison.py` 的模型RDM中增加 **M5: 记忆强度模型**（基于Stage6的1-7级回忆程度评分）
2. 可选：将后测trials按"是否自发回忆"拆分，比较回忆trial和未回忆trial的神经表征差异

***

# 脚本修改清单（按优先级）

## 必须修改的现有脚本（对应问题A-E）

| 优先级 | 脚本 | 修改内容 | 对应问题 |
|--------|------|---------|---------|
| **P0** | `common/final_utils.py` | `difference_in_differences()` 支持3条件；新增 `three_condition_anova()` | A |
| **P0** | `representation_analysis/stack_patterns.py` | `CONDITION_MAP` 添加baseline；`PHASE_MAP` 添加recall；增加假词排除逻辑 | A, E, F |
| **P0** | `representation_analysis/gps_analysis.py` | 条件循环添加baseline；统计升级为3×2 ANOVA | A |
| **P0** | `representation_analysis/rd_analysis.py` | 同上 | A |
| **P0** | `rsa_analysis/cross_roi_rsa.py` | 条件循环添加baseline | A |
| **P0** | `rsa_analysis/model_rdm_comparison.py` | 条件模型升级为3条件 | A |
| **P1** | `glm_analysis/glm_config.py` | 新增前后测Run配置和3条件对比 | B |
| **P1** | `glm_analysis/main_analysis.py` | 支持前后测GLM + 交互效应分析 | B |
| **P1** | `rsa_analysis/rsa_config.py` | 被试列表统一化 | C |
| **P1** | `fmri_rsa/fmri_rsa_pipline.py` | `conditions` 添加baseline（前后测分析时） | A |
| **P2** | `fmri_mvpa/cross_phase_decoding.py` | 可选：添加baseline泛化对比 | A |
| **P2** | `fmri_mvpa/mvpa_pre_post_comparison.py` | 可选：添加3-class分类或baseline控制 | A |

## 必须新建的脚本

| 优先级 | 脚本 | 功能 | 对应问题 |
|--------|------|------|---------|
| **P0** | `common/subjects.py` | 统一的被试排除列表和排除原因记录 | C |
| **P1** | `behavior_analysis/prepost_behavior.py` | 前后测真假词判断的3条件×2时间行为分析 | D |

***

# 分析流程：逐步详细方案

## 第0步：数据准备——LSS单试次Beta估计

### 0A. 学习阶段LSS（Run 3-4，供MVPA/RSA/RD使用）

- **目的**：为每个trial生成一个beta map（不平滑，保留空间模式信息）
- **方法**：LSS（Least Squares Separate），每个trial单独GLM
- **已有脚本**：
  - `metaphoric/fmri_lss/fmri_lss_enhance.py` ★★★★ — 增强版LSS，批处理+并行，三类事件编码（target/other_same/other_different）
  - `metaphoric/fmri_lss/fmri_lss_pipline.py` ★★★ — 基础版LSS
- **输出**：`learn_LSS/{sub}/run-{run}_LSS/` 下每个trial的beta和t统计图像

### 0B. 前后测LSS（Run 1-2, 5-6，供RSA/RD/GPS使用）

- **目的**：为前测和后测的每个trial生成beta map
- **已有脚本**：
  - `metaphoric/glm_analysis/run_lss.py` ★★★★ — RSA专用LSS，强制无平滑，unique_label=trial_type+'_'+pic_num
  - `metaphoric/glm_analysis/lss_config_override.py` — 配置LSS_RUNS=[1,2,5,6]
- **输出**：每个被试的前后测单试次beta maps

### 0C. LSS质量检查

- **已有脚本**：
  - `metaphoric/glm_analysis/check_lss_comprehensive.py` ★★★★ — 完整性、NaN、空脑、极端值检测
- **操作**：对所有LSS结果运行质量检查，记录并排除问题trial

### 0D. 模式堆叠为4D NIfTI（新增）

- **目的**：将单trial beta maps按条件×阶段堆叠为4D NIfTI，供RD/GPS/ROI-DSM等分析使用
- **可移植gjxx脚本**：
  - `gjxx/patterns.py` — `stack_maps()` 和 `stack_maps_from_metadata()`
- 🔴 **需新建**：`metaphoric/representation_analysis/stack_patterns.py`
  - 功能：读取LSS输出 + 事件文件，按(condition × phase)分组堆叠
  - 输出：`pre_yy.nii.gz`, `pre_kj.nii.gz`, `post_yy.nii.gz`, `post_kj.nii.gz`, `learn_yy.nii.gz`, `learn_kj.nii.gz`
  - 同时输出metadata TSV（含trial_id, condition, phase, run, pic_num列）

***

## 第1步（Layer 1）：行为分析——学习效果的条件差异

### 1A. 记忆准确率的条件差异

- **目的**：检验隐喻条件(yy)是否比空间条件(kj)的记忆效果更好
- **已有脚本**：
  - `metaphoric/behavior_analysis/getAccuracyScore.py` ★★★ — 配对t检验，Cohen's dz，Shapiro-Wilk正态性检验
- **统计**：配对t检验 + Cohen's dz + 95% CI
- **预期结果**：yy均值0.68 > kj均值0.59，t(26)=4.22, p=0.000261

### 1B. 反应时的条件差异

- **目的**：检验两条件的反应时差异
- **已有脚本**：
  - `metaphoric/behavior_analysis/getTimeActionScore.py` ★★★ — 同结构
- **统计**：配对t检验 + Cohen's dz

### 1C. 行为数据LMM分析（升级）

- **目的**：使用LMM替代t检验，同时建模被试和项目随机效应
- 🔴 **需新建**：`metaphoric/behavior_analysis/behavior_lmm.py`（或R脚本）
  - 模型：`accuracy ~ condition + (1 + condition | subject) + (1 | item)`
  - 工具：`pymer4` 或 R `lme4`
- **补充统计**：Wilcoxon signed-rank检验（非参数稳健性检查）、贝叶斯配对t检验（BF10）

### 1D. 行为数据预处理辅助

- **已有脚本**：
  - `metaphoric/behavior_analysis/addMemory.py` — 为run-7事件文件添加memory列
  - `metaphoric/behavior_analysis/getMemoryDetail.py` — 添加action/action_time列

**论文对应**：Figure 1A-B（行为结果柱状图+散点图）

***

## 第2步（Layer 1）：GLM单变量激活分析——学习阶段的条件差异

### 2A. 一阶GLM分析

- **目的**：对每个被试的学习阶段(run 3-4)拟合GLM，获得条件对比(yy>kj, kj>yy)的统计图
- **已有脚本**：
  - `metaphoric/glm_analysis/main_analysis.py` ★★★★★ — FirstLevelModel，SPM+derivative HRF，AR1噪声模型
  - `metaphoric/glm_analysis/glm_config.py` ★★★★★ — 对比定义：`"Metaphor_gt_Spatial": "yy - kj"`
  - `metaphoric/glm_analysis/glm_utils.py` ★★★★ — Friston-24去噪 + Scrubbing
- **参数**：TR=2.0s，6mm FWHM平滑，Friston-24头动参数，Scrubbing(FD>0.5, std_dvars>1.5)

### 2B. 二阶组水平分析

- **目的**：组水平检验条件效应
- **已有脚本**：同 `main_analysis.py`
- **统计方案**：
  - 主分析：Cluster-FWE（非参数置换检验，5000次）
  - 补充：FDR q<0.05
- **已有脚本（GPU加速）**：
  - `metaphoric/glm_analysis/gpu_permutation.py` ★★★★★ — PyTorch CUDA加速符号翻转置换检验

### 2C. 激活簇报告与ROI生成

- **目的**：自动标注显著簇的解剖位置，生成功能定义ROI mask
- **已有脚本**：`glm_utils.py` 中 `generate_cluster_report()` — Harvard-Oxford Atlas标注
- **输出**：功能ROI mask NIfTI文件，供后续MVPA/RSA/RD分析使用

### 2D. GLM阈值可视化

- **已有脚本**：
  - `metaphoric/get_nii.py` ★★ — 快速FDR阈值+切面可视化

**论文对应**：Figure 2（全脑激活图 + 激活簇表格）

***

## 第3步（Layer 2）：MVPA多变量模式分析——条件信息的编码与学习效应

### 3A. 学习阶段ROI-MVPA（Run 3-4）

- **目的**：检验关键ROI是否包含足以区分yy/kj的多变量信息
- **已有脚本**：
  - `metaphoric/fmri_mvpa/fmri_mvpa_roi/fmri_mvpa_roi_pipeline_merged.py` ★★★★★
    - SVM线性核分类器
    - DynamicSelectKBest + DynamicPCA特征选择
    - 嵌套交叉验证（外层LOGO/StratifiedKFold，内层GridSearchCV）
    - 组水平：单样本t检验(vs 0.5) + 符号翻转置换检验(10000次)
    - 数据泄露检查
  - `metaphoric/fmri_mvpa/fmri_mvpa_roi/create_optimized_roi_masks.py` ★★★★★ — 球形/解剖/网络ROI创建
  - `metaphoric/fmri_mvpa/fmri_mvpa_roi/visualization.py` ★★★ — 6种图表
  - `metaphoric/fmri_mvpa/fmri_mvpa_roi/report_generator.py` ★★★★ — HTML报告

### 3B. 学习阶段Searchlight-MVPA（Run 3-4）

- **目的**：全脑无偏搜索包含条件信息的区域
- **已有脚本**：
  - `metaphoric/fmri_mvpa/fmri_mvpa_searchlight/fmri_mvpa_searchlight_pipeline_multi.py` ★★★★★
    - 球形邻域（半径3体素）
    - 两步校正：分类阈值(>0.54)筛选 → t检验
    - FDR + FWE双校正
    - 被试内机会水平估计（标签置换重拟合）
    - 数据泄露诊断
  - `metaphoric/fmri_mvpa/fmri_mvpa_searchlight/optimized_config.py` — 预定义配置模板
  - `metaphoric/fmri_mvpa/fmri_mvpa_searchlight/visualization.py` ★★★★
  - `metaphoric/fmri_mvpa/fmri_mvpa_searchlight/report_generator.py` ★★★★

### 3C. 前后测MVPA对比（Run 1-2 vs Run 5-6）——学习效应

- **目的**：检验学习前后条件可解码性是否提升（MVPA分类准确率的时间×条件交互）
- **分析逻辑**：
  - 分别对pre(run1-2)和post(run5-6)做ROI-MVPA分类（yy vs kj）
  - 比较：post_accuracy vs pre_accuracy（配对t检验）
  - 如果post > pre，说明学习增强了条件间的神经可区分性
- **已有脚本基础**：`fmri_mvpa_roi_pipeline_merged.py` 可复用，需修改输入数据为pre/post的LSS beta
- 🔴 **需新建**：`metaphoric/fmri_mvpa/fmri_mvpa_roi/mvpa_pre_post_comparison.py`
  - 功能：调用现有MVPA pipeline分别对pre和post数据做分类，然后做配对t检验
  - 统计：配对t检验 + 置换检验 + Cohen's d

### 3D. 跨阶段泛化解码（Cross-phase decoding）

- **目的**：在学习阶段训练分类器，在前测/后测上测试——检验学习阶段获得的神经编码是否泛化到测试阶段
- **分析逻辑**：
  - 训练集：run 3-4的LSS beta（yy vs kj标签）
  - 测试集A：run 1-2的LSS beta → 泛化到pre的准确率
  - 测试集B：run 5-6的LSS beta → 泛化到post的准确率
  - 如果post泛化 > pre泛化，说明学习建立了可迁移的神经编码
- **可参考gjxx方法**：`gjxx/GJXX_paper_code_analysis_report.md` 中 E4 — SVM跨阶段解码
- 🔴 **需新建**：`metaphoric/fmri_mvpa/fmri_mvpa_roi/cross_phase_decoding.py`
  - 功能：train on learning → test on pre/post
  - 统计：置换检验(10000次) + 配对t检验(post vs pre泛化准确率)

**论文对应**：Figure 3（ROI-MVPA柱状图 + Searchlight全脑图 + 前后测MVPA变化）

***

## 第4步（Layer 3A）：RSA表征相似性分析——表征结构的学习重组

### 4A. 学习阶段RSA（Run 3-4）

- **目的**：分析学习阶段隐喻和空间概念的条件内/间表征距离
- **已有脚本**：
  - `metaphoric/fmri_rsa/fmri_rsa_pipline.py` ★★★★
    - 多种距离度量：correlation, euclidean, cosine
    - 多种相关方法：pearson, spearman
    - RDM类型：within_condition, between_condition, cross_condition
    - 置换检验(1000次) + 多重比较校正(fdr_bh/bonferroni/holm)
    - 条件内RDM + 跨条件RDM + 全局RDM
    - 并行处理（multiprocessing.Pool）

### 4B. 前后测RSA对比（Run 1-2 vs Run 5-6）

- **目的**：检验学习是否导致了表征结构的重组——核心分析
- **已有脚本**：
  - `metaphoric/rsa_analysis/run_rsa_optimized.py` ★★★★
    - Item-wise RSA：计算前后测每个刺激项的表征相似度变化
    - Z-score标准化（across voxels for each trial）
    - 输出：`rsa_summary_stats.csv`（用于ANOVA）+ `rsa_itemwise_details.csv`（用于LMM）
  - `metaphoric/rsa_analysis/rsa_config.py` — ROI来源、距离度量、Top-K选项
  - `metaphoric/rsa_analysis/check_data_integrity.py` — 数据完整性校验
  - `metaphoric/rsa_analysis/plot_rsa_results.py` — Delta Similarity交互柱状图

### 4C. RSA的LMM统计（升级）

- **目的**：对item-wise RSA数据用LMM替代ANOVA，建模被试和项目随机效应
- 🔴 **需新建**：`metaphoric/rsa_analysis/rsa_lmm.py`（或R脚本）
  - 模型：`similarity ~ condition * time + (1 + condition * time | subject) + (1 | item)`
  - 输入：`rsa_itemwise_details.csv`（已有）
  - 关键检验：condition × time 交互效应
  - 工具：`pymer4` 或 R `lme4`

### 4D. 模型RDM比较（新增高级分析）

- **目的**：构建理论驱动的模型RDM，检验哪个认知模型最能解释学习后的神经表征变化
- **候选模型RDM**：
  - M1: 语义类别模型（同条件=0，跨条件=1）
  - M2: 概念距离模型（基于BERT词向量余弦距离）
  - M3: 学习效果模型（基于行为记忆成绩的梯度）
  - M4: 视觉相似性模型（如果刺激有图片，基于DNN特征）
- 🔴 **需新建**：`metaphoric/rsa_analysis/model_rdm_comparison.py`
  - 功能：构建模型RDM → 与neural RDM做Spearman相关 → 偏相关控制竞争模型 → 置换检验
  - 参考方法：Kriegeskorte et al. (2008) 的 RSA toolbox 标准流程

### 4E. 跨脑区RSA一致性（Cross-ROI RSA）

- **目的**：检验学习后不同脑区之间的表征结构是否趋于一致（表征对齐）
- **分析逻辑**：
  - 在ROI_A和ROI_B中分别构建neural RDM
  - 计算 corr(RDM_A, RDM_B)
  - 比较：post的跨区RDM相关 vs pre的跨区RDM相关
  - 如果学习后增强 → 脑区间表征结构趋于对齐
- **可移植gjxx脚本**：
  - `gjxx/roi.py` — `compute_subject_roi_dsm_correlation()` — ROI-to-ROI DSM Pearson相关 + Fisher-Z
- 🔴 **需新建**：`metaphoric/rsa_analysis/cross_roi_rsa.py`
  - 基于gjxx/roi.py的ROI-DSM相关逻辑
  - 扩展为支持 pre/post 两个时间点的对比
  - 输出：每个被试×每对ROI的跨区RDM相关值

**论文对应**：Figure 4（RDM热图 + Delta Similarity交互效应 + 模型RDM比较 + 跨区对齐变化）

***

## 第5步（Layer 3B）：表征维度分析（RD）——表征几何的压缩/扩展

### 5A. ROI-level表征维度分析

- **目的**：检验学习前后隐喻/空间概念的表征维度是否发生变化
- **核心指标**：表征维度（Representational Dimensionality, RD）= 累计解释方差达到阈值所需的最少PCA成分数
  - 低维度 → 表征更紧凑/结构化
  - 高维度 → 表征更分散/无结构
- **分析逻辑**：
  ```
  Pre_yy_RD, Pre_kj_RD, Post_yy_RD, Post_kj_RD
  交互效应：(Post_yy_RD - Pre_yy_RD) vs (Post_kj_RD - Pre_kj_RD)
  ```
  - 如果隐喻学习导致RD降低（维度压缩），说明学习使隐喻表征变得更结构化
- **可移植gjxx核心算法**：
  - `gjxx/dimension_analysis_core.py`:
    - `correlation_distance_rdm()` — 相关距离RDM构建
    - `pca_explained_variance()` — PCA解释方差（SVD实现）
    - `dimensionality_from_samples()` — 从样本矩阵估计维度
    - `analyze_trial_split()` — 条件分割分析（配对t检验）
  - `gjxx/rd.py`:
    - `_rd_from_rdm()` — 基于RDM特征值的RD估计（替代实现）
    - `_rd_from_equalized_patterns()` — 等量化RD（trial数不等时的稳健估计）
  - `gjxx/utils.py`:
    - `fisher_z_from_samples()`, `components_to_variance_threshold()`, `paired_t_summary()`
- 🔴 **需新建**：`metaphoric/representation_analysis/rd_analysis.py`
  - 移植gjxx核心算法，适配 2(条件) × 2(时间) 设计
  - 输入：第0D步生成的4D NIfTI + ROI mask
  - 输出：每个被试×条件×时间的RD值 → CSV
  - 统计：2×2重复测量ANOVA + 交互效应配对t检验

### 5B. 全脑Searchlight维度映射

- **目的**：数据驱动地寻找学习后维度变化最大的脑区
- **分析逻辑**：
  - 对pre和post分别做searchlight RD map（每个体素周围100个邻居的RD值）
  - 组水平：逐体素配对t检验 post_yy_RD vs pre_yy_RD（及kj对照）
  - 交互效应map：(post_yy - pre_yy) vs (post_kj - pre_kj)
- **可移植gjxx脚本**：
  - `gjxx/dimension_story_utils.py`:
    - `compute_searchlight_dimension_map()` — cKDTree邻域 + 逐球PCA维度估计
    - `compute_group_paired_map_statistics()` — 逐体素配对t检验 → t-map/p-map
  - `gjxx/run_stage12_dimension_searchlight.py` — Searchlight运行脚本模板
- 🔴 **需新建**：`metaphoric/representation_analysis/rd_searchlight.py`
  - 移植searchlight框架，扩展为 condition × time 设计
  - 输出：个体level RD map (NIfTI) + 组level t-map/p-map

### 5C. RD稳健性检验

- **目的**：验证RD结果不受试次数差异、均值差异、PCA参数选择的影响
- **可移植gjxx脚本**：
  - `gjxx/dimension_story_utils.py`:
    - `compute_metric_with_optional_resampling()` — 等量化重采样(1000次)
    - `demean_condition_samples()` — 去均值策略
    - `build_cross_subject_voxel_keep_vector()` — 跨被试体素筛选
  - `gjxx/run_stage08_roi_robustness.py` — 7组ROI稳健性检验模板：
    - R1: Pattern PCA（不经RDM）
    - R3: 固定trial数重抽样
    - R4-R5: 去均值+最小/固定trial数
    - R6: 跨被试体素mask过滤
    - R7: PC窗口(3-5)解释方差和
- 🔴 **需新建**：`metaphoric/representation_analysis/rd_robustness.py`
  - 移植gjxx的稳健性检验套件
  - 至少实现：等量化重采样、去均值、多方差阈值(70%-90%)三种检验

**论文对应**：Figure 5（ROI-level RD条件×时间交互 + Searchlight RD变化map + 稳健性检验补充图）

***

## 第6步（Layer 3C）：GPS全局模式相似性——表征紧凑度的变化

### 6A. ROI-level GPS分析

- **目的**：检验学习是否使隐喻概念的表征变得更紧凑（trial间相似度更高）
- **核心指标**：GPS = 每个trial与同条件其他trial的平均Fisher-Z相似度
  - GPS高 → 同条件trial的neural patterns更一致 → 表征更紧凑
- **分析逻辑**：
  ```
  Pre_yy_GPS, Pre_kj_GPS, Post_yy_GPS, Post_kj_GPS
  交互效应：(Post_yy_GPS - Pre_yy_GPS) vs (Post_kj_GPS - Pre_kj_GPS)
  ```
  - 如果隐喻学习后yy的GPS增加 → 学习使隐喻表征趋于一致
- **可移植gjxx脚本**：
  - `gjxx/roi.py`:
    - `compute_gps_subject_metrics()` — GPS核心计算（Fisher-Z相关矩阵 → 行均值 → 按条件分组均值）
  - `gjxx/utils.py`:
    - `fisher_z_from_samples()` — trial间Pearson相关 → Fisher-Z
    - `mean_row_similarity()` — GPS核心函数
- 🔴 **需新建**：`metaphoric/representation_analysis/gps_analysis.py`
  - 移植gjxx/roi.py的GPS逻辑
  - 扩展为 condition × time 设计
  - 输入：4D NIfTI + metadata TSV + ROI mask
  - 输出：每个被试×条件×时间的GPS值 → CSV
  - 统计：2×2重复测量ANOVA + 交互效应

### 6B. GPS与RD的互补性

- **目的**：验证GPS和RD捕获了表征几何的不同方面
- **分析**：
  - 被试内相关：GPS变化(Δ) vs RD变化(Δ) 是否显著相关
  - 预期：GPS增加（更紧凑） ↔ RD降低（更低维） → 负相关
- **可移植gjxx方法**：`gjxx/GJXX_paper_code_analysis_report.md` 中 B4 — 被试内指标间相关
- 🔴 **需新建**：纳入 `gps_analysis.py` 或 `rd_analysis.py` 的补充分析函数

**论文对应**：Figure 5的补充panel（GPS结果与RD结果互补验证）

***

## 第7步（Layer 3D）：种子连接性分析——表征结构的脑区间传播

### 7A. PCA-based种子连接性Searchlight

- **目的**：检验学习后，关键种子区域（如IFG）的表征几何结构是否向更多脑区传播
- **核心算法**（来自gjxx）：
  1. 在seed ROI（如L-IFG）计算trial间相似性矩阵 → PCA得到成分向量
  2. 在全脑每个searchlight球内同样计算 → PCA得到成分向量
  3. 比较seed和local的匹配PCA成分的余弦相似度
  4. 取前min(seed_dim, local_dim)个成分的平均余弦相似度
  - 本质：测量两个脑区的表征几何相似程度
- **分析逻辑**：
  - 分别对pre和post生成全脑种子连接性map
  - 组水平：配对t检验 (post - pre) → 学习后哪些脑区与seed表征结构变得更匹配
- **可移植gjxx脚本**：
  - `gjxx/dimension_story_utils.py`:
    - `compute_seed_connectivity_map()` — 种子连接性searchlight核心算法
  - `gjxx/run_stage13_seed_connectivity.py` — 运行脚本模板
- 🔴 **需新建**：`metaphoric/representation_analysis/seed_connectivity.py`
  - 移植种子连接性算法
  - 种子选择：L-IFG（隐喻加工核心区）、L-AG（语义整合）
  - 分别对pre/post两个阶段运行
  - 组水平比较：post vs pre的连接性map差异

### 7B. ROI-to-ROI表征连接（替代全脑searchlight的精简版）

- **目的**：检验特定ROI对之间的表征结构匹配程度是否因学习而改变
- **已有gjxx方法**：即第4E步的Cross-ROI RSA，与此处互补
- **ROI对选择**：
  - IFG ↔ AG（额-顶语义网络连接）
  - IFG ↔ MTG（额-颞语言网络连接）
  - AG ↔ Precuneus（DMN内连接）

**论文对应**：Figure 6（种子连接性全脑map + ROI对连接变化柱状图）

***

## 第8步：PPI功能连接分析——任务调节的网络协作

### 8A. gPPI分析（学习阶段 Run 3-4）

- **目的**：检验隐喻vs空间条件下，种子区域与全脑的功能连接是否不同
- **分析逻辑**：
  - 种子：GLM显著簇（yy>kj）或先验ROI（L-IFG, L-AG）
  - PPI模型：Y_voxel = β1·Seed_timecourse + β2·Task + β3·Seed×Task + confounds
  - β3即为PPI效应：任务条件对功能连接的调节
- **gjxx参考**：`GJXX_paper_code_analysis_report.md` 中 P1 — PPI分析（论文提及但代码缺失）
- 🔴 **需新建**：`metaphoric/connectivity_analysis/gPPI_analysis.py`
  - 基于nilearn GLM框架实现gPPI
  - 包含：种子时间序列提取 → PPI交互项构建 → 一阶gPPI-GLM → 二阶组水平t检验
  - 多重比较：Cluster-FWE（非参数置换检验）

### 8B. PPI学习动态分析（可选扩展）

- **目的**：检验学习早期vs晚期的功能连接变化
- **分析**：将run3和run4分别做PPI → 比较两个run的PPI效应
- 🔴 **需新建**：纳入 `gPPI_analysis.py` 的扩展功能

**论文对应**：Figure 7（PPI功能连接结果——种子-全脑map + ROI对连接柱状图）

***

## 第9步：脑-行为关联——多层次神经指标预测学习效果

### 9A. 多层次神经指标提取

- **目的**：将前述各分析的核心指标汇总，形成每个被试的"神经变化特征向量"
- **指标列表**：

| 层次 | 指标 | 来源步骤 |
|------|------|---------|
| L1 | GLM yy>kj 对比值（ROI beta均值） | 第2步 |
| L2 | MVPA分类准确率（学习阶段） | 第3A步 |
| L2 | MVPA准确率变化（post - pre） | 第3C步 |
| L2 | 跨阶段泛化准确率（post - pre） | 第3D步 |
| L3a | RSA Delta Similarity（条件交互） | 第4B步 |
| L3b | RD变化（post - pre，条件交互） | 第5A步 |
| L3c | GPS变化（post - pre，条件交互） | 第6A步 |
| L3d | 种子连接性变化（post - pre） | 第7A步 |
| FC | PPI连接强度（yy vs kj） | 第8A步 |

### 9B. 脑-行为相关分析

- **目的**：检验哪些神经指标与行为学习效果（记忆准确率/反应时）相关
- **分析方法**：
  - Pearson/Spearman相关 + 置换检验
  - 多元回归：多个神经指标同时预测行为，检验各自独立贡献
  - FDR校正多重比较
- **可移植gjxx方法**：
  - `gjxx/group.py` — `_simple_voxelwise_regression()` — 体素级回归（如果需要全脑brain-behavior map）
  - `gjxx/group.py` — `_threshold_by_cluster()` — 簇阈值校正
- 🔴 **需新建**：`metaphoric/brain_behavior/brain_behavior_correlation.py`
  - 功能：汇总各步骤输出的CSV → 合并 → 相关矩阵 → 多元回归 → 可视化
  - 输出：相关矩阵热图 + 散点图 + 回归表格

### 9C. 全脑体素级脑-行为回归（补充分析）

- **目的**：全脑搜索与行为学习效果相关的体素
- **分析**：将行为学习增益(post_accuracy - pre_accuracy)作为回归因子，预测全脑的RD变化map或searchlight MVPA准确率map
- **可移植gjxx脚本**：
  - `gjxx/group.py` — `run_group_regression()` — 体素回归 + 秩转换 + 簇阈值

**论文对应**：Figure 8（脑-行为相关散点图矩阵 + 多元回归摘要表）

***

## 第9.5步（补充分析）：文献方法融合——6项关键补充

> 以下6项补充分析来自对14篇文献方法的系统Gap分析。它们沿着当前"多层次表征重组"主线，分别强化了理论深度、方法完备性和审稿人可能关注的遗漏点。

### 补充A. 隐喻熟悉度参数调制分析（Career of Metaphor效应）

- **理论来源**：文献3 Cardillo et al. (2012) — 隐喻从新颖到常规化伴随双侧语义网络负荷降低；文献4 Lai et al. (2015) — 熟悉度调节偏侧化
- **当前方案缺口**：方案将"前测vs后测"作为离散二分变量，未利用**连续熟悉度变化**这一更精细的分析维度
- **补充分析**：
  1. **参数调制GLM**：在一阶GLM中将每个trial的熟悉度评分（如果有行为评分）或学习暴露次数作为**参数调制变量**（parametric modulator），检验哪些脑区的激活与熟悉度线性相关
  2. **学习阶段内的时间梯度**：将run3-4的trial按时间顺序编码为线性参数（参考文献10 Heinonen et al., 2016的序列效应方法），检验学习过程中激活随时间的变化趋势
  3. **RSA的熟悉度梯度**：检验学习后RDM的变化是否与刺激的熟悉度变化量相关（item-level分析）
- **已有脚本基础**：`main_analysis.py` 的GLM框架支持参数调制（nilearn FirstLevelModel的 `add_regs` 参数）
- 🔴 **需新建**：`metaphoric/glm_analysis/parametric_modulation.py`
  - 功能：在一阶GLM中添加熟悉度/时间序列参数调制回归子
  - 二阶：检验参数调制效应的组水平显著性
- **论文价值**：将离散的pre/post对比提升为连续的"隐喻神经生涯"曲线，直接回应Career of Metaphor理论，增加理论深度

### 补充B. 偏侧化指数分析（半球贡献动态）

- **理论来源**：文献1 Mashal et al. (2007) — GSH假说，RH选择性加工新颖隐喻；文献4 Lai et al. (2015) — 偏侧化分析方法，LH贡献减少而非RH增加；文献6 Cardillo et al. (2018) — 左半球对隐喻必要性
- **当前方案缺口**：ROI体系已包含左右分离的IFG/AG/MFG，但**未单独计算偏侧化指数**（LI），无法回答"学习改变了半球贡献格局吗？"这一文献中的核心争议
- **补充分析**：
  1. **激活LI**：对GLM结果计算每个被试的LI = (L - R) / (L + R)，分别在前测/后测/学习阶段
  2. **MVPA LI**：分别在左侧和右侧同源ROI做MVPA分类 → 比较左右分类准确率
  3. **RSA LI**：分别在左右同源ROI计算前后测Delta Similarity → 比较左右半球的表征重组幅度
  4. **LI × 时间交互**：检验LI是否从pre到post发生变化——预测early RH bias → later LH dominance（隐喻常规化方向）
  5. **区分机制**：参考Lai et al.方法，分别检验LH和RH的独立变化量，区分"RH减少" vs "LH增加"两种偏侧化变化机制
- **已有脚本基础**：`create_optimized_roi_masks.py` 已包含左右分离的解剖ROI
- 🔴 **需新建**：`metaphoric/lateralization/lateralization_analysis.py`
  - 功能：计算激活LI + MVPA LI + RSA LI
  - 统计：配对t检验（pre_LI vs post_LI） + 2×2 ANOVA（半球×时间）
- **论文价值**：直接回应GSH假说和Career of Metaphor理论在偏侧化维度上的预测，是PNAS审稿人几乎必问的问题

### 补充C. 情绪效价调节分析

- **理论来源**：文献5 Subramaniam et al. (2013) — 积极效价通过mPFC促进新颖隐喻加工
- **当前方案缺口**：未考虑刺激的情绪效价作为调节变量
- **补充分析**：
  1. 如果刺激材料有效价评分（或可以事后收集评分）：将效价作为item-level协变量纳入LMM
  2. 在RSA模型RDM比较（第4D步）中新增**M5: 效价距离模型**——效价相似的刺激对是否有更相似的神经表征
  3. 检验mPFC的ROI激活/RD/GPS是否与效价调节隐喻学习效果相关
- **已有脚本基础**：`rsa_lmm.py`（待建N5）可以直接扩展协变量
- 🔴 **需新建**：在 `rsa_analysis/model_rdm_comparison.py`（N12）中增加效价模型RDM
- **ROI扩展**：在ROI体系中新增 **mPFC**（MNI坐标参考文献5: ~[0, 52, -6]）
- **论文价值**：揭示情绪-隐喻学习交互的神经机制，扩展单纯认知框架

### 补充D. 有效连接/信息流方向分析

- **理论来源**：文献8 De Pisapia et al. (2016) — 多元回归估计ROI间信息流方向；文献2 Beaty et al. (2017) — 时间窗连接动态分析
- **当前方案缺口**：第8步的gPPI只能检验功能连接的**强弱差异**，不能推断**信息流方向**（谁驱动谁）。文献8明确使用有效连接分析了BA47→DMN节点的方向性信息流
- **补充分析**（选择其一）：
  1. **DCM（动态因果模型）**：对3-4个核心节点（IFG→AG→Precuneus→mPFC）建模，比较不同连接结构的模型证据（BMS贝叶斯模型选择）
  2. **Granger因果/时间滞后相关**：用LSS beta的时间序列（按trial顺序）估计脑区间的先导-跟随关系
  3. **简化版有效连接**：参考De Pisapia et al.方法，用多元回归估计一个ROI对其他ROI的预测力
- 🔴 **需新建**：`metaphoric/connectivity_analysis/effective_connectivity.py`
  - 推荐先实现简化版（多元回归）作为探索性分析
  - 如需DCM：使用nilearn不支持，需SPM MATLAB或tapas/VBA工具箱
- **论文价值**：从"哪些区域联合参与"推进到"谁驱动了网络动态"，是PNAS级别对机制性解释的要求

### 补充E. 学习阶段内的动态时间窗分析

- **理论来源**：文献2 Beaty et al. (2017) — 将隐喻产出任务分为4个时间窗，发现早期SN→AG耦合 → 后期AG→DLPFC耦合；文献10 Heinonen et al. (2016) — 发散思维的序列效应
- **当前方案缺口**：方案主要比较pre vs post（时间跨度大），但**学习过程本身的在线动态**（run3 vs run4，或trial 1-10 vs trial 11-20 vs trial 21-30）未被充分利用
- **补充分析**：
  1. **学习早期 vs 晚期的MVPA准确率**：将run3-4的trials分为前半和后半，分别做MVPA → 准确率是否随学习增加
  2. **RD/GPS的学习曲线**：按时间窗（如每10个trial）计算RD和GPS值 → 绘制学习曲线
  3. **功能连接的时间窗分析**：参考Beaty et al.方法，将run3-4分为3-4个时间窗，分别做ROI-to-ROI连接 → 检验连接强度的线性/二次趋势
  4. **参数调制验证**：与补充A的参数调制GLM互为验证
- **已有脚本基础**：MVPA和RD/GPS分析框架已就绪，只需修改输入数据的时间窗分组
- 🔴 **需新建**：`metaphoric/temporal_dynamics/learning_dynamics.py`
  - 功能：按时间窗分组trial → 分别计算MVPA/RD/GPS/FC指标 → 绘制学习曲线 → 趋势检验
- **论文价值**：将静态的"前后对比"提升为动态的"学习过程追踪"，这是纵向实验设计的核心优势，不利用等于浪费数据

### 补充F. 中介分析——多层次指标间的因果路径

- **理论来源**：文献9 Zhu et al. (2017) — FPN介导DMN与创造力的关系（Bootstrap中介检验）
- **当前方案缺口**：第9步的脑-行为关联仅做相关和回归，未检验**多层次神经指标之间的因果路径**。例如：学习是否先改变了激活模式（L1）→ 进而改变了多变量编码（L2）→ 进而重组了表征几何（L3）→ 最终影响行为？
- **补充分析**：
  1. **序列中介模型**：
     ```
     学习条件(yy vs kj)
         → GLM激活差异(L1)
             → MVPA解码变化(L2)
                 → RD/GPS/RSA变化(L3)
                     → 记忆成绩差异(行为)
     ```
  2. **Bootstrap中介检验**：参考Zhu et al. (2017)使用PROCESS宏（Hayes）或Python `pingouin`/`semopy`
  3. 至少检验两个中介路径：
     - 路径1: 条件 → GLM激活 → MVPA准确率 → 行为
     - 路径2: 条件 → RSA变化 → RD变化 → 行为
  4. 如果样本量允许（26人较少），可用偏相关逐步排除替代完整SEM
- 🔴 **需新建**：`metaphoric/brain_behavior/mediation_analysis.py`
  - 功能：Bootstrap中介检验（PROCESS模型4/6） + 序列中介 + 路径图可视化
  - 工具：`pingouin`（简单中介）或 `semopy`/`lavaan`（SEM）
- **论文价值**：将"多层次并行证据"升级为"多层次因果路径"，是从描述到机制解释的关键跳跃。PNAS非常看重mechanism over description

***

## 第10步：稳健性与补充分析

### 10A. 统计稳健性

| 检验项 | 方法 | 状态 |
|--------|------|------|
| MVPA CV方案敏感性 | LOGO vs StratifiedKFold vs LeaveOneRunOut | 已有代码支持 |
| RSA距离度量敏感性 | correlation vs euclidean vs cosine | 已有代码支持 |
| RD方差阈值敏感性 | 70% / 75% / 80% / 85% / 90% | 可移植gjxx |
| RD等量化重采样 | 1000次无放回抽样 | 可移植gjxx |
| RD去均值控制 | demean_condition_samples | 可移植gjxx |
| RD体素筛选 | 跨被试t检验去除条件均值差异体素 | 可移植gjxx |
| 贝叶斯因子 | 对所有关键效应报告BF10 | 🔴 需新建 |
| Bootstrap CI | 对MVPA/RSA/RD报告bootstrap 95% CI | 🔴 需新建 |

### 10B. 控制分析

| 控制项 | 方法 | 状态 |
|--------|------|------|
| 头动控制 | 以FD为协变量纳入二阶分析 | 已有（glm_utils.py Friston-24） |
| 单变量激活控制 | ANCOVA控制GLM激活后多变量效应是否存在 | 🔴 需新建 |
| 试次数平衡 | 等量化重采样确保条件间试次数相等 | 可移植gjxx |

### 10C. 增强版Pipeline框架（工程支持）

- **已有脚本**：
  - `metaphoric/fmri_mvpa/fmri_mvpa_enhance/` 完整框架：
    - `enhanced_pipeline.py` — 5阶段自动化流程
    - `global_config.py` — 全局配置管理
    - `quality_control.py` — PerformanceMonitor + DataQualityAssessor
    - `error_handling.py` — 检查点保存/恢复
    - `data_flow.py` — DAG流水线引擎

**论文对应**：Supplementary Materials（稳健性检验完整结果表）

***

# 需新建脚本清单（优先级排序）

## 高优先级（核心分析）

| 序号 | 脚本 | 对应步骤 | 功能 | gjxx可移植程度 |
|------|------|---------|------|--------------|
| N1 | `representation_analysis/stack_patterns.py` | 0D | 按条件×阶段堆叠LSS beta为4D NIfTI | 高（gjxx/patterns.py） |
| N2 | `representation_analysis/rd_analysis.py` | 5A | ROI-level表征维度分析（PCA维度估计） | 高（gjxx/dimension_analysis_core.py + rd.py） |
| N3 | `representation_analysis/gps_analysis.py` | 6A | ROI-level全局模式相似性分析 | 高（gjxx/roi.py + utils.py） |
| N4 | `rsa_analysis/cross_roi_rsa.py` | 4E | 跨脑区RSA一致性（ROI-to-ROI DSM） | 高（gjxx/roi.py） |
| N5 | `rsa_analysis/rsa_lmm.py` | 4C | RSA item-wise数据的LMM统计 | 低（需独立编写） |
| N6 | `brain_behavior/brain_behavior_correlation.py` | 9B | 多层次脑-行为关联分析 | 中（gjxx/group.py部分可用） |

## 中优先级（扩展分析）

| 序号 | 脚本 | 对应步骤 | 功能 | gjxx可移植程度 |
|------|------|---------|------|--------------|
| N7 | `representation_analysis/rd_searchlight.py` | 5B | 全脑Searchlight维度映射 | 高（gjxx/dimension_story_utils.py） |
| N8 | `representation_analysis/seed_connectivity.py` | 7A | PCA种子连接性Searchlight | 高（gjxx/dimension_story_utils.py + run_stage13） |
| N9 | `representation_analysis/rd_robustness.py` | 5C/10A | RD稳健性检验套件 | 高（gjxx/run_stage08 + dimension_story_utils.py） |
| N10 | `fmri_mvpa/fmri_mvpa_roi/mvpa_pre_post_comparison.py` | 3C | 前后测MVPA对比 | 低（基于已有MVPA pipeline扩展） |
| N11 | `fmri_mvpa/fmri_mvpa_roi/cross_phase_decoding.py` | 3D | 跨阶段泛化解码 | 低（参考gjxx E4方法） |
| N12 | `rsa_analysis/model_rdm_comparison.py` | 4D | 模型RDM比较 | 低（需独立编写） |

## 中-低优先级（文献Gap补充分析）

| 序号 | 脚本 | 对应步骤 | 功能 | gjxx可移植程度 |
|------|------|---------|------|--------------|
| N13 | `glm_analysis/parametric_modulation.py` | 补充A | 熟悉度/时间参数调制GLM | 低（基于已有GLM扩展） |
| N14 | `lateralization/lateralization_analysis.py` | 补充B | 偏侧化指数LI（激活/MVPA/RSA三层次） | 低（需独立编写） |
| N15 | `temporal_dynamics/learning_dynamics.py` | 补充E | 学习阶段时间窗动态分析 | 低（需独立编写） |
| N16 | `brain_behavior/mediation_analysis.py` | 补充F | Bootstrap中介/序列中介分析 | 低（需独立编写） |
| N17 | `connectivity_analysis/effective_connectivity.py` | 补充D | 有效连接/信息流方向分析 | 低（需独立编写） |

## 低优先级（统计工具补充）

| 序号 | 脚本 | 对应步骤 | 功能 | gjxx可移植程度 |
|------|------|---------|------|--------------|
| N18 | `connectivity_analysis/gPPI_analysis.py` | 8A | gPPI功能连接分析 | 低（需独立编写） |
| N19 | `behavior_analysis/behavior_lmm.py` | 1C | 行为数据LMM分析 | 低（需独立编写） |
| N20 | `utils/bayesian_stats.py` | 10A | 贝叶斯因子计算 | 低（需独立编写） |
| N21 | `utils/bootstrap_ci.py` | 10A | Bootstrap置信区间 | 低（需独立编写） |

***

# 论文图表规划

| Figure | 内容 | 数据来源步骤 |
|--------|------|------------|
| **Fig. 1** | 实验范式图 + 分析框架概览 | — |
| **Fig. 2** | 行为结果：记忆准确率/反应时条件差异 | 第1步 |
| **Fig. 3** | GLM全脑激活图 + 激活簇表格 | 第2步 |
| **Fig. 4** | MVPA结果：ROI柱状图 + Searchlight全脑图 + 前后测变化 | 第3步 |
| **Fig. 5** | RSA结果：RDM热图 + Delta Similarity交互 + 模型RDM比较 | 第4步 |
| **Fig. 6** | 表征几何：RD条件×时间交互 + GPS变化 + Searchlight RD map | 第5-6步 |
| **Fig. 7** | 种子连接性 + PPI功能连接 | 第7-8步 |
| **Fig. 8** | 脑-行为关联：相关散点图矩阵 | 第9步 |

| Supp. Figure | 内容 |
|-------------|------|
| **Fig. S1** | 完整ROI坐标表和mask可视化 |
| **Fig. S2** | RD稳健性检验（7种策略） |
| **Fig. S3** | MVPA交叉验证敏感性分析 |
| **Fig. S4** | RSA距离度量敏感性分析 |
| **Fig. S5** | 跨阶段泛化解码详细结果 |
| **Fig. S6** | 所有被试个体水平结果 |
| **Fig. S7** | 贝叶斯因子汇总表 |
| **Fig. S8** | 偏侧化指数（LI）随学习阶段的变化（补充B） |
| **Fig. S9** | 学习阶段内动态时间窗：MVPA/RD/GPS学习曲线（补充E） |
| **Fig. S10** | 参数调制结果：熟悉度/时间梯度的全脑参数效应（补充A） |
| **Fig. S11** | 中介分析路径图（补充F） |
| **Fig. S12** | 有效连接方向性分析结果（补充D） |

***

# 统计方案总结

| 分析类型 | 主要统计方法 | 多重比较校正 | 效应量 |
|---------|------------|------------|--------|
| 行为分析 | LMM + 3×2重复测量ANOVA | N/A（单次比较） | Cohen's dz / η² |
| GLM全脑 | Cluster-FWE（置换检验5000次） | FWE | — |
| GLM ROI | FDR-BH | FDR q<0.05 | partial η² |
| MVPA | 置换检验(10000次) + 配对t检验 | FDR跨ROI | 准确率-0.5 |
| RSA | LMM + 置换检验 | FDR跨ROI | Cohen's d |
| RD | 3×2重复测量ANOVA + 等量化重采样 | FDR跨ROI | η² / Cohen's d |
| GPS | 3×2重复测量ANOVA | FDR跨ROI | η² / Cohen's d |
| PPI | Cluster-FWE（置换检验） | FWE | — |
| 脑-行为 | Pearson/Spearman + 置换检验 | FDR | r / R² |
| Searchlight | 配对t检验 + Cluster-FWE | FWE | — |

所有关键效应同时报告：p值 + 效应量 + 95% CI + BF10

***

# 执行路线图

### Phase 1：数据准备与基础分析（已完成/可直接运行）

- [x] LSS学习阶段（fmri_lss_enhance.py）
- [x] LSS前后测（run_lss.py）
- [x] LSS质量检查（check_lss_comprehensive.py）
- [x] 行为分析（getAccuracyScore.py, getTimeActionScore.py）
- [x] GLM一阶+二阶（main_analysis.py）
- [x] ROI-MVPA学习阶段（fmri_mvpa_roi_pipeline_merged.py）
- [x] Searchlight-MVPA学习阶段（fmri_mvpa_searchlight_pipeline_multi.py）
- [x] 前后测RSA（run_rsa_optimized.py）
- [x] 学习阶段RSA（fmri_rsa_pipline.py）

### Phase 2：新建核心脚本（高优先级 N1-N6）

- [ ] N1: stack_patterns.py — 模式堆叠
- [ ] N2: rd_analysis.py — 表征维度分析
- [ ] N3: gps_analysis.py — GPS分析
- [ ] N4: cross_roi_rsa.py — 跨脑区RSA
- [ ] N5: rsa_lmm.py — RSA LMM统计
- [ ] N6: brain_behavior_correlation.py — 脑-行为关联

### Phase 3：新建扩展脚本（中优先级 N7-N12）

- [ ] N7: rd_searchlight.py — 全脑RD map
- [ ] N8: seed_connectivity.py — 种子连接性
- [ ] N9: rd_robustness.py — RD稳健性
- [ ] N10: mvpa_pre_post_comparison.py — 前后测MVPA
- [ ] N11: cross_phase_decoding.py — 跨阶段解码
- [ ] N12: model_rdm_comparison.py — 模型RDM比较

### Phase 4：新建补充脚本（文献Gap补充 N13-N17）

- [ ] N13: parametric_modulation.py — 熟悉度参数调制GLM
- [ ] N14: lateralization_analysis.py — 偏侧化指数分析
- [ ] N15: learning_dynamics.py — 学习阶段时间窗动态
- [ ] N16: mediation_analysis.py — 中介分析
- [ ] N17: effective_connectivity.py — 有效连接

### Phase 5：新建统计工具脚本（N18-N21）

- [ ] N18: gPPI_analysis.py — gPPI功能连接
- [ ] N19: behavior_lmm.py — 行为LMM
- [ ] N20: bayesian_stats.py — 贝叶斯因子
- [ ] N21: bootstrap_ci.py — Bootstrap CI

### Phase 6：论文撰写

- [ ] 预注册（OSF）
- [ ] 方法部分撰写
- [ ] PNAS标准图表制作
- [ ] BIDS数据 + GitHub代码共享

***

> **总结**：本研究方案充分利用了metaphoric项目已实现的9大分析模块（行为、GLM、LSS、ROI-MVPA、Searchlight-MVPA、学习RSA、前后测RSA、GPU置换、增强Pipeline），并从gjxx项目移植了4类关键方法（PCA维度分析、GPS全局模式相似性、ROI-to-ROI DSM、种子连接性Searchlight），同时基于14篇文献的系统Gap分析补充了6项关键分析（参数调制、偏侧化指数、情绪效价调节、有效连接、学习动态时间窗、中介分析），形成了从行为到激活到多变量模式到表征几何的完整多层次证据链。共需新建21个脚本，其中6个为高优先级（N1-N6），6个为中优先级（N7-N12），5个为文献Gap补充（N13-N17），4个为统计工具补充（N18-N21）。gjxx的核心算法（dimension_analysis_core.py, roi.py, dimension_story_utils.py）可大幅降低新建脚本的开发工作量。
