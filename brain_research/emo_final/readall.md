# Cognitive Reappraisal Development Engages the Somato-Cognitive Action Network

## 面向 Nature Neuroscience 的完整故事框架

---

## 一句话概括

> 在 6–18 岁期间，情绪调控的发育并非单一脑区的增强，而是情绪神经表征的**"收敛中心"**沿着皮层大尺度梯度（S-A Axis）发生了从"感觉运动端"向"跨模态联合端"的**动态迁徙**。这一迁徙不仅涉及高阶认知网络（DMN/FPCN），还出人意料地涉及运动皮层中嵌入的**躯体-认知行动网络（SCAN）**——认知重评的发育成熟同时需要"想到"（认知重构）和"做到"（身体去动员化）两条通路的协同发育。

---

## 核心叙事逻辑（四段论）

### 第一幕：破局 — 空间双重分离（The Dissociation）

情绪感知（Passive）和调节（Reappraisal）的发育轨迹在空间上是**完全解离**的：

- **感知发育**锚定在躯体运动皮层（SomMot）与杏仁核 → 具身情绪反应的成熟
- **调节发育**锚定在默认模式网络（DMN）、背侧注意网络（DAN）和海马体 → 语义重构与注意部署策略的成熟

关键统计：通过 ΔISC 交互检验（而非简单对比两条件各自显著性），证明两种发育模式在统计上具有显著差异。

### 第二幕：升华 — 梯度迁徙与 SCAN 的浮现（The Gradient Shift & SCAN）

这种解离不是随机的。将发育效应量映射到 Margulies (2016) 的皮层主梯度（G1）上后发现：

- **Passive 条件**：发育效应量与 G1 呈负相关 → 收敛中心在感觉运动端
- **Reappraisal 条件**：发育效应量与 G1 呈正相关 → 收敛中心迁徙到联合皮层端

两条回归线在梯度空间中形成**完美的交叉**，直接可视化了"状态依赖的拓扑迁徙"。

**SCAN 子线索**：在梯度散点图上标注 Gordon (2023) 发现的**躯体-认知行动网络（SCAN）**对应 ROI 后发现，SCAN 的 inter-effector 和 medial motor 区域在 Reappraisal 条件下表现出比全皮层均值更强的发育效应。这一发现提示：认知重评的发育不仅依赖高阶认知网络的成熟，还出人意料地涉及运动皮层中嵌入的**行动控制网络**——情绪调控的本质是**取消已发起的行动准备**，而 SCAN 正是执行"认知重评 → 躯体去动员化"下行控制的关键枢纽。

### 第三幕：机制 — 表征连通性（The Connectivity）

皮层并不是孤立发育的。三条表征连通性通路随年龄分化发展：

- **海马-DMN**（Reappraisal 条件）：随年龄显著增强 → 支持"情境重构（Episodic Reframing）"的语义引擎
- **SCAN-DMN**（Reappraisal 条件）：随年龄增强 → 认知决策网络与行动控制网络之间的信息桥梁，体现"想到"→"做到"通路的发育成熟
- **杏仁核-SomMot**（Passive 条件作对照）：相对稳定 → 自动化情绪反应回路已在早期成熟

### 第四幕：意义 — 行为锚定（The Behavioral Anchor）

在重评任务中，跨模态皮层（DMN/DAN + SCAN）的神经表征越接近"18 岁成熟模板"的青少年，其真实的情绪调节能力越强。通过 `--include-scan` 标志将 SCAN ROI 纳入成熟度模板后，**表征成熟度指数（Maturity Index）**与行为指标的偏相关（控制年龄）得到增强，证明情绪调控的成熟同时需要"想到"（认知重构）和"做到"（身体去动员化）两条通路。

---

## 分析流程总览

### 已有 Pipeline（Phase 1 — 已完成）

```
LSS单试次GLM
  → build_roi_repr_matrix.py    # Step 1: 构建 232 ROI 的表征矩阵
    → calc_roi_isc_by_age.py    # Step 2: 计算被试间表征相似性（ISC）
      → joint_analysis_roi_isc_dev_models.py  # Step 3: 发育模型置换检验
        → plot_brain_surface_vol.py           # Step 5: 脑图渲染
        → plot_roi_isc_age_trajectory.py      # Step 6: 年龄轨迹图
```

**行为分支**：
```
behavior/build_behavior_trial_repr_matrix.py  → 行为表征矩阵
behavior/calc_behavior_isc_by_age.py          → 行为ISC
behavior/joint_analysis_roi_isc_behavior.py   → 脑-行为ISC关联
behavior/joint_analysis_roi_isc_behavior_age_regression.py → 年龄调制交互
```

### 新增 Pipeline（Phase 2 — 本次新增）

```
已有产出 (roi_isc_dev_models_perm_fwer.csv + ISC .npy)
  ↓
step1_task_dissociation.py     # Result 1: ΔISC 交互检验 → 空间双重分离
  ↓
step2_gradient_shift.py        # Result 2: S-A 梯度映射 + Spin Test → 拓扑迁徙
  ↓
step3_representational_connectivity.py  # Result 3: 海马-DMN 表征连通性发育
  ↓
step4_maturity_index.py        # Result 4: 成熟度指数 → 行为锚定
```

## 辅助工具

```
utils_network_assignment.py    # Schaefer 200 → 7Networks 映射 + Tian S2 → 皮下结构映射 + SCAN 坐标映射
```

---

## 各脚本详细说明

### utils_network_assignment.py

**功能**：提供 ROI 到脑网络/皮下结构的标准化映射，以及 SCAN 网络的坐标映射。

**皮层 & 皮下映射函数**：
- `get_schaefer200_network_labels()` → 200 行 DataFrame，含 roi/network/hemi
- `get_tian_s2_labels()` → 32 行 DataFrame，含 roi/structure/structure_group
- `get_all_roi_info()` → 232 行合并 DataFrame
- `get_roi_network_map()` → {roi: network} 字典
- `get_sensorimotor_association_order()` → S-A 轴排序的 7 网络列表

**SCAN 映射函数**（Gordon 2023 Nature）：
- `SCAN_REGIONS` → 14 个 SCAN 区域的 MNI 坐标（Table S1 Avg）
- `SCAN_FUNCTIONAL_GROUPS` → 3 组功能分组（inter_effector / medial_motor / subcortical）
- `get_scan_roi_mapping(distance_threshold=20.0)` → 每个 SCAN 区域的最近 Schaefer parcel + 距离
- `get_scan_cortical_rois()` → 阈值内的皮层 ROI 列表
- `get_scan_subcortical_rois()` → SCAN 对应的 Tian S2 ROI 列表
- `get_scan_all_rois()` → 皮层 + 皮下全部 SCAN ROI

**新增：一键生成 SCAN 映射 QC（推荐）**

该脚本新增 CLI 入口，用于输出 SCAN→ROI 映射的逐条明细与分布摘要，便于确认“最近 parcel + 距离阈值”的映射质量是否合理（以及离线环境是否能稳定运行）。

**输出**：
- `gradient_analysis/scan_mapping_qc.csv`：逐条映射明细（含 `scan_region/nearest_roi/distance_mm/within_threshold/scan_group` 等）
- `gradient_analysis/scan_mapping_qc_summary.csv`：汇总统计（overall + 分组 + cortical_only）

**离线 atlas/centroids 选项**（避免 `nilearn` 在线下载）：
- `--atlas-source auto|centroids|nifti|nilearn`
- `--atlas-nifti /path/to/Schaefer200.nii.gz`：本地 Schaefer200 atlas NIfTI（labels 1..200）
- `--centroids-file /path/to/centroids.csv|.npy`：本地 centroid 缓存（CSV/NPY）

**运行示例**：
```bash
# 1) 优先离线（建议）：有 centroids_file/atlas_nifti 就不会触发在线下载
python utils_network_assignment.py \
  --matrix-dir /path/to/roi_results_final \
  --atlas-source auto \
  --distance-threshold 20

# 2) 完全离线：指定本地 NIfTI atlas 计算 centroids
python utils_network_assignment.py \
  --matrix-dir /path/to/roi_results_final \
  --atlas-source nifti \
  --atlas-nifti /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz \
  --distance-threshold 20
```

**Schaefer 200 × 7Networks 映射**：

| 网络 | 缩写 | 左半球 ROI 范围 | 右半球 ROI 范围 |
|------|------|----------------|----------------|
| Visual | Vis | L_1–L_13 | R_1–R_13 |
| Somatomotor | SomMot | L_14–L_30 | R_14–R_30 |
| Dorsal Attention | DorsAttn | L_31–L_43 | R_31–R_43 |
| Salience/Ventral Attention | SalVentAttn | L_44–L_52 | R_44–R_53 |
| Limbic | Limbic | L_53–L_59 | R_54–R_59 |
| Frontoparietal Control | Cont | L_60–L_72 | R_60–R_72 |
| Default Mode | Default | L_73–L_100 | R_73–R_100 |

**Tian S2 × 32 皮下结构映射**（每侧 16 个）：

| V_id (左/右) | 结构 | 结构组 |
|---|---|---|
| V_1 / V_17 | HIP-head | Hippocampus |
| V_2 / V_18 | HIP-body | Hippocampus |
| V_3 / V_19 | AMY | Amygdala |
| V_4–V_7 / V_20–V_23 | THA 各亚区 | Thalamus |
| V_8–V_9 / V_24–V_25 | CAU 各亚区 | Caudate |
| V_10–V_11 / V_26–V_27 | PUT 各亚区 | Putamen |
| V_12 / V_28 | NAc | NAc |
| V_13–V_14 / V_29–V_30 | GP 各亚区 | GP |
| V_15–V_16 / V_31–V_32 | THA 各亚区 | Thalamus |

---

### step1_task_dissociation.py

**对应 Result 1：空间双重分离**

**科学问题**：被动感知和主动调节的发育趋同（M_conv）是否依赖完全不同的脑网络？

**输入**：
- `by_stimulus/Passive_Emo/roi_isc_mahalanobis_by_age.npy` (shape: n_rois × n_sub × n_sub)
- `by_stimulus/Reappraisal/roi_isc_mahalanobis_by_age.npy`

**核心算法**：
1. 对齐两个条件的被试（取交集，按年龄排序）
2. 对每个 ROI 计算 ΔISC = ISC_Passive - ISC_Reappraisal（上三角对应位置相减）
3. 将 ΔISC 向量与 M_conv 模型做 Spearman 关联
4. 置换检验（双尾）：打乱年龄标签 5000 次 → raw p + maxT FWER + BH-FDR
5. 分类标记：condition_a_dominant / condition_b_dominant / ns

**输出**：
- `gradient_analysis/roi_isc_task_dissociation.csv`
  - 新增 SCAN 标注列：`is_scan_overlap`、`nearest_scan_region`、`scan_distance_mm`
  - `dissociation_label` 分类：condition_a_dominant / condition_b_dominant / ns

**运行**：
```bash
python step1_task_dissociation.py \
  --matrix-dir /path/to/roi_results_final \
  --condition-a Passive_Emo \
  --condition-b Reappraisal \
  --model M_conv \
  --n-perm 5000
```

---

### step2_gradient_shift.py

**对应 Result 2：S-A 梯度拓扑迁徙 + SCAN 浮现 (文章最大亮点)**

**科学问题**：发育的"主战场"是否随任务状态沿 S-A 轴发生跨模态推移？SCAN 在此迁徙中扮演什么角色？

**依赖**：
- `neuromaps` 库（用于获取 Margulies 2016 G1 梯度 + Spin Test）
- Step 1 输出（可选，用于 ΔISC 梯度分析）

**新增：SCAN 映射稳健性与负对照（稳健性补充，不改变主分析）**

Step2 在生成 Figure 2 与核心梯度统计的同时，会额外输出两份“SCAN 映射层面”的稳健性文件：
- `gradient_analysis/scan_mapping_sensitivity.csv`：对 `--scan-threshold-grid`（默认 10/15/20/25mm）的阈值敏感性；记录阈值内命中 ROI 数、映射距离分布摘要，以及 SCAN ROI 上 `r_obs` 的汇总统计（A/B/Δ）
- `gradient_analysis/scan_negative_control.csv`：最小版负对照基线（默认从 `SomMot` 网络随机抽样与 SCAN 皮层 ROI 数量相同的 parcels，重复 `--scan-negctl-n` 次），并在 observed 行给出经验 p 值（two-sided/greater/less）

与 SCAN 映射相关的新增参数（支持离线）：
- `--scan-atlas-source`：`nilearn|nifti|centroids|auto`（含义同 `utils_network_assignment.py`）
- `--scan-atlas-nifti` / `--scan-centroids-file`：离线 atlas/centroids 输入
- `--scan-distance-threshold`：主分析使用的 SCAN 阈值（用于 `scan_overlap_analysis.csv`、Figure、负对照的“观测 ROI 集合”）
- `--scan-threshold-grid`：阈值敏感性网格（mm）
- `--scan-negctl-n`：负对照重复次数 N

**核心算法**：
1. 获取 G1 梯度并 parcellate 到 Schaefer 200（仅皮层 ROI）
2. 加载两个条件的 M_conv 发育效应量（r_obs）
3. 计算 Spearman 相关：r_obs vs G1
4. Spin Test（10000 次 Alexander-Bloch 旋转）验证空间相关性
5. 获取 SCAN 对应的皮层 ROI，标注在梯度散点图上
6. 按 SCAN 功能分组（inter_effector / medial_motor）与全皮层对比
7. 生成 2×2 四面板 Figure 2

**输出**：
- `gradient_analysis/gradient_shift_results.csv` — 条件级梯度-发育相关
- `gradient_analysis/roi_gradient_scores.csv` — 逐 ROI 的 G1 分数和 r_obs
- `gradient_analysis/scan_overlap_analysis.csv` — SCAN 各区域在两条件下的发育效应
- `gradient_analysis/fig2_gradient_shift.png` — 出版级 2×2 四面板图
- `gradient_analysis/scan_mapping_sensitivity.csv` — SCAN 阈值敏感性（稳健性补充）
- `gradient_analysis/scan_negative_control.csv` — SCAN 最小负对照（稳健性补充）

**Figure 2 面板说明**：
- [0,0] ROI 级散点：G1 vs 发育效应量（两条件双色 + 回归线交叉）
- [0,1] 网络级柱状图：按 S-A 轴排序的 7 网络发育效应
- [1,0] SCAN 高亮散点：灰底普通 ROI + 星标 SCAN ROI（标注区域名称）
- [1,1] SCAN 功能分组柱状图：Inter-Effector / Medial Motor / 全皮层对比

**预期结果**：
- Passive 条件：r_gradient < 0（感觉运动端效应更强）
- Reappraisal 条件：r_gradient > 0（联合皮层端效应更强）
- 两条线在梯度空间交叉 → **状态依赖的拓扑迁徙**

**运行**：
```bash
# 需要先安装 neuromaps: pip install neuromaps
python step2_gradient_shift.py \
  --matrix-dir /path/to/roi_results_final \
  --condition-a Passive_Emo \
  --condition-b Reappraisal \
  --n-spin 10000
```

**如果无法安装 neuromaps / spin test 失败**：
```bash
# 手动准备 gradient CSV (columns: roi, g1_score)，然后：
python step2_gradient_shift.py --gradient-source manual --gradient-file /path/to/g1_scores.csv
```

说明：
- 当 `neuromaps` 不可用或 `alexander_bloch` 旋转失败时，脚本会自动退回“随机置换近似”来估计 p 值；该近似不保留皮层空间自相关结构，因此应标注为 **exploratory**（用于快速 sanity check，而非确认性空间检验）。
- `gradient_shift_results.csv` 会记录 `spin_test_available/spin_test_method/nulls_shape/spin_test_note`，用于明确区分“确认性 spin test”与“探索性 fallback”。

---

### step3_representational_connectivity.py

**对应 Result 3：表征连通性的发育（含 SCAN-DMN 通路）**

**科学问题**：DMN 在重评中的核心地位如何通过网络间表征交换来实现？SCAN 与 DMN 之间是否也存在发育性表征耦合？

**核心算法**：
1. 对每个被试，提取两个 ROI 的 RSM → 展平上三角 → Spearman 相关 = RC_score
2. 网络级聚合：对所有跨组 ROI 对取均值
3. 支持 "SCAN" 作为特殊组名：自动调用 `get_scan_cortical_rois()` 获取 SCAN 对应的皮层 ROI
4. RC_score ~ age 回归 + 置换检验
5. 生成散点 + LOWESS 拟合图

**预设网络对**：
- **海马-DMN**（Reappraisal 条件）：预期随年龄增强 → 情境重构引擎
- **杏仁核-SomMot**（Passive 条件作对照）：预期相对稳定
- **SCAN-DMN**（Reappraisal 条件）：预期随年龄增强 → "想到→做到"通路

**新增参数**：
- `--condition-contrast`：可选，指定对比条件（如 Passive_Emo）
- `--rc-aggregate`：网络级 RC 聚合方式：`mean`（默认）或 `median`（稳健性补充），并在 stats 中记录 `n_roi_pairs`
- `--dev-modeling`：可选（默认关闭），对每个 `pair_label` 做线性 vs 二次模型对比（发育非线性补充）
- `--dev-model-criterion`：`AIC|R2_adj`（模型选择准则）
- `--dev-model-min-n`：模型对比所需最小样本数（默认 8）
- `--skip-repr-complexity`：跳过表征复杂性/区分度补充分析（默认会输出）
- `--repr-complexity-level`：`roi|network|both`（输出层级）

**输出**：
- `gradient_analysis/repr_connectivity_subject_scores.csv`
- `gradient_analysis/repr_connectivity_stats.csv`
  - 网络级行新增：`n_roi_pairs`、`n_rois_group_a`、`n_rois_group_b`、`rc_aggregate`
- `gradient_analysis/fig3_representational_connectivity.png`（N 面板，每个网络对一个）
- `gradient_analysis/fig3_representational_connectivity_contrast.png`（当恰好 2 个网络对时生成对比图）
- `gradient_analysis/repr_connectivity_subject_scores_<condition>.csv`（指定 `--condition-contrast` 时额外输出）
- `gradient_analysis/repr_connectivity_stats_<condition>.csv`（指定 `--condition-contrast` 时额外输出）
- `gradient_analysis/fig3_representational_connectivity_condition_overlay.png`（指定 `--condition-contrast` 时生成双条件叠加图）
- `gradient_analysis/repr_complexity_subject_scores.csv`（表征复杂性/区分度补充分析）
- `gradient_analysis/repr_complexity_stats.csv`（表征复杂性/区分度补充分析统计与置换）
- `gradient_analysis/repr_connectivity_dev_model_comparison.csv`（启用 `--dev-modeling` 时输出；线性 vs 二次对比摘要）
- `gradient_analysis/repr_connectivity_dev_model_comparison_<condition>.csv`（启用 `--dev-modeling` 且指定 `--condition-contrast` 时额外输出）

**运行**：
```bash
python step3_representational_connectivity.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --network-pairs "Hippocampus:Default" "Amygdala:SomMot" "SCAN:Default"

# 若需要与 Passive 条件直接对照
python step3_representational_connectivity.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --condition-contrast Passive_Emo \
  --network-pairs "Hippocampus:Default" "SCAN:Default"

# 若启用发育非线性模型对比（补充，默认关闭）
python step3_representational_connectivity.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --network-pairs "Hippocampus:Default" "SCAN:Default" \
  --dev-modeling --dev-model-criterion AIC
```

---

### step4_maturity_index.py

**对应 Result 4：行为锚定（SCAN 增强版成熟度模板）**

**科学问题**：大脑向 18 岁模板趋同，是否带来了现实中情绪调节能力的提升？将 SCAN ROI 纳入模板后预测力是否增强？

**核心算法**：
1. 筛选 ≥17 岁被试作为成熟组
2. 对重评专属网络（DMN + DAN + 海马）的 ROI，构建成熟模板（RSM 上三角均值）
3. 若启用 `--include-scan`，额外将 SCAN 对应的皮层 ROI 合并入目标 ROI 集
4. 对每个被试计算 Maturity Index = spearman(被试 mega_vec, 模板 mega_vec)
5. MI ~ age 相关（sanity check）
6. MI ~ 行为指标偏相关（控制年龄）
7. 生成 Figure 4

**新增参数**：
- `--include-scan`：布尔标志，启用后自动将 SCAN 皮层 ROI 添加到目标 ROI 集合中
- `--behavior-cols`：行为列名列表（建议在可用时包含 `reappraisal_success`；脚本会尝试从提供的行为表生成该列）
- `--mature-age-grid`：成熟阈值敏感性网格（不改变主分析的 `--mature-age-min`）
- `--bootstrap-template`：成熟模板 bootstrap 次数 B（0 关闭），输出 MI 稳定性统计
- `--loo-template`：最小版 leave-one-out 模板影响分析
- `--behavior-join`：行为表与 MI 表合并方式：`left`（默认，避免静默丢人）或 `inner`（仅保留有行为的被试）

**输出**：
- `gradient_analysis/maturity_index_subject_scores.csv`
- `gradient_analysis/maturity_index_stats.csv`
- `gradient_analysis/maturity_index_config.csv`（含 `include_scan` 标记和完整 target_rois 列表）
- `gradient_analysis/fig4_maturity_index.png`
- `gradient_analysis/maturity_index_sensitivity.csv`（`--mature-age-grid`：成熟阈值敏感性）
- `gradient_analysis/maturity_index_stability.csv`（`--bootstrap-template`：模板稳定性，含 subject-level mean/SD/CI/CV 与 global 汇总）
- `gradient_analysis/maturity_index_loo.csv`（`--loo-template`：LOO 模板影响）

**运行**：
```bash
# 标准版（不含 SCAN）
python step4_maturity_index.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --target-networks Default DorsAttn \
  --include-subcortical Hippocampus \
  --mature-age-min 17.0

# SCAN 增强版（推荐）
python step4_maturity_index.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --target-networks Default DorsAttn \
  --include-subcortical Hippocampus \
  --include-scan \
  --mature-age-min 17.0
```

**如果有外部行为数据**：
```bash
python step4_maturity_index.py \
  --behavior-file /path/to/behavior_scores.csv \
  --behavior-cols reappraisal_success anxiety_score \
  --include-scan
```

**稳健性补充示例**：
```bash
# 1) 成熟阈值敏感性
python step4_maturity_index.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --include-scan \
  --mature-age-grid 16.5 17 17.5 18

# 2) 模板稳定性（bootstrap）+ LOO
python step4_maturity_index.py \
  --matrix-dir /path/to/roi_results_final \
  --condition Reappraisal \
  --include-scan \
  --bootstrap-template 1000 \
  --loo-template
```

---

## 预期 Figures 对照表

| Figure | 脚本 | 内容 |
|--------|------|------|
| Fig 1A | step1 + plot_brain_surface_vol.py | 两张并列脑图：Passive 专属网络（蓝）vs Reappraisal 专属网络（红），SCAN 重叠 ROI 标注 |
| Fig 1B | plot_roi_isc_age_trajectory.py | 核心 ROI 的年龄 vs ISC 散点 + LOWESS（杏仁核 vs DMN） |
| Fig 2A | step2 [0,0] | G1 梯度 vs 发育效应量散点（两条件两色线交叉） |
| Fig 2B | step2 [0,1] | 网络级柱状图（按 S-A 轴排序） |
| Fig 2C | step2 [1,0] | SCAN ROI 高亮散点（星标 + 区域标注） |
| Fig 2D | step2 [1,1] | SCAN 功能分组柱状图（Inter-Effector / Medial Motor / 全皮层） |
| Fig 3A | step3 | 海马-DMN 表征连通性 ~ 年龄（上升趋势） |
| Fig 3B | step3 | 杏仁核-SomMot 表征连通性 ~ 年龄（对照，稳定） |
| Fig 3C | step3 | SCAN-DMN 表征连通性 ~ 年龄（"想到→做到"通路） |
| Fig 4 | step4 | 成熟度指数 ~ 年龄 + 成熟度 ~ 行为偏相关（SCAN 增强版模板） |

补充说明（非 Figure 主面板，但用于稳健性/复现）：
- Step2 额外输出：`scan_mapping_sensitivity.csv`、`scan_negative_control.csv`（SCAN 映射稳健性与负对照）
- Step3 额外输出：`repr_complexity_*.csv`（排他解释补充）、`repr_connectivity_dev_model_comparison*.csv`（非线性发育建模补充）
- Step4 额外输出：`maturity_index_sensitivity.csv`、`maturity_index_stability.csv`、`maturity_index_loo.csv`（模板稳健性）

---

## 执行顺序

```
# Phase 1（已有，无需重跑除非更改参数）
python run_pipeline.py --config config_trial.json

# （可选）如果希望 pipeline 自动跑 Phase 2（Step1-4）
python run_pipeline.py --config config_trial.json --run-phase2

# Phase 2（新增分析，按顺序执行）
python step1_task_dissociation.py --matrix-dir ... --n-perm 5000

# （可选但推荐）先生成/检查 SCAN 映射质量，尤其是离线环境
python utils_network_assignment.py --matrix-dir ... --atlas-source auto --distance-threshold 20

python step2_gradient_shift.py --matrix-dir ... --n-spin 10000
python step3_representational_connectivity.py --matrix-dir ... --condition Reappraisal --network-pairs "Hippocampus:Default" "Amygdala:SomMot" "SCAN:Default"
python step4_maturity_index.py --matrix-dir ... --condition Reappraisal --include-scan
```

所有 Phase 2 输出保存在 `matrix_dir/gradient_analysis/` 目录下，不影响 Phase 1 的已有产出。

---

## 与现有代码的兼容性

- **零修改现有脚本**：所有新增分析读取 Phase 1 的产出文件，不修改任何已有代码
- **输出隔离**：新增分析的所有产出保存在 `gradient_analysis/` 子目录
- **模块复用**：通过 import 复用 `build_models()`、`assoc()`、`flatten_upper()` 等已有函数
- **命名约定一致**：argparse 参数风格、文件命名、CSV 列名均与已有脚本保持一致
- **SCAN 分析为纯增量**：
  - SCAN 映射支持离线 atlas/centroids（`--scan-atlas-source auto|nifti|centroids`），避免在线下载
  - 当环境缺少 `neuromaps` 时，Step2 的空间检验会退回随机置换近似；建议将该 p 值作为 exploratory 标记使用

---

## SCAN 理论框架

### 什么是 SCAN？

SCAN（Somato-Cognitive Action Network）由 Gordon et al. (2023) 在 Nature 发表，基于对 HCP 等大规模数据集的精细个体化网络分析发现：传统运动皮层（M1/SMA）中嵌入了一个**跨效应器整合网络**，它并非编码具体的手/脚/口部运动执行，而是负责：

1. **跨效应器行动规划**（Inter-effector integration）
2. **认知-运动界面**（Cognitive-motor interface，通过 preSMA/dACC 枢纽）
3. **内脏-躯体调节**（Putamen/Thalamus 下行通路）

### 为什么 SCAN 与情绪调节发育相关？

**核心洞察：情绪 = 行动准备程序**

1. 看到威胁性情绪图片 → 杏仁核触发 → 身体进入"应战/逃跑"准备状态
2. 认知重评 = **取消已发起的行动准备** → 需要一个网络来执行这种"认知决策 → 身体去动员化"的下行控制
3. SCAN 正是执行这一功能的关键枢纽

### 三层故事结构

| 层级 | 传统观点 | 本研究发现 |
|------|---------|-----------|
| Layer 1 | 情绪调节 = PFC 抑制杏仁核 | 情绪调节的发育涉及整个 S-A 轴的拓扑重组 |
| Layer 2 | 运动皮层仅负责运动执行 | 运动皮层中的 SCAN 子网络参与认知重评的发育 |
| Layer 3 | 发育重点在"想到"（认知网络成熟） | 发育同时需要"想到"和"做到"两条通路协同成熟 |

---

## 理论框架参考文献

1. Margulies et al. (2016) PNAS — S-A 主梯度定义
2. Sydnor et al. (2021) PNAS — 保守/破坏型发育模式
3. Baum et al. (2020) PNAS — 结构-功能耦合发育
4. Cui et al. (2020) Nature Communications — 个体化功能网络拓扑
5. Dong et al. (2021) — 功能连接沿 S-A 轴的发育增强
6. Marek et al. (2022) Nature — 大样本可复现性
7. **Gordon et al. (2023) Nature** — SCAN 网络的发现与定义（本研究 SCAN 分析的直接理论来源）
