# brain_research/emo 项目脚本分析

## 一、整体工作流程

这个项目是一个情绪脑网络发育研究的 fMRI 数据分析流程，主要分为以下几个阶段：

```
1. 数据预处理 (GLM + LSS) 
    ↓
2. ROI 特征提取 
    ↓
3. 被试间相似性计算 (ISC)
    ↓
4. 与发育模型的联合分析 + 统计检验
```

---

## 二、各脚本详细功能介绍

### 阶段 1：数据预处理

| 脚本 | 功能说明 |
|------|---------|
| **glm_config.py** | 全局配置文件，定义路径、并行数、数据空间（surface/volume）等核心参数 |
| **glm_utils.py** | 工具函数库，包括：<br>- 复杂头动回归（Friston-24 + aCompCor + Scrubbing）<br>- 事件分类（Passive/Reappraisal + Neutral/Emotional）<br>- LSS 事件创建 |
| **lss_main.py** | LSS（Least Squares Separate）单试次建模主程序：<br>- 支持 Surface/Volume 两种数据空间<br>- 对每个 trial 单独建模提取 beta<br>- 输出包含 `stimulus_content` 的索引文件用于跨被试对齐 |

---

### 阶段 2：ROI 特征提取

这组脚本用于从全脑数据中提取特定 ROI（脑区）的特征。

| 脚本 | 功能说明 | 对比分析 |
|------|---------|---------|
| **build_roi_beta_matrix_232.py** | 构建 232 个 ROI 的 beta 矩阵<br>- 200个 Schaefer 脑区（左右各100）+ 32个 Tian 皮下核团<br>- 使用 **均值** 聚合 ROI 内体素<br>- 输出维度：`(232, n_subjects, n_stimuli)` | **核心差异**：这个脚本计算 ROI 内的**均值**，数据降维为标量 |
| **build_roi_beta_matrix_232_median.py** | 同上，但使用 **中位数** 聚合 ROI 内体素 | **核心差异**：使用**中位数**而非均值，对异常值更稳健 |

---

### 阶段 3：被试间相似性计算 (ISC)

这些脚本计算被试之间的相似性矩阵（Inter-Subject Correlation）。

| 脚本 | 功能说明 | 对比分析 |
|------|---------|---------|
| **calc_isc_combined_strict.py** | 计算**全脑**的被试间相似性矩阵（按年龄排序）<br>- 支持 Surface L/R 和 Volume<br>- 严格筛选：保被试、丢刺激<br>- 输出：`similarity_*_AgeSorted.csv` | **全脑级**分析，不做 ROI 分割 |
| **calc_isc_roi_aligned.py** | 计算**特定 ROI** 的相似性矩阵<br>- 预定义了 Visual、Salience_Insula、Control_PFC 三个网络<br>- 同样按年龄排序 | **ROI级**分析，但只针对预定义的3个网络 |
| **calc_roi_isc_by_age.py** | 基于 Step1 输出的 ROI beta 矩阵计算 ISC<br>- 使用 **Spearman 秩相关**<br>- 按年龄排序被试<br>- 输出：`(n_roi, n_sub, n_sub)` 的三维矩阵 | **Step2专用**，依赖 Step1 的输出，用 Spearman 相关 |
| **calc_roi_isc_pearson_direct.py** | 一站式计算 ROI 级 ISC<br>- 整合了 Step1 + Step2 的功能<br>- 保留空间维度（Flatten &amp; Concatenate，不求平均）<br>- 使用 **Pearson 相关** | **一站式**，不依赖中间文件，用 Pearson 相关，保留空间信息 |

---

### 阶段 4：联合分析与统计检验

这些脚本将相似性矩阵与发育模型进行关联分析，并做置换检验。

| 脚本 | 功能说明 | 对比分析 |
|------|---------|---------|
| **joint_analysis_similarity_dev_models_perm_fwer.py** | 对**全脑 + 预定义ROI** 的8个相似性矩阵做联合分析<br>- 3个发育模型：M_nn（最近邻）、M_conv（收敛）、M_div（发散）<br>- 统计量：Pearson r<br>- 多重比较：Model-wise FWER（每个模型内部校正） | **8个矩阵**（全脑L/R + 3ROI×L/R） |
| **combined_roi_joint_analysis_dev_models_perm_fwer.py** | 功能同上，但**强制被试对齐**<br>- 读取 `subject_order_surface_L.csv` 作为基准<br>- 支持多元回归（MRM/MRQAP 风格）<br>- 支持 "corr"、"mrm" 或 "both" 模式 | **被试严格对齐**，增加 MRM 模式 |
| **joint_analysis_roi_isc_dev_models_perm_fwer.py** | 专门用于**ROI级**矩阵的联合分析<br>- 输入：Step2 输出的三维 ISC 矩阵<br>- 同样做 3 个发育模型的置换检验 | **Step3专用**，依赖 Step2 的输出 |
| **parcel_roi_joint_analysis.py** | **网络级 + parcel级** 分析<br>- 支持按网络聚合或单个 parcel 分析<br>- 智能解析 Atlas 标签（兼容 Schaefer 和 Tian）<br>- 正确识别皮下组织（Subcortex） | **灵活的分析级别**，支持 network 或 parcel |
| **all_parcels_joint_analysis.py** | **所有脑区**的联合分析<br>- 不再进行"网络→脑区"的分层<br>- 直接提取所有脑区放入同一个池子<br>- 全局 FWER 校正 | **所有parcel一起分析**，不按网络分层 |

---

## 三、推荐分析流程

### 方案 A：旧链路（分步走）
```
1. lss_main.py                    # 单试次建模
   ↓
2. build_roi_beta_matrix_232.py   # 提取 ROI 特征（均值）
   ↓
3. calc_roi_isc_by_age.py         # 计算 Spearman ISC
   ↓
4. joint_analysis_roi_isc_dev_models_perm_fwer.py  # 统计检验
```

### 方案 B：新链路（一键式，在 emo_new 目录）
```
run_roi_isc_dev_models_pipeline.py  # 串联 Step1-3
```

### 方案 C：全脑 + ROI 分析
```
1. lss_main.py
   ↓
2. calc_isc_combined_strict.py     # 全脑相似性
3. calc_isc_roi_aligned.py         # ROI相似性
   ↓
4. joint_analysis_similarity_dev_models_perm_fwer.py  # 联合分析
```

---

## 四、关键概念解释

### 1. 3个发育模型
- **M_nn (Nearest Neighbor)**：年龄越相近，相似度越高
- **M_conv (Convergence)**：年龄越大，相似度越高（都大才像）
- **M_div (Divergence)**：年龄越大，相似度越低（越大约不一样）

### 2. 232个 ROI
- Schaefer 200 Parcels（左100 + 右100）
- Tian Subcortex S2（32个皮下核团）

### 3. 统计方法
- 置换检验（Label Permutation）
- 单侧检验（只检验正效应）
- Model-wise FWER 校正（每个模型内部独立校正）
