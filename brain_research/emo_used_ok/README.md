# Emo_used_ok 脚本使用说明

本目录包含了一套完整的 fMRI 分析流程，从单试次建模 (LSS) 到被试间相似性计算，再到发育模型联合分析。

---

## 一、脚本概览

| 脚本文件名 | 主要功能 | 依赖关系 |
|-----------|---------|---------|
| `glm_config.py` | **全局配置文件**：定义路径、数据空间、被试列表等 | 无 |
| `glm_utils.py` | **工具函数库**：头动处理、事件分类、LSS 事件生成 | 无 |
| `lss_main.py` | **LSS 分析主程序**：执行单试次 GLM 建模 | `glm_config.py`, `glm_utils.py` |
| `calc_isc_combined_strict.py` | **全脑 ISC 计算**：按年龄排序计算被试间相似性矩阵 | `lss_main.py` 的输出 |
| `calc_isc_roi_aligned.py` | **ROI 水平 ISC 计算**：基于 Schaefer 图谱计算特定脑区 | `lss_main.py` 的输出 |
| `joint_analysis_similarity_dev_models.py` | **发育模型联合分析**：相似性矩阵与 3 种发育模型的相关性 | `calc_isc_*` 的输出 |
| `joint_analysis_similarity_dev_models_perm_fwer.py` | **置换检验 + FWER 校正**：8 个矩阵的批量检验 | `calc_isc_*` 的输出 |

---

## 二、详细功能说明

### 1. 配置与工具层

#### `glm_config.py`
- **核心功能**：
  - 定义数据路径映射（影像在 midprep，头动在 miniprep）
  - 支持多种数据空间：`surface` (左/右脑) 和 `volume` (MNI)
  - 自动获取被试列表
  - 提供 `get_paths()` 方法统一获取 fMRI、事件、头动文件路径
- **关键配置项**：
  - `BIDS_DATA_DIR`：数据根目录（通过环境变量设置）
  - `OUTPUT_ROOT`：结果输出目录
  - `RUN_MAP`：定义 EMO 和 SOC 两个任务的路径映射

#### `glm_utils.py`
- **核心功能**：
  1. `load_complex_confounds()`：复杂头动回归策略
     - Friston-24（6参数 + 6导数 + 6平方 + 6导数平方）
     - aCompCor（前6个分量）
     - Scrubbing（motion_outlier + non_steady_state）
  2. `reclassify_events()`：事件 3 分类
     - `Reappraisal`：重评条件
     - `Passive_Neutral`：被动观看中性刺激
     - `Passive_Emo`：被动观看情绪刺激
  3. `create_lss_events()`：LSS 事件生成
     - Target vs Others 策略
     - 适配 E-Prime 点分隔列名格式

---

### 2. 单试次建模层

#### `lss_main.py`
- **主要功能**：
  - 兼容 Surface 和 Volume 两种数据空间
  - Surface 模式采用「伪装 NIfTI」策略处理
  - 并行处理多个被试和 run
  - **核心输出**：
    - `lss_index_*_aligned.csv`：包含 `stimulus_content` 字段的对齐索引（实现跨被试对齐）
    - `beta_*.gii` / `beta_*.nii.gz`：单试次 beta 图像
    - `lss_audit_*.csv`：审计日志
- **使用流程**：
  ```bash
  python lss_main.py
  ```

---

### 3. 被试间相似性计算层

#### `calc_isc_combined_strict.py`（全脑）
- **主要功能**：
  - 读取被试年龄表（支持 .xlsx/.tsv/.csv）
  - **按年龄从小到大排序**被试
  - 基于 `stimulus_content` 筛选公共刺激（阈值：90% 被试）
  - 计算被试×被试相似性矩阵（Pearson 相关）
- **输出**：
  - `similarity_*_AgeSorted.csv`：相似性矩阵
  - `subject_order_*.csv`：被试排序参考表

#### `calc_isc_roi_aligned.py`（ROI 水平）
- **与上一个脚本的对比**：
  | 特性 | `calc_isc_combined_strict.py` | `calc_isc_roi_aligned.py` |
  |-----|-------------------------------|---------------------------|
  | 分析范围 | 全脑（Surface/Volume） | 特定 ROI（仅 Surface） |
  | 使用图谱 | 无 | Schaefer 200 Parcels 7Networks |
  | 预定义 ROI | 无 | Visual, Salience_Insula, Control_PFC |
  - **核心差异**：前者计算全脑每个顶点的相似性，后者只提取特定脑区的顶点进行计算
- **输出**：每个 ROI 单独的相似性矩阵

---

### 4. 发育模型联合分析层

#### `joint_analysis_similarity_dev_models.py`
- **主要功能**：
  - 读取相似性矩阵和被试年龄
  - 构建 3 种发育模型矩阵：
    1. **M_nn（最近邻模型）**：`amax - |age_i - age_j|`
    2. **M_conv（收敛性模型）**：`min(age_i, age_j)`
    3. **M_div（发散性模型）**：`amax - 0.5*(age_i + age_j)`
  - 计算相似性矩阵上三角向量与模型向量的 Pearson 相关
- **输出**：
  - `dev_model_nn_*.csv`、`dev_model_conv_*.csv`、`dev_model_div_*.csv`：三个发育模型矩阵
  - `joint_analysis_dev_models_*.csv`：相关性结果（r, p, n）

#### `joint_analysis_similarity_dev_models_perm_fwer.py`
- **与上一个脚本的对比**：
  | 特性 | `joint_analysis_similarity_dev_models.py` | `joint_analysis_similarity_dev_models_perm_fwer.py` |
  |-----|-------------------------------------------|---------------------------------------------------|
  | 分析对象 | 单个相似性矩阵 | 8 个矩阵（全脑 L/R + 3 ROI×L/R） |
  | 显著性检验 | 参数 p 值 | 置换检验（label permutation） |
  | 多重比较 | 无 | Westfall–Young max-statistic FWER 校正 |
  - **核心差异**：前者是单个矩阵的简单相关分析，后者是批量统计检验，包含严格的多重比较校正
- **处理的 8 个矩阵**：
  1. surface_L（全脑左脑）
  2. surface_R（全脑右脑）
  3. surface_L_Visual
  4. surface_R_Visual
  5. surface_L_Salience_Insula
  6. surface_R_Salience_Insula
  7. surface_L_Control_PFC
  8. surface_R_Control_PFC

---

## 三、完整使用流程

```
1. 配置环境
   ↓
2. 运行 lss_main.py → 生成单试次 beta 和对齐索引
   ↓
3. 运行 calc_isc_combined_strict.py → 生成全脑相似性矩阵
   ↓
4. 运行 calc_isc_roi_aligned.py → 生成 ROI 水平相似性矩阵
   ↓
5. 运行 joint_analysis_similarity_dev_models.py → 单个矩阵的相关性分析（可选）
   ↓
6. 运行 joint_analysis_similarity_dev_models_perm_fwer.py → 批量置换检验 + FWER 校正
```

---

## 四、注意事项

1. **环境变量**：需设置 `BIDS_DATA_DIR` 指向数据根目录
2. **数据格式**：事件文件需包含 `condition` 和 `emotion` 列
3. **年龄表**：支持 `sub_id/age` 或 `被试编号/采集年龄` 两种列名格式
4. **并行处理**：`lss_main.py` 支持并行，Surface 模式建议 `N_JOBS=6`
