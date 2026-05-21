我看了一遍 `emo_final`，它不是单一脚本，而是一个“主分析 + 行为分支 + Phase 2 论文结果模块”的流程。最适合汇报的口径是：**用 RSA 构建每个 ROI 的情绪表征，再计算被试间表征相似性 ISC，最后把 ISC 与年龄模型、行为指标和皮层梯度联系起来。**

**总流程**

| 阶段 | 脚本 | 用到的数据分析方法 |
|---|---|---|
| 0. 单试次 beta 提取 | [lss_main.py](E:/PythonAnalysis/brain_research/emo_final/lss_main.py) | LSS 单试次 GLM；每个 trial 一个目标回归量，其他 trial 合并为干扰回归量；`FirstLevelModel`，AR(1) 噪声、SPM HRF + derivative、cosine drift、高通滤波 |
| 1. ROI 表征矩阵 | [build_roi_repr_matrix.py](E:/PythonAnalysis/brain_research/emo_final/build_roi_repr_matrix.py) | RSA / RSM；每个被试、每个 ROI，把 trial beta pattern 做刺激 × 刺激相似矩阵；默认 Spearman 相关，也支持 Pearson；默认做 cocktail-blank 去除整体均值 |
| 1.5 emotion-level 表征 | [build_roi_emotion_repr_matrix.py](E:/PythonAnalysis/brain_research/emo_final/build_roi_emotion_repr_matrix.py) | 先把同一情绪类别的 trial-level beta pattern 平均成 emotion pattern，再重新计算 emotion × emotion RSM；不是简单平均原来的 RSM |
| 2. 脑 ISC | [calc_roi_isc_by_age.py](E:/PythonAnalysis/brain_research/emo_final/calc_roi_isc_by_age.py) | 被试间表征相似性 ISC：把每个被试的 RSM 上三角展开成向量，再计算被试 × 被试相似矩阵；默认 Mahalanobis distance，再转成 similarity；也支持 Spearman/Pearson/Euclidean |
| 3. 年龄模型检验 | [joint_analysis_roi_isc_dev_models.py](E:/PythonAnalysis/brain_research/emo_final/joint_analysis_roi_isc_dev_models.py) | Matrix-level RSA / Mantel-like 思路：将每个 ROI 的被试间 ISC 向量，与三种年龄模型做 Spearman/Pearson 关联；多重比较用 BH-FDR 或 permutation + maxT FWER |
| 4. 可视化 | `plot_*` 系列 | 显著 ROI 热图、脑表面/皮下映射、年龄轨迹图；轨迹图支持 hexbin、线性拟合、LOWESS |

三个年龄模型可以这样给老师解释：

- `M_nn`：near-neighbor model，看年龄相近的被试是否神经表征更相似。公式近似是“最大年龄 - 年龄差”。
- `M_conv`：convergence / maturity model，看两个被试都越年长时，表征是否越相似。公式是两人年龄的较小值。
- `M_div`：young-centered / divergence model，看年轻被试之间是否更相似。公式和两人平均年龄负相关。

**行为分支**
行为分支在 [behavior](E:/PythonAnalysis/brain_research/emo_final/behavior) 下面，逻辑和脑数据平行：

| 阶段 | 脚本 | 方法 |
|---|---|---|
| 行为特征整理 | `behavior_data.py` | 清洗行为数据，导出 trial-level 行为特征，如 `emot_rating` |
| 行为表征矩阵 | `build_behavior_trial_repr_matrix.py` | 对每个被试、每个刺激提取行为特征；默认按 mean 聚合；用 Euclidean / Manhattan / Mahalanobis 距离构建刺激间差异矩阵，再转 similarity |
| 行为 ISC | `calc_behavior_isc_by_age.py` | 把行为 RSM 展开，计算被试 × 被试行为相似性；可按缺失比例过滤被试 |
| 脑-行为关联 | `joint_analysis_roi_isc_behavior.py` | 每个 ROI 的脑 ISC 向量与行为 ISC 向量做 Spearman/Pearson 关联；置换行为被试标签，做 permutation p、maxT FWER、BH-FDR |
| 年龄调节的脑-行为模型 | `joint_analysis_roi_isc_behavior_age_regression.py` | 多元回归 RSA：`Behavior_ISC = β1 Brain_ISC + β2 Age_Model + β3 Brain_ISC × Age_Model`；重点看交互项 β3 |

**Phase 2 论文式四个结果**
这部分是后加的“故事线”，在 `step1` 到 `step4`：

1. [step1_task_dissociation.py](E:/PythonAnalysis/brain_research/emo_final/step1_task_dissociation.py)  
   做 Passive vs Reappraisal 的差异检验。方法是先计算 `ΔISC = ISC_Passive - ISC_Reappraisal`，再把 ΔISC 与年龄模型，比如 `M_conv`，做 Spearman 关联。显著性用年龄标签置换，双尾检验，并做 FDR / FWER 校正。

2. [step2_gradient_shift.py](E:/PythonAnalysis/brain_research/emo_final/step2_gradient_shift.py)  
   把 ROI 的发育效应量 `r_obs` 映射到 Margulies 2016 皮层主梯度 G1 上。方法是 Spearman 相关 + spin test，用 Alexander-Bloch 空间旋转控制皮层空间自相关。还加入 SCAN ROI 映射、阈值敏感性分析和负对照抽样。

3. [step3_representational_connectivity.py](E:/PythonAnalysis/brain_research/emo_final/step3_representational_connectivity.py)  
   做 ROI/网络之间的 representational connectivity。方法是同一被试内，两个 ROI 的 RSM 上三角向量做 Spearman 相关，得到每个被试的 RC 分数；再用 Spearman + 年龄置换检验 RC 是否随年龄变化。可选线性 vs 二次模型比较，用 AIC 或 adjusted R²。

4. [step4_maturity_index.py](E:/PythonAnalysis/brain_research/emo_final/step4_maturity_index.py)  
   构建“表征成熟度指数”。方法是选目标网络 ROI，把每个被试的 RSM 向量拼接成 mega-vector；用 ≥17 岁被试平均向量作为成熟模板；每个被试与模板做 Spearman 相关，得到 maturity index。之后做 MI 与年龄的相关，以及控制年龄/性别后的 MI 与行为指标偏相关，显著性用置换检验。

**汇报时可以一句话收束**
这个流程的核心不是传统时间序列 ISC，而是 **representational ISC**：先用 RSA 描述每个 ROI 对情绪刺激的表征结构，再比较不同被试的表征结构是否相似，并检验这种相似性是否符合发育年龄模型、行为表现和皮层梯度组织。

一个小提醒：`config_trial.json` / `config_emotion.json` 里 `perm_*` 默认是 `correction_mode: "fdr_only"`，但绘图配置里有些地方默认按 `fwer` 画。如果汇报 FWER 结果，最好确认实际跑过的是 `perm_fwer_fdr`；否则就讲 BH-FDR 更稳。
