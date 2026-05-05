# result_new_meta_roi: meta ROI 全局结果汇总

更新时间：2026-05-06  
主分析 ROI：`meta_metaphor` + `meta_spatial`  
结果根目录：`E:/python_metaphor/paper_outputs`

## 0. 总结

这轮重跑把研究重心从“学习阶段 GLM 定位 ROI”移到了外部 meta 证据驱动的 ROI：`meta_metaphor` 负责语义/隐喻相关网络，`meta_spatial` 负责空间/情境/海马相关网络。学习阶段 GLM ROI 只保留为参考，不作为主故事的 ROI 来源。

整体结论是：**最稳定的新增机制证据不是简单的“隐喻整体增强”，而是学习后 trained relation edge 的表征区分增强**。具体表现为 YY trained edge 在多个 meta ROI 中出现显著 pre-post similarity drop，并且这个 drop 显著强于 baseline pseudo-edge、KJ trained edge 或 untrained non-edge。最强证据集中在右海马、双侧海马/空间情境 ROI、右 PPA/PHG、右 PPC/SPL，以及右颞极、IFG、AG、pMTG/pSTS 等隐喻语义 ROI。

MVPA 结果提供了与 RSA 主机制相互呼应的阶段动态证据。YY/KJ discriminative geometry 并不是 pre 阶段已经稳定存在的静态材料类别码，而是呈现 **pre 弱、learning 强、post 部分保留、retrieval 再次稳健** 的轨迹：学习阶段这些 ROI 明显区分 YY/KJ；post isolated-word 阶段右颞极保留可解码信息；run7 retrieval 阶段右海马、右颞极、右 PPA/PHG 和右 PPC/SPL 再次稳健可分类，并且与 retrieval pair-structure rebound 一致。因此，RSA 告诉我们 post 阶段 trained relation edge 如何被重组和分化；MVPA 告诉我们这种 YY/KJ 条件信息如何随任务阶段被调动、保留和重新提取。

最合适的主线可以概括为：**隐喻关系学习先在学习任务中调动 YY/KJ condition-discriminative geometry，随后在 post 阶段表现为 learned-edge differentiation / relation-edge reorganization，并在最终 retrieval 阶段以 pair-structure rebound 和 YY/KJ decoding 的形式重新出现。**

需要注意的边界是：relation-vector 模型本身没有形成“YY 比 KJ 更强”的简单结论。相反，temporal pole 中 relation-vector condition contrast 显示 KJ 的 relation-vector decoupling 更强。因此目前最合适的故事是：**YY 学习主要重组 pair/edge-level neural geometry；relation-vector 结果提供补充和边界条件，而不是当前最强主证据**。

## 1. ROI 与 mask 状态

本轮主分析使用 18 个 meta ROI：

- `meta_metaphor` 8 个：L/R IFG, L/R temporal pole/ATL, L/R AG, L/R pMTG_pSTS。
- `meta_spatial` 10 个：L/R hippocampus, L/R PPA/PHG, L/R RSC/PCC, L/R PPC/SPL, L/R precuneus。

这些 ROI 来自 Neurosynth/NeuroQuery 思路下的外部 meta peak 和当前显著/潜在效应脑区的交集式筛选。当前没有把学习阶段 GLM peak 当主 mask 来源。生成和审查文件包括：

- `E:/python_metaphor/roi_library/meta_sources/meta_roi_peaks.tsv`
- `E:/python_metaphor/roi_library/meta_sources/meta_roi_manifest_rows.tsv`
- `E:/PythonAnalysis/metaphoric/final_version/roi_management/meta_roi_peaks_curated.tsv`
- `E:/PythonAnalysis/metaphoric/final_version/roi_management/meta_roi_decision_review.tsv`
- `E:/PythonAnalysis/metaphoric/final_version/roi_management/legacy_mask_review.md`

QC 显示 `meta_metaphor` 和 `meta_spatial` 的 Step5C RSA completeness 均为 0 个 incomplete cell；relation-vector cell QC 也没有 non-ok cell。也就是说，新 ROI mask 本身可以继续作为主版本使用。

## 2. 行为结果

行为结果仍然支持 YY 条件学习效果更强：

| 指标 | KJ | YY | 统计 |
|---|---:|---:|---|
| Memory accuracy | 0.588 | 0.680 | logit beta = 0.398, p = 2.38e-06 |
| Correct-trial RT | 1100 ms | 1070 ms | log RT beta = -0.026, p = .009 或 covariate matched p = .022 |

协变量敏感性中，memory 的 YY 优势在加入 sentence length、word frequency、arousal、valence 后仍稳定，adjusted beta 约 0.548-0.553，p < 3.45e-07。RT 优势在未校正时显著，但加入词/句协变量后减弱到不显著，因此 RT 更适合作为补充行为结果，memory accuracy 才是更稳的行为主结果。

对应图：`fig_behavior_accuracy_rt.png`。当前行为图已重画并带显著性标识。

## 3. Step5C ROI-level RSA 主结果

Step5C 的核心问题是：学习后 metaphoric retrieval 是否改变同一关系/pair 的 neural similarity。这里应逐个 ROI 解读，不再把同类 ROI 混在一起统计。

主检验为每个 ROI 单独的 condition x time 交互。meta ROI 中显著 ROI 如下：

| ROI set | ROI | interaction estimate | p | q |
|---|---|---:|---:|---:|
| meta_spatial | meta_R_hippocampus | 0.125 | 9.02e-08 | 2.0e-06 |
| meta_metaphor | meta_R_temporal_pole | 0.159 | 2.00e-06 | 3.2e-05 |
| meta_spatial | meta_L_hippocampus | 0.092 | 1.07e-05 | 1.07e-04 |
| meta_metaphor | meta_L_IFG | 0.105 | 1.90e-05 | 1.52e-04 |
| meta_metaphor | meta_R_IFG | 0.109 | 1.79e-04 | 9.52e-04 |
| meta_metaphor | meta_R_AG | 0.105 | 4.80e-04 | 0.00175 |
| meta_metaphor | meta_L_temporal_pole | 0.101 | 5.46e-04 | 0.00175 |
| meta_spatial | meta_L_PPC_SPL | 0.089 | 5.09e-04 | 0.00339 |
| meta_spatial | meta_R_PPA_PHG | 0.078 | 9.93e-04 | 0.00496 |
| meta_spatial | meta_R_PPC_SPL | 0.078 | 0.00301 | 0.0120 |
| meta_spatial | meta_L_RSC_PCC | 0.083 | 0.00613 | 0.0204 |
| meta_metaphor | meta_L_pMTG_pSTS | 0.071 | 0.00855 | 0.0228 |

过程数据很关键。以 Metaphor condition 内 pre-post similarity 为例：

| ROI | pre | post | post-pre |
|---|---:|---:|---:|
| meta_R_temporal_pole | 0.209 | 0.122 | -0.087 |
| meta_L_IFG | 0.136 | 0.075 | -0.060 |
| meta_R_IFG | 0.160 | 0.106 | -0.053 |
| meta_L_temporal_pole | 0.167 | 0.113 | -0.053 |
| meta_R_AG | 0.140 | 0.083 | -0.057 |
| meta_L_pMTG_pSTS | 0.161 | 0.079 | -0.082 |

这说明显著交互不是凭空的“比较差异”，而是由 Metaphor/YY 相关 similarity 的下降驱动。这个结果更像是 learned-edge differentiation / relation-edge reorganization：学习后被训练关系不再保持原来的粗相似结构，而是被区分、重排。

对应图：`fig_step5c_main.png`, `fig_roi_set_robustness.png`。

## 4. Edge Specificity: 当前最强机制证据

Edge specificity 是这轮最值得放进主故事的新增机制分析，因为它直接比较 trained edge、untrained non-edge、baseline pseudo-edge 和 KJ trained edge。

最强结果集中在 spatial/hippocampal ROI：

| ROI | contrast | effect | p | q |
|---|---|---:|---:|---:|
| meta_R_hippocampus | YY trained drop > baseline pseudo drop | 0.125 | 4.33e-07 | 1.73e-05 |
| meta_R_PPA_PHG | YY trained drop > KJ trained drop | 0.090 | 1.90e-06 | 3.80e-05 |
| meta_R_hippocampus | YY trained drop > KJ trained drop | 0.093 | 1.25e-05 | 1.66e-04 |
| meta_R_hippocampus | YY specificity > KJ specificity | 0.087 | 1.95e-05 | 1.95e-04 |
| meta_R_PPC_SPL | YY specificity > KJ specificity | 0.081 | 3.12e-05 | 2.50e-04 |

隐喻语义 ROI 中也有很强结果：

| ROI | contrast | effect | p | q |
|---|---|---:|---:|---:|
| meta_R_temporal_pole | YY specificity > KJ specificity | 0.110 | 7.74e-05 | 7.13e-04 |
| meta_R_temporal_pole | YY trained drop > untrained non-edge drop | 0.079 | 8.58e-05 | 7.13e-04 |
| meta_R_temporal_pole | YY trained drop > baseline pseudo drop | 0.159 | 8.62e-05 | 7.13e-04 |
| meta_R_IFG | YY trained drop > baseline pseudo drop | 0.109 | 8.91e-05 | 7.13e-04 |
| meta_L_temporal_pole | YY trained drop > baseline pseudo drop | 0.101 | 1.99e-04 | 0.00127 |
| meta_L_IFG | YY trained drop > baseline pseudo drop | 0.105 | 3.38e-04 | 0.00180 |

过程数据支持“trained edge 特异性下降”：

- `meta_R_hippocampus`: YY trained edge 0.170 -> 0.064, drop = 0.106, q = 0.00021；baseline pseudo-edge 0.068 -> 0.087, drop = -0.019, q = 0.340。
- `meta_R_PPA_PHG`: YY trained edge 0.174 -> 0.084, drop = 0.091, q = 0.00374；KJ trained edge 0.154 -> 0.153, drop = 0.001, q = 0.961。
- `meta_R_PPC_SPL`: YY trained edge 0.156 -> 0.094, drop = 0.062, q = 0.017；KJ trained edge 0.134 -> 0.150, drop = -0.016, q = 0.579。
- `meta_R_temporal_pole`: YY trained edge 0.209 -> 0.122, drop = 0.087, q = 0.026；YY untrained non-edge 0.135 -> 0.128, drop = 0.007, q = 0.819；KJ trained edge 0.178 -> 0.205, drop = -0.027, q = 0.519。

这一块可以作为论文/汇报中最主要的神经机制证据：**学习主要改变被训练关系边，而不是所有词对或所有相似性都一起变化**。

对应图：`fig_edge_specificity.png`。

## 5. Single-word Stability: identity control

Single-word stability 的理论作用是排除一个过强解释：如果所有表征都整体崩塌，那么 pair similarity drop 可能只是噪声或表征不稳定。结果显示，多个 ROI 中 same-word stability 仍为正，说明 word identity 仍有可恢复结构。

显著结果包括：

| ROI | condition | mean stability | p | q |
|---|---|---:|---:|---:|
| meta_L_precuneus | YY | 0.046 | 1.96e-05 | 5.87e-04 |
| meta_L_PPC_SPL | YY | 0.035 | 0.00107 | 0.0161 |
| meta_L_precuneus | KJ | 0.030 | 0.00287 | 0.0287 |
| meta_L_RSC_PCC | YY | 0.028 | 0.00412 | 0.0309 |
| meta_L_temporal_pole | baseline | 0.024 | 0.00779 | 0.0449 |
| meta_L_AG | KJ | 0.030 | 0.00934 | 0.0449 |
| meta_R_pMTG_pSTS | YY | 0.028 | 0.00735 | 0.0449 |
| meta_L_AG | baseline | 0.039 | 0.00250 | 0.0449 |
| meta_R_AG | KJ | 0.031 | 0.00531 | 0.0449 |

因此，Step5C 和 edge-specificity 中的 similarity drop 更适合解释为 relation/edge-level reorganization，而不是单词身份表征整体丢失。

对应图：`fig_word_stability.png`。

## 6. RDM 模型与理论意义

当前模型家族可以这样解释：

| 模型 | 理论意义 |
|---|---|
| M1 condition | 检查神经 RDM 是否主要区分 Metaphor/Spatial 条件类别 |
| M2 pair | 检查具体 pair identity / learned pair structure |
| M3 embedding | 检查静态语义 embedding 几何是否解释神经结构 |
| M7 binary / continuous confidence | 检查行为记忆/信心是否映射到神经距离 |
| M8 reverse pair | M2 的方向反转控制，用于检查 pair 结构方向性 |
| M9 relation vector direct/abs/length | 检查 source-target relation vector，即关系转换结构是否解释神经 RDM |

在 meta ROI 的 global model-RSA 中，M2/M8 在右海马和左海马有 uncorrected evidence，但没有通过模型家族 FDR：

- `meta_R_hippocampus`, M8 reverse pair: p = 0.00427, q_model_family = 0.128, q_primary = 0.085。
- `meta_R_hippocampus`, M2 pair: p = 0.00427, q_model_family = 0.128。
- `meta_L_hippocampus`, M8 reverse pair: p = 0.0122, q_model_family = 0.169。

因此，model-RSA 不应作为主结果宣称“某模型显著解释全局神经 RDM”。它更适合作为探索性补充：海马/空间系统中有 pair-structure 方向的趋势，但主证据仍来自 Step5C 与 edge specificity。

对应图：`fig_model_rsa_mechanism.png`。

## 7. Relation-vector RSA

Relation-vector 分析试图回答：学习是否改变了 source-to-target relation vector 的神经几何。这是理论上最贴近“关系映射”的分析，但当前结果需要谨慎讲。

在 condition-specific pre-post 中，部分 ROI 有显著变化：

- `meta_L_temporal_pole`, YY, M9 relation vector abs: pre = -0.014, post = 0.029, delta = 0.043, p = 0.00045, q_primary = 0.00718。
- `meta_R_temporal_pole`, KJ, M9 relation vector abs: pre = 0.020, post = -0.013, delta = -0.034, p = 0.00120, q_primary = 0.0191。
- `meta_R_PPA_PHG`, YY, relation-vector length: delta = -0.035, p = 0.00063, q_model_family = 0.0262。

但是 condition contrast 显示 temporal pole 的核心差异方向是 KJ > YY decoupling：

- `meta_L_temporal_pole`, M9 abs, YY-KJ decoupling = -0.063, p = 0.00044, q_primary = 0.00702。
- `meta_R_temporal_pole`, M9 abs, YY-KJ decoupling = -0.057, p = 0.00214, q_primary = 0.0171。
- `meta_R_temporal_pole`, M9 direct, YY-KJ decoupling = -0.032, p = 0.00953, q_primary = 0.0509。

网络层面也一致：

- semantic/metaphor network, M9 abs: YY-KJ decoupling = -0.020, p = 0.0113, q = 0.0339。
- hippocampus core, M9 abs: YY-KJ decoupling = -0.022, p = 0.0569, q = 0.0750。
- spatial/context network, M9 abs: YY-KJ decoupling = -0.0167, p = 0.0750, q = 0.0750。

所以这一块的建议写法是：**relation-vector 结果提示 temporal pole 确实参与关系向量结构变化，但其方向不是 YY-specific relation decoupling，而是和 KJ 条件形成分离。这为主机制提供边界条件，而不是主支持证据。**

对应图：`fig_relation_vector_rsa.png`, `fig_relation_vector_condition_contrast.png`, `fig_relation_vector_network_dissociation.png`, `fig_relation_step5c_conjunction.png`。

## 8. Brain-behavior 连接

Relation behavior prediction 有一些未校正层面的桥接结果，但 FDR 后不稳：

- `meta_L_IFG`, YY, relation_pre -> memory accuracy: Pearson r = 0.500, p = 0.00790；Spearman r = 0.579, p = 0.00155；q_primary = 0.132。
- `meta_L_IFG`, YY, relation_pre -> retrieval efficiency: Pearson r = 0.439, p = 0.0219；Spearman r = 0.567, p = 0.00206；q_primary = 0.132。
- `meta_R_PPA_PHG`, YY-KJ relation decoupling -> memory accuracy: Pearson r = -0.521, p = 0.00531；q_primary = 0.334。

因此，brain-behavior 目前可以作为探索性连接：IFG relation structure 可能与 YY 记忆表现相关，但不能作为校正后主结论。

另一个重要 QC：`table_mechanism_behavior_prediction.tsv`、`table_itemlevel_coupling.tsv` 和 `table_repr_connectivity.tsv` 目前仍主要是 legacy ROI families，尚未完全迁移到 `meta_metaphor/meta_spatial`。因此这些图和表不能作为 meta ROI 主结论，只能保留为旧版本敏感性结果，或者后续单独改脚本重跑。

对应图：`fig_relation_behavior_prediction.png`。`fig_mechanism_behavior_prediction.png`, `fig_itemlevel_coupling.png`, `fig_repr_connectivity.png` 当前应标注为 legacy/sensitivity。

## 9. Run7 retrieval geometry

2026-05-04 已经把 run7 retrieval LSS 纳入 meta ROI 分析。run7 神经数据可用被试为 26 名；行为汇总有 27 名，缺失神经 run7 pattern 的是 `sub-23`，因此所有 run7 neural-behavior 分析均使用 26 名被试交集。新增脚本和输出：

- `rsa_analysis/retrieval_pair_similarity.py`
- `brain_behavior/retrieval_success_prediction.py`
- `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_item_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_metrics.tsv`
- `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_geometry_group_fdr.tsv`
- `E:/python_metaphor/paper_outputs/tables_main/table_retrieval_success_prediction.tsv`

QC：26 名被试，0 failure；item rows = 65,520，pair rows = 32,760。run7 结果需要先看 YY/KJ 各自的过程值，再看二者比较。

首先，retrieval 阶段同一 learned pair 的 similarity 在 YY 和 KJ 各自条件下都显著大于 0。`retrieval_pair_similarity > 0` 的 36 个 `ROI × condition` cell 全部通过 primary-family FDR，最大 q = 0.000818。这说明 run7 pattern 中保留了可检测的 pair-level structure，并不是只有 YY 才有基本的 retrieval pair structure。

其次，更有解释力的是 `retrieval - post pair similarity`。这个指标问的是：post 阶段之后，到 run7 retrieval 阶段，同一 learned pair 的 similarity 是否重新上升。YY 自身有多处显著正向结果：

| ROI set | ROI | condition | retrieval - post | p | q |
|---|---|---|---:|---:|---:|
| meta_spatial | meta_R_PPA_PHG | YY | 0.086 | 0.000720 | 0.00274 |
| meta_spatial | meta_R_PPC_SPL | YY | 0.081 | 0.00117 | 0.00427 |
| meta_spatial | meta_L_RSC_PCC | YY | 0.100 | 0.00169 | 0.00589 |
| meta_metaphor | meta_L_IFG | YY | 0.074 | 0.00178 | 0.00672 |
| meta_metaphor | meta_R_pMTG_pSTS | YY | 0.065 | 0.00305 | 0.0108 |
| meta_spatial | meta_R_hippocampus | YY | 0.047 | 0.00545 | 0.0182 |
| meta_metaphor | meta_L_temporal_pole | YY | 0.066 | 0.00751 | 0.0253 |
| meta_spatial | meta_R_RSC_PCC | YY | 0.085 | 0.0102 | 0.0326 |
| meta_metaphor | meta_R_AG | YY | 0.063 | 0.0106 | 0.0339 |
| meta_spatial | meta_L_hippocampus | YY | 0.035 | 0.0161 | 0.0495 |

KJ 自身没有任何 `retrieval - post pair similarity` 通过 primary q < .05。最接近的是 `meta_L_PPC_SPL`，但方向为负，mean = -0.051, p = 0.0326, q = 0.093；其他 KJ ROI 多数为不显著的弱正向或弱负向。也就是说，run7 的“post 后回弹”主要来自 YY 条件自身，而不是 YY/KJ 都同步增加。

在确认上述过程后，再看 YY-KJ 比较。YY 相对 KJ 的 `retrieval - post pair similarity` 在以下 ROI 中显著更强：

| ROI set | ROI | metric | YY-KJ effect | p | q |
|---|---|---|---:|---:|---:|
| meta_spatial | meta_R_PPA_PHG | retrieval - post pair similarity | 0.083 | 4.76e-07 | 1.91e-05 |
| meta_spatial | meta_R_PPC_SPL | retrieval - post pair similarity | 0.085 | 4.21e-05 | 8.42e-04 |
| meta_metaphor | meta_R_temporal_pole | retrieval - post pair similarity | 0.104 | 2.91e-04 | 0.00466 |
| meta_metaphor | meta_R_IFG | retrieval - post pair similarity | 0.084 | 1.85e-04 | 0.00466 |
| meta_spatial | meta_R_hippocampus | retrieval - post pair similarity | 0.048 | 0.00310 | 0.0413 |

解释上，这不是推翻 pre-to-post learned-edge differentiation，而是说明：YY 的 trained-pair similarity 在 post 阶段下降后，run7 retrieval 阶段自身出现 pair-structure rebound；这个 rebound 又显著强于 KJ，尤其在右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG 和右海马。它可以作为主机制之后的 retrieval-stage 补充证据：学习后的 pair/edge geometry 不是纯粹消失，而是在最终提取阶段以任务相关方式重新出现。

这个结果可以形成一个更完整的“分化 -> retrieval 回弹/再聚集”过程解释，但需要谨慎表述。以 YY 条件的几个关键 ROI 为例：

| ROI | pre pair similarity | post pair similarity | run7 retrieval similarity | 轨迹解释 |
|---|---:|---:|---:|---|
| meta_R_PPA_PHG | 0.174 | 0.086 | 0.172 | post 明显分化，run7 几乎回到 pre |
| meta_R_PPC_SPL | 0.158 | 0.098 | 0.179 | post 分化，run7 均值超过 pre |
| meta_R_IFG | 0.156 | 0.108 | 0.168 | post 分化，run7 均值超过 pre |
| meta_R_temporal_pole | 0.208 | 0.123 | 0.167 | post 分化，run7 部分回弹但仍低于 pre |
| meta_R_hippocampus | 0.173 | 0.064 | 0.111 | post 强分化，run7 部分回弹但仍低于 pre |

进一步把 `run7 - pre` 单独检验后，结论需要区分“均值轨迹”和“统计显著性”。在均值上，`meta_R_PPC_SPL` 和 `meta_R_IFG` 的 run7 确实高于 pre，`meta_R_PPA_PHG` 几乎回到 pre；但这些 `run7 - pre` 正向差异没有通过校正，也没有稳定的一致方向。

| ROI | run7 - pre | p | q | 说明 |
|---|---:|---:|---:|---|
| meta_R_PPA_PHG | -0.003 | 0.910 | 0.910 | 几乎回到 pre，但未超过 pre |
| meta_R_PPC_SPL | 0.021 | 0.435 | 0.725 | 均值超过 pre，但不显著 |
| meta_R_IFG | 0.012 | 0.700 | 0.934 | 均值超过 pre，但不显著 |
| meta_R_temporal_pole | -0.042 | 0.236 | 0.707 | 部分回弹，仍低于 pre |
| meta_R_hippocampus | -0.062 | 0.0096 | 0.076 | 未校正低于 pre，校正后趋势 |

因此，最稳妥的写法不是“run7 全面高于 pre”，而是：**学习后先出现 relation-edge differentiation，最终 retrieval 阶段出现 task-driven pair-structure rebound / partial re-binding；部分 ROI 的均值可回到或略高于 pre，但整体没有稳定的 `run7 > pre` 证据**。这能解释为什么 post 阶段看起来像“去相似化”，但 run7 提取时又能重新表现出 pair-level structure。

post-to-retrieval same-vs-different pair reinstatement 本身较弱：`meta_R_hippocampus` YY 的 pair reinstatement 为正，p = 0.0207，但 primary q = 0.061；`meta_L_hippocampus` YY p = 0.0431, q = 0.119。因此 reinstatement 可写成海马方向性趋势，不应作为校正后主结论。

run7 retrieval success prediction 使用 item-level GEE，并在 subject 内拆分 within-subject 与 between-subject 成分。主要结果：

- `meta_L_pMTG_pSTS`, KJ, retrieval pair similarity -> memory: within beta = 0.110, p = 0.000755, q_primary = 0.0362。
- `meta_R_AG`, YY, retrieval pair similarity -> memory: within beta = 0.159, p = 0.00217, q_primary = 0.0521。
- `meta_L_PPA_PHG`, YY, retrieval pair similarity -> log RT correct: within beta = 0.0164, p = 7.52e-05, q_primary = 0.00451。
- `meta_L_RSC_PCC`, YY, retrieval pair similarity -> log RT correct: within beta = 0.0170, p = 0.00180, q_primary = 0.0540。

RT 的 beta 为正，表示 retrieval pair similarity 更高时正确 trial 反应更慢，不是简单的效率增强。因此 run7 行为桥接目前应写为探索性：retrieval pair geometry 与最终表现有关，但方向和条件并不完全支持“更强 reinstatement -> 更好/更快记忆”的单一故事。

## 10. Run7 MVPA classification

为了检验 run7 中显著回升的 ROI 是否不仅有 pair similarity rebound，而且包含可解码的 YY/KJ 条件信息，新增脚本：

- `fmri_mvpa/fmri_mvpa_roi/run7_meta_roi_decoding.py`
- 输出目录：`E:/python_metaphor/paper_outputs/qc/run7_mvpa_decoding/`
- 主表：`E:/python_metaphor/paper_outputs/tables_main/table_run7_mvpa_decoding.tsv`
- 行为桥接表：`E:/python_metaphor/paper_outputs/tables_main/table_run7_mvpa_evidence_behavior.tsv`

分析使用 26 名有 run7 neural pattern 的被试，0 failure。预先聚焦 5 个有 run7 retrieval-minus-post rebound 的 ROI：`meta_R_PPA_PHG`, `meta_R_PPC_SPL`, `meta_R_temporal_pole`, `meta_R_IFG`, `meta_R_hippocampus`。QC 显示每个 subject x ROI 均有 140 个 trial，YY/KJ 各 70 个 trial，`w/ew` 四类各 35 个 trial。脚本最初按原始 `pair_id` 留出时发现 YY 的 `pair_id` 为 1-35、KJ 为 36-70，单个 fold 只有一个类别，因此已改为按 condition 内的同序号 pair 成对留出，避免训练集与测试集共享同序号 YY/KJ pair。

主分析是 pair-held-out YY/KJ decoding：每个 fold 同时留出一个 YY pair 和一个 KJ pair，训练其余 pair，测试被留出的两个条件。结果显示 4/5 个 ROI 的 balanced accuracy 显著高于 chance：

| ROI | balanced accuracy | above chance | p | q_primary | dz |
|---|---:|---:|---:|---:|---:|
| meta_R_hippocampus | 0.565 | 0.065 | 2.98e-06 | 3.04e-05 | 1.17 |
| meta_R_temporal_pole | 0.553 | 0.053 | 4.05e-06 | 3.04e-05 | 1.15 |
| meta_R_PPA_PHG | 0.564 | 0.064 | 1.72e-05 | 8.58e-05 | 1.04 |
| meta_R_PPC_SPL | 0.547 | 0.047 | 0.000190 | 0.000711 | 0.86 |
| meta_R_IFG | 0.513 | 0.013 | 0.187 | 0.357 | 0.27 |

这说明 run7 的右海马、右颞极、右 PPA/PHG、右 PPC/SPL 不只是 similarity 均值回升，也包含可用于区分 YY/KJ 的 multivariate condition information。这个证据可以作为 retrieval-stage pair/condition structure 的补充，但不能替代前面的 RSA 主结果。

为了检查这种分类是否是角色不变的抽象 YY/KJ code，又做了 cross-role generalization：`w -> ew` 与 `ew -> w`。结果没有任何 ROI 通过 FDR，最接近的是 `meta_R_PPA_PHG`：

- `w -> ew`: balanced accuracy = 0.518, p = 0.0671, q_primary = 0.201。
- `ew -> w`: balanced accuracy = 0.520, p = 0.126, q_primary = 0.314。

因此，原先设想的“用 yyw/kjw 训练后准确分类 yyew/kjew”的强版本目前不成立。更合理的解释是：run7 中存在 YY/KJ condition-discriminative information，但它不是稳定跨 `w/ew` 泛化的角色不变表征。

为了定位 cross-role 失败的原因，又做了 role-specific pair-held-out decoding。`w-only` 和 `ew-only` 在右 PPA/PHG、右海马、右颞极、右 PPC/SPL 中均显著：

| analysis | ROI | balanced accuracy | p | q_within_analysis |
|---|---|---:|---:|---:|
| w-only | meta_R_PPA_PHG | 0.592 | 1.44e-06 | 7.18e-06 |
| w-only | meta_R_hippocampus | 0.585 | 1.18e-05 | 1.97e-05 |
| w-only | meta_R_temporal_pole | 0.591 | 9.99e-06 | 1.97e-05 |
| w-only | meta_R_PPC_SPL | 0.584 | 3.95e-05 | 4.94e-05 |
| ew-only | meta_R_temporal_pole | 0.590 | 2.07e-06 | 1.03e-05 |
| ew-only | meta_R_PPA_PHG | 0.591 | 2.32e-05 | 5.80e-05 |
| ew-only | meta_R_hippocampus | 0.574 | 0.000105 | 0.000174 |
| ew-only | meta_R_PPC_SPL | 0.568 | 0.000603 | 0.000754 |

这组结果给出的边界条件很重要：分类信息在 `w` 和 `ew` 各自内部都存在，但跨角色泛化失败。因此论文中不建议写成“learned relation 的抽象类别码”；更稳妥的写法是 **run7 retrieval 阶段在 hippocampal-spatial/temporal 系统中出现 role-dependent YY/KJ discriminative geometry**。这与前面的 pair similarity rebound 共同支持 retrieval 阶段的 partial re-binding，但不支持完全角色不变的类别表征。

最后，classifier evidence 的 item-level brain-behavior GEE 没有通过 FDR：

- memory: 最小 q = 0.437。
- correct-trial log RT: 最小 q = 0.750。

因此 MVPA evidence-behavior 不进入主故事，只能记录为 negative/exploratory check。当前应停止继续扩展“classifier evidence 预测记忆/RT”的链条。

## 11. Learning-stage MVPA classification

为了检验 run7 中出现 YY/KJ 可解码信息的 ROI，在学习阶段 run3/run4 是否也已有类似分类效果，新增脚本：

- `fmri_mvpa/fmri_mvpa_roi/learning_meta_roi_decoding.py`
- 输出目录：`E:/python_metaphor/paper_outputs/qc/learning_mvpa_decoding/`
- 主表：`E:/python_metaphor/paper_outputs/tables_main/table_learning_mvpa_decoding.tsv`
- run4-run3 差异表：`E:/python_metaphor/paper_outputs/tables_main/table_learning_mvpa_run4_minus_run3.tsv`

学习阶段 pattern 与 run7 不同：这里是 pair/item-level learning pattern，不是 `w/ew` 单词角色 pattern，因此不能做 run7 那种 cross-role generalization。脚本改为三类检验：

1. `run3_pair_heldout`：run3 内部成对留出同一 `pair_id` 的 YY/KJ item，训练其余 pair。
2. `run4_pair_heldout`：run4 内部同样的 pair-held-out 分类。
3. `cross_run_3_to_4` / `cross_run_4_to_3`：检验学习阶段分类边界是否跨 run 稳定。

QC：28 名被试，0 failure；每个 subject x ROI 有 140 个 learning trials，run3/run4 各 70 个，YY/KJ 各 70 个。run3 和 run4 中 YY/KJ 共有的 `pair_id` 均为 30 个，因此 within-run pair-held-out 每个 subject x ROI x run 有 30 个 fold。

学习阶段 within-run MVPA 的结果很强。run3 中 5/5 个 ROI 均显著高于 chance：

| ROI | run3 balanced accuracy | p | q_primary | dz |
|---|---:|---:|---:|---:|
| meta_R_PPA_PHG | 0.591 | 1.81e-06 | 3.63e-05 | 1.14 |
| meta_R_hippocampus | 0.567 | 1.72e-05 | 0.000172 | 0.99 |
| meta_R_temporal_pole | 0.563 | 0.000170 | 0.000679 | 0.82 |
| meta_R_PPC_SPL | 0.545 | 0.00403 | 0.00896 | 0.59 |
| meta_R_IFG | 0.532 | 0.00770 | 0.0154 | 0.54 |

run4 中同样 5/5 个 ROI 显著高于 chance：

| ROI | run4 balanced accuracy | p | q_primary | dz |
|---|---:|---:|---:|---:|
| meta_R_temporal_pole | 0.571 | 5.74e-05 | 0.000382 | 0.90 |
| meta_R_PPA_PHG | 0.575 | 8.89e-05 | 0.000444 | 0.87 |
| meta_R_PPC_SPL | 0.573 | 0.000600 | 0.00165 | 0.73 |
| meta_R_IFG | 0.568 | 0.000591 | 0.00165 | 0.74 |
| meta_R_hippocampus | 0.570 | 0.000661 | 0.00165 | 0.73 |

这说明这些 run7 rebound ROI 在学习阶段已经可以区分 YY/KJ。它加强了“这些区域确实承载条件相关学习信息”的论点，而不是只在最终 retrieval 才出现 YY/KJ 差异。

但两个边界条件也很重要：

- run4 - run3 的 accuracy 提升没有任何 ROI 通过 FDR；最接近的是 `meta_R_IFG`，delta = 0.036, p = 0.088, q = 0.441。
- 跨 run 泛化没有任何 ROI 通过 FDR；最接近的是 `meta_R_PPA_PHG` 的 `run3 -> run4`，accuracy = 0.515, p = 0.0404, q_primary = 0.0735。

因此，学习阶段 MVPA 不应写成“学习使 YY/KJ classifier 从 run3 到 run4 稳定增强”，也不应写成“分类边界跨 run 稳定泛化”。更稳妥的写法是：**右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG、右海马在学习阶段 run-local pattern 中已经包含 YY/KJ discriminative information；这种信息与 run7 的 retrieval-stage decoding 相呼应，但当前没有证据表明它在 run3 到 run4 中显著增强或形成稳定跨 run 的分类边界。**

这条结果可以补充当前故事：post 阶段的 learned-edge differentiation 并不是凭空出现，学习阶段这些 ROI 已经能区分 YY/KJ 条件；但真正和“关系边重组”贴合的主证据仍然是 post 的 edge-specific RSA，以及 run7 的 pair-structure rebound / role-dependent decoding。

## 12. Pre/post isolated-word MVPA controls

为了把 learning/run7 的分类结果放回完整学习过程，新增 pre/post isolated-word 阶段的 YY/KJ 二分类分析：

- `fmri_mvpa/fmri_mvpa_roi/prepost_meta_roi_binary_decoding.py`

metadata 的真实 run 编码需要明确：pre 是 run1/2，post 是 run5/6；run3/4 是学习阶段，不是 post。由于 pre 的 run1+run2、post 的 run5+run6 合起来才覆盖完整 YY/KJ 词集，主报告口径采用 **phase-combined pair-held-out**：在每个 phase 内合并两个 run，每折同时留出一个 YY pair 和一个 KJ pair，训练其余 pair。这个检验回答的是“完整 pre/post 词集内是否存在 YY/KJ 可解码信息”，比跨 run 泛化更贴合当前问题。

YY/KJ 二分类 QC：28 名被试，0 failure；每个 subject x ROI 有 280 个 YY/KJ trial，其中 pre = 140、post = 140；每个阶段 YY = 70、KJ = 70。phase-combined pair-held-out 中，每个 subject x ROI x phase 有 35 个 fold。

pre 阶段整体较弱，只有右颞极达到未校正显著，校正后不稳：

| ROI | pre combined pair-held-out accuracy | p | q_primary |
|---|---:|---:|---:|
| meta_R_temporal_pole | 0.520 | 0.0393 | 0.0983 |
| meta_R_PPA_PHG | 0.514 | 0.244 | 0.443 |
| meta_R_PPC_SPL | 0.511 | 0.307 | 0.453 |
| meta_R_hippocampus | 0.510 | 0.359 | 0.453 |
| meta_R_IFG | 0.504 | 0.571 | 0.635 |

post 阶段出现更清楚的正向分类，右颞极通过 primary-family FDR，其他 ROI 多数为未校正显著或趋势：

| ROI | post combined pair-held-out accuracy | p | q_primary |
|---|---:|---:|---:|
| meta_R_temporal_pole | 0.539 | 0.00254 | 0.0354 |
| meta_R_hippocampus | 0.525 | 0.0162 | 0.0983 |
| meta_R_PPC_SPL | 0.516 | 0.0306 | 0.0983 |
| meta_R_IFG | 0.520 | 0.0377 | 0.0983 |
| meta_R_PPA_PHG | 0.520 | 0.0453 | 0.101 |

这组结果不需要被写成“post 显著强于 pre”。更合理的叙述是：pre 阶段没有稳定 YY/KJ 区分；学习阶段 YY/KJ 区分显著出现；post isolated-word 阶段保留了部分 condition information，尤其右颞极；最终 run7 retrieval 阶段又重新变得稳健。这个阶段性模式支持“学习-保留-提取”的动态过程，而不是一个贯穿全程的静态材料类别码。真正的机制增强证据仍然由 post edge-specific RSA 承担，MVPA 提供的是阶段状态证据。

把 MVPA 结果按四个阶段串联，可以得到更清晰的动态图景：

| 阶段 | 数据与口径 | 主要结果 | 解释 |
|---|---|---|---|
| pre | run1+run2, YY/KJ phase-combined pair-held-out | 整体弱；右颞极 0.520, q = 0.098 | 初始 isolated-word 阶段没有稳定 YY/KJ 区分 |
| learning | run3/run4, YY/KJ within-run pair-held-out | run3/run4 中 5 个 ROI 均显著 | 学习任务中 YY/KJ condition-discriminative geometry 被显著调动 |
| post | run5+run6, YY/KJ phase-combined pair-held-out | 右颞极显著，0.539, q = 0.035；其余 ROI 为趋势 | 学习后静态词项阶段保留部分 YY/KJ 信息 |
| retrieval | run7, YY/KJ pair-held-out | 右海马、右颞极、右 PPA/PHG、右 PPC/SPL 显著 | 提取阶段 YY/KJ 区分重新稳健出现，与 pair-structure rebound 对应 |

因此，MVPA 最适合写成：**YY/KJ discriminative geometry 并不是 pre 阶段已有的稳定静态类别码，而是在 learning 阶段显著出现，在 post isolated-word 阶段部分保留，并在 retrieval 阶段重新稳健出现。**

## 13. 学习动态、searchlight 与 noise ceiling

Relation learning dynamics 在 meta ROI 中没有稳定通过 FDR 的 run3-run4 轨迹结果：

- `meta_L_hippocampus`, YY, M9 abs: delta = -0.031, p = 0.0258, q_primary = 0.516。
- `meta_L_hippocampus`, KJ, M9 abs: delta = -0.030, p = 0.0296, q_primary = 0.592。

所以 learning-stage relation dynamics 目前不作为主结果。

Searchlight 结果已用已有 subject maps 复用生成，适合作为全脑补充图，但本轮故事应继续以 meta ROI 为主，避免又回到过分分散的全脑叙述。

Noise ceiling 表中已有 meta model-RSA neural RDM 行，整体显示 ROI 内 RDM 可靠性非零，但需要按具体分析类型过滤读取；部分旧 Step5C itemwise 行仍混有 literature ROI，引用时要避免误读。

对应图：`fig_learning_dynamics_meta_metaphor.png`, `fig_learning_dynamics_meta_spatial.png`, `fig_relation_learning_dynamics.png`, `fig_searchlight_delta.png`, `fig_searchlight_pair_similarity_delta.png`, `fig_noise_ceiling.png`。

## 14. 推荐故事线

推荐主线如下：

1. **行为层面**：YY 条件记忆显著优于 KJ，RT 作为补充。
2. **ROI 依据**：用外部 meta ROI 锁定隐喻语义网络和空间/海马情境网络，避免学习阶段 GLM circularity。
3. **pre 状态**：run1+run2 的 phase-combined YY/KJ MVPA 整体较弱，仅右颞极为未校正趋势，说明 YY/KJ 不是一开始就稳定可读出的静态材料类别码。
4. **learning 状态**：run3/run4 学习阶段在右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG、右海马均能显著解码 YY/KJ，说明学习任务显著调动 condition-discriminative geometry。
5. **post 主机制**：Step5C 和 edge specificity 显示 YY trained edge 出现 learned-edge differentiation，并且强于 baseline pseudo-edge、untrained non-edge 或 KJ trained edge。这是当前最强机制证据。
6. **post 状态补充**：run5+run6 的 phase-combined YY/KJ MVPA 显示右颞极显著可分类，其余 ROI 多为趋势，说明学习后 isolated-word 阶段保留了部分 condition information。
7. **retrieval 再调动**：run7 中 YY 相对 KJ 出现更强 retrieval-minus-post pair-structure rebound；同时右海马、右颞极、右 PPA/PHG、右 PPC/SPL 能解码 YY/KJ，支持 retrieval-stage re-binding / task-driven reinstatement。
8. **边界条件**：run7 cross-role generalization 不成立，因此不要写成角色不变抽象类别码；learning 的 cross-run 泛化和 run4-run3 增强也不显著，因此不要写成分类边界随学习单调增强。
9. **控制分析**：single-word stability 仍为正，说明不是 word identity 表征崩塌；relation-vector 和 model-RSA 作为补充和边界条件，不替代 edge-specific RSA 主机制。
10. **行为桥接**：IFG relation structure 与 YY memory、retrieval pair geometry 与 run7 memory/RT 有局部关联，但整体仍是探索性。

一句话版本：**YY/KJ 可区分表征在 pre 阶段较弱，在 learning 阶段显著出现，在 post 阶段以 relation-edge differentiation 和右颞极可解码信息的形式部分保留，并在 retrieval 阶段伴随 pair-structure rebound 重新稳健出现。**

## 15. 当前需要注意的问题

- MixedLM 重跑中出现过若干 convergence warning。主结果很强的 ROI 影响不大，但边缘 ROI 和探索性模型应谨慎。
- 部分全局表仍混有旧 ROI family，尤其 mechanism-behavior、itemlevel coupling、representational connectivity。写论文或汇报时不要把这些当成 meta ROI 主结果。
- Relation-vector 的方向需要讲清楚：temporal pole 是 KJ > YY decoupling，不是 YY > KJ。
- run7 神经分析只有 26 名被试，缺 `sub-23`；必须报告使用 neural-behavior intersection。
- run7 RT 桥接中正 beta 表示更慢，不要写成 retrieval geometry 提高提取效率。
- run7 MVPA 的 cross-role generalization 没有显著，不能写成 yyw/kjw 训练后可稳定泛化到 yyew/kjew。
- learning-stage MVPA 的 within-run 解码显著，但 cross-run 泛化和 run4-run3 增强不显著，不能写成学习过程中分类边界稳定增强。
- pre/post 中后测真实 run 是 run5/6，不是 run3/4；run3/4 是学习阶段。
- pre/post isolated-word 的主报告口径应使用 phase-combined pair-held-out；跨 run held-out 结果不再作为这部分主叙述。
- post 右颞极 YY/KJ 分类显著可作为静态阶段部分保留的证据，但 MVPA 不承担主机制增强证明，主机制仍是 edge-specific RSA。
- 所有 ROI-level 神经结果应继续逐 ROI 汇报，不建议再做“同类 ROI 合并”的主统计。
