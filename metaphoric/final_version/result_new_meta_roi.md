# result\_new\_meta\_roi: meta ROI 全局结果汇总

更新时间：2026-05-10\
主分析 ROI：`meta_metaphor` + `meta_spatial`\
结果根目录：`E:/python_metaphor/paper_outputs`

## 0. 总结

这轮重跑把研究重心从“学习阶段 GLM 定位 ROI”移到了外部 meta 证据驱动的 ROI：`meta_metaphor` 负责语义/隐喻相关网络，`meta_spatial` 负责空间/情境/海马相关网络。学习阶段 GLM ROI 只保留为参考，不作为主故事的 ROI 来源。

整体结论是：**最稳定的新增机制证据不是简单的“隐喻整体增强”，而是学习后 trained relation edge 的表征区分增强**。具体表现为 YY trained edge 在多个 meta ROI 中出现显著 pre-post similarity drop，并且这个 drop 显著强于 baseline pseudo-edge、KJ trained edge 或 untrained non-edge。最强证据集中在右海马、双侧海马/空间情境 ROI、右 PPA/PHG、右 PPC/SPL，以及右颞极、IFG、AG、pMTG/pSTS 等隐喻语义 ROI。

MVPA 结果提供了与 RSA 主机制相互呼应的阶段动态证据。YY/KJ discriminative geometry 并不是 pre 阶段已经稳定存在的静态材料类别码，而是呈现 **pre 弱、learning 强、post 部分保留、retrieval 再次稳健** 的轨迹：学习阶段这些 ROI 明显区分 YY/KJ；post isolated-word 阶段右颞极保留可解码信息；run7 retrieval 阶段右海马、右颞极、右 PPA/PHG 和右 PPC/SPL 再次稳健可分类，并且与 retrieval pair-structure rebound 一致。因此，RSA 告诉我们 post 阶段 trained relation edge 如何被重组和分化；MVPA 告诉我们这种 YY/KJ 条件信息如何随任务阶段被调动、保留和重新提取。最新 Stage 2 进一步显示，完整 run3 × run4 matrix 中存在 YY/KJ condition-level cross-run geometry，但更严格的 same-pair item trace 本身较弱，且不能稳定预测 post edge differentiation；因此 learning 阶段目前更适合写成 condition-level geometry 被调动，而不是已经形成强 pair-selective relation edge。Stage 3 的 model-RDM competition 没有显示稳定的 semantic-to-edge unique-fit shift，说明当前主机制仍应由 direct pair-similarity / edge-specificity contrast 承担，而不是写成全局 RDM 模型权重从 embedding 转向 edge。Stage 4 subsequent-memory rework 原本显示 later memory 与 YY 的 hpc-spatial re-binding difference score 相关；但 Stage 4.5 difference-score 分解进一步说明，这个效应主要由 remembered YY items 的 **lower post similarity / stronger prior post separation** 驱动，而不是 retrieval similarity 本身更高。因此行为桥接应写成 post-stage separation 与后续记忆相关的 endpoint 边界证据，而不是 retrieval-stage mnemonic reinstatement 的强证据。

最合适的主线可以概括为：**隐喻关系学习先在学习任务中调动 YY/KJ condition-discriminative geometry，随后在 post 阶段表现为 learned-edge differentiation / relation-edge reorganization，并在最终 retrieval 阶段以 pair-structure rebound 和 YY/KJ decoding 的形式重新出现。** 但需要谨慎的是，`post_edge_differentiation` 与 `retrieval_rebinding` 都包含 post similarity 项；经过 post-control 复核后，二者的 item/network-level 预测关系不再稳定。因此当前主线应写成阶段性表征链，而不是“post separation 直接预测 retrieval re-binding”的强桥接链。

需要注意的边界是：relation-vector 模型本身没有形成“YY 比 KJ 更强”的简单结论。相反，temporal pole 中 relation-vector condition contrast 显示 KJ 的 relation-vector decoupling 更强。因此目前最合适的故事是：**YY 学习主要重组 pair/edge-level neural geometry；relation-vector 结果提供补充和边界条件，而不是当前最强主证据**。

Reviewer-supplementary checks 进一步支持这种边界化解释：它们主要排除了 ROI mean activation、pre/post novelty/repetition、经典 learning-to-retrieval ERS 和 KJ-as-null 等更简单解释，但不改变 Stage 5 收束出的主线。

## 1. ROI 与 mask 状态

本轮主分析使用 18 个 meta ROI：

- `meta_metaphor` 8 个：L/R IFG, L/R temporal pole/ATL, L/R AG, L/R pMTG\_pSTS。
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

| 指标               |      KJ |      YY | 统计                                                          |
| ---------------- | ------: | ------: | ----------------------------------------------------------- |
| Memory accuracy  |   0.588 |   0.680 | logit beta = 0.398, p = 2.38e-06                            |
| Correct-trial RT | 1100 ms | 1070 ms | log RT beta = -0.026, p = .009 或 covariate matched p = .022 |

协变量敏感性中，memory 的 YY 优势在加入 sentence length、word frequency、arousal、valence 后仍稳定，adjusted beta 约 0.548-0.553，p < 3.45e-07。RT 优势在未校正时显著，但加入词/句协变量后减弱到不显著，因此 RT 更适合作为补充行为结果，memory accuracy 才是更稳的行为主结果。

对应图：`fig_behavior_accuracy_rt.png`。当前行为图已重画并带显著性标识。

## 3. Step5C ROI-level RSA 主结果

Step5C 的核心问题是：学习后 metaphoric retrieval 是否改变同一关系/pair 的 neural similarity。这里应逐个 ROI 解读，不再把同类 ROI 混在一起统计。

主检验为每个 ROI 单独的 condition x time 交互。meta ROI 中显著 ROI 如下：

| ROI set        | ROI                     | interaction estimate |        p |        q |
| -------------- | ----------------------- | -------------------: | -------: | -------: |
| meta\_spatial  | meta\_R\_hippocampus    |                0.125 | 9.02e-08 |  2.0e-06 |
| meta\_metaphor | meta\_R\_temporal\_pole |                0.159 | 2.00e-06 |  3.2e-05 |
| meta\_spatial  | meta\_L\_hippocampus    |                0.092 | 1.07e-05 | 1.07e-04 |
| meta\_metaphor | meta\_L\_IFG            |                0.105 | 1.90e-05 | 1.52e-04 |
| meta\_metaphor | meta\_R\_IFG            |                0.109 | 1.79e-04 | 9.52e-04 |
| meta\_metaphor | meta\_R\_AG             |                0.105 | 4.80e-04 |  0.00175 |
| meta\_metaphor | meta\_L\_temporal\_pole |                0.101 | 5.46e-04 |  0.00175 |
| meta\_spatial  | meta\_L\_PPC\_SPL       |                0.089 | 5.09e-04 |  0.00339 |
| meta\_spatial  | meta\_R\_PPA\_PHG       |                0.078 | 9.93e-04 |  0.00496 |
| meta\_spatial  | meta\_R\_PPC\_SPL       |                0.078 |  0.00301 |   0.0120 |
| meta\_spatial  | meta\_L\_RSC\_PCC       |                0.083 |  0.00613 |   0.0204 |
| meta\_metaphor | meta\_L\_pMTG\_pSTS     |                0.071 |  0.00855 |   0.0228 |

过程数据很关键。以 Metaphor condition 内 pre-post similarity 为例：

| ROI                     |   pre |  post | post-pre |
| ----------------------- | ----: | ----: | -------: |
| meta\_R\_temporal\_pole | 0.209 | 0.122 |   -0.087 |
| meta\_L\_IFG            | 0.136 | 0.075 |   -0.060 |
| meta\_R\_IFG            | 0.160 | 0.106 |   -0.053 |
| meta\_L\_temporal\_pole | 0.167 | 0.113 |   -0.053 |
| meta\_R\_AG             | 0.140 | 0.083 |   -0.057 |
| meta\_L\_pMTG\_pSTS     | 0.161 | 0.079 |   -0.082 |

这说明显著交互不是凭空的“比较差异”，而是由 Metaphor/YY 相关 similarity 的下降驱动。这个结果更像是 learned-edge differentiation / relation-edge reorganization：学习后被训练关系不再保持原来的粗相似结构，而是被区分、重排。

对应图：`fig_step5c_main.png`, `fig_roi_set_robustness.png`。

## 4. Edge Specificity: 当前最强机制证据

Edge specificity 是这轮最值得放进主故事的新增机制分析，因为它直接比较 trained edge、untrained non-edge、baseline pseudo-edge 和 KJ trained edge。

最强结果集中在 spatial/hippocampal ROI：

| ROI                  | contrast                               | effect |        p |        q |
| -------------------- | -------------------------------------- | -----: | -------: | -------: |
| meta\_R\_hippocampus | YY trained drop > baseline pseudo drop |  0.125 | 4.33e-07 | 1.73e-05 |
| meta\_R\_PPA\_PHG    | YY trained drop > KJ trained drop      |  0.090 | 1.90e-06 | 3.80e-05 |
| meta\_R\_hippocampus | YY trained drop > KJ trained drop      |  0.093 | 1.25e-05 | 1.66e-04 |
| meta\_R\_hippocampus | YY specificity > KJ specificity        |  0.087 | 1.95e-05 | 1.95e-04 |
| meta\_R\_PPC\_SPL    | YY specificity > KJ specificity        |  0.081 | 3.12e-05 | 2.50e-04 |

隐喻语义 ROI 中也有很强结果：

| ROI                     | contrast                                  | effect |        p |        q |
| ----------------------- | ----------------------------------------- | -----: | -------: | -------: |
| meta\_R\_temporal\_pole | YY specificity > KJ specificity           |  0.110 | 7.74e-05 | 7.13e-04 |
| meta\_R\_temporal\_pole | YY trained drop > untrained non-edge drop |  0.079 | 8.58e-05 | 7.13e-04 |
| meta\_R\_temporal\_pole | YY trained drop > baseline pseudo drop    |  0.159 | 8.62e-05 | 7.13e-04 |
| meta\_R\_IFG            | YY trained drop > baseline pseudo drop    |  0.109 | 8.91e-05 | 7.13e-04 |
| meta\_L\_temporal\_pole | YY trained drop > baseline pseudo drop    |  0.101 | 1.99e-04 |  0.00127 |
| meta\_L\_IFG            | YY trained drop > baseline pseudo drop    |  0.105 | 3.38e-04 |  0.00180 |

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

| ROI                     | condition | mean stability |        p |        q |
| ----------------------- | --------- | -------------: | -------: | -------: |
| meta\_L\_precuneus      | YY        |          0.046 | 1.96e-05 | 5.87e-04 |
| meta\_L\_PPC\_SPL       | YY        |          0.035 |  0.00107 |   0.0161 |
| meta\_L\_precuneus      | KJ        |          0.030 |  0.00287 |   0.0287 |
| meta\_L\_RSC\_PCC       | YY        |          0.028 |  0.00412 |   0.0309 |
| meta\_L\_temporal\_pole | baseline  |          0.024 |  0.00779 |   0.0449 |
| meta\_L\_AG             | KJ        |          0.030 |  0.00934 |   0.0449 |
| meta\_R\_pMTG\_pSTS     | YY        |          0.028 |  0.00735 |   0.0449 |
| meta\_L\_AG             | baseline  |          0.039 |  0.00250 |   0.0449 |
| meta\_R\_AG             | KJ        |          0.031 |  0.00531 |   0.0449 |

因此，Step5C 和 edge-specificity 中的 similarity drop 更适合解释为 relation/edge-level reorganization，而不是单词身份表征整体丢失。

对应图：`fig_word_stability.png`。

## 6. RDM 模型与理论意义

当前模型家族可以这样解释：

| 模型                                   | 理论意义                                               |
| ------------------------------------ | -------------------------------------------------- |
| M1 condition                         | 检查神经 RDM 是否主要区分 Metaphor/Spatial 条件类别              |
| M2 pair                              | 检查具体 pair identity / learned pair structure        |
| M3 embedding                         | 检查静态语义 embedding 几何是否解释神经结构                        |
| M7 binary / continuous confidence    | 检查行为记忆/信心是否映射到神经距离                                 |
| M8 reverse pair                      | M2 的方向反转控制，用于检查 pair 结构方向性                         |
| M9 relation vector direct/abs/length | 检查 source-target relation vector，即关系转换结构是否解释神经 RDM |

在 meta ROI 的 global model-RSA 中，M2/M8 在右海马和左海马有 uncorrected evidence，但没有通过模型家族 FDR：

- `meta_R_hippocampus`, M8 reverse pair: p = 0.00427, q\_model\_family = 0.128, q\_primary = 0.085。
- `meta_R_hippocampus`, M2 pair: p = 0.00427, q\_model\_family = 0.128。
- `meta_L_hippocampus`, M8 reverse pair: p = 0.0122, q\_model\_family = 0.169。

因此，model-RSA 不应作为主结果宣称“某模型显著解释全局神经 RDM”。它更适合作为探索性补充：海马/空间系统中有 pair-structure 方向的趋势，但主证据仍来自 Step5C 与 edge specificity。

对应图：`fig_model_rsa_mechanism.png`。

## 7. Relation-vector RSA

Relation-vector 分析试图回答：学习是否改变了 source-to-target relation vector 的神经几何。这是理论上最贴近“关系映射”的分析，但当前结果需要谨慎讲。

在 condition-specific pre-post 中，部分 ROI 有显著变化：

- `meta_L_temporal_pole`, YY, M9 relation vector abs: pre = -0.014, post = 0.029, delta = 0.043, p = 0.00045, q\_primary = 0.00718。
- `meta_R_temporal_pole`, KJ, M9 relation vector abs: pre = 0.020, post = -0.013, delta = -0.034, p = 0.00120, q\_primary = 0.0191。
- `meta_R_PPA_PHG`, YY, relation-vector length: delta = -0.035, p = 0.00063, q\_model\_family = 0.0262。

但是 condition contrast 显示 temporal pole 的核心差异方向是 KJ > YY decoupling：

- `meta_L_temporal_pole`, M9 abs, YY-KJ decoupling = -0.063, p = 0.00044, q\_primary = 0.00702。
- `meta_R_temporal_pole`, M9 abs, YY-KJ decoupling = -0.057, p = 0.00214, q\_primary = 0.0171。
- `meta_R_temporal_pole`, M9 direct, YY-KJ decoupling = -0.032, p = 0.00953, q\_primary = 0.0509。

网络层面也一致：

- semantic/metaphor network, M9 abs: YY-KJ decoupling = -0.020, p = 0.0113, q = 0.0339。
- hippocampus core, M9 abs: YY-KJ decoupling = -0.022, p = 0.0569, q = 0.0750。
- spatial/context network, M9 abs: YY-KJ decoupling = -0.0167, p = 0.0750, q = 0.0750。

所以这一块的建议写法是：**relation-vector 结果提示 temporal pole 确实参与关系向量结构变化，但其方向不是 YY-specific relation decoupling，而是和 KJ 条件形成分离。这为主机制提供边界条件，而不是主支持证据。**

对应图：`fig_relation_vector_rsa.png`, `fig_relation_vector_condition_contrast.png`, `fig_relation_vector_network_dissociation.png`, `fig_relation_step5c_conjunction.png`。

## 8. Brain-behavior 连接

Relation behavior prediction 有一些未校正层面的桥接结果，但 FDR 后不稳：

- `meta_L_IFG`, YY, relation\_pre -> memory accuracy: Pearson r = 0.500, p = 0.00790；Spearman r = 0.579, p = 0.00155；q\_primary = 0.132。
- `meta_L_IFG`, YY, relation\_pre -> retrieval efficiency: Pearson r = 0.439, p = 0.0219；Spearman r = 0.567, p = 0.00206；q\_primary = 0.132。
- `meta_R_PPA_PHG`, YY-KJ relation decoupling -> memory accuracy: Pearson r = -0.521, p = 0.00531；q\_primary = 0.334。

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

| ROI set        | ROI                     | condition | retrieval - post |        p |       q |
| -------------- | ----------------------- | --------- | ---------------: | -------: | ------: |
| meta\_spatial  | meta\_R\_PPA\_PHG       | YY        |            0.086 | 0.000720 | 0.00274 |
| meta\_spatial  | meta\_R\_PPC\_SPL       | YY        |            0.081 |  0.00117 | 0.00427 |
| meta\_spatial  | meta\_L\_RSC\_PCC       | YY        |            0.100 |  0.00169 | 0.00589 |
| meta\_metaphor | meta\_L\_IFG            | YY        |            0.074 |  0.00178 | 0.00672 |
| meta\_metaphor | meta\_R\_pMTG\_pSTS     | YY        |            0.065 |  0.00305 |  0.0108 |
| meta\_spatial  | meta\_R\_hippocampus    | YY        |            0.047 |  0.00545 |  0.0182 |
| meta\_metaphor | meta\_L\_temporal\_pole | YY        |            0.066 |  0.00751 |  0.0253 |
| meta\_spatial  | meta\_R\_RSC\_PCC       | YY        |            0.085 |   0.0102 |  0.0326 |
| meta\_metaphor | meta\_R\_AG             | YY        |            0.063 |   0.0106 |  0.0339 |
| meta\_spatial  | meta\_L\_hippocampus    | YY        |            0.035 |   0.0161 |  0.0495 |

KJ 自身没有任何 `retrieval - post pair similarity` 通过 primary q < .05。最接近的是 `meta_L_PPC_SPL`，但方向为负，mean = -0.051, p = 0.0326, q = 0.093；其他 KJ ROI 多数为不显著的弱正向或弱负向。也就是说，run7 的“post 后回弹”主要来自 YY 条件自身，而不是 YY/KJ 都同步增加。

在确认上述过程后，再看 YY-KJ 比较。YY 相对 KJ 的 `retrieval - post pair similarity` 在以下 ROI 中显著更强：

| ROI set        | ROI                     | metric                           | YY-KJ effect |        p |        q |
| -------------- | ----------------------- | -------------------------------- | -----------: | -------: | -------: |
| meta\_spatial  | meta\_R\_PPA\_PHG       | retrieval - post pair similarity |        0.083 | 4.76e-07 | 1.91e-05 |
| meta\_spatial  | meta\_R\_PPC\_SPL       | retrieval - post pair similarity |        0.085 | 4.21e-05 | 8.42e-04 |
| meta\_metaphor | meta\_R\_temporal\_pole | retrieval - post pair similarity |        0.104 | 2.91e-04 |  0.00466 |
| meta\_metaphor | meta\_R\_IFG            | retrieval - post pair similarity |        0.084 | 1.85e-04 |  0.00466 |
| meta\_spatial  | meta\_R\_hippocampus    | retrieval - post pair similarity |        0.048 |  0.00310 |   0.0413 |

解释上，这不是推翻 pre-to-post learned-edge differentiation，而是说明：YY 的 trained-pair similarity 在 post 阶段下降后，run7 retrieval 阶段自身出现 pair-structure rebound；这个 rebound 又显著强于 KJ，尤其在右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG 和右海马。它可以作为主机制之后的 retrieval-stage 补充证据：学习后的 pair/edge geometry 不是纯粹消失，而是在最终提取阶段以任务相关方式重新出现。

这个结果可以形成一个更完整的“分化 -> retrieval 回弹/再聚集”过程解释，但需要谨慎表述。以 YY 条件的几个关键 ROI 为例：

| ROI                     | pre pair similarity | post pair similarity | run7 retrieval similarity | 轨迹解释                       |
| ----------------------- | ------------------: | -------------------: | ------------------------: | -------------------------- |
| meta\_R\_PPA\_PHG       |               0.174 |                0.086 |                     0.172 | post 明显分化，run7 几乎回到 pre    |
| meta\_R\_PPC\_SPL       |               0.158 |                0.098 |                     0.179 | post 分化，run7 均值超过 pre      |
| meta\_R\_IFG            |               0.156 |                0.108 |                     0.168 | post 分化，run7 均值超过 pre      |
| meta\_R\_temporal\_pole |               0.208 |                0.123 |                     0.167 | post 分化，run7 部分回弹但仍低于 pre  |
| meta\_R\_hippocampus    |               0.173 |                0.064 |                     0.111 | post 强分化，run7 部分回弹但仍低于 pre |

进一步把 `run7 - pre` 单独检验后，结论需要区分“均值轨迹”和“统计显著性”。在均值上，`meta_R_PPC_SPL` 和 `meta_R_IFG` 的 run7 确实高于 pre，`meta_R_PPA_PHG` 几乎回到 pre；但这些 `run7 - pre` 正向差异没有通过校正，也没有稳定的一致方向。

| ROI                     | run7 - pre |      p |     q | 说明                |
| ----------------------- | ---------: | -----: | ----: | ----------------- |
| meta\_R\_PPA\_PHG       |     -0.003 |  0.910 | 0.910 | 几乎回到 pre，但未超过 pre |
| meta\_R\_PPC\_SPL       |      0.021 |  0.435 | 0.725 | 均值超过 pre，但不显著     |
| meta\_R\_IFG            |      0.012 |  0.700 | 0.934 | 均值超过 pre，但不显著     |
| meta\_R\_temporal\_pole |     -0.042 |  0.236 | 0.707 | 部分回弹，仍低于 pre      |
| meta\_R\_hippocampus    |     -0.062 | 0.0096 | 0.076 | 未校正低于 pre，校正后趋势   |

因此，最稳妥的写法不是“run7 全面高于 pre”，而是：**学习后先出现 relation-edge differentiation，最终 retrieval 阶段出现 task-driven pair-structure rebound / partial re-binding；部分 ROI 的均值可回到或略高于 pre，但整体没有稳定的** **`run7 > pre`** **证据**。这能解释为什么 post 阶段看起来像“去相似化”，但 run7 提取时又能重新表现出 pair-level structure。

post-to-retrieval same-vs-different pair reinstatement 本身较弱：`meta_R_hippocampus` YY 的 pair reinstatement 为正，p = 0.0207，但 primary q = 0.061；`meta_L_hippocampus` YY p = 0.0431, q = 0.119。因此 reinstatement 可写成海马方向性趋势，不应作为校正后主结论。

run7 retrieval success prediction 使用 item-level GEE，并在 subject 内拆分 within-subject 与 between-subject 成分。主要结果：

- `meta_L_pMTG_pSTS`, KJ, retrieval pair similarity -> memory: within beta = 0.110, p = 0.000755, q\_primary = 0.0362。
- `meta_R_AG`, YY, retrieval pair similarity -> memory: within beta = 0.159, p = 0.00217, q\_primary = 0.0521。
- `meta_L_PPA_PHG`, YY, retrieval pair similarity -> log RT correct: within beta = 0.0164, p = 7.52e-05, q\_primary = 0.00451。
- `meta_L_RSC_PCC`, YY, retrieval pair similarity -> log RT correct: within beta = 0.0170, p = 0.00180, q\_primary = 0.0540。

RT 的 beta 为正，表示 retrieval pair similarity 更高时正确 trial 反应更慢，不是简单的效率增强。因此 run7 行为桥接目前应写为探索性：retrieval pair geometry 与最终表现有关，但方向和条件并不完全支持“更强 reinstatement -> 更好/更快记忆”的单一故事。

## 10. Run7 MVPA classification

为了检验 run7 中显著回升的 ROI 是否不仅有 pair similarity rebound，而且包含可解码的 YY/KJ 条件信息，新增脚本：

- `fmri_mvpa/fmri_mvpa_roi/run7_meta_roi_decoding.py`
- 输出目录：`E:/python_metaphor/paper_outputs/qc/run7_mvpa_decoding/`
- 主表：`E:/python_metaphor/paper_outputs/tables_main/table_run7_mvpa_decoding.tsv`
- 行为桥接表：`E:/python_metaphor/paper_outputs/tables_main/table_run7_mvpa_evidence_behavior.tsv`

分析使用 26 名有 run7 neural pattern 的被试，0 failure。预先聚焦 5 个有 run7 retrieval-minus-post rebound 的 ROI：`meta_R_PPA_PHG`, `meta_R_PPC_SPL`, `meta_R_temporal_pole`, `meta_R_IFG`, `meta_R_hippocampus`。QC 显示每个 subject x ROI 均有 140 个 trial，YY/KJ 各 70 个 trial，`w/ew` 四类各 35 个 trial。脚本最初按原始 `pair_id` 留出时发现 YY 的 `pair_id` 为 1-35、KJ 为 36-70，单个 fold 只有一个类别，因此已改为按 condition 内的同序号 pair 成对留出，避免训练集与测试集共享同序号 YY/KJ pair。

主分析是 pair-held-out YY/KJ decoding：每个 fold 同时留出一个 YY pair 和一个 KJ pair，训练其余 pair，测试被留出的两个条件。结果显示 4/5 个 ROI 的 balanced accuracy 显著高于 chance：

| ROI                     | balanced accuracy | above chance |        p | q\_primary |   dz |
| ----------------------- | ----------------: | -----------: | -------: | ---------: | ---: |
| meta\_R\_hippocampus    |             0.565 |        0.065 | 2.98e-06 |   3.04e-05 | 1.17 |
| meta\_R\_temporal\_pole |             0.553 |        0.053 | 4.05e-06 |   3.04e-05 | 1.15 |
| meta\_R\_PPA\_PHG       |             0.564 |        0.064 | 1.72e-05 |   8.58e-05 | 1.04 |
| meta\_R\_PPC\_SPL       |             0.547 |        0.047 | 0.000190 |   0.000711 | 0.86 |
| meta\_R\_IFG            |             0.513 |        0.013 |    0.187 |      0.357 | 0.27 |

这说明 run7 的右海马、右颞极、右 PPA/PHG、右 PPC/SPL 不只是 similarity 均值回升，也包含可用于区分 YY/KJ 的 multivariate condition information。这个证据可以作为 retrieval-stage pair/condition structure 的补充，但不能替代前面的 RSA 主结果。

为了检查这种分类是否是角色不变的抽象 YY/KJ code，又做了 cross-role generalization：`w -> ew` 与 `ew -> w`。结果没有任何 ROI 通过 FDR，最接近的是 `meta_R_PPA_PHG`：

- `w -> ew`: balanced accuracy = 0.518, p = 0.0671, q\_primary = 0.201。
- `ew -> w`: balanced accuracy = 0.520, p = 0.126, q\_primary = 0.314。

因此，原先设想的“用 yyw/kjw 训练后准确分类 yyew/kjew”的强版本目前不成立。更合理的解释是：run7 中存在 YY/KJ condition-discriminative information，但它不是稳定跨 `w/ew` 泛化的角色不变表征。

为了定位 cross-role 失败的原因，又做了 role-specific pair-held-out decoding。`w-only` 和 `ew-only` 在右 PPA/PHG、右海马、右颞极、右 PPC/SPL 中均显著：

| analysis | ROI                     | balanced accuracy |        p | q\_within\_analysis |
| -------- | ----------------------- | ----------------: | -------: | ------------------: |
| w-only   | meta\_R\_PPA\_PHG       |             0.592 | 1.44e-06 |            7.18e-06 |
| w-only   | meta\_R\_hippocampus    |             0.585 | 1.18e-05 |            1.97e-05 |
| w-only   | meta\_R\_temporal\_pole |             0.591 | 9.99e-06 |            1.97e-05 |
| w-only   | meta\_R\_PPC\_SPL       |             0.584 | 3.95e-05 |            4.94e-05 |
| ew-only  | meta\_R\_temporal\_pole |             0.590 | 2.07e-06 |            1.03e-05 |
| ew-only  | meta\_R\_PPA\_PHG       |             0.591 | 2.32e-05 |            5.80e-05 |
| ew-only  | meta\_R\_hippocampus    |             0.574 | 0.000105 |            0.000174 |
| ew-only  | meta\_R\_PPC\_SPL       |             0.568 | 0.000603 |            0.000754 |

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

| ROI                     | run3 balanced accuracy |        p | q\_primary |   dz |
| ----------------------- | ---------------------: | -------: | ---------: | ---: |
| meta\_R\_PPA\_PHG       |                  0.591 | 1.81e-06 |   3.63e-05 | 1.14 |
| meta\_R\_hippocampus    |                  0.567 | 1.72e-05 |   0.000172 | 0.99 |
| meta\_R\_temporal\_pole |                  0.563 | 0.000170 |   0.000679 | 0.82 |
| meta\_R\_PPC\_SPL       |                  0.545 |  0.00403 |    0.00896 | 0.59 |
| meta\_R\_IFG            |                  0.532 |  0.00770 |     0.0154 | 0.54 |

run4 中同样 5/5 个 ROI 显著高于 chance：

| ROI                     | run4 balanced accuracy |        p | q\_primary |   dz |
| ----------------------- | ---------------------: | -------: | ---------: | ---: |
| meta\_R\_temporal\_pole |                  0.571 | 5.74e-05 |   0.000382 | 0.90 |
| meta\_R\_PPA\_PHG       |                  0.575 | 8.89e-05 |   0.000444 | 0.87 |
| meta\_R\_PPC\_SPL       |                  0.573 | 0.000600 |    0.00165 | 0.73 |
| meta\_R\_IFG            |                  0.568 | 0.000591 |    0.00165 | 0.74 |
| meta\_R\_hippocampus    |                  0.570 | 0.000661 |    0.00165 | 0.73 |

这说明这些 run7 rebound ROI 在学习阶段已经可以区分 YY/KJ。它加强了“这些区域确实承载条件相关学习信息”的论点，而不是只在最终 retrieval 才出现 YY/KJ 差异。

但两个边界条件也很重要：

- run4 - run3 的 accuracy 提升没有任何 ROI 通过 FDR；最接近的是 `meta_R_IFG`，delta = 0.036, p = 0.088, q = 0.441。
- 跨 run 泛化没有任何 ROI 通过 FDR；最接近的是 `meta_R_PPA_PHG` 的 `run3 -> run4`，accuracy = 0.515, p = 0.0404, q\_primary = 0.0735。

因此，学习阶段 MVPA 不应写成“学习使 YY/KJ classifier 从 run3 到 run4 稳定增强”，也不应写成“分类边界跨 run 稳定泛化”。更稳妥的写法是：**右 PPA/PHG、右 PPC/SPL、右颞极、右 IFG、右海马在学习阶段 run-local pattern 中已经包含 YY/KJ discriminative information；这种信息与 run7 的 retrieval-stage decoding 相呼应，但当前没有证据表明它在 run3 到 run4 中显著增强或形成稳定跨 run 的分类边界。**

这条结果可以补充当前故事：post 阶段的 learned-edge differentiation 并不是凭空出现，学习阶段这些 ROI 已经能区分 YY/KJ 条件；但真正和“关系边重组”贴合的主证据仍然是 post 的 edge-specific RSA，以及 run7 的 pair-structure rebound / role-dependent decoding。

## 12. Pre/post isolated-word MVPA controls

为了把 learning/run7 的分类结果放回完整学习过程，新增 pre/post isolated-word 阶段的 YY/KJ 二分类分析：

- `fmri_mvpa/fmri_mvpa_roi/prepost_meta_roi_binary_decoding.py`

metadata 的真实 run 编码需要明确：pre 是 run1/2，post 是 run5/6；run3/4 是学习阶段，不是 post。由于 pre 的 run1+run2、post 的 run5+run6 合起来才覆盖完整 YY/KJ 词集，主报告口径采用 **phase-combined pair-held-out**：在每个 phase 内合并两个 run，每折同时留出一个 YY pair 和一个 KJ pair，训练其余 pair。这个检验回答的是“完整 pre/post 词集内是否存在 YY/KJ 可解码信息”，比跨 run 泛化更贴合当前问题。

YY/KJ 二分类 QC：28 名被试，0 failure；每个 subject x ROI 有 280 个 YY/KJ trial，其中 pre = 140、post = 140；每个阶段 YY = 70、KJ = 70。phase-combined pair-held-out 中，每个 subject x ROI x phase 有 35 个 fold。

pre 阶段整体较弱，只有右颞极达到未校正显著，校正后不稳：

| ROI                     | pre combined pair-held-out accuracy |      p | q\_primary |
| ----------------------- | ----------------------------------: | -----: | ---------: |
| meta\_R\_temporal\_pole |                               0.520 | 0.0393 |     0.0983 |
| meta\_R\_PPA\_PHG       |                               0.514 |  0.244 |      0.443 |
| meta\_R\_PPC\_SPL       |                               0.511 |  0.307 |      0.453 |
| meta\_R\_hippocampus    |                               0.510 |  0.359 |      0.453 |
| meta\_R\_IFG            |                               0.504 |  0.571 |      0.635 |

post 阶段出现更清楚的正向分类，右颞极通过 primary-family FDR，其他 ROI 多数为未校正显著或趋势：

| ROI                     | post combined pair-held-out accuracy |       p | q\_primary |
| ----------------------- | -----------------------------------: | ------: | ---------: |
| meta\_R\_temporal\_pole |                                0.539 | 0.00254 |     0.0354 |
| meta\_R\_hippocampus    |                                0.525 |  0.0162 |     0.0983 |
| meta\_R\_PPC\_SPL       |                                0.516 |  0.0306 |     0.0983 |
| meta\_R\_IFG            |                                0.520 |  0.0377 |     0.0983 |
| meta\_R\_PPA\_PHG       |                                0.520 |  0.0453 |      0.101 |

这组结果不需要被写成“post 显著强于 pre”。更合理的叙述是：pre 阶段没有稳定 YY/KJ 区分；学习阶段 YY/KJ 区分显著出现；post isolated-word 阶段保留了部分 condition information，尤其右颞极；最终 run7 retrieval 阶段又重新变得稳健。这个阶段性模式支持“学习-保留-提取”的动态过程，而不是一个贯穿全程的静态材料类别码。真正的机制增强证据仍然由 post edge-specific RSA 承担，MVPA 提供的是阶段状态证据。

把 MVPA 结果按四个阶段串联，可以得到更清晰的动态图景：

| 阶段        | 数据与口径                                         | 主要结果                              | 解释                                                  |
| --------- | --------------------------------------------- | --------------------------------- | --------------------------------------------------- |
| pre       | run1+run2, YY/KJ phase-combined pair-held-out | 整体弱；右颞极 0.520, q = 0.098          | 初始 isolated-word 阶段没有稳定 YY/KJ 区分                    |
| learning  | run3/run4, YY/KJ within-run pair-held-out     | run3/run4 中 5 个 ROI 均显著           | 学习任务中 YY/KJ condition-discriminative geometry 被显著调动 |
| post      | run5+run6, YY/KJ phase-combined pair-held-out | 右颞极显著，0.539, q = 0.035；其余 ROI 为趋势 | 学习后静态词项阶段保留部分 YY/KJ 信息                              |
| retrieval | run7, YY/KJ pair-held-out                     | 右海马、右颞极、右 PPA/PHG、右 PPC/SPL 显著    | 提取阶段 YY/KJ 区分重新稳健出现，与 pair-structure rebound 对应     |

因此，MVPA 最适合写成：**YY/KJ discriminative geometry 并不是 pre 阶段已有的稳定静态类别码，而是在 learning 阶段显著出现，在 post isolated-word 阶段部分保留，并在 retrieval 阶段重新稳健出现。**

## 13. 学习动态、searchlight 与 noise ceiling

Relation learning dynamics 在 meta ROI 中没有稳定通过 FDR 的 run3-run4 轨迹结果：

- `meta_L_hippocampus`, YY, M9 abs: delta = -0.031, p = 0.0258, q\_primary = 0.516。
- `meta_L_hippocampus`, KJ, M9 abs: delta = -0.030, p = 0.0296, q\_primary = 0.592。

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

## 16. 本轮新增结果：Learning state 与 Learning-memory bridge

> 说明：本节只汇报本轮新增的 learning-state / learning-memory bridge 结果，不重复 Section 3/4/9 的既有主结果。
> 临时输出已整理回 `E:/python_metaphor/paper_outputs/`，下列数据来源均使用理论输出目录。

### 16.1 Learning state：Learning Sentence-Level Profile（run3 / run4）

数据来源：

- `E:/python_metaphor/paper_outputs/qc/learning_condition_rdm_meta_metaphor/learning_profile_group_one_sample.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_condition_rdm_meta_spatial/learning_profile_group_one_sample.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_condition_rdm_meta_metaphor/learning_profile_group_pairwise.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_condition_rdm_meta_spatial/learning_profile_group_pairwise.tsv`

先看 one-sample 结果（检验 `between_minus_within > 0`）。校正后显著 ROI 如下：

| ROI set        | ROI               | phase |  n | mean\_effect |     t |        p |   q\_bh |
| -------------- | ----------------- | ----- | -: | -----------: | ----: | -------: | ------: |
| meta\_metaphor | meta\_L\_AG       | run3  | 28 |       0.0154 | 3.973 | 4.75e-04 | 0.00760 |
| meta\_spatial  | meta\_L\_PPA\_PHG | run3  | 28 |       0.0119 | 4.352 | 1.73e-04 | 0.00347 |
| meta\_spatial  | meta\_R\_PPA\_PHG | run3  | 28 |       0.0082 | 3.649 |  0.00111 | 0.01110 |
| meta\_spatial  | meta\_L\_RSC\_PCC | run3  | 28 |       0.0091 | 3.171 |  0.00376 | 0.02510 |
| meta\_spatial  | meta\_R\_RSC\_PCC | run3  | 28 |       0.0123 | 2.966 |  0.00624 | 0.03122 |

从分布上看，虽然 `q<0.05` 的显著结果主要集中在 `run3`，但 `run4` 中 `meta_metaphor` 的 `meta_L_temporal_pole / meta_L_IFG / meta_L_AG` 与 `meta_spatial` 的 `meta_L_PPA_PHG` 仍保持未校正正向、且在较宽松校正下接近显著，说明 learning 阶段的 sentence-level condition geometry 并非只在单一 run 中短暂出现。

再看 `run4 − run3` 的 paired 结果：

| ROI set        | ROI | phase\_a | phase\_b | mean\_diff |  t |  p | q\_bh |
| -------------- | --- | -------- | -------- | ---------: | -: | -: | ----: |
| meta\_metaphor | —   | run4     | run3     | 无校正后显著 ROI |  — |  — |     — |
| meta\_spatial  | —   | run4     | run3     | 无校正后显著 ROI |  — |  — |     — |

因此，本节最稳妥的结论是：

- learning 阶段本身已经出现 sentence-level 的 YY/KJ condition geometry；
- 但当前没有证据表明该 geometry 从 `run3` 到 `run4` 稳定单调增强。

### 16.2 Learning within-item reliability

数据来源：

- `E:/python_metaphor/paper_outputs/qc/learning_within_item_reliability_meta_metaphor/learning_within_item_reliability_group_one_sample.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_within_item_reliability_meta_spatial/learning_within_item_reliability_group_one_sample.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_within_item_reliability_meta_metaphor/learning_within_item_reliability_group_yy_vs_kj.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_within_item_reliability_meta_spatial/learning_within_item_reliability_group_yy_vs_kj.tsv`

该分析检验同一 item 在 `run3` 与 `run4` 间是否有稳定表征。结果显示，学习阶段的 item-level reliability 很稳，但不是 YY 特异：

- `meta_metaphor` 中显著 ROI 主要集中在左侧语义/语言网络：`meta_L_pMTG_pSTS` YY mean = 0.127, q = 0.00023；`meta_L_AG` YY mean = 0.117, q = 0.00026；`meta_L_IFG` YY mean = 0.105, q = 0.00031；KJ 在 `meta_L_pMTG_pSTS / meta_L_IFG / meta_L_AG` 也显著。
- `meta_spatial` 中显著 ROI 更广泛，尤其 scene/context 网络：`meta_R_RSC_PCC` YY mean = 0.089, q = 2.68e-05；`meta_L_PPA_PHG` KJ mean = 0.094, q = 0.00391；`meta_L/R_RSC_PCC`、`meta_L/R_PPA_PHG`、`meta_L/R_precuneus`、`meta_L/R_PPC_SPL` 多数为正向显著。
- `YY - KJ` 对比没有任何 ROI 通过 FDR。

因此，这部分适合作为 learning-state 的质量控制和机制背景：学习阶段确实存在可重复的 item trace，但它反映的是学习材料/句子表征稳定性，不应被写成隐喻条件专属机制。


## 17. 已撤下的探索性分析

为避免主线被弱证据带偏，当前版本不再把 ERS、cross-phase reinstatement、learning-stage decomposition、learning-subsequent-memory、learning-to-later-stage continuity 写入主结果。撤下原因是：

- ERS 与 cross-phase continuity 在 pre 阶段同样很强，难以解释为 learning-induced reinstatement。
- learning-stage decomposition 只能说明 learning 阶段已有 condition geometry，不能证明 run3 到 run4 的稳定变化。
- learning-subsequent-memory 的方向复杂，不能支持“learning fidelity 越高，记忆越好”的简单机制。
- learning-to-pre/post/retrieval continuity 缺少 retrieval stronger 的线性梯度，不适合作为主桥接证据。

后续主线转为 item-level predictive mechanism：检验 learning-stage item trace 是否预测 post-stage learned-edge differentiation，以及 post-stage edge specificity 是否预测 run7 memory。对应执行方案见 `learning_post_memory_prediction_spec.md`。

## 18. Learning-Post-Memory predictive mechanism 执行结果

数据来源：

- `E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/learning_to_post_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/post_edge_memory_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/negative_control_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/staged_model_comparison.tsv`

本轮重新构建了 item-level mechanism table，并显式区分两个 ID：

- `pair_id / original_pair_id`：材料原始编号，YY 缺 `20,25,30,31,34`，KJ 缺 `4,18,32,35,40`。
- `template_pair_id`：`stimuli_template.csv` 与 pre/post/retrieval pair-similarity 输出使用的 1-35 词对编号。

QC 结果显示，YY 与 KJ 各保留 35 个 item，没有发生数字编号交集过滤。全表包含 28 名被试、18 个 meta ROI、35280 行。模型层面由于 pre/post/retrieval 与 run7 神经-行为交集限制，主模型通常为 26 名被试、70 个 condition-items、1820 行。

### 18.1 Learning specificity -> post edge specificity

主模型：

```text
post_edge_specificity ~ learning_specificity_z * condition + material_covariates
```

以及 pre-baseline control / residualized transformation control 均已执行。结果显示，`learning_specificity_z` 与 `learning_specificity_z × condition` 没有任何 ROI 在 meta family FDR 后显著。

最接近的探索性结果集中在 `meta_spatial`：

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| pre-control | meta_R_PPC_SPL | learning_specificity_z | -0.0247 | 0.0155 | 0.0839 |
| pre-control | meta_L_precuneus | learning_specificity_z | +0.0247 | 0.0168 | 0.0839 |
| residualized | meta_R_PPC_SPL | learning_specificity_z | -0.0244 | 0.0166 | 0.0871 |
| residualized | meta_L_precuneus | learning_specificity_z | +0.0251 | 0.0174 | 0.0871 |

因此，目前不能写成“learning-stage item trace 稳定预测 post-stage learned-edge differentiation”。这条路径最多作为探索性趋势，且方向在不同空间 ROI 中不一致。

### 18.2 Post edge specificity -> run7 memory

主模型：

```text
memory ~ post_edge_specificity_z * condition + material_covariates
```

结果也没有任何核心 post-edge 预测项通过 meta family FDR。最接近的结果包括：

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| primary | meta_R_precuneus | post_edge_specificity_z | +0.0888 | 0.0171 | 0.1710 |
| pre-control | meta_R_precuneus | post_edge_specificity_z | +0.1066 | 0.0153 | 0.1525 |
| residualized | meta_L_AG | post_edge_specificity_resid_z × YY | +0.1601 | 0.0119 | 0.0912 |
| residualized | meta_R_pMTG_pSTS | post_edge_specificity_resid_z × YY | +0.1177 | 0.0228 | 0.0912 |

这些结果提示 post-edge specificity 可能与后续记忆存在局部关联，尤其在 precuneus / AG / pMTG-pSTS 等区域，但校正后不足以作为主 brain-behavior bridge。

### 18.3 Negative controls

负控制结果提醒我们不要升级解释：

- `pseudo_edge_drop` 没有稳定被 `learning_specificity_z` 预测，这是好的。
- 但 `pre_pair_similarity ~ learning_specificity_z` 在 `meta_R_hippocampus` 中显著：beta = 0.0242, p = 0.000633, q = 0.00633。
- shuffled predictor 对 memory 在 `meta_R_AG` / `meta_L_IFG` 出现显著项，说明当前 item-level memory prediction 对 permutation / item structure 较敏感。

因此，当前 predictive mechanism 没有形成足够干净的因果链或行为桥接证据。

### 18.4 Staged model comparison

staged comparison 是 secondary，不作为 mediation。描述性结果显示，retrieval-pair model 最常成为 best model：

- `M3_retrieval_pair` 是 9/18 个 ROI 的最低 log-loss 模型。
- `M4_retrieval_minus_post` 是 6/18 个 ROI 的最低 log-loss 模型。
- 但平均 log-loss 改善很小，`M3` mean delta log-loss 约 0.00027。

因此 staged model comparison 只能支持一个很弱的倾向：run7 retrieval geometry 比 learning/post item-level predictor 更贴近 run7 memory，但增量很小，不能作为强行为机制。

### 18.5 结论

这轮 spec 分析没有补成强的 learning-to-post-to-memory 预测链。更稳妥的写法是：

> 学习阶段确实存在 item trace 与 condition geometry，post 阶段存在 learned-edge differentiation，retrieval 阶段出现 pair-structure rebound 与 YY/KJ 可分类表征；但 item-level learning trace 是否预测 post reorganization，以及 post edge specificity 是否进一步预测 run7 memory，目前缺少校正后稳定证据。

因此，当前论文主线仍应放在“learning 阶段调动 YY/KJ condition geometry -> post 阶段 relation-edge differentiation -> retrieval 阶段 re-binding / task-driven reinstatement”。Learning-post-memory predictive mechanism 可以作为严格探索性边界分析，用来说明我们检验过行为桥接，但不应升级为主机制。

## 19. Learning reactivation / directional mapping 扩展分析

本节对应新方案 `learning_reactivation_mapping_spec.md`。该分析的目的不是继续问“某个隐喻 ROI 是否显著”，而是更具体地检验老师建议的机制链：

```text
learning-stage semantic reactivation / behavioral update
    -> post-stage edge differentiation / directional mapping
    -> retrieval-stage re-binding / memory advantage
```

与第 18 节不同，本节不再使用泛化的 `learning_specificity = run3/run4 sentence self-specificity`。新的核心思想是：如果学习阶段真的在调动旧语义知识，那么学习句子的 neural pattern 应该重新激活 pre 阶段对应词语的 pattern；如果隐喻学习真的发生了 source-to-target mapping，那么 post 阶段 target/topic 应该相对更靠近 source/vehicle；如果这些学习阶段指标是机制性前因，它们应进一步预测 post edge differentiation 或 retrieval rebinding。

所有输出均写入独立目录，不覆盖旧分析：

- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_behavior_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_reactivation_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/directional_mapping_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/reactivation_to_post_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/network_bridge_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_reactivation_mapping_review.md`

### 19.1 学习行为：run3 理解判断与 run4 喜欢判断

原始 `data_events/sub-XX/sub-XX_run-3_events.csv` 和 `sub-XX_run-4_events.csv` 中包含之前整理 TSV 时丢掉的反应字段：

```text
run3: sentence3_RT / sentence3_RESP / sentence3_ACC
run4: sentence4_RT / sentence4_RESP / sentence4_ACC
```

由于 run3 是 5s 的理解判断，run4 是 3s 的喜欢判断，两者不能直接用 raw RT 差值解释为“学习变快”。因此本轮使用：

```text
run3_rt_z_subject_condition
run4_rt_z_subject_condition
learning_fluency_shift = run3_rt_z_subject_condition - run4_rt_z_subject_condition
```

作为 item 在各自任务下的相对理解负荷 / 二次加工流畅性 proxy。

QC：全表包含 28 名被试、1960 个 subject-item 行。`sub-10 run4` 有 59 个有效反应，`sub-20 run3` 有 50 个有效反应；这些是 no-response trial，表现为 `RT = 0`，不是解析错误。模型中这些 trial 的 RT 被设为缺失。

描述性结果：

| condition | run3 RT | run4 RT | fluency shift | understand yes | like yes |
| --- | ---: | ---: | ---: | ---: | ---: |
| KJ | 1947.9 ms | 1495.4 ms | +0.011 | 0.946 | 0.719 |
| YY | 2010.5 ms | 1406.9 ms | +0.003 | 0.916 | 0.810 |

模型结果显示：

| model | YY term beta | p | q | 解释 |
| --- | ---: | ---: | ---: | --- |
| run3 RT | ~0 | 0.759 | 0.759 | YY/KJ 初次理解 RT 无显著差异 |
| run4 RT | ~0 | 0.722 | 0.722 | YY/KJ 二次加工 RT 无显著差异 |
| fluency shift | -0.009 | 0.553 | 0.553 | 没有稳定的 YY 特异 learning-fluency shift |
| run3 understand yes | -0.483 | 0.0846 | 0.0846 | YY 理解确认比例低于 KJ 的趋势 |
| run4 like yes | +0.507 | 0.0309 | 0.0309 | YY 被判断为喜欢的比例高于 KJ |

因此，行为层面有一个可解释的现象：YY 在第二遍学习/评价阶段更容易获得喜欢判断，但这并没有转化为稳定的 RT-based fluency shift。因此后续不能把 `learning_fluency_shift` 写成强学习机制，只能作为探索性行为更新 proxy。

### 19.2 Learning-stage semantic reactivation

该分析检验学习句子 pattern 是否重新激活 pre 阶段组成词 pattern。对每个 subject x ROI x condition x item x run：

```text
react_role_a =
  Sim(LearningSentence_AB, PreWord_A)
  - mean Sim(LearningSentence_AB, OtherPreWords_same_condition_same_role)

react_role_b =
  Sim(LearningSentence_AB, PreWord_B)
  - mean Sim(LearningSentence_AB, OtherPreWords_same_condition_same_role)

react_pair_mean = mean(react_role_a, react_role_b)
```

其中角色定义为：

```text
YY: role_a = target/topic, role_b = source/vehicle
KJ: role_a = object, role_b = location/context
```

QC：28 名被试、18 个 ROI、35280 个 subject-item-ROI 行，learning reactivation QC pass rate = 1.0。

整体均值很小：

| ROI family | condition | react_pair_mean | run4-run3 delta |
| --- | ---: | ---: | ---: |
| meta_metaphor | KJ | +0.0001 | +0.0024 |
| meta_metaphor | YY | +0.0047 | -0.0021 |
| meta_spatial | KJ | -0.0006 | -0.0038 |
| meta_spatial | YY | +0.0011 | +0.0006 |

因此，learning reactivation 指标并没有显示出清晰、稳定、强幅度的 YY 特异 reactivation。

### 19.3 Directional mapping / assimilation

该分析原本用于检验隐喻学习后是否出现方向性 source-to-target 更新。最初版本使用：

```text
target_toward_source =
  Sim(PostRoleA, PreRoleB) - Sim(PreRoleA, PreRoleB)

source_toward_target =
  Sim(PostRoleB, PreRoleA) - Sim(PreRoleB, PreRoleA)

mapping_asymmetry =
  target_toward_source - source_toward_target
```

这个公式有一个关键问题：`Sim(PreRoleA, PreRoleB)` 与 `Sim(PreRoleB, PreRoleA)` 在相关相似性中是同一个值，因此在 `mapping_asymmetry` 中会被抵消。换句话说，旧版 `mapping_asymmetry` 实际近似退化为：

```text
Sim(PostRoleA, PreRoleB) - Sim(PostRoleB, PreRoleA)
```

它不能有效回答“target 是否相对自身原始表征更向 source 靠近”这个理论问题。因此，本轮将 directional mapping 改为 self-stability corrected 指标：

```text
target_source_over_self =
  Sim(PostRoleA, PreRoleB) - Sim(PostRoleA, PreRoleA)

source_target_over_self =
  Sim(PostRoleB, PreRoleA) - Sim(PostRoleB, PreRoleB)

mapping_asymmetry_self_corrected =
  target_source_over_self - source_target_over_self
```

其中，`target_source_over_self` 表示 post 阶段 RoleA 与对侧 RoleB 的 pre pattern 的相似性，是否超过 RoleA 自身的 pre-post stability。这个指标更接近“target/topic 是否相对自身基线向 source/vehicle 方向偏移”。旧版指标仍保留在输出中用于回溯，但不再作为方向性映射证据解释。

QC：28 名被试、18 个 ROI、35280 行，directional mapping QC pass rate = 1.0。修正后整体均值很小：

| ROI family | condition | old asymmetry | target_source_over_self | source_target_over_self | corrected asymmetry |
| --- | --- | ---: | ---: | ---: | ---: |
| meta_metaphor | KJ | +0.0008 | +0.0029 | +0.0003 | +0.0026 |
| meta_metaphor | YY | +0.0013 | -0.0017 | -0.0033 | +0.0016 |
| meta_spatial | KJ | +0.0036 | +0.0049 | +0.0027 | +0.0022 |
| meta_spatial | YY | +0.0025 | +0.0029 | +0.0017 | +0.0012 |

ROI-wise 条件模型中，`target_source_over_self` 与 `mapping_asymmetry_self_corrected` 没有任何 YY-KJ 条件效应通过 FDR。最接近的结果如下：

| model | ROI | outcome | YY beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| target over self | meta_R_pMTG_pSTS | target_source_over_self | -0.0391 | 0.107 | 0.620 |
| target over self | meta_L_RSC_PCC | target_source_over_self | -0.0267 | 0.130 | 0.841 |
| target over self | meta_R_AG | target_source_over_self | +0.0238 | 0.155 | 0.620 |
| target over self | meta_L_hippocampus | target_source_over_self | -0.0179 | 0.173 | 0.841 |
| corrected asymmetry | meta_L_PPA_PHG | mapping_asymmetry_self_corrected | -0.0265 | 0.229 | 0.776 |
| corrected asymmetry | meta_R_PPC_SPL | mapping_asymmetry_self_corrected | +0.0266 | 0.329 | 0.776 |

因此，目前不能写成“YY 学习后 target/topic 向 source/vehicle 方向性更新”。修正公式后，这条路径仍没有得到稳定的条件主效应支持。

### 19.4 Reactivation / fluency / mapping 是否预测 post edge differentiation

为检验学习阶段指标是否能解释 post 阶段 learned-edge differentiation，模型使用：

```text
post_edge_specificity ~ react_pair_mean_avg_z * condition
                      + pre_pair_similarity_z
                      + material_covariates

post_edge_specificity ~ learning_fluency_shift_z * condition
                      + pre_pair_similarity_z
                      + material_covariates

post_edge_specificity ~ mapping_asymmetry_self_corrected_z * condition
                      + pre_pair_similarity_z
                      + material_covariates

post_edge_specificity ~ target_source_over_self_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
```

修正后的 mapping 指标带来了一个更细的结果：在少数 meta_spatial ROI 中，mapping 指标与 post edge specificity 的关系存在 condition-dependent association，并且通过 FDR。

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| corrected asymmetry -> post | meta_L_PPA_PHG | corrected asymmetry × YY | +0.0365 | 0.000654 | 0.00654 |
| corrected asymmetry -> post | meta_R_PPC_SPL | corrected asymmetry × YY | -0.0424 | 0.00394 | 0.0197 |
| target over self -> post | meta_L_PPA_PHG | target_source_over_self × YY | +0.0355 | 0.00327 | 0.0327 |

这说明：虽然修正后的 directional mapping 没有整体 YY>KJ 条件效应，但在 item-level 上，mapping 指标与 post edge specificity 的关系在部分空间/场景相关 ROI 中存在 YY 特异调制。尤其是 `meta_L_PPA_PHG` 的方向为正，提示在 YY 中，target-source-over-self 或 corrected asymmetry 越强的 item，post edge specificity 越强；但 `meta_R_PPC_SPL` 的 corrected asymmetry interaction 为负，说明该效应不是一个方向统一、可直接升格为主机制的 source-to-target mapping 证据。

reactivation 与 fluency 的结果仍然没有稳定通过 FDR。最接近的探索性结果包括：

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| reactivation x fluency | meta_R_AG | react_pair_mean × YY | +0.0253 | 0.0338 | 0.149 |
| reactivation primary | meta_R_pMTG_pSTS | react_pair_mean × YY | +0.0285 | 0.0369 | 0.197 |
| reactivation x fluency | meta_R_pMTG_pSTS | react_pair_mean × YY | +0.0298 | 0.0373 | 0.149 |
| behavior fluency to post | meta_L_AG | learning_fluency_shift | +0.0215 | 0.0443 | 0.354 |

这些结果说明：learning-stage reactivation / fluency 目前不足以支持“学习阶段指标稳定预测 post edge differentiation”的主机制；修正后的 mapping 指标提供了局部的、探索性的 post edge 关联证据，但方向混合，需要谨慎写成“空间网络中存在 condition-dependent mapping-edge coupling”，而不是写成“YY 发生了稳定 source-to-target assimilation”。

更重要的是，负控制显示 `react_pair_mean_avg_z` 能显著预测 `pre_pair_similarity`：

| ROI set | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_RSC_PCC | react_pair_mean | +0.0587 | 0.000039 | 0.000389 |
| meta_metaphor | meta_R_AG | react_pair_mean | +0.0527 | 0.00314 | 0.01896 |
| meta_metaphor | meta_L_AG | react_pair_mean | +0.0449 | 0.00474 | 0.01896 |
| meta_spatial | meta_R_precuneus | react_pair_mean | +0.0485 | 0.00800 | 0.0400 |
| meta_metaphor | meta_L_IFG | react_pair_mean × YY | +0.0360 | 0.00593 | 0.0475 |

这说明当前 reactivation 指标受到 baseline item similarity / pre-existing semantic structure 的明显影响。换句话说，它不是一个足够干净的 learning-induced semantic reactivation 指标，因此不能作为强因果链证据。

### 19.5 Network-level bridge：semantic reactivation 与 hippocampal-spatial separation

为了提高稳定性并贴近记忆整合/分离文献，本轮还构建了两个预设网络：

```text
semantic network:
temporal pole, IFG, AG, pMTG/pSTS

hippocampal-spatial network:
hippocampus, PPA/PHG, RSC/PCC, PPC/SPL, precuneus
```

网络指标：

```text
semantic_reactivation = semantic ROI 的 react_pair_mean 平均
hpc_spatial_separation = hpc-spatial ROI 的 post_edge_specificity 平均
retrieval_rebinding = hpc-spatial ROI 的 retrieval_minus_post_pair_similarity 平均
```

重新 review 后发现，原始 `hpc_spatial_separation -> retrieval_rebinding` 模型存在 shared-post difference-score 风险：

```text
hpc_spatial_separation 主要来自 pre - post
retrieval_rebinding = retrieval - post
```

因此本轮按 Stage 1 的逻辑补充了 post-control 与 retrieval-similarity 控制模型。原始 raw model 仍显示强正向关系：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| hpc separation -> retrieval raw shared-post | hpc_spatial_separation_z | +0.1297 | 1.43e-18 | 4.29e-18 |

但在控制 hpc-spatial post similarity 后，这个 bridge 不再显著；直接预测 retrieval pair similarity 时也不显著：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| post-control rebinding | hpc_spatial_separation_z | +0.0236 | 0.653 | 0.653 |
| post-control rebinding | hpc_spatial_separation_z × YY | +0.0072 | 0.319 | 0.478 |
| retrieval similarity outcome | hpc_spatial_separation_z | +0.0236 | 0.653 | 0.653 |
| retrieval similarity outcome | hpc_spatial_separation_z × YY | +0.0072 | 0.319 | 0.478 |

这说明原始强相关很可能部分来自共同减去 post similarity 的数学耦合，不能再作为“post separation 预测 retrieval re-binding”的主机制证据。

独立的 Stage 1 复核也得到同样结论。对应输出位于：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage1_post_to_retrieval_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage1_post_to_retrieval_network_models.tsv`

Stage 1 中 raw shared-post network model 同样显著，但 post-control 与 retrieval-similarity outcome 后不显著：

| Stage 1 model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| raw shared-post | hpc_spatial_post_separation_z | +0.2550 | 2.24e-253 | 2.24e-253 |
| post-control rebinding | hpc_spatial_post_separation_z | +0.0094 | 0.732 | 0.732 |
| post-control rebinding | hpc_spatial_post_separation_z × YY | +0.0064 | 0.465 | 0.465 |
| retrieval similarity outcome | hpc_spatial_post_separation_z | +0.0094 | 0.732 | 0.732 |
| retrieval similarity outcome | hpc_spatial_post_separation_z × YY | +0.0064 | 0.465 | 0.465 |

因此，Stage 1、GEE network bridge、mixed-effects M5 三套复核一致指向同一判断：raw bridge 很强，但严格控制 post similarity 后不稳健。

同时，YY condition 本身显著预测更强的 hpc-spatial separation：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic_to_hpc_separation | YY condition | +0.0485 | 7.21e-09 | 1.51e-08 |
| mapping_to_hpc_separation | YY condition | +0.0485 | 7.54e-09 | 1.51e-08 |
| corrected_mapping_to_hpc_separation | YY condition | +0.0485 | 7.38e-09 | 1.46e-08 |
| target_over_self_to_hpc_separation | YY condition | +0.0483 | 9.70e-09 | 1.46e-08 |

但关键的 learning-stage predictor 并不显著：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic_to_hpc_separation | semantic_reactivation_z | -0.0012 | 0.844 | 0.844 |
| semantic_to_hpc_separation | semantic_reactivation_z × YY | +0.0018 | 0.805 | 0.805 |
| mapping_to_hpc_separation | semantic_mapping_asymmetry_z | +0.0060 | 0.328 | 0.328 |
| mapping_to_hpc_separation | semantic_mapping_asymmetry_z × YY | -0.0104 | 0.153 | 0.153 |
| corrected_mapping_to_hpc_separation | corrected semantic asymmetry_z | +0.0022 | 0.765 | 0.765 |
| corrected_mapping_to_hpc_separation | corrected semantic asymmetry_z × YY | -0.0065 | 0.464 | 0.464 |
| target_over_self_to_hpc_separation | semantic target over self_z | -0.0074 | 0.398 | 0.398 |
| target_over_self_to_hpc_separation | semantic target over self_z × YY | +0.0051 | 0.682 | 0.682 |

因此，network-level 结果支持“YY 在 hippocampal-spatial network 中有更强 post separation”，但不再支持“这种 separation 稳定预测 retrieval rebinding”。retrieval rebound 仍可作为 retrieval-stage 的相邻阶段证据，但不应写成由 post separation 直接驱动的 item-level bridge。

此外，retrieval rebinding 本身没有显著预测 memory：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| retrieval_to_memory | retrieval_rebinding_z | +0.0120 | 0.812 | 0.812 |
| retrieval_to_memory | retrieval_rebinding_z × YY | +0.0922 | 0.160 | 0.160 |

Subject-level 的 YY-KJ memory advantage 也没有被 YY-KJ neural advantage 稳定预测：

| model | predictor | beta | p | q | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| memory advantage | semantic reactivation advantage | +0.705 | 0.663 | 0.663 | 0.008 |
| memory advantage | hpc-spatial separation advantage | -0.066 | 0.870 | 0.870 | 0.001 |
| memory advantage | retrieval rebinding advantage | +0.195 | 0.627 | 0.627 | 0.010 |
| memory advantage | mapping asymmetry advantage | +0.115 | 0.802 | 0.802 | 0.003 |
| memory advantage | corrected mapping asymmetry advantage | +0.161 | 0.621 | 0.621 | 0.010 |
| memory advantage | target-source-over-self advantage | +1.042 | 0.082 | 0.082 | 0.116 |

其中 `target-source-over-self advantage` 对 YY-KJ memory advantage 有一个未校正趋势，但没有达到显著，因此只能作为后续探索线索，不能写成行为桥接证据。

### 19.6 本节结论

本轮扩展分析完整执行了老师建议中最关键的几条路径：learning-stage reactivation、learning behavior update、directional mapping、network-level separation/rebinding。结果给出了重要边界：

1. **行为层面**：YY 在 run4 liking 判断中高于 KJ，但 RT-based learning fluency shift 不显著。
2. **learning reactivation**：可以计算，但整体效应很小，且与 pre similarity 强相关。
3. **directional mapping**：旧公式存在 baseline cancellation 问题；改成 self-stability corrected 指标后，仍没有稳定证据支持 target/topic 向 source/vehicle 的整体方向性更新。
4. **learning -> post**：reactivation、fluency 没有稳定预测 post edge differentiation；修正后的 mapping 指标在 `meta_L_PPA_PHG` 与 `meta_R_PPC/SPL` 中显示 condition-dependent post-edge coupling，但方向混合，应作为探索性空间网络线索。
5. **network bridge**：YY 的 hpc-spatial post separation 很强；原始 hpc-spatial separation -> retrieval rebinding 模型显著，但 post-control 后不再显著，因此只能保留为 raw/sensitivity 结果，不能作为主机制桥。semantic reactivation / corrected mapping 也不是这个 separation 的显著上游预测因子。
6. **behavior bridge**：network-level neural advantage 仍未解释 YY-KJ memory advantage。

因此，这套扩展分析暂时不应升级为主机制。更稳妥的写法是：

> 当前结果支持 YY 学习后在 hippocampal-spatial network 中形成更强的 relation-edge separation；retrieval 阶段另有 pair-structure rebound 和 YY/KJ decoding 作为相邻阶段证据。修正后的 directional mapping 指标在部分空间 ROI 中与 post edge specificity 存在探索性耦合，但 learning-stage semantic reactivation、source-to-target directional mapping，以及 post separation -> retrieval re-binding 的严格 post-controlled bridge 目前都没有形成稳定证据，因此它们应作为探索性边界分析，而不是主因果链。

这意味着论文主线仍应保持在：

```text
learning-stage condition geometry
    -> post-stage learned-edge differentiation / hpc-spatial separation
    -> retrieval-stage re-binding
```

而不是写成：

```text
semantic reactivation / source-target mapping
    -> post edge differentiation
    -> memory advantage
```

## 20. Mixed-effects robustness：subject/item 层级结构复核

### 20.1 分析目的与模型设定

老师建议采用混合效应模型后，本轮将关键 item-level 结果重新用 mixed-effects model 复核。该分析不是新增一套主机制，而是用于检验：前面发现的 YY post-stage edge differentiation、hpc-spatial separation 与 retrieval re-binding 是否依赖于“先对 subject / item 求均值”的处理方式。

核心原则：

```text
unit = subject × condition × condition_item_id × ROI
random structure = (1 | subject) + (1 | condition_item_id)
condition_item_id = condition + "_" + original_pair_id
```

因此，YY 和 KJ 没有按照数字 `pair_id` 强行配对。例如 `yy_28` 与 `kj_28` 被视为两个不同 item pool 中的不同 item。`pre_pair_similarity_z` 使用的是 pre 阶段脑 pattern 的 pair similarity，不是 BERT、词标签或文本 embedding。

QC 结果显示：

| table | n rows | n subjects | n items | n ROIs | YY items | KJ items | YY/KJ item collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROI item table | 35280 | 28 | 70 | 18 | 35 | 35 | 0 |
| network item table | 1960 | 28 | 70 | 1 | 35 | 35 | 0 |
| trajectory long table | 133560 | 28 | 70 | 18 | 35 | 35 | 0 |

全部 M1-M5 模型均正常收敛，没有使用 fallback：

| model family | n models | status |
| --- | ---: | --- |
| M1 post edge condition | 18 | all ok |
| M2 four-phase trajectory | 18 | all ok |
| M3 reactivation / fluency -> post | 54 | all ok |
| M4 corrected mapping -> post | 36 | all ok |
| M5 network bridge | 6 | all ok |

M6 memory logistic mixed model 没有执行，因为当前 item-level memory 为 `0 / 0.5 / 1`，不是二元变量；按照预设 stop condition，不强行跑 logistic mixed model。

### 20.2 M1：post edge differentiation 的 mixed-effects 复核

模型：

```text
post_edge_specificity ~ condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject)
                      + (1 | condition_item_id)
```

结果支持前面的主线：在保留 subject 与 item 随机截距后，YY 在 meta_spatial 的 post edge specificity 仍更强。FDR 后最稳定的是空间/场景与空间注意相关 ROI：

| ROI set | ROI | YY beta | p | q |
| --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_PPA/PHG | +0.0566 | 0.00499 | 0.0345 |
| meta_spatial | meta_L_PPC/SPL | +0.0605 | 0.00691 | 0.0345 |
| meta_spatial | meta_L_PPA/PHG | +0.0496 | 0.0215 | 0.0536 |
| meta_spatial | meta_L_precuneus | +0.0606 | 0.0171 | 0.0536 |
| meta_metaphor | meta_L_AG | +0.0501 | 0.0407 | 0.171 |
| meta_metaphor | meta_L_IFG | +0.0420 | 0.0523 | 0.171 |

解释：mixed model 进一步支持“YY 学习后在 hippocampal-spatial / scene-context 相关系统中形成更强 relation-edge separation”。这个结果不是由简单 subject mean 或 item mean 驱动的。

### 20.3 M2：pre-learning-post-retrieval 四阶段轨迹

模型：

```text
pair_similarity ~ phase * condition
                + material_covariates
                + (1 | subject)
                + (1 | condition_item_id)
```

以 pre 为 reference 后，learning 阶段在多数 ROI 中显著低于 pre，说明学习句子 pattern 与 pre isolated-word pair similarity 不是简单同构，这与前面“学习阶段不是简单 pair similarity 增强”的结论一致。最强结果包括：

| ROI set | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_hippocampus | learning - pre | -0.1493 | 4.73e-34 | 4.73e-33 |
| meta_spatial | meta_L_hippocampus | learning - pre | -0.1234 | 1.29e-29 | 6.45e-29 |
| meta_metaphor | meta_R_pMTG/pSTS | learning - pre | -0.1400 | 6.89e-22 | 5.51e-21 |
| meta_metaphor | meta_R_temporal pole | learning - pre | -0.1777 | 2.05e-21 | 8.22e-21 |

同时，post × YY interaction 在若干空间与语义 ROI 为负，说明 YY 相比 KJ 有更强 post-pre differentiation：

| ROI set | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_hippocampus | post × YY | -0.0892 | 2.81e-07 | 1.74e-06 |
| meta_spatial | meta_R_PPA/PHG | post × YY | -0.0900 | 3.49e-07 | 1.74e-06 |
| meta_spatial | meta_L_PPC/SPL | post × YY | -0.0817 | 1.61e-05 | 5.38e-05 |
| meta_spatial | meta_L_PPA/PHG | post × YY | -0.0761 | 3.81e-05 | 7.63e-05 |
| meta_metaphor | meta_R_temporal pole | post × YY | -0.1205 | 5.35e-06 | 4.28e-05 |

这里的负向 interaction 对应 pair similarity 从 pre 到 post 的下降更强；与前面用 `post_edge_specificity = trained_edge_drop - pseudo_edge_drop` 得到的 YY learned-edge differentiation 是同一方向的证据。

### 20.4 M3：learning reactivation / fluency 是否预测 post edge

模型：

```text
post_edge_specificity ~ react_pair_mean_avg_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject)
                      + (1 | condition_item_id)

post_edge_specificity ~ learning_fluency_shift_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject)
                      + (1 | condition_item_id)
```

mixed model 没有支持 learning reactivation 或 learning fluency 稳定预测 post edge。最接近的 interaction 仍未通过 FDR：

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| reactivation -> post | meta_L_hippocampus | reactivation × YY | +0.0252 | 0.0527 | 0.488 |
| reactivation -> post | meta_R_PPC/SPL | reactivation × YY | -0.0266 | 0.0975 | 0.488 |
| fluency -> post | meta_L_PPA/PHG | fluency | -0.0078 | 0.482 | 0.640 |
| fluency -> post | meta_L_hippocampus | fluency | +0.0095 | 0.325 | 0.640 |

负控制再次显示 reactivation 与 pre-existing neural similarity 明显相关：

| ROI set | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_RSC/PCC | reactivation | +0.0589 | 3.82e-06 | 3.82e-05 |
| meta_spatial | meta_R_precuneus | reactivation | +0.0508 | 4.22e-05 | 0.000211 |
| meta_metaphor | meta_R_AG | reactivation | +0.0524 | 0.000210 | 0.00168 |
| meta_metaphor | meta_L_AG | reactivation | +0.0463 | 0.000473 | 0.00189 |
| meta_spatial | meta_L_PPA/PHG | reactivation | +0.0328 | 0.00228 | 0.00761 |
| meta_spatial | meta_R_PPA/PHG | reactivation | +0.0343 | 0.00308 | 0.00770 |

因此，mixed model 加强了一个边界结论：当前 reactivation 指标不是足够干净的 learning-induced semantic reactivation，不能作为 post edge differentiation 的上游因果证据。

### 20.5 M4：corrected directional mapping 的 mixed-effects 复核

模型：

```text
post_edge_specificity ~ target_source_over_self_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject)
                      + (1 | condition_item_id)

post_edge_specificity ~ mapping_asymmetry_self_corrected_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject)
                      + (1 | condition_item_id)
```

修正后的 mapping 指标在 mixed model 下仍只呈现边界性、方向混合的空间 ROI 结果：

| model | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| corrected asymmetry -> post | meta_R_PPC/SPL | corrected asymmetry × YY | -0.0469 | 0.00542 | 0.0503 |
| corrected asymmetry -> post | meta_L_PPA/PHG | corrected asymmetry × YY | +0.0392 | 0.0101 | 0.0503 |
| target over self -> post | meta_L_PPA/PHG | target_source_over_self × YY | +0.0374 | 0.0143 | 0.143 |

因此，mixed model 没有把 directional mapping 升级为稳定主机制。更合理的写法仍是：部分 spatial / scene ROI 中存在探索性的 mapping-edge coupling，但方向不统一，不能写成“YY 发生了统一的 source-to-target assimilation”。

### 20.6 M5：network-level bridge 的 mixed-effects 复核

模型：

```text
hpc_spatial_separation ~ condition
                       + semantic_reactivation_z / corrected_mapping_z
                       + pre_pair_similarity_z
                       + (1 | subject)
                       + (1 | condition_item_id)

retrieval_rebinding ~ hpc_spatial_separation_z * condition
                    + pre_pair_similarity_z
                    + (1 | subject)
                    + (1 | condition_item_id)

post-control sensitivity:

retrieval_rebinding ~ hpc_spatial_separation_z * condition
                    + pre_pair_similarity_z
                    + hpc_spatial_post_similarity_z
                    + (1 | subject)
                    + (1 | condition_item_id)

retrieval_pair_similarity ~ hpc_spatial_separation_z * condition
                          + pre_pair_similarity_z
                          + hpc_spatial_post_similarity_z
                          + (1 | subject)
                          + (1 | condition_item_id)
```

network-level mixed model 中，YY condition 仍显著预测更强 hpc-spatial separation：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic reactivation model | YY condition | +0.0493 | 2.78e-07 | 2.78e-07 |
| target-source model | YY condition | +0.0491 | 1.38e-07 | 1.38e-07 |
| corrected mapping model | YY condition | +0.0493 | 4.25e-07 | 4.25e-07 |

原始 shared-post 模型中，hpc-spatial separation 仍强烈预测 retrieval re-binding：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| hpc separation -> retrieval raw shared-post | hpc_spatial_separation_z | +0.2546 | 5.40e-254 | 5.40e-254 |

但这个结果不能再作为主 bridge，因为 `hpc_spatial_separation` 与 `retrieval_rebinding` 共享 post similarity 项。补充 post-control 后，结论明显降级：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| post-control rebinding | hpc_spatial_separation_z | +0.0093 | 0.0928 | 0.0928 |
| post-control rebinding | hpc_spatial_separation_z × YY | +0.0067 | 0.440 | 0.440 |
| retrieval similarity outcome | hpc_spatial_separation_z | +0.0093 | 0.736 | 0.736 |
| retrieval similarity outcome | hpc_spatial_separation_z × YY | +0.0067 | 0.440 | 0.440 |
| post-control rebinding | hpc_spatial_post_similarity_z | -0.1790 | 6.54e-220 | 6.54e-220 |

也就是说，difference-score outcome 中的剩余效应最多只是未校正趋势；当直接预测 retrieval pair similarity 时完全不显著。原始强 bridge 主要不应解释为心理/神经机制，而应视为 shared-post difference-score coupling 的敏感性警告。

但 semantic reactivation / corrected mapping 仍不能解释 hpc-spatial separation：

| model | term | beta | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic reactivation -> hpc separation | semantic_reactivation_z | -0.0012 | 0.861 | 0.861 |
| semantic reactivation -> hpc separation | semantic_reactivation_z × YY | +0.0025 | 0.795 | 0.795 |
| target-source -> hpc separation | semantic_target_source_over_self_z | -0.0087 | 0.171 | 0.171 |
| target-source -> hpc separation | semantic_target_source_over_self_z × YY | +0.0085 | 0.361 | 0.361 |
| corrected mapping -> hpc separation | corrected semantic asymmetry_z | +0.0019 | 0.768 | 0.768 |
| corrected mapping -> hpc separation | corrected semantic asymmetry_z × YY | -0.0056 | 0.564 | 0.564 |

这与前面的非 mixed model 结果更新后一致：最稳的结果不是“learning semantic reactivation -> post separation”，也不是严格意义上的“post separation -> retrieval rebinding”。更稳妥的阶段性链条是：

```text
YY condition
    -> hippocampal-spatial post separation
retrieval stage
    -> pair-structure rebound / YY-KJ decoding
```

### 20.7 本节结论

Mixed-effects 复核给出的结论很清楚：

1. **主结果更稳了**：YY 的 post-stage edge differentiation 在 meta_spatial ROI 中经 subject/item 层级控制后仍成立，尤其是 `R PPA/PHG` 与 `L PPC/SPL`。
2. **四阶段轨迹与主线一致**：learning 阶段不是 pre word-pair structure 的简单增强；post 阶段 YY 出现更强 pre-post differentiation。
3. **learning predictor 仍没有补上因果链**：reactivation 和 fluency 没有稳定预测 post edge；reactivation 还明显受 pre similarity 影响。
4. **directional mapping 仍是探索性**：corrected mapping 在 `L PPA/PHG` 与 `R PPC/SPL` 出现边界效应，但方向混合。
5. **network bridge 被降级**：raw model 中 hpc-spatial separation 非常稳定地预测 retrieval re-binding，但 post-control 与 retrieval-similarity outcome 后不稳健，因此不能作为主 bridge。
6. **memory logistic bridge 暂不成立**：当前 memory 编码不是二元变量，因此不能用 logistic mixed model 强行支持行为桥接。

因此，本轮 mixed-effects 模型最适合写成：

> 层级模型复核表明，YY 学习后的 post-stage relation-edge differentiation 并非 subject/item averaging 的产物，而是在 hippocampal-spatial / scene-context 系统中稳定存在。Retrieval 阶段仍显示 pair-structure rebound 与 YY/KJ decoding，但 post separation 是否进一步预测 retrieval-stage re-binding 在控制 post similarity 后不稳健。当前 learning-stage reactivation、fluency、corrected directional mapping 以及 post-to-retrieval bridge 都应保留为探索性候选，而不是主因果链。

## 21. Stagewise mechanism Stage 2：Learning cross-run geometry

### 21.1 分析目的与新版拆分

按照 `stagewise_mechanism_chain_spec.md` 的新版 Stage 2，本轮不再把旧的 row-only LET 指标当成 YY/KJ 主机制证据，而是把 Stage 2 拆成两层：

1. **Stage 2a：pooled item-specific trace**。直接从 learning beta 图像重算 run3 × run4 similarity matrix，并导出三个 item-specific 指标；YY/KJ 合并，只检验 learning item 是否整体存在 run3-to-run4 trace：

```text
row_specificity =
  Sim(run3_i, run4_i) - mean Sim(run3_i, run4_j, j != i)

column_specificity =
  Sim(run3_i, run4_i) - mean Sim(run3_j, run4_i, j != i)

bidirectional_specificity =
  Sim(run3_i, run4_i) - mean(row_other, column_other)
```

2. **Stage 2b：full cross-run matrix model**。保留完整 run3 × run4 matrix，不先扣掉 condition baseline，用同一个矩阵检验：

```text
cross_run_similarity ~ same_pair * condition
```

其中 `condition` 回答 YY/KJ 的整体 learning-stage cross-run geometry 是否不同；`same_pair` 回答是否存在 item-specific trace；`same_pair × condition` 回答 YY 是否有更强 item-specific trace。item-fixed sensitivity 使用 FWL residualization 先控制 row/column item fixed effects，再只解释 `same_pair` 与 `same_pair × condition`；不估计会被 item fixed effects 吸收的 condition 主效应。

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2a_learning_item_trace_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2a_learning_item_trace_network_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2a_learning_item_trace_one_sample.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2a_learning_item_trace_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2b_cross_run_matrix.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2b_cross_run_network_matrix.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2b_cross_run_matrix_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage2_learning_trace_review.md`

QC 通过：Stage 2a ROI item table 共 35,280 行，28 名被试、18 个 ROI，YY/KJ 各 17,640 行；2a 统计模型合并 YY/KJ，condition 只保留为数据列和 QC 标签。Stage 2b ROI matrix 共 1,234,800 行，network matrix 共 137,200 行。所有 Stage 2a pooled model、Stage 2b primary 与 item-fixed sensitivity 模型均为 `ok`，item-fixed focus terms 没有 NaN SE/p。

### 21.2 Stage 2a：item-specific trace

Stage 2a 的主结果只用于回答同一学习材料从 run3 到 run4 是否有 item-specific stability。因为 specificity 指标主动扣除了同条件其他 item 的平均相似性，它不适合作为 YY/KJ condition geometry 的主证据；因此新版 2a 把 YY/KJ 合并，只检验 pooled learning item trace。

Subject-level one-sample 检验中，pooled bidirectional specificity 的整体证据较弱。没有任何 ROI 通过 FDR，最接近的是左侧语义/隐喻 ROI：

| ROI set | ROI | bidirectional mean | p | q |
| --- | --- | ---: | ---: | ---: |
| meta_metaphor | meta_L_temporal_pole | +0.0163 | 0.0241 | 0.0963 |
| meta_metaphor | meta_L_pMTG/pSTS | +0.0135 | 0.0198 | 0.0963 |
| meta_spatial | meta_R_PPA/PHG | +0.00605 | 0.139 | 0.465 |
| meta_spatial | meta_L_PPA/PHG | +0.00916 | 0.0809 | 0.465 |
| meta_spatial | meta_R_RSC/PCC | +0.0111 | 0.118 | 0.465 |

因此，Stage 2a 不支持“learning 阶段存在稳定、广泛的 same-pair run3-run4 trace”。它最多提示少数 ROI 中有局部 item trace，但整体不强。

### 21.3 Stage 2a：pooled adjusted trace 与 post edge bridge

加入 `pre_pair_similarity_z` 与材料协变量后，pooled adjusted bidirectional specificity 仍没有任何 ROI 通过 FDR。最接近的仍是左颞极和左 pMTG/pSTS：

| ROI set | ROI | adjusted intercept | p | q |
| --- | --- | ---: | ---: | ---: |
| meta_metaphor | meta_L_temporal_pole | +0.0143 | 0.0390 | 0.156 |
| meta_metaphor | meta_L_pMTG/pSTS | +0.0144 | 0.0301 | 0.156 |
| meta_spatial | meta_R_RSC/PCC | +0.0122 | 0.115 | 0.620 |
| meta_spatial | meta_L_PPA/PHG | +0.00871 | 0.130 | 0.620 |
| meta_spatial | meta_R_PPA/PHG | +0.00645 | 0.231 | 0.620 |

进一步检验 pooled `post_edge_specificity ~ bidirectional_specificity_z` 时，也没有任何 ROI 通过 FDR。最接近的两个结果为：

| ROI set | ROI | term | beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_metaphor | meta_L_temporal_pole | bidirectional specificity | +0.0243 | 0.0114 | 0.0909 |
| meta_spatial | meta_L_hippocampus | bidirectional specificity | -0.0166 | 0.0104 | 0.104 |
| meta_spatial | meta_L_RSC/PCC | bidirectional specificity | +0.0116 | 0.222 | 0.444 |
| meta_spatial | meta_R_PPC/SPL | bidirectional specificity | -0.0105 | 0.208 | 0.444 |

因此，Stage 2a 没有把 learning-stage item trace 补成稳定的 post edge differentiation 上游 predictor。

### 21.4 Stage 2b：full run3 × run4 matrix

Stage 2b 是新版 Stage 2 的主检验，因为它保留完整 cross-run matrix，不把 condition-level shared geometry 先减掉。也就是说，2a 的 specificity 指标会主动扣掉同条件其他 item 的平均相似性，适合检验 item trace；但如果问题是“learning 阶段 YY/KJ 的整体几何是否不同”，就必须回到完整 matrix。

每个 subject × ROI × condition 都构建一个 run3 × run4 similarity matrix。矩阵的行是 run3 learning sentence item，列是 run4 learning sentence item；每个 cell 表示：

```text
cross_run_similarity_ij =
  Sim(run3_sentence_i, run4_sentence_j)
```

然后把所有 matrix cells 展开成长表，估计：

```text
cross_run_similarity ~ same_pair * condition
```

三个核心项分别回答不同问题：

| term | 含义 | 可以支持的结论 |
| --- | --- | --- |
| `condition` / `YY condition` | YY 相比 KJ 是否有整体更高的 run3-to-run4 cross-run similarity | learning 阶段存在 YY/KJ condition-level geometry |
| `same_pair` | 同一 item 的 run3-run4 similarity 是否高于不同 item | learning 阶段存在 item-specific trace |
| `same_pair × YY` | YY 的 same-pair trace 是否强于 KJ | YY 是否已经形成更强 pair-specific trace |

因此，`condition` 显著已经足以支持 learning-stage condition geometry；只有在要进一步声称“YY 在 learning 阶段已有更强 item-specific relation edge”时，才需要 `same_pair × YY` 显著。

Primary matrix model 的 ROI-level 关键结果如下：

| model | ROI set | ROI/network | term | beta | p | q |
| --- | --- | --- | --- | ---: | ---: | ---: |
| primary | meta_metaphor | meta_L_AG | YY condition | +0.0531 | 0.000257 | 0.00206 |
| primary | meta_spatial | meta_R_RSC/PCC | YY condition | +0.0315 | 0.00712 | 0.0712 |
| primary | meta_metaphor | meta_L_temporal_pole | same_pair | +0.0268 | 0.00309 | 0.0247 |
| primary | meta_spatial | meta_R_PPA/PHG | same_pair | +0.0138 | 0.0531 | 0.531 |

ROI-level 结果说明：`meta_L_AG` 中有很强的 YY condition effect，提示 learning sentence pattern 的 cross-run geometry 在 YY/KJ 之间不同；`meta_R_RSC/PCC` 有未过 FDR 的 spatial/context 趋势。same-pair trace 只有 `meta_L_temporal_pole` 通过 FDR，因此 item-specific trace 是局部的，不是全网络稳定现象。

更适合写主线的是预设 network-level 结果。这里把 8 个 `meta_metaphor` ROI 合成 semantic/metaphor network，把 10 个 `meta_spatial` ROI 合成 hpc-spatial/context network：

| network | term | beta | p | q | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| semantic | YY condition | +0.0131 | 0.00858 | 0.0172 | learning-stage YY/KJ condition geometry 显著 |
| hpc_spatial | YY condition | +0.00186 | 0.672 | 0.672 | learning 阶段无稳定 hpc-spatial condition effect |
| semantic | same_pair | +0.00647 | 0.0965 | 0.193 | item trace 仅为趋势 |
| hpc_spatial | same_pair | +0.00102 | 0.831 | 0.831 | 无 network-level item trace |

这个 network dissociation 很关键：learning 阶段的正证据主要来自 semantic/metaphor network，而不是 hpc-spatial network。换句话说，learning 阶段更像是在语义/隐喻网络中调动 YY/KJ condition-level geometry；hpc-spatial network 的强贡献主要出现在后续 post-stage relation-edge separation 和 retrieval-stage re-binding，而不是 sentence-level learning matrix 中。

`same_pair × YY` interaction 没有稳定证据。最接近的两个 metaphor ROI 都未过 FDR：

| model | ROI set | ROI | same_pair × YY beta | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| primary | meta_metaphor | meta_L_pMTG/pSTS | +0.0191 | 0.155 | 0.618 |
| primary | meta_metaphor | meta_L_temporal_pole | -0.0210 | 0.124 | 0.618 |
| primary | network | semantic | -0.00013 | 0.983 | 0.983 |
| primary | network | hpc_spatial | +0.00404 | 0.538 | 0.983 |

item-fixed sensitivity 得到几乎相同的解释：`same_pair` 在 `meta_L_temporal_pole` 中稳定为正，beta = +0.0268, p = 0.00309, q = 0.0247；semantic network 的 `same_pair` 仍只是趋势，beta = +0.00647, p = 0.0965, q = 0.193；`same_pair × YY` 仍完全不稳健。

item-fixed sensitivity 的作用是进一步控制 row item 和 column item 的固定差异，避免结果只反映某些 item 本身更容易产生高相似性。因为 row/column item fixed effects 会吸收 condition 主效应，所以 sensitivity 不解释 `condition`，只检查 `same_pair` 与 `same_pair × YY`。结果与 primary model 一致：有局部 same-pair trace，但没有 YY-specific same-pair enhancement。

### 21.5 Stage 2 结论

新版 Stage 2 给出的结论比旧 LET 版本更清楚：

1. Stage 2a 合并 YY/KJ 后，item-specific trace 仍较弱，没有任何 ROI 的 pooled bidirectional specificity 通过 FDR。
2. Stage 2a 的 bidirectional trace 没有稳定预测 post edge differentiation，因此 learning-to-post 的 item-level bridge 仍未补成。
3. Stage 2b full matrix 支持 learning 阶段存在 YY/KJ condition-level geometry，尤其在 `meta_L_AG` 与 semantic/metaphor network 中；hpc-spatial network 在 learning matrix 中不显著。
4. Stage 2b 对 same-pair item trace 的支持有限，主要集中在 `meta_L_temporal_pole`；没有证据显示 YY 有更强 same-pair trace。

更稳妥的写法是：

> Learning-stage patterns show YY/KJ condition-level cross-run geometry, especially in semantic/metaphor regions, but the stricter item-specific same-pair trace is weak and not reliably stronger for YY. Thus, Stage 2 supports learning-stage condition geometry or task-state geometry rather than fully formed pair-specific relation edges during learning.

这会让主线更清楚：learning 阶段提供的是 **condition geometry / task-state evidence**；真正稳定的 relation-edge reorganization 仍主要出现在 post 阶段，并由 Step5C / edge specificity 承担主机制证据。

## 22. Stagewise mechanism Stage 3：Semantic-to-edge model shift

### 22.1 分析目的与模型设定

Stage 3 检验一个更全局的机制问题：post 阶段的 neural RDM 是否从预存语义 embedding geometry 转向 learned-edge geometry。它不同于 Step5C / edge specificity 的 direct pair-similarity contrast；这里把完整 word-level neural RDM 同时放进多个模型解释框架中，问 edge model 的**独特解释量**是否从 pre 到 post 上升，并且是否相对 embedding model 上升。

本轮复用既有 meta ROI 的 model-RSA audit vectors，并额外生成 by-condition audit：

- pooled all-condition audit：`E:/python_metaphor/paper_outputs/qc/model_rdm_results_meta_metaphor/model_rdm_audit/`
- pooled all-condition audit：`E:/python_metaphor/paper_outputs/qc/model_rdm_results_meta_spatial/model_rdm_audit/`
- by-condition audit：`E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_model_rdm_by_condition_meta_metaphor/`
- by-condition audit：`E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_model_rdm_by_condition_meta_spatial/`

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_rdm_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_model_collinearity.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_shift.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_shift_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_condition_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage3_semantic_edge_shift_review.md`

All-condition pooled RDM 使用三模型回归：

```text
neural_RDM ~ M3_embedding + M8_reverse_pair + M1_condition
```

其中 `M3_embedding` 是预训练语义 embedding RDM；`M8_reverse_pair` 是 learned-edge / pair differentiation probe，同一 pair = 1、不同 pair = 0；`M1_condition` 控制 YY/KJ/baseline 条件类别结构。

YY/KJ condition-specific RDM 中，`M1_condition` 在单一 condition 内不可估，因此模型为：

```text
neural_RDM ~ M3_embedding + M8_reverse_pair
```

所有模型向量和 neural RDM 都做 rank transform，再用 reduced-model subtraction 计算 unique R2：

```text
unique_edge_r2 = R2(full model) - R2(model without edge)
unique_embedding_r2 = R2(full model) - R2(model without embedding)

semantic_to_edge_shift =
  (post_unique_edge_r2 - pre_unique_edge_r2)
  - (post_unique_embedding_r2 - pre_unique_embedding_r2)
```

正向 `semantic_to_edge_shift` 才能支持“post 阶段从 semantic embedding geometry 转向 edge geometry”。

QC：3024 个 subject × ROI × time × condition_group RDM cell 全部 `ok`，包括 `all / yy / kj` 三类 condition group；28 名被试、18 个 ROI。模型共线性检查没有发现高共线性 cell，`|rho| > 0.7` 的数量为 0；all-condition 中最大模型相关约为 condition-edge rho = 0.090，YY/KJ condition-specific 中 edge-embedding rho 约为 0.020。因此 Stage 3 的阴性结果不是由模型 RDM 高共线性造成的。

### 22.2 All-condition global RDM shift

先看 pooled all-condition RDM。Network-level 中没有任何 global semantic-to-edge shift 通过 FDR：

| network | metric | estimate | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic | semantic_to_edge_shift | +0.000100 | 0.399 | 0.679 |
| hpc_spatial | semantic_to_edge_shift | -0.000045 | 0.679 | 0.679 |
| semantic | delta_unique_edge_r2 | -0.0000129 | 0.422 | 0.422 |
| hpc_spatial | delta_unique_edge_r2 | -0.0000253 | 0.0822 | 0.164 |

ROI-level 也没有稳定正向 semantic-to-edge shift。最接近的是 `meta_R_AG`，但未过 FDR：

| ROI set | ROI | metric | estimate | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| meta_metaphor | meta_R_AG | semantic_to_edge_shift | +0.000464 | 0.0107 | 0.0856 |
| meta_spatial | meta_R_PPA/PHG | semantic_to_edge_shift | -0.000422 | 0.0522 | 0.340 |
| meta_spatial | meta_R_RSC/PCC | semantic_to_edge_shift | +0.000418 | 0.0868 | 0.340 |

因此，all-condition global RDM 没有支持“post 阶段整体从 embedding model 转向 edge model”。

### 22.3 YY/KJ condition-specific shift

Condition-specific 分析进一步问：semantic-to-edge shift 是否在 YY 中强于 KJ。结果也不支持。Network-level 的 `semantic_to_edge_shift` 是负向，semantic network 有未校正趋势但 FDR 后不显著：

| network | metric | YY-KJ estimate | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic | semantic_to_edge_shift | -0.00131 | 0.0391 | 0.0782 |
| hpc_spatial | semantic_to_edge_shift | -0.000455 | 0.452 | 0.452 |

更明显的是，`delta_unique_edge_r2` 的 YY-KJ 差异为负，而且在两个 network 都显著：

| network | metric | YY-KJ estimate | p | q |
| --- | --- | ---: | ---: | ---: |
| hpc_spatial | delta_unique_edge_r2 | -0.000666 | 2.65e-06 | 5.30e-06 |
| semantic | delta_unique_edge_r2 | -0.000602 | 1.22e-05 | 1.22e-05 |

这个方向意味着：edge model 的 unique fit 并不是 YY 比 KJ 增加更多，而是 KJ 更正向、YY 更负向。Network mean 显示：

| network | condition | delta unique edge R2 |
| --- | --- | ---: |
| semantic | KJ | +0.000291 |
| semantic | YY | -0.000311 |
| hpc_spatial | KJ | +0.000294 |
| hpc_spatial | YY | -0.000372 |

ROI-level 的 `delta_unique_edge_r2` 也大多呈负向 YY-KJ，其中多处通过 FDR：

| ROI set | ROI | YY-KJ delta unique edge R2 | p | q |
| --- | --- | ---: | ---: | ---: |
| meta_spatial | meta_R_hippocampus | -0.000879 | 5.62e-05 | 0.000562 |
| meta_spatial | meta_R_PPC/SPL | -0.000874 | 0.000545 | 0.00273 |
| meta_spatial | meta_L_hippocampus | -0.000890 | 0.00101 | 0.00336 |
| meta_spatial | meta_L_PPC/SPL | -0.000907 | 0.00215 | 0.00537 |
| meta_metaphor | meta_R_AG | -0.000853 | 0.000903 | 0.00722 |
| meta_metaphor | meta_R_IFG | -0.000834 | 0.00182 | 0.00727 |
| meta_metaphor | meta_L_temporal_pole | -0.000894 | 0.00302 | 0.00806 |

这说明 Stage 3 不仅没有支持 YY-specific semantic-to-edge shift，反而提示这个 global RDM edge-probe 在 KJ 中更容易出现 post-pre 增量。由于这与 Step5C / edge specificity 的 direct pair-similarity 主证据方向不同，最稳妥的解释是：**global model-RDM unique-fit shift 捕捉到的不是当前 YY learned-edge differentiation 的主机制层级**。

### 22.4 Stage 3 结论

Stage 3 是一个重要边界结果：

1. 模型共线性很低，因此阴性结果不是因为 embedding、condition、edge model 无法区分。
2. All-condition global RDM 没有稳定的 semantic-to-edge shift。
3. YY/KJ condition-specific RDM 也没有显示 YY 更强 semantic-to-edge shift；`semantic_to_edge_shift` 在 semantic network 中反而是负向趋势，未过 FDR。
4. `delta_unique_edge_r2` 的 YY-KJ 差异显著为负，说明 edge-model unique fit 的 post-pre 增量不是 YY 更强，而是 KJ 更强或 YY 更弱。

因此，Stage 3 不应升级为主机制证据。更稳妥的写法是：

> Model-RDM competition did not support a reliable YY-specific shift from embedding-like semantic geometry toward learned-edge geometry. The strongest evidence for YY post-learning reorganization therefore remains the direct edge-specific pair-similarity contrast rather than a global RDM model-weight shift.

这让主线更清楚：Stage 2 支持 learning-stage semantic condition geometry；Stage 3 说明这种语义状态并没有在 global RDM model space 中直接转化为可检出的 YY-specific edge-model dominance；真正稳定的 post-stage 机制仍是 Step5C / edge specificity 所显示的 trained-edge differentiation。

## 23. Stagewise mechanism Stage 4：Subsequent-memory rework

### 23.1 目的和方法

Stage 4 把行为桥接改成经典 subsequent-memory 逻辑：不再把 memory 当作到处回归的因变量，而是问 later remembered item 是否在预定义 neural mechanism metric 上更高。模型方向为：

```text
neural_metric_z ~ memory_score * condition
                + material_covariates
                + (1 | subject)
                + (1 | condition_item_id)
```

这里的 `memory_score` 保留 0/0.5/1 连续分数；正向 slope 表示后来记得更好的 item 有更高 neural metric。分析只使用 network composite，不做 18 ROI 全扫。预定义指标为：

- `semantic_learning_edge_trace`：semantic network 的 Stage 2a bidirectional learning trace。
- `hpc_spatial_learning_edge_trace`：hpc-spatial network 的 Stage 2a bidirectional learning trace。
- `hpc_spatial_post_separation`：Stage 1 hpc-spatial post edge separation。
- `hpc_spatial_rebinding`：Stage 1 hpc-spatial retrieval re-binding。

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage4_subsequent_memory_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage4_subsequent_memory_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage4_subsequent_memory_sensitivity.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage4_subsequent_memory_review.md`

QC：长表共有 7840 行，覆盖 28 名被试、70 个 condition item、4 个预定义 network metric；YY/KJ item ID collision 为 0。主模型进入 26-27 名被试，取决于 metric 的有效 neural rows。所有 mixed models 均正常收敛，没有使用 OLS fallback。

### 23.2 主模型结果

主模型最清楚的结果出现在 `hpc_spatial_rebinding`：YY item 中，later memory 越好，retrieval re-binding 越强；而 KJ 中这个 slope 接近 0。因此 YY-KJ interaction 也显著：

| metric | term | estimate | p | q |
| --- | --- | ---: | ---: | ---: |
| hpc_spatial_rebinding | memory slope in YY | +0.209 | 0.00968 | 0.0387 |
| hpc_spatial_rebinding | YY-KJ slope difference | +0.220 | 0.00173 | 0.00690 |
| hpc_spatial_rebinding | memory slope in KJ | -0.0111 | 0.854 | 0.854 |

其他三个指标没有稳定 subsequent-memory 差异：

| metric | YY memory slope | p | q | YY-KJ slope difference | p | q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic_learning_edge_trace | +0.108 | 0.238 | 0.391 | +0.0477 | 0.703 | 0.821 |
| hpc_spatial_learning_edge_trace | -0.0080 | 0.930 | 0.930 | -0.118 | 0.340 | 0.680 |
| hpc_spatial_post_separation | +0.0785 | 0.293 | 0.391 | +0.0245 | 0.821 | 0.821 |

因此，Stage 4 不支持“learning trace 或 post separation 本身已经区分 later remembered vs forgotten items”。在 Stage 4 单独看 difference score 时，行为桥接主要出现在 hpc-spatial re-binding；但这一点必须经过 Stage 4.5 difference-score 分解后再解释。

### 23.3 Sensitivity

两个 sensitivity 与主结果方向一致，但强度下降：

1. `memory_successes` 把 0/0.5/1 转成 0/1/2 remembered trials。`hpc_spatial_rebinding` 的 YY slope 为 +0.104，p = 0.0229，q = 0.0918；YY-KJ slope difference 为 +0.110，p = 0.0751，q = 0.301。
2. `remembered_strict` 排除 0.5 item，只比较 0 vs 1。`hpc_spatial_rebinding` 的 YY slope 为 +0.190，p = 0.0535，q = 0.214；YY-KJ slope difference 为 +0.221，p = 0.0871，q = 0.244。

这个 sensitivity pattern 说明主模型结果不是完全由方向错误造成，但也提示 item-level memory 桥接并不算强：一旦丢弃 0.5 item 或改成 success-count 编码，FDR 稳定性下降。

### 23.4 Stage 4 结论

Stage 4 初步支持一个中等强度、位置更靠后的行为桥接：

> Later remembered YY items show stronger hpc-spatial retrieval re-binding, whereas learning trace and post edge separation do not show reliable subsequent-memory differences.

Stage 4.5 复核后，这句话需要降级：`hpc_spatial_rebinding = retrieval_pair_similarity - post_pair_similarity`，其 memory effect 主要来自 post similarity 分量。因此，当前机制链不宜写成完整的 `learning edge trace -> post separation -> retrieval re-binding -> memory`。更稳妥的写法是：learning 阶段存在 condition-level geometry，post 阶段存在 trained-edge differentiation，而 item-level behavior bridge 更接近“remembered YY items already show stronger post-stage separation / lower post similarity”的边界证据。

## 24. Stagewise mechanism Stage 4.5：Accuracy audits before integration

### 24.1 目的

Stage 4.5 是进入最终整合前的 accuracy audit，专门检查三个最容易误读的位置：

1. Stage 4 的 `hpc_spatial_rebinding` difference score 到底由 retrieval similarity 还是 post similarity 驱动。
2. Stage 2b 的 condition effect 是否只是 beta norm / pattern variance / pattern quality 差异。
3. Stage 3 的 edge model 方向和 unique R2 解释是否有编码误读。

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage45_memory_difference_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage45_stage2_condition_quality_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage45_stage3_edge_direction_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage45_accuracy_audits_manifest.json`

QC：Stage 4.5 生成 5880 行 memory component rows、300 行 pattern-quality descriptive model rows、520 行 Stage2b quality-sensitivity model rows、3024 行 Stage3 direction-audit rows。Stage2 pattern-quality QC 覆盖 28 名被试、18 个 ROI、70560 个 ROI-level quality rows；没有 failed QC。

### 24.2 Audit 1：Stage 4 difference-score 分解

Stage 4 的关键指标是：

```text
hpc_spatial_rebinding = retrieval_pair_similarity - post_pair_similarity
```

分解后发现，YY 的 memory effect 不是主要由 retrieval similarity 更高驱动，而是由 post similarity 更低驱动：

| component | YY memory slope | p | q | YY-KJ slope difference | p | q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hpc_spatial_rebinding difference | +0.209 | 0.00968 | 0.0182 | +0.220 | 0.00173 | 0.00259 |
| hpc_spatial_post_pair_similarity | -0.169 | 0.0121 | 0.0182 | -0.293 | 0.000775 | 0.00232 |
| hpc_spatial_retrieval_pair_similarity | +0.105 | 0.254 | 0.254 | -0.0175 | 0.888 | 0.888 |

因此，Stage 4 不能强写成“remembered YY items show stronger retrieval-stage mnemonic reinstatement”。更准确的解释是：

> Remembered YY items show a stronger hpc-spatial rebinding difference score, but this is mainly because they already have lower post-stage pair similarity / stronger prior post separation, not because retrieval pair similarity itself is reliably higher.

这对主线很重要：behavior bridge 更接近 post-stage separation 的后续记忆相关，而不是 retrieval endpoint 的纯 re-binding 证据。

### 24.3 Audit 2：Stage 2b pattern-quality QC

Stage2b 的 concern 是：cross-run similarity 的 YY/KJ condition effect 可能来自整体 beta amplitude 或 pattern quality。复核中计算了：

- pattern norm
- pattern mean
- voxelwise SD / variance
- within-run mean similarity to other items in the same condition/run
- valid voxel count

严格 item-level split-half reliability 不能从当前 learning beta 表中估计，因为每个 item 每个 run 只有一个 beta。因此 `within_run_mean_similarity` 只能作为 run 内几何/quality proxy，而不是严格 reliability。

Network-level YY/KJ quality difference 中，amplitude 类指标没有明显差异：

| network | quality metric | YY-KJ estimate | p | q |
| --- | --- | ---: | ---: | ---: |
| semantic | pattern_norm | -0.0219 | 0.159 | 0.159 |
| semantic | pattern_sd | -0.0217 | 0.160 | 0.160 |
| semantic | pattern_variance | -0.0158 | 0.207 | 0.207 |
| hpc_spatial | pattern_norm | -0.0244 | 0.142 | 0.159 |
| hpc_spatial | pattern_sd | -0.0220 | 0.149 | 0.160 |
| hpc_spatial | pattern_variance | -0.0132 | 0.174 | 0.207 |

Stage2b sensitivity 显示，semantic network 的 condition effect 在 amplitude-only control 后仍保留：

| model | network | YY condition effect | p | q |
| --- | --- | ---: | ---: | ---: |
| amplitude-controlled | semantic | +0.0122 | 0.0149 | 0.0298 |
| amplitude-controlled | hpc_spatial | +0.00169 | 0.703 | 0.703 |

但是，当进一步加入 `within_run_mean_similarity` 这个 run 内几何 proxy 后，semantic effect 衰减为趋势：

| model | network | YY condition effect | p | q |
| --- | --- | ---: | ---: | ---: |
| full quality-controlled | semantic | +0.00852 | 0.0629 | 0.126 |
| full quality-controlled | hpc_spatial | -0.000908 | 0.816 | 0.816 |

ROI-level 中，`meta_L_AG` 在 amplitude-controlled 模型下仍显著：

| ROI | YY condition effect | p | q |
| --- | ---: | ---: | ---: |
| meta_L_AG | +0.0526 | 0.000389 | 0.00311 |

解释：Stage2b semantic condition geometry 不是简单 beta norm 或 voxel variance artifact；但它与更广义的 within-run geometry/quality proxy 有重叠。因此 Stage2b 可以保留为 learning-stage semantic condition geometry，但不应写成强 item-specific learning edge trace，也不应声称完全独立于 run 内几何结构。

### 24.4 Audit 3：Stage 3 edge-model direction audit

Stage3 的方向编码没有发现错误：

- neural RDM 是 correlation distance，即 `1 - r`，数值越大代表 neural patterns 越不相似。
- `M8_reverse_pair` 编码为 same pair = 1, different pair = 0。
- 因此正向 neural-edge association 表示 same-pair words 的 neural distance 更大，即 stronger differentiation。
- `unique_edge_r2 = full R2 - reduced R2` 是 explained-variance increment，本身没有方向，不能单独解释为 same-pair 更近或更远。

3024 个 audit rows 全部 `ok`。平均方向检查显示 same-pair neural distance 并没有更大，反而略低：

| network | condition group | n cells | same-minus-other neural distance | mean neural-edge rho |
| --- | --- | ---: | ---: | ---: |
| semantic | YY | 448 | -0.0221 | -0.00762 |
| semantic | KJ | 448 | -0.0391 | -0.0125 |
| hpc_spatial | YY | 560 | -0.0184 | -0.00731 |
| hpc_spatial | KJ | 560 | -0.0333 | -0.0121 |

因此 Stage 3 不需要重跑。它应继续作为 boundary result：global model-RDM competition 没有支持 YY-specific semantic-to-edge shift；这不是 coding direction error，而是该模型层级没有捕捉到当前 YY learned-edge differentiation 主机制。

### 24.5 Stage 4.5 结论

Stage 4.5 使主线进一步收束：

1. Stage4 行为桥接不能写成纯 retrieval re-binding；difference-score effect 主要来自 remembered YY items 的 lower post similarity / stronger prior post separation。
2. Stage2b semantic condition effect 不是简单 amplitude/variance artifact，但在控制 within-run geometry proxy 后变弱，因此应写成 learning-stage condition geometry，而不是 item-specific edge formation。
3. Stage3 阴性/反向结果不是编码错误；它是一个真实边界，说明 global RDM model competition 不适合作为主机制证据。

更新后的主线应是：

> YY learning first engages semantic condition-level geometry; post learning then shows trained-edge differentiation, especially in hpc-spatial / scene-context systems. Later remembered YY items are associated with stronger prior post-stage separation, while learning item trace, global semantic-to-edge RDM shift, and strong post-to-retrieval-to-memory path evidence remain boundary results.

## 25. Stagewise mechanism Stage 5 / 5.5: Final integration and exploratory readiness gate

### 25.1 Stage 5 目的

Stage 5 不再新增强路径模型，而是把 Stage 1-4.5 的结果收束成最终主线、模型图和边界条件表。输出文件为：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage5_evidence_integration_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage5_boundary_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage5_story_model.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage5_storyline_review.md`

Stage 5 evidence integration 生成 6 行核心证据：

| stage | best metric | conclusion | key estimate | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| Behavior | YY-KJ memory accuracy | YY memory advantage is stable | YY-KJ accuracy = +0.0921 | 0.000261 |  |
| Learning | Stage2b semantic condition geometry | semantic/metaphor network 中存在 YY/KJ condition-level geometry | semantic network beta = +0.0131 | 0.00858 | 0.0172 |
| Post | Step5C / hpc-spatial post edge specificity | YY trained relation edges show specific differentiation | hpc-spatial YY beta = +0.0493 | 2.78e-07 | 2.78e-07 |
| Retrieval | run7 pair-structure rebound + MVPA decoding | retrieval 阶段 pair structure / YY-KJ decoding 重新出现 | R hippocampus pair similarity = +0.111; MVPA acc = 0.565 | 7.02e-09 | 3.47e-07 |
| Memory | Stage4/4.5 rebinding difference decomposition | remembered YY items 有更大的 rebinding difference score，但主要由 lower post similarity / stronger prior separation 驱动 | rebinding slope = +0.209; post component = -0.169; retrieval component = +0.105 | 0.00968 | 0.0182 |
| Boundary | Stage3 / reactivation / mapping / post-control bridge | global semantic-to-edge shift、reactivation 上游因果、source-to-target mapping、强 post->retrieval->memory path 均不支持升为主线 | Stage3 semantic shift = -0.00131 | 0.0391 | 0.0782 |

### 25.2 Stage 5 最终主线

最终主线应写成阶段性表征链，而不是强因果桥接链：

> The present results do not support a strong full chain from item-specific learning trace to post separation to retrieval re-binding to memory. A more defensible account is that metaphor learning first engages YY/KJ condition-level semantic geometry during learning; after learning, YY trained relation edges differentiate in hpc-spatial / scene-context systems; during retrieval, pair-level structure reappears as task-driven rebound and YY/KJ decoding. Behavioral memory is robustly better for YY, but the item-level memory bridge is best interpreted as remembered YY items showing stronger prior post-stage separation, not reliably stronger retrieval similarity itself.

因此，Stage 5 后的写作规则是：

1. 可以保留：learning-stage condition geometry、post-stage trained-edge differentiation、retrieval-stage rebound / decoding。
2. post edge specificity / hpc-spatial separation 是当前最强机制层。
3. memory bridge 只能写成 endpoint boundary evidence，不能写成 pure retrieval reinstatement。
4. semantic reactivation、directional mapping、global semantic-to-edge shift、post->retrieval->memory path 都只能作为 boundary / exploratory，不进入主机制裁决。

### 25.3 Stage 5.5 目的

Stage 5.5 是 Stage 6/7 的轻量 precheck，不运行 Stage 6/7 主分析，只回答 matching、role coding 和 baseline asymmetry 是否足以允许后续探索性分析。输出文件为：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage6_control_balance.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage6_matching_manifest.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage6_precheck_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage7_role_coding_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage7_baseline_asymmetry.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_stage7_precheck_review.md`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage55_exploratory_readiness_review.md`

Stage 5.5 overall status 为 **pass_with_limitations**。

| check | status | detail |
| --- | --- | --- |
| stage6_matching_rules | pass | 140 true/control role-level matches built |
| stage6_balance | review | core lexical/material variables balanced, but pre_word_stability unavailable |
| stage7_role_coding | pass | 70/70 role rows ok |
| stage7_baseline_asymmetry | pass | baseline asymmetry = 0 under symmetric Fisher-z correlation |

Stage 6 matched-control balance 表显示，已有可用材料变量没有明显不平衡：

| variable | true mean | control mean | difference | p | status |
| --- | ---: | ---: | ---: | ---: | --- |
| role_char_len | 2.193 | 2.179 | +0.0143 | 0.319 | ok |
| word_frequency_mean | 3.845 | 3.871 | -0.0267 | 0.413 | ok |
| stroke_count_mean | 18.321 | 18.136 | +0.186 | 0.373 | ok |
| valence_mean | 3.105 | 3.116 | -0.0110 | 0.204 | ok |
| arousal_mean | 3.099 | 3.094 | +0.00477 | 0.406 | ok |
| pre_pair_similarity | 0.157 | 0.161 | -0.00356 | 0.457 | ok |
| pre_word_stability |  |  |  |  | unavailable |

`pre_word_stability` 被标为 unavailable，是因为当前 word-stability 输出是 condition/ROI summary，不是可直接用于 true-control matching 的 word-level covariate。因此它不是 matching fail，但必须作为 Stage 6 的 readiness limitation 明确写出。

### 25.4 是否可以执行 Stage 6 和 Stage 7

结论：**可以执行，但只能按 exploratory / boundary 执行，不能改变 Stage 5 主线。**

Stage 6 可以执行的条件：

1. 只能作为 exploratory matched-control constituent ERS。
2. 必须在模型和结果解释中报告 Stage5.5 matching limitation，尤其是 `pre_word_stability` unavailable。
3. 如果 Stage 6 只得到 `ERS > 0`，只能解释为 general constituent reinstatement，不能写成 metaphor-specific mechanism。
4. 只有在 `YY > KJ` 且进一步能预测 post separation 时，才可作为 exploratory upstream candidate；即使阳性，也不能升为主线证据。

Stage 7 可以执行的条件：

1. 只能作为 directional-mapping boundary analysis。
2. role coding 已通过：YY 的 RoleA/RoleB 锁定为 target/topic 与 source/vehicle，KJ 的 RoleA/RoleB 锁定为 object 与 location/context。
3. baseline asymmetry 已通过：所有 ROI/condition summary 的 baseline asymmetry 均为 0。
4. 只允许使用 corrected mapping metrics；结果无论阳性阴性，都不能写成主机制，除非方向稳定、理论一致且跨 meta_metaphor ROI robust。

因此下一步 gate 决策是：

> Proceed to Stage 6 and Stage 7 only as exploratory/supplementary analyses. Do not use either stage to revise the primary storyline unless they pass much stricter downstream criteria, especially YY-specificity and prediction of post-stage separation for Stage 6, and stable theory-aligned source-to-target direction across meta_metaphor ROIs for Stage 7.

## 26. Stagewise mechanism Stage 6 / 7: Exploratory boundary analyses

### 26.1 Stage 6: Matched-control constituent ERS

Stage 6 在 Stage5.5 `pass_with_limitations` gate 之后执行，只作为 exploratory matched-control ERS。指标定义为：

```text
constituent_ERS =
  Sim(learning_sentence, true_pre_constituent_word)
  - Sim(learning_sentence, matched_control_pre_word)
```

匹配控制来自 Stage5.5 的 `stage55_stage6_matching_manifest.tsv`，要求 same condition、same role position、not paired constituent、not same learned sentence，并尽量平衡词频、笔画/长度、valence、arousal、pre_pair_similarity。`pre_word_stability` 仍不可用，因此 Stage 6 结果必须带这个 limitation。

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage6_matched_control_ers_role_run.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage6_matched_control_ers_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage6_matched_control_ers_group_tests.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage6_matched_control_ers_to_post_models.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage6_matched_control_ers_review.md`

QC：Stage 6 生成 141120 行 role-run ERS、35280 行 item-level ERS；QC ok rate = 1.0。所有 subject × ROI × condition 单元均成功生成，每个单元 140 行 role-run rows。

主结果是一个很局部的 true-minus-control ERS 信号：

| test | ROI set | ROI | condition / contrast | mean effect | p | q |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ERS > 0 | meta_metaphor | meta_L_AG | YY | +0.0210 | 0.00181 | 0.0289 |
| YY - KJ ERS | meta_metaphor | meta_L_AG | YY-KJ | +0.0216 | 0.0145 | 0.116 |
| ERS > 0 | meta_metaphor | meta_L_IFG | KJ | -0.0122 | 0.0155 | 0.124 |
| YY - KJ ERS | meta_metaphor | meta_L_pMTG/pSTS | YY-KJ | +0.0156 | 0.101 | 0.300 |

因此，Stage 6 支持的最多是 **meta_L_AG 中 YY learning sentence 对 true constituent word 有轻度 matched-control reinstatement**。但这个结果没有形成稳健的 YY>KJ 条件特异性：meta_L_AG 的 YY-KJ contrast 未通过 FDR，`q = 0.116`；meta_spatial 中也没有稳定的 YY-specific ERS。

ERS -> post edge specificity 模型显示一个空间网络中的非条件特异趋势：

| model term | ROI set | ROI | estimate | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| ERS main effect | meta_spatial | meta_L_precuneus | +0.0407 | 0.00237 | 0.0236 |
| ERS main effect | meta_spatial | meta_R_RSC/PCC | +0.0354 | 0.0197 | 0.0987 |
| ERS × YY | meta_spatial | meta_L_RSC/PCC | -0.0311 | 0.0270 | 0.155 |
| ERS × YY | meta_spatial | meta_L_PPA/PHG | +0.0304 | 0.0311 | 0.155 |
| ERS × YY | meta_metaphor | meta_R_AG | +0.0262 | 0.0408 | 0.326 |

只有 `meta_L_precuneus` 的 ERS main effect 通过 FDR；YY-specific interaction 没有通过 FDR。因此它不能作为“YY-specific constituent reactivation drives post edge differentiation”的证据。

Stage 6 结论：

> A stricter matched-control ERS analysis provides limited evidence for general constituent reinstatement in left AG for YY items, but it does not show robust YY-specific constituent reactivation and does not provide a YY-specific upstream predictor of post-stage edge differentiation. Stage 6 therefore remains an exploratory boundary result and does not revise the Stage 5 storyline.

### 26.2 Stage 7: Corrected directional-mapping boundary

Stage 7 在 Stage5.5 role coding 与 baseline asymmetry precheck 通过后执行，只作为 directional-mapping boundary。使用 corrected metrics：

```text
target_source_over_self =
  Sim(PostTarget, PreSource) - Sim(PostTarget, PreTarget)

source_target_over_self =
  Sim(PostSource, PreTarget) - Sim(PostSource, PreSource)

mapping_asymmetry_self_corrected =
  target_source_over_self - source_target_over_self
```

输出文件：

- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage7_directional_mapping_boundary_item.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage7_directional_mapping_primary_tests.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage7_directional_mapping_readiness.tsv`
- `E:/python_metaphor/paper_outputs/qc/stagewise_mechanism/stage7_directional_mapping_boundary_review.md`

QC：role coding 70/70 rows ok；baseline asymmetry max abs = 0。Stage 7 item 表包含 35280 行，primary tests 包含 54 行。

Primary scope 固定为 `meta_metaphor`。三个 primary exploratory metrics 均未显示稳定 directional mapping：

| primary metric | best ROI in meta_metaphor | mean effect | p | q |
| --- | --- | ---: | ---: | ---: |
| target_source_over_self in YY | meta_R_IFG | +0.0232 | 0.191 | 0.984 |
| mapping_asymmetry_self_corrected in YY | meta_R_IFG | +0.0261 | 0.319 | 0.902 |
| YY-KJ mapping_asymmetry_self_corrected | meta_R_pMTG/pSTS | -0.0369 | 0.344 | 0.922 |

方向上也不稳定：部分 ROI 为正，部分 ROI 为负，而且没有任何 meta_metaphor ROI 在 primary metric scope 内通过 FDR。secondary `meta_spatial` 中也没有可解释为 metaphor source-to-target assimilation 的稳定结果。

Stage 7 结论：

> Corrected directional-mapping analyses do not provide evidence for stable source-to-target assimilation in the metaphor network. The current data are better described by post-stage trained-edge differentiation than by a uniform target-toward-source representational shift.

### 26.3 Stage 6/7 总结

Stage 6/7 的整体状态为：

| stage | decision | interpretation |
| --- | --- | --- |
| Stage 6 | mixed_exploratory_boundary | limited general ERS in left AG; no robust YY-specific ERS; no YY-specific post-edge prediction |
| Stage 7 | no_stable_directional_mapping | role/baseline QC passed, but corrected directional mapping is null/mixed |

因此，Stage 5 主线不改变：

> learning-stage condition geometry -> post-stage trained-edge differentiation -> retrieval-stage rebound / decoding.

Stage 6/7 只能放在 supplementary / boundary 部分，用来说明更严格的 constituent reactivation 和 source-to-target mapping 分析没有补出一条新的强上游机制链。

## 27. Reviewer-requested supplementary checks

本节汇总 `reviewer_supp` 中按审稿人问题补充的 sandbox 分析。它们不修改 Section 1-26 的主线，也不把新的探索性路径升级为主机制；作用是检查若干常见替代解释，并明确哪些结果只能作为 boundary / supplementary evidence。

输出目录为：

- `E:/python_metaphor/paper_outputs/qc/reviewer_supp/`

### 27.1 Scope and decision

这些补充分析的整体结论是：**当前主线仍应保持为 learning-stage condition geometry -> post-stage trained-edge differentiation -> retrieval-stage task-driven rebound / decoding**。Reviewer-supp analyses 没有补出一条比 Stage 5 更强的新机制链，但它们提高了当前解释的防守性：

1. post edge / trained-edge RSA 结果不太可能只是简单 ROI mean activation 或 novelty/repetition amplitude 差异。
2. learning 阶段有强 univariate subsequent-memory Dm，但没有稳定 learning-stage RSA Dm。
3. learning-to-retrieval ERS 没有形成广泛、稳健的 same-pair reinstatement 或 memory moderation。
4. strict binary memory 下仍能看到 post separation / hpc-spatial trace 的正向记忆相关，但 lenient coding 下减弱，因此二值化结果只能作为 sensitivity check。
5. KJ 应被写成 active spatial-relational learning condition，而不是 pure null control。

### 27.2 M2 / M6: univariate and amplitude controls

M2 检查 ROI mean activation 是否解释 Step5C / edge-specificity 主结果。核心输出是：

- `m2_univariate_sanity/step5c_univariate_controlled.tsv`
- `m2_univariate_sanity/roi_mean_activation_models.tsv`

Step5C univariate-controlled models 共 252 行，全部 `ok`。加入 `roi_mean_activation_z` 和 `pre_pair_similarity_z` 后，YY condition effect 仍在大量 post-edge / trained-edge 模型中保留：condition terms 中 44/54 行通过 FDR，其中 `post_edge_specificity` 有 15 个 ROI 显著，`trained_edge_drop` 有 16 个 ROI 显著。`roi_mean_activation_z` 本身主要影响 `pseudo_edge_drop`，说明平均激活并非完全无关，但它不能解释核心 YY trained-edge differentiation。

M6 检查 pre/post isolated-word 的 novelty/repetition 单变量幅度。核心输出是：

- `m6_novelty_repetition/post_minus_pre_condition_model.tsv`

YY-vs-KJ condition difference 没有任何 ROI 通过 FDR。由于不少 mixed models 出现 random-effect singularity，脚本保留 mixed-preferred 结果并使用 HC3 nonmixed fallback；这些 fallback 只用于 sanity-check 层级解释。

因此，M2/M6 的合适写法是：

> The post-learning edge-differentiation results are not reducible to simple ROI mean activation or pre/post novelty-repetition amplitude differences, although these controls do not rule out all possible pattern-quality, SNR, or variance-related influences.

### 27.3 M3 / M4: learning Dm and ERS boundary checks

M3 检查 learning-stage subsequent-memory Dm。核心输出是：

- `m3_learning_dm/learning_dm_univariate.tsv`
- `m3_learning_dm/learning_dm_rsa.tsv`

结果显示 learning-stage univariate Dm 很强：univariate remembered / remembered × condition 项 36/36 行通过 FDR。相反，learning-stage RSA Dm 没有稳定证据：RSA remembered / interaction 项 0/36 行通过 FDR。也就是说，later memory 信息在学习期平均激活中很明显，但没有形成稳定的 learning-phase representational geometry Dm。

M4 检查 learning-to-retrieval same-vs-other ERS 及其 memory moderation。核心输出是：

- `m4_correct_ers/ers_same_vs_other.tsv`
- `m4_correct_ers/ers_dm_interaction.tsv`

same-vs-other ERS 只有 1/36 行通过 FDR，且唯一显著项为 `meta_metaphor / meta_R_IFG / YY` 的负向 same-vs-other effect。ERS memory moderation 为 0/36 行通过 FDR。

因此，M3/M4 的合适写法是：

> Learning-stage subsequent-memory information is visible in univariate activation, but not in robust learning-stage RSA geometry. Learning-to-retrieval ERS does not provide a broad same-pair reinstatement mechanism and does not reliably explain later memory. These analyses therefore remain exploratory boundary checks rather than a replacement for the post-stage edge-differentiation account.

### 27.4 M9: strict/lenient binary-memory sensitivity

M9 检查连续 memory-score 结果在 strict / lenient 二值化下是否脆弱。核心输出是：

- `m9_logistic_dm/logistic_dm_input_item.tsv`
- `m9_logistic_dm/logistic_dm_strict.tsv`
- `m9_logistic_dm/logistic_dm_lenient.tsv`

输入表包含 1890 行，`condition_item_id` 已生成且无 YY/KJ item-ID collision。原始 memory 分布为 1.0 = 855、0.5 = 686、0.0 = 349；strict coding 只把 1.0 作为 remembered，lenient coding 把 0.5 和 1.0 都作为 remembered。

strict logistic Dm 中，三个 neural main predictors 通过 FDR，且方向均为正：

| metric | estimate | q |
| --- | ---: | ---: |
| `hpc_spatial_learning_edge_trace_z` | +0.146 | 0.0137 |
| `post_edge_specificity_z` | +0.141 | 0.0150 |
| `hpc_spatial_post_separation_z` | +0.141 | 0.0150 |

lenient logistic Dm 中，neural main predictors 没有通过 FDR。这个结果说明 neural-memory association 主要出现在更严格的完整记忆成功定义下；lenient coding 会稀释这些 neural predictors。

因此，M9 的合适写法是：

> Binary-memory sensitivity analyses partly support the continuous-memory interpretation under the strict remembered-only definition, but the effects weaken under lenient coding. M9 should therefore be treated as a sensitivity check rather than a replacement for the Stage 4/4.5 continuous memory-score model.

### 27.5 M11: KJ fairness and evidence tiering

M11 从预定义上游表中抓取 explicit KJ / spatial signatures，目的是避免把 KJ 简化为 pure null control。核心输出是：

- `m11_kj_fair_characterization/kj_significant_results.tsv`
- `m11_kj_fair_characterization/kj_specific_signature.md`

结果共找到 71 条 candidate KJ/spatial rows：

| evidence tier / source | n |
| --- | ---: |
| existing ERS, FDR-corrected | 23 |
| existing ERS, uncorrected-only | 5 |
| semantic edge model, FDR-corrected | 16 |
| semantic edge model, uncorrected-only | 2 |
| relation vector, uncorrected-only | 18 |
| retrieval rebinding, FDR-corrected but shared-post downgraded | 5 |
| retrieval rebinding, uncorrected-only and shared-post downgraded | 2 |

因此，KJ 的合适写法是：

> KJ should be treated as an active spatial-relational learning condition rather than a pure null baseline. FDR-corrected KJ signatures can be described as supplementary evidence for KJ-specific spatial/relational structure; uncorrected-only rows should remain descriptive, and retrieval-rebinding rows should be downgraded when they come from shared-post sources.

### 27.6 M5: network-level theoretical framing

M5 不做统计检验，只整理网络解释语言。核心输出是：

- `m5_theoretical_reframing/network_reframing.md`
- `m5_theoretical_reframing/network_reframing_table.tsv`

推荐术语是：

| network | preferred framing |
| --- | --- |
| hippocampus | pattern separation |
| PPA/PHG/RSC/PPC | scene-context binding reorganization |
| ATL/AG/IFG/semantic network | relational semantic restructuring |

这意味着全文不应把所有 ROI 的结果都写成单一的 similarity enhancement。更稳妥的表述是：YY learning produces relation-edge differentiation, with different network-level interpretations across hippocampal, spatial-context, and semantic-control systems.

### 27.7 Section 27 conclusion

Reviewer-supplementary analyses 支持当前的保守解释：主机制仍是 post-learning trained-edge differentiation，并在 retrieval 阶段以 task-driven rebound / decoding 的形式重新出现。M2/M6 排除了简单单变量替代解释，M3/M4 限定了 learning Dm 和 ERS 的解释范围，M9 说明 memory coding 的敏感性，M11 纠正了 KJ-as-null 的过度简化，M5 提供更精确的网络术语。

因此，Section 27 的结论不是扩张主线，而是给当前主线加上审稿层面的保护：

> The reviewer-requested supplementary checks rule out several simpler alternatives and clarify important boundary conditions, but they do not revise the primary Stage 5 account.

## 28. NC convergence after bug fix: 主线收束、稳健性验证与机制升级

本节整理 bug 修复后重新执行的 `nc_converge` 结果。所有新增输出均来自：

- `E:/python_metaphor/paper_outputs/qc/nc_converge/`

本轮重跑包括：

```text
A2 schema lock
A3 shared-post review
A4 post/retrieval component + stage trajectory
A5 network composite
B1 staged trajectory characterization
B2 crossnobis robustness
B3 network-level coupling
C1 input-output transformation
C2 global similarity / centrality
C3 novelty / familiarity moderation
A6 FDR tiering
Go/No-Go
Isolation check
```

核心判断仍是 **GO**，但机制升级部分比旧结果更保守：C1 支持 YY post-stage output similarity 更低，但 pre-to-post slope 交互不稳定；C2 支持 global-centrality 作为 Tier C refinement；C3 不再支持强 novelty moderation，semantic 模型还出现 mixed-model non-convergence，因此只能作为材料协变量边界检查。

### 28.1 Go/No-Go 总判断

核心输出：

- `go_no_go/go_no_go_stage6.md`
- `go_no_go/go_no_go_evidence.tsv`

最终判定为 **GO**。

| evidence block | status | interpretation |
| --- | --- | --- |
| A4 post/retrieval component + trajectory | stable | 支持 post-stage differentiation、retrieval rebound 与 memory bridge |
| B3 network-level coupling | stable | 支持 semantic/metaphor 与 hpc-spatial 的预定义 network coupling |
| B2 Mahalanobis robustness proxy | partial/supportive | 已补充 diagonal-shrinkage Mahalanobis proxy；YY network composite 方向与 correlation Step5C 一致，但不能称为严格 crossnobis |
| B1 staged trajectory characterization | unresolved | direct association 表缺少匹配 predictor/outcome，仅作为边界条件 |

Go/No-Go 的实际规则是：

```text
Primary GO rule:
at least two of A4/B2/B3 support the main direction.
```

本轮满足该规则的强证据仍然是 A4 与 B3；B2 现在提供 supportive robustness proxy，但由于不是严格 crossnobis，只能作为稳健性补充，不能写成 definitive crossnobis confirmation。

### 28.2 A2 schema lock：ID 结构可用

核心输出：

- `a2_schema_lock/id_audit.tsv`
- `a2_schema_lock/subject_intersection.tsv`
- `a2_schema_lock/schema_lock.log`

A2 确认 `condition_item_id` 可以作为后续 item-level 合并主键。少数重复来自 repeated-measure 长表，已经通过 `run / role / control_condition_item_id` 等额外维度解释，因此标记为 `ok_repeated_measure_key`，不再作为 fatal schema error。

结论：

> item-level 分析可以继续使用 `condition_item_id`，但不能把所有上游长表强行理解为 subject × roi × condition_item_id 的唯一行表。

### 28.3 A4-1：memory component model

核心输出：

- `a4_post_memory_component/memory_component_model.tsv`
- `a4_post_memory_component/memory_component_roi_followup.tsv`

A4-1 将 post similarity 与 retrieval similarity 分开进入 subsequent-memory model：

```text
memory ~ condition * post_pair_similarity_z
       + condition * retrieval_pair_similarity_z
       + covariates
       + subject / condition_item_id random terms
```

YY memory advantage 稳定存在：

| network | outcome | term | estimate | q |
| --- | --- | --- | ---: | ---: |
| hpc_spatial | memory_prop | `condition[T.yy]` | +0.106 | 2.20e-52 |
| semantic | memory_prop | `condition[T.yy]` | +0.107 | 9.10e-43 |
| hpc_spatial | memory_strict | `condition[T.yy]` | +0.565 | 2.00e-111 |
| semantic | memory_strict | `condition[T.yy]` | +0.570 | 8.84e-91 |

更关键的是，YY 的 memory bridge 主要表现为 post-stage separation：

| network | outcome | term | estimate | q |
| --- | --- | --- | ---: | ---: |
| hpc_spatial | memory_prop | `condition[T.yy]:post_pair_similarity_network_z` | -0.0177 | 4.78e-04 |
| semantic | memory_prop | `condition[T.yy]:post_pair_similarity_network_z` | -0.0261 | 3.41e-06 |
| hpc_spatial | memory_lenient | `condition[T.yy]:post_pair_similarity_network_z` | -0.290 | 1.71e-14 |
| semantic | memory_lenient | `condition[T.yy]:post_pair_similarity_network_z` | -0.358 | 1.93e-18 |

retrieval similarity 的结果更混合：

| network | outcome | term | estimate | q |
| --- | --- | --- | ---: | ---: |
| semantic | memory_strict | `condition[T.yy]:retrieval_pair_similarity_network_z` | +0.129 | 3.99e-06 |
| semantic | memory_prop | `condition[T.yy]:retrieval_pair_similarity_network_z` | +0.0131 | 0.0210 |
| hpc_spatial | memory_prop | `condition[T.yy]:retrieval_pair_similarity_network_z` | +0.0016 | 0.768 |

因此，最稳妥解释是：

> remembered YY items show stronger prior post-stage separation. Retrieval similarity contributes in some semantic-network memory codings, but the main behavioral bridge should not be written as pure retrieval reinstatement.

### 28.4 A4-2：stage trajectory model

核心输出：

- `a4_post_memory_component/stage_trajectory_model.tsv`
- `a4_post_memory_component/stage_trajectory_descriptives.tsv`

模型：

```text
pair_similarity ~ condition * stage
                + covariates
                + subject / condition_item_id random terms
```

关键统计：

| network | term | estimate | q |
| --- | --- | ---: | ---: |
| hpc_spatial | `condition[T.yy]` | -0.0580 | 5.39e-07 |
| hpc_spatial | `condition[T.yy]:C(stage)[T.pre]` | +0.0652 | 3.93e-06 |
| hpc_spatial | `condition[T.yy]:C(stage)[T.retrieval]` | +0.0482 | 6.18e-04 |
| semantic | `condition[T.yy]` | -0.0688 | 1.58e-08 |
| semantic | `condition[T.yy]:C(stage)[T.pre]` | +0.0680 | 4.64e-06 |
| semantic | `condition[T.yy]:C(stage)[T.retrieval]` | +0.0534 | 3.46e-04 |

描述性均值也支持同一方向：

| network | condition | pre mean | post mean | retrieval mean |
| --- | --- | ---: | ---: | ---: |
| hpc_spatial | KJ | 0.153 | 0.146 | 0.157 |
| hpc_spatial | YY | 0.169 | 0.096 | 0.155 |
| semantic | KJ | 0.147 | 0.153 | 0.161 |
| semantic | YY | 0.158 | 0.097 | 0.158 |

解释：

> YY shows a post-stage similarity reduction with pre/retrieval rebound-like offsets. This supports stagewise relation-edge differentiation rather than a monotonic similarity enhancement account.

### 28.5 A5 / B1 / B2：收束性与边界条件

A5 输出：

- `a5_network_composite/network_trajectory_long.tsv`
- `a5_network_composite/network_composite_summary.tsv`

A5 已修订为逐表识别 value column，再生成 network composite，避免异构表拼接后 `metric_value` 全空。当前 A5 主要作为 network-level 汇总和后续 B3 的输入辅助，不作为单独主证据。

B1 输出：

- `b1_staged_path/stage5_network_trajectory_table.tsv`
- `b1_staged_path/staged_trajectory.md`

B1 三条 staged direct association 均未找到可匹配 predictor/outcome 字段，状态为 `missing_predictor_or_outcome`。这不是反证，而是说明当前上游表还不能支持 direct cross-stage path。B1 只能写成边界条件，不能写成 causal path 或 mediation。

B2 输出：

- `b2a_mahalanobis_from_patterns/mahalanobis_pair_distance.tsv`
- `b2a_mahalanobis_from_patterns/mahalanobis_subject_delta.tsv`
- `b2a_mahalanobis_from_patterns/mahalanobis_step5c.tsv`
- `b2a_mahalanobis_from_patterns/mahalanobis_method_note.md`
- `b2_crossnobis_robustness/crossnobis_step5c.tsv`

B2 原本缺少 matched crossnobis / Mahalanobis 输入表；现在已补充基于 `pattern_root` 4D pattern 与 `meta_metaphor/meta_spatial` ROI masks 的 diagonal-shrinkage Mahalanobis robustness proxy。由于每个 lexical item 在单一 stage 内没有重复独立估计，严格 item-level crossnobis 仍不可计算，因此脚本明确标注：

```text
diagonal_shrinkage_mahalanobis_not_crossvalidated
```

覆盖率：

| output | rows | coverage |
| --- | ---: | --- |
| `mahalanobis_pair_distance.tsv` | 70,560 | 28 subjects × 18 ROI × 2 conditions × pre/post pair rows |
| `mahalanobis_subject_delta.tsv` | 39,200 | YY/KJ 各 35 个 condition items，含 ROI 与 network composite |
| `crossnobis_step5c.tsv` | 38 | 18 ROI × 2 conditions + 2 networks × 2 conditions |

关键 network composite 方向：

| network | condition | Mahalanobis proxy delta | correlation delta | status |
| --- | --- | ---: | ---: | --- |
| hpc_spatial | YY/Metaphor | -0.0138 | -0.0746 | ok |
| hpc_spatial | KJ/Spatial | +0.0030 | +0.0736 | ok |
| semantic | YY/Metaphor | -0.0102 | -0.0668 | ok |
| semantic | KJ/Spatial | +0.0075 | +0.0675 | ok |

因此可以写：

> A covariance-normalized Mahalanobis proxy reproduced the direction of the YY post-stage separation at the predefined semantic and hpc-spatial network-composite level.

但不能写：

```text
strict crossnobis robustness supports the main result
```

### 28.6 B3：network-level representational coupling

核心输出：

- `b3_repr_coupling/repr_coupling.tsv`

B3 只做预定义 network-level coupling，不做 ROI × ROI 扫描。两个 planned couplings 是：

```text
learning semantic specificity -> post hpc-spatial trained-edge drop
retrieval semantic rebound -> retrieval hpc-spatial rebound
```

结果：

| coupling | term | estimate | q | interpretation |
| --- | --- | ---: | ---: | --- |
| learning semantic -> post hpc-spatial | `condition[T.yy]` | +0.217 | 9.37e-06 | YY 条件下 hpc-spatial trained-edge drop 更强 |
| learning semantic -> post hpc-spatial | semantic learning specificity × YY | -0.0158 | 0.731 | item-level semantic learning specificity 交互不稳定 |
| retrieval semantic -> retrieval hpc-spatial | semantic retrieval rebound | +0.659 | 1.43e-146 | retrieval rebound 在两个 network 间强耦合 |
| retrieval semantic -> retrieval hpc-spatial | semantic retrieval rebound × YY | -0.0429 | 0.358 | YY-specific coupling 交互不稳定 |

写作建议：

> Retrieval-stage rebound is strongly coupled across semantic/metaphor and hpc-spatial networks. Learning-to-post coupling is better described as a YY condition-level hpc-spatial differentiation effect, not as a strong item-by-item semantic-to-hippocampal path.

### 28.7 C1：input-output transformation

核心输出：

- `c1_input_output_transformation/input_output_transformation.tsv`
- `c1_input_output_transformation/input_output_descriptives.tsv`

C1 检验：

```text
post_pair_similarity ~ pre_pair_similarity * condition
                     + covariates
                     + subject / condition_item_id random terms
```

bug 修复后的结果比旧版更保守：

| network | term | estimate | q | interpretation |
| --- | --- | ---: | ---: | --- |
| hpc_spatial | `condition[T.yy]` | -0.176 | 0.0229 | YY post output similarity 更低 |
| hpc_spatial | pre similarity | +0.061 | 0.163 | pre-to-post slope 不稳定 |
| hpc_spatial | pre similarity × YY | -0.009 | 0.945 | slope 交互不支持 |
| semantic | `condition[T.yy]` | -0.186 | 0.0229 | YY post output similarity 更低 |
| semantic | pre similarity | +0.060 | 0.163 | pre-to-post slope 不稳定 |
| semantic | pre similarity × YY | -0.063 | 0.284 | slope 交互不支持 |

因此 C1 现在应写成：

> YY shows lower post-stage output similarity after accounting for pre-stage input similarity, but the condition-specific pre-to-post slope change is not robust after the bug fix.

这仍支持 “post-stage differentiation”，但不能强写成 “YY 改变了输入到输出的转换斜率”。

### 28.8 C2：global similarity / representational centrality

核心输出：

- `c2_global_similarity_centrality/global_similarity_item.tsv`
- `c2_global_similarity_centrality/global_similarity_models.tsv`
- `c2_global_similarity_centrality/global_similarity_manifest.tsv`

C2 从 RDM audit npz 中计算 item global similarity / centrality。共读取 1008 个 RDM 文件，manifest 全部为 `ok`。

关键结果：

| network | model | term | estimate | q | interpretation |
| --- | --- | --- | ---: | ---: | --- |
| hpc_spatial | global change by memory | YY × memory | -0.195 | 0.682 | memory-specific global centrality effect 不稳定 |
| semantic | global change by memory | YY × memory | -0.333 | 0.361 | 同上，不稳定 |
| hpc_spatial | edge differentiation by global change | `condition[T.yy]` | +0.435 | 9.43e-04 | YY edge differentiation 在 global centrality 控制下仍保留 |
| hpc_spatial | edge differentiation by global change | global change × YY | -0.260 | 0.0234 | hpc-spatial 中 global change 与 YY edge differentiation 的关系不同 |
| semantic | edge differentiation by global change | `condition[T.yy]` | +0.267 | 0.0918 | semantic 中 condition effect 降为趋势 |
| semantic | edge differentiation by global change | global change × YY | -0.0346 | 0.837 | semantic 交互不稳定 |

C2 应作为 Tier C refinement：

> Global-centrality analyses suggest that hpc-spatial YY edge differentiation is not reducible to pair similarity alone, but memory-specific global centrality effects are not stable enough to serve as a main mechanism.

### 28.9 C3：novelty / familiarity moderation

核心输出：

- `c3_novelty_familiarity_moderation/novelty_familiarity_moderation.tsv`
- `c3_novelty_familiarity_moderation/novelty_familiarity_input_coverage.tsv`

coverage：

| network | rows | n subjects | n items | novelty/familiarity/comprehensibility/difficulty coverage |
| --- | ---: | ---: | ---: | ---: |
| hpc_spatial | 1960 | 28 | 70 | 1008 |
| semantic | 1960 | 28 | 70 | 1008 |

bug 修复后的 C3 不再支持强 novelty moderation：

| network | term | estimate | q / status | interpretation |
| --- | --- | ---: | --- | --- |
| hpc_spatial | `condition[T.yy]` | -0.094 | q = 0.590 | condition effect 不稳定 |
| hpc_spatial | novelty | +0.081 | q = 0.590 | novelty 主效应不稳定 |
| hpc_spatial | YY × novelty | -0.093 | q = 0.590 | novelty moderation 不稳定 |
| semantic | `condition[T.yy]` | -0.173 | `mixed_not_converged` | 不能解释 |
| semantic | novelty | +0.063 | `mixed_not_converged` | 不能解释 |
| semantic | YY × novelty | -0.045 | `mixed_not_converged` | 不能解释 |

因此 C3 的最新写法应明显降调：

> Novelty/familiarity moderation does not provide stable support after the bug fix. It should be reported as a material-covariate boundary check rather than as evidence that novelty drives or explains YY edge differentiation.

### 28.10 A6 FDR tiering

核心输出：

- `a6_fdr_tiering/fdr_tier_manifest.tsv`
- `a6_fdr_tiering/fdr_tier_summary.md`

A6 现在按 `(tier, module, network)` 分区，而不是把异质模型全部塞进同一个 family。当前 FDR families 为：

| tier | family | n tests |
| --- | --- | ---: |
| Tier A | `a4_post_memory_component::hpc_spatial` | 154 |
| Tier A | `a4_post_memory_component::semantic` | 132 |
| Tier A | `b3_repr_coupling::all` | 8 |
| Tier C | `c1_input_output_transformation::hpc_spatial` | 9 |
| Tier C | `c1_input_output_transformation::semantic` | 9 |
| Tier C | `c2_global_similarity_centrality::hpc_spatial` | 18 |
| Tier C | `c2_global_similarity_centrality::semantic` | 18 |
| Tier C | `c3_novelty_familiarity_moderation::hpc_spatial` | 12 |
| Tier C | `c3_novelty_familiarity_moderation::semantic` | 12 |

主文结论应主要依赖 Tier A，即 A4 与 B3；C1/C2/C3 只作为机制 refinement 和边界条件。

### 28.11 Phase 5 / Phase 6 决策

Phase 5 learning-update proxy 暂不执行。原因是当前 `behavior_results/refined/behavior_trials.tsv` 中没有 run3/run4 行为 RT 数据；已有 run3/run4 learning dynamics 主要是 neural RDM / relation-vector 指标，不是老师建议的 RT reduction / fluency proxy。强行替代会改变问题性质。

Phase 6 `hpc_subfield_three_axis` 已作为 exploratory extension 执行，结果见 §29。B2 已不再是 input missing，且 Mahalanobis proxy 在 network-composite 层面对 YY post-stage separation 给出方向一致的稳健性补充；但它仍不是严格 crossnobis。因此海马 head/body/tail 长轴分析最多进入：

```text
exploratory hippocampal long-axis extension
```

不能写成：

```text
definitive hippocampal subfield mechanism
CA3/DG mechanism
```

### 28.12 最新主线写法

当前最稳主线：

```text
Behavior:
YY memory advantage

Learning:
YY/KJ condition-level geometry emerges in semantic/metaphor network

Post:
YY trained relation edges show specific differentiation,
especially in hippocampal-spatial / scene-context network

Retrieval:
YY pair-structure rebounds and semantic/hpc-spatial rebound is coupled

Memory:
remembered YY items show stronger prior post-stage separation,
not pure retrieval reinstatement
```

一句话版本仍然可以保留：

> Metaphorical learning transforms pre-existing semantic geometry into differentiated, task-retrievable relation-edge representations.

中文：

> 隐喻学习将既有语义几何转化为分化的、可在提取阶段重新调动的关系边表征。

但需要避免以下过度表述：

- 不写完整因果链：`learning trace -> post separation -> retrieval rebinding -> memory`。
- 不把 C1 写成强 input-output slope transformation；bug 修复后 slope 交互不稳定。
- 不把 C3 写成 novelty moderation 支持；bug 修复后不稳定且 semantic 模型未收敛。
- 不把 B2 写成 strict crossnobis 支持；当前是 Mahalanobis robustness proxy。
- 不把 KJ 写成 null control；KJ 是 active spatial-relational learning comparison condition。
- 不把海马长轴写成 definitive subfield 或 CA3/DG mechanism。

本节结论：

> Bug-fixed NC convergence results support moving forward with a stagewise relation-edge reorganization account. The strongest upgraded evidence comes from A4 and B3, with B2 now adding a supportive Mahalanobis robustness proxy rather than a strict crossnobis confirmation. C1 and C2 provide useful Tier-C refinements, while C3 now functions mainly as a material-covariate boundary check. The paper should stop expanding analyses and shift to Results/Discussion consolidation.

## 29. HPC long-axis exploratory extension

本节整理 `hpc_subfield_three_axis` 的实际执行结果。所有输出位于：

- `E:/python_metaphor/paper_outputs/qc/hpc_subfield_three_axis/`

本轮新增或修正的执行层包括：

- `p0_inputs/hpc_items.tsv`
- `p0_inputs/hpc_pair_table.tsv`
- `s1_segmentation/subfield_manifest.tsv`
- `s2_beta_extract/beta_long.tsv`
- `s3_step5c_subfield/subfield_step5c.tsv`
- `s4_edge_specificity_subfield/subfield_edge_specificity.tsv`
- `s5_anterior_posterior_contrast/anterior_posterior_contrast.tsv`
- `v1_mds_visualisation/mds_coords.tsv`
- `n1_three_axis_evidence/three_axis_evidence_table.tsv`
- `append_section29/section29_preview.md`

### 29.1 执行与 QC

S1 没有检测到 FreeSurfer FS60 hipposubfield 分割，因此全部使用既有 `meta_L/R_hippocampus` mask 的 MNI y 轴三等分：

| provenance | qc_status | n masks |
| --- | --- | ---: |
| mni_split | ok | 112 |
| mni_split | qc_warn | 56 |

因此，本节必须写成：

```text
exploratory MNI long-axis segmentation
```

不能写成：

```text
FreeSurfer subfield mechanism
CA3/DG mechanism
definitive hippocampal subfield evidence
```

S2 成功提取 subROI beta vectors：

| output | value |
| --- | ---: |
| `beta_qc.tsv` rows | 168 |
| extracted vectors in `beta_long.tsv` | 68,880 |
| subROI | 6 head/body/tail masks across L/R |
| subjects | 28 |

V1 MDS 也已完成，5 个 panel 均有 pre/post 坐标与 1200 dpi 图像：

```text
meta_R_hippocampus
hpc_R_head
hpc_R_body
hpc_R_tail
meta_R_temporal_pole
```

每个 panel 的 pre/post 均包含 140 个 item。MDS 只作为几何可视化，不作为统计证据。

### 29.2 S3：subfield Step5C 不作为证据

S3 试图在 6 个 subROI 上重做 Step5C 型模型：

```text
pair_similarity ~ C(condition) * C(time)
```

但 6 个 subROI 的 mixed model 全部返回：

```text
mixed_failed_no_fallback: Singular matrix
```

因此 S3 不能用于支持或反驳主线。它只能说明：当前 subROI 数据结构不适合再用同一 mixed model 重跑 full Step5C。

### 29.3 S4：trained-edge differentiation 支持较强

S4 是本轮 hpc extension 最有价值的结果。它直接检验 trained edge 的 post-pre drop 及其 specificity。

#### C1：YY trained drop vs zero

负值表示 post 阶段 pair similarity 低于 pre，即 trained-edge separation / differentiation。

| subROI | estimate | p | q | tier |
| --- | ---: | ---: | ---: | --- |
| hpc_L_head | -0.0860 | 7.21e-06 | 5.05e-05 | A |
| hpc_L_body | -0.0698 | 2.64e-04 | 8.10e-04 | A |
| hpc_L_tail | -0.0525 | 0.0275 | 0.0501 | B |
| hpc_R_head | -0.1059 | 8.75e-08 | 1.22e-06 | A |
| hpc_R_body | -0.0947 | 2.38e-05 | 1.25e-04 | A |
| hpc_R_tail | -0.0897 | 2.47e-04 | 8.10e-04 | A |

解释：YY trained-edge differentiation 在 5/6 subROI 通过 FDR，左 tail 为边缘。

#### C2：YY trained drop vs KJ trained drop

| subROI | estimate | p | q | tier |
| --- | ---: | ---: | ---: | --- |
| hpc_L_head | -0.0954 | 2.01e-04 | 7.80e-04 | A |
| hpc_L_body | -0.0765 | 2.04e-04 | 7.80e-04 | A |
| hpc_L_tail | -0.0440 | 0.0417 | 0.0730 | B |
| hpc_R_head | -0.0820 | 0.00156 | 0.00438 | A |
| hpc_R_body | -0.1013 | 8.75e-06 | 5.25e-05 | A |
| hpc_R_tail | -0.0677 | 0.00459 | 0.0121 | A |

解释：YY 的 trained-edge drop 大于 KJ，在 5/6 subROI 通过 FDR，左 tail 为趋势。

#### C3：YY trained drop vs pseudo edge

| subROI | estimate | p | q | tier |
| --- | ---: | ---: | ---: | --- |
| hpc_L_head | -0.0626 | 0.0151 | 0.0302 | A |
| hpc_L_body | -0.0458 | 0.0246 | 0.0470 | A |
| hpc_L_tail | -0.0340 | 0.118 | 0.184 | B |
| hpc_R_head | -0.0261 | 0.316 | 0.402 | B |
| hpc_R_body | -0.0570 | 0.0122 | 0.0270 | A |
| hpc_R_tail | -0.0334 | 0.161 | 0.233 | B |

解释：trained > pseudo 的 specificity 只在 L head、L body、R body 稳定；不是全亚区一致。

#### C4：YY trained drop vs non-edge

| subROI | estimate | p | q | tier |
| --- | ---: | ---: | ---: | --- |
| hpc_L_head | -0.1224 | 1.48e-06 | 1.56e-05 | A |
| hpc_L_body | -0.1189 | 7.37e-09 | 3.09e-07 | A |
| hpc_L_tail | -0.0524 | 0.0143 | 0.0301 | A |
| hpc_R_head | -0.1073 | 3.24e-05 | 1.51e-04 | A |
| hpc_R_body | -0.1190 | 8.24e-08 | 1.22e-06 | A |
| hpc_R_tail | -0.1078 | 4.26e-06 | 3.58e-05 | A |

解释：YY trained edge 相比 non-edge 的 differentiation 在 6/6 subROI 稳定成立。

因此 S4 的结论是：

> Exploratory hippocampal long-axis masks show robust YY trained-edge differentiation across head/body/tail segments, especially when trained edges are compared with KJ trained edges and non-edges.

### 29.4 S5：没有稳定 anterior-posterior gradient

S5 只对 trained edge 的 drop 做 anterior-posterior 模型：

```text
drop ~ axis_position * C(condition)
```

其中：

```text
head = +1
body = 0
tail = -1
```

关键结果：

| hemisphere set | term | estimate | p | q | tier |
| --- | --- | ---: | ---: | ---: | --- |
| hemi_L | `C(condition)[T.YY]` | -0.0720 | 2.74e-05 | 1.10e-04 | A |
| hemi_R | `C(condition)[T.YY]` | -0.0837 | 7.13e-06 | 4.28e-05 | A |
| pooled_LR | `C(condition)[T.YY]` | -0.0779 | 1.59e-07 | 1.90e-06 | A |
| hemi_L | `axis_position:C(condition)[T.YY]` | -0.0258 | 0.0470 | 0.141 | B |
| hemi_R | `axis_position:C(condition)[T.YY]` | -0.0072 | 0.594 | 0.792 | B |
| pooled_LR | `axis_position:C(condition)[T.YY]` | -0.0165 | 0.100 | 0.241 | B |

解释：

- YY trained-edge drop 主效应在左、右、双侧 pooled 都稳定。
- anterior-posterior gradient 交互没有通过 FDR。
- 因此不能写成 “head/body/tail 梯度机制”。

最稳写法：

> YY trained-edge differentiation generalizes across exploratory hippocampal long-axis segments, but the anterior-posterior gradient itself is not statistically stable.

### 29.5 与主线的关系

本节可以加强 §28 的 hpc-spatial post-stage differentiation 主线，但只能作为 exploratory extension。

可以写：

```text
Exploratory hippocampal long-axis analyses showed that YY trained-edge
differentiation was broadly present across head/body/tail segments.
```

不建议写：

```text
YY selectively engages posterior hippocampal pattern separation.
The hippocampal long axis explains the effect.
The result supports a CA3/DG subfield mechanism.
```

本节对最终主线的贡献是：

> 海马长轴探索性分析支持 YY relation-edge differentiation 不是单个整体海马 ROI 的偶然结果，而是在 MNI 三分的 head/body/tail segments 中均有一定泛化；但它没有提供稳定的 anterior-posterior gradient 证据，因此只能作为解剖稳健性补充，而不是新的机制主线。

### 29.6 更新后的论文定位

结合 §28 与 §29，当前最稳论文结构应是：

```text
Main evidence:
A4 post/retrieval component + trajectory
B3 semantic-hpc network coupling
B2 Mahalanobis robustness proxy

Exploratory extension:
HPC MNI long-axis trained-edge differentiation

Boundary conditions:
No strict crossnobis
No stable input-output slope transformation
No novelty moderation
No stable anterior-posterior hippocampal gradient
No CA3/DG or definitive subfield claim
```

一句话更新：

> Metaphorical learning transforms semantic relation geometry into differentiated, task-retrievable relation-edge representations, with exploratory hippocampal long-axis analyses showing broad trained-edge differentiation but no reliable anterior-posterior gradient.

## 30. Final writing/figure integration after advisor-guided consolidation

### 30.1 本轮更新目的

根据老师建议和前面所有分析结果，当前工作已经从“继续扩分析”转入：

```text
收束主线
+ 稳健性边界
+ 少量机制升级
+ 结果/讨论写作
+ 图件工程化整理
```

因此，本轮不再新增统计分析，而是把已经完成的结果整理成投稿写作可以直接调用的三层文档：

| 文件 | 当前定位 |
| --- | --- |
| `result_new_meta_roi.md` | 完整结果台账与证据边界 |
| `result_story.md` | Results / Discussion 叙事骨架与图表规划 |
| `result_final.md` | 中文 Results / Discussion 终稿草案，已嵌入本地图 |

### 30.2 figure_final 工程化整理

所有 `result_final.md` 当前使用到的图片已经集成到项目内：

```text
metaphoric/final_version/figure_final
```

当前该目录包含 43 个文件，其中包含：

- 21 个 PNG，可直接在 Markdown 中嵌入；
- 19 个 PDF，作为后续排版或矢量图备用；
- 1 个 `figure_manifest.tsv`；
- 额外保留的 temporal-pole MDS PNG/PDF。

`result_final.md` 当前嵌入 21 张 PNG，所有路径均为项目内相对路径：

```text
figure_final/*.png
```

已确认：

- 所有 Markdown 图片引用均存在；
- `result_final.md` 中没有残留外部 `E:/python_metaphor` 绝对路径；
- 文件实际编码为 UTF-8，中文内容正常。

### 30.3 图件 review 后修正

本轮图件 review 发现并修正了两个重要问题：

#### B2 图空图问题

初版 B2 图筛选条件错误：

```text
roi == "network_composite"
```

但真实数据中 network composite 的字段结构是：

```text
roi_set == "network_composite"
roi == "__network_composite__"
```

因此初版 B2 图虽然有坐标轴和图例，但没有真实数据。现已修正为：

```text
roi_set == "network_composite"
condition in ["Metaphor", "Spatial"]
```

修正后 B2 图使用 4 行 network-composite 数据：

| network | condition | correlation delta | Mahalanobis proxy delta |
| --- | --- | ---: | ---: |
| hpc_spatial | YY/Metaphor | -0.0746 | -0.0138 |
| hpc_spatial | KJ/Spatial | +0.0736 | +0.0030 |
| semantic | YY/Metaphor | -0.0668 | -0.0102 |
| semantic | KJ/Spatial | +0.0675 | +0.0075 |

更新后的 B2 图为：

```text
figure_final/fig_result_final_b2_mahalanobis_proxy.png
```

解释保持不变：B2 是 covariance-normalized Mahalanobis robustness proxy，不是 strict crossnobis evidence。

#### HPC long-axis 图空 panel 问题

初版 HPC long-axis summary 图中，C2/C4 panel 的 contrast 名称与真实表不一致，导致部分 panel 为空。真实 contrast 名称为：

```text
C2_YY_vs_KJ_trained_drop
C4_YY_trained_vs_non_edge_drop
```

并且需要只取对应 model term：

```text
C2: C(condition)[T.YY]
C4: C(edge_type)[T.trained]
```

修正后每个核心 panel 均有 6 个 subROI 数据点：

| panel | selected rows |
| --- | ---: |
| C1 YY trained drop vs zero | 6 |
| C2 YY drop vs KJ drop | 6 |
| C4 YY drop vs non-edge | 6 |

更新后的 HPC 图为：

```text
figure_final/fig_result_final_hpc_long_axis_summary.png
```

### 30.4 当前 result_final.md 的图文结构

`result_final.md` 已改为中文写作，并按以下逻辑嵌入图件：

| 文档段落 | 插图 |
| --- | --- |
| 图件索引 | `fig_result_final_contact_sheet.png` |
| 行为记忆优势 | `fig_behavior_accuracy_rt.png` |
| 学习阶段 condition geometry | `fig_learning_dynamics_meta_metaphor.png`; `fig_learning_dynamics_meta_spatial.png` |
| post trained-edge differentiation | `fig_step5c_main.png`; `fig_edge_specificity.png`; `fig_word_stability.png` |
| stage trajectory | `fig_four_phase_trajectory_meta_metaphor.png`; `fig_four_phase_trajectory_meta_spatial.png` |
| retrieval / behavior bridge | `fig_relation_behavior_prediction.png`; `fig_result_final_b3_network_coupling.png`; `fig_repr_connectivity.png` |
| memory component | `fig_result_final_a4_memory_components.png`; `fig_mechanism_behavior_prediction.png` |
| robustness boundary | `fig_result_final_b2_mahalanobis_proxy.png`; `fig_result_final_c1_c2_c3_boundaries.png` |
| HPC exploratory extension | `fig_result_final_hpc_long_axis_summary.png`; right hippocampus/head/body/tail MDS |

### 30.5 是否需要更新 result_story.md

需要更新。

原因是 `result_story.md` 原先只覆盖到 §29，尚未记录：

1. `figure_final/` 已作为最终图件目录；
2. B2 图件筛选 bug 已修正；
3. HPC long-axis summary 图件的 C2/C4 panel 已修正；
4. `result_final.md` 已改为中文 Results/Discussion 草稿；
5. 主文图件和补充图件的落位应以 `figure_final` 版本为准。

因此，story 的更新重点不是改变主线，而是同步最终写作状态和 figure blueprint。

### 30.6 最终主线保持不变

本轮图件与写作整理没有改变统计结论。最终主线仍为：

```text
Behavior:
YY memory advantage

Learning:
YY/KJ condition-level geometry emerges in semantic/metaphor network

Post:
YY trained relation edges show specific differentiation,
especially in hippocampal-spatial / scene-context network

Retrieval:
YY pair-structure rebounds and semantic/hpc-spatial rebound is coupled

Memory:
remembered YY items show stronger prior post-stage separation,
not pure retrieval reinstatement
```

最终一句话版本仍为：

> 隐喻学习将既有语义几何转化为分化的、可在提取阶段重新调动的关系边表征。

### 30.7 下一步建议

当前不建议继续新增主线分析。更合适的下一步是：

1. 基于 `result_final.md` 继续打磨中文 Results / Discussion；
2. 再将中文稿翻译或重写为英文 manuscript prose；
3. 把 `figure_final` 中主文图和补充图重新编号；
4. 将 C1/C2/C3、B2 方法细节、HPC S1-S5、MDS 全部降级到 robustness/supplement；
5. 保持 B2、HPC、novelty、input-output slope 的边界化表述，避免写成强机制证据。

## 31. relation_edge_story_revision：材料调节、轨迹几何、网络耦合与 MVPA 整合

### 31.1 执行状态

本轮按 spec relation_edge_story_revision 完成 r0-r8 全流程。输出目录为 paper_outputs/qc/revision_relation_edge/、paper_outputs/figures_revision/，并已将新增图复制到 figure_final/fig_revision_*.png/pdf。

核心脚本均已执行：r0_build_revision_master_table.py、r1_material_distance_novelty_moderation.py、r2_high_low_post_separation_item_profile.py、r3_pre_post_retrieval_geometry.py、r4_network_coupling_revision.py、r5_mvpa_stage_state_summary.py、r6_update_evidence_tier_and_story.py、r7_make_revision_figures.py、r8_append_revision_story.py。

### 31.2 R0 schema 与 master table

R0 生成统一 revision master table，避免后续重复 merge 造成 item ID 错配。revision_schema_audit.tsv 显示：YY_28 与 KJ_28 保持区分，condition_item_id 没有 condition collision；network-level master table 包含 26 名被试、70 个 condition items、3,640 行，覆盖 semantic/metaphor 与 hpc-spatial 两个主网络。

需要记录的边界是：clean refined behavior table 中目前只有 run7 行为数据，因此 run3_understand、run4_like、run3_rt_z、run4_rt_z 在 revision master 中作为显式缺失 placeholder 保留，不能在当前版本中解释 learning behavior proxy。

### 31.3 R1 材料语义距离 / 新颖性调节

R1 回答三个问题：原始语义距离是否影响记忆；控制语义距离后 YY novelty 是否改善记忆；semantic distance / novelty 是否解释 post-stage separation。

memory model 中，YY-specific novelty moderation 不稳定。hpc-spatial 中 YY x novelty estimate = +0.0119, q = 0.897；semantic/metaphor 中 estimate = +0.0044, q = 0.932。YY x pre_pair_similarity 对 memory 也没有通过校正：hpc-spatial estimate = -0.0491, q = 0.250；semantic/metaphor estimate = -0.0120, q = 0.775。因此不能写“更 novel 的 YY metaphors 记忆更好”，也不能写“原始语义距离解释 YY memory advantage”。

post separation model 中，pre_pair_similarity 对 post separation 有很强总体关系：hpc-spatial estimate = +0.7309, q = 2.84e-89；semantic/metaphor estimate = +0.6952, q = 5.08e-91。YY-specific pre-similarity interaction 不稳定：hpc-spatial q = 0.751；semantic/metaphor q = 0.450。YY-specific novelty moderation 对 post separation 呈边界性支持：hpc-spatial estimate = +0.2210, within-model q = 0.0494, across-model q = 0.0593；semantic/metaphor estimate = +0.2077, q = 0.0599。因此可以写成 supplementary support / boundary，不应升为主机制。

### 31.4 R2 high vs low post separation profile

R2 在 YY 内部比较 high vs low post-separation items。最稳定的材料特征是 pre_pair_similarity：top/bottom 30% split 中，hpc-spatial low vs high estimate = -0.6269, p = 6.69e-06；semantic/metaphor low vs high estimate = -0.6917, p = 7.38e-07。median split 中方向一致：hpc-spatial estimate = -0.5615, p = 7.77e-07；semantic/metaphor estimate = -0.5260, p = 1.12e-05。

novelty、familiarity、word frequency、valence、arousal 与 memory score 都没有形成稳定 high/low profile。semantic median split 中 difficulty 有一定差异，estimate = -0.9186, q = 0.0485，但 top/bottom 30% 没有同样稳定。因此 R2 最适合写成：post-stage separation 更强的 YY items 往往具有更高的原始 pre-pair similarity，但 post separation 不是由 novelty 或单一材料维度决定。

### 31.5 R3 pre-post-retrieval trajectory geometry

R3 使用 pair-similarity scalar proxy 计算 d_pre_post、d_post_retrieval、d_pre_retrieval、return_to_pre_index 与 retrieval_new_state_index。这不是 voxel-level displacement vector，因此不能作为严格几何方向证据。

描述统计显示，YY 在两个网络中都有更大的 post drop 与 retrieval rebound：hpc-spatial 中 YY post_drop_raw = 0.0724、retrieval_rebound_raw = 0.0589；KJ 分别为 0.0072 与 0.0107。semantic/metaphor 中 YY post_drop_raw = 0.0616、retrieval_rebound_raw = 0.0616；KJ 分别为 -0.0063 与 0.0082。

但是 trajectory metric 模型没有给出稳定的 return-to-pre 或 new-state reconstruction 主效应。hpc-spatial return_to_pre_index 的 YY effect q = 0.561；semantic/metaphor q = 0.957。memory interaction 也不稳定。结论应写为：当前 scalar proxy 支持 YY 的 post drop / retrieval rebound 描述，但不能证明 retrieval 是完全回到 pre，也不能强证 new-state reconstruction；voxel-level vector alignment 在本轮不可用。

### 31.6 R4 semantic-hpc network coupling revision

R4 只在预定义 semantic/metaphor 与 hpc-spatial 网络之间做 coupling。retrieval rebound coupling 保持最强结果：semantic_retrieval_rebound_z 强预测 hpc_spatial_retrieval_rebound_z，estimate = +0.6977, q = 5.57e-63。YY-specific interaction 为负，estimate = -0.1095, p = 0.0392，但校正后 q = 0.196，不能写成 YY-specific coupling。

stage coupling 中 YY x retrieval 方向为正，estimate = +0.1526, p = 0.0213，但 q = 0.128，只能作为趋势或补充。memory coupling 不稳定：YY x semantic_hpc_coupling_retrieval_z estimate = +0.0119, q = 0.783；YY x semantic_hpc_coupling_post_z estimate = +0.0018, q = 0.944。因此 R4 支持 retrieval-stage inter-network coordination，但不支持 YY-specific causal communication 或 coupling-memory bridge。

### 31.7 R5 MVPA stage-state summary

R5 将 MVPA 放回 stage-state evidence 的位置。learning 阶段有 10 个 q < .05 的 decoding rows，best q = 0.000036；post 阶段有 2 个 q < .05，best q = 0.0354；retrieval 阶段有 4 个 q < .05，best q = 0.000030。pre 阶段没有 q < .05，best q = 0.0983；MVPA-behavior bridge 没有稳定校正结果。

因此 MVPA 应承担“YY/KJ condition information 在哪些阶段被调动”的证据，而不是 edge-reorganization mechanism evidence。RSA 解释 pair/edge geometry，MVPA 解释 stage-state condition information。

### 31.8 本轮故事影响

本轮 revision 不推翻原主线，而是让主线更收束：YY memory advantage、learning condition-level geometry、post trained-edge differentiation、retrieval task-driven rebound、remembered YY items stronger prior post-stage separation 仍是主线。新增分析主要提供三类边界：YY memory advantage 不能由 novelty 或 semantic distance 简单解释；high-separation YY items 与更高 pre-pair similarity 相关，但 novelty / memory profile 不稳定；retrieval-stage semantic-hpc coordination 很强，但不是 YY-specific causal coupling，trajectory geometry 不能强写成 return-to-pre 或 new-state proof。

最安全的新表述是：隐喻学习的关键不是让两个概念更相似，而是把 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 能在 retrieval 阶段被任务依赖地重新组织。后续被记住的 YY items 主要表现为 stronger prior post-stage separation。
## 32. Source-first audit 与 run3/run4 行为接入更正

### 32.1 更正 run3/run4 placeholder 判断

第 31 节中“run3_understand、run4_like、run3_rt_z、run4_rt_z 只能作为 placeholder”的判断已经过时。重新审计后发现，精简版 `*_events.tsv` 中确实没有这些字段，但原始 E-Prime CSV 中存在完整学习阶段行为列：

```text
E:/python_metaphor/data_events/sub-XX/sub-XX_run-3_events.csv
run3: sentence3_RESP, sentence3_RT, sentence3_ACC

E:/python_metaphor/data_events/sub-XX/sub-XX_run-4_events.csv
run4: sentence4_RESP, sentence4_RT, sentence4_ACC
```

已有脚本 `brain_behavior/learning_reactivation/build_learning_behavior_table.py` 会从这些源 CSV 中抽取 run3/run4 行为。我已更新 `paper_analysis/revision_relation_edge/r0_build_revision_master_table.py`，使 revision master table 自动读取 `E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_behavior_item.tsv`；如果该表不存在，则从源 CSV 重新生成。

### 32.2 source inventory 输出

新增脚本：

```text
paper_analysis/revision_relation_edge/r0a_source_inventory.py
```

输出目录：

```text
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/
```

变量来源分层如下：

```text
source:
run3_understand / run3_rt_z
run4_like / run4_rt_z
memory_ord / memory_strict / memory_lenient
novelty / familiarity / comprehensibility / difficulty

source_or_material_summary:
word frequency / length / valence / arousal

locked_upstream_neural_summary:
pre/post/retrieval pair similarity
post_separation / retrieval_rebound
semantic/hpc-spatial network composites
MVPA stage-state summary
```

因此，目前 revision 已经修复学习行为遗漏，但神经 RSA/MVPA 指标仍是读取上游锁定汇总表，而不是在 revision 脚本中从 `pattern_root/*.nii.gz` 重新计算。若要满足最严格的 source-only rerun，需要额外加入 neural recomputation step。

### 32.3 重新接入 run3/run4 后的覆盖率

重跑 `r0_build_revision_master_table.py` 后，学习阶段行为字段覆盖率为：

```text
run3_understand: 98.57%
run4_like: 97.86%
run3_rt: 98.57%
run4_rt: 97.86%
run3_rt_z: 98.57%
run4_rt_z: 97.86%
learning_fluency_shift: 96.43%
```

### 32.4 重新接入 run3/run4 后的 R2 更新

重跑 `r2_high_low_post_separation_item_profile.py` 后，high vs low post-separation item profile 中已包含 run3/run4 行为特征。结果显示，post-separation 更强的 YY items 并没有稳定表现出更高 run3 RT、run4 RT 或更大的 learning_fluency_shift。

top/bottom 30% split 中：

```text
hpc-spatial:
high learning_fluency_shift = 0.060
low  learning_fluency_shift = 0.100
low - high estimate = 0.039, p = 0.777

semantic/metaphor:
high learning_fluency_shift = 0.055
low  learning_fluency_shift = 0.100
low - high estimate = 0.045, p = 0.754
```

median split sensitivity 也不支持 learning_fluency_shift 区分 high/low separation：

```text
hpc-spatial: estimate = 0.100, p = 0.403
semantic/metaphor: estimate = 0.121, p = 0.309
```

因此，当前更稳妥的解释是：post-stage separation 更强的 YY items 主要仍与更高 pre-pair similarity 相关，而不是由 run3/run4 行为流畅性或学习反应模式直接解释。

### 32.5 对故事的影响

这次 source-first 更正增强了一个边界结论：学习阶段行为 proxy 没有解释哪些 YY items 更容易发生 post-stage separation。也就是说，当前数据仍不支持把故事写成：

```text
learning behavior update -> post separation -> memory
```

更安全的写法仍是：

```text
learning 阶段出现 condition-level semantic geometry；
post 阶段出现 YY trained-edge differentiation；
后续 remembered YY items 表现为 stronger prior post-stage separation；
run3/run4 行为 proxy 尚不能解释这种 post separation 的 item 差异。
```

## 33. append_518 ROI-level 与 learning-boundary 整合

append_518 的主要作用不是产生新主线，而是把第 31-32 节的 relation-edge revision 拆到 ROI、network、learning behavior、activation/connectivity sensitivity 和 trajectory decomposition 层面。整体结论是：主故事仍保持为 learning condition geometry -> post trained-edge differentiation -> retrieval task-driven rebound / coordination -> remembered YY items show stronger prior post-stage separation；518 新增的是更清楚的边界，尤其是 run3/run4 learning behavior 不能解释 post-stage separation，也不能被写成正向 memory mechanism。

### 33.1 pre-existing pair similarity 是 post separation 的强总体边界

E1 pre-similarity tertile 结果显示，pre_pair_similarity 对后续 post_edge_differentiation_z 的影响非常强，而且是跨 ROI 的总体结构。high vs low main effect 在 18/18 ROI FDR 显著；mid vs low 也在 18/18 ROI FDR 显著。condition-specific 检查中，KJ 和 YY 内部 high-low 均显著。

但是，这个结果不能写成 YY-specific moderation。YY-specific interaction 为 0/18 ROI FDR 显著，`did_yy_minus_kj_high_low` 也没有 FDR 稳定证据。因此最稳妥的写法是：items with stronger pre-existing pair structure tend to show stronger post-stage separation overall；这解释的是材料/结构边界，不是 YY 独有机制。

network-level 辅助分析给出同一方向：pre-sim tertile high vs low 预测 post_edge_differentiation_z，在 hpc_spatial 中 estimate = +1.5789, p = 8.01e-127, q = 1.76e-125；在 semantic 中 estimate = +1.6592, p = 1.08e-130, q = 4.76e-129。

### 33.2 run3/run4 learning behavior 不能解释 post separation

E2 learning bridge 显示，run3_understand 与 run4_like 不稳定预测 post_edge_differentiation_z，也不稳定预测 post pair similarity。learning behavior -> post_edge_differentiation_z 的 focal terms 均为 0/18 ROI FDR：

```text
YY x run3_understand_yes: 0/18 ROI FDR, estimate range -0.1101 to +0.2787, min p = 0.0680, min q = 0.4270
YY x run4_like_yes:       0/18 ROI FDR, estimate range -0.1241 to +0.1109, min p = 0.1362, min q = 0.7130
```

learning behavior -> post pair similarity 的 focal terms 也均为 0/18 ROI FDR：

```text
YY x run3_understand_yes: 0/18 ROI FDR, estimate range -0.3507 to +0.1784, min p = 0.1056, min q = 0.8674
YY x run4_like_yes:       0/18 ROI FDR, estimate range -0.1105 to +0.1761, min p = 0.1281, min q = 0.8674
```

同一模型中，pre_pair_similarity_z 仍为 18/18 ROI 显著，说明 post-stage separation 更稳定地受 pre-existing pair structure 约束，而不是由 run3/run4 understand/like 行为 proxy 解释。

### 33.3 learning behavior 与 memory 的方向不能主线化

direct memory model 中，YY x run3_understand_yes 与 YY x run4_like_yes 在所有 ROI 中均为 FDR 显著负向：

```text
YY x run3_understand_yes: 18/18 ROI FDR, estimate range -0.7267 to -0.6712, min p = 3.38e-08, min q = 4.33e-07
YY x run4_like_yes:       18/18 ROI FDR, estimate range -0.5531 to -0.4527, min p = 5.04e-05, min q = 1.55e-04
```

加入 post_edge_differentiation_z 后，两个 interaction 仍然是 18/18 ROI FDR 显著负向：

```text
YY x run3_understand_yes: 18/18 ROI FDR, estimate range -0.7417 to -0.6675, min p = 1.85e-08, min q = 2.53e-07
YY x run4_like_yes:       18/18 ROI FDR, estimate range -0.5308 to -0.4530, min p = 1.05e-04, min q = 3.45e-04
```

因此不能写成 YY items 在学习时越理解/越喜欢，后续记忆越好。更安全的解释是：这些二值学习行为项捕捉到的可能是 item/task/response-selection boundary，而不是正向学习强度机制。

network-level direct memory 模型方向一致：YY x run3_understand 为负，hpc_spatial estimate = -0.6904, p = 1.55e-07, q = 7.77e-07；semantic estimate = -0.7058, p = 8.38e-08, q = 5.03e-07。YY x run4_like 也为负，hpc_spatial estimate = -0.5089, p = 1.89e-04, q = 5.15e-04；semantic estimate = -0.4917, p = 3.12e-04, q = 7.80e-04。

### 33.4 post_edge_differentiation_z 的 memory bridge 是局部 ROI 证据

E2 中加入 learning behavior 后，post_edge_differentiation_z 对 memory 的预测在 4/18 ROI FDR 显著，estimate range = -0.0336 to +0.2619, min p = 0.0027, min q = 0.0072。FDR-significant ROI 主要是：

```text
meta_L_PPC_SPL
meta_R_PPC_SPL
meta_R_precuneus
meta_R_pMTG_pSTS
```

E6 focused learning-to-post bridge 在这 4 个 memory-linked ROI 中进一步显示，learning behavior / activation -> post edge 没有 FDR 稳定证据；post_edge_differentiation_z 本身仍在 4/4 ROI 中显著，estimate range = +0.1977 to +0.2591, min p = 0.0025, min q = 0.0075。也就是说，post edge 与 memory 的桥接可以作为 ROI-level 支持，但 learning behavior / activation 不能被写成上游中介。

network-level D1 learning bridge 中，post_edge_differentiation_z -> memory 在 hpc_spatial 显著，estimate = +0.3010, p = 3.29e-04, q = 9.56e-04；semantic 为 trend/boundary，estimate = +0.1500, p = 0.0634, q = 0.0922。

### 33.5 trajectory decomposition 没有支持 return-to-pre 或 new-state 主张

E3 trajectory decomposition 将 pair trajectory 拆成 return_to_pre_cos、new_state_fraction、return_new_log_ratio、differentiation_axis_cos 等指标。核心结果均没有 FDR 稳定证据：

```text
return_to_pre_cos | YY x memory | 18 ROI | 0 FDR | estimate range -0.0759 to +0.0572 | min p = 0.1574 | min q = 0.9120
new_state_fraction | YY x memory | 18 ROI | 0 FDR | estimate range -0.0280 to +0.0393 | min p = 0.2299 | min q = 0.9670
return_new_log_ratio | YY x memory | 18 ROI | 0 FDR | estimate range -0.3015 to +0.1200 | min p = 0.1509 | min q = 0.9120
differentiation_axis_cos | YY x memory | 18 ROI | 0 FDR | estimate range -0.0539 to +0.0594 | min p = 0.3007 | min q = 0.9670
```

因此，trajectory 结果只能维持为 post-drop / retrieval-rebound 的描述性支持，不能写成 retrieval 回到 pre，也不能写成形成 new state。

### 33.6 activation/connectivity SME 与 pooled understand 分析只作 sensitivity

E4 activation~memory 的 focal terms 没有稳定 FDR 证据；YY x memory 为 0/36，min p = 0.0066, q = 0.4349。反向 memory~activation 模型中，YY x activation 有 5/36 FDR，min p = 0.0011, q = 0.0030，但方向混合，不能主线化。

E5 beta-series connectivity 的 YY x memory_strict 为 0/306 FDR，min p = 0.0041, q = 0.0850；且 1060/1224 coefficients 使用 fallback_after_mixed_failure，因此只能作为 sensitivity boundary。E7 pooled run3 understand -> post edge 的 network 结果均不显著；ROI simple pooled 仅 `meta_R_hippocampus` 进入 family-level caveat，adjusted focal family 不稳定。结论是 pooled run3 understand 不是稳定正向 predictor。

### 33.7 对故事的影响

518 之后，故事应更保守但更干净：

```text
pre-existing pair structure constrains post-stage separation;
run3/run4 learning behavior does not explain this separation;
post_edge_differentiation_z has a focal ROI/network bridge to memory;
trajectory and activation/connectivity sensitivity analyses do not create a new mechanism.
```

这支持把 memory endpoint 写成 remembered YY items show stronger prior post-stage separation，但不支持 learning behavior -> post edge -> memory 的因果中介链。

## 34. append_519 Dm panorama、ERS revival、subfield/run-level boundary 整合

append_519 继续沿着 memory endpoint 和 robustness boundary 展开。整体结果是：ordinal Dm 对 A4 memory bridge 有支持，尤其是 YY x post_pair_similarity_network_z 的负向交互；但 composite、per-subject Dm、ERS revival、subfield 和 run-level trajectory 都没有生成新的主机制。因此 519 应写进故事的部分主要是 ordinal robustness；其余进入 evidence boundary。

### 34.1 A4b ordinal Dm 支持 stronger prior post-stage separation

A4b ordered_model 使用 5 个 input paths，n_rows = 38340，n_subjects = 28，网络为 hpc_spatial 与 semantic。ordinal focal 结果中，YY x post_pair_similarity_network_z 在两个网络均为显著负向：

```text
semantic YY x post_pair_similarity_network_z:    estimate = -0.2085, p = 6.43e-11, q = 1.71e-10
hpc_spatial YY x post_pair_similarity_network_z: estimate = -0.1395, p = 9.97e-07, q = 1.99e-06
```

这与 A4 binary/strict 结果一致，支持 remembered YY items had stronger prior post-stage separation。semantic 网络中 YY x retrieval_pair_similarity_network_z 为正，estimate = +0.1082, p = 5.66e-04, q = 7.15e-04；semantic post_pair_similarity_network_z 主效应为正，estimate = +0.0466, p = 0.0350, q = 0.0420。hpc_spatial YY x retrieval_pair_similarity_network_z 不稳定，estimate = +0.0415, p = 0.139, q = 0.158。

binary sanity check 也保留同一边界：

```text
semantic YY x retrieval_pair_similarity_network_z:    estimate = +0.1291, p = 2.48e-06, q = 4.20e-06
hpc_spatial post_pair_similarity_network_z:           estimate = -0.0727, p = 4.83e-05, q = 7.59e-05
hpc_spatial retrieval_pair_similarity_network_z:      estimate = +0.0669, p = 1.94e-04, q = 2.85e-04
semantic YY x post_pair_similarity_network_z:         estimate = -0.1022, p = 3.46e-04, q = 4.75e-04
```

解释上，ordinal Dm 可以写成对 A4 的 robustness：memory endpoint 不是高 post similarity，而是 YY remembered items 更强的 prior post-stage separation；retrieval 相关项存在网络特异支持，但不足以替代 memory bridge 的主表述。

### 34.2 A4c composite 没有改写 network-level 结论

A4c composite 使用 n_composite_rows = 1820，n_subjects = 26。PCA1 为 post-dominant component，loadings 约为 post hpc = -0.6742、post semantic = -0.6675；PCA2 更偏 retrieval-dominant。strict memory composite focal predictors 全部不显著：

```text
strict YY x PCA1: estimate = +0.0347, p = 0.572, q = 0.626
strict YY x mean: estimate = +0.0103, p = 0.932, q = 0.932
```

lenient YY x PCA1 为正，estimate = +0.2307, p = 0.00748, q = 0.0195，但这属于 sensitivity，不能改写 strict 主结论。`composite_vs_network_comparison.tsv` 因 a4 table 中缺少 `__model__` rows，状态为 `missing_model_row`，因此不能用该表做正式 model-vs-network 强比较。

### 34.3 D1c per-subject multivariate Dm 没有形成稳定 group-level bridge

D1c 合并后 merged_rows = 1960，n_subjects_raw = 28；eligible subjects = 27，dropped subject 为 sub-12。模型状态为 fits_logit = 21，fits_ols_fallback = 14，fits_skipped = 46。

group inference 全部不显著：

```text
KJ run3: mean_beta = +5.5812, p = 0.1429, q = 0.9677, n = 9
KJ run4: mean_beta = -2.7249, p = 0.3629, q = 0.9677, n = 9
YY run3: mean_beta = +0.4311, p = 0.9287, q = 0.9782, n = 12
YY run4: mean_beta = -2.0705, p = 0.6448, q = 0.9782, n = 12
pooled run3: mean_beta = +0.1068, p = 0.9782, q = 0.9782, n = 14
pooled run4: mean_beta = -0.0883, p = 0.6109, q = 0.9782, n = 14
paired YY-KJ run3: mean_beta = -5.6807, p = 0.2422, q = 0.9677, n = 8
paired YY-KJ run4: mean_beta = -0.6988, p = 0.8990, q = 0.9782, n = 8
```

因此 D1c 不能写成 subject-level multivariate Dm 机制，只能作为 negative/sensitivity boundary。

### 34.4 D3 ERS revival 不支持 retrieval reinstatement explains memory

D3 使用 `nifti_pattern_root_pair_mean` 作为 pattern_source，n_rows_item = 32760，n_subjects = 26，网络为 hpc_spatial 与 semantic。post-to-retrieval own-other diff 的描述统计接近 0：

```text
hpc_spatial KJ: n = 9100, mean = -0.0039, sd = 0.2906
hpc_spatial YY: n = 9100, mean = -0.0011, sd = 0.2922
semantic KJ:    n = 7280, mean = +0.0044, sd = 0.3149
semantic YY:    n = 7280, mean = -0.0002, sd = 0.3185
```

memory moderation models 共 288 rows，模型状态均 ok，但没有 FDR 显著结果。strict key terms：

```text
hpc_spatial YY:          estimate = -0.0143, p = 0.184, q = 0.947
hpc_spatial memory:      estimate = +0.0051, p = 0.606, q = 0.947
hpc_spatial YY x memory: estimate = +0.0110, p = 0.388, q = 0.947
semantic YY:             estimate = +0.0063, p = 0.568, q = 0.947
semantic memory:         estimate = +0.0081, p = 0.426, q = 0.947
semantic YY x memory:    estimate = -0.0175, p = 0.181, q = 0.947
```

baseline pre-to-retrieval own semantic intercept 为正，estimate = +0.0271, p = 0.000323, q = 0.00258，但这不是 condition 或 memory moderation。因此 D3 的结论是：ERS/reinstatement 不能解释 YY memory advantage，也不能替代 A4 的 post-stage separation memory endpoint。

### 34.5 D4 hippocampal subfield 是 supplementary anatomical boundary

D4 使用 n_beta_rows = 68880，n_pair_rows = 34440，segments = body/head/tail，hemispheres = L/R。per-seg strict focal YY x pair 没有 FDR 稳定结果，最接近的是：

```text
body R: estimate = -0.1521, p = 0.054, q = 0.0729
head R: estimate = -0.0856, p = 0.269, q = 0.320
body L: estimate = -0.0793, p = 0.312, q = 0.366
```

lenient per-seg focal 有负向显著结果：

```text
body L: estimate = -0.4277, p = 7.16e-05, q = 3.09e-04
head L: estimate = -0.2657, p = 0.0131, q = 0.0313
head R: estimate = -0.2474, p = 0.0142, q = 0.0333
body R: estimate = -0.2510, p = 0.0204, q = 0.0438
tail L: estimate = -0.2142, p = 0.0373, q = 0.0611
```

pooled three-way 模型给出更稳定的整体方向：

```text
strict YY x pair:        estimate = -0.1240, p = 1.30e-04, q = 3.11e-04
strict YY x pair x tail: estimate = +0.1306, p = 0.0189, q = 0.0377
lenient pair:            estimate = +0.1651, p = 7.08e-09, q = 2.55e-08
lenient YY x pair:       estimate = -0.3558, p = 1.21e-15, q = 6.24e-15
lenient YY x pair x tail: estimate = +0.2198, p = 0.00406, q = 0.00861
```

解释应保持保守：subfield 支持 hippocampal involvement 的 supplementary robustness，但没有清晰 AP gradient；tail 相对 body 的负向 YY slope 较弱，不能写成强 subfield mechanism。

### 34.6 D5 post run5/6 run-level trajectory 没有积累证据

D5 使用 `post_nifti_stream`，n_pair_rows = 22176，n_subjects = 28，runs_present = 5/6。重要边界是：run3/run4 learning behavior 来自 sentence/item trial，不是 isolated-word two-word pair similarity；因此不能把 run3/run4 与 D5 的 post run5/6 pair similarity 强行合并成一条学习到测试的连续 trajectory。D5 只回答 post 阶段 run5/6 内是否有 run-level accumulation。

描述统计如下：

```text
hpc_spatial KJ run5: n = 3640, mean_raw = 0.2524, mean_z = +0.0289
hpc_spatial KJ run6: n = 3640, mean_raw = 0.2361, mean_z = +0.0157
hpc_spatial YY run5: n = 2520, mean_raw = 0.2111, mean_z = -0.0417
hpc_spatial YY run6: n = 2520, mean_raw = 0.2158, mean_z = -0.0227

semantic KJ run5: n = 2912, mean_raw = 0.2812, mean_z = +0.0214
semantic KJ run6: n = 2912, mean_raw = 0.2372, mean_z = -0.0026
semantic YY run5: n = 2016, mean_raw = 0.2461, mean_z = -0.0309
semantic YY run6: n = 2016, mean_raw = 0.2412, mean_z = +0.0038
```

models 共 104 rows，均 ok，但没有 FDR 显著。最接近的 terms 为：

```text
hpc_spatial strict linear YY x run_index x memory: estimate = -0.4355, p = 0.0193, q = 0.328
hpc_spatial strict YY x memory:                    estimate = +2.3509, p = 0.0221, q = 0.328
semantic strict YY x run_index x memory:           estimate = -0.3370, p = 0.0429, q = 0.430
semantic categorical YY x run6 x memory:           estimate = -0.3369, p = 0.0435, q = 0.430
```

因此 D5 不能写成 run-level accumulation evidence。它只支持一个边界：post-stage YY/KJ pair similarity 差异存在描述性 run variation，但没有稳定 memory-moderated run trajectory。

### 34.7 519 对故事的影响

519 之后，能写进主故事的是 ordinal robustness：A4 的 memory bridge 在 ordered memory 上仍显示 YY x post similarity 负向，即 remembered YY items 更像是已有 post-stage separation 更强，而不是 retrieval reinstatement 更强。不能写进主故事的是：composite strict predictor、per-subject multivariate Dm、ERS memory moderation、subfield AP gradient、post run5/6 run-level accumulation。

与 518 合并后，当前最稳版本仍是：

```text
learning 阶段出现 condition-level semantic geometry；
post 阶段出现 YY trained-edge differentiation，并受 pre-existing pair structure 约束；
retrieval 阶段出现 task-driven rebound / semantic-hpc coordination；
后续 remembered YY items 表现为 stronger prior post-stage separation；
run3/run4 行为、ERS、subfield/run-level trajectory 尚不能解释该 memory endpoint。
```
