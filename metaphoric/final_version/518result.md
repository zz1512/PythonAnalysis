# append_518 新增分析结果整合：ROI 级主结果与网络级辅助结果

更新时间：2026-05-19  
执行环境：`C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe`  
正式输出根目录：`E:/python_metaphor/paper_outputs/qc/nc_converge/`

本文件按当前写作策略重写：**ROI 级结果作为本轮 append_518 的主结果**，`semantic/metaphor network` 与 `hpc-spatial network` 的 composite 结果作为辅助验证。统计解释严格以 FDR 后结果为准；未通过 FDR 的 uncorrected 线索只作为边界信息，不作为主证据。

本轮新增或重跑的 ROI 级脚本包括：

```text
nc_converge/e1_all_roi_pre_similarity_tertile.py
nc_converge/e2_all_roi_learning_bridge.py
nc_converge/e3_all_roi_trajectory_decomposition.py
```

对应的 network 级辅助脚本/结果包括：

```text
nc_converge/c1b_pre_similarity_tertile.py
nc_converge/d1_learning_understand_like_bridge.py
nc_converge/d2_voxel_trajectory_alignment.py
nc_converge/e3_all_roi_trajectory_decomposition.py 中的 network aggregation
```

---

## 0. 总体结论

ROI 级分析没有推翻当前主线，而是把三个问题说得更清楚：

1. **pre-existing pair similarity 确实强烈影响 post-stage separation，但不是 YY 特异。**  
   在 18 个 ROI 中，KJ 和 YY 的 high-pre-similarity items 都表现出显著更强的 post edge differentiation。`YY high-low - KJ high-low` 的 difference-in-differences 在所有 ROI 中均未通过 FDR。

2. **run3 理解和 run4 喜欢不稳定预测 post-stage separation 或 post pair similarity。**  
   ROI 级结果与 network 级结果一致：learning-stage subjective response 不能解释 post-stage separation，因此目前不能写成 `learning understand/like -> post separation -> memory`。

3. **run3 理解与记忆有关，但不是 YY 专属正向机制。**  
   在 memory direct model 中，`run3_understand_yes` 的主效应反映的是 KJ baseline 下的正向关系；同时 `YY × run3_understand_yes` 和 `YY × run4_like_yes` 均为负向且通过 FDR。合成 YY slope 后并不支持“YY 理解/喜欢越高，记忆越好”。因此 learning behavior 可以写成 condition-dependent behavioral marker，不能写成 YY-specific positive learning pathway。

4. **post edge differentiation 与 memory 的 ROI 级桥接主要落在 PPC/precuneus 与右 pMTG-pSTS。**  
   在加入 learning behavior 与材料协变量后，`post_edge_differentiation_z` 仍在 4 个 ROI 中正向预测 memory 并通过 FDR：`meta_L_PPC_SPL`、`meta_R_PPC_SPL`、`meta_R_precuneus`、`meta_R_pMTG_pSTS`。这比单纯 network 平均更适合写成本轮 ROI 级主发现。

5. **trajectory decomposition 显示 retrieval 不是简单 return-to-pre，但 YY/memory 特异性不成立。**  
   ROI 描述性结果显示 post -> retrieval 位移同时包含 return-to-pre 成分和较大的 orthogonal/new-state 成分；但 condition、memory、condition × memory 在 ROI 和 network 层面都未通过 FDR。因此这部分应写成“retrieval 包含重构成分”的边界证据，而不是 YY-specific 或 memory-specific 的强机制。

建议当前故事写法：

> 隐喻学习的核心证据不在于学习阶段主观理解/喜欢直接导致 post separation，也不在于 retrieval 简单恢复 pre 几何；更稳妥的说法是，post-stage trained-edge differentiation 是后续记忆优势的关键表征桥接，其中 ROI 级证据主要集中在 PPC/precuneus 与右 pMTG-pSTS。retrieval 阶段呈现部分 return-to-pre 和明显 new-state component，但这种轨迹结构并非 YY 特异。

---

## 1. ROI 级主结果一：pre-similarity tertile 与 post-stage separation

### 1.1 分析目的

本分析回答：

```text
原本 pre 阶段神经相似性更高的词对，
在 post 阶段是否更容易发生 edge differentiation？
这个效应是否在 YY 隐喻学习中比 KJ 空间关系学习更强？
```

ROI 级脚本：

```text
nc_converge/e1_all_roi_pre_similarity_tertile.py
```

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/e1_all_roi_pre_similarity_tertile/
```

输入与样本：

```text
input rows      = 35280
prepared rows   = 35280
n ROI           = 18
low quantile    = 0.30
high quantile   = 0.70
status          = ok
```

18 个 ROI 包括：

```text
semantic/metaphor ROI:
meta_L_IFG, meta_R_IFG,
meta_L_temporal_pole, meta_R_temporal_pole,
meta_L_AG, meta_R_AG,
meta_L_pMTG_pSTS, meta_R_pMTG_pSTS

hpc-spatial ROI:
meta_L_hippocampus, meta_R_hippocampus,
meta_L_PPA_PHG, meta_R_PPA_PHG,
meta_L_RSC_PCC, meta_R_RSC_PCC,
meta_L_PPC_SPL, meta_R_PPC_SPL,
meta_L_precuneus, meta_R_precuneus
```

核心模型：

```text
post_edge_differentiation_z ~ condition * pre_tertile + covariates
post_pair_similarity_z      ~ condition * pre_tertile + covariates
```

其中 `pre_tertile` 在每个 `ROI × condition` 内部划分为 low / mid / high，避免 YY/KJ 的整体分布差异影响 tertile 定义。

### 1.2 ROI 级结果：high pre similarity 强烈预测 post separation

在 `post_edge_differentiation_z` 模型中，`high` tertile 相对 `low` tertile 的主效应在 18/18 个 ROI 中均通过 FDR。

```text
term:
C(pre_tertile, Treatment(reference='low'))[T.high]

n ROI             = 18
FDR-significant   = 18 / 18
estimate range    = 1.2242 to 1.9620
minimum p         = 5.43e-164
minimum q_BH      = 2.15e-161
```

mid tertile 相对 low tertile 也在 18/18 个 ROI 中通过 FDR：

```text
term:
C(pre_tertile, Treatment(reference='low'))[T.mid]

n ROI             = 18
FDR-significant   = 18 / 18
estimate range    = 0.6451 to 1.1392
minimum p         = 4.43e-65
minimum q_BH      = 7.98e-64
```

解释：

```text
pre 阶段神经相似性越高的词对，
post 阶段越容易表现为更高的 edge differentiation。
这个效应在全部 ROI 中非常稳定。
```

但这个结果本身不是 YY 特异，因为模型的 `high` 主效应是在 KJ reference 下估计，且下面的 condition-specific contrast 显示 KJ 和 YY 中都存在同方向效应。

### 1.3 condition-specific high-low contrast：KJ 和 YY 均显著

在每个 ROI 内计算：

```text
KJ high - KJ low
YY high - YY low
YY[high-low] - KJ[high-low]
```

结果如下：

| contrast | ROI 数 | FDR 显著 ROI | estimate range | 最小 q_BH | 解释 |
|---|---:|---:|---:|---:|---|
| KJ high - low | 18 | 18 | 1.2145 to 1.9702 | 0.0000 | KJ 内 high pre similarity items 有更强 post separation |
| YY high - low | 18 | 18 | 1.2471 to 2.2003 | 0.0000 | YY 内 high pre similarity items 也有更强 post separation |
| YY high-low - KJ high-low | 18 | 0 | -0.1097 to 0.2301 | 0.0850 | 没有稳定 YY-specific 加成 |

`did_yy_minus_kj_high_low` 中最接近显著的 ROI 为：

| ROI | network | estimate | p | q_BH |
|---|---|---:|---:|---:|
| meta_R_temporal_pole | semantic | 0.2301 | 0.0583 | 0.0850 |
| meta_R_precuneus | hpc_spatial | 0.1632 | 0.0910 | 0.1293 |
| meta_R_PPA_PHG | hpc_spatial | -0.1097 | 0.1723 | 0.2386 |

这些结果不能作为主证据，只能作为后续探索线索。

### 1.4 YY-specific interaction

关键交互项：

```text
C(condition, Treatment(reference='kj'))[T.yy]:
C(pre_tertile, Treatment(reference='low'))[T.high]
```

结果：

```text
n ROI             = 18
FDR-significant   = 0 / 18
estimate range    = -0.1184 to 0.2474
minimum p         = 0.0345
minimum q_BH      = 0.1520
```

因此不能写：

```text
YY 中原本语义/神经距离更近的词对更难分化，
或 YY 对 high-pre-similarity items 有特异分化加成。
```

可以写：

```text
Across individual ROIs, pre-existing pair similarity strongly predicted post-stage edge differentiation in both YY and KJ. However, this relationship was not selectively amplified in YY, indicating that pre-existing similarity shaped post-stage geometry as a general relational-learning constraint rather than a metaphor-specific differentiation cost.
```

### 1.5 ROI 级小结

本分析支持：

```text
pre-existing pair geometry strongly constrains post-stage edge differentiation.
```

但不支持：

```text
YY-specific semantic-distance cost
YY-specific high-pre-similarity differentiation boost
```

因此在论文中应作为材料/表征初始状态的边界分析，而不是隐喻特异机制。

---

## 2. ROI 级主结果二：learning understand/like bridge

### 2.1 分析目的

本分析回答：

```text
run3 理解、run4 喜欢是否预测 post-stage separation？
run3 理解、run4 喜欢是否直接预测后续记忆？
post edge differentiation 在控制 learning behavior 后是否仍与 memory 有关？
```

ROI 级脚本：

```text
nc_converge/e2_all_roi_learning_bridge.py
```

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/e2_all_roi_learning_bridge/
```

输入与样本：

```text
input rows                = 35280
prepared rows             = 35280
n ROI                     = 18
learning behavior merge   = ok
run3_understand non-null  = 34812
run4_like non-null        = 34488
memory non-null           = 21672
```

核心输出：

```text
roi_understand_like_to_post_separation.tsv
roi_understand_like_to_post_pair_similarity.tsv
roi_understand_like_to_memory_direct.tsv
roi_understand_like_to_memory_with_neural.tsv
```

---

### 2.2 learning behavior 不稳定预测 post-stage separation

模型：

```text
post_edge_differentiation_z ~
    condition * (run3_understand_yes + run4_like_yes)
  + pre_pair_similarity_z
  + RT / fluency / material covariates
```

关键 learning terms：

| term | ROI 数 | FDR 显著 ROI | estimate range | 最小 p | 最小 q_BH |
|---|---:|---:|---:|---:|---:|
| run3_understand_yes | 18 | 0 | -0.2658 to 0.1502 | 0.0281 | 0.2514 |
| run4_like_yes | 18 | 0 | -0.0694 to 0.0754 | 0.1604 | 0.7144 |
| YY × run3_understand_yes | 18 | 0 | -0.1101 to 0.2787 | 0.0680 | 0.4270 |
| YY × run4_like_yes | 18 | 0 | -0.1241 to 0.1109 | 0.1362 | 0.7130 |

同时，`pre_pair_similarity_z` 在该模型中 18/18 个 ROI 显著正向预测 post edge differentiation：

```text
term             = pre_pair_similarity_z
FDR-significant  = 18 / 18
estimate range   = 0.6591 to 0.7349
minimum q_BH     = 0.0000
```

解释：

```text
post separation 更强地受 pre-existing pair similarity 约束，
而不是由 run3 理解或 run4 喜欢直接解释。
```

---

### 2.3 learning behavior 不稳定预测 post pair similarity

模型：

```text
post_pair_similarity_z ~
    pre_pair_similarity_z
  + condition * (run3_understand_yes + run4_like_yes)
  + RT / fluency / material covariates
```

关键结果：

| term | ROI 数 | FDR 显著 ROI | estimate range | 最小 p | 最小 q_BH |
|---|---:|---:|---:|---:|---:|
| run3_understand_yes | 18 | 0 | -0.2090 to 0.3853 | 0.0275 | 0.4388 |
| run4_like_yes | 18 | 0 | -0.1083 to 0.0969 | 0.1285 | 0.8674 |
| YY × run3_understand_yes | 18 | 0 | -0.3507 to 0.1784 | 0.1056 | 0.8674 |
| YY × run4_like_yes | 18 | 0 | -0.1105 to 0.1761 | 0.1281 | 0.8674 |

解释：

```text
run3 理解与 run4 喜欢既不能稳定预测 post edge differentiation，
也不能稳定预测 post pair similarity。
```

因此不应写：

```text
学习阶段理解/喜欢直接导致 post-stage neural separation。
```

---

### 2.4 learning behavior 直接预测 memory，但方向是 condition-dependent

模型：

```text
memory_strict ~
    condition * (run3_understand_yes + run4_like_yes)
  + pre_pair_similarity_z
  + RT / fluency / material covariates
```

ROI 级 direct memory model 的关键结果：

| term | ROI 数 | FDR 显著 ROI | estimate range | 最小 p | 最小 q_BH | 解释 |
|---|---:|---:|---:|---:|---:|---|
| YY main effect | 18 | 18 | 2.6075 to 2.6651 | 3.51e-97 | 9.47e-95 | YY memory advantage 稳定 |
| run3_understand_yes | 18 | 18 | 0.3322 to 0.4024 | 7.07e-06 | 3.47e-05 | KJ baseline 下正向预测 memory |
| run4_like_yes | 18 | 0 | -0.0961 to -0.0503 | 0.3284 | 0.4661 | run4 喜欢主效应不稳定 |
| YY × run3_understand_yes | 18 | 18 | -0.7267 to -0.6712 | 3.38e-08 | 4.33e-07 | YY 中 run3 理解的额外效应为负 |
| YY × run4_like_yes | 18 | 18 | -0.5531 to -0.4527 | 5.04e-05 | 1.55e-04 | YY 中 run4 喜欢的额外效应为负 |
| pre_pair_similarity_z | 18 | 0 | -0.1310 to 0.1085 | 0.0862 | 0.1428 | pre similarity 不稳定预测 memory |

解释需要非常谨慎。这个结果说明：

```text
learning behavior 与 subsequent memory 有关，
但不是 YY-specific positive learning benefit。
```

尤其是：

```text
run3_understand_yes 是 KJ baseline 下的正向预测因子；
YY × run3_understand_yes 为负；
YY × run4_like_yes 也为负。
```

因此不能写成：

```text
YY 中理解越好或越喜欢，记忆越好。
```

更稳妥写法：

```text
Learning-stage subjective responses were directly associated with later memory, but these associations were condition-dependent and did not provide a simple YY-specific positive learning pathway.
```

---

### 2.5 控制 learning behavior 后，post edge differentiation 仍预测 memory

模型：

```text
memory_strict ~
    condition * (run3_understand_yes + run4_like_yes)
  + post_edge_differentiation_z
  + pre_pair_similarity_z
  + RT / fluency / material covariates
```

关键结果：

| term | ROI 数 | FDR 显著 ROI | estimate range | 最小 p | 最小 q_BH |
|---|---:|---:|---:|---:|---:|
| YY main effect | 18 | 18 | 2.5500 to 2.6660 | 6.25e-97 | 1.72e-94 |
| run3_understand_yes | 18 | 18 | 0.2722 to 0.4164 | 3.42e-06 | 1.79e-05 |
| YY × run3_understand_yes | 18 | 18 | -0.7417 to -0.6675 | 1.85e-08 | 2.53e-07 |
| YY × run4_like_yes | 18 | 18 | -0.5308 to -0.4530 | 1.05e-04 | 3.45e-04 |
| post_edge_differentiation_z | 18 | 4 | -0.0336 to 0.2619 | 0.0027 | 0.0072 |

通过 FDR 的 `post_edge_differentiation_z` ROI：

| ROI | network | estimate | p | q_BH |
|---|---|---:|---:|---:|
| meta_L_PPC_SPL | hpc_spatial | 0.2619 | 0.0027 | 0.0072 |
| meta_R_PPC_SPL | hpc_spatial | 0.2126 | 0.0135 | 0.0271 |
| meta_R_precuneus | hpc_spatial | 0.2022 | 0.0170 | 0.0330 |
| meta_R_pMTG_pSTS | semantic | 0.1947 | 0.0207 | 0.0400 |

解释：

```text
post-stage edge differentiation 与 memory 的桥接不是全 ROI 均匀分布，
而是主要集中在 bilateral PPC/SPL、right precuneus 和 right pMTG-pSTS。
```

这个结果比单纯 network composite 更有写作价值，因为它给出更清楚的 ROI-level locus：

```text
hpc-spatial / scene-context system 中的 PPC-precuneus 成分，
以及 semantic/metaphor system 中的 right pMTG-pSTS，
是 post edge differentiation 与 subsequent memory 之间最稳定的 ROI 级连接点。
```

---

## 3. ROI 级主结果三：pre-post-retrieval trajectory decomposition

### 3.1 分析目的

原 D2 只计算了 voxel-level trajectory cosine，例如：

```text
return_to_pre_cos
continued_diff_cos
differentiation_axis_cos
```

本轮新增 E3 将 `post -> retrieval` 位移分解为两个部分：

```text
return component:
post -> retrieval 中沿 post -> pre 轴的成分

new-state component:
post -> retrieval 中不沿 post -> pre 轴的正交成分
```

这样可以回答：

```text
retrieval rebound 是简单回到 pre，
还是包含新的 task-driven reconstruction？
```

ROI 级脚本：

```text
nc_converge/e3_all_roi_trajectory_decomposition.py
```

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/e3_all_roi_trajectory_decomposition/
```

输入与样本：

```text
input rows      = 35282
prepared rows   = 32760
n ROI           = 18
n networks      = 2
status          = ok
```

### 3.2 ROI 描述性结果：retrieval 同时有 return 和 new-state 成分

ROI 级 condition descriptives 的范围如下：

| condition | return_to_pre_cos | continued_diff_cos | new_state_fraction | return_fraction_abs | return_new_log_ratio | differentiation_axis_cos |
|---|---:|---:|---:|---:|---:|---:|
| KJ | 0.4196 to 0.4905 | -0.4905 to -0.4196 | 0.7312 to 0.7872 | 0.4842 to 0.5535 | -0.6511 to -0.3689 | -0.4876 to -0.4374 |
| YY | 0.4260 to 0.4908 | -0.4908 to -0.4260 | 0.7293 to 0.7831 | 0.4810 to 0.5543 | -0.6420 to -0.3575 | -0.4979 to -0.4372 |

解释：

```text
return_to_pre_cos 为正，说明 post -> retrieval 位移有朝向 pre 的回拉成分。
continued_diff_cos 为负，说明 retrieval 不是继续沿 pre -> post differentiation 方向前进。
new_state_fraction 大约在 0.73 到 0.79，说明 post -> retrieval 位移中还有大量不沿 post -> pre 轴的正交成分。
```

因此，最稳妥的几何解释是：

```text
retrieval 不是简单保持 post differentiation，
也不是完整恢复 pre geometry；
它包含 partial return-to-pre pull，同时包含 substantial orthogonal/new-state reconstruction。
```

### 3.3 ROI 模型结果：没有 YY-specific 或 memory-specific trajectory effect

模型：

```text
trajectory_metric ~ condition * memory_strict + covariates
```

关键 focal terms：

```text
C(condition, Treatment('kj'))[T.yy]
memory_strict
C(condition, Treatment('kj'))[T.yy]:memory_strict
```

结果汇总：

| outcome | term | ROI 数 | FDR 显著 ROI | estimate range | 最小 p | 最小 q_BH |
|---|---|---:|---:|---:|---:|---:|
| return_to_pre_cos | YY main | 18 | 0 | -0.0617 to 0.0558 | 0.2027 | 0.9670 |
| return_to_pre_cos | memory | 18 | 0 | -0.0922 to 0.0311 | 0.0190 | 0.2349 |
| return_to_pre_cos | YY × memory | 18 | 0 | -0.0759 to 0.0572 | 0.1574 | 0.9120 |
| new_state_fraction | YY main | 18 | 0 | -0.0387 to 0.0356 | 0.2278 | 0.9670 |
| new_state_fraction | memory | 18 | 0 | -0.0087 to 0.0429 | 0.0704 | 0.6153 |
| new_state_fraction | YY × memory | 18 | 0 | -0.0280 to 0.0393 | 0.2299 | 0.9670 |
| return_new_log_ratio | YY main | 18 | 0 | -0.1894 to 0.2104 | 0.3065 | 0.9670 |
| return_new_log_ratio | memory | 18 | 0 | -0.2788 to 0.0625 | 0.0598 | 0.5367 |
| return_new_log_ratio | YY × memory | 18 | 0 | -0.3015 to 0.1200 | 0.1509 | 0.9120 |
| differentiation_axis_cos | YY main | 18 | 0 | -0.0193 to 0.1023 | 0.0522 | 0.4849 |
| differentiation_axis_cos | memory | 18 | 0 | -0.0182 to 0.0570 | 0.0926 | 0.6539 |
| differentiation_axis_cos | YY × memory | 18 | 0 | -0.0539 to 0.0594 | 0.3007 | 0.9670 |

解释：

```text
trajectory decomposition 可以支持“retrieval 包含重构成分”的描述，
但不能支持“YY 比 KJ 更 new-state reconstruction”，
也不能支持“remembered YY items 有更强 return/new-state trajectory”的结论。
```

因此本部分应写成边界机制：

```text
Retrieval-stage rebound should not be interpreted as a simple reinstatement of pre-learning geometry. ROI-level trajectory decomposition indicated both a partial return-to-pre component and a substantial orthogonal component, but these trajectory components were not selectively amplified for YY or remembered items after FDR correction.
```

---

## 4. Network 级辅助结果

本节只作为 ROI 级主结果的辅助验证。当前写作中不建议再把 network composite 作为唯一主结果，因为 ROI 级结果能更具体地说明效应来源。

### 4.1 C1b network-level pre-similarity tertile

network 级输入：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/c1b_pre_similarity_tertile/
```

主要结果：

| network | outcome | high vs low estimate | p | q_BH |
|---|---|---:|---:|---:|
| hpc_spatial | post_edge_differentiation_z | 1.5789 | 8.01e-127 | 1.76e-125 |
| semantic | post_edge_differentiation_z | 1.6592 | 1.08e-130 | 4.76e-129 |

condition-specific contrast：

| network | contrast | estimate | p | q_BH |
|---|---|---:|---:|---:|
| hpc_spatial | KJ high-low | 1.5184 | 0.0000 | 0.0000 |
| hpc_spatial | YY high-low | 1.4896 | 0.0000 | 0.0000 |
| hpc_spatial | YY high-low - KJ high-low | -0.0288 | 0.7706 | 0.7706 |
| semantic | KJ high-low | 1.6410 | 0.0000 | 0.0000 |
| semantic | YY high-low | 1.7220 | 0.0000 | 0.0000 |
| semantic | YY high-low - KJ high-low | 0.0810 | 0.4221 | 0.5065 |

network 级结论与 ROI 级一致：

```text
high pre similarity -> stronger post separation
but no YY-specific DiD
```

---

### 4.2 D1 network-level learning bridge

network 级输入：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/d1_learning_understand_like_bridge/
```

#### post separation model

关键 learning terms 均未通过 FDR：

| network | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| hpc_spatial | run3_understand_yes | -0.0555 | 0.6423 | 0.9788 |
| semantic | run3_understand_yes | 0.0111 | 0.9358 | 0.9788 |
| hpc_spatial | run4_like_yes | -0.0062 | 0.9153 | 0.9788 |
| semantic | run4_like_yes | 0.0239 | 0.6916 | 0.9788 |
| hpc_spatial | YY × run3_understand_yes | 0.1062 | 0.4535 | 0.9077 |
| semantic | YY × run3_understand_yes | 0.0043 | 0.9788 | 0.9788 |
| hpc_spatial | YY × run4_like_yes | 0.0036 | 0.9644 | 0.9788 |
| semantic | YY × run4_like_yes | -0.0875 | 0.3195 | 0.7988 |

但 `pre_pair_similarity_z` 显著预测 post separation：

| network | estimate | p | q_BH |
|---|---:|---:|---:|
| hpc_spatial | 0.6913 | 0.0000 | 0.0000 |
| semantic | 0.7299 | 0.0000 | 0.0000 |

#### direct memory model

| network | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| hpc_spatial | YY main | 2.6286 | 1.19e-94 | 3.56e-93 |
| semantic | YY main | 2.6203 | 6.06e-94 | 9.10e-93 |
| hpc_spatial | run3_understand_yes | 0.3662 | 4.31e-05 | 1.29e-04 |
| semantic | run3_understand_yes | 0.3867 | 1.57e-05 | 6.74e-05 |
| hpc_spatial | YY × run3_understand_yes | -0.6904 | 1.55e-07 | 7.77e-07 |
| semantic | YY × run3_understand_yes | -0.7058 | 8.38e-08 | 5.03e-07 |
| hpc_spatial | YY × run4_like_yes | -0.5089 | 1.89e-04 | 5.15e-04 |
| semantic | YY × run4_like_yes | -0.4917 | 3.12e-04 | 7.80e-04 |

这里同样需要按 condition-dependent relationship 解释：`run3_understand_yes` 是 KJ baseline 下的正向项，而 `YY × run3_understand_yes` 为稳定负向，因此不能写成 YY 中理解越好记忆越好。

#### memory model with post edge differentiation

| network | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| hpc_spatial | post_edge_differentiation_z | 0.3010 | 3.29e-04 | 9.56e-04 |
| semantic | post_edge_differentiation_z | 0.1500 | 0.0634 | 0.0922 |

network 级结果说明：

```text
hpc-spatial network 的 post edge differentiation 与 memory bridge 更稳定；
semantic network 为同方向但未达 FDR。
```

ROI 级结果进一步显示，这个 bridge 主要来自 `meta_L_PPC_SPL`、`meta_R_PPC_SPL`、`meta_R_precuneus` 与 `meta_R_pMTG_pSTS`。

---

### 4.3 Network-level trajectory decomposition

network aggregation 的描述性结果：

| network | condition | n | return_to_pre | continued | new_state | return_abs | return_new_log_ratio | diff_axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hpc_spatial | KJ | 9100 | 0.4553 | -0.4553 | 0.7630 | 0.5188 | -0.5045 | -0.4682 |
| hpc_spatial | YY | 9100 | 0.4653 | -0.4653 | 0.7542 | 0.5236 | -0.4797 | -0.4748 |
| semantic | KJ | 7280 | 0.4496 | -0.4496 | 0.7567 | 0.5192 | -0.5076 | -0.4633 |
| semantic | YY | 7280 | 0.4663 | -0.4663 | 0.7437 | 0.5308 | -0.4542 | -0.4730 |

network-level focal terms 均未通过 FDR：

| outcome | term | hpc_spatial q_BH | semantic q_BH | 解释 |
|---|---|---:|---:|---|
| return_to_pre_cos | YY main | 0.9775 | 0.9775 | no YY effect |
| return_to_pre_cos | memory | 0.9775 | 0.9775 | no memory effect |
| new_state_fraction | YY main | 0.9775 | 0.9848 | no YY effect |
| new_state_fraction | memory | 0.9775 | 0.9775 | no memory effect |
| return_new_log_ratio | YY main | 0.9775 | 0.9977 | no YY effect |
| differentiation_axis_cos | YY main | 0.9775 | 0.9775 | no YY effect |

结论：

```text
network 级 trajectory 与 ROI 级 trajectory 一致：
retrieval 有 return-to-pre pull 和 new-state component，
但没有稳定 YY-specific 或 memory-specific trajectory signature。
```

---

## 5. 对当前论文故事的影响

### 5.1 可以加强的部分

本轮 ROI 级结果可以加强以下说法：

```text
1. pre-existing pair geometry strongly constrains post-stage edge differentiation.
2. post-stage edge differentiation remains linked to subsequent memory after accounting for learning behavior.
3. memory-related post edge differentiation is not uniformly distributed across all ROIs, but is strongest in PPC/precuneus and right pMTG-pSTS.
4. retrieval-stage geometry is not a simple reinstatement of pre-learning geometry; it contains both return-to-pre and orthogonal/new-state components.
```

尤其值得进入主文的是：

```text
post_edge_differentiation_z -> memory_strict
FDR-significant ROIs:
meta_L_PPC_SPL
meta_R_PPC_SPL
meta_R_precuneus
meta_R_pMTG_pSTS
```

这可以作为“分化后的 relation-edge 更容易被记住”的 ROI 级行为桥接证据。

### 5.2 必须弱化的部分

本轮结果不支持：

```text
1. 原始 pre similarity 越高，YY 越难记住。
2. YY 中 high-pre-similarity items 有特异的 post separation 加成。
3. run3 理解或 run4 喜欢直接导致 post-stage separation。
4. YY 的 retrieval trajectory 比 KJ 更明显进入 new-state reconstruction。
5. remembered YY items 有更强 return-to-pre 或 new-state trajectory。
```

### 5.3 推荐写法

中文：

> ROI 级分析显示，学习前已经更相似的词对在学习后更容易表现出 post-stage edge differentiation，这一效应在 18 个 ROI 中均稳定存在，但并非 YY 特异。学习阶段的理解与喜欢反应不能稳定预测 post-stage separation 或 post pair similarity，说明主观学习反应不能解释 post-stage neural differentiation。相反，在控制 learning behavior 和材料协变量后，post edge differentiation 仍在 bilateral PPC/SPL、right precuneus 和 right pMTG-pSTS 中正向预测后续记忆。trajectory decomposition 进一步显示 retrieval 阶段并不是简单恢复 pre-learning geometry，而是同时包含 return-to-pre 成分和较大的 orthogonal/new-state component；不过这些轨迹成分没有表现出 YY 或 memory 特异性。

英文：

> Across individual ROIs, pre-existing pair similarity robustly predicted post-stage edge differentiation in both YY and KJ, but this relationship was not selectively amplified for YY. Learning-stage understand/like responses did not reliably predict post-stage separation or post pair similarity, indicating that subjective learning responses did not explain post-stage neural differentiation. In contrast, post-stage edge differentiation remained positively associated with subsequent memory in bilateral PPC/SPL, right precuneus, and right pMTG-pSTS after accounting for learning behavior and material covariates. Trajectory decomposition further suggested that retrieval-stage geometry was not a simple reinstatement of pre-learning geometry, but contained both a partial return-to-pre component and a substantial orthogonal/new-state component; however, these trajectory components were not selectively amplified for YY or remembered items.

---

## 6. 最终结论

本轮 append_518 的 ROI 级主结果应服务于以下故事：

```text
隐喻学习的关键证据不是学习阶段理解/喜欢直接带来 post separation，
也不是 retrieval 阶段简单恢复 pre geometry。

更稳妥的机制链是：
pre-existing pair geometry constrains post-stage differentiation;
post-stage edge differentiation in specific ROI loci predicts subsequent memory;
retrieval later reconfigures the relation representation with both return and new-state components,
but this trajectory signature is not YY-specific.
```

因此当前主线可以写成：

> 隐喻学习并不是简单增强两个概念之间的相似性，也不是由学习阶段主观理解/喜欢直接驱动 post-stage separation。更稳妥的解释是，学习前已有的 pair geometry 约束了学习后的 edge differentiation，而在 PPC/precuneus 与右 pMTG-pSTS 中更强的 post-stage edge differentiation 与后续记忆表现相联系。提取阶段的表征变化包含部分 return-to-pre 与明显 new-state reconstruction 成分，但这种轨迹结构目前应作为 retrieval-stage reconfiguration 的边界证据，而不是 YY 特异或记忆特异机制。

---

## 7. 追加分析：learning-stage activation / connectivity SME 与 pooled run3 检验

本节是在文献检索和代码复核后更新的补充分析。目标不是扩展主线，而是回答一个具体问题：

```text
run3 理解与 run7 记忆有关，
但 learning-stage local RSA 没有稳定机制；
那么 learning 阶段是否存在 activation、beta-series connectivity，
或者 pooled behavioral-response 层面的 post-stage separation 预测信号？
```

新增脚本：

```text
nc_converge/e4_learning_activation_sme.py
nc_converge/e5_learning_beta_connectivity_sme.py
nc_converge/e6_focused_learning_to_post_memory_rois.py
nc_converge/e7_pooled_run3_understand_post_edge.py
```

新增输出：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/e4_learning_activation_sme/
E:/python_metaphor/paper_outputs/qc/nc_converge/e5_learning_beta_connectivity_sme/
E:/python_metaphor/paper_outputs/qc/nc_converge/e6_focused_learning_to_post_memory_rois/
E:/python_metaphor/paper_outputs/qc/nc_converge/e7_pooled_run3_understand_post_edge/
```

代码复核没有发现 ID 合并错位、volume-metadata 不匹配、condition_item_id 混淆或 YY/KJ 被误合并这类硬 bug。需要注意的是，部分表格中的 `q_bh` 是脚本对整张模型表全部 term 的统一校正，不等同于“只针对 focal term family”的校正；同时 e5/e6/e7 有较多 mixed model fallback，因此这些结果应作为 sensitivity / exploratory，而不是主机制证据。

---

### 7.1 E4：learning-stage ROI activation SME

E4 直接从源 4D pattern 中提取 learning-stage ROI mean beta：

```text
source:
E:/python_metaphor/pattern_root/sub-*/learn_yy.nii.gz
E:/python_metaphor/pattern_root/sub-*/learn_kj.nii.gz

rows        = 70560
n ROI       = 18
n subjects  = 28
n failures  = 0
duplicate key rows = 0
```

#### 7.1.1 Activation SME: activation ~ condition * memory

模型：

```text
activation_z_subject_roi_run ~ condition * memory_strict
                             + run3/run4 behavior covariates
                             + material covariates
```

结果：

| focal term | n terms | FDR significant | minimum p | minimum q_BH |
|---|---:|---:|---:|---:|
| memory_strict | 36 | 0 | 0.0426 | 0.7035 |
| YY × memory_strict | 36 | 0 | 0.0066 | 0.4349 |

解释：learning-stage activation 本身没有形成稳定的 remembered-vs-forgotten SME。这与前面的 RSA 结果一致：learning 阶段没有稳定的 item-specific neural trace。

#### 7.1.2 Memory-from-activation: memory ~ condition * activation

模型：

```text
memory_strict ~ condition * activation_z_subject_roi_run
              + condition * run3_understand_yes / run4_like_yes
              + material covariates
```

该模型中，learning activation 有若干 ROI/run 层面的 memory-predictive signal：

| focal term | n terms | FDR significant | minimum p | minimum q_BH |
|---|---:|---:|---:|---:|
| activation_z_subject_roi_run | 36 | 7 | 0.0011 | 0.0029 |
| YY × activation_z_subject_roi_run | 36 | 5 | 0.0011 | 0.0030 |

通过 FDR 的 activation main effects：

| ROI | run | network | estimate | p | q_BH |
|---|---:|---|---:|---:|---:|
| meta_R_PPC_SPL | 4 | hpc_spatial | -0.3177 | 0.0011 | 0.0029 |
| meta_L_precuneus | 3 | hpc_spatial | 0.2794 | 0.0022 | 0.0060 |
| meta_R_temporal_pole | 4 | semantic | -0.2312 | 0.0146 | 0.0250 |
| meta_R_hippocampus | 3 | hpc_spatial | 0.2159 | 0.0193 | 0.0319 |
| meta_L_temporal_pole | 3 | semantic | 0.2206 | 0.0199 | 0.0327 |
| meta_L_precuneus | 4 | hpc_spatial | -0.2221 | 0.0215 | 0.0351 |
| meta_L_temporal_pole | 4 | semantic | -0.2135 | 0.0309 | 0.0498 |

`YY × activation` 通过 FDR 的 ROI/run：

| ROI | run | network | estimate | p | q_BH |
|---|---:|---|---:|---:|---:|
| meta_R_precuneus | 4 | hpc_spatial | 0.5591 | 0.0011 | 0.0030 |
| meta_L_temporal_pole | 3 | semantic | -0.3629 | 0.0110 | 0.0200 |
| meta_L_precuneus | 4 | hpc_spatial | 0.2896 | 0.0279 | 0.0455 |
| meta_R_PPC_SPL | 4 | hpc_spatial | 0.2874 | 0.0292 | 0.0474 |
| meta_L_IFG | 4 | semantic | 0.2757 | 0.0299 | 0.0484 |

同时，memory model 中再次出现 learning behavior 与 memory 的 condition-dependent relationship：

```text
run3_understand_yes:
FDR-significant = 18 / 18 ROI
estimate range = 0.2256 to 0.2449
minimum q_BH = 0.0142

YY × run3_understand_yes:
FDR-significant = 18 / 18 ROI
estimate range = -0.8344 to -0.7056
minimum q_BH = 5.61e-10

YY × run4_like_yes:
FDR-significant = 18 / 18 ROI
estimate range = -0.6016 to -0.5066
minimum q_BH = 4.36e-05
```

这里的 `run3_understand_yes` 主效应应理解为 KJ baseline 下的正向关系；因为 YY interaction 为稳定负向，不能写成“YY 中理解越好，记忆越好”。更稳妥的解释是：learning-stage activation carried limited memory-predictive information, but did not form a consistent subsequent-memory activation signature.

---

### 7.2 E5：learning-stage beta-series connectivity SME

E5 使用 E4 的 ROI activation table，在每个 `subject × run × condition × memory_strict` 内，以 item 为 beta series 计算 ROI-pair correlation，并 Fisher-z 转换。

```text
activation rows     = 70560
connectivity rows   = 23256
n ROI pairs         = 153
minimum items/group = 5
finite connectivity = 100%
```

模型：

```text
connectivity_z ~ condition * memory_strict
```

focal terms 结果：

| term | n pair/run terms | FDR significant | minimum p | minimum q_BH |
|---|---:|---:|---:|---:|
| YY main | 306 | 0 | 0.0056 | 0.1062 |
| memory_strict | 306 | 0 | 0.0091 | 0.1431 |
| YY × memory_strict | 306 | 0 | 0.0041 | 0.0850 |

最接近显著的 `YY × memory_strict` connectivity 为：

| run | ROI pair | pair family | estimate | p | q_BH |
|---:|---|---|---:|---:|---:|
| 4 | meta_L_AG__meta_R_precuneus | cross_network_memory_locus | 1.3421 | 0.0041 | 0.0850 |
| 4 | meta_L_PPA_PHG__meta_R_precuneus | within_network_memory_locus | 1.3682 | 0.0108 | 0.1570 |
| 4 | meta_R_PPA_PHG__meta_R_precuneus | within_network_memory_locus | 1.0322 | 0.0154 | 0.2063 |

代码复核发现该模型有大量 mixed-model fallback：`1224` 个系数中 `1060` 个为 `fallback_after_mixed_failure: Singular matrix`。因此，E5 只能说明 learning-stage beta-series connectivity 没有提供稳定 FDR 证据，不能作为当前主机制；接近显著的 cross-network / memory-locus edges 只能作为 future direction。

---

### 7.3 E6：focused learning-to-post bridge in memory-linked ROIs

E6 只检验 E2 中已经发现 `post_edge_differentiation -> memory` 的 4 个 ROI：

```text
meta_L_PPC_SPL
meta_R_PPC_SPL
meta_R_precuneus
meta_R_pMTG_pSTS
```

```text
rows  = 7840
n ROI = 4
duplicate key rows = 0
```

#### 7.3.1 Learning behavior / activation -> post edge differentiation

模型：

```text
post_edge_differentiation_z ~
    condition * (run3_understand_yes + run4_like_yes
               + run3_activation_z + run4_activation_z)
  + pre_pair_similarity_z
  + covariates
```

结果：

| predictor | ROI 数 | FDR significant ROI | minimum p | minimum q_BH |
|---|---:|---:|---:|---:|
| run3_understand_yes | 4 | 0 | 0.1829 | 0.6043 |
| run4_like_yes | 4 | 0 | 0.3292 | 0.7055 |
| run3_activation_z | 4 | 0 | 0.1592 | 0.5981 |
| run4_activation_z | 4 | 0 | 0.0274 | 0.2083 |
| YY × run3_understand_yes | 4 | 0 | 0.6138 | 0.8971 |
| YY × run4_like_yes | 4 | 0 | 0.1698 | 0.5981 |
| YY × run3_activation_z | 4 | 0 | 0.3234 | 0.7055 |
| YY × run4_activation_z | 4 | 0 | 0.0076 | 0.0818 |

但 `pre_pair_similarity_z` 仍在 4/4 ROI 显著预测 post edge differentiation：

```text
estimate range = 0.6520 to 0.7034
minimum q_BH = 0.0000
```

需要降调的是，该模型 76/76 个系数均为 mixed-model fallback，因此结论应写成 sensitivity result：即便聚焦到 memory-linked ROIs，也没有证据显示 run3 understand 或 learning activation 可以解释 post-stage edge differentiation；post differentiation 仍主要受 pre-existing pair geometry 约束。

#### 7.3.2 Memory model: post edge bridge remains after learning behavior and activation

模型：

```text
memory_strict ~
    condition * (run3_understand_yes + run4_like_yes
               + run3_activation_z + run4_activation_z)
  + post_edge_differentiation_z
  + pre_pair_similarity_z
  + covariates
```

关键结果：

| term | ROI 数 | FDR significant ROI | estimate range | minimum p | minimum q_BH |
|---|---:|---:|---:|---:|---:|
| run3_understand_yes | 4 | 4 | 0.2834 to 0.4432 | 9.39e-07 | 6.75e-06 |
| YY × run3_understand_yes | 4 | 4 | -0.7059 to -0.6498 | 1.08e-07 | 1.23e-06 |
| YY × run4_like_yes | 4 | 4 | -0.5704 to -0.4649 | 3.40e-05 | 1.43e-04 |
| post_edge_differentiation_z | 4 | 4 | 0.1977 to 0.2591 | 0.0025 | 0.0075 |
| run3_activation_z | 4 | 0 | 0.0112 to 0.1396 | 0.1622 | 0.2448 |
| run4_activation_z | 4 | 1 | -0.3548 to -0.1893 | 0.0003 | 0.0010 |

`post_edge_differentiation_z` 在 4 个 ROI 中全部保持显著：

| ROI | network | estimate | p | q_BH |
|---|---|---:|---:|---:|
| meta_L_PPC_SPL | hpc_spatial | 0.2591 | 0.0025 | 0.0075 |
| meta_R_precuneus | hpc_spatial | 0.2230 | 0.0074 | 0.0198 |
| meta_R_PPC_SPL | hpc_spatial | 0.2112 | 0.0126 | 0.0305 |
| meta_R_pMTG_pSTS | semantic | 0.1977 | 0.0165 | 0.0366 |

这里最重要的解释不是“run3 understand 正向预测 YY memory”，而是：

```text
run3_understand_yes 是 KJ baseline 下的正向预测因子；
YY × run3_understand_yes 为稳定负向；
合成 YY slope 后不支持 YY 理解越好、记忆越好。
```

因此，E6 支持的是“双入口”而不是中介链：learning-stage behavioral response 与 memory 有 condition-dependent 关系；post-stage edge differentiation 也预测 memory；但 run3 understand / learning activation 不能解释 post edge differentiation。

---

### 7.4 E7：去掉 condition 后，run3 理解是否与 post edge differentiation 相关？

E7 回答一个更直接的问题：如果不放 YY/KJ condition，直接把两组 pooled 在一起，`run3_understand_yes` 是否能预测 `post_edge_differentiation_z`？

模型分两种：

```text
simple:
post_edge_differentiation_z ~ run3_understand_yes

adjusted:
post_edge_differentiation_z ~ run3_understand_yes
                            + run4_like_yes
                            + pre_pair_similarity_z
                            + RT/material covariates
```

网络层面结果完全不显著：

| level | model | network | estimate | p |
|---|---|---|---:|---:|
| network | simple | semantic | -0.0572 | 0.5706 |
| network | simple | hpc_spatial | -0.0492 | 0.5768 |
| network | adjusted | semantic | 0.0031 | 0.9684 |
| network | adjusted | hpc_spatial | 0.0020 | 0.9766 |

ROI 层面结果显示：simple pooled model 中只有 `meta_R_hippocampus` 通过 run3-term family FDR：

```text
meta_R_hippocampus:
estimate = -0.2387
p = 0.0020
run3-term family q = 0.0367
```

但它的方向是负向，并且 adjusted model 后不稳定：

```text
meta_R_hippocampus adjusted:
estimate = -0.1599
p = 0.0136
run3-term family q = 0.1222
```

adjusted model 中 `meta_L_precuneus` 的表内 `q_bh` 为 0.0287，但这是对整张模型表全部 term 的校正；如果只在 18 个 ROI 的 run3 focal term 内做 family FDR，结果为：

```text
meta_L_precuneus adjusted:
estimate = 0.2018
p = 0.0033
run3-term family q = 0.0594
```

因此，E7 的结论是：去掉 condition 后，run3 理解与 post-stage edge differentiation 没有稳定正相关。memory-linked focused ROI 中也没有显著关系。

---

### 7.5 对“learning 脑机制没发现”的最终解释

新增 E4-E7 后，当前证据更清楚：

```text
1. learning activation SME: activation ~ memory 没有稳定 FDR 证据。
2. learning activation -> memory 有若干 ROI/run 信号，但方向不统一。
3. learning beta-series connectivity SME 没有 FDR 稳定证据，且 mixed-model fallback 很多。
4. run3 understand / learning activation -> post edge differentiation 没有稳定证据。
5. 去掉 condition 后，run3 understand 仍不能稳定预测 post edge differentiation。
6. post edge differentiation -> memory 在 4 个 memory-linked ROI 中仍稳定。
7. run3 understand 与 memory 的关系是 condition-dependent，不是 YY-specific positive pathway。
```

因此，不应强行写成：

```text
run3 understand -> learning neural mechanism -> post separation -> memory
```

更适合写成：

```text
Learning-stage behavioral responses are condition-dependent behavioral markers,
but the neural consequence of learning is more clearly detectable as
post-stage edge differentiation rather than as an immediate learning-stage
local RSA, activation, or connectivity signature.
```

中文写法：

> run3 理解与后续记忆有关，但这种关系受 condition 调节，并不能解释为 YY 中理解越好、记忆越好。当前数据也没有显示 run3 理解通过 learning-stage 局部 RSA、activation 或 beta-series connectivity 直接转化为 post-stage separation。相反，后续记忆相关的神经证据主要出现在 post 阶段的 edge differentiation，尤其集中在 bilateral PPC/SPL、right precuneus 和 right pMTG-pSTS。
