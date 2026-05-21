# append_519 新增分析结果整合：Dm 全景补充、ERS 复核与 subfield/run-level 边界

更新日期：2026-05-20  
执行环境：`C:\Users\dell\anaconda3\envs\PythonAnalysis\python.exe`  
正式输出根目录：`E:/python_metaphor/paper_outputs/qc/nc_converge/`

本文件整理 `append_519.md` 规划的新增分析。写作策略与 `518result.md` 保持一致：统计解释严格以 FDR 后结果为准；未通过 FDR 的 nominal 线索只作为边界信息，不作为主证据。

本轮新增或修复后执行的脚本包括：

```text
nc_converge/a4b_ordinal_dm.py
nc_converge/a4c_multinetwork_composite.py
nc_converge/d1c_per_subject_multivariate_dm.py
nc_converge/d3_encoding_retrieval_similarity_revived.py
nc_converge/d4_subfield_dm.py
nc_converge/d5_run_level_trajectory_dm.py
```

另有两个入口脚本做了代码层面的关键修正：

```text
nc_converge/voxel_pattern_export.py
nc_converge/d5_run_level_trajectory_dm.py
```

---

## 0. 总体结论

append_519 没有推翻 `518result.md` 的主线，而是把若干 reviewer 可能追问的问题进一步封口：

1. **ordinal Dm 与 binary Dm 方向一致，支持 post-stage separation 是关键记忆桥接。**  
   A4b 显示 `memory_successes` 三档评分模型中，`YY × post_pair_similarity_network_z` 在 semantic 与 hpc-spatial 两个 network 均为负且通过 FDR。换句话说，在 YY 中 post pair similarity 越低，后续记忆越好；这与之前“post-stage trained-edge differentiation / separation 预测 memory”的主线一致。

2. **cross-network composite 没有提供比 single-network 更强的 strict-memory 主证据。**  
   A4c 的 strict memory composite focal predictor 没有通过 FDR。lenient memory 下有一个 `YY × PCA1` 信号，但 lenient coding 本身在前文已经被定位为 sensitivity，因此不能升级为主机制。

3. **per-subject 行为回归不支持 learning understand/like 是稳定记忆机制。**  
   D1c 中 `run3_understand_yes` 与 `run4_like_yes` 的个体斜率，在 YY、KJ、pooled 以及 YY-KJ paired contrast 中均未通过 FDR。这与 518result 的结论一致：learning behavior 可以作为 condition-dependent marker，但不能写成 YY-specific positive pathway。

4. **ERS revived 没有补出 memory-moderated reactivation 证据。**  
   D3 的 own-other ERS 在 `condition × memory` 框架中全部未通过 FDR。post-to-retrieval own-other diff 的均值接近 0。因此 §31.5 的 reactivation caveat 应维持为 boundary/null check，而不是机制升级。

5. **HPC subfield Dm 支持“subfield supplementary specificity”，但不是新的 long-axis 主线。**  
   D4 在 pooled subfield 模型中显示 `YY × pair_similarity_z` 为负且通过 FDR，说明海马 subfield 层面也呈现 YY 中较低 pair similarity 与更好记忆相关。tail 的三阶项为正且通过 FDR，表示 tail 相对 body 的负向 YY slope 较弱。这个结果适合写成 supplementary anatomical robustness，不应写成稳定 anterior-posterior gradient。

6. **D5 只支持 post run5/run6 层面的边界检查，不支持 run-level Dm accumulation 主张。**  
   当前源数据中 run3/run4 learning pattern 不是 isolated-word 双词 pair similarity 的同构输入，因此不能与 post run5/run6 用同一公式合并解释。D5 对 post run5/run6 的模型没有任何 FDR 显著项，不能支持 YY-specific run-level Dm 累积。

推荐写法：

> The append_519 analyses further support a conservative account in which post-learning representational separation, rather than learning-stage subjective response or encoding-retrieval reinstatement, provides the most stable memory-linked representational evidence. Ordinal memory modeling preserves the post-similarity/separation pattern, whereas per-subject learning-behavior slopes and own-other ERS do not yield FDR-corrected memory-moderation evidence. Hippocampal subfield analyses provide supplementary anatomical robustness but do not establish a new anterior-posterior mechanism.

---

## 1. 代码与逻辑复核

### 1.1 已修复的问题

本轮执行前发现并修复了以下会影响结果有效性的代码问题：

| 脚本 | 问题 | 修复 |
|---|---|---|
| `voxel_pattern_export.py` | 默认 mask root 指到 `roi_library/meta_sources`，那里主要是 source map，不是 final meta ROI masks | 改为 `roi_library/masks`，并只选 `meta_metaphor` / `meta_spatial` 下的 `meta_*.nii.gz` |
| `voxel_pattern_export.py` | metadata 中 isolated-word label 未正确派生 `condition_item_id` / `word_index` | 从 `word_label` / `unique_label` 派生 pair id 与 word role，避免 pair 错位 |
| `voxel_pattern_export.py` | long-tsv 没有保留 run，D5 无法拆 run5/run6 | 输出 schema 加入 `run` |
| `d3_encoding_retrieval_similarity_revived.py` | 只支持 long-tsv，缺少 NIfTI `pattern_root` fallback | 复用 D2 的 NIfTI loader，直接从 `pattern_root` 计算 pair-mean pattern |
| `d3/d4/d5` | `item_mechanism_table.tsv` 中原始列为 `memory`，脚本期待 `memory_strict/memory_lenient` | 统一派生 `memory_prop`、`memory_strict`、`memory_lenient`、`memory_successes` |
| `d4_subfield_dm.py` | 默认 beta_long 路径写成旧目录 | 改为实际存在的 `s2_beta_extract/beta_long.tsv` |
| `d4_subfield_dm.py` | subfield `item_id` 如 `yyew_28` 未正确映射到 `yy_28` | 从 `item_id` 抽取数字并重建 `condition_item_id` |
| `d4_subfield_dm.py` | condition 为 `YY/KJ` 大写，模型 reference 写 `kj` | 统一转小写 |
| `d5_run_level_trajectory_dm.py` | merge 后出现 `condition_x/condition_y`，模型找不到 `condition` | merge 时保留 pattern 侧 condition，不从 item table 重复带入 |

### 1.2 当前执行状态

修复后所有核心脚本均通过：

```text
python -m py_compile
```

D4/D5 初跑出现过模型失败输出，已删除对应失败模块目录并重跑。最终 D4/D5 模型状态均为 `ok`。

---

## 2. A4b：三档 ordinal Dm

### 2.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/a4b_ordinal_dm/
```

manifest：

```text
engine        = ordered_model
n_input_paths = 5
n_rows        = 38340
n_subjects    = 28
networks      = hpc_spatial, semantic
```

主要输出：

```text
ordinal_dm_models.tsv
binary_sanity_models.tsv
ordinal_dm_descriptives.tsv
manifest.tsv
```

### 2.2 Ordinal 结果

ordinal outcome 为 `memory_successes`。focal pair-similarity 结果如下：

| network | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| semantic | `YY × post_pair_similarity_network_z` | -0.2085 | 6.43e-11 | 1.71e-10 |
| hpc_spatial | `YY × post_pair_similarity_network_z` | -0.1395 | 9.97e-7 | 1.99e-6 |
| semantic | `YY × retrieval_pair_similarity_network_z` | +0.1082 | 5.66e-4 | 7.15e-4 |
| semantic | `post_pair_similarity_network_z` | +0.0466 | 0.0350 | 0.0420 |
| hpc_spatial | `YY × retrieval_pair_similarity_network_z` | +0.0415 | 0.139 | 0.158 |

核心解释：

`YY × post_pair_similarity_network_z` 为负，说明在 YY 中 post pair similarity 越高，三档记忆评分越低；反过来说，post-stage separation 越强，记忆越好。这个方向与 518result 中的 post edge differentiation bridge 完全一致。

### 2.3 Binary sanity

binary sanity 中 focal 结果也大体一致：

| network | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| semantic | `YY × retrieval_pair_similarity_network_z` | +0.1291 | 2.48e-6 | 4.20e-6 |
| hpc_spatial | `post_pair_similarity_network_z` | -0.0727 | 4.83e-5 | 7.59e-5 |
| hpc_spatial | `retrieval_pair_similarity_network_z` | +0.0669 | 1.94e-4 | 2.85e-4 |
| semantic | `YY × post_pair_similarity_network_z` | -0.1022 | 3.46e-4 | 4.75e-4 |

binary sanity 仍支持 post / retrieval representational similarity 与 memory 有关，但 ordinal 主模型更适合写“评分粒度不改变主结论”。

### 2.4 A4b 小结

A4b 是本轮最稳妥的主线支持项。它不新增机制，而是说明 memory coding 从 binary 换成 ordinal 后，post-stage separation 与 memory 的关系仍保留。

推荐写法：

> Treating memory as an ordinal outcome preserved the key post-learning representational effect: in both semantic and hpc-spatial networks, YY items with lower post-learning pair similarity showed better later memory. Thus, the post-stage separation-memory link is not an artifact of binary memory coding.

---

## 3. A4c：cross-network composite Dm

### 3.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/a4c_multinetwork_composite/
```

manifest：

```text
n_composite_rows = 1820
n_subjects       = 26
a4_network_table = default
```

PCA loadings：

| component | feature | loading | explained variance |
|---|---|---:|---:|
| pca1 | post hpc_spatial | -0.6742 | 0.4248 |
| pca1 | post semantic | -0.6675 | 0.4248 |
| pca1 | retrieval hpc_spatial | -0.2158 | 0.4248 |
| pca1 | retrieval semantic | -0.2309 | 0.4248 |
| pca2 | post hpc_spatial | +0.2119 | 0.3928 |
| pca2 | post semantic | +0.2348 | 0.3928 |
| pca2 | retrieval hpc_spatial | -0.6737 | 0.3928 |
| pca2 | retrieval semantic | -0.6679 | 0.3928 |

PCA1 主要是 post-dominant composite；PCA2 主要是 retrieval-dominant composite。

### 3.2 Composite focal 结果

| outcome | predictor | term | estimate | p | q_BH |
|---|---|---|---:|---:|---:|
| memory_lenient | PCA1 | `YY × composite_score_pca1` | +0.2307 | 0.00748 | 0.0195 |
| memory_lenient | mean | `YY × composite_score_mean` | -0.3337 | 0.0480 | 0.0845 |
| memory_strict | PCA1 | `composite_score_pca1` | +0.0504 | 0.233 | 0.299 |
| memory_strict | PCA1 | `YY × composite_score_pca1` | +0.0347 | 0.572 | 0.626 |
| memory_strict | mean | `YY × composite_score_mean` | +0.0103 | 0.932 | 0.932 |

严格记忆模型没有 focal composite 证据。唯一通过 FDR 的是 lenient memory 下的 `YY × PCA1`，不应升级为主证据。

### 3.3 AIC/BIC 对比

`composite_vs_network_comparison.tsv` 为：

```text
status = missing_model_row
```

原因是可读入的 a4 network table 不含脚本期望的 `__model__` 对比行。因此本轮不能使用 A4c 做 composite-vs-single-network 的 AIC/BIC 结论。

### 3.4 A4c 小结

A4c 不支持“cross-network composite 比 single network 更强”的 strict-memory 主张。它最多作为 supplementary sensitivity：lenient memory 中可能存在 broad composite signal，但这不改变主线。

推荐写法：

> A cross-network composite did not improve the strict-memory account. Although a lenient-memory interaction emerged for the post-dominant PCA component, strict-memory focal predictors were not FDR-significant; therefore, the primary interpretation should remain at the ROI/network-specific post-separation level.

---

## 4. D1c：per-subject understand/like 多元行为回归

### 4.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/d1c_per_subject_multivariate_dm/
```

manifest：

```text
merged_rows          = 1960
n_subjects_raw       = 28
n_subjects_eligible  = 27
n_subjects_dropped   = 1
dropped_subjects     = sub-12
min_trials           = 12
fits_logit           = 21
fits_ols_fallback    = 14
fits_skipped         = 46
```

per-subject fit status：

| status | n |
|---|---:|
| design_rank_deficient | 31 |
| ok | 21 |
| ok_fallback: Singular matrix | 14 |
| single_class_outcome | 12 |
| too_few_rows | 3 |

这说明 D1c 可作为行为斜率 sanity check，但结果需要保守解释。

### 4.2 Group inference

全部 group inference 均未通过 FDR：

| subset/test | predictor | n | mean_beta | p | q_BH |
|---|---|---:|---:|---:|---:|
| KJ one-sample | `run3_understand_yes` | 9 | +5.5812 | 0.1429 | 0.9677 |
| KJ one-sample | `run4_like_yes` | 9 | -2.7249 | 0.3629 | 0.9677 |
| YY one-sample | `run3_understand_yes` | 12 | +0.4311 | 0.9287 | 0.9782 |
| YY one-sample | `run4_like_yes` | 12 | -2.0705 | 0.6448 | 0.9782 |
| pooled one-sample | `run3_understand_yes` | 14 | +0.1068 | 0.9782 | 0.9782 |
| pooled one-sample | `run4_like_yes` | 14 | -0.0883 | 0.6109 | 0.9782 |
| paired YY-KJ | `run3_understand_yes` | 8 | -5.6807 | 0.2422 | 0.9677 |
| paired YY-KJ | `run4_like_yes` | 8 | -0.6988 | 0.8990 | 0.9782 |

### 4.3 D1c 小结

D1c 支持 518result 的降调判断：学习阶段理解/喜欢不能写成稳定正向记忆机制。

推荐写法：

> A per-subject two-stage regression did not reveal stable understand- or like-related memory slopes in YY, KJ, or pooled trials. Thus, learning-stage subjective responses should remain descriptive or condition-dependent behavioral markers rather than a direct memory mechanism.

---

## 5. D3：ERS revived, own-other reactivation

### 5.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/d3_encoding_retrieval_similarity_revived/
```

manifest：

```text
pattern_source       = nifti_pattern_root_pair_mean
n_rows_item          = 32760
n_subjects           = 26
networks             = hpc_spatial, semantic
include_pre_baseline = True
```

行数：

| output | rows |
|---|---:|
| `ers_item.tsv` | 32760 |
| hpc_spatial item rows | 18200 |
| semantic item rows | 14560 |
| condition YY | 16380 |
| condition KJ | 16380 |

### 5.2 描述性结果

post-to-retrieval own-other diff 均值接近 0：

| network | condition | n | mean diff | sd |
|---|---|---:|---:|---:|
| hpc_spatial | KJ | 9100 | -0.0039 | 0.2906 |
| hpc_spatial | YY | 9100 | -0.0011 | 0.2922 |
| semantic | KJ | 7280 | +0.0044 | 0.3149 |
| semantic | YY | 7280 | -0.0002 | 0.3185 |

pre-to-retrieval own-other diff 同样很小：

| network | condition | n | mean diff | sd |
|---|---|---:|---:|---:|
| hpc_spatial | KJ | 9100 | +0.0039 | 0.2669 |
| hpc_spatial | YY | 9100 | +0.0051 | 0.2695 |
| semantic | KJ | 7280 | +0.0076 | 0.2814 |
| semantic | YY | 7280 | +0.0039 | 0.2826 |

### 5.3 Memory-moderation 模型

`ers_models.tsv` 共 288 行，全部 status 为 `ok`，但没有任何 FDR 显著项。

最关键的 post-to-retrieval diff 结果：

| network | memory_var | term | estimate | p | q_BH |
|---|---|---|---:|---:|---:|
| hpc_spatial | memory_strict | `YY` | -0.0143 | 0.184 | 0.947 |
| hpc_spatial | memory_strict | `memory_strict` | +0.0051 | 0.606 | 0.947 |
| hpc_spatial | memory_strict | `YY × memory_strict` | +0.0110 | 0.388 | 0.947 |
| semantic | memory_strict | `YY` | +0.0063 | 0.568 | 0.947 |
| semantic | memory_strict | `memory_strict` | +0.0081 | 0.426 | 0.947 |
| semantic | memory_strict | `YY × memory_strict` | -0.0175 | 0.181 | 0.947 |

### 5.4 Baseline pre-to-retrieval

`ers_baseline_pre_to_retrieval.tsv` 有一个 semantic `ers_pre_to_retrieval_own` intercept 通过 FDR：

| network | outcome | term | estimate | p | q_BH |
|---|---|---|---:|---:|---:|
| semantic | `ers_pre_to_retrieval_own` | Intercept | +0.0271 | 0.000323 | 0.00258 |

这只是整体 own similarity 高于 0 的 baseline，不是 condition 或 memory moderation。

### 5.5 D3 小结

D3 没有补出 reactivation-memory bridge。它应写作对 §31.5 caveat 的闭合：own-other ERS 在当前数据中不支持 YY-specific 或 memory-specific reinstatement 机制。

推荐写法：

> Revisiting encoding-retrieval similarity with an own-minus-other baseline did not yield FDR-corrected memory moderation or YY-specific reinstatement effects. Thus, reinstatement should remain a boundary analysis rather than a central mechanism.

---

## 6. D4：HPC subfield Dm

### 6.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/d4_subfield_dm/
```

manifest：

```text
n_beta_rows = 68880
n_pair_rows = 34440
segments    = body, head, tail
hemispheres = L, R
```

`pair_similarity_long.tsv` 覆盖：

| dimension | levels |
|---|---|
| condition | YY 17220 rows; KJ 17220 rows |
| run_phase | pre 11760; post 11760; retrieval 10920 |
| segment | head/body/tail 各 11480 |
| hemisphere | L/R 各 17220 |

### 6.2 Per-segment model

`subfield_models_per_segment.tsv` 共 108 行，全部 status 为 `ok`。

strict memory 中主要显著项是 condition 主效应与材料协变量；focal `YY × pair_similarity_z` 未通过 FDR：

| segment | hemi | outcome | term | estimate | p | q_BH |
|---|---|---|---|---:|---:|---:|
| body | R | strict | `YY × pair_similarity_z` | -0.1521 | 0.0540 | 0.0729 |
| head | R | strict | `YY × pair_similarity_z` | -0.0856 | 0.269 | 0.320 |
| body | L | strict | `YY × pair_similarity_z` | -0.0793 | 0.312 | 0.366 |

lenient memory 中多个 `YY × pair_similarity_z` 为负且通过 FDR：

| segment | hemi | outcome | term | estimate | p | q_BH |
|---|---|---|---|---:|---:|---:|
| body | L | lenient | `YY × pair_similarity_z` | -0.4277 | 7.16e-5 | 3.09e-4 |
| head | L | lenient | `YY × pair_similarity_z` | -0.2657 | 0.0131 | 0.0313 |
| head | R | lenient | `YY × pair_similarity_z` | -0.2474 | 0.0142 | 0.0333 |
| body | R | lenient | `YY × pair_similarity_z` | -0.2510 | 0.0204 | 0.0438 |
| tail | L | lenient | `YY × pair_similarity_z` | -0.2142 | 0.0373 | 0.0611 |

### 6.3 Pooled three-way model

`subfield_pooled_three_way.tsv` 共 36 行，全部 status 为 `ok`。

关键结果：

| outcome | term | estimate | p | q_BH |
|---|---|---:|---:|---:|
| strict | `YY × pair_similarity_z` | -0.1240 | 1.30e-4 | 3.11e-4 |
| strict | `YY × pair_similarity_z × tail` | +0.1306 | 0.0189 | 0.0377 |
| lenient | `pair_similarity_z` | +0.1651 | 7.08e-9 | 2.55e-8 |
| lenient | `YY × pair_similarity_z` | -0.3558 | 1.21e-15 | 6.24e-15 |
| lenient | `YY × pair_similarity_z × tail` | +0.2198 | 0.00406 | 0.00861 |

解释：

- `YY × pair_similarity_z` 为负，说明在 YY 中 subfield post pair similarity 越低，记忆越好。
- tail 三阶项为正，说明 tail 相对 body 的负向 YY slope 较弱。
- head 三阶项不显著，不能写成稳定 anterior/posterior gradient。

### 6.4 D4 小结

D4 是一个可用的 supplementary anatomical robustness 结果。它支持 hippocampal subfield 层面也存在与 memory 相关的 YY separation pattern，但不能把 story 升级成明确 long-axis gradient。

推荐写法：

> Hippocampal subfield analyses provided supplementary evidence that the YY memory association is compatible with lower post-learning pair similarity across MNI-defined hippocampal segments. However, the segment interaction pattern did not support a clean anterior-posterior gradient; therefore, the subfield result should be framed as anatomical robustness rather than a new mechanistic axis.

---

## 7. D5：post run5/run6 run-level Dm

### 7.1 执行与输入

输出目录：

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/d5_run_level_trajectory_dm/
```

manifest：

```text
pattern_source = post_nifti_stream
n_pair_rows    = 22176
n_subjects     = 28
runs_present   = 5,6
```

重要边界：

原规划希望 run3/run4/run5/run6 全部拆 run。但当前 `pattern_root` 中 run3/run4 是 learning sentence/item trial，不是 post isolated-word 双词 pattern；它们不能与 run5/run6 用同一个 within-pair word similarity 公式合并。为避免构造非同构指标，最终 D5 只对 post run5/run6 做 run-level boundary check。

### 7.2 描述性结果

| network | condition | run | n | mean raw | mean z |
|---|---|---:|---:|---:|---:|
| hpc_spatial | KJ | 5 | 3640 | 0.2524 | +0.0289 |
| hpc_spatial | KJ | 6 | 3640 | 0.2361 | +0.0157 |
| hpc_spatial | YY | 5 | 2520 | 0.2111 | -0.0417 |
| hpc_spatial | YY | 6 | 2520 | 0.2158 | -0.0227 |
| semantic | KJ | 5 | 2912 | 0.2812 | +0.0214 |
| semantic | KJ | 6 | 2912 | 0.2372 | -0.0026 |
| semantic | YY | 5 | 2016 | 0.2461 | -0.0309 |
| semantic | YY | 6 | 2016 | 0.2412 | +0.0038 |

### 7.3 模型结果

`run_level_models.tsv` 共 104 行，全部 status 为 `ok`，无任何 FDR 显著项。

最接近显著的 focal terms：

| network | outcome | model | term | estimate | p | q_BH |
|---|---|---|---|---:|---:|---:|
| hpc_spatial | strict | linear slope | `YY × run_index × memory_strict` | -0.4355 | 0.0193 | 0.328 |
| hpc_spatial | strict | linear slope | `YY × memory_strict` | +2.3509 | 0.0221 | 0.328 |
| semantic | strict | linear slope | `YY × run_index × memory_strict` | -0.3370 | 0.0429 | 0.430 |
| semantic | strict | categorical | `YY × run6 × memory_strict` | -0.3369 | 0.0435 | 0.430 |

这些 nominal signals 均不能作为证据。

### 7.4 D5 小结

D5 不支持 run5/run6 内部存在稳定的 YY-specific run-level Dm slope。由于 run3/run4 learning 与 run5/run6 post 在输入结构上不同构，本节不能回答“learning 阶段是否已经开始累积 pair-level separation”；这个问题应保持为当前数据结构下不可判定。

推荐写法：

> A post-stage run-level check did not reveal FDR-corrected YY-specific memory slopes across run5 and run6. Because learning runs do not provide the same isolated-word pair-similarity structure as post runs, the data do not support a homogeneous run3-run6 accumulation analysis.

---

## 8. 与 518result 的关系

append_519 应作为 518result 后的 robustness layer，而不是新主线。

可以加强：

1. `post-stage separation -> memory` 不是 binary memory artifact，ordinal Dm 仍支持。
2. hippocampal subfield 层面存在与 memory 兼容的 YY separation pattern，可作为 anatomical robustness。
3. D1c 与 D3 进一步说明 learning behavior 与 ERS reinstatement 不能替代 post separation 主线。

必须弱化或避免：

1. 不写“learning understand/like 直接导致 memory”。
2. 不写“own-other ERS 是 YY-specific reactivation bridge”。
3. 不写“cross-network composite 优于 ROI/network-specific evidence”。
4. 不写“run3-run6 连续累积 Dm slope”，因为 run3/run4 与 run5/run6 的输入结构不同构。
5. 不写“海马 long-axis anterior-posterior gradient”，D4 只能支持 subfield supplementary specificity。

---

## 9. 推荐整合段落

中文：

> append_519 的补充分析进一步支持一个保守但更稳固的解释：后续记忆优势主要与学习后表征分离有关，而不是由学习阶段主观理解/喜欢反应或 encoding-retrieval reinstatement 直接驱动。将记忆作为三档 ordinal outcome 建模后，YY 中 post pair similarity 与 memory 的负向关系仍在 semantic 与 hpc-spatial network 中保持，说明该结果不是二值化记忆评分的产物。相反，per-subject learning behavior 回归和 own-other ERS 复核均未提供 FDR 校正后的 memory-moderation 证据。海马 subfield 分析显示 YY 中较低 post pair similarity 与更好记忆的关系可在 head/body/tail 层面作为补充稳健性证据，但没有形成清晰 anterior-posterior gradient。post run5/run6 的 run-level 检查也未显示稳定的 YY-specific slope，因此 run-level accumulation 不能作为当前机制主线。

英文：

> The append_519 analyses further support a conservative post-separation account. Modeling memory as an ordinal outcome preserved the negative YY association between post-learning pair similarity and later memory in both semantic and hpc-spatial networks, indicating that the memory-linked separation effect is not an artifact of binary coding. In contrast, per-subject learning-behavior regressions and own-minus-other encoding-retrieval similarity did not yield FDR-corrected memory-moderation evidence. Hippocampal subfield analyses provided supplementary anatomical robustness for the YY separation-memory pattern, but did not establish a clean anterior-posterior gradient. A post-run-level check across run5/run6 also failed to reveal stable YY-specific slopes, and learning runs were not structurally comparable to post isolated-word pair similarity. Thus, the primary storyline should remain centered on post-learning representational separation rather than learning-stage behavior, reinstatement, or run-level accumulation.

---

## 10. 最终结论

append_519 的最终定位：

```text
GO as supplementary robustness layer.
Do not revise the primary mechanism away from post-stage representational separation.
Do not upgrade ERS, learning behavior, composite, or run-level accumulation to main evidence.
Use D4 subfield results as anatomical robustness, not as a new long-axis mechanism.
```

最稳妥的论文故事仍然是：

```text
learning-stage condition geometry / task state
    -> post-stage trained-edge differentiation / lower post pair similarity
    -> subsequent memory advantage
with retrieval and subfield/run-level analyses serving as boundary and robustness checks.
```
