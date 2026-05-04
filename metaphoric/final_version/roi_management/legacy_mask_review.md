# Legacy Mask Review: H:/metaphor/mask vs Meta ROI

最后更新：2026-05-03

## 结论

师兄的 `H:/metaphor/mask` 不是一套严格区分 metaphor / spatial 的 mask library，而是一组混合来源 ROI：语义控制/整合节点、空间-记忆-情境节点、海马细分、若干探索性 sphere，以及一个 vision Neurosynth map。新版主分析不建议照师兄那样全部放在一起做一个大 family；更合理的做法是继续把主故事分成 `meta_metaphor` 与 `meta_spatial`，但在图和解释中允许它们共同支持 relation-learning / memory-binding 机制。

## 师兄 mask 内容审计

审计文件：

```text
E:/python_metaphor/roi_library/meta_sources/legacy_mask_file_audit.tsv
E:/python_metaphor/roi_library/meta_sources/legacy_vs_meta_mask_overlap.tsv
E:/python_metaphor/roi_library/meta_sources/legacy_vs_meta_nearest_centroid.tsv
```

主要发现：

- 语义/隐喻相关：`IFG_L/R`、`MFG_L/R`、`angular_L/R`、`IPL_L/R`。
- 空间/记忆/情境相关：`Hippocampus_L/R`、`aHPC/pHPC`、`rostral/caudal hippocampus`、`parahippocampal_gyrus_L/R`、`PCC_L/R`、`RSC_L/R`、`Precuneus_L/R`。
- 探索性或视觉相关：`sphere_*`、`vision_association-test_z_FDR_0.01.nii`。
- 多数 atlas mask 是 2 mm MNI `91x109x91`；部分 rostral/caudal hippocampus 是 3 mm `61x73x61` 或需要重采样；`aHPC_L`、`pHPC_L` 文件较大但实际空间仍可读。

## 与新版 meta ROI 的关系

新版 `meta_metaphor` 已覆盖师兄语义核心：

- `meta_L_IFG` 完全落在师兄 `IFG_L` 内。
- `meta_R_IFG` 完全落在师兄 `IFG_R` 内。
- `meta_L_AG` 与师兄 `angular_L` 有明确重叠。
- `meta_R_AG` 几乎完全落在师兄 `angular_R` 内。

新版 `meta_spatial` 已覆盖师兄空间-记忆核心：

- `meta_L/R_hippocampus` 与师兄 hippocampus / aHPC / rostral-caudal hippocampus 高度一致。
- `meta_L/R_PPA_PHG` 与师兄 parahippocampal gyrus 有同源关系，但新版是 Neurosynth parahippocampal peak sphere，更小、更偏 peak。
- `meta_L/R_PPC_SPL` 与师兄 IPL/PPC 相关，但新版更偏 spatial-attention/SPL peak；师兄 IPL 是较大的 atlas 区域。
- `meta_L/R_precuneus` 与师兄 precuneus 同源，但新版是小 sphere。
- `meta_L/R_RSC_PCC` 与师兄 RSC 有同源关系；对师兄 PCC mask 的覆盖较弱。

## 是否应该把隐喻和空间放一起分析

不建议作为主分析完全合并。

原因：

- 师兄混合分析适合 exploratory network / whole-brain gPPI 时代，但现在老师要求主线来自外部 meta-analysis ROI。
- 如果把 metaphor 与 spatial 全部合并，FDR family 会变大，解释会变成“泛语义-记忆-空间网络”，削弱隐喻学习的故事。
- 更好的主分析口径是：`meta_metaphor` 与 `meta_spatial` 分开建模、分开报告；再做一个跨 family 的 conjunction / summary，说明 learned-edge differentiation 是否同时出现在 semantic-metaphor network 与 spatial/hippocampal-binding network。

## 相对师兄 mask，新版可参考补充

主 mask 暂不建议加入：

- `MFG_L/R`：师兄有大 MFG mask，但当前主结果没有稳定 region-level 交集；可作为 semantic-control sensitivity，不进主文主 family。
- `OPA/LOC/vision`：师兄有 vision map 和若干视觉/探索 sphere；新版主故事不是视觉处理，建议 backup/SI。

需要重点留意：

- `PCC_L/R`：师兄 PCC 与 left-HPC network 线索重要，但新版 `meta_L/R_RSC_PCC` 更贴近 RSC/navigation peak，对师兄 PCC atlas mask 覆盖较弱。建议在 legacy network confirmation 中把 PCC/Precuneus 作为 planned target family；若老师特别重视海马-PCC，可在 `meta_spatial` 里增加一个 `meta_L/R_PCC` backup 或把 PCC peak 合并进 `meta_L/R_RSC_PCC`。
- `aHPC/pHPC`、`rostral/caudal hippocampus`：师兄做过海马细分。新版主分析先用 L/R hippocampus；海马细分适合 SI 或 network seed sensitivity，不建议先进入主 mask family。

## 新版 mask 是否有文件错误

当前未发现文件层面的错误：

- 18 个 mask 均成功写入 `E:/python_metaphor/roi_library/masks/meta_metaphor` 与 `meta_spatial`。
- 每个 mask 都是 6 mm sphere，2 mm MNI reference 下 `123` voxels。
- shape 均为 `91x109x91`，与当前 reference pattern 对齐。
- 同一 ROI set 内无 mask overlap。

需要在论文口径中说明的不是文件错误，而是来源限制：

- Neurosynth 没有直接的 `metaphor` / `figurative` term analysis；`meta_metaphor` 当前是 semantic/language/abstract proxy map 与当前显著 ROI 的交集。
- `meta_spatial` 更接近 spatial/navigation/hippocampal-binding 网络，不应被写成单纯视觉空间网络。
