# hpc\_subfield\_three\_axis — 海马亚区重跑与三轴叙事

本目录实现海马亚区切分、subfield 级 Step5C / edge-specificity / anterior-posterior 检验，
并把这些结果组织成 `Favila / Schlichting / Bein` 三轴叙事，最终服务于
`result_new_meta_roi.md` 的 §29 追加。

所有脚本遵守同一隔离约束：

- 只读上游 `paper_outputs/`、`nc_converge/` 与主结果目录，不改写既有分析输出。
- 新产物只写到 `paper_outputs/qc/hpc_subfield_three_axis/<module>/`。
- 默认禁止覆盖既有文件；已有文件存在时，要么 `resume` 跳过，要么直接报错退出。
- 主结果文档 `result_new_meta_roi.md` 只允许通过 `append_section29.py` 末尾追加 §29。

***

## 当前流程

- `S1`：生成 `hpc_<L|R>_<head|body|tail>` 六个 subROI mask。
- `S2`：在六个 subROI 上提取 item-level multi-voxel beta 向量。
- `S3`：在 subROI 上重做 Step5C 型 `condition × time` RSA。
- `S4`：计算 trained / pseudo / non-edge 的 post-pre drop 对比。
- `S5`：仅对 `trained` edge 的 drop 做 anterior-posterior 检验。
- `V1`：为 `meta_R_hippocampus`、`hpc_R_head/body/tail`、`meta_R_temporal_pole` 画 pre/post MDS。
- `N1`：整合主结果与 S3/S4/S5 统计，生成三轴证据表。

***

## 目录与状态

| 脚本                                       | 模块                               | 当前作用                                                                                                                                                                                |
| ---------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shared_subfield.py`                     | 公共工具                             | 复用 `shared_nc` 的 IO、FDR、mixed model 与 markdown helper；补充 subfield 命名、mask 切分与轴位置工具。                                                                                                 |
| `s1_subfield_segmentation.py`            | `s1_segmentation`                | 生成每个被试的 6 个 subROI mask。当前逻辑是：若检测到 FS60 文件，则在 FS60 标签体积的非零体素空间内按 y 轴三等分；若 FS60 缺失或读取失败，则回退到既有 `meta_L/R_hippocampus` mask 的 MNI y 轴三等分。                                             |
| `s2_subfield_extract_beta.py`            | `s2_beta_extract`                | 读取既有 LSS beta，并为每个 `subject × phase × item × subROI` 落盘 beta 向量。当前 `beta_qc.tsv` 输出 `mean_of_voxel_means`、`mean_of_voxel_vars`、`zero_variance_voxels`；其中零方差体素统计已修正为“跨 item 的逐体素方差”。 |
| `s3_subfield_step5c.py`                  | `s3_step5c_subfield`             | 在每个 subROI 上构造 pair similarity，并拟合 `pair_similarity ~ C(condition) * C(time)`。                                                                                                      |
| `s4_subfield_edge_specificity.py`        | `s4_edge_specificity_subfield`   | 计算 `drop = post - pre`，并跑 C1-C4 四个 trained-edge specificity 对比。                                                                                                                     |
| `s5_subfield_anterior_posterior_test.py` | `s5_anterior_posterior_contrast` | 基于 `axis_position` 跑 anterior-posterior mixed model；当前实现已修正为仅使用 `trained` edge 的 drop。                                                                                              |
| `v1_mds_rhpc_temporal_pole.py`           | `v1_mds_visualisation`           | 为右海马整体、右 head/body/tail 与右 temporal pole 生成 pre/post MDS 图和坐标表。                                                                                                                     |
| `n1_three_axis_evidence_table.py`        | `n1_three_axis_evidence`         | 汇总 S3/S4/S5 和可选主结果表，输出 `three_axis_evidence_table.tsv` 与 markdown 摘要；当前并不直接汇总 V1 图坐标。                                                                                               |
| `append_section29.py`                    | `append_section29`               | 生成并按需追加 §29“Three-Axis Narrative Reframing (Favila / Schlichting / Bein)”。                                                                                                          |
| `isolation_check.py`                     | `isolation_check`                | 输出沙箱内所有文件的 sha256，并检查主结果、`nc_converge` 等上游路径 git status。                                                                                                                            |
| `run_all_hpc_subfield.py`                | 编排器                              | 默认顺序为 `S1 → S2 → S3 → S4 → S5 → V1 → N1 → ISO → §29`。                                                                                                                               |

***

## 三轴对应

- `Favila axis`：主要看 `S3` 的 `condition × time` 交互与 `S4` 的 trained-edge specificity。
- `Schlichting axis`：主要看 `S5` 中 `axis_position × condition` 的 anterior-posterior 梯度。
- `Bein axis`：主要通过主分析里左侧 semantic-control / temporal pole 证据与亚区结果并置，由 `N1` 统一整理；不是由 `S5` 单独“产出 Bein 证据”。

***

## 关键输出

- `s1_segmentation/subfield_manifest.tsv`
- `s2_beta_extract/beta_long.tsv`
- `s2_beta_extract/beta_qc.tsv`
- `s3_step5c_subfield/subfield_step5c.tsv`
- `s3_step5c_subfield/subfield_pair_similarity.tsv`
- `s4_edge_specificity_subfield/subfield_edge_specificity.tsv`
- `s4_edge_specificity_subfield/subfield_drop_long.tsv`
- `s5_anterior_posterior_contrast/anterior_posterior_contrast.tsv`
- `v1_mds_visualisation/mds_coords.tsv`
- `n1_three_axis_evidence/three_axis_evidence_table.tsv`
- `append_section29/section29_preview.md`
- `isolation_check/output_manifest.tsv`

***

## 执行建议

```bash
python run_all_hpc_subfield.py \
  --base-dir /path/to/python_metaphor \
  --paper-output-root /path/to/python_metaphor/paper_outputs \
  --fs60-root /path/to/freesurfer_hipposubfields \
  --mask-root /path/to/meta_roi_masks
```

- 默认 `resume`：若某模块的预期输出已齐全，则直接跳过。
- 默认跳过 §29 写入；需要文档追加时显式传 `--no-skip-append-section29`。
- 仅想预览 §29 时，可加 `--dry-run-section29`。
- 如果要全量重跑，不建议直接配合旧输出目录使用 `--no-resume`；更稳的是先清空旧的 `paper_outputs/qc/hpc_subfield_three_axis`，或指定新的 `--output-root`。

***

## 当前实现备注

- `S1` 已修正为“FS60 可用时优先走 FS60 体素空间”，不再无条件回退到 MNI split。
- `S2` 已修正 `zero_variance_voxels` 的统计逻辑，不再把单个向量里的全部体素误记为零方差。
- `S5` 已修正为只对 `trained` edge 做 anterior-posterior 模型，避免把 `pseudo` 与 `non_edge` 混入轴检验。
- `N1` 当前汇总的是统计证据表；V1 图像结果仍由 `mds_coords.tsv` 与图文件单独承载。

## 依赖

- Python 3.10+
- `numpy`
- `pandas`
- `statsmodels`
- `scikit-learn`
- `matplotlib`
- `nibabel`
- 上游主分析 `paper_outputs/` 与 `nc_converge` 输出已就绪

