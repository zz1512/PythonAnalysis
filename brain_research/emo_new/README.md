# emo_new（重构版）

这个版本按你的 5 点要求实现为**单脚本流程**：
- trial type 固定：`Passive_Emo` / `Reappraisal` / `Passive_Neutral`
- 用 `lss_main` 输出的 `lss_index_surface_L_aligned.csv` 和 `lss_index_surface_R_aligned.csv` 对齐 trial beta
- 仅保留 L/R 都可用的被试交集
- 每种 trial type 保存被试全脑平均结果
- 构建 `(surface_L + surface_R) × 被试 × 被试` 相似性矩阵（年龄升序）
- 输出同顺序的 3 个发育模型矩阵（`M_nn/M_conv/M_div`）

## 运行

> 参数都在脚本顶部配置区，默认无需命令行参数。

```bash
python brain_research/emo_new/build_trialtype_means_and_similarity.py
```

## 主要输出

输出根目录（默认）：
`/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/emo_new_trialtype_wholebrain`

- `mean_betas/<trial_type>/<sub>/mean_beta_surface_L.gii`
- `mean_betas/<trial_type>/<sub>/mean_beta_surface_R.gii`
- `mean_betas/<trial_type>/<sub>/mean_beta_wholebrain_lr_concat.npy`
- `similarity/similarity_wholebrain_LR_<trial_type>_AgeSorted.csv`
- `dev_models/M_nn_<trial_type>_AgeSorted.csv`
- `dev_models/M_conv_<trial_type>_AgeSorted.csv`
- `dev_models/M_div_<trial_type>_AgeSorted.csv`
- `meta/subject_order_<trial_type>_AgeSorted.csv`
- `meta/subject_intersection_LR.csv`
