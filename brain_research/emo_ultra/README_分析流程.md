# emo_ultra 分析流程

## 目标
在 `emo`（偏保守）与 `emo_new`（偏敏感）之间，提供一个折中的新流程：
- 先在每个 ROI 内构建刺激×刺激表征矩阵（RSM）
- 再做被试间相似性（ISC）
- 再做发育模型置换检验（模型内 FWER，正向单尾）

## 运行顺序
1. `build_roi_repr_matrix_232.py`
2. `calc_roi_isc_by_age.py`
3. `joint_analysis_roi_isc_dev_models_perm_fwer.py`
4. `plot_roi_isc_dev_models_perm_fwer_sig.py`

## data_check
- `data_check/check_stage1_repr_matrix.py`
- `data_check/check_stage2_isc.py`
- `data_check/check_stage3_perm_results.py`

> 建议每一步后都运行对应 `data_check` 脚本，确认维度、被试排序与显著性统计数量。
