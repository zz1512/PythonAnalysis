# Tasks

说明：同事指出 Step2 里 `nulls[:, pi]` 的处理**当前并非确定性 bug**（neuromaps 现版本返回通常为 (200, n_perm)，顺序也与 `L_1..L_100, R_1..R_100` 一致），但确实存在**版本变更导致静默错配**的风险，因此需要加断言与元信息记录（P0）。

- [x] Task 1（P0）: Step2 spin test 形状断言 + 可追溯元信息（必须）
  - [x] 在 `_spin_test()` 增加对 `nulls` 形状的断言/显式分支：
    - [x] 断言 parcel 维度为 200（`nulls.shape[0] == 200` 或等价逻辑）
    - [x] 明确支持的返回形状（例如 2D: (200,n_perm)；如遇 3D/其它则抛出可读错误或降级为 fallback 并标注）
  - [x] 在 `gradient_shift_results.csv` 增加列（每个 condition 一行）：
    - [x] `spin_test_available`（True/False）
    - [x] `spin_test_method`（alexander_bloch / random_permutation）
    - [x] `nulls_shape`（字符串，如 "200x10000"）
    - [x] `spin_test_note`（例如 "exploratory_fallback_no_spatial_autocorr"）
  - [x] 验证：neuromaps 可用/不可用两种环境都能跑通，且 CSV 标注正确

- [x] Task 2（P0）: Step4 行为合并不再静默丢人（必须）
  - [x] 将 `_merge_behavior()` 默认改为 left join（或增加 `--behavior-join {left,inner}`，默认 left；推荐后者以保留可控性）
  - [x] merge 后报告样本量：
    - [x] `n_total_subjects`（有 MI 的被试数）
    - [x] `n_with_behavior`（行为列非空可用于相关的被试数）
    - [x] `n_lost_behavior = n_total_subjects - n_with_behavior`
  - [x] 将上述信息写入 `maturity_index_config.csv`，并在控制台 `warning` 提示（缺失>0 时）
  - [x] 验证：行为缺失>0 时会提示；MI~age 仍基于全样本不受影响

- [x] Task 3（P0）: Step3 网络级 RC 的 `n_roi_pairs`（必须）
  - [x] 在 `repr_connectivity_stats.csv` 增加 `n_roi_pairs`（以及可选 `n_rois_group_a/n_rois_group_b`）
  - [x] 在日志中打印每个网络对的 pair 数（便于审稿/复现）

- [x] Task 4（P1）: Step3 RC 聚合方式（推荐）
  - [x] 增加参数 `--rc-aggregate {mean,median}`（默认 mean，保持兼容）
  - [x] 可选：stats 同时输出 `mean_rc` 与 `median_rc`（便于 supplementary）

- [x] Task 5（P1）: Step1 ΔISC 方向一致性检查（提示型，避免叙事自相矛盾）
  - [x] 增加 `--direction-check` 开关（默认关闭，避免改变旧行为；需要时再启用）
  - [x] 输出：按网络汇总的 `r_obs_delta` 符号/分布摘要，并打印“是否与当前叙事假设冲突”的提示（不阻断）

- [x] Task 6（P1）: Step4 `reappraisal_success` suffix 匹配白名单过滤（推荐）
  - [x] suffix 匹配仅允许列名包含关键词（`rating|emot|score` 等，提供可配置列表或常量）
  - [x] 将匹配来源写入 `maturity_index_config.csv`（explicit_pairs / suffix_match / long_table / unpaired）
  - [x] 验证：存在无关 `_passive/_reappraisal` 列时不会误配

- [x] Task 7（P2）: Step3 复杂性指标增强为 distinctiveness（可选但有意义）
  - [x] 在现有 `offdiag_std_upper` 基础上新增 `pattern_distinctiveness`（更可解释）
  - [x] 输出同时包含两者，便于“成熟化 vs 噪声/退化”排他解释
  - [x] 文档/注释补充两指标的解释边界

- [x] Task 8（P2）: Mahalanobis 协方差 shrinkage 可选项（可选，默认不变）
  - [x] 增加参数 `--cov-estimator {ridge_pinv,ledoit_wolf}`（默认 ridge_pinv）
  - [x] 将 ridge 默认值是否调整为 1e-3 作为可配置项（默认保持旧值，避免改动历史结果）
  - [x] 验证：两种估计器都可运行；输出 schema 不变

- [x] Task 9（P2）: Step2 负对照扩展为多网络（可选）
  - [x] 允许指定负对照抽样网络集合（例如 `--scan-negctl-networks SomMot Vis Limbic`）
  - [x] 输出分网络的 null 分布与经验 p 值（不破坏现有 `scan_negative_control.csv`，可新增文件或新增列）

- [x] Task 10（P2）: run_pipeline.py 集成 Step1-4（可选且默认关闭）
  - [x] 增加配置项/参数 `--run-phase2`（默认 False）
  - [x] 仅在开启时按顺序调用 Step1-4（沿用现有 matrix_dir 与 out_dir 约定）

- [x] Task 11（P0）: 文档同步与验证（必须）
  - [x] 更新 `readall.md`：补充新增列/参数含义（spin_test_available、behavior merge n、n_roi_pairs、rc-aggregate 等）
  - [x] `python -m py_compile` 覆盖相关脚本
  - [x] 最小 mock 数据跑通 Step2/Step4/Step3 新增分支，确认新增 CSV/列存在

# Task Dependencies
- Task 11 depends on Task 1-3；若实现 Task 4-10，也需覆盖对应文档与验证
