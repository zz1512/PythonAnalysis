# Bug / Risk Log (metaphoric/final_version)

本文件用于记录 pipeline 中已确认或高风险的缺陷点，便于后续逐一修复与回归验证。

## 1. Model-RSA Trial 对齐缺口（M2/M3/M7 可能静默失效）

### 1.1 缺 `word_label`：M3/M7 无法对齐（高优先级）
- **现象/风险**：
  - `model_rdm_comparison.py` 的 M3（embedding）与 M7（memory）依赖 `metadata.tsv` 中的 `word_label` 列。
  - 但 `stack_patterns.py` 生成的 `*_metadata.tsv` 由 `lss_metadata_index_final.csv` 透传，当前只保证 `unique_label/pic_num/condition/...`，不保证有 `word_label`。
  - 结果是：M3/M7 会在 `model_rdm_subject_metrics.tsv` 里大量出现 `skip_reason=missing_*`，表面上“跑完了”，但核心模型没有真正参与分析（静默失败）。
- **证据**：
  - `stack_patterns.py` 依赖 `lss_metadata_index_final.csv` 并输出 `*_metadata.tsv`：`representation_analysis/stack_patterns.py`
  - `model_rdm_comparison.py` 默认 `--word-col word_label`：`rsa_analysis/model_rdm_comparison.py`
- **影响范围**：Model-RSA（M3、M7）、偏相关、Δρ LMM。
- **建议修复**：
  - 在 LSS 阶段或 `stack_patterns.py` 后处理阶段，把 events/刺激表中的“实际中文词”合并写回 `*_metadata.tsv`：
    - `metadata = metadata.merge(events[['unique_label','word_label']], on='unique_label')`
  - 或提供一致的 ID：让 embedding/memory 表也支持用 `unique_label` 作为对齐键（需要同步修改预计算脚本的 key）。
- **建议最小 QC**：
  - 对任一被试任一条件的 metadata：`assert 'word_label' in metadata.columns`

### 1.2 缺 `pair_id`：M2 目前依赖 `pic_num` 回退（高优先级）
- **现象/风险**：
  - M2（配对身份）理应使用 `pair_id`；当前实现若无 `pair_id` 会回退到 `pic_num`。
  - 若实验材料中“同一对的两个 concept word”并不共享同一 `pic_num`（或 1 个 pic_num 对应多于 2 个 word），则 M2 会退化甚至完全失效，但不会明确报错。
- **证据**：
  - M2 回退逻辑：`rsa_analysis/model_rdm_comparison.py` 中 `model_from_pair_identity`
  - `unique_label` 的构造被记录为 `trial_type + pic_num`：`readme_detail.md`
- **影响范围**：Model-RSA 的核心学习证据（M2）、偏相关、Δρ LMM。
- **建议修复**：
  - 在 metadata 中显式写入 `pair_id`（推荐），来源可为刺激材料表（pair-level mapping）。
  - 若短期无法补 `pair_id`，至少在跑 M2 前做强制断言：每个 `pic_num` 必须恰好出现 2 次（对应同一对的两个 word）。
- **建议最小 QC**：
  - `assert 'pair_id' in metadata.columns or metadata.groupby('pic_num').size().eq(2).all()`

### 1.3 Neural RDM vs Model RDM 顺序对齐缺少断言（中优先级）
- **现象/风险**：
  - 当前 `model_rdm_comparison.py` 通过“同一循环顺序拼接 samples 与 metadata”来隐式保证 trial 对齐。
  - 若未来有人调整拼接逻辑或上游写盘顺序变化，可能出现 samples 与 metadata 行顺序错位，结果数值会变得不可解释但不一定报错。
- **影响范围**：所有模型（M1/M2/M3/M7）与 neural RDM 的相关值。
- **建议修复**：
  - 增加硬性断言：
    - `assert samples.shape[0] == len(metadata)`
  - 在 `stack_patterns.py` 写盘前也断言：
    - `assert pattern_array.shape[0] == len(metadata_out)`

## 2. Δρ LMM 的随机效应结构说明（中优先级）
- **现象/风险**：
  - 目标写法常表述为 `(1|subject) + (1|roi)` 的交叉随机效应；但 statsmodels 对真正 crossed RE 支持有限。
  - 当前实现采用 `groups=subject` + `vc_formula={'roi': '0 + C(roi)'}`，更接近“roi 作为 subject 内的方差成分近似”，语义可能被误读。
- **当前处理**：
  - `delta_rho_lmm_model.json` 已记录 `converged` 与 `random_effects_note`，用于提醒该近似。
- **建议**：
  - 若后续要提交顶刊并被要求严格 crossed RE，可考虑迁移到 R `lme4` 或 `pymer4` 重跑验证。

