# Tasks

- [x] Task 1: 统一 SCAN 映射配置与离线支持（P0）
  - [x] 在 `utils_network_assignment.py` 中增加可配置参数：`distance_threshold` 统一入口、atlas/centroid 的离线加载选项（本地 NIfTI 或缓存 CSV/NPY）
  - [x] 增加映射质量报告函数：输出每个 SCAN 区域映射距离、网络归属、阈值命中情况与分布摘要
  - [x] 增加 CLI/脚本入口（可复用 step2 或新增轻量入口）用于一键生成 `scan_mapping_qc.csv`
  - [x] 验证：无网络环境下可运行（不触发 nilearn 在线下载）

- [x] Task 2: SCAN 映射稳健性与负对照（P0）
  - [x] 实现阈值敏感性：对 10/15/20/25mm 输出命中 ROI 数、关键统计量（与 r_obs 的汇总）并保存 `scan_mapping_sensitivity.csv`
  - [x] 实现负对照基线（任选其一先做最小版）：
    - [x] 随机 parcel 负对照：从同网络/同数量 parcel 随机抽样，重复 N 次形成基线分布
    - [ ] 随机 MNI 点负对照：在约束脑空间内随机点→最近 parcel 映射
  - [x] 输出：`scan_negative_control.csv` + 经验 p 值摘要（观测 SCAN 效应在随机分布中的位置）
  - [x] 验证：固定 seed 可复现

- [x] Task 3: Step1/Step2 的 SCAN 标注行为改进（P1）
  - [x] 将关键 try/except “静默失败”改为明确 warning（保留不中断主流程）
  - [x] 把 `distance_threshold` 从硬编码改为参数/配置透传（默认仍为 20mm，保持兼容）
  - [x] 验证：旧命令行不变，新参数可选

- [x] Task 4: MI 模板稳定性（Bootstrap + LOO + 阈值敏感性）（P0）
  - [x] 在 `step4_maturity_index.py` 增加：
    - [x] `--mature-age-grid`（例如 16.5 17 17.5 18）输出敏感性对照表
    - [x] `--bootstrap-template B` 生成模板 bootstrap 稳定性输出（MI 的 CI/SD/CV）
    - [x] `--loo-template` 可选，评估单个成熟被试对模板的影响
  - [x] 输出：`maturity_index_stability.csv`、`maturity_index_sensitivity.csv`、必要的图（可选）
  - [x] 验证：小成熟组时给出清晰提示；结果可复现（seed）

- [x] Task 5: 行为锚定增强（P1）
  - [x] 明确行为 success 的计算路径（优先不改动既有行为 pipeline）：
    - [x] 若 `behavior_subject_stimulus_scores.csv` 同刺激可配对：生成被试级 `reappraisal_success`
    - [x] 若不可配对：提供退化定义（如被试级条件均值差）并在输出中标注
  - [x] 让 `step4_maturity_index.py` 支持把 success 作为主 `--behavior-cols` 示例/默认推荐（保持向后兼容）
  - [x] 验证：在缺失列/缺失文件时提示清晰

- [x] Task 6: 表征复杂性/区分度补充分析（P1）
  - [x] 选择一个最小可行指标（例如：RSM 的有效秩/谱熵/平均非对角值分布宽度）并实现 ROI/网络级输出
  - [x] 输出：`repr_complexity_subject_scores.csv`、`repr_complexity_stats.csv`
  - [ ] 在 `readall.md` 增加“排他解释”补充说明与输出引用

- [x] Task 7: 非线性发育建模（P2，可选最小版）
  - [x] 最小版：在 `step3_representational_connectivity.py` 增加 `--dev-modeling` 开关（默认关闭），对每个 pair_label 做线性 vs 二次模型对比（AIC/R2_adj）
  - [x] 输出：`repr_connectivity_dev_model_comparison.csv`（含最佳模型、AIC 差、二次拐点年龄/是否落在年龄范围内）；若有 `--condition-contrast` 则额外输出 `repr_connectivity_dev_model_comparison_<contrast>.csv`
  - [ ] 可选增强：若 statsmodels 可用再支持样条/GAM，否则优雅降级

- [x] Task 8: 文档与复现收尾（P1）
  - [x] 更新 `readall.md`：新增参数、输出文件、如何解释负对照与模板稳定性
  - [x] 增加“确认性 vs 探索性”标记：例如 neuromaps 不可用时的结果定位
  - [x] 统一所有新增输出到 `gradient_analysis/`，不影响 Phase 1 产物

- [x] Task 9: 验证与回归检查（P0）
  - [x] `python -m py_compile` 覆盖所有改动脚本
  - [x] 在一份小规模数据/抽样（或最小 mock）上跑通 Step2/Step4 新增分支，确认输出文件齐全
  - [x] 关键输出 CSV schema 检查（列名稳定、向后兼容）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 5（行为列准备）仅当要把 success 作为默认展示时
- Task 8 depends on Task 1-6
- Task 9 depends on Task 1-8
