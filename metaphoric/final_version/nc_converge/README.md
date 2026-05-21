# nc_converge — NC 投稿收束分析

本目录实现当前的 NC 收束工作流：在既有主分析与 `reviewer_supp/` 基础上，把故事线统一到
`stage-specific relation-edge reorganization`，并补齐 network-level 主模型、稳健性检查、
FDR 分层与写作级收束材料。

当前对应 spec：

- `.trae/specs/converge-for-nc-submission/`

所有脚本遵守同一隔离约束：

- 只读上游 `paper_outputs/`、`reviewer_supp/` 与主结果目录，不回写既有分析产物。
- 新产物只写到 `paper_outputs/qc/nc_converge/<module>/`。
- 默认禁止覆盖既有文件；已有文件存在时，要么 `resume` 跳过，要么直接报错退出。
- 主结果文档 `result_new_meta_roi.md` 只允许通过 `append_section28.py` 末尾追加 §28。

---

## 当前主线

- A 线：锁定 `condition_item_id` 与被试交集，避免 stage / run7 / memory bridge 混表。
- A 线：把 post 与 retrieval 分量拆开，重新定义更保守的 memory endpoint。
- B 线：在 `semantic` 与 `hpc_spatial` 两个 network composite 上做 staged path、crossnobis 和 representational coupling。
- D 线：索引或按需触发 `reviewer_supp`，但 Reviewer family 不回灌主表。
- 收束：通过 `a6_fdr_tiering.py`、`go_no_go.py` 与 `append_section28.py` 把结果转成投稿决策与文稿段落。

---

## 目录与状态

| 脚本 | 模块 | 当前作用 |
| --- | --- | --- |
| `shared_nc.py` | 公共工具 | 统一 IO、BH-FDR、network 映射、`condition_item_id` 构造、`fit_formula()`、隔离校验辅助函数。 |
| `a2_schema_lock.py` | `a2_schema_lock` | 审计上游长表的 `condition_item_id` 唯一性，并输出 `subject_intersection.tsv`。 |
| `a3_shared_post_review.py` | `a3_shared_post_review` | 复核 post / retrieval shared component 是否污染主效应；difference-score 只作 sensitivity。 |
| `a4_post_memory_component.py` | `a4_post_memory_component` | 同时输出 memory component 模型、ROI follow-up 与 stage trajectory；当前实现已修正为“仅对显著 network 展开 ROI follow-up”。 |
| `a5_network_composite.py` | `a5_network_composite` | 把 Step5C、edge specificity、retrieval-related 指标聚合到 `semantic` / `hpc_spatial` network composite。 |
| `b1_staged_path.py` | `b1_staged_path` | 做阶段到阶段的直接关联描述，不声称 mediation。 |
| `b2_crossnobis_robustness.py` | `b2_crossnobis_robustness` | 检查 correlation 版结果与 crossnobis / Mahalanobis 版方向和量级是否一致。 |
| `b3_repr_coupling.py` | `b3_repr_coupling` | 检验预定义的 semantic network 与 hpc-spatial network 耦合。 |
| `d_reviewer_supp_index.py` | `d_reviewer_supp_index` | 默认只索引 `reviewer_supp` 产物；加 `--run-reviewer-supp` 时才主动运行 reviewer_supp 流程。 |
| `a6_fdr_tiering.py` | `a6_fdr_tiering` | 对结果做 Tier A / B / C / Reviewer 分层；当前 family 粒度按 `(tier, module, network)` 拆开。 |
| `go_no_go.py` | `go_no_go` | 汇总 A4 / B1 / B2 / B3 的稳定性，输出 GO / HOLD / NO-GO 建议。 |
| `append_section28.py` | `append_section28` | 生成并按需追加 §28“Storyline Reframing (NC target)”。 |
| `isolation_check.py` | `isolation_check` | 输出沙箱内所有文件的 sha256，并检查上游关键路径 git status。 |
| `run_all_nc_converge.py` | 编排器 | 默认顺序为 `A2 → A3 → A4 → A5 → B1 → B2 → B3 → D → A6 → Go/No-Go → ISO → §28`。 |
| `c1_input_output_transformation.py` | `c1_input_output_transformation` | 已实现的补充机制分析；当前不在默认编排器内，属于备用/下一轮分析模块。 |
| `c2_global_similarity_centrality.py` | `c2_global_similarity_centrality` | 已实现的补充机制分析；当前不在默认编排器内。 |
| `c3_novelty_familiarity_moderation.py` | `c3_novelty_familiarity_moderation` | 已实现的补充机制分析；当前不在默认编排器内。 |

---

## 关键输出

- `a2_schema_lock/id_audit.tsv`
- `a2_schema_lock/subject_intersection.tsv`
- `a3_shared_post_review/shared_post_review.tsv`
- `a4_post_memory_component/memory_component_model.tsv`
- `a4_post_memory_component/memory_component_roi_followup.tsv`
- `a4_post_memory_component/stage_trajectory_model.tsv`
- `a5_network_composite/network_composite_summary.tsv`
- `b1_staged_path/stage5_network_trajectory_table.tsv`
- `b2_crossnobis_robustness/crossnobis_step5c.tsv`
- `b3_repr_coupling/repr_coupling.tsv`
- `d_reviewer_supp_index/reviewer_supp_index.tsv`
- `a6_fdr_tiering/fdr_tier_manifest.tsv`
- `go_no_go/go_no_go_stage6.md`
- `append_section28/section28_preview.md`
- `isolation_check/output_manifest.tsv`

---

## 执行建议

```bash
python run_all_nc_converge.py \
  --base-dir /path/to/python_metaphor \
  --paper-output-root /path/to/python_metaphor/paper_outputs
```

- 默认 `resume`：若某模块的预期输出已齐全，则直接跳过。
- 默认跳过 §28 写入；需要文档追加时显式传 `--no-skip-append-section28`。
- 仅想预览 §28 时，可加 `--dry-run-section28`。
- 如果要全量重跑，不建议直接配合旧输出目录使用 `--no-resume`；更稳的是先清空旧的 `paper_outputs/qc/nc_converge`，或指定新的 `--output-root`。

---

## 当前实现备注

- `a4_post_memory_component.py` 已修正：ROI follow-up 只对显著 network 运行，不再“一处显著，全部 network 都展开”。
- `a6_fdr_tiering.py` 当前按 `(tier, module, network)` 作为 family 粒度，避免把异质模型混到同一 BH family。
- `c1/c2/c3` 已落地，但默认流程仍以 A/B/D 线为主；若后续需要纳入主流程，应同步更新 `run_all_nc_converge.py` 与本文档。

## 依赖

- Python 3.10+
- `numpy`
- `pandas`
- `statsmodels`
- `scikit-learn`
- 上游 `paper_outputs/`、`reviewer_supp/` 与主结果目录已就绪
