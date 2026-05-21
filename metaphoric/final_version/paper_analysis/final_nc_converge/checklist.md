# Final NC Converge Checklist

- [x] 所有脚本只读上游 paper_outputs，不覆盖旧结果
- [x] 所有 item-level mixed model 使用 condition_item_id
- [x] 所有 difference-score 结果都完成 component decomposition
- [x] memory_ord / memory_strict / memory_lenient 均完成敏感性分析
- [x] material covariates 已 z-score 并进入关键模型
- [x] mean activation / pattern quality control 已完成
- [x] Mahalanobis proxy 不误写成 strict crossnobis
- [x] input-output transformation 若 slope 不显著，不使用 transformation 强表述
- [x] global centrality 若 memory effect 不稳，只写 refinement
- [x] novelty moderation 若不稳，只写边界检查
- [x] hpc long-axis 若只有 MNI fallback，只写 exploratory
- [x] MDS 图不作为统计证据
- [x] KJ 全文写成 active spatial-relational comparison
- [x] 不写完整 causal mediation chain
- [x] 不写 pure retrieval reinstatement explains memory
- [x] 输出 final_evidence_tier_table.tsv
- [x] 输出 main figure manifest

验收输出：

- `E:/python_metaphor/paper_outputs/qc/final_nc_converge/s11_fdr_tiering_evidence_table/final_evidence_tier_table.tsv`
- `E:/python_metaphor/paper_outputs/qc/final_nc_converge/s12_make_final_figures/figure_manifest.tsv`
