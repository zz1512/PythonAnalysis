#!/usr/bin/env python3
"""Task 7: global similarity / representational centrality final wrapper."""

from __future__ import annotations

import shutil

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s6_global_similarity_centrality"


def main() -> None:
    item = read_table(old_nc("c2_global_similarity_centrality", "global_similarity_item.tsv"))
    models = read_table(old_nc("c2_global_similarity_centrality", "global_similarity_models.tsv"))
    memory = models[models["mechanism_model"].astype(str).eq("global_change_by_memory")].copy()
    write_tsv(item, final_out(MODULE, "global_similarity_item_metrics.tsv"))
    write_tsv(models, final_out(MODULE, "global_centrality_models.tsv"))
    write_tsv(memory, final_out(MODULE, "global_centrality_memory_models.tsv"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_c1_c2_c3_boundaries.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_global_similarity_summary.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_global_similarity_summary.pdf"))


if __name__ == "__main__":
    main()

