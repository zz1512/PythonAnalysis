#!/usr/bin/env python3
"""Task 4: Mahalanobis robustness proxy, explicitly not strict crossnobis."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_text, write_tsv

MODULE = "s3_mahalanobis_robustness_proxy"


def main() -> None:
    pair = read_table(old_nc("b2a_mahalanobis_from_patterns", "mahalanobis_pair_distance.tsv"))
    summary = read_table(old_nc("b2a_mahalanobis_from_patterns", "mahalanobis_step5c.tsv"))
    comp = read_table(old_nc("b2_crossnobis_robustness", "crossnobis_step5c.tsv"))
    write_tsv(pair, final_out(MODULE, "mahalanobis_proxy_pair_metrics.tsv"))
    write_tsv(summary, final_out(MODULE, "mahalanobis_proxy_network_summary.tsv"))
    write_tsv(comp, final_out(MODULE, "mahalanobis_vs_correlation_comparison.tsv"))
    rows = comp[comp["roi_set"].astype(str).eq("network_composite")].copy()
    rows["strict_crossnobis_available"] = False
    rows["method_label"] = "covariance-normalized Mahalanobis robustness proxy"
    write_tsv(rows, final_out(MODULE, "strict_crossnobis_gate.tsv"))
    write_text("Strict item-level crossnobis is not available; use Mahalanobis robustness proxy wording only.\n", final_out(MODULE, "mahalanobis_method_boundary.md"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_b2_mahalanobis_proxy.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_mahalanobis_proxy_summary.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_mahalanobis_proxy_summary.pdf"))


if __name__ == "__main__":
    main()

