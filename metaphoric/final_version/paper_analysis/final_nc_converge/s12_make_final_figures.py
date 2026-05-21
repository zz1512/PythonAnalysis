#!/usr/bin/env python3
"""Task 13: collect final figures into project-local figure_final."""

from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, FINAL_ROOT, final_out, final_qc_root, write_tsv

MODULE = "s12_make_final_figures"

FIGURES = [
    "fig_behavior_accuracy_rt",
    "fig_learning_dynamics_meta_metaphor",
    "fig_learning_dynamics_meta_spatial",
    "fig_step5c_main",
    "fig_edge_specificity",
    "fig_word_stability",
    "fig_four_phase_trajectory_meta_metaphor",
    "fig_four_phase_trajectory_meta_spatial",
    "fig_mechanism_behavior_prediction",
    "fig_relation_behavior_prediction",
    "fig_repr_connectivity",
]


def copy_if_exists(src: Path, dest_dir: Path, rows: list[dict]) -> None:
    if src.exists():
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        rows.append({"figure": src.stem, "source": str(src), "dest": str(dest), "status": "ok"})
    else:
        rows.append({"figure": src.stem, "source": str(src), "dest": "", "status": "missing"})


def main() -> None:
    dest_dir = FINAL_ROOT / "figure_final"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for stem in FIGURES:
        for ext in [".png", ".pdf"]:
            copy_if_exists(DEFAULT_PAPER_OUTPUTS / "figures_main" / f"{stem}{ext}", dest_dir, rows)
    for p in (DEFAULT_PAPER_OUTPUTS / "figures_result_final").glob("fig_result_final_*.*"):
        if p.suffix.lower() in {".png", ".pdf"}:
            copy_if_exists(p, dest_dir, rows)
    for p in (DEFAULT_PAPER_OUTPUTS / "qc" / "hpc_subfield_three_axis" / "v1_mds_visualisation").glob("fig_mds_*.*"):
        if p.suffix.lower() in {".png", ".pdf"}:
            copy_if_exists(p, dest_dir, rows)
    manifest = pd.DataFrame(rows)
    write_tsv(manifest, final_out(MODULE, "figure_manifest.tsv"))
    write_tsv(manifest, dest_dir / "figure_manifest.tsv")


if __name__ == "__main__":
    main()

