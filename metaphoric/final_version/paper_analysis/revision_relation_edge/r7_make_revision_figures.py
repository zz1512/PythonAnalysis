#!/usr/bin/env python3
"""Task r7: build revision figure manifest from generated and existing figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from shared_revision import PROJECT_ROOT, REVISION_FIG, out_path, write_tsv

MODULE = "r7_make_revision_figures"


FIGURE_MAP = [
    ("Fig1", "Design + behavior", PROJECT_ROOT / "figure_final" / "fig_behavior_accuracy_rt.png", "existing_final"),
    ("Fig2", "Learning condition geometry + MVPA inset", PROJECT_ROOT / "figure_final" / "fig_learning_dynamics_meta_metaphor.png", "existing_final"),
    ("Fig2_inset", "MVPA stage-state summary", REVISION_FIG / "fig_mvpa_stage_state_summary.png", "revision"),
    ("Fig3", "Post trained-edge differentiation", PROJECT_ROOT / "figure_final" / "fig_step5c_main.png", "existing_final"),
    ("Fig3_control", "Word identity control", PROJECT_ROOT / "figure_final" / "fig_word_stability.png", "existing_final"),
    ("Fig4", "Stage trajectory", PROJECT_ROOT / "figure_final" / "fig_four_phase_trajectory_meta_metaphor.png", "existing_final"),
    ("Fig4_revision", "Trajectory geometry", REVISION_FIG / "fig_pre_post_retrieval_triangle.png", "revision"),
    ("Fig4_vector", "Displacement direction proxy", REVISION_FIG / "fig_displacement_vector_alignment.png", "revision"),
    ("Fig5", "Subsequent memory prior post separation", PROJECT_ROOT / "figure_final" / "fig_result_final_a4_memory_components.png", "existing_final"),
    ("Fig6", "Semantic-hpc network coupling", REVISION_FIG / "fig_network_coupling_revision.png", "revision"),
    ("Fig6_robustness", "Material moderation boundary", REVISION_FIG / "fig_material_moderation_effects.png", "revision"),
    ("S5", "High/low post-separation profile", REVISION_FIG / "fig_high_low_separation_profile.png", "revision"),
]


def main() -> None:
    rows = []
    REVISION_FIG.mkdir(parents=True, exist_ok=True)
    for fig_id, title, src, source_type in FIGURE_MAP:
        exists = src.exists()
        dest = REVISION_FIG / f"{fig_id.lower()}_{src.name}" if exists and source_type == "existing_final" else src
        if exists and source_type == "existing_final":
            shutil.copy2(src, dest)
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, dest.with_suffix(".pdf"))
        rows.append({
            "figure_id": fig_id,
            "title": title,
            "source_type": source_type,
            "source_path": str(src),
            "revision_path": str(dest),
            "exists": exists,
            "use_in_main": fig_id.startswith("Fig"),
        })
    manifest = pd.DataFrame(rows)
    write_tsv(manifest, REVISION_FIG / "figure_revision_manifest.tsv")
    write_tsv(manifest, out_path(MODULE, "figure_revision_manifest.tsv"))
    print(f"Wrote r7 manifest to {REVISION_FIG}")


if __name__ == "__main__":
    main()
