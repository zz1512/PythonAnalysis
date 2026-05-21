#!/usr/bin/env python3
"""Run the final NC convergence freeze layer in dependency order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "s0_build_master_long_table.py",
    "s1_stage_trajectory_model.py",
    "s2_memory_component_model.py",
    "s3_mahalanobis_robustness_proxy.py",
    "s4_activation_pattern_quality_control.py",
    "s5_input_output_transformation.py",
    "s6_global_similarity_centrality.py",
    "s7_material_moderation.py",
    "s8_learning_update_proxy.py",
    "s9_network_representational_coupling.py",
    "s10_hpc_long_axis_gate_summary.py",
    "s11_fdr_tiering_evidence_table.py",
    "s12_make_final_figures.py",
    "s13_append_final_section.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = root / script
        print(f"[final-nc] running {script}")
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()

