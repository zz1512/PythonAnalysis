#!/usr/bin/env python3
"""M5: write the network-level theoretical reframing fragment."""

from __future__ import annotations

import argparse
import pandas as pd

from shared import add_common_args, default_config, write_standard_outputs

MODULE = "m5_theoretical_reframing"


def build_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"network": "hippocampus", "preferred_term": "pattern separation", "interpretation": "trained relational edges become more discriminable to reduce interference", "key_refs": "Yassa & Stark 2011; Schlichting et al. 2015; Favila et al."},
        {"network": "PPA/PHG/RSC/PPC", "preferred_term": "scene-context binding reorganization", "interpretation": "spatial/contextual representational structure is reorganized around learned pair context", "key_refs": "Schlichting et al. 2015; Kim et al."},
        {"network": "ATL/AG/IFG/semantic", "preferred_term": "relational semantic restructuring", "interpretation": "semantic control and combinatorial systems encode task-relevant relation states", "key_refs": "Kim et al.; metaphor semantic-control literature"},
    ])


def build_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    table_lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    lines = [
        "# M5 Network-Level Theoretical Reframing",
        "",
        "This fragment is designed for appending to Section 27 only. It does not alter Sections 1-26.",
        "",
        "## Interpretation Table",
        "",
        "\n".join(table_lines),
        "",
        "## Draft Claim",
        "",
        "The core result should be described as relation-edge differentiation rather than simple similarity enhancement. In hippocampal ROIs, this is most naturally framed as pattern separation of newly trained conceptual relations. In scene/spatial-context regions, the same empirical direction is better described as reorganization of contextual binding. In semantic-control and combinatorial regions, differentiation reflects relational semantic restructuring rather than a global collapse of word representations.",
        "",
        "## Citation Placeholders",
        "",
        "- Yassa & Stark (2011): hippocampal pattern separation.",
        "- Schlichting, Mumford & Preston (2015): dissociable integration and separation signatures.",
        "- Favila / Chanales / Kuhl line of work: learning-induced differentiation and memory.",
        "- Kim / Norman / Turk-Browne line of work: differentiation during integration and schema learning.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = default_config(args)
    table = build_table()
    outputs = {"network_reframing_table.tsv": table, "network_reframing.md": build_markdown(table)}
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
