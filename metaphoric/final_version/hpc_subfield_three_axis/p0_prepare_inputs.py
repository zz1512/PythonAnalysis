#!/usr/bin/env python3
"""P0: prepare item and pair tables for the hippocampal long-axis workflow."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from shared_subfield import add_common_args, default_config, write_outputs

MODULE = "p0_inputs"
PHASES = ("pre", "learning", "post", "retrieval")
CONDITIONS = ("yy", "kj")
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pattern-root", type=Path, default=None)
    return parser.parse_args()


def template_id(value: object) -> str:
    match = re.search(r"(\d+)\s*$", str(value))
    return match.group(1) if match else ""


def load_metadata(pattern_root: Path) -> pd.DataFrame:
    frames = []
    for subject_dir in sorted(pattern_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        for phase in PHASES:
            for cond in CONDITIONS:
                path = subject_dir / f"{phase}_{cond}_metadata.tsv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path, sep="\t", low_memory=False)
                frame["subject"] = subject_dir.name
                frame["run_phase"] = phase
                frame["condition_code"] = cond
                frame["condition"] = CONDITION_LABELS[cond]
                frame["item_id"] = frame["word_label"].astype(str)
                frame["template_pair_id"] = frame["word_label"].map(template_id)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_items(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame()
    keep = [
        "subject",
        "run_phase",
        "condition",
        "condition_code",
        "item_id",
        "template_pair_id",
        "word_label",
        "real_word",
        "run",
        "beta_path",
    ]
    out = meta[[c for c in keep if c in meta.columns]].copy()
    out = out[out["beta_path"].map(lambda item: Path(str(item)).exists())]
    return out.drop_duplicates(["subject", "run_phase", "condition", "item_id"]).reset_index(drop=True)


def pair_rows_for_subject_condition(sub: pd.DataFrame, subject: str, condition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    templates: list[tuple[str, list[str]]] = []
    phase = "pre" if (sub["run_phase"] == "pre").any() else str(sub["run_phase"].iloc[0])
    base = sub[sub["run_phase"] == phase].copy()
    for template, grp in base.groupby("template_pair_id", dropna=False):
        labels = sorted(set(grp["item_id"].dropna().astype(str)))
        if template and len(labels) == 2:
            templates.append((str(template), labels))
    templates = sorted(templates, key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
    n = len(templates)
    if n < 3:
        return rows
    for idx, (template, labels) in enumerate(templates):
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "template_pair_id": template,
                "item_id_a": labels[0],
                "item_id_b": labels[1],
                "edge_type": "trained",
            }
        )
        next_labels = templates[(idx + 1) % n][1]
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "template_pair_id": f"{template}_pseudo",
                "item_id_a": labels[0],
                "item_id_b": next_labels[1],
                "edge_type": "pseudo",
            }
        )
        far_labels = templates[(idx + max(2, n // 2)) % n][1]
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "template_pair_id": f"{template}_non_edge",
                "item_id_a": labels[0],
                "item_id_b": far_labels[1],
                "edge_type": "non_edge",
            }
        )
    return rows


def build_pairs(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (subject, condition), sub in items.groupby(["subject", "condition"], dropna=False):
        rows.extend(pair_rows_for_subject_condition(sub, str(subject), str(condition)))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    pattern_root = Path(args.pattern_root or cfg.base_dir / "pattern_root")
    meta = load_metadata(pattern_root)
    items = build_items(meta)
    pairs = build_pairs(items)
    manifest = pd.DataFrame(
        [
            {
                "pattern_root": str(pattern_root),
                "metadata_rows": len(meta),
                "item_rows": len(items),
                "pair_rows": len(pairs),
                "n_subjects": items["subject"].nunique() if not items.empty else 0,
                "n_conditions": items["condition"].nunique() if not items.empty else 0,
                "status": "ok" if not items.empty and not pairs.empty else "empty",
            }
        ]
    )
    write_outputs(
        cfg,
        MODULE,
        {
            "hpc_items.tsv": items,
            "hpc_pair_table.tsv": pairs,
            "p0_manifest.tsv": manifest,
        },
    )


if __name__ == "__main__":
    main()
