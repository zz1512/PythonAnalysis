#!/usr/bin/env python3
"""Task r5: summarize MVPA as stage-state evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared_revision import DEFAULT_PAPER_OUTPUTS, fig_path, out_path, read_table, write_tsv

MODULE = "r5_mvpa_stage_state_summary"


SOURCES = [
    ("learning", DEFAULT_PAPER_OUTPUTS / "qc" / "learning_mvpa_decoding" / "learning_mvpa_group_fdr.tsv"),
    ("post", DEFAULT_PAPER_OUTPUTS / "qc" / "prepost_binary_mvpa_decoding" / "prepost_binary_mvpa_group_fdr.tsv"),
    ("prepost_delta", DEFAULT_PAPER_OUTPUTS / "qc" / "prepost_binary_mvpa_decoding" / "prepost_binary_mvpa_post_minus_pre_fdr.tsv"),
    ("retrieval", DEFAULT_PAPER_OUTPUTS / "qc" / "run7_mvpa_decoding" / "run7_mvpa_group_fdr.tsv"),
    ("retrieval_behavior_bridge", DEFAULT_PAPER_OUTPUTS / "qc" / "run7_mvpa_decoding" / "run7_mvpa_evidence_behavior_fdr.tsv"),
]


def stage_label(source: str, analysis_type: str) -> str:
    text = f"{source}:{analysis_type}".lower()
    if source == "learning":
        return "learning"
    if "pre_" in text or ":pre" in text:
        return "pre"
    if "post" in text:
        return "post"
    if source == "retrieval":
        return "retrieval"
    if source == "retrieval_behavior_bridge":
        return "behavior_bridge"
    return source


def load_summary() -> pd.DataFrame:
    rows = []
    for source, path in SOURCES:
        if not path.exists():
            rows.append({"source": source, "status": "missing", "source_path": str(path)})
            continue
        df = read_table(path)
        if df.empty:
            rows.append({"source": source, "status": "empty", "source_path": str(path)})
            continue
        df["source"] = source
        df["source_path"] = str(path)
        if "analysis_type" not in df.columns:
            df["analysis_type"] = source
        df["stage_state"] = df["analysis_type"].map(lambda x: stage_label(source, x))
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False)


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "q_bh_primary_family" in out.columns:
        out["primary_q"] = out["q_bh_primary_family"]
    elif "q_bh_within_analysis" in out.columns:
        out["primary_q"] = out["q_bh_within_analysis"]
    elif "q_bh_within_response" in out.columns:
        out["primary_q"] = out["q_bh_within_response"]
    else:
        out["primary_q"] = pd.NA
    if "mean_accuracy" not in out.columns:
        out["mean_accuracy"] = pd.NA
    if "mean_minus_chance" not in out.columns:
        out["mean_minus_chance"] = pd.NA
    keep = [c for c in ["stage_state", "source", "roi_set", "roi", "analysis_type", "metric", "n_subjects", "mean_accuracy", "mean_minus_chance", "t", "p", "primary_q", "response", "status", "source_path"] if c in out.columns]
    return out[keep].sort_values([c for c in ["stage_state", "primary_q"] if c in keep], na_position="last")


def boundary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [
        ("pre", "pre YY/KJ decoding", "weak / not stable unless q-corrected rows appear"),
        ("learning", "learning YY/KJ decoding", "stage-state evidence; not edge mechanism"),
        ("post", "post YY/KJ decoding", "partial retention, especially temporal pole if q-corrected"),
        ("retrieval", "run7 YY/KJ decoding", "robust retrieval stage-state evidence"),
        ("behavior_bridge", "MVPA-behavior bridge", "boundary unless q-corrected behavior bridge appears"),
    ]
    for stage, claim, interpretation in rules:
        sub = summary[summary["stage_state"].eq(stage)].copy() if "stage_state" in summary.columns else pd.DataFrame()
        qcol = "primary_q" if "primary_q" in sub.columns else None
        n_sig = int((pd.to_numeric(sub[qcol], errors="coerce") < 0.05).sum()) if qcol else 0
        best_q = pd.to_numeric(sub[qcol], errors="coerce").min() if qcol and not sub.empty else pd.NA
        rows.append({"stage_state": stage, "claim": claim, "n_rows": len(sub), "n_q_lt_05": n_sig, "best_q": best_q, "interpretation": interpretation})
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return

    plot = summary.copy()
    if "mean_minus_chance" not in plot.columns:
        return
    plot = plot.dropna(subset=["mean_minus_chance"]).copy()
    plot = plot.sort_values("primary_q").groupby("stage_state", as_index=False).head(3)
    if plot.empty:
        return
    labels = plot["stage_state"].astype(str) + " | " + plot["roi"].astype(str)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(plot))))
    ax.barh(y, plot["mean_minus_chance"], color="#6b5b95", alpha=0.85)
    ax.axvline(0, color="0.7", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean accuracy minus chance")
    ax.set_title("MVPA stage-state summary")
    fig.tight_layout()
    fig.savefig(fig_path("fig_mvpa_stage_state_summary.png"), dpi=200)
    fig.savefig(fig_path("fig_mvpa_stage_state_summary.pdf"))
    plt.close(fig)


def main() -> None:
    raw = load_summary()
    summary = compact_table(raw)
    boundary = boundary_table(summary)
    write_tsv(summary, out_path(MODULE, "mvpa_stage_state_summary.tsv"))
    write_tsv(boundary, out_path(MODULE, "mvpa_boundary_table.tsv"))
    make_figure(summary)
    print(f"Wrote r5 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
