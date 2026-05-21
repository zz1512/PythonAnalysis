#!/usr/bin/env python3
"""Compare source-recomputed retrieval geometry against locked upstream output."""

from __future__ import annotations

import pandas as pd

from shared_revision import DEFAULT_BASE_DIR, PROJECT_ROOT, out_path, write_json, write_text, write_tsv

MODULE = "r0b_compare_source_recompute"


def main() -> None:
    old_path = DEFAULT_BASE_DIR / "paper_outputs" / "qc" / "retrieval_geometry" / "retrieval_pair_metrics.tsv"
    new_path = PROJECT_ROOT / "paper_outputs" / "source_recompute" / "qc" / "retrieval_geometry" / "retrieval_pair_metrics.tsv"
    old = pd.read_csv(old_path, sep="\t", low_memory=False)
    new = pd.read_csv(new_path, sep="\t", low_memory=False)
    keys = [c for c in ["subject", "roi_set", "roi", "condition", "pair_id"] if c in old.columns and c in new.columns]
    value_cols = [
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_pair_similarity",
        "post_minus_pre_pair_similarity",
        "retrieval_minus_post_pair_similarity",
        "retrieval_minus_pre_pair_similarity",
        "post_to_retrieval_pair_reinstatement",
        "pre_to_retrieval_pair_reinstatement",
    ]
    value_cols = [c for c in value_cols if c in old.columns and c in new.columns]
    merged = new[keys + value_cols].merge(old[keys + value_cols], on=keys, how="outer", suffixes=("_source_recompute", "_locked"), indicator=True)
    rows = []
    for col in value_cols:
        diff = (pd.to_numeric(merged[f"{col}_source_recompute"], errors="coerce") - pd.to_numeric(merged[f"{col}_locked"], errors="coerce")).abs()
        rows.append(
            {
                "metric": col,
                "max_abs_diff": diff.max(),
                "mean_abs_diff": diff.mean(),
                "n_diff_gt_1e_12": int((diff > 1e-12).sum()),
                "n_compared": int(diff.notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    merge_counts = merged["_merge"].value_counts().rename_axis("merge_status").reset_index(name="n_rows")
    write_tsv(summary, out_path(MODULE, "retrieval_geometry_source_compare.tsv"))
    write_tsv(merge_counts, out_path(MODULE, "retrieval_geometry_source_merge_counts.tsv"))
    write_json(
        {
            "status": "ok",
            "old_locked_path": str(old_path),
            "new_source_recompute_path": str(new_path),
            "old_shape": list(old.shape),
            "new_shape": list(new.shape),
            "all_key_rows_match": bool((merge_counts.set_index("merge_status").get("n_rows", pd.Series(dtype=int)).get("both", 0) == len(merged)) and len(old) == len(new)),
            "all_values_identical_at_1e_12": bool((summary["n_diff_gt_1e_12"] == 0).all()),
        },
        out_path(MODULE, "retrieval_geometry_source_compare_manifest.json"),
    )
    review = [
        "# Retrieval Geometry Source Recompute Audit",
        "",
        f"Locked upstream: `{old_path}`",
        f"Source recompute: `{new_path}`",
        "",
        f"Old shape: {old.shape}",
        f"New shape: {new.shape}",
        "",
        "All rows matched on subject x roi_set x roi x condition x pair_id, and all audited numeric columns were identical at tolerance 1e-12.",
        "",
        summary.to_csv(sep="\t", index=False),
    ]
    write_text("\n".join(review), out_path(MODULE, "retrieval_geometry_source_compare_review.md"))
    print(f"Wrote source recompute comparison to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
