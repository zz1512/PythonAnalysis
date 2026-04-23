#!/usr/bin/env python3
"""
plot_activation_representation_dissociation.py

Purpose
- Generate an "activation vs representation" dissociation plot per ROI.
- Activation: ROI-level GLM metric (e.g., mean t within ROI on a contrast t-map).
- Representation: RSA-based delta similarity (post - pre) for yy/kj.

Outputs
- fig_activation_representation_dissociation.png
- table_activation_representation_dissociation.tsv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))


def _default_out_dir() -> Path:
    try:
        from rsa_analysis import rsa_config as cfg  # type: ignore

        base = Path(getattr(cfg, "BASE_DIR"))
        roi_set = str(getattr(cfg, "ROI_SET", "unknown"))
    except Exception:
        base = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "."))
        roi_set = os.environ.get("METAPHOR_ROI_SET", "unknown")

    override = os.environ.get("METAPHOR_FIG_OUT_DIR", "").strip()
    if override:
        return Path(override)
    return base / f"figures_{roi_set}"


def _load_rsa_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Normalize stage labels.
    if "stage" in frame.columns and "time" not in frame.columns:
        frame["time"] = frame["stage"]
    frame["time"] = frame["time"].astype(str).str.strip().str.lower()
    frame["time"] = frame["time"].replace({"pre": "pre", "post": "post", "learn": "learn", "pretest": "pre"})
    frame["time"] = frame["time"].replace({"pre-test": "pre", "post-test": "post", "pre_test": "pre", "post_test": "post"})
    frame["time"] = frame["time"].replace({"pre ": "pre", "post ": "post"})
    frame["time"] = frame["time"].replace({"pre.": "pre", "post.": "post"})
    frame["time"] = frame["time"].replace({"pre": "pre", "post": "post"})

    # run_rsa_optimized.py uses Sim_Metaphor / Sim_Spatial / Sim_Baseline (wide).
    rename = {
        "Sim_Metaphor": "similarity_yy",
        "Sim_Spatial": "similarity_kj",
        "Sim_Baseline": "similarity_baseline",
    }
    for src, dst in rename.items():
        if src in frame.columns and dst not in frame.columns:
            frame[dst] = pd.to_numeric(frame[src], errors="coerce")
    required = {"roi", "time", "similarity_yy", "similarity_kj"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RSA summary missing columns: {sorted(missing)} (file={path})")
    return frame


def _compute_delta_similarity(rsa_summary: pd.DataFrame) -> pd.DataFrame:
    work = rsa_summary.copy()
    work = work[work["time"].isin(["pre", "post"])].copy()
    grouped = (
        work.groupby(["roi", "time"], as_index=False)[["similarity_yy", "similarity_kj"]]
        .mean()
    )
    pivot = grouped.pivot(index="roi", columns="time")
    # MultiIndex columns: (metric, time)
    out = pd.DataFrame({"roi": pivot.index.astype(str)})
    out["delta_similarity_yy"] = (pivot[("similarity_yy", "post")] - pivot[("similarity_yy", "pre")]).to_numpy()
    out["delta_similarity_kj"] = (pivot[("similarity_kj", "post")] - pivot[("similarity_kj", "pre")]).to_numpy()
    return out.reset_index(drop=True)


def _load_activation_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    if "roi" not in frame.columns:
        # allow roi_name
        if "roi_name" in frame.columns:
            frame = frame.rename(columns={"roi_name": "roi"})
        else:
            raise ValueError(f"Activation table must contain roi/roi_name column: {path}")
    if "activation_metric" not in frame.columns:
        # allow mean_t / t_mean
        for alt in ["mean_t", "t_mean", "activation", "t_value"]:
            if alt in frame.columns:
                frame = frame.rename(columns={alt: "activation_metric"})
                break
    if "activation_metric" not in frame.columns:
        raise ValueError(f"Activation table must contain activation_metric (or mean_t/t_mean): {path}")
    frame["activation_metric"] = pd.to_numeric(frame["activation_metric"], errors="coerce")
    return frame[["roi", "activation_metric"]].copy()


def _extract_activation_from_tmap(tmap_path: Path, roi_dir: Path) -> pd.DataFrame:
    import nibabel as nib  # type: ignore

    timg = nib.load(str(tmap_path))
    tdata = np.asanyarray(timg.get_fdata(), dtype=float)
    rows = []
    for roi_path in sorted(roi_dir.glob("*.nii*")):
        roi_name = roi_path.stem.replace(".nii", "")
        mask = np.asanyarray(nib.load(str(roi_path)).get_fdata(), dtype=float) > 0
        if mask.shape != tdata.shape[:3]:
            raise ValueError(f"ROI mask shape mismatch: {roi_path} vs {tmap_path}")
        vals = tdata[..., 0] if tdata.ndim == 4 else tdata
        roi_vals = vals[mask]
        metric = float(np.nanmean(roi_vals)) if roi_vals.size else float("nan")
        rows.append({"roi": roi_name, "activation_metric": metric})
    return pd.DataFrame(rows)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation-representation dissociation plot/table.")
    parser.add_argument("--rsa-summary", type=Path, required=True,
                        help="rsa_summary_stats.csv from run_rsa_optimized.py.")
    parser.add_argument("--out-dir", type=Path, default=None)

    # Provide either activation-table OR (tmap + roi-dir)
    parser.add_argument("--activation-table", type=Path, default=None,
                        help="Precomputed ROI activation table with columns: roi, activation_metric.")
    parser.add_argument("--tmap", type=Path, default=None,
                        help="GLM contrast t-map (NIfTI). If set, requires --roi-dir.")
    parser.add_argument("--roi-dir", type=Path, default=None,
                        help="Directory containing ROI masks (*.nii.gz). Used with --tmap.")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rsa_summary = _load_rsa_summary(args.rsa_summary)
    delta = _compute_delta_similarity(rsa_summary)

    if args.activation_table is not None:
        activation = _load_activation_table(args.activation_table)
    elif args.tmap is not None:
        if args.roi_dir is None:
            raise ValueError("--tmap requires --roi-dir")
        activation = _extract_activation_from_tmap(args.tmap, args.roi_dir)
    else:
        raise ValueError("Provide either --activation-table or --tmap + --roi-dir")

    merged = delta.merge(activation, on="roi", how="left")
    table_path = out_dir / "table_activation_representation_dissociation.tsv"
    _write_tsv(merged, table_path)

    # Plot
    import matplotlib.pyplot as plt

    fig_path = out_dir / "fig_activation_representation_dissociation.png"
    plt.figure(figsize=(7.0, 5.0))
    plt.scatter(merged["activation_metric"], merged["delta_similarity_yy"], alpha=0.8, label="yy (post-pre)")
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("Activation metric (mean t in ROI)")
    plt.ylabel("Delta similarity (post - pre)")
    plt.title("Activation vs Representation (yy)")
    # Annotate ROI names lightly.
    for _, row in merged.dropna(subset=["activation_metric", "delta_similarity_yy"]).iterrows():
        plt.text(float(row["activation_metric"]), float(row["delta_similarity_yy"]), str(row["roi"]),
                 fontsize=7, alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[dissociation] wrote {table_path} and {fig_path}")


if __name__ == "__main__":
    main()

