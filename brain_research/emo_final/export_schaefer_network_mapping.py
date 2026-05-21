#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Schaefer-200 ROI to Yeo 7-network mapping.

The network assignment is reused from utils_network_assignment.py, which is the
same logic used by downstream ROI/network summaries in this project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from utils_network_assignment import get_schaefer200_network_labels, get_sensorimotor_association_order
except ModuleNotFoundError:
    SCHAEFER_7NETWORKS = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Frontoparietal Control",
        "Default": "Default Mode Network",
    }

    LEFT_NETWORK_RANGES = [
        ("Vis", 1, 13),
        ("SomMot", 14, 30),
        ("DorsAttn", 31, 43),
        ("SalVentAttn", 44, 52),
        ("Limbic", 53, 59),
        ("Cont", 60, 72),
        ("Default", 73, 100),
    ]

    RIGHT_NETWORK_RANGES = [
        ("Vis", 1, 13),
        ("SomMot", 14, 30),
        ("DorsAttn", 31, 43),
        ("SalVentAttn", 44, 53),
        ("Limbic", 54, 59),
        ("Cont", 60, 72),
        ("Default", 73, 100),
    ]

    def get_schaefer200_network_labels() -> pd.DataFrame:
        rows = []
        for hemi, ranges in [("L", LEFT_NETWORK_RANGES), ("R", RIGHT_NETWORK_RANGES)]:
            idx_to_net = {}
            for net, start, end in ranges:
                for i in range(start, end + 1):
                    idx_to_net[i] = net
            for i in range(1, 101):
                net = idx_to_net[i]
                rows.append(
                    {
                        "roi": f"{hemi}_{i}",
                        "roi_id": i,
                        "hemi": hemi,
                        "network": net,
                        "network_full": SCHAEFER_7NETWORKS[net],
                    }
                )
        return pd.DataFrame(rows)

    def get_sensorimotor_association_order() -> list[str]:
        return ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_OUT_NAME = "schaefer200_7network_roi_mapping.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Schaefer200 ROI -> 7-network mapping table.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output CSV path. Default: <matrix-dir>/{DEFAULT_OUT_NAME}",
    )
    p.add_argument(
        "--matrix-dir",
        type=Path,
        default=DEFAULT_MATRIX_DIR,
        help="Matrix/result root used only when --out is not provided.",
    )
    p.add_argument(
        "--rois-file",
        type=Path,
        default=None,
        help=(
            "Optional existing *_rois.csv file. When provided, matrix_order is "
            "taken from this file and non-Schaefer ROIs such as V_* are filtered."
        ),
    )
    return p.parse_args()


def _load_matrix_order(rois_file: Optional[Path]) -> Optional[pd.DataFrame]:
    if rois_file is None:
        return None
    path = Path(rois_file)
    if not path.exists():
        raise FileNotFoundError(f"ROI order file not found: {path}")
    df = pd.read_csv(path)
    if "roi" not in df.columns:
        raise ValueError(f"{path} must contain a 'roi' column")
    out = df[["roi"]].copy()
    out["roi"] = out["roi"].astype(str)
    out["matrix_order"] = range(1, len(out) + 1)
    return out


def _add_repo_roi_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["roi_schaefer_style"] = out["roi"].astype(str)
    out["roi_matrix_style"] = out.apply(
        lambda r: f"L_{int(r['roi_id'])}" if str(r["hemi"]) == "L" else f"R_{100 + int(r['roi_id'])}",
        axis=1,
    )
    return out


def build_mapping_table(rois_file: Optional[Path] = None) -> pd.DataFrame:
    df = _add_repo_roi_aliases(get_schaefer200_network_labels().copy())

    # Logical Schaefer order: L_1..L_100, then R_1..R_100. In this repo, the
    # right hemisphere atlas labels can be stored as R_101..R_200 in *_rois.csv.
    df["mapping_index"] = range(1, len(df) + 1)
    df["roi_id_in_hemisphere"] = df["roi_id"].astype(int)
    df["schaefer_200_label_id"] = df.apply(
        lambda r: int(r["roi_id"]) if str(r["hemi"]) == "L" else 100 + int(r["roi_id"]),
        axis=1,
    )

    network_order = {name: i + 1 for i, name in enumerate(get_sensorimotor_association_order())}
    df["network_order"] = df["network"].map(network_order).astype(int)

    order_df = _load_matrix_order(rois_file)
    if order_df is None:
        df["matrix_order"] = df["mapping_index"]
        df["roi"] = df["roi_schaefer_style"]
    else:
        if order_df["roi"].duplicated().any():
            dup = order_df.loc[order_df["roi"].duplicated(), "roi"].tolist()
            raise ValueError(f"Duplicated ROI names in rois-file: {', '.join(dup[:10])}")
        order_map = dict(zip(order_df["roi"], order_df["matrix_order"]))
        df["matrix_order"] = df["roi_matrix_style"].map(order_map)
        df["matched_roi_name"] = df["roi_matrix_style"]

        miss_mask = df["matrix_order"].isna()
        df.loc[miss_mask, "matrix_order"] = df.loc[miss_mask, "roi_schaefer_style"].map(order_map)
        df.loc[miss_mask, "matched_roi_name"] = df.loc[miss_mask, "roi_schaefer_style"]

        missing = df[df["matrix_order"].isna()]["roi_schaefer_style"].tolist()
        if missing:
            raise ValueError(
                "Some Schaefer ROIs are missing from rois-file: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        df["matrix_order"] = df["matrix_order"].astype(int)
        df["roi"] = df["matched_roi_name"]

    cols = [
        "matrix_order",
        "mapping_index",
        "roi",
        "roi_schaefer_style",
        "hemi",
        "roi_id_in_hemisphere",
        "schaefer_200_label_id",
        "network_order",
        "network",
        "network_full",
    ]
    return df[cols].sort_values("matrix_order").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out is not None else Path(args.matrix_dir) / DEFAULT_OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = build_mapping_table(rois_file=args.rois_file)
    table.to_csv(out_path, index=False)

    print(f"Saved Schaefer200 7-network mapping: {out_path}")
    print(f"Rows: {len(table)}")
    print(table.groupby(["network_order", "network"], as_index=False).size().to_string(index=False))


if __name__ == "__main__":
    main()
