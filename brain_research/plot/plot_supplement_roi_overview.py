#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_supplement_roi_overview.py

补充材料推荐图：ROI 概览（以 Schaefer label.gii 计算 ROI 顶点数）
说明：这是 ROI 空间覆盖的“可审计替代图”，用于补充材料说明 ROI 大小与左右半球一致性。
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")

ROI_CONFIG = {
    "Visual": {"L": [1, 2, 3, 4, 5, 6, 7], "R": [101, 102, 103, 104, 105, 106, 107]},
    "Salience_Insula": {"L": [38, 39, 40, 41, 42], "R": [138, 139, 140, 141, 142]},
    "Control_PFC": {"L": [57, 58, 59, 60, 61, 62], "R": [157, 158, 159, 160, 161, 162]},
}


def _load_label_gii(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(int)


def _count_vertices(atlas: np.ndarray, roi_ids: List[int]) -> int:
    return int(np.isin(atlas, roi_ids).sum())


def main() -> None:
    if not ATLAS_SURF_L.exists() or not ATLAS_SURF_R.exists():
        print("未找到 Schaefer label.gii，请检查 atlas 路径。")
        return

    atlas_l = _load_label_gii(ATLAS_SURF_L)
    atlas_r = _load_label_gii(ATLAS_SURF_R)

    rows = []
    for roi, cfg in ROI_CONFIG.items():
        n_l = _count_vertices(atlas_l, cfg["L"])
        n_r = _count_vertices(atlas_r, cfg["R"])
        rows.append({"ROI": roi, "Hemisphere": "L", "Vertices": n_l})
        rows.append({"ROI": roi, "Hemisphere": "R", "Vertices": n_r})

    df = pd.DataFrame(rows)

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sns.barplot(data=df, x="ROI", y="Vertices", hue="Hemisphere", ax=ax, palette=["#4c72b0", "#dd8452"])
    ax.set_title("Supplementary: ROI Vertex Counts (Schaefer 200)")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Vertices")
    ax.legend(title="")
    plt.tight_layout()
    out_dir = Path("./figures_supplement")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "S_ROI_vertex_counts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

