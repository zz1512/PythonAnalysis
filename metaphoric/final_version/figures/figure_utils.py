#!/usr/bin/env python3
"""Shared figure helpers for metaphor project."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def abbreviate_roi_name(name: str) -> str:
    text = str(name)

    direct = {
        "func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort": "TP-L1",
        "func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort": "TP-L2",
        "func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort": "Fus-L",
        "func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort": "Prec-L1",
        "func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort": "Prec-L2",
        "func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort": "PHG-R",
        "func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort": "LOC-L",
        "atlas_L_AG": "Atlas L-AG",
        "atlas_L_IFGop": "Atlas L-IFGop",
        "atlas_L_IFGtri": "Atlas L-IFGtri",
        "atlas_L_fusiform": "Atlas L-Fus",
        "atlas_L_pMTG": "Atlas L-pMTG",
        "atlas_L_precuneus": "Atlas L-Prec",
        "atlas_L_temporal_pole": "Atlas L-TP",
        "atlas_R_parahippocampal": "Atlas R-PHG",
        "atlas_R_precuneus": "Atlas R-Prec",
    }
    if text in direct:
        return direct[text]

    if text.startswith("lit_"):
        return text.replace("lit_", "").replace("_", "-")
    if text.startswith("litspat_"):
        return text.replace("litspat_", "").replace("_", "-")

    text = re.sub(r"_HO_cort$", "", text)
    text = re.sub(r"_posterior_division", "", text)
    text = re.sub(r"_superior_division", "", text)
    text = text.replace("Temporal_Pole", "TP")
    text = text.replace("Precuneous_Cortex", "Prec")
    text = text.replace("Parahippocampal_Gyrus", "PHG")
    text = text.replace("Lateral_Occipital_Cortex", "LOC")
    text = text.replace("Temporal_Fusiform_Cortex", "Fus")
    text = text.replace("Metaphor_gt_Spatial", "M>S")
    text = text.replace("Spatial_gt_Metaphor", "S>M")
    text = text.replace("_", " ")
    return text


def apply_publication_rcparams() -> None:
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="black",
    )


def save_png_pdf(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def make_half_violins(ax: plt.Axes, centers: list[float], side: str = "left") -> None:
    collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for poly, center in zip(collections, centers):
        try:
            path = poly.get_paths()[0]
        except Exception:
            continue
        verts = path.vertices
        if side == "left":
            verts[:, 0] = verts[:, 0].clip(max=center)
        else:
            verts[:, 0] = verts[:, 0].clip(min=center)
