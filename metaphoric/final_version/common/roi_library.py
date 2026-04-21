from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.final_utils import read_table


def _as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def load_roi_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest = read_table(manifest_path).copy()
    if "mask_path" not in manifest.columns or "roi_name" not in manifest.columns:
        raise ValueError(f"ROI manifest missing required columns: {manifest_path}")
    manifest["mask_path"] = manifest["mask_path"].map(lambda item: str(Path(str(item)).resolve()))
    for column in ["include_in_main", "include_in_rsa", "include_in_mvpa", "include_in_rd", "include_in_gps"]:
        if column in manifest.columns:
            manifest[column] = manifest[column].map(_as_bool)
    return manifest


def filter_roi_manifest(
    manifest: pd.DataFrame,
    *,
    roi_set: str | None = None,
    include_flag: str = "include_in_rsa",
    require_exists: bool = True,
) -> pd.DataFrame:
    frame = manifest.copy()
    if roi_set and roi_set.lower() != "all":
        frame = frame[frame["roi_set"].astype(str).str.lower() == roi_set.strip().lower()]
    if include_flag in frame.columns:
        frame = frame[frame[include_flag].map(_as_bool)]
    if require_exists:
        frame = frame[frame["mask_path"].map(lambda item: Path(str(item)).exists())]
    frame = frame.sort_values(["roi_set", "roi_name"]).reset_index(drop=True)
    return frame


def select_roi_masks(
    manifest_path: str | Path,
    *,
    roi_set: str | None = None,
    include_flag: str = "include_in_rsa",
    require_exists: bool = True,
) -> dict[str, Path]:
    manifest = load_roi_manifest(manifest_path)
    selected = filter_roi_manifest(
        manifest,
        roi_set=roi_set,
        include_flag=include_flag,
        require_exists=require_exists,
    )
    roi_masks: dict[str, Path] = {}
    for row in selected.itertuples(index=False):
        roi_masks[str(row.roi_name)] = Path(str(row.mask_path))
    return roi_masks
