from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import pandas as pd

from .utils import ensure_dir, write_table


def stack_maps(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    images = [nib.load(str(path)) for path in image_paths]
    stacked = nib.concat_images(images, axis=3)
    stacked.to_filename(output_path)


def stack_maps_from_metadata(
    metadata_path: str | Path,
    output_root: str | Path,
    *,
    groups: Sequence[str],
    combined_name: str | None = None,
    filter_query: str | None = None,
) -> None:
    metadata_path = Path(metadata_path)
    frame = pd.read_csv(metadata_path, sep="	")
    if filter_query:
        frame = frame.query(filter_query).copy()
    output_root = ensure_dir(output_root)

    if combined_name:
        selected = frame.loc[frame["analysis_group"].isin(groups)].copy()
        if not selected.empty:
            paths = [metadata_path.parent / item for item in selected["output_map"].tolist()]
            stack_maps(paths, output_root / combined_name)
            write_table(selected, output_root / f"{Path(combined_name).stem}_metadata.tsv")

    for group in groups:
        selected = frame.loc[frame["analysis_group"] == group].copy()
        if selected.empty:
            continue
        paths = [metadata_path.parent / item for item in selected["output_map"].tolist()]
        name = f"{group}.nii.gz"
        stack_maps(paths, output_root / name)
        write_table(selected, output_root / f"{group}_metadata.tsv")


def build_story_pattern_outputs(
    trial_map_root: str | Path,
    output_root: str | Path,
    *,
    story: str,
    filter_query: str | None = None,
) -> None:
    """Create the stacked pattern images used by the downstream analyses."""

    trial_map_root = Path(trial_map_root)
    output_root = ensure_dir(output_root)

    if story == "no_too_easy_or_hard_gps":
        groups = ["HSC", "LSC", "median0"]
        combined_name = "glm_T_gps.nii.gz"
    elif story == "gps":
        groups = ["HSC", "LSC", "tianchong"]
        combined_name = "glm_T_gps.nii.gz"
    elif story == "rca":
        groups = ["HSC", "LSC"]
        combined_name = None
    elif story == "rd_all_examples":
        groups = ["example"]
        combined_name = "glm_T_allexamples.nii.gz"
    else:
        raise ValueError(f"Unknown pattern story: {story}")

    for subject_dir in sorted([path for path in trial_map_root.iterdir() if path.is_dir()]):
        metadata_path = subject_dir / "trial_maps.tsv"
        if not metadata_path.exists():
            continue
        stack_maps_from_metadata(
            metadata_path,
            output_root / subject_dir.name,
            groups=groups,
            combined_name=combined_name,
            filter_query=filter_query,
        )
