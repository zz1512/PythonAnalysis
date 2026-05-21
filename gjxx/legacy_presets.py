from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegacyPaths:
    """Legacy, project-specific filesystem defaults.

    These values exist to preserve historical behavior, but callers should prefer:
    CLI args > config JSON > legacy preset.
    """

    pattern_dir: Path
    patterns_hlrf_dir: Path
    events_dir: Path
    mask_left: Path
    mask_right: Path
    output_root: Path

    searchlight_mask_root: Path
    rsa_all_dir: Path
    alt_events_dir: Path


# The original MATLAB / Windows project layout.
FLXX1_WINDOWS = LegacyPaths(
    pattern_dir=Path(r"I:\FLXX1\Dimension\pattern"),
    patterns_hlrf_dir=Path(r"I:\FLXX1\First_level\patterns_hlrf"),
    events_dir=Path(r"I:\FLXX1\events"),
    mask_left=Path(r"I:\FLXX1\mask\Hippocampus_L.nii"),
    mask_right=Path(r"I:\FLXX1\mask\Hippocampus_R.nii"),
    output_root=Path(r"I:\FLXX1\Dimension\python_results"),
    searchlight_mask_root=Path(r"L:\FLXX_1\mvpa\GLM_item_allexample"),
    rsa_all_dir=Path(r"L:\FLXX_1\mvpa\GLM_item_allexample\rsa"),
    alt_events_dir=Path(r"L:\FLXX_1\First_level\get_onset\newdata_to_use"),
)
