from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class StudyPaths:
    """Shared filesystem locations used throughout the pipeline."""

    project_root: Path
    raw_bold_root: Path | None = None
    onset_root: Path | None = None
    output_root: Path | None = None
    roi_root: Path | None = None
    group_mask: Path | None = None

    def resolve_output_root(self) -> Path:
        return self.output_root or self.project_root

    @classmethod
    def from_json(cls, path: str | Path) -> "StudyPaths":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapped: dict[str, Any] = {}
        for key, value in payload.items():
            mapped[key] = Path(value) if value is not None else None
        return cls(**mapped)

    def to_json_dict(self) -> dict[str, str | None]:
        return {
            "project_root": str(self.project_root),
            "raw_bold_root": str(self.raw_bold_root) if self.raw_bold_root else None,
            "onset_root": str(self.onset_root) if self.onset_root else None,
            "output_root": str(self.output_root) if self.output_root else None,
            "roi_root": str(self.roi_root) if self.roi_root else None,
            "group_mask": str(self.group_mask) if self.group_mask else None,
        }


@dataclass(slots=True)
class GlmSettings:
    tr: float = 2.0
    hrf_model: str = "spm"
    high_pass_seconds: float = 128.0
    noise_model: str = "ar1"
    smoothing_fwhm: float | None = None
    signal_scaling: bool | int = False
    minimize_memory: bool = False


@dataclass(slots=True)
class RdSettings:
    variance_threshold: float = 80.0
    equalized_iterations: int = 1000
    random_seed: int = 0
    searchlight_neighbors: int = 100
    min_cluster_size: int = 5


@dataclass(slots=True)
class StoryConfig:
    paths: StudyPaths
    glm: GlmSettings = field(default_factory=GlmSettings)
    rd: RdSettings = field(default_factory=RdSettings)
