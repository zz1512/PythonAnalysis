from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from common.final_utils import read_table


def _as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def current_roi_set(default: str = "main_functional") -> str:
    """当前 ROI 层级，统一从 METAPHOR_ROI_SET 环境变量读取。"""
    return os.environ.get("METAPHOR_ROI_SET", default).strip() or default


def sanitize_roi_tag(roi_set: str | None) -> str:
    """
    把 ROI_SET 字符串清洗成合法目录后缀。
    - 只保留字母数字与 `-_`
    - 其他字符替换为 `_`
    - 若清洗后为空则回落 `default`
    """
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(roi_set or "").strip())
    return tag or "default"


def default_roi_tagged_out_dir(
    base_dir: str | Path,
    analysis_name: str,
    *,
    override_env: str | None = None,
    roi_set: str | None = None,
) -> Path:
    """
    统一生成带 ROI_SET 后缀的默认输出目录。

    规则
    1. 若 `override_env` 指定的环境变量非空，则直接使用该值作为最终路径。
    2. 否则返回 `base_dir / f"{analysis_name}_{roi_tag}"`。
    3. `roi_set` 默认从 `METAPHOR_ROI_SET` 读取（兜底 `main_functional`）。

    例：
        base_dir=/data/python_metaphor, analysis_name="rd_results"
        METAPHOR_ROI_SET=literature
        -> /data/python_metaphor/rd_results_literature

    设计原因
    - 让 RSA / RD / GPS / MVPA-ROI / gPPI 等 ROI-based 分析在同一环境变量驱动下，
      各自的默认输出目录自动隔离，避免"同一目录被三层 ROI 互相覆盖"。
    """
    if override_env:
        override = os.environ.get(override_env, "").strip()
        if override:
            return Path(override)
    rs = roi_set if roi_set is not None else current_roi_set()
    return Path(base_dir) / f"{analysis_name}_{sanitize_roi_tag(rs)}"


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
