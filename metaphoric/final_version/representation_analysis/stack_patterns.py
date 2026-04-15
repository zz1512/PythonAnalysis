from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

from common.final_utils import ensure_dir, read_table, write_table
from common.pattern_metrics import concat_images


CONDITION_MAP = {
    "metaphor": "yy",
    "yy": "yy",
    "spatial": "kj",
    "kj": "kj",
    "hsc": "yy",
    "lsc": "kj",
}

PHASE_MAP = {
    "pre": "pre",
    "pre-test": "pre",
    "post": "post",
    "post-test": "post",
    "learn": "learn",
    "learning": "learn",
}


def normalize_label(value: str, mapping: dict[str, str], default: str | None = None) -> str:
    text = str(value).strip().lower()
    return mapping.get(text, default if default is not None else text)


def normalize_metadata(frame: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    frame = frame.copy()
    rename_candidates = {
        "beta_file": "beta_path",
        "output_map": "beta_path",
        "map_path": "beta_path",
        "trial_phase": "phase",
        "stage": "phase",
        "analysis_group": "condition",
    }
    for source, target in rename_candidates.items():
        if source in frame.columns and target not in frame.columns:
            frame[target] = frame[source]
    required = {"subject", "run", "condition", "phase", "beta_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    frame["subject"] = frame["subject"].astype(str)
    frame["run"] = frame["run"].astype(int)
    frame["condition"] = frame["condition"].map(lambda item: normalize_label(item, CONDITION_MAP))
    frame["phase"] = frame["phase"].map(lambda item: normalize_label(item, PHASE_MAP))
    frame["beta_path"] = frame["beta_path"].map(lambda item: str((metadata_path.parent / str(item)).resolve()) if not Path(str(item)).is_absolute() else str(Path(str(item)).resolve()))
    if "trial_id" not in frame.columns:
        frame["trial_id"] = range(1, len(frame) + 1)
    return frame


def stack_subject(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)
    for (phase, condition), cell in frame.groupby(["phase", "condition"]):
        if cell.empty:
            continue
        output_image = output_dir / f"{phase}_{condition}.nii.gz"
        output_meta = output_dir / f"{phase}_{condition}_metadata.tsv"
        concat_images(cell["beta_path"].tolist(), output_image)
        write_table(cell.reset_index(drop=True), output_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack single-trial beta maps into phase x condition 4D images.")
    parser.add_argument("metadata_path", type=Path, help="Trial-level metadata TSV/CSV with beta map paths.")
    parser.add_argument("output_root", type=Path, help="Output root, one folder per subject.")
    args = parser.parse_args()

    metadata = normalize_metadata(read_table(args.metadata_path), args.metadata_path)
    for subject, subject_frame in metadata.groupby("subject"):
        stack_subject(subject_frame.sort_values(["phase", "condition", "run", "trial_id"]), args.output_root / subject)


if __name__ == "__main__":
    main()
