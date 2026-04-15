from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json
from common.pattern_metrics import compute_group_paired_map_statistics, compute_searchlight_dimension_map, save_scalar_map


def compute_cell_maps(subject_dirs, subject_mask_root: Path, output_dir: Path, time: str, condition: str, filename_template: str, explained_threshold: float, voxel_count: int):
    paths = []
    for subject_dir in subject_dirs:
        image_path = subject_dir / filename_template.format(time=time, condition=condition)
        subject_mask = subject_mask_root / subject_dir.name / "mask.nii"
        if not image_path.exists() or not subject_mask.exists():
            continue
        reference_img, mask, values = compute_searchlight_dimension_map(image_path, subject_mask, explained_threshold, voxel_count)
        subject_output = ensure_dir(output_dir / subject_dir.name)
        out_path = subject_output / f"rd_{time}_{condition}.nii.gz"
        save_scalar_map(reference_img, mask, values, out_path)
        paths.append(out_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-brain RD searchlight analysis.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("subject_mask_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--voxel-count", type=int, default=100)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    cell_paths = {}
    for time in ["pre", "post"]:
        for condition in ["yy", "kj"]:
            cell_paths[(time, condition)] = compute_cell_maps(
                subject_dirs,
                args.subject_mask_root,
                output_dir,
                time,
                condition,
                args.filename_template,
                args.threshold,
                args.voxel_count,
            )

    summaries = {
        "yy_post_vs_pre": compute_group_paired_map_statistics(cell_paths[("post", "yy")], cell_paths[("pre", "yy")], output_dir / "group_yy_post_vs_pre", prefix="yy_post_vs_pre"),
        "kj_post_vs_pre": compute_group_paired_map_statistics(cell_paths[("post", "kj")], cell_paths[("pre", "kj")], output_dir / "group_kj_post_vs_pre", prefix="kj_post_vs_pre"),
    }
    save_json(summaries, output_dir / "rd_searchlight_summary.json")


if __name__ == "__main__":
    main()
