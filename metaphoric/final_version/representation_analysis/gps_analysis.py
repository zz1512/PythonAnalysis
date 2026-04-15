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

from common.final_utils import difference_in_differences, ensure_dir, paired_t_summary, save_json, write_table
from common.pattern_metrics import gps_from_samples, load_masked_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-level global pattern similarity analysis.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, object]] = []
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                for condition in ["yy", "kj"]:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    if not image_path.exists():
                        continue
                    samples = load_masked_samples(image_path, roi_path)
                    rows.append({
                        "subject": subject_dir.name,
                        "roi": roi_name,
                        "time": time,
                        "condition": condition,
                        "value": gps_from_samples(samples),
                    })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "gps_subject_metrics.tsv")

    summaries = []
    for roi_name, roi_frame in frame.groupby("roi"):
        interaction = difference_in_differences(roi_frame)
        summary = {"roi": roi_name, **paired_t_summary(interaction["yy_delta"], interaction["kj_delta"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_gps_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "gps_group_summary.tsv")


if __name__ == "__main__":
    main()
