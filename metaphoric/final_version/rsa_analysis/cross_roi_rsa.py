from __future__ import annotations

import argparse
from itertools import combinations
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
from common.pattern_metrics import dsm_correlation, load_masked_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-ROI RSA consistency analysis.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--metric", default="correlation")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])
    rows: list[dict[str, object]] = []

    for roi_a, roi_b in combinations(roi_paths, 2):
        roi_pair = f"{roi_a.stem.replace('.nii', '')}__{roi_b.stem.replace('.nii', '')}"
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                for condition in ["yy", "kj"]:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    if not image_path.exists():
                        continue
                    samples_a = load_masked_samples(image_path, roi_a)
                    samples_b = load_masked_samples(image_path, roi_b)
                    rows.append({
                        "subject": subject_dir.name,
                        "roi_pair": roi_pair,
                        "time": time,
                        "condition": condition,
                        "value": dsm_correlation(samples_a, samples_b, metric=args.metric),
                    })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "cross_roi_rsa_subject_metrics.tsv")

    summaries = []
    for roi_pair, pair_frame in frame.groupby("roi_pair"):
        interaction = difference_in_differences(pair_frame)
        summary = {"roi_pair": roi_pair, **paired_t_summary(interaction["yy_delta"], interaction["kj_delta"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_pair}_cross_roi_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "cross_roi_rsa_group_summary.tsv")


if __name__ == "__main__":
    main()
