from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, paired_t_summary, save_json, write_table
from common.pattern_metrics import load_masked_samples


def classify_binary(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    x = np.vstack([samples_a, samples_b])
    y = np.array([0] * len(samples_a) + [1] * len(samples_b))
    n_splits = min(5, np.bincount(y).min())
    if n_splits < 2:
        return float("nan")
    clf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0))])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return float(np.mean(cross_val_score(clf, x, y, cv=cv)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ROI-MVPA accuracy between pre and post phases.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows = []
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                yy_path = subject_dir / args.filename_template.format(time=time, condition="yy")
                kj_path = subject_dir / args.filename_template.format(time=time, condition="kj")
                if not yy_path.exists() or not kj_path.exists():
                    continue
                yy_samples = load_masked_samples(yy_path, roi_path)
                kj_samples = load_masked_samples(kj_path, roi_path)
                rows.append({
                    "subject": subject_dir.name,
                    "roi": roi_name,
                    "time": time,
                    "accuracy": classify_binary(yy_samples, kj_samples),
                })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "mvpa_pre_post_subject_metrics.tsv")

    summaries = []
    for roi_name, roi_frame in frame.groupby("roi"):
        pivot = roi_frame.pivot(index="subject", columns="time", values="accuracy").dropna()
        summary = {"roi": roi_name, **paired_t_summary(pivot["post"], pivot["pre"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_mvpa_pre_post_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "mvpa_pre_post_group_summary.tsv")


if __name__ == "__main__":
    main()
