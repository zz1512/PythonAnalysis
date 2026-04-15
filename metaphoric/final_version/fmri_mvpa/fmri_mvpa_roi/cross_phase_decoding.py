from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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


def fit_and_score(train_a: np.ndarray, train_b: np.ndarray, test_a: np.ndarray, test_b: np.ndarray) -> float:
    x_train = np.vstack([train_a, train_b])
    y_train = np.array([0] * len(train_a) + [1] * len(train_b))
    x_test = np.vstack([test_a, test_b])
    y_test = np.array([0] * len(test_a) + [1] * len(test_b))
    clf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0))])
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-phase decoding: train on learning, test on pre/post.")
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
            learn_yy = subject_dir / args.filename_template.format(time="learn", condition="yy")
            learn_kj = subject_dir / args.filename_template.format(time="learn", condition="kj")
            if not learn_yy.exists() or not learn_kj.exists():
                continue
            train_yy = load_masked_samples(learn_yy, roi_path)
            train_kj = load_masked_samples(learn_kj, roi_path)
            for time in ["pre", "post"]:
                test_yy_path = subject_dir / args.filename_template.format(time=time, condition="yy")
                test_kj_path = subject_dir / args.filename_template.format(time=time, condition="kj")
                if not test_yy_path.exists() or not test_kj_path.exists():
                    continue
                test_yy = load_masked_samples(test_yy_path, roi_path)
                test_kj = load_masked_samples(test_kj_path, roi_path)
                rows.append({
                    "subject": subject_dir.name,
                    "roi": roi_name,
                    "time": time,
                    "accuracy": fit_and_score(train_yy, train_kj, test_yy, test_kj),
                })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "cross_phase_subject_metrics.tsv")

    summaries = []
    for roi_name, roi_frame in frame.groupby("roi"):
        pivot = roi_frame.pivot(index="subject", columns="time", values="accuracy").dropna()
        summary = {"roi": roi_name, **paired_t_summary(pivot["post"], pivot["pre"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_cross_phase_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "cross_phase_group_summary.tsv")


if __name__ == "__main__":
    main()
