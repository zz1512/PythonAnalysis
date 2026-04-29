#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_learning_metadata_index.py

Purpose
- Build a learn-only LSS metadata index for runs 3/4.
- This is intentionally separate from `lss_metadata_index_final.csv`, so the
  existing pre/post pipeline for runs 1/2/5/6 remains untouched.

Inputs
- `${PYTHON_METAPHOR_ROOT}/data_events/sub-xx/sub-xx_run-{3,4}_events.tsv`
- `${PYTHON_METAPHOR_ROOT}/lss_betas_final/sub-xx/run-{3,4}/beta_trial-*.nii.gz`

Output
- `${PYTHON_METAPHOR_ROOT}/lss_betas_final/lss_metadata_index_learning.csv`

Design notes
- Rows are aligned by trial order between the run-level events table and the
  sorted beta filenames.
- `unique_label` is written as `yy_<pic_num>` / `kj_<pic_num>` so run3/run4 can
  be matched as repeated exposures of the same learning item.
- `pair_id` is set to `pic_num`, which is sufficient for the learning-dynamics
  repeated-exposure analysis.
"""

from __future__ import annotations

import argparse
import os
import re
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

from common.final_utils import ensure_dir, write_table  # noqa: E402


PYTHON_METAPHOR_ROOT = Path(os.environ.get("PYTHON_METAPHOR_ROOT", r"E:\python_metaphor"))
DEFAULT_EVENTS_ROOT = PYTHON_METAPHOR_ROOT / "data_events"
DEFAULT_LSS_ROOT = PYTHON_METAPHOR_ROOT / "lss_betas_final"
DEFAULT_OUTPUT = DEFAULT_LSS_ROOT / "lss_metadata_index_learning.csv"
VALID_CONDITIONS = {"yy", "kj"}


def _subject_id(raw: int | str) -> str:
    text = str(raw).strip()
    if text.lower().startswith("sub-"):
        return text
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse subject id from {raw!r}")
    return f"sub-{int(match.group(1)):02d}"


def _read_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"trial_type", "pic_num", "run"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["trial_type"] = frame["trial_type"].astype(str).str.strip().str.lower()
    frame = frame[frame["trial_type"].isin(VALID_CONDITIONS)].copy()
    frame["pic_num"] = pd.to_numeric(frame["pic_num"], errors="coerce")
    frame = frame.dropna(subset=["pic_num"])
    frame["pic_num"] = frame["pic_num"].astype(int)
    return frame.reset_index(drop=True)


def _beta_sort_key(path: Path) -> int:
    match = re.search(r"beta_trial-(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse beta trial index from {path.name}")
    return int(match.group(1))


def _collect_run_rows(subject_dir: Path, run: int, events_root: Path, lss_root: Path) -> list[dict[str, object]]:
    subject = subject_dir.name
    events_path = events_root / subject / f"{subject}_run-{run}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(f"Learning events file not found: {events_path}")

    beta_dir = lss_root / subject / f"run-{run}"
    if not beta_dir.exists():
        raise FileNotFoundError(f"LSS beta directory not found: {beta_dir}")

    events = _read_events(events_path)
    beta_files = sorted(beta_dir.glob("beta_trial-*.nii.gz"), key=_beta_sort_key)
    if len(events) != len(beta_files):
        raise ValueError(
            f"{subject} run-{run}: events/betas length mismatch "
            f"({len(events)} events vs {len(beta_files)} betas)."
        )

    rows: list[dict[str, object]] = []
    for trial_id, (event_row, beta_path) in enumerate(zip(events.itertuples(index=False), beta_files), start=1):
        condition = str(event_row.trial_type).strip().lower()
        pic_num = int(event_row.pic_num)
        rows.append(
            {
                "trial_index": trial_id - 1,
                "trial_id": trial_id,
                "unique_label": f"{condition}_{pic_num}",
                "word_label": f"{condition}_{pic_num}",
                "pair_id": pic_num,
                "pic_num": pic_num,
                "condition": condition,
                "beta_file": beta_path.name,
                "beta_path": str(beta_path.resolve()),
                "onset": getattr(event_row, "onset", pd.NA),
                "subject": subject,
                "run": run,
                "stage": "learn",
                "phase": "learn",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learn-only LSS metadata index for runs 3/4.")
    parser.add_argument("--events-root", type=Path, default=DEFAULT_EVENTS_ROOT)
    parser.add_argument("--lss-root", type=Path, default=DEFAULT_LSS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", nargs="+", type=int, default=[3, 4])
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    subject_dirs = sorted([p for p in args.lss_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    all_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for run in args.runs:
            all_rows.extend(_collect_run_rows(subject_dir, run, args.events_root, args.lss_root))

    frame = pd.DataFrame(all_rows).sort_values(["subject", "run", "trial_id"]).reset_index(drop=True)
    write_table(frame, args.output)
    print(f"[build_learning_metadata_index] wrote {len(frame)} rows -> {args.output}")


if __name__ == "__main__":
    main()
