from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import ensure_dir, list_run_event_files, read_table, write_table


@dataclass(slots=True)
class ExtremeItemResult:
    pic_ids: list[int]
    summary: pd.DataFrame


def _load_subject_events(subject_dir: Path) -> pd.DataFrame:
    frames = [read_table(item.path) for item in list_run_event_files(subject_dir)]
    if not frames:
        raise FileNotFoundError(f"No run event files found in {subject_dir}")
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values("pic_num").reset_index(drop=True)


def find_extreme_items(
    subject_dirs: Iterable[str | Path],
    *,
    trial_type_column: str = "trial_type",
    pic_column: str = "pic_num",
    score_column: str = "condition_median_jiaocha_1",
) -> ExtremeItemResult:
    """Replicate the exact item-flagging logic of get_events.m."""

    item_scores: dict[int, float] = {}
    for subject_dir in subject_dirs:
        frame = _load_subject_events(Path(subject_dir))
        example_rows = frame.loc[frame[trial_type_column].astype(str) == "e", [pic_column, score_column]]
        for pic_id, score in example_rows.itertuples(index=False):
            item_scores[int(pic_id)] = item_scores.get(int(pic_id), 0.0) + float(score)

    summary = pd.DataFrame(
        {"pic_num": sorted(item_scores), "aggregate_score": [item_scores[key] for key in sorted(item_scores)]}
    )
    mean_score = summary["aggregate_score"].mean()
    std_score = summary["aggregate_score"].std(ddof=1)
    low = summary.loc[summary["aggregate_score"] < mean_score - std_score, "pic_num"].tolist()
    high = summary.loc[summary["aggregate_score"] > mean_score + std_score, "pic_num"].tolist()
    summary["extreme_label"] = "middle"
    summary.loc[summary["pic_num"].isin(low), "extreme_label"] = "tl"
    summary.loc[summary["pic_num"].isin(high), "extreme_label"] = "th"
    return ExtremeItemResult(pic_ids=sorted(low + high), summary=summary)


def relabel_events_excluding_extremes(
    input_root: str | Path,
    output_root: str | Path,
    *,
    low_label: str = "tl",
    high_label: str = "th",
    trial_type_column: str = "trial_type",
    pic_column: str = "pic_num",
    score_column: str = "condition_median_jiaocha_1",
) -> ExtremeItemResult:
    subject_dirs = sorted({path.parent for path in Path(input_root).glob("sub*/*run-*_events.*")})
    result = find_extreme_items(
        subject_dirs,
        trial_type_column=trial_type_column,
        pic_column=pic_column,
        score_column=score_column,
    )

    summary = result.summary
    low_ids = set(summary.loc[summary["extreme_label"] == "tl", "pic_num"].astype(int).tolist())
    high_ids = set(summary.loc[summary["extreme_label"] == "th", "pic_num"].astype(int).tolist())

    output_root = ensure_dir(output_root)
    for subject_dir in subject_dirs:
        out_subject_dir = ensure_dir(output_root / Path(subject_dir).name)
        for run_file in list_run_event_files(subject_dir):
            frame = read_table(run_file.path).copy()
            frame.loc[frame[pic_column].isin(low_ids), trial_type_column] = low_label
            frame.loc[frame[pic_column].isin(high_ids), trial_type_column] = high_label
            write_table(frame, out_subject_dir / run_file.path.name)

    write_table(summary, output_root / "extreme_item_summary.tsv")
    return result
