from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import GlmSettings
from .utils import (
    ensure_dir,
    infer_subject_mapping,
    list_bold_runs,
    list_run_event_files,
    read_motion_confounds,
    read_table,
    sanitize_name,
    save_json,
    write_table,
)


ItemGroupingFn = Callable[[pd.Series], str | None]


@dataclass(slots=True)
class TrialMapRecord:
    subject: str
    run: int
    trial_name: str
    analysis_group: str
    output_map: str
    onset: float
    duration: float
    pic_num: int | None
    memory: float | None


def _as_seconds(value: float, default_duration: float) -> float:
    if float(value) > 1000:
        return default_duration
    return float(value)


def _group_no_too_easy_or_hard(row: pd.Series) -> str | None:
    trial_type = str(row["trial_type"])
    cond = row.get("condition_median_jiaocha_1")
    if trial_type == "e" and cond == 1:
        return "HSC"
    if trial_type == "e" and cond == -1:
        return "LSC"
    if trial_type in {"tl", "th"}:
        return "median0"
    if trial_type == "t":
        return "tianchong"
    return None


def _group_gps(row: pd.Series) -> str | None:
    trial_type = str(row["trial_type"])
    cond = row.get("condition_median_jiaocha_1")
    if cond == 1:
        return "HSC"
    if cond == -1:
        return "LSC"
    if cond == 0 and trial_type == "e":
        return "tianchong"
    if cond == 0 and trial_type == "t":
        return "notuse"
    return None


def _group_biggerthan3(row: pd.Series) -> str | None:
    trial_type = str(row["trial_type"])
    score = row.get("jiaocha_score")
    if trial_type == "e" and score >= 3:
        return "HSC"
    if trial_type == "e" and score < 3:
        return "LSC"
    if trial_type == "t":
        return "tianchong"
    return None


def _group_tichu_jiaocha(row: pd.Series) -> str | None:
    trial_type = str(row["trial_type"])
    cond = row.get("condition_median_tichujiaocha_1")
    if trial_type == "e" and cond == 1:
        return "HSC"
    if trial_type == "e" and cond == -1:
        return "LSC"
    if trial_type == "t" or cond == 0:
        return "tianchong"
    return None


def _group_all_examples(row: pd.Series) -> str | None:
    return "example" if str(row["trial_type"]) == "e" else "tianchong"


ITEM_STORIES: dict[str, ItemGroupingFn] = {
    "no_too_easy_or_hard": _group_no_too_easy_or_hard,
    "gps": _group_gps,
    "biggerthan3": _group_biggerthan3,
    "tichu_jiaocha": _group_tichu_jiaocha,
    "rd_all_examples": _group_all_examples,
}


def _build_item_events(
    run_frame: pd.DataFrame,
    grouping_fn: ItemGroupingFn,
    *,
    default_duration: float = 6.0,
    run: int,
) -> tuple[pd.DataFrame, list[TrialMapRecord]]:
    event_rows: list[dict[str, float | str]] = []
    records: list[TrialMapRecord] = []
    trial_index = 0
    for _, row in run_frame.iterrows():
        group = grouping_fn(row)
        if group is None:
            continue
        trial_index += 1
        trial_name = sanitize_name(f"{group}_trial_{run:02d}_{trial_index:03d}")
        duration = _as_seconds(row.get("duration", default_duration), default_duration)
        event_rows.append(
            {"onset": float(row["onset"]), "duration": duration, "trial_type": trial_name}
        )
        records.append(
            TrialMapRecord(
                subject="",
                run=run,
                trial_name=trial_name,
                analysis_group=group,
                output_map="",
                onset=float(row["onset"]),
                duration=duration,
                pic_num=int(row["pic_num"]) if "pic_num" in row and not pd.isna(row["pic_num"]) else None,
                memory=float(row["memory"]) if "memory" in row and not pd.isna(row["memory"]) else None,
            )
        )
    return pd.DataFrame(event_rows), records


def fit_item_level_story(
    bold_root: str | Path,
    events_root: str | Path,
    output_root: str | Path,
    *,
    story: str,
    glm_settings: GlmSettings | None = None,
    bold_pattern: str = "sub-*.nii*",
    default_duration: float = 6.0,
) -> None:
    """Fit the item-level story used by the MATLAB project."""

    from nilearn.glm.first_level import FirstLevelModel

    if story not in ITEM_STORIES:
        raise ValueError(f"Unknown item-level story: {story}")

    glm_settings = glm_settings or GlmSettings()
    output_root = ensure_dir(output_root)
    subject_map = infer_subject_mapping(bold_root, events_root)
    grouping_fn = ITEM_STORIES[story]

    for _, (bold_subject_dir, event_subject_dir) in subject_map.items():
        subject_out = ensure_dir(output_root / bold_subject_dir.name)
        bold_runs = list_bold_runs(bold_subject_dir, pattern=bold_pattern)
        event_files = list_run_event_files(event_subject_dir)
        if len(bold_runs) != len(event_files):
            raise ValueError(
                f"{bold_subject_dir.name}: {len(bold_runs)} bold runs but {len(event_files)} event files"
            )

        events: list[pd.DataFrame] = []
        records: list[TrialMapRecord] = []
        for event_file in event_files:
            frame = read_table(event_file.path)
            event_df, run_records = _build_item_events(
                frame,
                grouping_fn,
                default_duration=default_duration,
                run=event_file.run,
            )
            for item in run_records:
                item.subject = bold_subject_dir.name
            events.append(event_df)
            records.extend(run_records)

        confounds = read_motion_confounds(bold_subject_dir)
        if confounds and len(confounds) != len(bold_runs):
            raise ValueError(f"{bold_subject_dir.name}: confound count does not match bold runs")
        confounds_arg = confounds if confounds else None

        model = FirstLevelModel(
            t_r=glm_settings.tr,
            hrf_model=glm_settings.hrf_model,
            drift_model="cosine",
            high_pass=1.0 / glm_settings.high_pass_seconds,
            noise_model=glm_settings.noise_model,
            smoothing_fwhm=glm_settings.smoothing_fwhm,
            signal_scaling=glm_settings.signal_scaling,
            minimize_memory=glm_settings.minimize_memory,
        )
        model = model.fit(bold_runs, events=events, confounds=confounds_arg)

        record_map: dict[str, TrialMapRecord] = {record.trial_name: record for record in records}
        for run_index, design in enumerate(model.design_matrices_):
            for column in design.columns:
                if column not in record_map:
                    continue
                contrast_list: list[np.ndarray] = []
                for design_index, this_design in enumerate(model.design_matrices_):
                    vector = np.zeros(this_design.shape[1], dtype=float)
                    if design_index == run_index:
                        vector[this_design.columns.get_loc(column)] = 1.0
                    contrast_list.append(vector)
                stat_map = model.compute_contrast(contrast_list, output_type="stat")
                output_name = f"{column}_stat.nii.gz"
                stat_map.to_filename(subject_out / output_name)
                record_map[column].output_map = output_name

        metadata = pd.DataFrame([asdict(record) for record in records])
        write_table(metadata, subject_out / "trial_maps.tsv")
        save_json(
            {
                "story": story,
                "subject": bold_subject_dir.name,
                "glm_settings": {
                    "tr": glm_settings.tr,
                    "hrf_model": glm_settings.hrf_model,
                    "high_pass_seconds": glm_settings.high_pass_seconds,
                    "noise_model": glm_settings.noise_model,
                },
            },
            subject_out / "model_summary.json",
        )


def _build_activation_run_events(
    run_frame: pd.DataFrame,
    *,
    story: str,
    default_duration: float = 6.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if story == "four_types":
        conditions = {
            "HSC_r": (run_frame["condition_median_jiaocha_1"] == 1) & (run_frame["memory"] == 1),
            "HSC_f": (run_frame["condition_median_jiaocha_1"] == 1) & (run_frame["memory"] == -1),
            "LSC_r": (run_frame["condition_median_jiaocha_1"] == -1) & (run_frame["memory"] == 1),
            "LSC_f": (run_frame["condition_median_jiaocha_1"] == -1) & (run_frame["memory"] == -1),
            "tianchong": run_frame["condition_median_jiaocha_1"] == 0,
        }
    elif story == "remember_forget":
        conditions = {
            "remembered": run_frame["memory_1"] == 1,
            "forgotten": run_frame["memory_1"] == -1,
            "filler": run_frame["trial_type"].astype(str) == "t",
        }
    else:
        raise ValueError(f"Unknown activation story: {story}")

    for name, mask in conditions.items():
        for _, row in run_frame.loc[mask].iterrows():
            rows.append(
                {
                    "onset": float(row["onset"]),
                    "duration": _as_seconds(row.get("duration", default_duration), default_duration),
                    "trial_type": name,
                }
            )
    return pd.DataFrame(rows)


def fit_activation_story(
    bold_root: str | Path,
    events_root: str | Path,
    output_root: str | Path,
    *,
    story: str,
    glm_settings: GlmSettings | None = None,
    bold_pattern: str = "sub-*.nii*",
    default_duration: float = 6.0,
) -> None:
    """Python counterpart to the MATLAB activation scripts."""

    from nilearn.glm.first_level import FirstLevelModel

    glm_settings = glm_settings or GlmSettings()
    output_root = ensure_dir(output_root)
    subject_map = infer_subject_mapping(bold_root, events_root)

    for _, (bold_subject_dir, event_subject_dir) in subject_map.items():
        subject_out = ensure_dir(output_root / bold_subject_dir.name)
        bold_runs = list_bold_runs(bold_subject_dir, pattern=bold_pattern)
        event_files = list_run_event_files(event_subject_dir)
        events = [
            _build_activation_run_events(
                read_table(event_file.path), story=story, default_duration=default_duration
            )
            for event_file in event_files
        ]
        confounds = read_motion_confounds(bold_subject_dir)
        confounds_arg = confounds if confounds else None

        model = FirstLevelModel(
            t_r=glm_settings.tr,
            hrf_model=glm_settings.hrf_model,
            drift_model="cosine",
            high_pass=1.0 / glm_settings.high_pass_seconds,
            noise_model=glm_settings.noise_model,
            smoothing_fwhm=glm_settings.smoothing_fwhm,
            signal_scaling=glm_settings.signal_scaling,
            minimize_memory=glm_settings.minimize_memory,
        )
        model = model.fit(bold_runs, events=events, confounds=confounds_arg)

        condition_names = sorted({name for event_df in events for name in event_df["trial_type"].unique()})
        for condition_name in condition_names:
            stat_map = model.compute_contrast(condition_name, output_type="stat")
            stat_map.to_filename(subject_out / f"{condition_name}_stat.nii.gz")

        save_json({"story": story, "subject": bold_subject_dir.name}, subject_out / "model_summary.json")
