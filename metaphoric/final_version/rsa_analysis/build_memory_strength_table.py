"""
build_memory_strength_table.py

Purpose
- Build per-subject memory-strength tables for Model-RSA M7 variants.
- Keep the historical binary version, while adding continuous variants that
  incorporate response-speed information.

Default outputs
- memory_strength_sub-01.tsv                         # legacy alias = binary
- memory_strength_binary_sub-01.tsv
- memory_strength_continuous_confidence_sub-01.tsv
- memory_strength_manifest.tsv

Design notes
- `binary`: preserves the current 0/1 memory definition.
- `continuous_confidence`: conservative continuous variant:
    recall_score = binary_memory + rt_weight * speed_component * binary_memory
  so incorrect items stay at 0, while correct items are separated by retrieval speed.
- Additional modes (`learning_weighted`, `irt`) are reserved for the new plan and
  currently recorded as not yet implemented unless explicitly added later.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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
from common.stimulus_text_mapping import attach_real_word_columns  # noqa: E402


DEFAULT_SCORE_MODES = ["binary", "continuous_confidence"]
SUPPORTED_SCORE_MODES = {"binary", "continuous_confidence", "learning_weighted", "irt"}


def default_events_root() -> Path:
    base = os.environ.get("METAPHOR_DATA_EVENTS")
    if base:
        return Path(base)
    root = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    return root / "data_events"


def pick_column(frame: pd.DataFrame, preferred: list[str], *, kind: str) -> str:
    for col in preferred:
        if col in frame.columns:
            return col
    raise ValueError(f"None of {preferred} columns found for {kind}. Available: {list(frame.columns)}")


def _normalize_to_unit_interval(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low = float(valid.min())
    high = float(valid.max())
    if np.isclose(low, high):
        return pd.Series(1.0, index=values.index, dtype=float)
    return (numeric - low) / (high - low)


def derive_binary_memory(frame: pd.DataFrame) -> pd.Series:
    if "memory" in frame.columns:
        return pd.to_numeric(frame["memory"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if "action" in frame.columns:
        action_numeric = pd.to_numeric(frame["action"], errors="coerce")
        return (action_numeric == 3).astype(float)
    raise ValueError("No memory/action column available to derive binary recall.")


def derive_explicit_memory_score(frame: pd.DataFrame, score_col: str | None) -> pd.Series | None:
    if score_col and score_col in frame.columns:
        return pd.to_numeric(frame[score_col], errors="coerce")
    return None


def derive_rt_components(frame: pd.DataFrame, rt_cols: list[str]) -> tuple[pd.Series, pd.Series, str | None]:
    available = [col for col in rt_cols if col in frame.columns]
    if not available:
        empty = pd.Series(np.nan, index=frame.index, dtype=float)
        return empty, empty, None
    rt_col = available[0]
    rt_raw = pd.to_numeric(frame[rt_col], errors="coerce")
    rt_raw = rt_raw.where(rt_raw > 0)
    log_rt = np.log(rt_raw)
    norm_log_rt = _normalize_to_unit_interval(log_rt)
    speed_component = 1.0 - norm_log_rt
    return rt_raw, speed_component, rt_col


def _group_col(out: pd.DataFrame) -> str:
    if "real_word" in out.columns and out["real_word"].notna().any():
        return "real_word"
    return "word_label"


def _aggregate_variant_rows(frame: pd.DataFrame) -> pd.DataFrame:
    group_col = _group_col(frame)
    aggregations: dict[str, tuple[str, str]] = {
        "recall_score": ("recall_score", "mean"),
        "binary_memory": ("binary_memory", "mean"),
    }
    for optional in ["explicit_memory_score", "rt_raw", "speed_component"]:
        if optional in frame.columns:
            aggregations[optional] = (optional, "mean")
    if group_col == "real_word":
        aggregations["word_label"] = ("word_label", "first")

    out = frame.groupby(group_col, as_index=False).agg(**aggregations)
    if group_col == "real_word":
        ordered = ["word_label", "real_word", "recall_score", "binary_memory"]
    else:
        ordered = ["word_label", "recall_score", "binary_memory"]
    for optional in ["explicit_memory_score", "rt_raw", "speed_component"]:
        if optional in out.columns:
            ordered.append(optional)
    return out[ordered]


def build_variant_table(
    base_rows: pd.DataFrame,
    *,
    score_mode: str,
    rt_weight: float,
) -> pd.DataFrame:
    out = base_rows.copy()

    if score_mode == "binary":
        out["recall_score"] = out["binary_memory"].astype(float)
    elif score_mode == "continuous_confidence":
        if "speed_component" not in out.columns:
            raise ValueError("continuous_confidence requires RT-derived speed_component.")
        speed = pd.to_numeric(out["speed_component"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        out["recall_score"] = out["binary_memory"].astype(float) + rt_weight * speed * out["binary_memory"].astype(float)
    elif score_mode in {"learning_weighted", "irt"}:
        raise NotImplementedError(f"{score_mode} is reserved in the plan but not implemented yet.")
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    out["score_mode"] = score_mode
    return _aggregate_variant_rows(out)


def process_subject(
    events_path: Path,
    *,
    subject: str,
    score_col: str | None,
    word_cols: list[str],
    rt_cols: list[str],
    output_dir: Path,
    score_modes: list[str],
    rt_weight: float,
) -> list[dict[str, object]]:
    frame = pd.read_csv(events_path, sep="\t")
    word_col = pick_column(frame, word_cols, kind="word label")
    binary_memory = derive_binary_memory(frame)
    explicit_score = derive_explicit_memory_score(frame, score_col)
    rt_raw, speed_component, rt_col = derive_rt_components(frame, rt_cols)

    base_rows = pd.DataFrame({
        "word_label": frame[word_col].astype(str).str.strip(),
        "binary_memory": binary_memory.astype(float),
        "explicit_memory_score": explicit_score if explicit_score is not None else np.nan,
        "rt_raw": rt_raw.astype(float),
        "speed_component": speed_component.astype(float),
    })
    base_rows = attach_real_word_columns(base_rows, column_map={"word_label": "real_word"})
    base_rows = base_rows.dropna(subset=["word_label"])
    base_rows = base_rows[base_rows["word_label"].ne("")]

    summaries: list[dict[str, object]] = []
    for score_mode in score_modes:
        try:
            variant_table = build_variant_table(base_rows, score_mode=score_mode, rt_weight=rt_weight)
        except NotImplementedError as exc:
            summaries.append({
                "subject": subject,
                "score_mode": score_mode,
                "skip_reason": str(exc),
            })
            continue

        if score_mode == "binary":
            legacy_path = output_dir / f"memory_strength_{subject}.tsv"
            write_table(variant_table, legacy_path)

        out_path = output_dir / f"memory_strength_{score_mode}_{subject}.tsv"
        write_table(variant_table, out_path)
        summaries.append({
            "subject": subject,
            "score_mode": score_mode,
            "n_words": int(len(variant_table)),
            "mean_recall": float(np.nanmean(variant_table["recall_score"])) if len(variant_table) else float("nan"),
            "rt_source_col": rt_col,
            "output": str(out_path),
        })

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-subject memory-strength tables for Model-RSA M7 variants.")
    parser.add_argument("--events-root", type=Path, default=None)
    parser.add_argument("--target-run", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to {BASE_DIR}/memory_strength.")
    parser.add_argument("--score-col", default=None,
                        help="Optional explicit rating column. If provided, it is preserved for audit and future variants.")
    parser.add_argument("--word-cols", nargs="+", default=["real_word", "word_label", "pic_out", "ciyu", "word5_word"],
                        help="Candidate word-label columns (first match wins).")
    parser.add_argument("--rt-cols", nargs="+", default=["word7_RT", "rt", "RT", "response_time", "action_time"],
                        help="Candidate RT columns for continuous-confidence scoring.")
    parser.add_argument("--score-modes", nargs="+", default=DEFAULT_SCORE_MODES,
                        help="Memory-strength variants to build. Default: binary continuous_confidence")
    parser.add_argument("--rt-weight", type=float, default=0.4,
                        help="Weight for the speed component in continuous_confidence scoring.")
    parser.add_argument("--subject-range", nargs=2, type=int, default=[1, 28])
    args = parser.parse_args()

    unknown_modes = [mode for mode in args.score_modes if mode not in SUPPORTED_SCORE_MODES]
    if unknown_modes:
        raise ValueError(f"Unsupported score modes: {unknown_modes}. Supported: {sorted(SUPPORTED_SCORE_MODES)}")

    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

    events_root = args.events_root or default_events_root()
    output_dir = ensure_dir(args.output_dir or (BASE_DIR / "memory_strength"))

    summaries: list[dict[str, object]] = []
    for i in range(args.subject_range[0], args.subject_range[1] + 1):
        subject = f"sub-{i:02d}"
        events_path = events_root / subject / f"{subject}_run-{args.target_run}_events.tsv"
        if not events_path.exists():
            summaries.append({"subject": subject, "skip_reason": "events_file_missing", "path": str(events_path)})
            continue
        try:
            summaries.extend(
                process_subject(
                    events_path,
                    subject=subject,
                    score_col=args.score_col,
                    word_cols=args.word_cols,
                    rt_cols=args.rt_cols,
                    output_dir=output_dir,
                    score_modes=args.score_modes,
                    rt_weight=args.rt_weight,
                )
            )
        except Exception as exc:
            summaries.append({"subject": subject, "skip_reason": f"error: {exc}"})

    summary_frame = pd.DataFrame(summaries)
    write_table(summary_frame, output_dir / "memory_strength_manifest.tsv")
    print(f"[build_memory_strength_table] processed {len(summaries)} subject-variant rows -> {output_dir}")


if __name__ == "__main__":
    main()
