"""
build_memory_strength_table.py

用途
- 为 Model-RSA 中的 M7（被试回忆强度 RDM）准备 per-subject 的记忆强度表。
- 从 Run-7 的 events.tsv / events.csv 中提取每个 word_label 的 1-7 级回忆评分
  （如果只有 0/1 成功失败编码，则用 0/1 作为记忆强度）。

输入
- `--events-root`：events 根目录（默认 `${METAPHOR_DATA_EVENTS:-${PYTHON_METAPHOR_ROOT}/data_events}`）
  约定每个被试在 `events-root/sub-xx/sub-xx_run-7_events.tsv`

输出
- `memory_strength_{subject}.tsv`：列 `word_label`, `recall_score`

下游
- `model_rdm_comparison.py --memory-strength-dir <output-dir>` 用于 M7。

常见坑
- run-7 的 TSV 必须已经由 `getMemoryDetail.py` + `addMemory.py` 回填了 action/memory。
- 若 CSV 中存在 `word7_RT`（反应时）或 1-7 级主观评分列，可通过 `--score-col` 指定。
- 默认优先顺序：`score_col` > `memory` > `action` 数值化（3→1，其余→0）。
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


def default_events_root() -> Path:
    base = os.environ.get("METAPHOR_DATA_EVENTS")
    if base:
        return Path(base)
    root = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    return root / "data_events"


def derive_recall_score(frame: pd.DataFrame, score_col: str | None) -> pd.Series:
    if score_col and score_col in frame.columns:
        return pd.to_numeric(frame[score_col], errors="coerce")
    if "memory" in frame.columns:
        return pd.to_numeric(frame["memory"], errors="coerce")
    if "action" in frame.columns:
        action_numeric = pd.to_numeric(frame["action"], errors="coerce")
        return (action_numeric == 3).astype(float)
    raise ValueError("No score/memory/action column available to derive recall score.")


def pick_word_column(frame: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in frame.columns:
            return col
    raise ValueError(f"None of {preferred} columns found in events file. Got: {list(frame.columns)}")


def process_subject(events_path: Path, subject: str, score_col: str | None,
                    word_cols: list[str], output_dir: Path) -> dict:
    frame = pd.read_csv(events_path, sep="\t")
    word_col = pick_word_column(frame, word_cols)
    recall = derive_recall_score(frame, score_col)
    out = pd.DataFrame({
        "word_label": frame[word_col].astype(str).str.strip(),
        "recall_score": recall.astype(float),
    })
    out = attach_real_word_columns(out, column_map={"word_label": "real_word"})
    out = out.dropna(subset=["word_label"])
    out = out[out["word_label"].ne("")]
    group_col = "real_word" if "real_word" in out.columns and out["real_word"].notna().any() else "word_label"
    aggregations = {"recall_score": ("recall_score", "mean")}
    if group_col == "real_word":
        aggregations["word_label"] = ("word_label", "first")
    out = out.groupby(group_col, as_index=False).agg(**aggregations)
    if group_col == "real_word":
        out = out[["word_label", "real_word", "recall_score"]]
    else:
        out = out[["word_label", "recall_score"]]
    out_path = output_dir / f"memory_strength_{subject}.tsv"
    write_table(out, out_path)
    return {
        "subject": subject,
        "n_words": int(len(out)),
        "mean_recall": float(np.nanmean(out["recall_score"])) if len(out) else float("nan"),
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-subject memory strength tables for Model-RSA M7.")
    parser.add_argument("--events-root", type=Path, default=None)
    parser.add_argument("--target-run", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to {BASE_DIR}/memory_strength.")
    parser.add_argument("--score-col", default=None,
                        help="Optional column name with 1-7 recall ratings.")
    parser.add_argument("--word-cols", nargs="+", default=["real_word", "word_label", "pic_out", "ciyu", "word5_word"],
                        help="Candidate word-label columns (first match wins).")
    parser.add_argument("--subject-range", nargs=2, type=int, default=[1, 28])
    args = parser.parse_args()

    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

    events_root = args.events_root or default_events_root()
    output_dir = ensure_dir(args.output_dir or (BASE_DIR / "memory_strength"))

    summaries = []
    for i in range(args.subject_range[0], args.subject_range[1] + 1):
        subject = f"sub-{i:02d}"
        events_path = events_root / subject / f"{subject}_run-{args.target_run}_events.tsv"
        if not events_path.exists():
            summaries.append({"subject": subject, "skip_reason": "events_file_missing", "path": str(events_path)})
            continue
        try:
            summaries.append(process_subject(events_path, subject, args.score_col, args.word_cols, output_dir))
        except Exception as exc:
            summaries.append({"subject": subject, "skip_reason": f"error: {exc}"})

    summary_frame = pd.DataFrame(summaries)
    write_table(summary_frame, output_dir / "memory_strength_manifest.tsv")
    print(f"[build_memory_strength_table] processed {len(summaries)} subjects -> {output_dir}")


if __name__ == "__main__":
    main()
