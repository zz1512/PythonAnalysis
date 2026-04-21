"""
stack_patterns.py

用途
- 将 LSS 产生的单 trial beta maps，按 `phase x condition` 堆叠为 4D NIfTI（最后一维是 trials）。
- 这是 final_version 里“主线粘合点”：下游 RD/GPS/RSA/MVPA 依赖这里生成的固定文件名。

输入
- metadata_path: TSV/CSV（默认 `${PYTHON_METAPHOR_ROOT}/lss_betas_final/lss_metadata_index_final.csv`）
  必需字段（列名可有别名，会在 normalize_metadata 中统一）：
  - subject: `sub-01` 形式
  - beta_path 或 beta_file: 单 trial beta 的路径
  - condition: 原始 trial_type（支持 yyw/yyew/kjw/kjew 等，会归并到 yy/kj）
  - phase/time: pre/post/learn（可由 run 推断或直接给）
  - run, trial_id: 用于排序与追溯（推荐有）

输出（output_root/sub-xx/）
- `{phase}_{condition}.nii.gz`：4D patterns，例如 `pre_yy.nii.gz`、`post_kj.nii.gz`
- `{phase}_{condition}_metadata.tsv`：与 4D patterns 对齐的 trial 元数据（用于 debug/追溯）

关键约定
- 会把 `yyw/yyew -> yy`、`kjw/kjew -> kj`，避免生成下游找不到的文件名（如 pre_yyw）。
- 会把前后测中的 `jx -> baseline`、`jc -> fake`，以支持 baseline 控制验证。
- 默认排除 `fake`（可用 `--exclude-conditions` 调整）。
"""

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

from common.final_utils import ensure_dir, read_table, write_table
from common.pattern_metrics import concat_images


PYTHON_METAPHOR_ROOT = Path(r"E:\python_metaphor")
DEFAULT_METADATA_PATH = PYTHON_METAPHOR_ROOT / "lss_betas_final" / "lss_metadata_index_final.csv"
DEFAULT_OUTPUT_ROOT = PYTHON_METAPHOR_ROOT / "pattern_root"


CONDITION_MAP = {
    "metaphor": "yy",
    "yy": "yy",
    "yyw": "yy",
    "yyew": "yy",
    "spatial": "kj",
    "kj": "kj",
    "kjw": "kj",
    "kjew": "kj",
    "hsc": "yy",
    "lsc": "kj",
    "baseline": "baseline",
    "base": "baseline",
    "bl": "baseline",
    "nonlink": "baseline",
    "no_link": "baseline",
    "unlinked": "baseline",
    "jx": "baseline",
    "fake": "fake",
    "jc": "fake",
    "pseudoword": "fake",
    "pseudo": "fake",
    "nonword": "fake",
}

PHASE_MAP = {
    "pre": "pre",
    "pre-test": "pre",
    "post": "post",
    "post-test": "post",
    "learn": "learn",
    "learning": "learn",
}


def normalize_label(value: str, mapping: dict[str, str], default: str | None = None) -> str:
    text = str(value).strip().lower()
    mapped = mapping.get(text)
    if mapped is not None:
        return mapped

    # Heuristic normalization for common trial_type variants.
    if text.startswith("yy"):
        return "yy"
    if text.startswith("kj"):
        return "kj"
    if text == "jx":
        return "baseline"
    if "base" in text or text.startswith("bl"):
        return "baseline"
    if text == "jc":
        return "fake"
    if "fake" in text or "pseudo" in text or "nonword" in text or "jia" in text:
        return "fake"

    return default if default is not None else text


def resolve_beta_path(beta_value: str, metadata_path: Path, subject: str, run: int) -> str:
    beta_text = str(beta_value).strip()
    beta_path = Path(beta_text)
    if beta_path.is_absolute():
        return str(beta_path.resolve())

    candidates = []
    if beta_path.parent == Path("."):
        candidates.append(metadata_path.parent / subject / f"run-{run}" / beta_path.name)
    candidates.append(metadata_path.parent / beta_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Fall back to the most likely layout used by lss_betas_final.
    if beta_path.parent == Path("."):
        return str((metadata_path.parent / subject / f"run-{run}" / beta_path.name).resolve())
    return str((metadata_path.parent / beta_path).resolve())


def normalize_metadata(frame: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    frame = frame.copy()
    rename_candidates = {
        "beta_file": "beta_path",
        "output_map": "beta_path",
        "map_path": "beta_path",
        "trial_phase": "phase",
        "stage": "phase",
        "analysis_group": "condition",
    }
    for source, target in rename_candidates.items():
        if source in frame.columns and target not in frame.columns:
            frame[target] = frame[source]
    required = {"subject", "run", "condition", "phase", "beta_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    frame["subject"] = frame["subject"].astype(str)
    frame["run"] = frame["run"].astype(int)
    frame["condition"] = frame["condition"].map(lambda item: normalize_label(item, CONDITION_MAP))
    frame["phase"] = frame["phase"].map(lambda item: normalize_label(item, PHASE_MAP))
    frame["beta_path"] = [
        resolve_beta_path(beta_value, metadata_path, subject, run)
        for beta_value, subject, run in zip(frame["beta_path"], frame["subject"], frame["run"])
    ]
    if "trial_id" not in frame.columns:
        frame["trial_id"] = range(1, len(frame) + 1)
    return frame


def stack_subject(frame: pd.DataFrame, output_dir: Path, exclude_conditions: set[str]) -> None:
    output_dir = ensure_dir(output_dir)
    for (phase, condition), cell in frame.groupby(["phase", "condition"]):
        if cell.empty:
            continue
        if condition in exclude_conditions:
            continue
        output_image = output_dir / f"{phase}_{condition}.nii.gz"
        output_meta = output_dir / f"{phase}_{condition}_metadata.tsv"
        concat_images(cell["beta_path"].tolist(), output_image)
        write_table(cell.reset_index(drop=True), output_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack single-trial beta maps into phase x condition 4D images.")
    parser.add_argument(
        "metadata_path",
        nargs="?",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Trial-level metadata TSV/CSV with beta map paths (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root, one folder per subject (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--exclude-conditions",
        nargs="*",
        default=["fake"],
        help="Condition labels to exclude from stacking (default: fake).",
    )
    args = parser.parse_args()

    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")

    metadata = normalize_metadata(read_table(args.metadata_path), args.metadata_path)
    exclude_conditions = {str(item).strip().lower() for item in args.exclude_conditions if str(item).strip()}
    for subject, subject_frame in metadata.groupby("subject"):
        stack_subject(
            subject_frame.sort_values(["phase", "condition", "run", "trial_id"]),
            args.output_root / subject,
            exclude_conditions=exclude_conditions,
        )


if __name__ == "__main__":
    main()
