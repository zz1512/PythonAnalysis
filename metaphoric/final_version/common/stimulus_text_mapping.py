from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def default_mapping_path() -> Path:
    root = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    return Path(os.environ.get("METAPHOR_STIMULUS_MAPPING_FILE", root / "materials_detail" / "映射.txt"))


def normalize_stimulus_label(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    stem = Path(text).stem.strip()
    stem = stem.replace(" ", "")
    match = re.fullmatch(r"([A-Za-z]+)_?(\d+)", stem)
    if match:
        prefix, number = match.groups()
        return f"{prefix.lower()}{int(number)}"
    return stem.lower()


def load_stimulus_mapping(mapping_path: Path | None = None) -> dict[str, str]:
    path = Path(mapping_path) if mapping_path is not None else default_mapping_path()
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("编号"):
            continue
        parts = re.split(r"\t+", line, maxsplit=1)
        if len(parts) < 2:
            parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) < 2:
            continue
        label, word = parts[0].strip(), parts[1].strip()
        key = normalize_stimulus_label(label)
        if key and word:
            mapping[key] = word
    return mapping


def resolve_stimulus_text(value: object, mapping: dict[str, str]) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    normalized = normalize_stimulus_label(text)
    return mapping.get(normalized, text)


def attach_real_word_columns(
    frame: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    mapping_path: Path | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    mapping = load_stimulus_mapping(mapping_path)
    if not mapping:
        return out

    pairs = column_map or {
        "word_label": "real_word",
        "partner_label": "partner_real_word",
        "unique_label": "real_word",
        "pic_out": "real_word",
    }
    for source_col, target_col in pairs.items():
        if source_col not in out.columns:
            continue
        resolved = out[source_col].map(lambda item: resolve_stimulus_text(item, mapping))
        resolved = resolved.where(resolved.astype(str).str.strip().ne(""), pd.NA)
        if target_col in out.columns:
            existing = out[target_col].astype(str).str.strip()
            mask = out[target_col].isna() | existing.eq("") | existing.eq("nan") | existing.eq("<NA>")
            out.loc[mask, target_col] = resolved.loc[mask]
        else:
            out[target_col] = resolved
    return out
