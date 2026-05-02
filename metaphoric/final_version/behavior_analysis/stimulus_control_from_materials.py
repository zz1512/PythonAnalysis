#!/usr/bin/env python
"""
Extract and audit historical stimulus-control ratings from the materials workbook.

The workbook is a manually arranged legacy file. This script keeps the original
workbook untouched, exports machine-readable control tables, and reports whether
the available ratings cover the current formal stimuli in stimuli_template.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - allows coverage export without scipy
    stats = None


FINAL_ROOT = Path(__file__).resolve().parents[1]


def _default_base_dir() -> Path:
    override = os.environ.get("PYTHON_METAPHOR_ROOT", "").strip()
    if override:
        return Path(override)
    return Path("E:/python_metaphor")


def _find_default_workbook(base_dir: Path) -> Path:
    materials_dir = base_dir / "materials_detail"
    candidates = sorted(
        p for p in materials_dir.glob("*.xlsx")
        if "+" in p.name and not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No materials workbook with '+' in filename found under {materials_dir}"
        )
    return candidates[0]


def _find_default_mapping(base_dir: Path) -> Path:
    materials_dir = base_dir / "materials_detail"
    candidates = sorted(p for p in materials_dir.glob("*.txt") if not p.name.startswith("~$"))
    if not candidates:
        raise FileNotFoundError(f"No mapping .txt found under {materials_dir}")
    return candidates[0]


def _read_mapping(mapping_file: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    code_to_word: dict[str, str] = {}
    with mapping_file.open("r", encoding="utf-8-sig") as f:
        header = next(f, "")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            stem = parts[0].replace(".jpg", "").strip()
            word = parts[1].strip()
            if stem and word:
                code_to_word[stem] = word
    word_to_codes: dict[str, list[str]] = {}
    for stem, word in code_to_word.items():
        word_to_codes.setdefault(word, []).append(stem)
    return code_to_word, word_to_codes


def _first_word_from_sentence(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    for sep in ("\u662f", "\u5728"):  # shi4 / zai4
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return None


def _word_to_stem(word: object, word_to_codes: dict[str, list[str]], prefix: str | None = None) -> str | None:
    if not isinstance(word, str):
        return None
    candidates = word_to_codes.get(word.strip(), [])
    if prefix:
        candidates = [c for c in candidates if c.startswith(prefix)]
    return candidates[0] if candidates else None


def _canonical_word_label(stem: str | None) -> str | None:
    if not stem:
        return None
    for prefix in ("yyew", "yyw", "kjew", "kjw", "jx", "jc"):
        if stem.startswith(prefix):
            return f"{prefix}_{stem[len(prefix):]}"
    return stem


def _pair_norm(stems: Iterable[str | None]) -> str | None:
    vals = [str(s) for s in stems if isinstance(s, str) and s]
    if len(vals) != 2:
        return None
    return "|".join(sorted(vals))


def _pair_to_stems(pair_text: object, word_to_codes: dict[str, list[str]]) -> tuple[str | None, str | None]:
    if not isinstance(pair_text, str) or "-" not in pair_text:
        return None, None
    left, right = [x.strip() for x in pair_text.split("-", 1)]
    return _word_to_stem(left, word_to_codes), _word_to_stem(right, word_to_codes)


def _load_current_stimuli(stimuli_template: Path, code_to_word: dict[str, str]) -> pd.DataFrame:
    stim = pd.read_csv(stimuli_template)
    stim["stem"] = stim["word_label"].astype(str).str.replace("_", "", regex=False)
    stim["word"] = stim["stem"].map(code_to_word)
    return stim


def _sentence_controls(workbook: Path, word_to_codes: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Sheet index 1: metaphor sentences, columns 0 sentence, 3:7 ratings.
    # Sheet index 2: spatial sentences, columns 0 sentence, 3:7 ratings.
    specs = [
        (1, "Metaphor", "yyw", ["comprehensibility", "difficulty", "novelty", "familiarity"]),
        (2, "Spatial", "kjw", ["comprehensibility", "difficulty", "novelty", "familiarity"]),
    ]
    for sheet_idx, condition, prefix, variables in specs:
        df = pd.read_excel(workbook, sheet_name=sheet_idx)
        valid = df[df.iloc[:, 3].notna()].copy()
        for _, row in valid.iterrows():
            cue = _first_word_from_sentence(row.iloc[0])
            stem = _word_to_stem(cue, word_to_codes, prefix)
            out = {
                "control_scope": "sentence",
                "condition": condition,
                "source_sheet_index": sheet_idx,
                "sentence": row.iloc[0],
                "cue_word": cue,
                "stem": stem,
                "word_label": _canonical_word_label(stem),
                "pair_norm": None,
            }
            for idx, var in enumerate(variables, start=3):
                out[var] = pd.to_numeric(row.iloc[idx], errors="coerce")
            rows.append(out)
    return pd.DataFrame(rows)


def _wordpair_controls(workbook: Path, word_to_codes: dict[str, list[str]]) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=4)
    rating_rows = df[pd.to_numeric(df.iloc[:, 4], errors="coerce").notna()].copy()
    rating_rows = rating_rows[pd.to_numeric(rating_rows.iloc[:, 4], errors="coerce").between(1, 20)]
    rows: list[dict[str, object]] = []
    # Legacy layout: rating row with sub=n is offset by one row from material list.
    # The matching material appears at zero-based row n-1 in columns 0/1/2.
    for _, rating_row in rating_rows.iterrows():
        item_index = int(pd.to_numeric(rating_row.iloc[4], errors="coerce"))
        material_row = df.iloc[item_index - 1]
        for col_idx, condition in [(0, "Metaphor"), (1, "Spatial"), (2, "Baseline")]:
            pair_text = material_row.iloc[col_idx]
            stem_a, stem_b = _pair_to_stems(pair_text, word_to_codes)
            rows.append({
                "control_scope": "word_pair",
                "condition": condition,
                "source_sheet_index": 4,
                "legacy_item_index": item_index,
                "pair_text": pair_text,
                "stem_a": stem_a,
                "stem_b": stem_b,
                "word_label_a": _canonical_word_label(stem_a),
                "word_label_b": _canonical_word_label(stem_b),
                "pair_norm": _pair_norm([stem_a, stem_b]),
                "association": pd.to_numeric(rating_row.iloc[5 + col_idx], errors="coerce"),
                "pair_familiarity": pd.to_numeric(rating_row.iloc[11 + col_idx], errors="coerce"),
                "alignment_note": "rating sub=n aligned to material-list row n-1",
            })
    return pd.DataFrame(rows)


def _coverage(stim: pd.DataFrame, sentence: pd.DataFrame, wordpair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, type_code in [("Metaphor", "yyw"), ("Spatial", "kjw")]:
        formal = stim[(stim["condition"] == condition) & (stim["type"] == type_code)].copy()
        formal_set = set(formal["stem"])
        control_set = set(sentence.loc[sentence["condition"] == condition, "stem"].dropna())
        rows.append({
            "control_scope": "sentence",
            "condition": condition,
            "formal_n": len(formal_set),
            "covered_n": len(formal_set & control_set),
            "missing_n": len(formal_set - control_set),
            "coverage_rate": len(formal_set & control_set) / len(formal_set) if formal_set else math.nan,
            "missing_stems": ";".join(sorted(formal_set - control_set)),
        })

    pair_df = stim.groupby(["condition", "pair_id"]).agg(
        pair_norm=("stem", lambda x: _pair_norm(x)),
    ).reset_index()
    for condition in ["Metaphor", "Spatial", "Baseline"]:
        formal_set = set(pair_df.loc[pair_df["condition"] == condition, "pair_norm"].dropna())
        control_set = set(wordpair.loc[wordpair["condition"] == condition, "pair_norm"].dropna())
        rows.append({
            "control_scope": "word_pair",
            "condition": condition,
            "formal_n": len(formal_set),
            "covered_n": len(formal_set & control_set),
            "missing_n": len(formal_set - control_set),
            "coverage_rate": len(formal_set & control_set) / len(formal_set) if formal_set else math.nan,
            "missing_stems": ";".join(sorted(formal_set - control_set)),
        })
    return pd.DataFrame(rows)


def _pvalue_welch(a: pd.Series, b: pd.Series) -> float:
    if stats is None:
        return math.nan
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return math.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _pvalue_kruskal(groups: list[pd.Series]) -> float:
    if stats is None:
        return math.nan
    clean = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    if any(len(g) < 2 for g in clean):
        return math.nan
    return float(stats.kruskal(*clean).pvalue)


def _balance(sentence: pd.DataFrame, wordpair: pd.DataFrame, stim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    formal_cues = {
        "Metaphor": set(stim[(stim["condition"] == "Metaphor") & (stim["type"] == "yyw")]["stem"]),
        "Spatial": set(stim[(stim["condition"] == "Spatial") & (stim["type"] == "kjw")]["stem"]),
    }
    sent_formal = sentence[sentence.apply(
        lambda r: r["stem"] in formal_cues.get(r["condition"], set()), axis=1
    )]
    for var in ["comprehensibility", "difficulty", "novelty", "familiarity"]:
        a = sent_formal.loc[sent_formal["condition"] == "Metaphor", var]
        b = sent_formal.loc[sent_formal["condition"] == "Spatial", var]
        rows.append({
            "control_scope": "sentence",
            "variable": var,
            "conditions": "Metaphor_vs_Spatial",
            "n_metaphor": int(a.notna().sum()),
            "n_spatial": int(b.notna().sum()),
            "mean_metaphor": float(pd.to_numeric(a, errors="coerce").mean()),
            "mean_spatial": float(pd.to_numeric(b, errors="coerce").mean()),
            "difference_metaphor_minus_spatial": float(pd.to_numeric(a, errors="coerce").mean() - pd.to_numeric(b, errors="coerce").mean()),
            "test": "Welch_t_available_formal_items",
            "p_value": _pvalue_welch(a, b),
            "interpretation_note": "Partial coverage only; not a full formal stimulus-control test.",
        })

    pair_df = stim.groupby(["condition", "pair_id"]).agg(
        pair_norm=("stem", lambda x: _pair_norm(x)),
    ).reset_index()
    wp_formal = wordpair.merge(pair_df[["condition", "pair_norm"]].drop_duplicates(), on=["condition", "pair_norm"], how="inner")
    for var in ["association", "pair_familiarity"]:
        groups = [wp_formal.loc[wp_formal["condition"] == c, var] for c in ["Metaphor", "Spatial", "Baseline"]]
        rows.append({
            "control_scope": "word_pair",
            "variable": var,
            "conditions": "Metaphor_vs_Spatial_vs_Baseline",
            "n_metaphor": int(groups[0].notna().sum()),
            "n_spatial": int(groups[1].notna().sum()),
            "n_baseline": int(groups[2].notna().sum()),
            "mean_metaphor": float(pd.to_numeric(groups[0], errors="coerce").mean()),
            "mean_spatial": float(pd.to_numeric(groups[1], errors="coerce").mean()),
            "mean_baseline": float(pd.to_numeric(groups[2], errors="coerce").mean()),
            "test": "Kruskal_available_formal_pairs",
            "p_value": _pvalue_kruskal(groups),
            "interpretation_note": "Partial coverage for Metaphor/Spatial; baseline is complete under legacy alignment assumption.",
        })
    return pd.DataFrame(rows)


def _clean_for_json(value):
    if isinstance(value, dict):
        return {k: _clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=_default_base_dir())
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--mapping-file", type=Path, default=None)
    parser.add_argument("--stimuli-template", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    workbook = args.workbook or _find_default_workbook(base_dir)
    mapping_file = args.mapping_file or _find_default_mapping(base_dir)
    stimuli_template = args.stimuli_template or base_dir / "stimuli_template.csv"
    paper_output_root = args.paper_output_root or base_dir / "paper_outputs"

    tables_si = paper_output_root / "tables_si"
    qc_dir = paper_output_root / "qc" / "stimulus_control"
    tables_si.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    code_to_word, word_to_codes = _read_mapping(mapping_file)
    stim = _load_current_stimuli(stimuli_template, code_to_word)
    sentence = _sentence_controls(workbook, word_to_codes)
    wordpair = _wordpair_controls(workbook, word_to_codes)
    controls = pd.concat([sentence, wordpair], ignore_index=True, sort=False)
    coverage = _coverage(stim, sentence, wordpair)
    balance = _balance(sentence, wordpair, stim)

    control_path = tables_si / "table_stimulus_control_from_materials.tsv"
    coverage_path = qc_dir / "stimulus_control_materials_coverage.tsv"
    balance_path = tables_si / "table_stimulus_balance_materials_available.tsv"
    manifest_path = qc_dir / "stimulus_control_from_materials_manifest.json"

    controls.to_csv(control_path, sep="\t", index=False)
    coverage.to_csv(coverage_path, sep="\t", index=False)
    balance.to_csv(balance_path, sep="\t", index=False)
    manifest = {
        "workbook": str(workbook),
        "mapping_file": str(mapping_file),
        "stimuli_template": str(stimuli_template),
        "control_table": str(control_path),
        "coverage_table": str(coverage_path),
        "balance_table": str(balance_path),
        "n_control_rows": int(len(controls)),
        "coverage": coverage.to_dict(orient="records"),
        "limitations": [
            "Sentence ratings cover only part of the current formal Metaphor/Spatial stimuli.",
            "Word-pair ratings use a legacy offset alignment assumption and cover only part of current Metaphor/Spatial pairs.",
            "The workbook does not contain low-level lexical controls such as frequency, stroke count, concreteness, imageability, valence, arousal, or embedding distance.",
        ],
    }
    manifest_path.write_text(
        json.dumps(_clean_for_json(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {control_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {balance_path}")
    print(f"Wrote {manifest_path}")
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nAvailable balance:")
    print(balance.to_string(index=False))


if __name__ == "__main__":
    main()
