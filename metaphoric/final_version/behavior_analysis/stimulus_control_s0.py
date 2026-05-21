#!/usr/bin/env python
"""
Build a submission-oriented S0 stimulus-control table from the reorganized
materials workbook.

The reorganized workbook fully defines the current formal stimulus set. This
script derives variables available from the workbook itself (word/sentence
length and pair semantic distance from local embeddings), optionally merges
external lexical norms, and exports balance/missingness reports.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


LEXICAL_NORM_COLUMNS = [
    "word_frequency",
    "stroke_count",
    "concreteness",
    "familiarity",
    "imageability",
    "valence",
    "arousal",
]

UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
VAD_MODEL_NAME = "Pectics/vad-macbert"


def _default_base_dir() -> Path:
    override = os.environ.get("PYTHON_METAPHOR_ROOT", "").strip()
    if override:
        return Path(override)
    return Path("E:/python_metaphor")


def _find_workbook(base_dir: Path) -> Path:
    materials_dir = base_dir / "materials_detail"
    candidates = sorted(
        p for p in materials_dir.glob("*.xlsx")
        if "\u6574\u7406" in p.name and not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(f"No reorganized materials workbook found under {materials_dir}")
    return candidates[0]


def _label_from_stem(stem: str) -> str:
    stem = str(stem).replace(".jpg", "")
    for prefix in ("yyew", "yyw", "kjew", "kjw", "jx"):
        if stem.startswith(prefix):
            return f"{prefix}_{stem[len(prefix):]}"
    return stem


def _stem_from_label(label: str) -> str:
    return str(label).replace("_", "")


def _char_count(text: object) -> float:
    if not isinstance(text, str):
        return math.nan
    return float(len(text.strip()))


def _sentence_char_count(text: object) -> float:
    if not isinstance(text, str):
        return math.nan
    stripped = text.strip()
    # Count Chinese/word characters while excluding common punctuation/spaces.
    return float(sum(1 for ch in stripped if not ch.isspace() and ch not in "，。！？；：,.!?;:"))


def _unique_words(table: pd.DataFrame) -> pd.DataFrame:
    first = table[["first_word", "first_word_label"]].rename(
        columns={"first_word": "word", "first_word_label": "word_label"}
    )
    second = table[["second_word", "second_word_label"]].rename(
        columns={"second_word": "word", "second_word_label": "word_label"}
    )
    words = pd.concat([first, second], ignore_index=True).drop_duplicates("word_label")
    return words.sort_values("word_label").reset_index(drop=True)


def _zipf_frequency(word: object) -> float:
    if not isinstance(word, str) or not word.strip():
        return math.nan
    try:
        from wordfreq import zipf_frequency
    except Exception:
        return math.nan
    return float(zipf_frequency(word.strip(), "zh"))


def _download_unihan_strokes(cache_dir: Path, *, force: bool = False) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = cache_dir / "unihan_total_strokes.tsv"
    if parsed_path.exists() and not force:
        df = pd.read_csv(parsed_path, sep="\t")
        return {str(row["char"]): int(row["stroke_count"]) for _, row in df.iterrows()}

    zip_path = cache_dir / "Unihan.zip"
    if not zip_path.exists() or force:
        with urlopen(UNIHAN_URL, timeout=120) as response:
            zip_path.write_bytes(response.read())

    strokes: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Unihan_IRGSources.txt") as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                if not line.startswith("U+"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[1] != "kTotalStrokes":
                    continue
                char = chr(int(parts[0][2:], 16))
                values = [int(x) for x in parts[2].split() if x.isdigit()]
                if values:
                    strokes[char] = values[0]

    pd.DataFrame(
        [{"char": char, "stroke_count": count} for char, count in sorted(strokes.items())]
    ).to_csv(parsed_path, sep="\t", index=False)
    return strokes


def _word_strokes(word: object, stroke_map: dict[str, int]) -> float:
    if not isinstance(word, str) or not word.strip():
        return math.nan
    counts = []
    for char in word.strip():
        if char.isspace():
            continue
        count = stroke_map.get(char)
        if count is None:
            return math.nan
        counts.append(count)
    return float(sum(counts)) if counts else math.nan


def _predict_vad(words: pd.DataFrame, model_name: str, *, batch_size: int = 32) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("torch/transformers are required for --vad-source hf_model") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    texts = words["word"].astype(str).tolist()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            result = model(**encoded)
            values = result.logits.detach().cpu().numpy()
            outputs.append(values)
    pred = np.vstack(outputs)
    if pred.shape[1] < 2:
        raise RuntimeError(f"VAD model returned {pred.shape[1]} outputs; expected at least valence/arousal.")
    out = words.copy()
    out["valence"] = pred[:, 0]
    out["arousal"] = pred[:, 1]
    if pred.shape[1] >= 3:
        out["dominance"] = pred[:, 2]
    out["vad_source"] = model_name
    return out


def _build_auto_lexical_norms(
    table: pd.DataFrame,
    cache_dir: Path,
    *,
    include_vad: bool,
    vad_model: str,
    force_unihan_download: bool,
) -> pd.DataFrame:
    words = _unique_words(table)
    words["word_frequency"] = words["word"].map(_zipf_frequency)
    stroke_map = _download_unihan_strokes(cache_dir, force=force_unihan_download)
    words["stroke_count"] = words["word"].map(lambda word: _word_strokes(word, stroke_map))

    if include_vad:
        vad = _predict_vad(words[["word", "word_label"]], vad_model)
        words = words.merge(vad[["word_label", "valence", "arousal"]], on="word_label", how="left")
        words["vad_source"] = vad_model
    else:
        words["valence"] = np.nan
        words["arousal"] = np.nan
        words["vad_source"] = np.nan

    for col in ["concreteness", "familiarity", "imageability"]:
        words[col] = np.nan
    return words


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return math.nan
    return float(1.0 - np.dot(a, b) / denom)


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    dim_cols = [c for c in df.columns if str(c).startswith("dim_")]
    if not dim_cols:
        raise ValueError(f"No dim_* columns found in embedding file: {path}")
    return {
        str(row["word_label"]): row[dim_cols].to_numpy(dtype=float)
        for _, row in df.iterrows()
    }


def _load_materials(workbook: Path) -> pd.DataFrame:
    spatial = pd.read_excel(workbook, sheet_name=0)
    metaphor = pd.read_excel(workbook, sheet_name=1)
    baseline = pd.read_excel(workbook, sheet_name=2)

    rows: list[dict[str, object]] = []
    for condition, df, sentence_col in [
        ("Spatial", spatial, 1),
        ("Metaphor", metaphor, 1),
    ]:
        for _, row in df.iterrows():
            first_stem = str(row.iloc[2]).replace(".jpg", "")
            second_stem = str(row.iloc[4]).replace(".jpg", "")
            rows.append({
                "condition": condition,
                "source_id": str(row.iloc[0]).replace(".jpg", ""),
                "sentence": row.iloc[sentence_col],
                "first_word": row.iloc[3],
                "second_word": row.iloc[5],
                "first_word_label": _label_from_stem(first_stem),
                "second_word_label": _label_from_stem(second_stem),
            })

    # Current baseline pair logic follows stimuli_template: jx_1/jx_2, jx_3/jx_4, ...
    baseline = baseline.reset_index(drop=True)
    if len(baseline) % 2 != 0:
        raise ValueError("Baseline sheet must contain an even number of words.")
    for i in range(0, len(baseline), 2):
        first = baseline.iloc[i]
        second = baseline.iloc[i + 1]
        first_stem = str(first.iloc[0]).replace(".jpg", "")
        second_stem = str(second.iloc[0]).replace(".jpg", "")
        rows.append({
            "condition": "Baseline",
            "source_id": f"baseline_pair_{i // 2 + 1:02d}",
            "sentence": np.nan,
            "first_word": first.iloc[1],
            "second_word": second.iloc[1],
            "first_word_label": _label_from_stem(first_stem),
            "second_word_label": _label_from_stem(second_stem),
        })

    table = pd.DataFrame(rows)
    table["pair_id_s0"] = np.arange(1, len(table) + 1)
    table["first_word_char_count"] = table["first_word"].map(_char_count)
    table["second_word_char_count"] = table["second_word"].map(_char_count)
    table["pair_char_count_sum"] = table["first_word_char_count"] + table["second_word_char_count"]
    table["pair_char_count_mean"] = table[["first_word_char_count", "second_word_char_count"]].mean(axis=1)
    table["sentence_char_count"] = table["sentence"].map(_sentence_char_count)
    return table


def _merge_lexical_norms(table: pd.DataFrame, lexical_norms: Path | None) -> pd.DataFrame:
    out = table.copy()
    if lexical_norms is None:
        for var in LEXICAL_NORM_COLUMNS:
            out[f"first_{var}"] = np.nan
            out[f"second_{var}"] = np.nan
            out[f"pair_{var}_mean"] = np.nan
            if var in {"stroke_count"}:
                out[f"pair_{var}_sum"] = np.nan
        return out

    if isinstance(lexical_norms, Path):
        norms = pd.read_csv(lexical_norms, sep=None, engine="python")
    else:
        norms = lexical_norms.copy()
    if "word" not in norms.columns:
        raise ValueError("Lexical norms file must contain a 'word' column.")
    available = [c for c in LEXICAL_NORM_COLUMNS if c in norms.columns]
    missing = [c for c in LEXICAL_NORM_COLUMNS if c not in norms.columns]
    for col in missing:
        norms[col] = np.nan
    first = norms[["word", *LEXICAL_NORM_COLUMNS]].rename(
        columns={"word": "first_word", **{c: f"first_{c}" for c in LEXICAL_NORM_COLUMNS}}
    )
    second = norms[["word", *LEXICAL_NORM_COLUMNS]].rename(
        columns={"word": "second_word", **{c: f"second_{c}" for c in LEXICAL_NORM_COLUMNS}}
    )
    out = out.merge(first, on="first_word", how="left").merge(second, on="second_word", how="left")
    for var in LEXICAL_NORM_COLUMNS:
        out[f"pair_{var}_mean"] = out[[f"first_{var}", f"second_{var}"]].mean(axis=1)
        if var == "stroke_count":
            out[f"pair_{var}_sum"] = out[[f"first_{var}", f"second_{var}"]].sum(axis=1, min_count=1)
    out.attrs["available_lexical_norm_columns"] = available
    return out


def _add_embedding_distance(table: pd.DataFrame, embedding_file: Path) -> pd.DataFrame:
    out = table.copy()
    embeddings = _load_embeddings(embedding_file)
    distances = []
    for _, row in out.iterrows():
        a = embeddings.get(str(row["first_word_label"]))
        b = embeddings.get(str(row["second_word_label"]))
        distances.append(_cosine_distance(a, b) if a is not None and b is not None else math.nan)
    out["embedding_cosine_distance"] = distances
    return out


def _load_stimuli_template(path: Path) -> pd.DataFrame:
    stim = pd.read_csv(path)
    stim["stem"] = stim["word_label"].astype(str).map(_stem_from_label)
    return stim


def _coverage(table: pd.DataFrame, stimuli_template: Path) -> pd.DataFrame:
    stim = _load_stimuli_template(stimuli_template)
    rows = []
    for condition in ["Metaphor", "Spatial", "Baseline"]:
        formal = set(stim.loc[stim["condition"] == condition, "word_label"].astype(str))
        got = set(table.loc[table["condition"] == condition, "first_word_label"].astype(str)) | set(
            table.loc[table["condition"] == condition, "second_word_label"].astype(str)
        )
        rows.append({
            "condition": condition,
            "formal_word_n": len(formal),
            "table_word_n": len(got),
            "covered_n": len(formal & got),
            "missing_n": len(formal - got),
            "extra_n": len(got - formal),
            "coverage_rate": len(formal & got) / len(formal) if formal else math.nan,
            "missing_labels": ";".join(sorted(formal - got)),
            "extra_labels": ";".join(sorted(got - formal)),
        })
    return pd.DataFrame(rows)


def _pvalue(groups: list[pd.Series]) -> float:
    if stats is None:
        return math.nan
    clean = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return math.nan
    if len(clean) == 2:
        return float(stats.ttest_ind(clean[0], clean[1], equal_var=False).pvalue)
    return float(stats.kruskal(*clean).pvalue)


def _balance(table: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "first_word_char_count",
        "second_word_char_count",
        "pair_char_count_sum",
        "pair_char_count_mean",
        "sentence_char_count",
        "embedding_cosine_distance",
        *[f"pair_{v}_mean" for v in LEXICAL_NORM_COLUMNS],
        "pair_stroke_count_sum",
    ]
    rows = []
    for var in variables:
        if var not in table.columns:
            continue
        groups = {c: pd.to_numeric(table.loc[table["condition"] == c, var], errors="coerce") for c in ["Metaphor", "Spatial", "Baseline"]}
        rows.append({
            "variable": var,
            "n_metaphor": int(groups["Metaphor"].notna().sum()),
            "n_spatial": int(groups["Spatial"].notna().sum()),
            "n_baseline": int(groups["Baseline"].notna().sum()),
            "mean_metaphor": groups["Metaphor"].mean(),
            "mean_spatial": groups["Spatial"].mean(),
            "mean_baseline": groups["Baseline"].mean(),
            "diff_metaphor_minus_spatial": groups["Metaphor"].mean() - groups["Spatial"].mean(),
            "test": "Welch_t_Metaphor_vs_Spatial" if groups["Baseline"].notna().sum() < 2 else "Kruskal_3_condition",
            "p_value": _pvalue([groups["Metaphor"], groups["Spatial"]] if groups["Baseline"].notna().sum() < 2 else [groups["Metaphor"], groups["Spatial"], groups["Baseline"]]),
        })
    bal = pd.DataFrame(rows)
    if "p_value" in bal.columns:
        bal["q_bh"] = _bh(bal["p_value"])
    return bal


def _bh(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    valid = np.isfinite(arr)
    if not valid.any():
        return pd.Series(out, index=values.index)
    idx = np.flatnonzero(valid)
    order = np.argsort(arr[valid])
    ranked = arr[valid][order]
    m = float(len(ranked))
    adj = ranked * m / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[idx[order]] = np.clip(adj, 0, 1)
    return pd.Series(out, index=values.index)


def _missingness(table: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "embedding_cosine_distance",
        "sentence_char_count",
        *[f"pair_{v}_mean" for v in LEXICAL_NORM_COLUMNS],
        "pair_stroke_count_sum",
    ]
    rows = []
    for var in variables:
        if var not in table.columns:
            continue
        for condition, sub in table.groupby("condition"):
            rows.append({
                "variable": var,
                "condition": condition,
                "n_rows": int(len(sub)),
                "n_missing": int(sub[var].isna().sum()),
                "missing_rate": float(sub[var].isna().mean()),
            })
    return pd.DataFrame(rows)


def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=_default_base_dir())
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--stimuli-template", type=Path)
    parser.add_argument("--lexical-norms", type=Path, default=None, help="Optional CSV/TSV with word and lexical norm columns.")
    parser.add_argument(
        "--auto-lexical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically compute wordfreq frequency and Unihan stroke count when --lexical-norms is not provided.",
    )
    parser.add_argument(
        "--vad-source",
        choices=["none", "hf_model"],
        default="hf_model",
        help="How to obtain valence/arousal. hf_model uses a Chinese VAD regression model.",
    )
    parser.add_argument("--vad-model", default=VAD_MODEL_NAME)
    parser.add_argument("--vad-batch-size", type=int, default=32)
    parser.add_argument("--force-unihan-download", action="store_true")
    parser.add_argument("--paper-output-root", type=Path)
    args = parser.parse_args()

    base_dir = args.base_dir
    workbook = args.workbook or _find_workbook(base_dir)
    embedding_file = args.embedding_file or base_dir / "stimulus_embeddings" / "stimulus_embeddings_bert.tsv"
    stimuli_template = args.stimuli_template or base_dir / "stimuli_template.csv"
    paper_output_root = args.paper_output_root or base_dir / "paper_outputs"
    tables_si = paper_output_root / "tables_si"
    qc_dir = paper_output_root / "qc" / "stimulus_control_s0"
    tables_si.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    table = _load_materials(workbook)
    table = _add_embedding_distance(table, embedding_file)
    lexical_source = None
    auto_norms_path = qc_dir / "stimulus_control_s0_auto_lexical_norms.tsv"
    if args.lexical_norms is not None:
        lexical_source = args.lexical_norms
    elif args.auto_lexical:
        lexical_source = _build_auto_lexical_norms(
            table,
            qc_dir,
            include_vad=args.vad_source == "hf_model",
            vad_model=args.vad_model,
            force_unihan_download=args.force_unihan_download,
        )
        lexical_source.to_csv(auto_norms_path, sep="\t", index=False)
    table = _merge_lexical_norms(table, lexical_source)
    coverage = _coverage(table, stimuli_template)
    balance = _balance(table)
    missing = _missingness(table)

    control_path = tables_si / "table_stimulus_control_s0.tsv"
    balance_path = tables_si / "table_stimulus_balance_s0.tsv"
    coverage_path = qc_dir / "stimulus_control_s0_coverage.tsv"
    missing_path = qc_dir / "stimulus_control_s0_missingness.tsv"
    manifest_path = qc_dir / "stimulus_control_s0_manifest.json"

    table.to_csv(control_path, sep="\t", index=False)
    balance.to_csv(balance_path, sep="\t", index=False)
    coverage.to_csv(coverage_path, sep="\t", index=False)
    missing.to_csv(missing_path, sep="\t", index=False)

    manifest = {
        "workbook": str(workbook),
        "embedding_file": str(embedding_file),
        "stimuli_template": str(stimuli_template),
        "lexical_norms": str(args.lexical_norms) if args.lexical_norms else None,
        "auto_lexical_norms": str(auto_norms_path) if args.lexical_norms is None and args.auto_lexical else None,
        "vad_source": args.vad_source,
        "vad_model": args.vad_model if args.vad_source == "hf_model" else None,
        "control_table": str(control_path),
        "balance_table": str(balance_path),
        "coverage_table": str(coverage_path),
        "missingness_table": str(missing_path),
        "n_pairs": int(len(table)),
        "available_from_workbook": [
            "condition",
            "sentence",
            "first_word",
            "second_word",
            "word labels",
            "word/sentence character counts",
        ],
        "available_from_local_embeddings": ["embedding_cosine_distance"],
        "requires_external_lexical_norms": [
            "concreteness",
            "familiarity",
            "imageability",
        ],
        "automatic_lexical_variables": [
            "word_frequency via wordfreq.zipf_frequency(lang='zh')",
            "stroke_count via Unicode Unihan kTotalStrokes",
            "valence/arousal via VAD model when --vad-source hf_model",
        ],
    }
    manifest_path.write_text(json.dumps(_json_clean(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {control_path}")
    print(f"Wrote {balance_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {missing_path}")
    print(f"Wrote {manifest_path}")
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nBalance:")
    print(balance.to_string(index=False))
    print("\nMissingness:")
    print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
