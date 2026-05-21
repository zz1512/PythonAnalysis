#!/usr/bin/env python3
"""
Build relation-vector model RDMs for the PNAS-oriented mechanism analysis.

The key model treats each learned pair as a directed semantic transformation:

    relation_vector = target_embedding - cue_embedding

Outputs are condition-specific pair manifests and model RDMs that can be used by
`relation_vector_rsa.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402
from common.stimulus_text_mapping import attach_real_word_columns  # noqa: E402


CONDITION_ALIASES = {
    "metaphor": "yy",
    "yy": "yy",
    "yyw": "yy",
    "yyew": "yy",
    "spatial": "kj",
    "kj": "kj",
    "kjw": "kj",
    "kjew": "kj",
    "baseline": "baseline",
    "jx": "baseline",
}

MODEL_ROLES = {
    "M9_relation_vector_direct": "primary",
    "M9_relation_vector_abs": "primary",
    "M9_relation_vector_reverse": "secondary",
    "M9_relation_vector_direction_only": "secondary",
    "M9_relation_vector_length": "control",
    "M3_embedding_pair_distance": "control",
    "M3_embedding_pair_centroid": "control",
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _normalize_condition(value: object) -> str | None:
    text = str(value).strip().lower()
    return CONDITION_ALIASES.get(text)


def _read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    return read_table(path)


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in frame.columns if str(col).startswith("dim_")]
    if not cols:
        raise ValueError("Embedding table must contain columns named dim_0, dim_1, ...")
    return sorted(cols, key=lambda col: int(str(col).split("_", 1)[1]))


def _load_embeddings(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    frame = _read_any_table(path).copy()
    dim_cols = _embedding_columns(frame)
    lookup: dict[str, np.ndarray] = {}
    key_cols = [col for col in ["word_label", "real_word"] if col in frame.columns]
    if not key_cols:
        raise ValueError(f"Embedding table {path} must contain word_label or real_word.")
    for row in frame.itertuples(index=False):
        values = np.asarray([getattr(row, col) for col in dim_cols], dtype=float)
        if not np.isfinite(values).all():
            continue
        for key_col in key_cols:
            key = str(getattr(row, key_col)).strip()
            if key and key.lower() != "nan":
                lookup[key] = values
    return frame, lookup, dim_cols


def _load_stimuli(path: Path, *, condition_col: str, pair_col: str, word_col: str) -> pd.DataFrame:
    frame = _read_any_table(path).copy()
    if "real_word" not in frame.columns and "word_label" in frame.columns:
        frame = attach_real_word_columns(frame, column_map={"word_label": "real_word"})
    if word_col not in frame.columns and word_col == "real_word":
        frame = attach_real_word_columns(frame, column_map={"word_label": "real_word"})
    required = {condition_col, pair_col, word_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stimulus template missing columns {sorted(missing)}. Available: {list(frame.columns)}")
    frame["condition_norm"] = frame[condition_col].map(_normalize_condition)
    frame["pair_id"] = frame[pair_col]
    frame["word_key"] = frame[word_col].astype(str).str.strip()
    if "sort_id" in frame.columns:
        frame["_sort_key"] = pd.to_numeric(frame["sort_id"], errors="coerce")
    else:
        frame["_sort_key"] = np.arange(len(frame), dtype=float)
    return frame


def _vector_for_word(
    row: pd.Series,
    embedding_lookup: dict[str, np.ndarray],
    *,
    word_col: str,
) -> tuple[np.ndarray | None, str, str]:
    candidates: list[str] = []
    for col in [word_col, "word_label", "real_word"]:
        if col in row.index:
            value = str(row[col]).strip()
            if value and value.lower() != "nan" and value not in candidates:
                candidates.append(value)
    for key in candidates:
        if key in embedding_lookup:
            return embedding_lookup[key], key, ""
    return None, candidates[0] if candidates else "", "missing_embedding"


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 0:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def _cosine_rdm(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    distances = pdist(matrix, metric="cosine")
    distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
    out = squareform(distances)
    np.fill_diagonal(out, 0.0)
    return out


def _scalar_distance_rdm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.abs(values[:, None] - values[None, :])
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _rdm_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _build_model_rdms(pair_frame: pd.DataFrame, vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pair_ids = pair_frame["pair_uid"].tolist()
    direct = np.vstack([vectors[f"{pair_id}__direct"] for pair_id in pair_ids])
    reverse = -direct
    abs_direct = np.abs(direct)
    direction_only = np.vstack([_unit_vector(row) for row in direct])
    lengths = np.linalg.norm(direct, axis=1)
    centroids = np.vstack([vectors[f"{pair_id}__centroid"] for pair_id in pair_ids])
    within_distances = np.asarray([vectors[f"{pair_id}__within_distance"][0] for pair_id in pair_ids], dtype=float)

    return {
        "M9_relation_vector_direct": _cosine_rdm(direct),
        "M9_relation_vector_reverse": _cosine_rdm(reverse),
        "M9_relation_vector_abs": _cosine_rdm(abs_direct),
        "M9_relation_vector_direction_only": _cosine_rdm(direction_only),
        "M9_relation_vector_length": _scalar_distance_rdm(lengths),
        "M3_embedding_pair_distance": _scalar_distance_rdm(within_distances),
        "M3_embedding_pair_centroid": _cosine_rdm(centroids),
    }


def _collinearity_rows(rdm_store: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in sorted({key.split("__", 1)[0] for key in rdm_store}):
        model_names = sorted(key.split("__", 1)[1] for key in rdm_store if key.startswith(f"{condition}__"))
        for idx, model_a in enumerate(model_names):
            vec_a = _rdm_vector(rdm_store[f"{condition}__{model_a}"])
            for model_b in model_names[idx + 1 :]:
                vec_b = _rdm_vector(rdm_store[f"{condition}__{model_b}"])
                valid = np.isfinite(vec_a) & np.isfinite(vec_b)
                if valid.sum() < 3 or np.isclose(np.std(vec_a[valid]), 0.0) or np.isclose(np.std(vec_b[valid]), 0.0):
                    rho = float("nan")
                    p_value = float("nan")
                else:
                    rho, p_value = stats.spearmanr(vec_a[valid], vec_b[valid])
                    rho = float(rho) if np.isfinite(rho) else float("nan")
                    p_value = float(p_value) if np.isfinite(p_value) else float("nan")
                rows.append(
                    {
                        "condition": condition,
                        "model_a": model_a,
                        "model_b": model_b,
                        "spearman_r": rho,
                        "p_value": p_value,
                        "abs_r": abs(rho) if np.isfinite(rho) else float("nan"),
                        "high_collinearity": bool(np.isfinite(rho) and abs(rho) > 0.70),
                    }
                )
    return rows


def _build_pairs(
    stimuli: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    conditions: Iterable[str],
    word_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    condition_set = {_normalize_condition(item) or str(item).strip().lower() for item in conditions}
    work = stimuli[stimuli["condition_norm"].isin(condition_set)].copy()
    if work.empty:
        raise ValueError(f"No stimuli remained for conditions: {sorted(condition_set)}")

    pair_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}

    for (condition, pair_id), group in work.groupby(["condition_norm", "pair_id"], sort=True):
        group = group.sort_values("_sort_key").reset_index(drop=True)
        if len(group) != 2:
            for _, row in group.iterrows():
                coverage_rows.append(
                    {
                        "condition": condition,
                        "pair_id": pair_id,
                        "word_label": row.get("word_label", ""),
                        "real_word": row.get("real_word", ""),
                        "lookup_key": "",
                        "status": f"invalid_pair_size_{len(group)}",
                    }
                )
            continue

        cue = group.iloc[0]
        target = group.iloc[1]
        cue_vec, cue_key, cue_status = _vector_for_word(cue, embedding_lookup, word_col=word_col)
        target_vec, target_key, target_status = _vector_for_word(target, embedding_lookup, word_col=word_col)
        coverage_rows.extend(
            [
                {
                    "condition": condition,
                    "pair_id": pair_id,
                    "role": "cue",
                    "word_label": cue.get("word_label", ""),
                    "real_word": cue.get("real_word", ""),
                    "lookup_key": cue_key,
                    "status": cue_status or "ok",
                },
                {
                    "condition": condition,
                    "pair_id": pair_id,
                    "role": "target",
                    "word_label": target.get("word_label", ""),
                    "real_word": target.get("real_word", ""),
                    "lookup_key": target_key,
                    "status": target_status or "ok",
                },
            ]
        )
        if cue_vec is None or target_vec is None:
            continue

        pair_uid = f"{condition}:{pair_id}"
        direct = target_vec - cue_vec
        centroid = (target_vec + cue_vec) / 2.0
        within_distance = float(pdist(np.vstack([cue_vec, target_vec]), metric="cosine")[0])
        pair_rows.append(
            {
                "pair_uid": pair_uid,
                "condition": condition,
                "pair_id": pair_id,
                "cue_word_label": cue.get("word_label", ""),
                "target_word_label": target.get("word_label", ""),
                "cue_real_word": cue.get("real_word", ""),
                "target_real_word": target.get("real_word", ""),
                "cue_lookup_key": cue_key,
                "target_lookup_key": target_key,
                "relation_norm": float(np.linalg.norm(direct)),
                "pair_embedding_cosine_distance": within_distance,
            }
        )
        vectors[f"{pair_uid}__direct"] = direct
        vectors[f"{pair_uid}__centroid"] = centroid
        vectors[f"{pair_uid}__within_distance"] = np.asarray([within_distance], dtype=float)

    return pd.DataFrame(pair_rows), pd.DataFrame(coverage_rows), vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relation-vector model RDMs.")
    base_dir = _default_base_dir()
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--embedding-file", type=Path, default=base_dir / "stimulus_embeddings" / "stimulus_embeddings_bert.tsv")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "relation_vectors")
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj"])
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--pair-col", default="pair_id")
    parser.add_argument("--word-col", default="word_label")
    parser.add_argument("--fail-on-missing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    embedding_frame, embedding_lookup, dim_cols = _load_embeddings(args.embedding_file)
    stimuli = _load_stimuli(
        args.stimuli_template,
        condition_col=args.condition_col,
        pair_col=args.pair_col,
        word_col=args.word_col,
    )

    pair_manifest, coverage, vectors = _build_pairs(
        stimuli,
        embedding_lookup,
        conditions=args.conditions,
        word_col=args.word_col,
    )
    write_table(coverage, output_dir / "relation_embedding_coverage.tsv")
    if pair_manifest.empty:
        raise RuntimeError("No valid relation pairs were built. Check coverage table.")

    bad_coverage = coverage[coverage["status"].astype(str).ne("ok")]
    if args.fail_on_missing and not bad_coverage.empty:
        raise RuntimeError(
            f"Relation vector coverage has {len(bad_coverage)} non-ok rows. "
            f"See {output_dir / 'relation_embedding_coverage.tsv'}"
        )

    rdm_store: dict[str, np.ndarray] = {}
    rdm_manifest_rows: list[dict[str, object]] = []
    for condition, condition_pairs in pair_manifest.groupby("condition", sort=True):
        condition_pairs = condition_pairs.sort_values("pair_id").reset_index(drop=True)
        model_rdms = _build_model_rdms(condition_pairs, vectors)
        for model_name, matrix in model_rdms.items():
            key = f"{condition}__{model_name}"
            rdm_store[key] = matrix.astype(np.float32)
            rdm_manifest_rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "model_role": MODEL_ROLES.get(model_name, "exploratory"),
                    "npz_key": key,
                    "n_pairs": int(matrix.shape[0]),
                    "shape": json.dumps(list(matrix.shape)),
                    "has_nan": bool(np.isnan(matrix).any()),
                    "symmetric": bool(np.allclose(matrix, matrix.T, equal_nan=False)),
                }
            )

    write_table(pair_manifest, output_dir / "relation_pair_manifest.tsv")
    write_table(pd.DataFrame(rdm_manifest_rows), output_dir / "relation_model_rdm_manifest.tsv")
    write_table(pd.DataFrame(_collinearity_rows(rdm_store)), output_dir / "relation_model_collinearity.tsv")
    np.savez_compressed(output_dir / "relation_model_rdms.npz", **rdm_store)
    save_json(
        {
            "stimuli_template": str(args.stimuli_template),
            "embedding_file": str(args.embedding_file),
            "output_dir": str(output_dir),
            "conditions": list(args.conditions),
            "condition_col": args.condition_col,
            "pair_col": args.pair_col,
            "word_col": args.word_col,
            "embedding_rows": int(len(embedding_frame)),
            "embedding_dim": int(len(dim_cols)),
            "n_pairs": int(len(pair_manifest)),
            "n_coverage_rows": int(len(coverage)),
            "n_bad_coverage_rows": int(len(bad_coverage)),
            "models": sorted({row["model"] for row in rdm_manifest_rows}),
        },
        output_dir / "relation_vectors_manifest.json",
    )
    print(f"[build_relation_vectors] wrote {len(pair_manifest)} pairs and {len(rdm_store)} model RDMs to {output_dir}")


if __name__ == "__main__":
    main()
