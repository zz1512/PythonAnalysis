#!/usr/bin/env python3
"""
ROI-level Relation-vector Model-RSA.

This script tests whether local neural geometry encodes pair-level semantic
relations such as target-cue embedding vectors. It is deliberately independent
from `rsa_config.py` so aggregate and multi-ROI-set reruns do not depend on a
single `METAPHOR_ROI_SET` environment variable.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Iterable

import nibabel as nib
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
from common.roi_library import select_roi_masks  # noqa: E402


MODEL_ROLES = {
    "M9_relation_vector_direct": "primary",
    "M9_relation_vector_abs": "primary",
    "M9_relation_vector_reverse": "secondary",
    "M9_relation_vector_direction_only": "secondary",
    "M9_relation_vector_length": "control",
    "M3_embedding_pair_distance": "control",
    "M3_embedding_pair_centroid": "control",
}

DEFAULT_MODELS = [
    "M9_relation_vector_direct",
    "M9_relation_vector_abs",
    "M9_relation_vector_length",
    "M3_embedding_pair_distance",
    "M3_embedding_pair_centroid",
]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    return read_table(path)


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(float("nan"), index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    sorted_values = values.loc[order].to_numpy(dtype=float)
    n = len(sorted_values)
    adjusted = sorted_values * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _rdm_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _cosine_rdm(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 2:
        return np.zeros((samples.shape[0], samples.shape[0]), dtype=float)
    vec = pdist(samples, metric="cosine")
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    out = squareform(vec)
    np.fill_diagonal(out, 0.0)
    return out


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan"), float("nan"), int(valid.sum())
    a = a[valid]
    b = b[valid]
    if np.isclose(np.std(a), 0.0) or np.isclose(np.std(b), 0.0):
        return float("nan"), float("nan"), int(valid.sum())
    rho, pvalue = stats.spearmanr(a, b)
    return (
        float(rho) if np.isfinite(rho) else float("nan"),
        float(pvalue) if np.isfinite(pvalue) else float("nan"),
        int(valid.sum()),
    )


def _load_masked_samples(image_path: Path, mask_path: Path) -> tuple[np.ndarray, int]:
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {image_path}")
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.get_fdata()) > 0
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    samples = data[mask, :].T.astype(np.float64, copy=False)
    return samples, int(mask.sum())


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.reset_index(drop=True).copy()
    if "time" not in out.columns and "stage" in out.columns:
        out["time"] = out["stage"]
    if "word_label" in out.columns:
        out["word_label"] = out["word_label"].astype(str).str.strip()
    if "real_word" in out.columns:
        out["real_word"] = out["real_word"].astype(str).str.strip()
    if "pair_id" in out.columns:
        out["pair_id"] = pd.to_numeric(out["pair_id"], errors="coerce")
    return out


def _metadata_index_for_word(meta: pd.DataFrame, *, word_label: object, real_word: object) -> int | None:
    candidates = []
    word_text = str(word_label).strip()
    real_text = str(real_word).strip()
    if word_text and word_text.lower() != "nan" and "word_label" in meta.columns:
        candidates.append(meta.index[meta["word_label"].astype(str).str.strip().eq(word_text)].tolist())
    if real_text and real_text.lower() != "nan" and "real_word" in meta.columns:
        candidates.append(meta.index[meta["real_word"].astype(str).str.strip().eq(real_text)].tolist())
    for hits in candidates:
        if len(hits) == 1:
            return int(hits[0])
    return None


def _pair_samples(
    samples: np.ndarray,
    metadata: pd.DataFrame,
    pair_manifest: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    relation_vectors: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    used_pair_ids: list[object] = []
    qc_rows: list[dict[str, object]] = []

    for row in pair_manifest.sort_values("pair_id").itertuples(index=False):
        cue_idx = _metadata_index_for_word(
            metadata,
            word_label=getattr(row, "cue_word_label", ""),
            real_word=getattr(row, "cue_real_word", ""),
        )
        target_idx = _metadata_index_for_word(
            metadata,
            word_label=getattr(row, "target_word_label", ""),
            real_word=getattr(row, "target_real_word", ""),
        )
        status = "ok" if cue_idx is not None and target_idx is not None else "missing_pair_member"
        qc_rows.append(
            {
                "condition": getattr(row, "condition"),
                "pair_id": getattr(row, "pair_id"),
                "cue_word_label": getattr(row, "cue_word_label", ""),
                "target_word_label": getattr(row, "target_word_label", ""),
                "cue_index": cue_idx if cue_idx is not None else "",
                "target_index": target_idx if target_idx is not None else "",
                "status": status,
            }
        )
        if status != "ok":
            continue
        cue_pattern = samples[cue_idx, :]
        target_pattern = samples[target_idx, :]
        relation_vectors.append(target_pattern - cue_pattern)
        centroids.append((target_pattern + cue_pattern) / 2.0)
        used_pair_ids.append(getattr(row, "pair_id"))

    if not relation_vectors:
        raise ValueError("No valid pair samples were built.")
    return (
        {
            "relation_vector": np.vstack(relation_vectors),
            "pair_centroid": np.vstack(centroids),
            "pair_ids": np.asarray(used_pair_ids),
        },
        qc_rows,
    )


def _paired_t_summary(post: pd.Series, pre: pd.Series) -> dict[str, float]:
    post_arr = pd.to_numeric(post, errors="coerce").to_numpy(dtype=float)
    pre_arr = pd.to_numeric(pre, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(post_arr) & np.isfinite(pre_arr)
    post_arr = post_arr[valid]
    pre_arr = pre_arr[valid]
    if post_arr.size < 2:
        return {
            "n_subjects": int(post_arr.size),
            "mean_pre": float("nan"),
            "mean_post": float("nan"),
            "mean_delta": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
        }
    diff = post_arr - pre_arr
    t_val, p_val = stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
    sd = float(np.std(diff, ddof=1))
    return {
        "n_subjects": int(post_arr.size),
        "mean_pre": float(np.mean(pre_arr)),
        "mean_post": float(np.mean(post_arr)),
        "mean_delta": float(np.mean(diff)),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": float(np.mean(diff) / sd) if sd > 0 and np.isfinite(sd) else float("nan"),
    }


def _resolve_roi_masks(manifest_path: Path, roi_set: str) -> dict[str, Path]:
    masks = select_roi_masks(manifest_path, roi_set=roi_set, include_flag="include_in_rsa")
    if not masks:
        raise RuntimeError(f"No ROI masks found for roi_set={roi_set} in {manifest_path}")
    return masks


def _subject_dirs(pattern_root: Path) -> list[Path]:
    if not pattern_root.exists():
        raise FileNotFoundError(f"Missing pattern root: {pattern_root}")
    return sorted(path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-"))


def _add_fdr(summary: pd.DataFrame, roi_set: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out.insert(0, "roi_set", roi_set)
    out["model_role"] = out["model"].map(lambda value: MODEL_ROLES.get(str(value), "exploratory"))
    out["is_primary_model"] = out["model_role"].eq("primary")
    out["q_bh_model_family"] = float("nan")
    out["q_bh_primary_family"] = float("nan")
    out["q_bh_model_role_family"] = float("nan")

    for _, idx in out.groupby(["condition", "neural_rdm_type"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_model_family"] = _bh_fdr(out.loc[idx, "p"])
        primary_idx = out.index[out.index.isin(idx) & out["is_primary_model"]]
        if len(primary_idx):
            out.loc[primary_idx, "q_bh_primary_family"] = _bh_fdr(out.loc[primary_idx, "p"])

    for _, idx in out.groupby(["condition", "neural_rdm_type", "model_role"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_model_role_family"] = _bh_fdr(out.loc[idx, "p"])
    return out


def _write_global_table(frame: pd.DataFrame, target: Path, *, roi_set: str) -> None:
    ensure_dir(target.parent)
    if target.exists():
        existing = read_table(target)
        if "roi_set" in existing.columns:
            existing = existing[existing["roi_set"].astype(str) != str(roi_set)]
        frame = pd.concat([existing, frame], ignore_index=True)
    sort_cols = [col for col in ["roi_set", "condition", "neural_rdm_type", "model_role", "roi", "model"] if col in frame.columns]
    write_table(frame.sort_values(sort_cols).reset_index(drop=True), target)


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_masks = _resolve_roi_masks(args.roi_manifest, args.roi_set)
    relation_pairs = read_table(args.relation_pair_manifest)
    relation_pairs["condition"] = relation_pairs["condition"].astype(str).str.strip().str.lower()
    relation_npz = np.load(args.relation_rdm_npz)
    requested_models = list(args.models)
    requested_types = list(args.neural_rdm_types)

    metric_rows: list[dict[str, object]] = []
    cell_qc_rows: list[dict[str, object]] = []
    pair_qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in _subject_dirs(args.pattern_root):
        subject = subject_dir.name
        for roi_name, roi_path in roi_masks.items():
            for condition in args.conditions:
                condition = str(condition).strip().lower()
                condition_pairs = relation_pairs[relation_pairs["condition"].eq(condition)].copy()
                if condition_pairs.empty:
                    continue
                for time in ["pre", "post"]:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    metadata_path = subject_dir / args.metadata_template.format(time=time, condition=condition)
                    cell_base = {
                        "roi_set": args.roi_set,
                        "subject": subject,
                        "roi": roi_name,
                        "condition": condition,
                        "time": time,
                        "image_path": str(image_path),
                        "metadata_path": str(metadata_path),
                        "roi_mask_path": str(roi_path),
                    }
                    try:
                        if not image_path.exists():
                            raise FileNotFoundError(f"Missing pattern: {image_path}")
                        if not metadata_path.exists():
                            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
                        samples, n_voxels = _load_masked_samples(image_path, roi_path)
                        metadata = _normalize_metadata(_read_any_table(metadata_path))
                        if len(metadata) != samples.shape[0]:
                            raise ValueError(f"metadata rows={len(metadata)} but samples={samples.shape[0]}")
                        pair_data, pair_qc = _pair_samples(samples, metadata, condition_pairs)
                        for row in pair_qc:
                            pair_qc_rows.append({**cell_base, **row})
                        n_pairs = int(pair_data["relation_vector"].shape[0])
                        cell_qc_rows.append(
                            {
                                **cell_base,
                                "status": "ok",
                                "skip_reason": "",
                                "n_voxels": int(n_voxels),
                                "n_trials": int(samples.shape[0]),
                                "n_pairs": n_pairs,
                            }
                        )
                        for neural_type in requested_types:
                            if neural_type not in pair_data:
                                raise ValueError(f"Unknown neural_rdm_type={neural_type}")
                            neural_rdm = _cosine_rdm(pair_data[neural_type])
                            neural_vec = _rdm_vector(neural_rdm)
                            for model_name in requested_models:
                                key = f"{condition}__{model_name}"
                                if key not in relation_npz:
                                    raise KeyError(f"Missing model RDM key in npz: {key}")
                                model_rdm = np.asarray(relation_npz[key], dtype=float)
                                if model_rdm.shape[0] != n_pairs:
                                    raise ValueError(
                                        f"Model RDM shape {model_rdm.shape} does not match n_pairs={n_pairs} for {key}"
                                    )
                                rho, p_value, n_edges = _safe_spearman(neural_vec, _rdm_vector(model_rdm))
                                metric_rows.append(
                                    {
                                        **cell_base,
                                        "neural_rdm_type": neural_type,
                                        "model": model_name,
                                        "model_role": MODEL_ROLES.get(model_name, "exploratory"),
                                        "rho": rho,
                                        "p_value": p_value,
                                        "n_pairs": n_pairs,
                                        "n_rdm_edges": n_edges,
                                        "n_voxels": int(n_voxels),
                                    }
                                )
                    except Exception as exc:
                        cell_qc_rows.append(
                            {
                                **cell_base,
                                "status": "failed",
                                "skip_reason": repr(exc),
                                "n_voxels": 0,
                                "n_trials": 0,
                                "n_pairs": 0,
                            }
                        )
                        failure_rows.append(
                            {
                                **cell_base,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        if not args.allow_partial:
                            raise

    metrics = pd.DataFrame(metric_rows)
    cell_qc = pd.DataFrame(cell_qc_rows)
    pair_qc = pd.DataFrame(pair_qc_rows)
    failures = pd.DataFrame(failure_rows)
    return metrics, cell_qc, pair_qc, failures


def summarize(metrics: pd.DataFrame, roi_set: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["roi", "condition", "neural_rdm_type", "model"]
    for keys, frame in metrics.groupby(group_cols, sort=False):
        roi, condition, neural_rdm_type, model = keys
        pivot = frame.pivot_table(index="subject", columns="time", values="rho", aggfunc="mean").dropna()
        if "pre" not in pivot.columns or "post" not in pivot.columns:
            continue
        summary = _paired_t_summary(pivot["post"], pivot["pre"])
        rows.append(
            {
                "roi": roi,
                "condition": condition,
                "neural_rdm_type": neural_rdm_type,
                "model": model,
                **summary,
            }
        )
    return _add_fdr(pd.DataFrame(rows), roi_set)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="ROI-level relation-vector Model-RSA.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-set", required=True)
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--relation-rdm-npz", type=Path, default=base_dir / "paper_outputs" / "qc" / "relation_vectors" / "relation_model_rdms.npz")
    parser.add_argument("--relation-pair-manifest", type=Path, default=base_dir / "paper_outputs" / "qc" / "relation_vectors" / "relation_pair_manifest.tsv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--neural-rdm-types", nargs="+", default=["relation_vector", "pair_centroid"])
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--metadata-template", default="{time}_{condition}_metadata.tsv")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.output_dir = ensure_dir(args.output_dir or (args.paper_output_root / "qc" / f"relation_vector_rsa_{args.roi_set}"))
    metrics, cell_qc, pair_qc, failures = run_analysis(args)

    write_table(metrics, args.output_dir / "relation_vector_subject_metrics.tsv")
    write_table(cell_qc, args.output_dir / "relation_vector_cell_qc.tsv")
    write_table(pair_qc, args.output_dir / "relation_vector_pair_qc.tsv")
    write_table(failures, args.output_dir / "relation_vector_model_failures.tsv")

    if failures.empty and not cell_qc.empty:
        failed_cells = int(cell_qc["status"].astype(str).eq("failed").sum())
    else:
        failed_cells = int(len(failures))
    if failed_cells and not args.allow_partial:
        raise RuntimeError(f"Relation-vector RSA has {failed_cells} failed cells.")

    summary = summarize(metrics, args.roi_set)
    write_table(summary, args.output_dir / "relation_vector_group_summary_fdr.tsv")
    write_table(summary.drop(columns=[col for col in ["q_bh_model_family", "q_bh_primary_family", "q_bh_model_role_family"] if col in summary.columns]), args.output_dir / "relation_vector_group_summary.tsv")

    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")
    if not summary.empty:
        main_table = summary[
            (summary["model"].isin(["M9_relation_vector_direct", "M9_relation_vector_abs"]))
            & (summary["neural_rdm_type"].eq("relation_vector"))
        ].copy()
        _write_global_table(main_table, tables_main / "table_relation_vector_rsa.tsv", roi_set=args.roi_set)
        _write_global_table(summary, tables_si / "table_relation_vector_rsa_fdr.tsv", roi_set=args.roi_set)
        _write_global_table(metrics.assign(roi_set=args.roi_set), tables_si / "table_relation_vector_subject_metrics.tsv", roi_set=args.roi_set)

    save_json(
        {
            "roi_set": args.roi_set,
            "pattern_root": str(args.pattern_root),
            "roi_manifest": str(args.roi_manifest),
            "relation_rdm_npz": str(args.relation_rdm_npz),
            "relation_pair_manifest": str(args.relation_pair_manifest),
            "output_dir": str(args.output_dir),
            "conditions": list(args.conditions),
            "models": list(args.models),
            "neural_rdm_types": list(args.neural_rdm_types),
            "n_metric_rows": int(len(metrics)),
            "n_failed_cells": failed_cells,
            "n_subjects": int(metrics["subject"].nunique()) if not metrics.empty else 0,
            "n_rois": int(metrics["roi"].nunique()) if not metrics.empty else 0,
        },
        args.output_dir / "relation_vector_manifest.json",
    )
    print(f"[relation_vector_rsa] wrote {len(metrics)} subject metric rows and {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
