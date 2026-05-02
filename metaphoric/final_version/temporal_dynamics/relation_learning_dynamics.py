#!/usr/bin/env python3
"""
Learning-stage relation-vector trajectory.

Learning patterns contain pair/item-level trials repeated across run 3 and run 4.
This script tests whether each run's pair-pattern RDM aligns with relation-vector
model RDMs, then summarizes run4 - run3 changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import traceback

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
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _load_image_data(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def _cosine_rdm(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 2:
        return np.zeros((samples.shape[0], samples.shape[0]), dtype=float)
    vec = pdist(samples, metric="cosine")
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    out = squareform(vec)
    np.fill_diagonal(out, 0.0)
    return out


def _rdm_vec(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


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
    return float(rho), float(pvalue), int(valid.sum())


def _pair_index_from_label(value: object) -> int | None:
    text = str(value).strip()
    match = re.search(r"_(\d+)$", text)
    if match:
        return int(match.group(1))
    try:
        return int(float(text))
    except Exception:
        return None


def _build_model_lookup(pair_manifest: pd.DataFrame, relation_npz: np.lib.npyio.NpzFile, models: list[str]) -> dict[tuple[str, str], dict[str, object]]:
    manifest = pair_manifest.copy()
    manifest["condition"] = manifest["condition"].astype(str).str.strip().str.lower()
    manifest["pair_index"] = manifest["cue_word_label"].map(_pair_index_from_label)
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for condition, cond_frame in manifest.groupby("condition", sort=False):
        cond_frame = cond_frame.sort_values("pair_id").reset_index(drop=True)
        pair_indices = cond_frame["pair_index"].astype(int).to_numpy()
        for model in models:
            key = f"{condition}__{model}"
            if key not in relation_npz:
                raise KeyError(f"Missing model RDM key: {key}")
            lookup[(condition, model)] = {
                "rdm": np.asarray(relation_npz[key], dtype=float),
                "pair_indices": pair_indices,
            }
    return lookup


def _normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    out["run_num"] = pd.to_numeric(out["run"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if "pair_id" not in out.columns:
        out["pair_id"] = out.get("unique_label", pd.Series(index=out.index, dtype=object))
    out["pair_index"] = out["pair_id"].map(_pair_index_from_label)
    if out["pair_index"].isna().any() and "unique_label" in out.columns:
        out.loc[out["pair_index"].isna(), "pair_index"] = out.loc[out["pair_index"].isna(), "unique_label"].map(_pair_index_from_label)
    return out


def _run_samples(samples: np.ndarray, metadata: pd.DataFrame, run_num: int) -> tuple[np.ndarray, np.ndarray]:
    subset = metadata[metadata["run_num"].eq(run_num)].copy()
    subset = subset[np.isfinite(pd.to_numeric(subset["pair_index"], errors="coerce"))].copy()
    if subset.empty:
        return np.empty((0, samples.shape[1])), np.asarray([], dtype=int)
    subset["pair_index"] = subset["pair_index"].astype(int)
    rows = []
    indices = []
    work = subset.reset_index().rename(columns={"index": "_row"})
    for pair_index, group in work.groupby("pair_index", sort=True):
        row_indices = group["_row"].to_numpy(dtype=int)
        rows.append(np.nanmean(samples[row_indices, :], axis=0))
        indices.append(int(pair_index))
    return np.vstack(rows), np.asarray(indices, dtype=int)


def _paired_t_summary(run4: pd.Series, run3: pd.Series) -> dict[str, float]:
    a = pd.to_numeric(run4, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(run3, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2:
        return {"n_subjects": int(a.size), "mean_run3": np.nan, "mean_run4": np.nan, "mean_delta": np.nan, "t": np.nan, "p": np.nan, "cohens_dz": np.nan}
    diff = a - b
    tval, pval = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
    sd = float(np.std(diff, ddof=1))
    return {
        "n_subjects": int(a.size),
        "mean_run3": float(np.mean(b)),
        "mean_run4": float(np.mean(a)),
        "mean_delta": float(np.mean(diff)),
        "t": float(tval) if np.isfinite(tval) else np.nan,
        "p": float(pval) if np.isfinite(pval) else np.nan,
        "cohens_dz": float(np.mean(diff) / sd) if sd > 0 and np.isfinite(sd) else np.nan,
    }


def _resolve_roi_masks(manifest_path: Path, roi_sets: list[str]) -> dict[str, dict[str, Path]]:
    resolved = {}
    for roi_set in roi_sets:
        masks = select_roi_masks(manifest_path, roi_set=roi_set, include_flag="include_in_rsa")
        if not masks:
            raise RuntimeError(f"No ROI masks found for roi_set={roi_set}")
        resolved[roi_set] = {name: Path(path) for name, path in masks.items()}
    return resolved


def _subject_dirs(pattern_root: Path) -> list[Path]:
    return sorted(path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-"))


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_sets = _resolve_roi_masks(args.roi_manifest, args.roi_sets)
    all_masks = {str(path): _load_mask(path) for masks in roi_sets.values() for path in masks.values()}
    relation_npz = np.load(args.relation_rdm_npz)
    pair_manifest = read_table(args.relation_pair_manifest)
    model_lookup = _build_model_lookup(pair_manifest, relation_npz, list(args.models))

    metric_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in _subject_dirs(args.pattern_root):
        subject = subject_dir.name
        for condition in args.conditions:
            condition = str(condition).strip().lower()
            image_path = subject_dir / args.filename_template.format(condition=condition)
            metadata_path = subject_dir / args.metadata_template.format(condition=condition)
            try:
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                if not metadata_path.exists():
                    raise FileNotFoundError(metadata_path)
                data = _load_image_data(image_path)
                metadata = _normalize_metadata(_read_any_table(metadata_path))
                if len(metadata) != data.shape[3]:
                    raise ValueError(f"metadata rows={len(metadata)} but samples={data.shape[3]}")
            except Exception as exc:
                failure_rows.append({"subject": subject, "condition": condition, "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
                if not args.allow_partial:
                    raise
                continue

            for roi_set, masks in roi_sets.items():
                for roi, roi_path in masks.items():
                    base = {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi,
                        "condition": condition,
                        "image_path": str(image_path),
                        "metadata_path": str(metadata_path),
                        "roi_mask_path": str(roi_path),
                    }
                    try:
                        mask = all_masks[str(roi_path)]
                        if data.shape[:3] != mask.shape:
                            raise ValueError(f"Image/mask shape mismatch: {image_path} vs {roi_path}")
                        samples = data[mask, :].T.astype(np.float64, copy=False)
                        for run_num in args.runs:
                            run_patterns, learned_indices = _run_samples(samples, metadata, int(run_num))
                            qc_rows.append(
                                {
                                    **base,
                                    "run": int(run_num),
                                    "status": "ok" if len(learned_indices) >= args.min_pairs else "too_few_pairs",
                                    "n_pairs": int(len(learned_indices)),
                                    "n_voxels": int(mask.sum()),
                                }
                            )
                            if len(learned_indices) < args.min_pairs:
                                continue
                            learned_map = {int(pair_index): pos for pos, pair_index in enumerate(learned_indices)}
                            for model in args.models:
                                lookup = model_lookup[(condition, model)]
                                model_indices = np.asarray(lookup["pair_indices"], dtype=int)
                                common = [idx for idx in model_indices.tolist() if idx in learned_map]
                                if len(common) < args.min_pairs:
                                    continue
                                sample_pos = [learned_map[idx] for idx in common]
                                model_pos = [int(np.where(model_indices == idx)[0][0]) for idx in common]
                                neural_rdm = _cosine_rdm(run_patterns[sample_pos, :])
                                model_rdm = np.asarray(lookup["rdm"], dtype=float)[np.ix_(model_pos, model_pos)]
                                rho, pvalue, n_edges = _safe_spearman(_rdm_vec(neural_rdm), _rdm_vec(model_rdm))
                                metric_rows.append(
                                    {
                                        **base,
                                        "run": int(run_num),
                                        "model": model,
                                        "model_role": MODEL_ROLES.get(model, "exploratory"),
                                        "rho": rho,
                                        "p_value": pvalue,
                                        "n_pairs": int(len(common)),
                                        "n_rdm_edges": n_edges,
                                        "n_voxels": int(mask.sum()),
                                        "trajectory_type": "run3_to_run4",
                                    }
                                )
                    except Exception as exc:
                        qc_rows.append({**base, "run": "", "status": "failed", "n_pairs": 0, "n_voxels": 0, "skip_reason": repr(exc)})
                        failure_rows.append({**base, "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
                        if not args.allow_partial:
                            raise

    return pd.DataFrame(metric_rows), pd.DataFrame(qc_rows), pd.DataFrame(failure_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["roi_set", "roi", "condition", "model"]
    for keys, frame in metrics.groupby(group_cols, sort=False):
        roi_set, roi, condition, model = keys
        pivot = frame.pivot_table(index="subject", columns="run", values="rho", aggfunc="mean")
        if 3 not in pivot.columns or 4 not in pivot.columns:
            continue
        summary = _paired_t_summary(pivot[4], pivot[3])
        rows.append(
            {
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "model": model,
                "model_role": MODEL_ROLES.get(model, "exploratory"),
                "is_primary_model": MODEL_ROLES.get(model, "exploratory") == "primary",
                **summary,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_model_family"] = np.nan
    out["q_bh_primary_family"] = np.nan
    out["q_bh_model_role_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "condition"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_model_family"] = _bh_fdr(out.loc[idx, "p"])
        primary_idx = out.index[out.index.isin(idx) & out["is_primary_model"]]
        if len(primary_idx):
            out.loc[primary_idx, "q_bh_primary_family"] = _bh_fdr(out.loc[primary_idx, "p"])
    for _, idx in out.groupby(["roi_set", "condition", "model_role"], dropna=False).groups.items():
        out.loc[list(idx), "q_bh_model_role_family"] = _bh_fdr(out.loc[list(idx), "p"])
    return out.sort_values(["q_bh_primary_family", "p"], na_position="last").reset_index(drop=True)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Learning-stage relation-vector trajectory.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-sets", nargs="+", default=["main_functional", "literature", "literature_spatial"])
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--relation-rdm-npz", type=Path, default=base_dir / "paper_outputs" / "qc" / "relation_vectors" / "relation_model_rdms.npz")
    parser.add_argument("--relation-pair-manifest", type=Path, default=base_dir / "paper_outputs" / "qc" / "relation_vectors" / "relation_pair_manifest.tsv")
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--filename-template", default="learn_{condition}.nii.gz")
    parser.add_argument("--metadata-template", default="learn_{condition}_metadata.tsv")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.paper_output_root / "qc" / "relation_learning_dynamics")
    metrics, qc, failures = run_analysis(args)
    summary = summarize(metrics)

    write_table(metrics, out_dir / "relation_learning_dynamics_long.tsv")
    write_table(qc, out_dir / "relation_learning_window_qc.tsv")
    write_table(failures, out_dir / "relation_learning_failures.tsv")
    write_table(summary, out_dir / "relation_learning_group_summary_fdr.tsv")
    write_table(metrics, args.paper_output_root / "tables_si" / "table_relation_learning_dynamics_subject.tsv")
    main_summary = summary[summary["is_primary_model"].astype(bool)].copy() if not summary.empty else summary
    write_table(main_summary, args.paper_output_root / "tables_main" / "table_relation_learning_dynamics.tsv")
    save_json(
        {
            "pattern_root": str(args.pattern_root),
            "roi_sets": list(args.roi_sets),
            "conditions": list(args.conditions),
            "models": list(args.models),
            "runs": list(args.runs),
            "n_metric_rows": int(len(metrics)),
            "n_qc_rows": int(len(qc)),
            "n_failures": int(len(failures)),
            "trajectory_type": "run3_to_run4_pair_pattern_rdm_alignment",
            "interpretation_note": "Learning-stage patterns are pair/item patterns, not separate cue/target word patterns.",
        },
        out_dir / "relation_learning_manifest.json",
    )
    if len(failures) and not args.allow_partial:
        raise RuntimeError(f"relation learning dynamics had {len(failures)} failures")
    print(f"[relation-learning] wrote {len(metrics)} metric rows and {len(summary)} summary rows to {out_dir}")


if __name__ == "__main__":
    main()
