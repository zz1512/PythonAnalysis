"""
noise_ceiling.py

B5: Noise ceiling (supporting information).

Goal
- Provide a simple, reproducible noise ceiling estimate for:
  1) Step 5C item-wise delta similarity profiles (post - pre) from `rsa_itemwise_details.csv`
  2) Model-RSA neural RDM vectors from `model_rdm_audit/*_rdms.npz` (exported by model_rdm_comparison.py)

Definition (Nili et al., 2014 style; LOSO)
- For each subject i with vector v_i:
  - lower_i = corr(v_i, mean(v_{-i}))
  - upper_i = corr(v_i, mean(v_all))
- Report the mean/median across subjects.

Notes
- This is support information for figures/tables; it does not replace main statistics.
- We default to Spearman correlation, matching typical RSA practice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


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


CONDITION_MAP = {
    "metaphor": "yy",
    "yy": "yy",
    "spatial": "kj",
    "kj": "kj",
    "baseline": "baseline",
}


def _default_base_dir() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

    return Path(BASE_DIR)


def _default_step5c_files(base_dir: Path) -> list[Path]:
    return [
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_main_functional" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature_spatial" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_atlas_robustness" / "rsa_itemwise_details.csv",
    ]


def _default_model_rdm_dirs(base_dir: Path) -> list[Path]:
    qc_root = base_dir / "paper_outputs" / "qc"
    if not qc_root.exists():
        return []
    return sorted([p for p in qc_root.glob("model_rdm_results_*") if p.is_dir()])


def _canonical_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "time" not in out.columns and "stage" in out.columns:
        out["time"] = out["stage"]
    out["time"] = out["time"].astype(str).str.strip().str.lower().map({"pre": "pre", "post": "post", "1": "pre", "2": "post"})
    return out


def _infer_roi_set_from_rsa_file(path: Path) -> str:
    name = path.parent.name
    prefix = "rsa_results_optimized_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return name


def _infer_main_family(roi_name: str) -> str:
    if "Metaphor_gt_Spatial" in roi_name:
        return "Metaphor_gt_Spatial"
    if "Spatial_gt_Metaphor" in roi_name:
        return "Spatial_gt_Metaphor"
    return "main_functional_other"


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    av = a[valid]
    bv = b[valid]
    if np.isclose(np.std(av, ddof=1), 0.0) or np.isclose(np.std(bv, ddof=1), 0.0):
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(av, bv).statistic)
    return float(stats.spearmanr(av, bv, nan_policy="omit").correlation)


def _noise_ceiling_vectors(
    vectors_by_subject: dict[str, np.ndarray],
    *,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    subjects = sorted(vectors_by_subject.keys())
    if len(subjects) < 3:
        subject_rows = pd.DataFrame(
            [{"subject": s, "lower_r": float("nan"), "upper_r": float("nan")} for s in subjects]
        )
        summary = {
            "n_subjects": float(len(subjects)),
            "lower_mean_r": float("nan"),
            "upper_mean_r": float("nan"),
            "lower_median_r": float("nan"),
            "upper_median_r": float("nan"),
        }
        return subject_rows, summary

    # Stack as (n_subjects, n_features) assuming consistent shapes.
    mat = np.vstack([vectors_by_subject[s] for s in subjects]).astype(float)
    mean_all = np.nanmean(mat, axis=0)

    rows: list[dict[str, Any]] = []
    for idx, subject in enumerate(subjects):
        v = mat[idx]
        mean_others = np.nanmean(np.delete(mat, idx, axis=0), axis=0)
        lower_r = _safe_corr(v, mean_others, method)
        upper_r = _safe_corr(v, mean_all, method)
        rows.append({"subject": subject, "lower_r": lower_r, "upper_r": upper_r})

    subject_rows = pd.DataFrame(rows)
    lower = subject_rows["lower_r"].to_numpy(dtype=float)
    upper = subject_rows["upper_r"].to_numpy(dtype=float)
    summary = {
        "n_subjects": float(len(subjects)),
        "lower_mean_r": float(np.nanmean(lower)),
        "upper_mean_r": float(np.nanmean(upper)),
        "lower_median_r": float(np.nanmedian(lower)),
        "upper_median_r": float(np.nanmedian(upper)),
    }
    return subject_rows, summary


def _build_step5c_item_frame(rsa_files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in rsa_files:
        if not Path(path).exists():
            continue
        df = _canonical_time(read_table(path))
        if df.empty:
            continue
        df["roi_set"] = _infer_roi_set_from_rsa_file(Path(path))
        df["condition"] = df["condition"].astype(str).str.strip().str.lower().map(CONDITION_MAP)
        df = df[df["condition"].isin(["yy", "kj", "baseline"])].copy()
        df["word_label"] = df["word_label"].astype(str).str.strip()
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
        df = df.dropna(subset=["subject", "roi", "roi_set", "condition", "time", "word_label", "similarity"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _step5c_scopes(item_frame: pd.DataFrame) -> pd.DataFrame:
    if item_frame.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for roi_set, df in item_frame.groupby("roi_set"):
        pooled = (
            df.groupby(["subject", "condition", "time", "word_label"], as_index=False)["similarity"]
            .mean()
            .assign(roi_scope=f"{roi_set}__all", roi_set=roi_set)
        )
        rows.append(pooled)
        if roi_set == "main_functional":
            fam = df.copy()
            fam["family"] = fam["roi"].astype(str).map(_infer_main_family)
            fam = (
                fam.groupby(["subject", "condition", "time", "word_label", "family"], as_index=False)["similarity"]
                .mean()
                .assign(roi_set=roi_set)
            )
            fam["roi_scope"] = fam["family"].map(lambda x: f"main_functional__{x}")
            fam = fam.drop(columns=["family"])
            rows.append(fam)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = out.drop_duplicates(subset=["subject", "condition", "time", "word_label", "roi_scope"], keep="last")
    return out


def _step5c_noise_ceiling(
    scoped_item_frame: pd.DataFrame,
    *,
    method: str,
    min_subjects_per_feature: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
    - summary table rows
    - per-subject rows (for QC)
    """
    if scoped_item_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = scoped_item_frame.pivot_table(
        index=["subject", "roi_scope", "roi_set", "condition", "word_label"],
        columns="time",
        values="similarity",
        aggfunc="mean",
    ).reset_index()
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        return pd.DataFrame(), pd.DataFrame()

    pivot = pivot.dropna(subset=["pre", "post"]).copy()
    pivot["delta_similarity"] = pivot["post"] - pivot["pre"]

    summary_rows: list[dict[str, Any]] = []
    subject_rows: list[pd.DataFrame] = []

    for (roi_scope, condition), df in pivot.groupby(["roi_scope", "condition"]):
        mat = df.pivot_table(index="subject", columns="word_label", values="delta_similarity", aggfunc="mean")
        if mat.empty:
            continue
        # Feature (item) inclusion: keep columns present in enough subjects.
        frac = mat.notna().mean(axis=0).to_numpy(dtype=float)
        keep_mask = frac >= float(min_subjects_per_feature)
        kept_cols = mat.columns.to_numpy()[keep_mask]
        mat = mat.loc[:, kept_cols] if kept_cols.size else mat.iloc[:, :0]
        if mat.shape[1] < 3 or mat.shape[0] < 3:
            continue

        vectors = {str(subj): mat.loc[subj].to_numpy(dtype=float) for subj in mat.index.astype(str)}
        subj_df, summary = _noise_ceiling_vectors(vectors, method=method)
        subj_df = subj_df.assign(
            analysis="step5c_itemwise_delta",
            roi_scope=roi_scope,
            roi_set=str(df["roi_set"].iloc[0]),
            condition=condition,
            n_features=int(mat.shape[1]),
            method=method,
        )
        subject_rows.append(subj_df)
        summary_rows.append(
            {
                "analysis": "step5c_itemwise_delta",
                "roi_scope": roi_scope,
                "roi_set": str(df["roi_set"].iloc[0]),
                "condition": condition,
                "time": "delta",
                "method": method,
                "n_subjects": int(summary["n_subjects"]),
                "n_features": int(mat.shape[1]),
                "lower_mean_r": summary["lower_mean_r"],
                "upper_mean_r": summary["upper_mean_r"],
                "lower_median_r": summary["lower_median_r"],
                "upper_median_r": summary["upper_median_r"],
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    subject_table = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    return summary_table, subject_table


def _infer_roi_set_from_model_dir(model_dir: Path) -> str:
    name = model_dir.name.strip()
    prefix = "model_rdm_results_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return name


def _iter_model_rdm_audit_npz(model_dir: Path) -> Iterable[tuple[str, str, str, str, Path]]:
    """
    Yield (roi_set, subject, roi, cell_key, npz_path) where cell_key encodes {time}_{condition_group}.
    """
    roi_set = _infer_roi_set_from_model_dir(model_dir)
    audit_root = model_dir / "model_rdm_audit"
    if not audit_root.exists():
        return []
    results: list[tuple[str, str, str, str, Path]] = []
    for npz_path in audit_root.glob("sub-*/*/*_rdms.npz"):
        subject = npz_path.parents[1].name
        roi = npz_path.parent.name
        stem = npz_path.stem
        if stem.endswith("_rdms"):
            stem = stem[: -len("_rdms")]
        # Expect "{time}_{condition_group}"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        cell_key = stem
        results.append((roi_set, subject, roi, cell_key, npz_path))
    return results


def _parse_cell_key(cell_key: str) -> tuple[str, str]:
    parts = str(cell_key).split("_", 1)
    if len(parts) != 2:
        return "unknown", cell_key
    return parts[0], parts[1]


def _model_rsa_noise_ceiling(
    model_dirs: Iterable[Path],
    *,
    method: str,
    min_subjects_per_feature: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: dict[tuple[str, str, str, str], list[tuple[str, Path]]] = {}
    # key: (roi_set, roi, time, condition_group) -> [(subject, npz_path)]
    for model_dir in model_dirs:
        for roi_set, subject, roi, cell_key, npz_path in _iter_model_rdm_audit_npz(Path(model_dir)):
            time, condition_group = _parse_cell_key(cell_key)
            key = (roi_set, roi, time, condition_group)
            groups.setdefault(key, []).append((subject, npz_path))

    summary_rows: list[dict[str, Any]] = []
    subject_rows: list[pd.DataFrame] = []

    for (roi_set, roi, time, condition_group), entries in sorted(groups.items()):
        feature_maps_by_subject: dict[str, dict[str, float]] = {}
        for subject, npz_path in entries:
            try:
                payload = np.load(npz_path)
                if "neural_rdm" not in payload or "pair_i" not in payload or "pair_j" not in payload:
                    continue
                metadata_path = npz_path.with_name(npz_path.name.replace("_rdms.npz", "_metadata.tsv"))
                if not metadata_path.exists():
                    continue
                metadata = read_table(metadata_path).reset_index(drop=True)
            except Exception:
                continue

            if metadata.empty or int(payload["neural_rdm"].size) < 3:
                continue

            labels: list[str] = []
            for row in metadata.itertuples(index=False):
                condition = str(getattr(row, "condition", "")).strip().lower()
                word_label = str(getattr(row, "word_label", "")).strip()
                unique_label = str(getattr(row, "unique_label", "")).strip()
                pair_id = str(getattr(row, "pair_id", "")).strip()
                if word_label and word_label.lower() != "nan":
                    base_label = word_label
                elif unique_label and unique_label.lower() != "nan":
                    base_label = unique_label
                elif pair_id and pair_id.lower() != "nan":
                    base_label = pair_id
                else:
                    base_label = f"trial_{len(labels)}"
                labels.append(f"{condition}::{base_label}" if condition else base_label)

            pair_i = np.asarray(payload["pair_i"], dtype=int).ravel()
            pair_j = np.asarray(payload["pair_j"], dtype=int).ravel()
            vec = np.asarray(payload["neural_rdm"], dtype=float).ravel()
            if len(labels) == 0 or pair_i.size != vec.size or pair_j.size != vec.size:
                continue

            feature_map: dict[str, float] = {}
            for idx_a, idx_b, value in zip(pair_i, pair_j, vec):
                if idx_a >= len(labels) or idx_b >= len(labels):
                    continue
                key = "||".join(sorted((labels[int(idx_a)], labels[int(idx_b)])))
                feature_map[key] = float(value)
            if len(feature_map) < 3:
                continue
            feature_maps_by_subject[str(subject)] = feature_map

        if len(feature_maps_by_subject) < 3:
            continue

        feature_counts = pd.Series(
            [key for fmap in feature_maps_by_subject.values() for key in fmap.keys()],
            dtype="string",
        ).value_counts()
        required = float(min_subjects_per_feature) * len(feature_maps_by_subject)
        kept_features = sorted(feature_counts[feature_counts >= required].index.tolist())
        if len(kept_features) < 3:
            continue

        vectors_by_subject = {
            subject: np.asarray([fmap.get(key, np.nan) for key in kept_features], dtype=float)
            for subject, fmap in feature_maps_by_subject.items()
        }
        subj_df, summary = _noise_ceiling_vectors(vectors_by_subject, method=method)
        subj_df = subj_df.assign(
            analysis="model_rsa_neural_rdm",
            roi_scope=str(roi),
            roi_set=str(roi_set),
            condition=str(condition_group),
            time=str(time),
            n_features=int(len(kept_features)),
            method=method,
        )
        subject_rows.append(subj_df)
        summary_rows.append(
            {
                "analysis": "model_rsa_neural_rdm",
                "roi_scope": str(roi),
                "roi_set": str(roi_set),
                "condition": str(condition_group),
                "time": str(time),
                "method": method,
                "n_subjects": int(summary["n_subjects"]),
                "n_features": int(len(kept_features)),
                "lower_mean_r": summary["lower_mean_r"],
                "upper_mean_r": summary["upper_mean_r"],
                "lower_median_r": summary["lower_median_r"],
                "upper_median_r": summary["upper_median_r"],
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    subject_table = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    return summary_table, subject_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LOSO noise ceilings for Step 5C and Model-RSA.")
    parser.add_argument("--rsa-details", nargs="*", type=Path, default=None,
                        help="One or more rsa_itemwise_details files for Step 5C. "
                             "Default: main_functional + literature + literature_spatial under paper_outputs/qc.")
    parser.add_argument("--model-rdm-dirs", nargs="*", type=Path, default=None,
                        help="One or more model_rdm_results_* directories produced by model_rdm_comparison.py. "
                             "Default: auto-detect under paper_outputs/qc/model_rdm_results_*.")
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--min-subjects-per-feature", type=float, default=0.8,
                        help="Keep Step 5C items / Model-RSA pair features present in at least this fraction of subjects.")
    args = parser.parse_args()

    base_dir = _default_base_dir()
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_dir = ensure_dir(paper_root / "qc")

    rsa_files = args.rsa_details or _default_step5c_files(base_dir)
    model_dirs = args.model_rdm_dirs or _default_model_rdm_dirs(base_dir)

    # Step 5C: item-wise delta similarity profile noise ceiling.
    item_frame = _build_step5c_item_frame(rsa_files)
    scoped = _step5c_scopes(item_frame)
    step5c_summary, step5c_subject = _step5c_noise_ceiling(
        scoped,
        method=args.method,
        min_subjects_per_feature=args.min_subjects_per_feature,
    )

    # Model-RSA: neural RDM vector noise ceiling (uses audit exports).
    model_summary, model_subject = _model_rsa_noise_ceiling(
        model_dirs,
        method=args.method,
        min_subjects_per_feature=args.min_subjects_per_feature,
    )

    combined = pd.concat([step5c_summary, model_summary], ignore_index=True) if not step5c_summary.empty or not model_summary.empty else pd.DataFrame()
    write_table(combined, tables_si / "table_noise_ceiling.tsv")
    if not step5c_subject.empty:
        write_table(step5c_subject, qc_dir / "noise_ceiling_step5c_subject.tsv")
    if not model_subject.empty:
        write_table(model_subject, qc_dir / "noise_ceiling_model_rsa_subject.tsv")
    save_json(
        {
            "rsa_details_files": [str(p) for p in rsa_files],
            "model_rdm_dirs": [str(p) for p in model_dirs],
            "method": args.method,
            "min_subjects_per_feature": float(args.min_subjects_per_feature),
            "n_step5c_summary_rows": int(len(step5c_summary)),
            "n_step5c_subject_rows": int(len(step5c_subject)),
            "n_model_summary_rows": int(len(model_summary)),
            "n_model_subject_rows": int(len(model_subject)),
            "output_table": str(tables_si / "table_noise_ceiling.tsv"),
        },
        qc_dir / "noise_ceiling_meta.json",
    )


if __name__ == "__main__":
    main()
