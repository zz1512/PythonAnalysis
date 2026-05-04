"""Learning-stage YY/KJ decoding in planned meta ROIs.

The learning patterns are pair/item-level LSS patterns from run 3 and run 4,
not separate word-role patterns.  This script therefore tests within-run
pair-held-out condition decoding and cross-run generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


CONDITION_TO_LABEL = {"kj": 0, "yy": 1}
DEFAULT_ROIS = [
    "meta_R_PPA_PHG",
    "meta_R_PPC_SPL",
    "meta_R_temporal_pole",
    "meta_R_IFG",
    "meta_R_hippocampus",
]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p.astype(float))
    if not valid.any():
        return out
    values = p.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _cohens_dz(values: np.ndarray, chance: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)] - chance
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _one_sample_accuracy(values: pd.Series, chance: float = 0.5) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return {"n_subjects": int(arr.size), "mean_accuracy": np.nan, "mean_minus_chance": np.nan, "t": np.nan, "p": np.nan, "cohens_dz": np.nan}
    t_val, p_val = stats.ttest_1samp(arr, chance, nan_policy="omit")
    return {
        "n_subjects": int(arr.size),
        "mean_accuracy": float(arr.mean()),
        "mean_minus_chance": float(arr.mean() - chance),
        "t": float(t_val),
        "p": float(p_val),
        "cohens_dz": _cohens_dz(arr, chance),
    }


def _paired_delta(run4: pd.Series, run3: pd.Series) -> dict[str, float]:
    a = pd.to_numeric(run4, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(run3, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    diff = a[valid] - b[valid]
    if diff.size < 2:
        return {"n_subjects": int(diff.size), "mean_delta": np.nan, "t": np.nan, "p": np.nan, "cohens_dz": np.nan}
    t_val, p_val = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
    sd = float(diff.std(ddof=1))
    return {
        "n_subjects": int(diff.size),
        "mean_delta": float(diff.mean()),
        "t": float(t_val),
        "p": float(p_val),
        "cohens_dz": float(diff.mean() / sd) if sd > 0 and math.isfinite(sd) else np.nan,
    }


def _load_4d(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=42)),
        ]
    )


def _selected_masks(manifest_path: Path, roi_sets: list[str], roi_names: list[str]) -> dict[tuple[str, str], Path]:
    manifest = read_table(manifest_path)
    manifest = manifest[manifest["roi_set"].isin(roi_sets)].copy()
    selected: dict[tuple[str, str], Path] = {}
    for _, row in manifest.iterrows():
        roi_name = str(row["roi_name"])
        if roi_names and roi_name not in roi_names:
            continue
        path_col = "path" if "path" in manifest.columns else "mask_path"
        selected[(str(row["roi_set"]), roi_name)] = Path(str(row[path_col]))
    if not selected:
        raise ValueError(f"No ROI masks selected from {manifest_path}")
    return selected


def _load_condition(subject_dir: Path, condition: str) -> tuple[pd.DataFrame, np.ndarray]:
    image_path = subject_dir / f"learn_{condition}.nii.gz"
    metadata_path = subject_dir / f"learn_{condition}_metadata.tsv"
    if not image_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing learning pattern for {subject_dir.name}, condition={condition}")
    metadata = read_table(metadata_path).reset_index(drop=True)
    data = _load_4d(image_path)
    if data.shape[3] != len(metadata):
        raise ValueError(f"Volume/metadata mismatch: {image_path}: {data.shape[3]} vs {len(metadata)}")
    metadata["subject"] = metadata.get("subject", subject_dir.name).astype(str)
    metadata["condition"] = condition
    metadata["label"] = CONDITION_TO_LABEL[condition]
    metadata["run_num"] = pd.to_numeric(metadata["run"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    metadata["pair_id"] = pd.to_numeric(metadata["pair_id"], errors="coerce").astype("Int64")
    return metadata, data


def _make_dataset(subject_dir: Path, mask: np.ndarray, mask_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in ["yy", "kj"]:
        metadata, data = _load_condition(subject_dir, condition)
        if data.shape[:3] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {subject_dir / f'learn_{condition}.nii.gz'} vs {mask_path}")
        samples = data[mask, :].T.astype(np.float64, copy=False)
        metadata = metadata.copy()
        metadata["_sample_ref"] = list(samples)
        frames.append(metadata)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["run_num", "pair_id"]).copy()
    data["run_num"] = data["run_num"].astype(int)
    data["pair_id"] = data["pair_id"].astype(int)
    return data


def _samples(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(frame["_sample_ref"].to_numpy())


def _balanced_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return float(accuracy_score(y_true, y_pred)), float(balanced_accuracy_score(y_true, y_pred))


def within_run_pair_heldout(data: pd.DataFrame, *, subject: str, roi_set: str, roi: str, run_num: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    run_data = data[data["run_num"].eq(run_num)].copy()
    yy_pairs = set(run_data.loc[run_data["condition"].eq("yy"), "pair_id"].astype(int))
    kj_pairs = set(run_data.loc[run_data["condition"].eq("kj"), "pair_id"].astype(int))
    common_pairs = sorted(yy_pairs & kj_pairs)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for pair_id in common_pairs:
        test = run_data[run_data["pair_id"].eq(pair_id)].copy()
        train = run_data[~run_data["pair_id"].eq(pair_id)].copy()
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        clf = _classifier()
        x_train = _samples(train)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = _samples(test)
        y_test = test["label"].to_numpy(dtype=int)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test).astype(int)
        acc, bal = _balanced_score(y_test, pred)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        fold_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": f"run{run_num}_pair_heldout",
                "run": int(run_num),
                "heldout_pair_id": int(pair_id),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(test)),
            }
        )
    if y_true_all:
        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        acc, bal = _balanced_score(y_true_arr, y_pred_arr)
        rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": f"run{run_num}_pair_heldout",
                "train_run": int(run_num),
                "test_run": int(run_num),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(y_true_arr)),
                "n_folds": int(len(fold_rows)),
                "n_common_pairs": int(len(common_pairs)),
            }
        )
    return rows, fold_rows


def cross_run_decoding(data: pd.DataFrame, *, subject: str, roi_set: str, roi: str, train_run: int, test_run: int) -> list[dict[str, object]]:
    train = data[data["run_num"].eq(train_run)].copy()
    test = data[data["run_num"].eq(test_run)].copy()
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return []
    clf = _classifier()
    x_train = _samples(train)
    y_train = train["label"].to_numpy(dtype=int)
    x_test = _samples(test)
    y_test = test["label"].to_numpy(dtype=int)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test).astype(int)
    acc, bal = _balanced_score(y_test, pred)
    return [
        {
            "subject": subject,
            "roi_set": roi_set,
            "roi": roi,
            "analysis_type": f"cross_run_{train_run}_to_{test_run}",
            "train_run": int(train_run),
            "test_run": int(test_run),
            "accuracy": acc,
            "balanced_accuracy": bal,
            "n_test": int(len(test)),
            "n_folds": 1,
            "n_common_pairs": np.nan,
        }
    ]


def summarize_decoding(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, subset in subject_metrics.groupby(["roi_set", "roi", "analysis_type"], sort=False):
        roi_set, roi, analysis_type = keys
        summary = _one_sample_accuracy(subset["balanced_accuracy"], chance=0.5)
        rows.append({"roi_set": roi_set, "roi": roi, "analysis_type": analysis_type, "metric": "balanced_accuracy", **summary})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_analysis"] = np.nan
    out["q_bh_primary_family"] = np.nan
    for _, idx in out.groupby("analysis_type", dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_analysis"] = _bh_fdr(out.loc[idx, "p"])
    primary = out["analysis_type"].isin(["run3_pair_heldout", "run4_pair_heldout", "cross_run_3_to_4", "cross_run_4_to_3"])
    idx = out[primary].index.tolist()
    out.loc[idx, "q_bh_primary_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_primary_family", "q_bh_within_analysis", "p"], na_position="last")


def summarize_run4_minus_run3(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairheld = subject_metrics[subject_metrics["analysis_type"].isin(["run3_pair_heldout", "run4_pair_heldout"])].copy()
    for keys, subset in pairheld.groupby(["roi_set", "roi"], sort=False):
        roi_set, roi = keys
        pivot = subset.pivot_table(index="subject", columns="analysis_type", values="balanced_accuracy", aggfunc="mean")
        if {"run3_pair_heldout", "run4_pair_heldout"} - set(pivot.columns):
            continue
        summary = _paired_delta(pivot["run4_pair_heldout"], pivot["run3_pair_heldout"])
        rows.append({"roi_set": roi_set, "roi": roi, "analysis_type": "run4_minus_run3_pair_heldout", **summary})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_analysis"] = _bh_fdr(out["p"])
    return out.sort_values(["q_bh_within_analysis", "p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Learning-stage meta ROI YY/KJ decoding.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    output_dir = args.paper_output_root / "qc" / "learning_mvpa_decoding"
    tables_main = args.paper_output_root / "tables_main"
    tables_si = args.paper_output_root / "tables_si"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_main.mkdir(parents=True, exist_ok=True)
    tables_si.mkdir(parents=True, exist_ok=True)

    selected_masks = _selected_masks(args.roi_manifest, args.roi_sets, args.rois)
    loaded_masks = {str(path): _load_mask(path) for path in selected_masks.values()}
    subjects = sorted(path for path in args.pattern_root.glob("sub-*") if path.is_dir())

    subject_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        for (roi_set, roi_name), mask_path in selected_masks.items():
            try:
                mask = loaded_masks[str(mask_path)]
                data = _make_dataset(subject_dir, mask, mask_path)
                qc_base = {
                    "subject": subject_dir.name,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "n_trials": int(len(data)),
                    "n_voxels": int(mask.sum()),
                    "n_run3": int(data["run_num"].eq(3).sum()),
                    "n_run4": int(data["run_num"].eq(4).sum()),
                    "n_yy": int(data["condition"].eq("yy").sum()),
                    "n_kj": int(data["condition"].eq("kj").sum()),
                }
                for run_num in [3, 4]:
                    run_data = data[data["run_num"].eq(run_num)]
                    yy_pairs = set(run_data.loc[run_data["condition"].eq("yy"), "pair_id"].astype(int))
                    kj_pairs = set(run_data.loc[run_data["condition"].eq("kj"), "pair_id"].astype(int))
                    qc_base[f"n_common_pairs_run{run_num}"] = int(len(yy_pairs & kj_pairs))
                qc_rows.append(qc_base)
                for run_num in [3, 4]:
                    rows, folds = within_run_pair_heldout(data, subject=subject_dir.name, roi_set=roi_set, roi=roi_name, run_num=run_num)
                    subject_rows.extend(rows)
                    fold_rows.extend(folds)
                subject_rows.extend(cross_run_decoding(data, subject=subject_dir.name, roi_set=roi_set, roi=roi_name, train_run=3, test_run=4))
                subject_rows.extend(cross_run_decoding(data, subject=subject_dir.name, roi_set=roi_set, roi=roi_name, train_run=4, test_run=3))
            except Exception as exc:
                failure_rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if not args.allow_partial:
                    raise

    subject_metrics = pd.DataFrame(subject_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group = summarize_decoding(subject_metrics)
    delta = summarize_run4_minus_run3(subject_metrics)

    write_table(subject_metrics, output_dir / "learning_mvpa_subject_metrics.tsv")
    write_table(fold_metrics, output_dir / "learning_mvpa_fold_metrics.tsv")
    write_table(qc, output_dir / "learning_mvpa_qc.tsv")
    write_table(failures, output_dir / "learning_mvpa_failures.tsv")
    write_table(group, output_dir / "learning_mvpa_group_fdr.tsv")
    write_table(delta, output_dir / "learning_mvpa_run4_minus_run3_fdr.tsv")
    write_table(group, tables_main / "table_learning_mvpa_decoding.tsv")
    write_table(delta, tables_main / "table_learning_mvpa_run4_minus_run3.tsv")
    write_table(subject_metrics, tables_si / "table_learning_mvpa_subject_metrics.tsv")

    save_json(
        {
            "subjects": [path.name for path in subjects],
            "n_subjects": len(subjects),
            "roi_sets": args.roi_sets,
            "rois": args.rois,
            "n_subject_metric_rows": int(len(subject_metrics)),
            "n_fold_rows": int(len(fold_metrics)),
            "n_failures": int(len(failures)),
            "interpretation_note": "Learning-stage patterns are pair/item-level patterns; no w/ew cross-role decoding is available.",
        },
        output_dir / "learning_mvpa_manifest.json",
    )
    if len(failures) and not args.allow_partial:
        raise RuntimeError(f"learning MVPA had {len(failures)} failures")


if __name__ == "__main__":
    main()
