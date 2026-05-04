#!/usr/bin/env python3
"""Run-7 ROI decoding for meta ROI retrieval-rebound regions.

Primary questions
- Can run7 patterns decode YY vs KJ in ROIs that showed retrieval-stage rebound?
- Does decoding generalize across word roles (yyw/kjw -> yyew/kjew, and reverse)?
- Does cross-validated classifier evidence predict run7 memory/RT?

The main decoding uses leave-one-pair-out CV so the classifier cannot solve the
task by memorizing a single pair_id.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.roi_library import load_roi_manifest, select_roi_masks  # noqa: E402


DEFAULT_ROIS = [
    "meta_R_PPA_PHG",
    "meta_R_PPC_SPL",
    "meta_R_temporal_pole",
    "meta_R_IFG",
    "meta_R_hippocampus",
]
CONDITION_TO_LABEL = {"kj": 0, "yy": 1}
LABEL_TO_CONDITION = {0: "kj", 1: "yy"}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


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
        return {
            "n_subjects": int(arr.size),
            "mean_accuracy": float("nan"),
            "mean_minus_chance": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
        }
    t_val, p_val = stats.ttest_1samp(arr, chance, nan_policy="omit")
    return {
        "n_subjects": int(arr.size),
        "mean_accuracy": float(arr.mean()),
        "mean_minus_chance": float(arr.mean() - chance),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": _cohens_dz(arr, chance),
    }


def _load_4d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def _load_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(mask_path)).get_fdata()) > 0


def _masked_samples(data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _normalize_role(value: object) -> str:
    text = str(value).strip().lower()
    if text.endswith("ew"):
        return "ew"
    if text.endswith("w"):
        return "w"
    return text


def _load_condition(subject_dir: Path, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"retrieval_{condition}.nii.gz"
    meta_path = subject_dir / f"retrieval_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing retrieval_{condition} for {subject_dir.name}")
    meta = _read_table(meta_path).reset_index(drop=True)
    data = _load_4d(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(f"Volume/metadata mismatch: {image_path}: {data.shape[3]} vs {len(meta)}")
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["label"] = CONDITION_TO_LABEL[condition]
    meta["pair_id"] = pd.to_numeric(meta["pair_id"], errors="coerce").astype("Int64")
    meta["word_label"] = meta["word_label"].astype(str).str.strip()
    role_source = meta["original_condition"] if "original_condition" in meta.columns else meta["word_label"].str.split("_").str[0]
    meta["word_role"] = role_source.map(_normalize_role)
    meta["memory"] = pd.to_numeric(meta.get("memory", np.nan), errors="coerce")
    meta["action_time"] = pd.to_numeric(meta.get("action_time", np.nan), errors="coerce")
    return meta, data, image_path


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=42)),
        ]
    )


def _decision_values(clf: Pipeline, x: np.ndarray) -> np.ndarray:
    decision = clf.decision_function(x)
    return np.asarray(decision, dtype=float).reshape(-1)


def _make_dataset(subject_dir: Path, mask: np.ndarray, mask_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in ["yy", "kj"]:
        meta, data, image_path = _load_condition(subject_dir, condition)
        samples = _masked_samples(data, mask, image_path, mask_path)
        meta = meta.copy()
        meta["_sample_index"] = np.arange(len(meta), dtype=int)
        meta["_condition_image_path"] = str(image_path)
        # Store sample matrix outside the dataframe, then attach row-wise object
        # references only after filtering to keep logic explicit.
        meta["_sample_ref"] = list(samples)
        frames.append(meta)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["pair_id"])
    data["pair_id"] = data["pair_id"].astype(int)
    data = data[data["word_role"].isin(["w", "ew"])].copy()
    data["condition_pair_index"] = np.nan
    for condition, idx in data.groupby("condition").groups.items():
        pair_ids = sorted(data.loc[idx, "pair_id"].astype(int).unique().tolist())
        pair_map = {pair_id: rank + 1 for rank, pair_id in enumerate(pair_ids)}
        data.loc[idx, "condition_pair_index"] = data.loc[idx, "pair_id"].map(pair_map)
    data["condition_pair_index"] = data["condition_pair_index"].astype(int)
    return data


def _samples(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(frame["_sample_ref"].to_numpy())


def _balanced_fold_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return float(accuracy_score(y_true, y_pred)), float(balanced_accuracy_score(y_true, y_pred))


def pair_heldout_decoding(data: pd.DataFrame, *, subject: str, roi_set: str, roi: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    yy_pair_idx = set(data.loc[data["condition"].eq("yy"), "condition_pair_index"].dropna().astype(int))
    kj_pair_idx = set(data.loc[data["condition"].eq("kj"), "condition_pair_index"].dropna().astype(int))
    pair_indices = sorted(yy_pair_idx & kj_pair_idx)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for pair_index in pair_indices:
        test = data[data["condition_pair_index"].eq(pair_index)].copy()
        train = data[~data["condition_pair_index"].eq(pair_index)].copy()
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        clf = _classifier()
        x_train = _samples(train)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = _samples(test)
        y_test = test["label"].to_numpy(dtype=int)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test).astype(int)
        decision = _decision_values(clf, x_test)
        acc, bal = _balanced_fold_score(y_test, pred)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        fold_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": "pair_heldout",
                "heldout_condition_pair_index": int(pair_index),
                "heldout_pair_ids": ",".join(str(x) for x in sorted(test["pair_id"].astype(int).unique().tolist())),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(test)),
            }
        )
        for row, y, yhat, dec in zip(test.itertuples(index=False), y_test, pred, decision):
            evidence_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi,
                    "analysis_type": "pair_heldout",
                    "pair_id": int(row.pair_id),
                    "condition_pair_index": int(row.condition_pair_index),
                    "word_label": row.word_label,
                    "word_role": row.word_role,
                    "condition": LABEL_TO_CONDITION[int(y)],
                    "label": int(y),
                    "predicted_label": int(yhat),
                    "decision_value": float(dec),
                    "true_class_evidence": float(dec if int(y) == 1 else -dec),
                    "memory": row.memory,
                    "action_time": row.action_time,
                    "is_correct_prediction": bool(int(y) == int(yhat)),
                }
            )
    if y_true_all:
        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        acc, bal = _balanced_fold_score(y_true_arr, y_pred_arr)
        rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": "pair_heldout",
                "train_role": "both",
                "test_role": "both",
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(y_true_arr)),
                "n_folds": int(len(fold_rows)),
            }
        )
    return rows, fold_rows, evidence_rows


def role_specific_pair_heldout_decoding(data: pd.DataFrame, *, subject: str, roi_set: str, roi: str, role: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    role_data = data[data["word_role"].eq(role)].copy()
    yy_pair_idx = set(role_data.loc[role_data["condition"].eq("yy"), "condition_pair_index"].dropna().astype(int))
    kj_pair_idx = set(role_data.loc[role_data["condition"].eq("kj"), "condition_pair_index"].dropna().astype(int))
    pair_indices = sorted(yy_pair_idx & kj_pair_idx)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for pair_index in pair_indices:
        test = role_data[role_data["condition_pair_index"].eq(pair_index)].copy()
        train = role_data[~role_data["condition_pair_index"].eq(pair_index)].copy()
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        clf = _classifier()
        x_train = _samples(train)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = _samples(test)
        y_test = test["label"].to_numpy(dtype=int)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test).astype(int)
        acc, bal = _balanced_fold_score(y_test, pred)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        fold_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": f"pair_heldout_{role}_only",
                "heldout_condition_pair_index": int(pair_index),
                "heldout_pair_ids": ",".join(str(x) for x in sorted(test["pair_id"].astype(int).unique().tolist())),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(test)),
            }
        )
    if y_true_all:
        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        acc, bal = _balanced_fold_score(y_true_arr, y_pred_arr)
        rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": f"pair_heldout_{role}_only",
                "train_role": role,
                "test_role": role,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(y_true_arr)),
                "n_folds": int(len(fold_rows)),
            }
        )
    return rows, fold_rows


def cross_role_decoding(data: pd.DataFrame, *, subject: str, roi_set: str, roi: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_role, test_role in [("w", "ew"), ("ew", "w")]:
        train = data[data["word_role"].eq(train_role)].copy()
        test = data[data["word_role"].eq(test_role)].copy()
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        clf = _classifier()
        x_train = _samples(train)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = _samples(test)
        y_test = test["label"].to_numpy(dtype=int)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test).astype(int)
        acc, bal = _balanced_fold_score(y_test, pred)
        rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": f"cross_role_{train_role}_to_{test_role}",
                "train_role": train_role,
                "test_role": test_role,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "n_test": int(len(test)),
                "n_folds": 1,
            }
        )
    return rows


def summarize_decoding(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, subset in subject_metrics.groupby(["roi_set", "roi", "analysis_type"], sort=False):
        roi_set, roi, analysis_type = keys
        summary = _one_sample_accuracy(subset["balanced_accuracy"], chance=0.5)
        rows.append(
            {
                "roi_set": roi_set,
                "roi": roi,
                "analysis_type": analysis_type,
                "metric": "balanced_accuracy",
                **summary,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_analysis"] = np.nan
    out["q_bh_primary_family"] = np.nan
    for _, idx in out.groupby(["analysis_type"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_analysis"] = _bh_fdr(out.loc[idx, "p"])
    primary = out["analysis_type"].isin(["pair_heldout", "cross_role_w_to_ew", "cross_role_ew_to_w"])
    idx = out[primary].index.tolist()
    out.loc[idx, "q_bh_primary_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_primary_family", "q_bh_within_analysis", "p"], na_position="last")


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def _fit_gee(frame: pd.DataFrame, response: str) -> dict[str, object] | None:
    data = frame.copy()
    data[response] = pd.to_numeric(data[response], errors="coerce")
    data["true_class_evidence"] = pd.to_numeric(data["true_class_evidence"], errors="coerce")
    data = data.dropna(subset=[response, "true_class_evidence", "subject"])
    if data["subject"].nunique() < 8 or data[response].nunique() < 2:
        return None
    data["evidence_between_subject"] = data.groupby("subject")["true_class_evidence"].transform("mean")
    data["evidence_within_subject"] = data["true_class_evidence"] - data["evidence_between_subject"]
    data["evidence_within_z"] = _zscore(data["evidence_within_subject"])
    data["evidence_between_z"] = _zscore(data["evidence_between_subject"])
    data = data.dropna(subset=[response, "evidence_within_z", "evidence_between_z"])
    if len(data) < 30:
        return None
    family = Binomial() if response == "memory" else Gaussian()
    try:
        fit = GEE.from_formula(
            f"{response} ~ evidence_within_z + evidence_between_z + C(condition)",
            groups="subject",
            data=data,
            family=family,
            cov_struct=Exchangeable(),
        ).fit()
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "n_subjects": int(data["subject"].nunique()),
            "n_trials": int(len(data)),
        }
    return {
        "status": "ok",
        "n_subjects": int(data["subject"].nunique()),
        "n_trials": int(len(data)),
        "within_beta": float(fit.params.get("evidence_within_z", np.nan)),
        "within_se": float(fit.bse.get("evidence_within_z", np.nan)),
        "within_p": float(fit.pvalues.get("evidence_within_z", np.nan)),
        "between_beta": float(fit.params.get("evidence_between_z", np.nan)),
        "between_se": float(fit.bse.get("evidence_between_z", np.nan)),
        "between_p": float(fit.pvalues.get("evidence_between_z", np.nan)),
        "family": "binomial" if response == "memory" else "gaussian",
    }


def summarize_evidence_behavior(evidence: pd.DataFrame) -> pd.DataFrame:
    frame = evidence.copy()
    frame["memory"] = pd.to_numeric(frame["memory"], errors="coerce")
    frame["action_time"] = pd.to_numeric(frame["action_time"], errors="coerce")
    frame["log_rt_correct"] = np.nan
    valid_rt = frame["memory"].eq(1) & frame["action_time"].gt(0)
    frame.loc[valid_rt, "log_rt_correct"] = np.log(frame.loc[valid_rt, "action_time"])
    rows: list[dict[str, object]] = []
    for keys, subset in frame.groupby(["roi_set", "roi", "analysis_type"], sort=False):
        roi_set, roi, analysis_type = keys
        for response in ["memory", "log_rt_correct"]:
            fit = _fit_gee(subset, response)
            if fit is None:
                continue
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "analysis_type": analysis_type,
                    "response": response,
                    **fit,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_response"] = np.nan
    ok = out["status"].eq("ok")
    for _, idx in out[ok].groupby(["response"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_response"] = _bh_fdr(out.loc[idx, "within_p"])
    return out.sort_values(["q_bh_within_response", "within_p"], na_position="last")


def _selected_masks(manifest_path: Path, roi_sets: list[str], roi_names: list[str]) -> dict[tuple[str, str], Path]:
    selected: dict[tuple[str, str], Path] = {}
    manifest = load_roi_manifest(manifest_path)
    for roi_set in roi_sets:
        masks = select_roi_masks(manifest_path, roi_set=roi_set, include_flag="include_in_rsa")
        for roi_name, mask_path in masks.items():
            if roi_names and roi_name not in roi_names:
                continue
            selected[(roi_set, roi_name)] = mask_path
    if not selected:
        available = manifest[manifest["roi_set"].isin(roi_sets)]["roi_name"].tolist()
        raise ValueError(f"No ROI masks selected. Requested={roi_names}; available={available}")
    return selected


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Run7 YY/KJ decoding in meta ROI rebound regions.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    args = parser.parse_args()

    output_dir = ensure_dir(args.paper_output_root / "qc" / "run7_mvpa_decoding")
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    subjects = sorted(
        sub for sub in args.pattern_root.glob("sub-*")
        if sub.is_dir() and (sub / "retrieval_yy.nii.gz").exists() and (sub / "retrieval_kj.nii.gz").exists()
    )
    masks = _selected_masks(args.roi_manifest, args.roi_sets, args.rois)

    subject_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        for (roi_set, roi_name), mask_path in masks.items():
            try:
                mask = _load_mask(mask_path)
                data = _make_dataset(subject_dir, mask, mask_path)
                n_by_cell = data.groupby(["condition", "word_role"]).size().to_dict()
                qc_rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "n_trials": int(len(data)),
                        "n_pairs": int(data["pair_id"].nunique()),
                        "n_yy_w": int(n_by_cell.get(("yy", "w"), 0)),
                        "n_yy_ew": int(n_by_cell.get(("yy", "ew"), 0)),
                        "n_kj_w": int(n_by_cell.get(("kj", "w"), 0)),
                        "n_kj_ew": int(n_by_cell.get(("kj", "ew"), 0)),
                        "ok": True,
                    }
                )
                rows, folds, evidence = pair_heldout_decoding(
                    data,
                    subject=subject_dir.name,
                    roi_set=roi_set,
                    roi=roi_name,
                )
                subject_rows.extend(rows)
                fold_rows.extend(folds)
                evidence_rows.extend(evidence)
                for role in ["w", "ew"]:
                    role_rows, role_folds = role_specific_pair_heldout_decoding(
                        data,
                        subject=subject_dir.name,
                        roi_set=roi_set,
                        roi=roi_name,
                        role=role,
                    )
                    subject_rows.extend(role_rows)
                    fold_rows.extend(role_folds)
                subject_rows.extend(
                    cross_role_decoding(data, subject=subject_dir.name, roi_set=roi_set, roi=roi_name)
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                qc_rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "ok": False,
                        "error": str(exc),
                    }
                )

    subject_metrics = pd.DataFrame(subject_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    evidence = pd.DataFrame(evidence_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group = summarize_decoding(subject_metrics)
    behavior = summarize_evidence_behavior(evidence) if not evidence.empty else pd.DataFrame()

    write_table(subject_metrics, output_dir / "run7_mvpa_subject_metrics.tsv")
    write_table(fold_metrics, output_dir / "run7_mvpa_fold_metrics.tsv")
    write_table(evidence, output_dir / "run7_mvpa_trial_evidence.tsv")
    write_table(qc, output_dir / "run7_mvpa_qc.tsv")
    write_table(failures, output_dir / "run7_mvpa_failures.tsv")
    write_table(group, output_dir / "run7_mvpa_group_fdr.tsv")
    write_table(behavior, output_dir / "run7_mvpa_evidence_behavior_fdr.tsv")

    write_table(group, tables_main / "table_run7_mvpa_decoding.tsv")
    write_table(behavior, tables_main / "table_run7_mvpa_evidence_behavior.tsv")
    write_table(subject_metrics, tables_si / "table_run7_mvpa_subject_metrics.tsv")
    write_table(evidence, tables_si / "table_run7_mvpa_trial_evidence.tsv")

    save_json(
        {
            "subjects": [sub.name for sub in subjects],
            "n_subjects": len(subjects),
            "roi_sets": args.roi_sets,
            "rois": args.rois,
            "n_subject_metric_rows": int(len(subject_metrics)),
            "n_trial_evidence_rows": int(len(evidence)),
            "n_failures": int(len(failures)),
        },
        output_dir / "run7_mvpa_manifest.json",
    )


if __name__ == "__main__":
    main()
