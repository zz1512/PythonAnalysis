#!/usr/bin/env python3
"""Compute learning sentence -> pre word semantic reactivation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [FINAL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common.roi_library import filter_roi_manifest, load_roi_manifest  # noqa: E402
from lr_utils import (  # noqa: E402
    CONDITIONS,
    CONDITION_LABELS,
    VALID_ORIGINAL_IDS,
    condition_item_id,
    extract_original_id,
    fisher_corr_one_to_many,
    load_mask,
    load_template_map,
    masked_samples,
    read_any,
    write_tsv,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _load_metadata(path: Path) -> pd.DataFrame:
    frame = read_any(path).reset_index(drop=True)
    if "word_label" in frame.columns:
        frame["word_label"] = frame["word_label"].astype(str).str.strip()
        frame["original_pair_id"] = frame["word_label"].map(extract_original_id).astype("Int64")
    if "pair_id" in frame.columns:
        frame["template_pair_id"] = pd.to_numeric(frame["pair_id"], errors="coerce").astype("Int64")
    return frame


def _word_index(meta: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, row in meta.iterrows():
        label = str(row.get("word_label", "")).strip()
        if label and label.lower() != "nan" and label not in out:
            out[label] = int(idx)
    return out


def _learning_index(meta: pd.DataFrame, run: int) -> dict[int, int]:
    sub = meta[pd.to_numeric(meta["run"], errors="coerce").eq(run)].copy()
    counts = sub.groupby("original_pair_id").size()
    dup = set(counts[counts > 1].index.tolist())
    out: dict[int, int] = {}
    for idx, row in sub.iterrows():
        original = row.get("original_pair_id")
        if pd.isna(original) or original in dup:
            continue
        out[int(original)] = int(idx)
    return out


def _role_indices(pre_meta: pd.DataFrame, labels: list[str], index: dict[str, int]) -> list[int]:
    return [index[label] for label in labels if label in index]


def _compute_for_subject_roi_condition(
    *,
    subject_dir: Path,
    roi_set: str,
    roi: str,
    mask_path: Path,
    condition: str,
    template_condition: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name
    learn_img = subject_dir / f"learn_{condition}.nii.gz"
    learn_meta_path = subject_dir / f"learn_{condition}_metadata.tsv"
    pre_img = subject_dir / f"pre_{condition}.nii.gz"
    pre_meta_path = subject_dir / f"pre_{condition}_metadata.tsv"
    try:
        learn_meta = _load_metadata(learn_meta_path)
        pre_meta = _load_metadata(pre_meta_path)
        learn_samples = masked_samples(learn_img, mask, mask_path)
        pre_samples = masked_samples(pre_img, mask, mask_path)
        if len(learn_meta) != learn_samples.shape[0] or len(pre_meta) != pre_samples.shape[0]:
            raise ValueError("metadata/image row mismatch")

        pre_index = _word_index(pre_meta)
        role_a_labels = template_condition["role_a_label"].tolist()
        role_b_labels = template_condition["role_b_label"].tolist()
        role_a_all = _role_indices(pre_meta, role_a_labels, pre_index)
        role_b_all = _role_indices(pre_meta, role_b_labels, pre_index)
        run_indices = {3: _learning_index(learn_meta, 3), 4: _learning_index(learn_meta, 4)}

        for _, item in template_condition.iterrows():
            original = int(item["original_pair_id"])
            if original not in VALID_ORIGINAL_IDS[condition]:
                continue
            a_label = str(item["role_a_label"])
            b_label = str(item["role_b_label"])
            a_idx = pre_index.get(a_label)
            b_idx = pre_index.get(b_label)
            if a_idx is None or b_idx is None:
                continue
            for run in [3, 4]:
                learn_idx = run_indices[run].get(original)
                if learn_idx is None:
                    continue
                learn_vec = learn_samples[learn_idx]
                a_other = [idx for idx in role_a_all if idx != a_idx]
                b_other = [idx for idx in role_b_all if idx != b_idx]
                a_same = fisher_corr_one_to_many(learn_vec, pre_samples[[a_idx]])[0]
                b_same = fisher_corr_one_to_many(learn_vec, pre_samples[[b_idx]])[0]
                a_base_vals = fisher_corr_one_to_many(learn_vec, pre_samples[a_other]) if a_other else np.array([])
                b_base_vals = fisher_corr_one_to_many(learn_vec, pre_samples[b_other]) if b_other else np.array([])
                a_base = float(np.nanmean(a_base_vals)) if np.isfinite(a_base_vals).any() else np.nan
                b_base = float(np.nanmean(b_base_vals)) if np.isfinite(b_base_vals).any() else np.nan
                react_a = float(a_same - a_base) if np.isfinite(a_same) and np.isfinite(a_base) else np.nan
                react_b = float(b_same - b_base) if np.isfinite(b_same) and np.isfinite(b_base) else np.nan
                rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "run": run,
                        "original_pair_id": original,
                        "template_pair_id": int(item["template_pair_id"]),
                        "condition_item_id": condition_item_id(condition, original),
                        "role_a_label": a_label,
                        "role_b_label": b_label,
                        "role_a_same_z": float(a_same),
                        "role_a_baseline_z": a_base,
                        "react_role_a": react_a,
                        "role_b_same_z": float(b_same),
                        "role_b_baseline_z": b_base,
                        "react_role_b": react_b,
                        "react_pair_mean": float(np.nanmean([react_a, react_b])),
                        "react_asymmetry_b_minus_a": float(react_b - react_a) if np.isfinite(react_a) and np.isfinite(react_b) else np.nan,
                        "n_role_a_baseline": len(a_other),
                        "n_role_b_baseline": len(b_other),
                    }
                )
        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": True,
                "fail_reason": "",
                "n_rows": len([r for r in rows if r["subject"] == subject and r["roi"] == roi and r["condition"] == condition]),
                "n_pre_role_a": len(role_a_all),
                "n_pre_role_b": len(role_b_all),
            }
        )
    except Exception as exc:
        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": False,
                "fail_reason": str(exc),
                "n_rows": 0,
                "n_pre_role_a": 0,
                "n_pre_role_b": 0,
            }
        )
    return rows, qc_rows


def _wide_item(run_long: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "condition_label",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
        "role_a_label",
        "role_b_label",
    ]
    metric_cols = ["react_role_a", "react_role_b", "react_pair_mean", "react_asymmetry_b_minus_a"]
    parts = []
    for run in [3, 4]:
        sub = run_long[run_long["run"].eq(run)][index_cols + metric_cols].copy()
        sub = sub.rename(columns={col: f"run{run}_{col}" for col in metric_cols})
        parts.append(sub)
    out = parts[0].merge(parts[1], on=index_cols, how="outer")
    for col in metric_cols:
        out[f"{col}_avg"] = out[[f"run3_{col}", f"run4_{col}"]].mean(axis=1)
        out[f"{col}_delta_run4_minus_run3"] = out[f"run4_{col}"] - out[f"run3_{col}"]
    return out.sort_values(["subject", "roi_set", "roi", "condition", "original_pair_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    template = load_template_map(args.stimuli_template)
    manifest = load_roi_manifest(args.roi_manifest)
    roi_frames = [filter_roi_manifest(manifest, roi_set=roi_set, include_flag="include_in_rsa") for roi_set in args.roi_sets]
    rois = pd.concat(roi_frames, ignore_index=True)
    subject_dirs = sorted([p for p in args.pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if args.subjects:
        wanted = set(args.subjects)
        subject_dirs = [p for p in subject_dirs if p.name in wanted]

    rows: list[dict[str, object]] = []
    qcs: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for roi_row in rois.itertuples(index=False):
            mask_path = Path(str(roi_row.mask_path))
            try:
                mask = load_mask(mask_path)
            except Exception as exc:
                for condition in CONDITIONS:
                    qcs.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": str(roi_row.roi_set),
                            "roi": str(roi_row.roi_name),
                            "condition": condition,
                            "ok": False,
                            "fail_reason": f"mask failed: {exc}",
                            "n_rows": 0,
                            "n_pre_role_a": 0,
                            "n_pre_role_b": 0,
                        }
                    )
                continue
            for condition in CONDITIONS:
                item_template = template[template["condition"].eq(condition)].copy()
                item_rows, item_qc = _compute_for_subject_roi_condition(
                    subject_dir=subject_dir,
                    roi_set=str(roi_row.roi_set),
                    roi=str(roi_row.roi_name),
                    mask_path=mask_path,
                    condition=condition,
                    template_condition=item_template,
                    mask=mask,
                )
                rows.extend(item_rows)
                qcs.extend(item_qc)

    long = pd.DataFrame(rows)
    wide = _wide_item(long) if not long.empty else pd.DataFrame()
    qc = pd.DataFrame(qcs)
    write_tsv(long, args.out_dir / "learning_reactivation_run.tsv")
    write_tsv(wide, args.out_dir / "learning_reactivation_item.tsv")
    write_tsv(qc, args.out_dir / "learning_reactivation_qc.tsv")
    summary = {
        "script": str(Path(__file__).resolve()),
        "n_run_rows": int(len(long)),
        "n_item_rows": int(len(wide)),
        "n_subjects": int(wide["subject"].nunique()) if not wide.empty else 0,
        "n_rois": int(wide[["roi_set", "roi"]].drop_duplicates().shape[0]) if not wide.empty else 0,
        "qc_ok_rate": float(qc["ok"].mean()) if "ok" in qc else None,
    }
    (args.out_dir / "learning_reactivation_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
