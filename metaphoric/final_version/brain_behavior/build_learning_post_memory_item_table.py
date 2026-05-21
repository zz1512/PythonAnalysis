#!/usr/bin/env python3
"""Build item-level learning -> post -> memory mechanism table.

This script deliberately keeps two IDs:
- original_pair_id: the material/picture number, e.g. yy28 / kj5.
- template_pair_id: the compact 1..35 pair ID used by stimuli_template.csv and
  existing pre/post/retrieval pair-similarity outputs.

The learning sentence patterns are aligned by original_pair_id. Existing
word-pair similarity tables are aligned by template_pair_id. YY and KJ are
never item-paired with each other.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


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
from common.roi_library import load_roi_manifest, filter_roi_manifest  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
MISSING_ORIGINAL_IDS = {
    "yy": {20, 25, 30, 31, 34},
    "kj": {4, 18, 32, 35, 40},
}
VALID_ORIGINAL_IDS = {
    condition: [idx for idx in range(1, 41) if idx not in missing]
    for condition, missing in MISSING_ORIGINAL_IDS.items()
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"metaphor", "yy", "yyw", "yyew"}:
        return "yy"
    if text in {"spatial", "kj", "kjw", "kjew"}:
        return "kj"
    return text


def _extract_original_id(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    stem = Path(text).stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return float("nan")
    return float(int(match.group(1)))


def _condition_item_id(condition: str, original_pair_id: object) -> str:
    try:
        original = int(float(original_pair_id))
    except Exception:
        return f"{condition}_nan"
    return f"{condition}_{original}"


def _load_template_map(template_path: Path) -> pd.DataFrame:
    template = _read_any(template_path).copy()
    template["condition"] = template["condition"].map(_normalize_condition)
    template = template[template["condition"].isin(CONDITIONS)].copy()
    template["template_pair_id"] = pd.to_numeric(template["pair_id"], errors="coerce").astype("Int64")
    template["word_label"] = template["word_label"].astype(str).str.strip()
    template["original_pair_id"] = template["word_label"].map(_extract_original_id).astype("Int64")
    rows = []
    for (condition, template_pair_id), group in template.groupby(["condition", "template_pair_id"], dropna=True):
        originals = sorted(set(int(v) for v in group["original_pair_id"].dropna().tolist()))
        if len(originals) != 1:
            raise ValueError(
                f"Template pair {condition}/{template_pair_id} maps to original IDs {originals}; expected one."
            )
        word_labels = sorted(group["word_label"].astype(str).tolist())
        rows.append(
            {
                "condition": condition,
                "template_pair_id": int(template_pair_id),
                "original_pair_id": originals[0],
                "condition_item_id": _condition_item_id(condition, originals[0]),
                "word_labels": "|".join(word_labels),
            }
        )
    return pd.DataFrame(rows)


def _load_material_covariates(base_dir: Path, template_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    materials_dir = base_dir / "materials_detail"
    candidates = list(materials_dir.glob("*.xlsx"))
    target = None
    for path in candidates:
        if path.name.endswith("实验材料整理.xlsx") or path.stat().st_size == 17429:
            target = path
            break
    if target is not None:
        sheet_map = {"yy": "隐喻", "kj": "空间"}
        for condition, sheet in sheet_map.items():
            try:
                frame = pd.read_excel(target, sheet_name=sheet)
            except Exception:
                continue
            sentence_col = "隐喻" if condition == "yy" else "空间"
            for _, row in frame.iterrows():
                original = _extract_original_id(row.get("编号", ""))
                if not math.isfinite(original):
                    continue
                rows.append(
                    {
                        "condition": condition,
                        "original_pair_id": int(original),
                        "sentence_text": str(row.get(sentence_col, "")).strip(),
                        "word_a_text": str(row.get("首词", "")).strip(),
                        "word_b_text": str(row.get("末词", "")).strip(),
                        "sentence_char_len": len(str(row.get(sentence_col, "")).strip()),
                        "word_a_char_len": len(str(row.get("首词", "")).strip()),
                        "word_b_char_len": len(str(row.get("末词", "")).strip()),
                    }
                )
    materials = pd.DataFrame(rows)
    if materials.empty:
        materials = template_map[["condition", "original_pair_id"]].drop_duplicates().copy()

    norms_path = base_dir / "paper_outputs" / "qc" / "stimulus_control_s0" / "stimulus_control_s0_auto_lexical_norms.tsv"
    if norms_path.exists():
        norms = read_table(norms_path).copy()
        norms["word_label_norm"] = norms["word_label"].astype(str).str.replace("_", "", regex=False).str.lower()
        template_words = template_map.copy()
        split_words = template_words["word_labels"].str.split("|", expand=True)
        word_rows = []
        for idx, row in template_words.iterrows():
            for label in split_words.loc[idx].dropna().tolist():
                word_rows.append(
                    {
                        "condition": row["condition"],
                        "original_pair_id": row["original_pair_id"],
                        "word_label_norm": str(label).replace("_", "").lower(),
                    }
                )
        word_frame = pd.DataFrame(word_rows)
        joined = word_frame.merge(norms, on="word_label_norm", how="left")
        numeric_cols = [
            col for col in [
                "word_frequency",
                "stroke_count",
                "valence",
                "arousal",
                "concreteness",
                "familiarity",
                "imageability",
            ]
            if col in joined.columns
        ]
        if numeric_cols:
            agg = joined.groupby(["condition", "original_pair_id"], as_index=False)[numeric_cols].mean()
            agg = agg.rename(columns={col: f"{col}_mean" for col in numeric_cols})
            materials = materials.merge(agg, on=["condition", "original_pair_id"], how="left")

    return materials


def _load_4d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def _masked_samples(data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = a @ b.T / denom
    corr = np.clip(corr, -0.999999, 0.999999)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(corr)
    z[~np.isfinite(z)] = np.nan
    return z


def _corr_similarity(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    centered = samples - np.nanmean(samples, axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1)[:, None] * np.linalg.norm(centered, axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = centered @ centered.T / denom
    sim = np.clip(sim, -1.0, 1.0)
    sim[~np.isfinite(sim)] = np.nan
    np.fill_diagonal(sim, np.nan)
    return sim


def _load_learning_condition(subject_dir: Path, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"learn_{condition}.nii.gz"
    meta_path = subject_dir / f"learn_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing learning image or metadata: {subject_dir.name} {condition}")
    meta = read_table(meta_path).reset_index(drop=True)
    data = _load_4d(image_path)
    if len(meta) != data.shape[3]:
        raise ValueError(f"Learning metadata/image mismatch: {meta_path} rows={len(meta)} vols={data.shape[3]}")
    meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    if "word_label" in meta.columns:
        meta["original_pair_id"] = meta["word_label"].map(_extract_original_id).astype("Int64")
    elif "pair_id" in meta.columns:
        meta["original_pair_id"] = pd.to_numeric(meta["pair_id"], errors="coerce").astype("Int64")
    else:
        raise ValueError(f"Learning metadata missing word_label/pair_id: {meta_path}")
    meta["condition"] = condition
    return meta, data, image_path


def _compute_learning_rows(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    template_map: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name
    mask = _load_mask(mask_path)

    for condition in CONDITIONS:
        try:
            meta, data, image_path = _load_learning_condition(subject_dir, condition)
            samples = _masked_samples(data, mask, image_path, mask_path)
            run3 = meta[meta["run"] == 3].copy()
            run4 = meta[meta["run"] == 4].copy()
            run3 = run3.drop_duplicates("original_pair_id", keep=False)
            run4 = run4.drop_duplicates("original_pair_id", keep=False)
            common_ids = sorted(
                set(int(v) for v in run3["original_pair_id"].dropna().tolist())
                & set(int(v) for v in run4["original_pair_id"].dropna().tolist())
            )
            valid_ids = set(VALID_ORIGINAL_IDS[condition])
            common_ids = [idx for idx in common_ids if idx in valid_ids]
            run3_index = {int(row.original_pair_id): int(idx) for idx, row in run3.iterrows()}
            run4_index = {int(row.original_pair_id): int(idx) for idx, row in run4.iterrows()}
            mat3 = samples[[run3_index[idx] for idx in common_ids], :]
            mat4 = samples[[run4_index[idx] for idx in common_ids], :]
            sim = _corr_matrix(mat3, mat4)
            id_to_template = (
                template_map[template_map["condition"] == condition]
                .set_index("original_pair_id")["template_pair_id"]
                .to_dict()
            )
            finite_cells = int(np.isfinite(sim).sum())
            total_cells = int(sim.size)
            for pos, original_id in enumerate(common_ids):
                self_z = float(sim[pos, pos]) if np.isfinite(sim[pos, pos]) else float("nan")
                others = np.delete(sim[pos, :], pos)
                other_mean = float(np.nanmean(others)) if np.isfinite(others).any() else float("nan")
                specificity = self_z - other_mean if np.isfinite(self_z) and np.isfinite(other_mean) else float("nan")
                rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "original_pair_id": int(original_id),
                        "template_pair_id": int(id_to_template.get(original_id)) if original_id in id_to_template else np.nan,
                        "condition_item_id": _condition_item_id(condition, original_id),
                        "valid_condition_item": True,
                        "learning_self_similarity": self_z,
                        "learning_other_similarity": other_mean,
                        "learning_specificity": specificity,
                        "n_learning_run3_rows": int((meta["run"] == 3).sum()),
                        "n_learning_run4_rows": int((meta["run"] == 4).sum()),
                        "learning_metric_reliability": finite_cells / total_cells if total_cells else np.nan,
                    }
                )
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": True,
                    "fail_reason": "",
                    "n_common_valid_items": len(common_ids),
                    "n_run3_rows": int((meta["run"] == 3).sum()),
                    "n_run4_rows": int((meta["run"] == 4).sum()),
                    "finite_similarity_cells": finite_cells,
                    "total_similarity_cells": total_cells,
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": str(exc),
                    "n_common_valid_items": 0,
                    "n_run3_rows": 0,
                    "n_run4_rows": 0,
                    "finite_similarity_cells": 0,
                    "total_similarity_cells": 0,
                }
            )
    return rows, qc_rows


def _load_pair_metrics(base_dir: Path, template_map: pd.DataFrame) -> pd.DataFrame:
    path = base_dir / "paper_outputs" / "qc" / "retrieval_geometry" / "retrieval_pair_metrics.tsv"
    pair = read_table(path).copy()
    pair["condition"] = pair["condition"].map(_normalize_condition)
    pair = pair[pair["condition"].isin(CONDITIONS)].copy()
    pair["template_pair_id"] = pd.to_numeric(pair["pair_id"], errors="coerce").astype("Int64")
    pair = pair.drop(columns=["pair_id"])
    pair = pair.merge(
        template_map[["condition", "template_pair_id", "original_pair_id", "condition_item_id"]],
        on=["condition", "template_pair_id"],
        how="left",
    )
    pair["trained_edge_drop"] = pair["pre_pair_similarity"] - pair["post_pair_similarity"]
    keep = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "template_pair_id",
        "original_pair_id",
        "condition_item_id",
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_pair_similarity",
        "post_minus_pre_pair_similarity",
        "retrieval_minus_post_pair_similarity",
        "retrieval_minus_pre_pair_similarity",
        "trained_edge_drop",
    ]
    keep = [col for col in keep if col in pair.columns]
    return pair[keep].copy()


def _load_pseudo_edge_drop(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "paper_outputs" / "qc" / "edge_specificity" / "edge_delta_subject.tsv"
    edge = read_table(path).copy()
    edge["condition"] = edge["condition"].map(_normalize_condition)
    edge = edge[edge["condition"].isin(CONDITIONS)].copy()
    pseudo = edge[edge["edge_status"].astype(str).eq("untrained_nonedge")].copy()
    pseudo = pseudo.rename(columns={"drop_pre_minus_post": "pseudo_edge_drop"})
    keep = ["subject", "roi_set", "roi", "condition", "pseudo_edge_drop"]
    return pseudo[keep].drop_duplicates().copy()


def _metadata_template_ids(meta: pd.DataFrame, condition: str, template_map: pd.DataFrame) -> pd.Series:
    mapping = template_map[template_map["condition"].eq(condition)].set_index("original_pair_id")["template_pair_id"].to_dict()
    if "word_label" in meta.columns:
        original = meta["word_label"].map(_extract_original_id)
    elif "pic_num" in meta.columns:
        original = pd.to_numeric(meta["pic_num"], errors="coerce")
    else:
        original = pd.to_numeric(meta.get("pair_id", np.nan), errors="coerce")
    return original.map(lambda item: mapping.get(int(item)) if pd.notna(item) and int(item) in mapping else np.nan)


def _mean_nonedge_similarity(image_path: Path, meta_path: Path, mask: np.ndarray, mask_path: Path, condition: str, template_map: pd.DataFrame) -> tuple[float, int]:
    if not image_path.exists() or not meta_path.exists():
        return float("nan"), 0
    meta = read_table(meta_path).reset_index(drop=True)
    data = _load_4d(image_path)
    if len(meta) != data.shape[3]:
        raise ValueError(f"Metadata/image mismatch: {meta_path} rows={len(meta)} vols={data.shape[3]}")
    template_ids = _metadata_template_ids(meta, condition, template_map)
    samples = _masked_samples(data, mask, image_path, mask_path)
    sim = _corr_similarity(samples)
    ids = template_ids.to_numpy()
    iu, ju = np.triu_indices(len(ids), k=1)
    valid = pd.notna(ids[iu]) & pd.notna(ids[ju]) & (ids[iu] != ids[ju])
    values = sim[iu[valid], ju[valid]]
    values = values[np.isfinite(values)]
    return (float(np.mean(values)) if values.size else float("nan"), int(values.size))


def _compute_pseudo_edge_drops(
    subject_dirs: list[Path],
    rois: pd.DataFrame,
    template_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for roi in rois.itertuples(index=False):
            mask_path = Path(str(roi.mask_path))
            try:
                mask = _load_mask(mask_path)
            except Exception as exc:
                for condition in CONDITIONS:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": str(roi.roi_set),
                            "roi": str(roi.roi_name),
                            "condition": condition,
                            "ok": False,
                            "fail_reason": f"mask failed: {exc}",
                            "n_pre_nonedges": 0,
                            "n_post_nonedges": 0,
                        }
                    )
                continue
            for condition in CONDITIONS:
                try:
                    pre_mean, n_pre = _mean_nonedge_similarity(
                        subject_dir / f"pre_{condition}.nii.gz",
                        subject_dir / f"pre_{condition}_metadata.tsv",
                        mask,
                        mask_path,
                        condition,
                        template_map,
                    )
                    post_mean, n_post = _mean_nonedge_similarity(
                        subject_dir / f"post_{condition}.nii.gz",
                        subject_dir / f"post_{condition}_metadata.tsv",
                        mask,
                        mask_path,
                        condition,
                        template_map,
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "roi_set": str(roi.roi_set),
                            "roi": str(roi.roi_name),
                            "condition": condition,
                            "pre_pseudo_similarity": pre_mean,
                            "post_pseudo_similarity": post_mean,
                            "pseudo_edge_drop": pre_mean - post_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan,
                            "n_pre_nonedges": n_pre,
                            "n_post_nonedges": n_post,
                        }
                    )
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": str(roi.roi_set),
                            "roi": str(roi.roi_name),
                            "condition": condition,
                            "ok": bool(np.isfinite(pre_mean) and np.isfinite(post_mean)),
                            "fail_reason": "",
                            "n_pre_nonedges": n_pre,
                            "n_post_nonedges": n_post,
                        }
                    )
                except Exception as exc:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": str(roi.roi_set),
                            "roi": str(roi.roi_name),
                            "condition": condition,
                            "ok": False,
                            "fail_reason": str(exc),
                            "n_pre_nonedges": 0,
                            "n_post_nonedges": 0,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def _load_behavior(base_dir: Path, template_map: pd.DataFrame) -> pd.DataFrame:
    path = base_dir / "paper_outputs" / "qc" / "behavior_results" / "refined" / "behavior_trials.tsv"
    behavior = read_table(path).copy()
    behavior["condition"] = behavior["condition"].map(_normalize_condition)
    behavior = behavior[behavior["condition"].isin(CONDITIONS)].copy()
    if "pic_num" in behavior.columns:
        behavior["original_pair_id"] = pd.to_numeric(behavior["pic_num"], errors="coerce").astype("Int64")
    else:
        behavior["original_pair_id"] = behavior["item"].map(_extract_original_id).astype("Int64")
    behavior = behavior.merge(
        template_map[["condition", "original_pair_id", "template_pair_id", "condition_item_id"]],
        on=["condition", "original_pair_id"],
        how="left",
    )
    grouped = behavior.groupby(["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"], dropna=False)
    out = grouped.agg(
        memory=("memory", "mean"),
        log_rt_correct=("log_rt_correct", "mean"),
        n_behavior_trials=("memory", "size"),
    ).reset_index()
    return out


def _add_z_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        grouped = []
        for _, sub in out.groupby(["roi_set", "roi"], sort=False):
            s = pd.to_numeric(sub[col], errors="coerce")
            sd = s.std(ddof=1)
            if not np.isfinite(sd) or math.isclose(float(sd), 0.0):
                z = pd.Series(0.0, index=sub.index)
            else:
                z = (s - s.mean()) / sd
            grouped.append(z)
        if grouped:
            out[f"{col}_z"] = pd.concat(grouped).sort_index()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.paper_output_root / "qc" / "learning_post_memory_prediction")
    template_map = _load_template_map(args.stimuli_template)
    materials = _load_material_covariates(args.base_dir, template_map)

    manifest = load_roi_manifest(args.roi_manifest)
    roi_frames = [
        filter_roi_manifest(manifest, roi_set=roi_set, include_flag="include_in_rsa")
        for roi_set in args.roi_sets
    ]
    rois = pd.concat(roi_frames, ignore_index=True)
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])
    if args.subjects:
        wanted = set(args.subjects)
        subject_dirs = [path for path in subject_dirs if path.name in wanted]

    all_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for roi in rois.itertuples(index=False):
            rows, qcs = _compute_learning_rows(
                subject_dir,
                str(roi.roi_set),
                str(roi.roi_name),
                Path(str(roi.mask_path)),
                template_map,
            )
            all_rows.extend(rows)
            qc_rows.extend(qcs)

    learning = pd.DataFrame(all_rows)
    qc = pd.DataFrame(qc_rows)
    pair = _load_pair_metrics(args.base_dir, template_map)
    pseudo, pseudo_qc = _compute_pseudo_edge_drops(subject_dirs, rois, template_map)
    behavior = _load_behavior(args.base_dir, template_map)

    table = learning.merge(
        pair,
        on=["subject", "roi_set", "roi", "condition", "template_pair_id", "original_pair_id", "condition_item_id"],
        how="left",
    )
    table = table.merge(pseudo, on=["subject", "roi_set", "roi", "condition"], how="left")
    table = table.merge(
        behavior,
        on=["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"],
        how="left",
    )
    table = table.merge(materials, on=["condition", "original_pair_id"], how="left")
    table["post_edge_specificity"] = table["trained_edge_drop"] - table["pseudo_edge_drop"]
    table["pair_id"] = table["original_pair_id"]
    table["material_covariate_missing"] = table[
        [col for col in ["sentence_char_len", "word_frequency_mean", "stroke_count_mean", "valence_mean", "arousal_mean"] if col in table.columns]
    ].isna().sum(axis=1)
    table = _add_z_columns(
        table,
        [
            "learning_self_similarity",
            "learning_specificity",
            "pre_pair_similarity",
            "post_pair_similarity",
            "trained_edge_drop",
            "pseudo_edge_drop",
            "post_edge_specificity",
            "retrieval_pair_similarity",
            "retrieval_minus_post_pair_similarity",
            "retrieval_minus_pre_pair_similarity",
            "word_frequency_mean",
            "stroke_count_mean",
            "valence_mean",
            "arousal_mean",
            "sentence_char_len",
        ],
    )

    qc_summary = qc.groupby(["roi_set", "roi", "condition"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        ok_rate=("ok", "mean"),
        mean_common_valid_items=("n_common_valid_items", "mean"),
        min_common_valid_items=("n_common_valid_items", "min"),
        mean_finite_cell_rate=("finite_similarity_cells", "sum"),
        total_similarity_cells=("total_similarity_cells", "sum"),
    )
    qc_summary["mean_finite_cell_rate"] = qc_summary["mean_finite_cell_rate"] / qc_summary["total_similarity_cells"].replace(0, np.nan)

    write_table(table, out_dir / "item_mechanism_table.tsv")
    write_table(qc, out_dir / "item_mechanism_qc.tsv")
    write_table(qc_summary, out_dir / "item_mechanism_qc_summary.tsv")
    write_table(pseudo, out_dir / "item_mechanism_pseudo_edge.tsv")
    write_table(pseudo_qc, out_dir / "item_mechanism_pseudo_edge_qc.tsv")
    manifest_payload = {
        "script": str(Path(__file__).resolve()),
        "roi_sets": args.roi_sets,
        "n_rows": int(len(table)),
        "n_subjects": int(table["subject"].nunique()) if not table.empty else 0,
        "n_rois": int(table[["roi_set", "roi"]].drop_duplicates().shape[0]) if not table.empty else 0,
        "id_policy": {
            "pair_id": "original material ID, 1..40 with condition-specific missing IDs",
            "template_pair_id": "compact 1..35 ID used by stimuli_template.csv and pair-similarity outputs",
            "condition_item_id": "condition + '_' + original_pair_id",
            "yy_missing_original_ids": sorted(MISSING_ORIGINAL_IDS["yy"]),
            "kj_missing_original_ids": sorted(MISSING_ORIGINAL_IDS["kj"]),
        },
    }
    save_json(manifest_payload, out_dir / "item_mechanism_manifest.json")
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
