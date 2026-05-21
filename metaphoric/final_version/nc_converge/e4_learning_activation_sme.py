#!/usr/bin/env python3
"""Append 518 E4: learning-stage ROI activation SME from source 4D patterns.

Extract ROI-mean beta values from ``pattern_root/sub-*/learn_{yy,kj}.nii.gz``
and test whether learning-stage activation relates to later memory or
run3/run4 subjective responses. This complements RSA results because encoding
SME literature often reports activation/connectivity rather than local
pattern-similarity effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
    zscore,
)

MODULE = "e4_learning_activation_sme"
DEFAULT_ROI_MANIFEST = Path("roi_library/manifest.tsv")
DEFAULT_LEARNING_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
DEFAULT_ITEM_MECHANISM = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--learning-behavior", type=Path, default=None)
    parser.add_argument("--item-mechanism", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


def _normal_pair_id(value: object) -> str:
    try:
        val = float(str(value).strip())
        return str(int(val)) if np.isfinite(val) and val.is_integer() else str(value).strip()
    except Exception:
        return str(value).strip()


def _condition_item_id(condition: object, pair_id: object) -> str:
    cond = str(condition).strip().lower()
    pair = _normal_pair_id(pair_id)
    return f"{cond}_{pair}" if cond and pair else ""


def _coerce_binary(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "t": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "1": 1.0,
            "1.0": 1.0,
            "false": 0.0,
            "f": 0.0,
            "no": 0.0,
            "n": 0.0,
            "0": 0.0,
            "0.0": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def _load_roi_table(path: Path, roi_sets: list[str]) -> pd.DataFrame:
    table = read_table(path).copy()
    selected = {x.lower() for x in roi_sets}
    table = table[table["roi_set"].astype(str).str.lower().isin(selected)].copy()
    if "include_in_rsa" in table.columns:
        table = table[table["include_in_rsa"].astype(str).str.lower().isin({"true", "1", "yes", "y"})]
    table["mask_path"] = table["mask_path"].map(lambda x: Path(str(x)))
    table = table[table["mask_path"].map(lambda p: p.exists())].copy()
    if table.empty:
        raise FileNotFoundError(f"No ROI masks selected from {path}")
    return add_network_column(table.rename(columns={"roi_name": "roi"})).reset_index(drop=True)


def _load_masks(roi_table: pd.DataFrame) -> dict[str, np.ndarray]:
    import nibabel as nib

    masks = {}
    for row in roi_table.itertuples(index=False):
        masks[str(row.roi)] = np.asarray(nib.load(str(row.mask_path)).get_fdata()) > 0
    return masks


def extract_activation(pattern_root: Path, roi_table: pd.DataFrame, *, max_subjects: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nibabel as nib

    masks = _load_masks(roi_table)
    subject_dirs = sorted(p for p in pattern_root.glob("sub-*") if p.is_dir())
    if max_subjects is not None and max_subjects > 0:
        subject_dirs = subject_dirs[: int(max_subjects)]
    rows: list[dict] = []
    failures: list[dict] = []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for condition in ("yy", "kj"):
            image_path = subject_dir / f"learn_{condition}.nii.gz"
            meta_path = subject_dir / f"learn_{condition}_metadata.tsv"
            if not image_path.exists() or not meta_path.exists():
                failures.append({"subject": subject, "condition": condition, "status": "missing_file"})
                continue
            meta = read_table(meta_path).copy()
            data = np.asarray(nib.load(str(image_path)).get_fdata(), dtype=float)
            if data.ndim == 3:
                data = data[..., np.newaxis]
            if data.shape[3] != len(meta):
                failures.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "status": f"volume_metadata_mismatch:{data.shape[3]}!={len(meta)}",
                    }
                )
                continue
            meta["condition"] = condition
            meta["subject"] = subject
            id_source = "pic_num" if "pic_num" in meta.columns else "pair_id"
            meta["condition_item_id"] = [_condition_item_id(condition, x) for x in meta[id_source]]
            for roi_row in roi_table.itertuples(index=False):
                roi = str(roi_row.roi)
                network = getattr(roi_row, "network", None)
                mask = masks[roi]
                if mask.shape != data.shape[:3]:
                    failures.append({"subject": subject, "condition": condition, "roi": roi, "status": "mask_shape_mismatch"})
                    continue
                values = data[mask, :].mean(axis=0)
                for idx, meta_row in meta.iterrows():
                    rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "condition_item_id": str(meta_row["condition_item_id"]),
                            "original_pair_id": _normal_pair_id(meta_row[id_source]),
                            "run": int(meta_row["run"]) if "run" in meta.columns and pd.notna(meta_row["run"]) else np.nan,
                            "trial_index": int(meta_row["trial_index"]) if "trial_index" in meta.columns and pd.notna(meta_row["trial_index"]) else np.nan,
                            "roi": roi,
                            "network": network,
                            "activation_raw": float(values[int(idx)]),
                            "source_image": str(image_path),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _memory_meta(item_mechanism_path: Path) -> pd.DataFrame:
    frame = build_condition_item_id(read_table(item_mechanism_path))
    for col in ["memory", "memory_score", "memory_successes", "remembered_strict", *COVARIATES]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keys = [c for c in ["subject", "condition", "condition_item_id"] if c in frame.columns]
    cols = [c for c in ["memory", "memory_score", "memory_successes", "remembered_strict", *COVARIATES] if c in frame.columns]
    meta = frame[keys + cols].groupby(keys, dropna=False, as_index=False).mean()
    if "remembered_strict" in meta.columns:
        meta["memory_strict"] = meta["remembered_strict"]
    elif "memory_score" in meta.columns:
        ms = meta["memory_score"]
        meta["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory" in meta.columns:
        ms = meta["memory"]
        meta["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory_successes" in meta.columns:
        ms = meta["memory_successes"]
        meta["memory_strict"] = np.where(ms.ge(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    return meta


def _behavior_meta(path: Path) -> pd.DataFrame:
    frame = build_condition_item_id(read_table(path))
    keys = [c for c in ["subject", "condition", "condition_item_id"] if c in frame.columns]
    cols = [
        "run3_understand_yes",
        "run4_like_yes",
        "run3_rt_z_subject_condition",
        "run4_rt_z_subject_condition",
        "learning_fluency_shift",
    ]
    keep = [c for c in cols if c in frame.columns]
    for col in keep:
        if col in {"run3_understand_yes", "run4_like_yes"}:
            frame[col] = _coerce_binary(frame[col])
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[keys + keep].groupby(keys, dropna=False, as_index=False).mean()


def prepare(activation: pd.DataFrame, behavior_path: Path, item_mechanism_path: Path) -> pd.DataFrame:
    out = activation.copy()
    out = out.merge(_behavior_meta(behavior_path), on=["subject", "condition", "condition_item_id"], how="left")
    out = out.merge(_memory_meta(item_mechanism_path), on=["subject", "condition", "condition_item_id"], how="left")
    out["activation_z_subject_roi_run"] = out.groupby(["subject", "roi", "run"], dropna=False)["activation_raw"].transform(zscore)
    out["activation_z_subject_roi"] = out.groupby(["subject", "roi"], dropna=False)["activation_raw"].transform(zscore)
    if "learning_fluency_shift" in out.columns:
        out["learning_fluency_shift_z"] = zscore(out["learning_fluency_shift"])
    return out


def _usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().sum() >= 10]


def fit_models(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    rows_activation: list[pd.DataFrame] = []
    rows_memory: list[pd.DataFrame] = []
    for (roi, run), sub in data.groupby(["roi", "run"], dropna=False, sort=False):
        network = sub["network"].dropna().iloc[0] if "network" in sub.columns and sub["network"].notna().any() else ""
        covs = _usable_columns(sub, COVARIATES)
        cov_expr = " + " + " + ".join(covs) if covs else ""
        if int(run) == 3:
            behavior_terms = [c for c in ["run3_understand_yes", "run3_rt_z_subject_condition"] if c in sub.columns]
            behavior_expr = (" + " + " + ".join(behavior_terms)) if behavior_terms else ""
            behavior_interaction = " + C(condition, Treatment('kj')) * run3_understand_yes" if "run3_understand_yes" in sub.columns else ""
        else:
            behavior_terms = [c for c in ["run4_like_yes", "run4_rt_z_subject_condition"] if c in sub.columns]
            behavior_expr = (" + " + " + ".join(behavior_terms)) if behavior_terms else ""
            behavior_interaction = " + C(condition, Treatment('kj')) * run4_like_yes" if "run4_like_yes" in sub.columns else ""

        # Activation SME: does later memory explain learning activation?
        keep = ["activation_z_subject_roi_run", "memory_strict", *behavior_terms, *covs]
        model_data = sub.dropna(subset=[c for c in keep if c in sub.columns]).copy()
        if len(model_data) >= 30 and model_data["memory_strict"].nunique(dropna=True) >= 2:
            formula = (
                "activation_z_subject_roi_run ~ C(condition, Treatment('kj')) * memory_strict"
                + behavior_expr
                + cov_expr
            )
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "run", int(run))
            res.insert(3, "mechanism_model", "learning_activation_sme")
            rows_activation.append(res)

        # Neural predictor model: does learning activation predict memory?
        keep = ["memory_strict", "activation_z_subject_roi_run", *behavior_terms, *covs]
        model_data = sub.dropna(subset=[c for c in keep if c in sub.columns]).copy()
        if len(model_data) >= 30 and model_data["memory_strict"].nunique(dropna=True) >= 2:
            formula = (
                "memory_strict ~ C(condition, Treatment('kj')) * activation_z_subject_roi_run"
                + behavior_interaction
                + cov_expr
            )
            res = fit_formula(model_data, formula, family="binomial")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "run", int(run))
            res.insert(3, "mechanism_model", "memory_from_learning_activation")
            rows_memory.append(res)

    outputs["learning_activation_sme_models.tsv"] = (
        q_for_ok_rows(pd.concat(rows_activation, ignore_index=True, sort=False)) if rows_activation else pd.DataFrame()
    )
    outputs["memory_from_learning_activation_models.tsv"] = (
        q_for_ok_rows(pd.concat(rows_memory, ignore_index=True, sort=False)) if rows_memory else pd.DataFrame()
    )
    return outputs


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in data.groupby(["network", "roi", "run", "condition"], dropna=False, sort=False):
        rec = dict(zip(["network", "roi", "run", "condition"], keys))
        rec["n"] = int(len(sub))
        rec["n_subjects"] = int(sub["subject"].nunique())
        rec["n_items"] = int(sub["condition_item_id"].nunique())
        for col in ["activation_raw", "activation_z_subject_roi_run", "memory_strict", "run3_understand_yes", "run4_like_yes"]:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                rec[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    pattern_root = Path(args.pattern_root or cfg.base_dir / "pattern_root")
    roi_manifest = Path(args.roi_manifest or cfg.base_dir / DEFAULT_ROI_MANIFEST)
    behavior_path = Path(args.learning_behavior or cfg.paper_output_root / DEFAULT_LEARNING_BEHAVIOR)
    item_mechanism_path = Path(args.item_mechanism or cfg.paper_output_root / DEFAULT_ITEM_MECHANISM)
    roi_table = _load_roi_table(roi_manifest, list(args.roi_sets))
    activation_raw, failures = extract_activation(pattern_root, roi_table, max_subjects=args.max_subjects)
    data = prepare(activation_raw, behavior_path, item_mechanism_path)
    models = fit_models(data)
    write_outputs(
        cfg,
        MODULE,
        {
            "learning_activation_roi_item.tsv": data,
            "learning_activation_descriptives.tsv": descriptives(data),
            **models,
            "learning_activation_failures.tsv": failures,
            "manifest.tsv": pd.DataFrame(
                [
                    {
                        "pattern_root": str(pattern_root),
                        "roi_manifest": str(roi_manifest),
                        "roi_sets": ",".join(args.roi_sets),
                        "learning_behavior": str(behavior_path),
                        "item_mechanism": str(item_mechanism_path),
                        "n_rows": int(len(data)),
                        "n_rois": int(data["roi"].nunique()) if not data.empty else 0,
                        "n_subjects": int(data["subject"].nunique()) if not data.empty else 0,
                        "n_failures": int(len(failures)),
                        "status": "ok" if not data.empty else "empty",
                    }
                ]
            ),
        },
    )


if __name__ == "__main__":
    main()
