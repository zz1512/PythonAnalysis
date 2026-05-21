#!/usr/bin/env python3
"""B2a: build a covariance-normalized distance robustness table from 4D patterns.

The original B2 audit expects a crossnobis/mahalanobis-family input. The current
dataset has pre/post split across runs, but each lexical item is estimated once
per stage, so item-level pair distances cannot be cross-validated in the strict
crossnobis sense. This module therefore computes a conservative diagonal
shrinkage Mahalanobis proxy and writes explicit estimator labels.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

FINAL_ROOT = Path(__file__).resolve().parents[1]
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from shared_nc import add_common_args, bh_fdr, default_config, roi_to_network, write_outputs, zscore  # noqa: E402

MODULE = "b2a_mahalanobis_from_patterns"
STAGES = ("pre", "post")
CONDITIONS = ("yy", "kj")
CONDITION_LABELS = {"yy": "Metaphor", "kj": "Spatial"}
ESTIMATOR = "diagonal_shrinkage_mahalanobis_not_crossvalidated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--max-subjects", type=int, default=None, help="Development-only subject limit.")
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def normalize_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        numeric = float(text)
    except Exception:
        return text
    if math.isnan(numeric):
        return ""
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def condition_item_id(condition: str, pair_id: object) -> str:
    pair = normalize_pair_id(pair_id)
    return f"{condition}_{pair}" if pair else ""


def template_item_id(row: pd.Series, condition: str) -> str:
    for col in ("word_label", "unique_label", "real_word"):
        if col not in row.index:
            continue
        match = re.search(r"(\d+)\s*$", str(row[col]))
        if match:
            return match.group(1)
    return normalize_pair_id(row.get("pair_id", ""))


def load_roi_manifest(path: Path, roi_sets: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    needed = {"roi_set", "roi_name", "mask_path"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"ROI manifest missing columns: {sorted(missing)}")
    out = frame[frame["roi_set"].astype(str).isin(roi_sets)].copy()
    if "include_in_rsa" in out.columns:
        out = out[out["include_in_rsa"].map(truthy)]
    out = out[out["mask_path"].map(lambda item: Path(str(item)).exists())]
    out["network"] = [roi_to_network(r, s) for r, s in zip(out["roi_name"], out["roi_set"])]
    out = out[out["network"].notna()].sort_values(["roi_set", "roi_name"]).reset_index(drop=True)
    if out.empty:
        raise FileNotFoundError(f"No usable ROI masks found in {path} for {roi_sets}")
    return out


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(nib.load(str(path)).get_fdata()) > 0
    if not mask.any():
        raise ValueError(f"Empty ROI mask: {path}")
    return mask


def load_stage(subject_dir: Path, stage: str, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"{stage}_{condition}.nii.gz"
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing pattern image/metadata: {image_path}")
    meta = pd.read_csv(meta_path, sep="\t", low_memory=False).reset_index(drop=True)
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.shape[3] != len(meta):
        raise ValueError(f"Volume/metadata mismatch: {image_path} has {data.shape[3]}, metadata has {len(meta)}")
    meta["subject"] = subject_dir.name
    meta["stage"] = stage
    meta["condition_code"] = condition
    meta["condition"] = CONDITION_LABELS[condition]
    meta["pair_id_norm"] = meta["pair_id"].map(normalize_pair_id) if "pair_id" in meta.columns else ""
    meta["template_pair_id"] = [template_item_id(row, condition) for _, row in meta.iterrows()]
    meta["condition_item_id"] = [condition_item_id(condition, pid) for pid in meta["template_pair_id"]]
    if "run" in meta.columns:
        meta["run"] = pd.to_numeric(meta["run"], errors="coerce")
    return meta, data, image_path


def diagonal_mahalanobis_pair_rows(
    meta: pd.DataFrame,
    samples: np.ndarray,
    *,
    subject: str,
    stage: str,
    condition_code: str,
    roi_set: str,
    roi: str,
    network: str,
    n_voxels: int,
) -> list[dict[str, object]]:
    finite_cols = np.isfinite(samples).all(axis=0)
    samples = samples[:, finite_cols]
    if samples.shape[1] < 3:
        return []
    variance = np.nanvar(samples, axis=0, ddof=1)
    valid_var = variance[np.isfinite(variance) & (variance > 0)]
    if valid_var.size == 0:
        return []
    ridge = max(float(np.nanmedian(valid_var)) * 0.05, 1e-6)
    denom = variance + ridge
    rows: list[dict[str, object]] = []
    for pair_id, sub in meta.groupby("template_pair_id", dropna=False):
        if not pair_id or len(sub) != 2:
            continue
        idx = sub.index.to_numpy(dtype=int)
        delta = samples[idx[0], :] - samples[idx[1], :]
        valid = np.isfinite(delta) & np.isfinite(denom) & (denom > 0)
        if valid.sum() < 3:
            continue
        distance = float(np.mean((delta[valid] ** 2) / denom[valid]))
        run_values = sorted({int(v) for v in sub.get("run", pd.Series(dtype=float)).dropna().tolist()})
        rows.append(
            {
                "subject": subject,
                "stage": stage,
                "condition_code": condition_code,
                "condition": CONDITION_LABELS[condition_code],
                "condition_item_id": condition_item_id(condition_code, pair_id),
                "pair_id": pair_id,
                "source_pair_id": ",".join(sorted(set(sub["pair_id_norm"].dropna().astype(str)))),
                "roi_set": roi_set,
                "roi": roi,
                "network": network,
                "n_voxels_mask": n_voxels,
                "n_voxels_used": int(valid.sum()),
                "distance_mahalanobis": distance,
                "mahalanobis_similarity": -distance,
                "metric": "mahalanobis_similarity",
                "estimator": ESTIMATOR,
                "crossvalidated": False,
                "runs_with_pair_words": ",".join(map(str, run_values)),
            }
        )
    return rows


def compute_pair_table(pattern_root: Path, rois: pd.DataFrame, max_subjects: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(path for path in pattern_root.glob("sub-*") if path.is_dir())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    mask_cache = {row.roi_name: load_mask(Path(str(row.mask_path))) for row in rois.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for subject_dir in subjects:
        for stage in STAGES:
            for condition in CONDITIONS:
                try:
                    meta, data, image_path = load_stage(subject_dir, stage, condition)
                except Exception as exc:
                    manifest_rows.append(
                        {
                            "subject": subject_dir.name,
                            "stage": stage,
                            "condition": condition,
                            "status": f"missing_or_failed_stage: {exc}",
                        }
                    )
                    continue
                for row in rois.itertuples(index=False):
                    mask = mask_cache[row.roi_name]
                    if data.shape[:3] != mask.shape:
                        manifest_rows.append(
                            {
                                "subject": subject_dir.name,
                                "stage": stage,
                                "condition": condition,
                                "roi": row.roi_name,
                                "status": "image_mask_shape_mismatch",
                                "image_path": str(image_path),
                                "mask_path": str(row.mask_path),
                            }
                        )
                        continue
                    samples = data[mask, :].T.astype(np.float64, copy=False)
                    before = len(rows)
                    rows.extend(
                        diagonal_mahalanobis_pair_rows(
                            meta,
                            samples,
                            subject=subject_dir.name,
                            stage=stage,
                            condition_code=condition,
                            roi_set=str(row.roi_set),
                            roi=str(row.roi_name),
                            network=str(row.network),
                            n_voxels=int(mask.sum()),
                        )
                    )
                    manifest_rows.append(
                        {
                            "subject": subject_dir.name,
                            "stage": stage,
                            "condition": condition,
                            "roi_set": row.roi_set,
                            "roi": row.roi_name,
                            "network": row.network,
                            "status": "ok",
                            "n_pair_rows": len(rows) - before,
                            "image_path": str(image_path),
                            "mask_path": str(row.mask_path),
                        }
                    )
                del data
    return pd.DataFrame(rows), pd.DataFrame(manifest_rows)


def add_network_composite(pair_table: pd.DataFrame) -> pd.DataFrame:
    if pair_table.empty:
        return pd.DataFrame()
    out = pair_table.copy()
    out["mahalanobis_similarity_z"] = out.groupby(["subject", "roi"], dropna=False)["mahalanobis_similarity"].transform(zscore)
    group_cols = ["subject", "stage", "condition_code", "condition", "condition_item_id", "pair_id", "network", "metric", "estimator"]
    return (
        out.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            mahalanobis_similarity=("mahalanobis_similarity_z", "mean"),
            distance_mahalanobis=("distance_mahalanobis", "mean"),
            n_roi=("roi", "nunique"),
            n_voxels_used_mean=("n_voxels_used", "mean"),
        )
        .assign(roi="__network_composite__", roi_set="network_composite")
    )


def build_delta_table(stage_table: pd.DataFrame) -> pd.DataFrame:
    if stage_table.empty:
        return pd.DataFrame()
    keys = ["subject", "condition_code", "condition", "condition_item_id", "pair_id", "roi_set", "roi", "network", "metric", "estimator"]
    wide = stage_table.pivot_table(index=keys, columns="stage", values="mahalanobis_similarity", aggfunc="mean").reset_index()
    if "pre" not in wide.columns or "post" not in wide.columns:
        return pd.DataFrame()
    wide["pre_mahalanobis_similarity"] = pd.to_numeric(wide["pre"], errors="coerce")
    wide["post_mahalanobis_similarity"] = pd.to_numeric(wide["post"], errors="coerce")
    wide["delta_post_minus_pre"] = wide["post_mahalanobis_similarity"] - wide["pre_mahalanobis_similarity"]
    wide["estimate"] = wide["delta_post_minus_pre"]
    wide["term"] = "post_minus_pre_mahalanobis_similarity"
    return wide.drop(columns=[c for c in ["pre", "post"] if c in wide.columns])


def one_sample_rows(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta.empty:
        return pd.DataFrame()
    for keys, sub in delta.groupby(["network", "roi_set", "roi", "condition"], dropna=False):
        values = pd.to_numeric(sub["delta_post_minus_pre"], errors="coerce").dropna()
        if len(values) < 2:
            t_value = p_value = mean = sd = dz = np.nan
        else:
            t_value, p_value = stats.ttest_1samp(values, 0.0, nan_policy="omit")
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            dz = mean / sd if sd and np.isfinite(sd) else np.nan
        network, roi_set, roi, condition = keys
        rows.append(
            {
                "network": network,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "term": "post_minus_pre_mahalanobis_similarity",
                "metric": "mahalanobis_similarity",
                "estimate": mean,
                "se": sd / math.sqrt(len(values)) if len(values) > 0 and np.isfinite(sd) else np.nan,
                "stat": t_value,
                "p": p_value,
                "cohens_dz": dz,
                "n_obs": int(len(values)),
                "n_subjects": int(sub["subject"].nunique()),
                "n_items": int(sub["condition_item_id"].nunique()),
                "status": "ok" if len(values) >= 2 else "too_few_rows",
                "estimator": ESTIMATOR,
                "crossvalidated": False,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out["status"].eq("ok")
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def condition_contrast_rows(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta.empty:
        return pd.DataFrame()
    subject_item = (
        delta.groupby(["network", "roi_set", "roi", "subject", "condition"], dropna=False, as_index=False)
        .agg(delta_post_minus_pre=("delta_post_minus_pre", "mean"))
    )
    for keys, sub in subject_item.groupby(["network", "roi_set", "roi"], dropna=False):
        piv = sub.pivot_table(index="subject", columns="condition", values="delta_post_minus_pre", aggfunc="mean")
        if not {"Metaphor", "Spatial"}.issubset(piv.columns):
            continue
        diff = (piv["Metaphor"] - piv["Spatial"]).dropna()
        if len(diff) < 2:
            t_value = p_value = mean = sd = dz = np.nan
            status = "too_few_rows"
        else:
            t_value, p_value = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = mean / sd if sd and np.isfinite(sd) else np.nan
            status = "ok"
        network, roi_set, roi = keys
        rows.append(
            {
                "network": network,
                "roi_set": roi_set,
                "roi": roi,
                "condition": "Metaphor_minus_Spatial",
                "term": "condition_contrast_metaphor_minus_spatial_on_post_minus_pre",
                "metric": "mahalanobis_similarity",
                "estimate": mean,
                "se": sd / math.sqrt(len(diff)) if len(diff) > 0 and np.isfinite(sd) else np.nan,
                "stat": t_value,
                "p": p_value,
                "cohens_dz": dz,
                "n_obs": int(len(diff)),
                "n_subjects": int(len(diff)),
                "n_items": np.nan,
                "status": status,
                "estimator": ESTIMATOR,
                "crossvalidated": False,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out["status"].eq("ok")
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def method_note(pair_table: pd.DataFrame, delta: pd.DataFrame) -> str:
    n_subjects = pair_table["subject"].nunique() if "subject" in pair_table else 0
    n_rois = pair_table["roi"].nunique() if "roi" in pair_table else 0
    return "\n".join(
        [
            "# B2a Mahalanobis Robustness Method Note",
            "",
            f"- Estimator: `{ESTIMATOR}`.",
            "- Strict item-level crossnobis was not computed because the pattern metadata contain unique lexical-item estimates rather than repeated independent estimates for the same item in both pre/post partitions.",
            "- The robustness metric is therefore a diagonal shrinkage Mahalanobis distance between the two words in each trained pair, scaled as a similarity-like score by multiplying distance by -1.",
            "- Negative `delta_post_minus_pre` means post-stage pair representations are more separated than pre-stage pair representations.",
            f"- Pair rows: {len(pair_table)}; delta rows: {len(delta)}; subjects: {n_subjects}; ROI entries: {n_rois}.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    pattern_root = Path(args.pattern_root or cfg.base_dir / "pattern_root")
    roi_manifest = Path(args.roi_manifest or cfg.base_dir / "roi_library" / "manifest.tsv")
    rois = load_roi_manifest(roi_manifest, list(args.roi_sets))
    pair_table, manifest = compute_pair_table(pattern_root, rois, args.max_subjects)
    if pair_table.empty and not args.allow_empty:
        raise FileNotFoundError("No Mahalanobis pair rows could be computed from pattern_root and ROI masks.")
    network_stage = add_network_composite(pair_table)
    stage_table = pd.concat([pair_table, network_stage], ignore_index=True, sort=False) if not network_stage.empty else pair_table
    delta = build_delta_table(stage_table)
    step5c = pd.concat([one_sample_rows(delta), condition_contrast_rows(delta)], ignore_index=True, sort=False)
    write_outputs(
        cfg,
        MODULE,
        {
            "mahalanobis_pair_distance.tsv": pair_table,
            "mahalanobis_network_stage.tsv": network_stage,
            "mahalanobis_subject_delta.tsv": delta,
            "mahalanobis_step5c.tsv": step5c,
            "mahalanobis_input_manifest.tsv": manifest,
            "mahalanobis_method_note.md": method_note(pair_table, delta),
        },
    )


if __name__ == "__main__":
    main()
