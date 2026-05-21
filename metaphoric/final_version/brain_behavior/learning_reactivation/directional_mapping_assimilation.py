#!/usr/bin/env python3
"""Compute post-pre directional mapping / assimilation indices."""

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
    fisher_corr,
    load_mask,
    load_template_map,
    masked_samples,
    read_any,
    write_tsv,
    fit_gee_by_roi,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _load_word_meta(path: Path) -> pd.DataFrame:
    frame = read_any(path).reset_index(drop=True)
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    return frame


def _index(meta: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, row in meta.iterrows():
        label = str(row.get("word_label", "")).strip()
        if label and label.lower() != "nan" and label not in out:
            out[label] = int(idx)
    return out


def _subject_roi_condition(
    *,
    subject_dir: Path,
    roi_set: str,
    roi: str,
    mask_path: Path,
    mask: np.ndarray,
    condition: str,
    template_condition: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    qcs: list[dict[str, object]] = []
    subject = subject_dir.name
    try:
        pre_meta = _load_word_meta(subject_dir / f"pre_{condition}_metadata.tsv")
        post_meta = _load_word_meta(subject_dir / f"post_{condition}_metadata.tsv")
        pre_samples = masked_samples(subject_dir / f"pre_{condition}.nii.gz", mask, mask_path)
        post_samples = masked_samples(subject_dir / f"post_{condition}.nii.gz", mask, mask_path)
        if len(pre_meta) != pre_samples.shape[0] or len(post_meta) != post_samples.shape[0]:
            raise ValueError("metadata/image row mismatch")
        pre_idx = _index(pre_meta)
        post_idx = _index(post_meta)
        for _, item in template_condition.iterrows():
            original = int(item["original_pair_id"])
            if original not in VALID_ORIGINAL_IDS[condition]:
                continue
            a_label = str(item["role_a_label"])
            b_label = str(item["role_b_label"])
            if a_label not in pre_idx or b_label not in pre_idx or a_label not in post_idx or b_label not in post_idx:
                continue
            pre_a = pre_samples[pre_idx[a_label]]
            pre_b = pre_samples[pre_idx[b_label]]
            post_a = post_samples[post_idx[a_label]]
            post_b = post_samples[post_idx[b_label]]
            pre_ab = fisher_corr(pre_a, pre_b)
            post_a_pre_a = fisher_corr(post_a, pre_a)
            post_b_pre_b = fisher_corr(post_b, pre_b)
            post_a_pre_b = fisher_corr(post_a, pre_b)
            post_b_pre_a = fisher_corr(post_b, pre_a)
            target_toward_source = post_a_pre_b - pre_ab if np.isfinite(post_a_pre_b) and np.isfinite(pre_ab) else np.nan
            source_toward_target = post_b_pre_a - pre_ab if np.isfinite(post_b_pre_a) and np.isfinite(pre_ab) else np.nan
            target_source_over_self = (
                post_a_pre_b - post_a_pre_a
                if np.isfinite(post_a_pre_b) and np.isfinite(post_a_pre_a)
                else np.nan
            )
            source_target_over_self = (
                post_b_pre_a - post_b_pre_b
                if np.isfinite(post_b_pre_a) and np.isfinite(post_b_pre_b)
                else np.nan
            )
            rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "original_pair_id": original,
                    "template_pair_id": int(item["template_pair_id"]),
                    "condition_item_id": condition_item_id(condition, original),
                    "role_a_label": a_label,
                    "role_b_label": b_label,
                    "pre_role_a_role_b": pre_ab,
                    "post_role_a_pre_role_a": post_a_pre_a,
                    "post_role_b_pre_role_b": post_b_pre_b,
                    "post_role_a_pre_role_b": post_a_pre_b,
                    "post_role_b_pre_role_a": post_b_pre_a,
                    "target_toward_source": target_toward_source,
                    "source_toward_target": source_toward_target,
                    "mapping_asymmetry": target_toward_source - source_toward_target if np.isfinite(target_toward_source) and np.isfinite(source_toward_target) else np.nan,
                    "target_source_over_self": target_source_over_self,
                    "source_target_over_self": source_target_over_self,
                    "mapping_asymmetry_self_corrected": (
                        target_source_over_self - source_target_over_self
                        if np.isfinite(target_source_over_self) and np.isfinite(source_target_over_self)
                        else np.nan
                    ),
                }
            )
        qcs.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": True,
                "fail_reason": "",
                "n_rows": len([r for r in rows if r["condition"] == condition]),
            }
        )
    except Exception as exc:
        qcs.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": False,
                "fail_reason": str(exc),
                "n_rows": 0,
            }
        )
    return rows, qcs


def _fit_directional_models(item: pd.DataFrame) -> pd.DataFrame:
    formulas = [
        ("mapping_asymmetry_condition", "mapping_asymmetry ~ C(condition, Treatment(reference='kj'))", "mapping_asymmetry"),
        ("target_toward_source_condition", "target_toward_source ~ C(condition, Treatment(reference='kj'))", "target_toward_source"),
        (
            "mapping_asymmetry_self_corrected_condition",
            "mapping_asymmetry_self_corrected ~ C(condition, Treatment(reference='kj'))",
            "mapping_asymmetry_self_corrected",
        ),
        (
            "target_source_over_self_condition",
            "target_source_over_self ~ C(condition, Treatment(reference='kj'))",
            "target_source_over_self",
        ),
    ]
    results = []
    for model, formula, outcome in formulas:
        results.append(fit_gee_by_roi(item, formula=formula, outcome=outcome, model_name=model))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


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
                        }
                    )
                continue
            for condition in CONDITIONS:
                item_rows, item_qc = _subject_roi_condition(
                    subject_dir=subject_dir,
                    roi_set=str(roi_row.roi_set),
                    roi=str(roi_row.roi_name),
                    mask_path=mask_path,
                    mask=mask,
                    condition=condition,
                    template_condition=template[template["condition"].eq(condition)].copy(),
                )
                rows.extend(item_rows)
                qcs.extend(item_qc)
    item = pd.DataFrame(rows)
    qc = pd.DataFrame(qcs)
    models = _fit_directional_models(item) if not item.empty and item["subject"].nunique() > 1 else pd.DataFrame()
    write_tsv(item, args.out_dir / "directional_mapping_item.tsv")
    write_tsv(qc, args.out_dir / "directional_mapping_qc.tsv")
    write_tsv(models, args.out_dir / "directional_mapping_models.tsv")
    summary = {
        "script": str(Path(__file__).resolve()),
        "n_item_rows": int(len(item)),
        "n_subjects": int(item["subject"].nunique()) if not item.empty else 0,
        "n_rois": int(item[["roi_set", "roi"]].drop_duplicates().shape[0]) if not item.empty else 0,
        "qc_ok_rate": float(qc["ok"].mean()) if "ok" in qc else None,
        "n_model_rows": int(len(models)),
    }
    (args.out_dir / "directional_mapping_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
