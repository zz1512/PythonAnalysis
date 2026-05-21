#!/usr/bin/env python3
"""M6: pre/post isolated-word univariate repetition/novelty check."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from shared import add_common_args, bh_fdr, default_config, fit_formula, load_mask, load_stage, masked_samples, one_sample_summary, roi_masks, subject_dirs, write_standard_outputs, make_condition_item_id

MODULE = "m6_novelty_repetition"


def run_analysis(cfg):
    rows, qc_rows = [], []
    masks_by_set = roi_masks(cfg)
    subjects = subject_dirs(cfg.pattern_root, ["pre", "post"])
    for subject_dir in subjects:
        for condition in ("yy", "kj"):
            try:
                pre_meta, pre_data, pre_img = load_stage(subject_dir, "pre", condition)
                post_meta, post_data, post_img = load_stage(subject_dir, "post", condition)
            except Exception as exc:
                qc_rows.append({"subject": subject_dir.name, "condition": condition, "ok": False, "message": str(exc)})
                continue
            for roi_set, masks in masks_by_set.items():
                for roi, mask_path in masks.items():
                    try:
                        _, mask = load_mask(mask_path)
                        pre_values = np.nanmean(masked_samples(pre_data, mask, pre_img, mask_path), axis=1)
                        post_values = np.nanmean(masked_samples(post_data, mask, post_img, mask_path), axis=1)
                        pre = pre_meta.copy(); pre["activation"] = pre_values; pre["phase"] = "pre"
                        post = post_meta.copy(); post["activation"] = post_values; post["phase"] = "post"
                        key = "pair_id" if "pair_id" in pre.columns and "pair_id" in post.columns else None
                        if key:
                            pre_pair = pre.groupby(key, dropna=False)["activation"].mean().rename("pre_activation")
                            post_pair = post.groupby(key, dropna=False)["activation"].mean().rename("post_activation")
                            merged = pd.concat([pre_pair, post_pair], axis=1).dropna().reset_index()
                            for rec in merged.to_dict("records"):
                                pair_id = rec.get(key)
                                rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "pair_id": pair_id, "condition_item_id": make_condition_item_id(condition, pair_id), "pre_activation": rec["pre_activation"], "post_activation": rec["post_activation"], "post_minus_pre": rec["post_activation"] - rec["pre_activation"]})
                        else:
                            rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "pair_id": "__mean__", "condition_item_id": make_condition_item_id(condition, "__mean__"), "pre_activation": float(np.nanmean(pre_values)), "post_activation": float(np.nanmean(post_values)), "post_minus_pre": float(np.nanmean(post_values) - np.nanmean(pre_values))})
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": True})
                    except Exception as exc:
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": False, "message": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def condition_model(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    rows = []
    subject_level = item.groupby(["subject", "roi_set", "roi", "condition"], dropna=False)["post_minus_pre"].mean().reset_index()
    for (roi_set, roi), sub in subject_level.groupby(["roi_set", "roi"], dropna=False):
        model = fit_formula(
            sub,
            "post_minus_pre ~ C(condition, Treatment(reference='kj'))",
            group_col="subject",
            prefer_mixed=True,
            allow_fallback=True,
        )
        model.insert(0, "roi", roi)
        model.insert(0, "roi_set", roi_set)
        rows.append(model)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "p" in out:
        ok = out["status"].astype(str).str.startswith(("ok", "fallback"))
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = default_config(args)
    item, qc = run_analysis(cfg)
    subject = item.groupby(["subject", "roi_set", "roi", "condition"], dropna=False)["post_minus_pre"].mean().reset_index() if not item.empty else pd.DataFrame()
    group = one_sample_summary(subject, ["roi_set", "roi", "condition"], "post_minus_pre") if not subject.empty else pd.DataFrame()
    outputs = {"post_minus_pre_amplitude.tsv": item, "post_minus_pre_subject.tsv": subject, "post_minus_pre_group.tsv": group, "post_minus_pre_condition_model.tsv": condition_model(item), "novelty_repetition_qc.tsv": qc}
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
