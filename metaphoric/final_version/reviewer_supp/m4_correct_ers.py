#!/usr/bin/env python3
"""M4: learning-to-retrieval same-vs-other ERS with memory moderation."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from shared import add_common_args, bh_fdr, corr, default_config, fit_formula, load_mask, load_stage, masked_samples, one_sample_summary, roi_masks, subject_dirs, write_standard_outputs, zscore, make_condition_item_id

MODULE = "m4_correct_ers"


def mean_by_pair(meta: pd.DataFrame, samples: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    if "pair_id" not in meta.columns:
        return out
    for pair_id, sub in meta.groupby("pair_id", dropna=False):
        if not pair_id:
            continue
        idx = sub.index.to_numpy(dtype=int)
        out[str(pair_id)] = np.nanmean(samples[idx, :], axis=0)
    return out


def memory_by_pair(meta: pd.DataFrame) -> dict[str, float]:
    if "pair_id" not in meta.columns or "memory" not in meta.columns:
        return {}
    return meta.groupby("condition_item_id", dropna=False)["memory"].mean().to_dict()


def run_analysis(cfg):
    item_rows, qc_rows = [], []
    masks_by_set = roi_masks(cfg)
    subjects = subject_dirs(cfg.pattern_root, ["learn", "retrieval"])
    for subject_dir in subjects:
        for condition in ("yy", "kj"):
            try:
                learn_meta, learn_data, learn_img = load_stage(subject_dir, "learn", condition)
                ret_meta, ret_data, ret_img = load_stage(subject_dir, "retrieval", condition)
            except Exception as exc:
                qc_rows.append({"subject": subject_dir.name, "condition": condition, "ok": False, "message": str(exc)})
                continue
            for roi_set, masks in masks_by_set.items():
                for roi, mask_path in masks.items():
                    try:
                        _, mask = load_mask(mask_path)
                        learn_samples = masked_samples(learn_data, mask, learn_img, mask_path)
                        ret_samples = masked_samples(ret_data, mask, ret_img, mask_path)
                        learn_pairs = mean_by_pair(learn_meta, learn_samples)
                        ret_pairs = mean_by_pair(ret_meta, ret_samples)
                        memories = memory_by_pair(ret_meta)
                        shared = sorted(set(learn_pairs) & set(ret_pairs))
                        for pair_id in shared:
                            condition_item_id = make_condition_item_id(condition, pair_id)
                            memory = memories.get(condition_item_id, np.nan)
                            same = corr(learn_pairs[pair_id], ret_pairs[pair_id])
                            other_values = [corr(learn_pairs[pair_id], ret_pairs[other]) for other in ret_pairs if other != pair_id]
                            other = float(np.nanmean(other_values)) if other_values else np.nan
                            item_rows.append({
                                "subject": subject_dir.name,
                                "roi_set": roi_set,
                                "roi": roi,
                                "condition": condition,
                                "pair_id": pair_id,
                                "condition_item_id": condition_item_id,
                                "ers_same": same,
                                "ers_other_mean": other,
                                "ers_same_minus_other": same - other if np.isfinite(same) and np.isfinite(other) else np.nan,
                                "memory": memory,
                                "remembered": 1 if memory >= 0.5 else (0 if np.isfinite(memory) else np.nan),
                                "memory_binary_rule": "lenient_memory_ge_0.5",
                                "n_other_pairs": len(other_values),
                            })
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": True, "n_shared_pairs": len(shared)})
                    except Exception as exc:
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": False, "message": str(exc)})
    return pd.DataFrame(item_rows), pd.DataFrame(qc_rows)


def dm_model(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    work = item.dropna(subset=["ers_same_minus_other", "remembered", "condition"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["ers_z"] = zscore(work["ers_same_minus_other"])
    rows = []
    for (roi_set, roi), sub in work.groupby(["roi_set", "roi"], dropna=False):
        model = fit_formula(
            sub,
            "ers_z ~ remembered * C(condition, Treatment(reference='kj'))",
            group_col="subject",
            item_col="condition_item_id",
            prefer_mixed=True,
        )
        model.insert(0, "roi", roi)
        model.insert(0, "roi_set", roi_set)
        rows.append(model)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "p" in out:
        ok = out["status"].eq("ok")
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = default_config(args)
    item, qc = run_analysis(cfg)
    group = one_sample_summary(item, ["roi_set", "roi", "condition"], "ers_same_minus_other") if not item.empty else pd.DataFrame()
    outputs = {
        "ers_item.tsv": item,
        "ers_same_vs_other.tsv": group,
        "ers_dm_interaction.tsv": dm_model(item),
        "ers_qc.tsv": qc,
    }
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
