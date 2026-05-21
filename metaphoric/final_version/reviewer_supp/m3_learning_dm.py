#!/usr/bin/env python3
"""M3: learning-stage subsequent-memory (Dm) analysis."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from shared import add_common_args, corr, default_config, fit_formula, load_mask, load_stage, masked_samples, one_sample_summary, roi_masks, subject_dirs, write_standard_outputs, zscore, make_condition_item_id

MODULE = "m3_learning_dm"


def retrieval_memory_map(subject_dir, condition: str) -> dict[str, float]:
    try:
        meta, _, _ = load_stage(subject_dir, "retrieval", condition)
    except Exception:
        return {}
    if "pair_id" not in meta.columns or "memory" not in meta.columns:
        return {}
    return meta.groupby("condition_item_id", dropna=False)["memory"].mean().to_dict()


def run_analysis(cfg):
    item_rows, qc_rows = [], []
    masks_by_set = roi_masks(cfg)
    subjects = subject_dirs(cfg.pattern_root, ["learn", "retrieval"])
    for subject_dir in subjects:
        for condition in ("yy", "kj"):
            mem = retrieval_memory_map(subject_dir, condition)
            try:
                meta, data, image_path = load_stage(subject_dir, "learn", condition)
            except Exception as exc:
                qc_rows.append({"subject": subject_dir.name, "condition": condition, "ok": False, "message": str(exc)})
                continue
            if "pair_id" not in meta.columns:
                qc_rows.append({"subject": subject_dir.name, "condition": condition, "ok": False, "message": "learn metadata missing pair_id"})
                continue
            for roi_set, masks in masks_by_set.items():
                for roi, mask_path in masks.items():
                    try:
                        _, mask = load_mask(mask_path)
                        samples = masked_samples(data, mask, image_path, mask_path)
                        meta_work = meta.copy()
                        meta_work["trial_mean_activation"] = np.nanmean(samples, axis=1)
                        for pair_id, sub in meta_work.groupby("pair_id", dropna=False):
                            if not pair_id:
                                continue
                            condition_item_id = make_condition_item_id(condition, pair_id)
                            memory = mem.get(condition_item_id, np.nan)
                            remembered = np.nan
                            if np.isfinite(memory):
                                remembered = 1 if memory >= 0.5 else 0
                            pair_samples = samples[sub.index.to_numpy(dtype=int), :]
                            row = {
                                "subject": subject_dir.name,
                                "roi_set": roi_set,
                                "roi": roi,
                                "condition": condition,
                                "pair_id": pair_id,
                                "condition_item_id": condition_item_id,
                                "memory": memory,
                                "remembered": remembered,
                                "memory_binary_rule": "lenient_memory_ge_0.5",
                                "learning_beta": float(np.nanmean(meta_work.loc[sub.index, "trial_mean_activation"])),
                                "within_pair_learning_similarity": np.nan,
                                "n_learning_trials": int(len(sub)),
                            }
                            if "run" in sub.columns:
                                r3 = sub.index[sub["run"].astype("Int64").eq(3)].to_numpy(dtype=int)
                                r4 = sub.index[sub["run"].astype("Int64").eq(4)].to_numpy(dtype=int)
                                if r3.size and r4.size:
                                    row["within_pair_learning_similarity"] = corr(np.nanmean(samples[r3, :], axis=0), np.nanmean(samples[r4, :], axis=0))
                            elif pair_samples.shape[0] >= 2:
                                row["within_pair_learning_similarity"] = corr(pair_samples[0], pair_samples[1])
                            item_rows.append(row)
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": True, "n_pairs": int(meta_work["pair_id"].nunique())})
                    except Exception as exc:
                        qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "condition": condition, "ok": False, "message": str(exc)})
    item = pd.DataFrame(item_rows)
    qc = pd.DataFrame(qc_rows)
    return item, qc


def model_dm(item: pd.DataFrame, response: str) -> pd.DataFrame:
    if item.empty or response not in item.columns:
        return pd.DataFrame()
    work = item.dropna(subset=[response, "remembered", "condition"]).copy()
    if work.empty:
        return pd.DataFrame()
    work[f"{response}_z"] = zscore(work[response])
    rows = []
    for (roi_set, roi), sub in work.groupby(["roi_set", "roi"], dropna=False):
        model = fit_formula(
            sub,
            f"{response}_z ~ remembered * C(condition, Treatment(reference='kj'))",
            group_col="subject",
            item_col="condition_item_id",
            prefer_mixed=True,
        )
        model.insert(0, "response", response)
        model.insert(0, "roi", roi)
        model.insert(0, "roi_set", roi_set)
        rows.append(model)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "p" in out:
        out["q_bh"] = np.nan
        ok = out["status"].eq("ok")
        from shared import bh_fdr
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = default_config(args)
    item, qc = run_analysis(cfg)
    group = one_sample_summary(item.dropna(subset=["within_pair_learning_similarity"]) if not item.empty else pd.DataFrame(), ["roi_set", "roi", "condition", "remembered"], "within_pair_learning_similarity") if not item.empty else pd.DataFrame()
    outputs = {
        "learning_dm_item.tsv": item,
        "learning_dm_univariate.tsv": model_dm(item, "learning_beta"),
        "learning_dm_rsa.tsv": model_dm(item, "within_pair_learning_similarity"),
        "learning_dm_group.tsv": group,
        "learning_dm_qc.tsv": qc,
    }
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
