#!/usr/bin/env python3
"""C2: global similarity / representational centrality from RDM audit files.

Unit policy
-----------
Both ``global_similarity_change`` and ``post_edge_differentiation`` are now
aggregated to ``subject x network x condition_item_id`` with raw means before a
single global z-score is applied. The previous version z-scored within each
network, which forced YY/KJ condition main effects toward symmetric reflections
and pulled estimates into z-of-z space (the same pattern as the §28.4 mirror
artefact). Mixed-model rows that fail to converge or report singular random
effects are now flagged via ``status`` so downstream FDR (``q_for_ok_rows``)
skips them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import add_common_args, add_network_column, build_condition_item_id, default_config, fit_formula, q_for_ok_rows, read_table, roi_to_network, write_outputs, zscore

MODULE = "c2_global_similarity_centrality"
RDM_PATTERNS = [
    "qc/model_rdm_results_meta_metaphor/model_rdm_audit/*/*/*_all_rdms.npz",
    "qc/model_rdm_results_meta_spatial/model_rdm_audit/*/*/*_all_rdms.npz",
]
ITEM_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
COVARIATES = ["sentence_char_len_z", "word_frequency_mean_z", "stroke_count_mean_z", "valence_mean_z", "arousal_mean_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--item-input", type=Path, default=None)
    parser.add_argument("--rdm-input", nargs="*", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def condition_norm(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"metaphor", "yy"}:
        return "yy"
    if text in {"spatial", "kj"}:
        return "kj"
    return text


def condition_item_id(condition: object, pair_id: object) -> str:
    cond = condition_norm(condition)
    try:
        pid = int(float(str(pair_id)))
    except Exception:
        return ""
    return f"{cond}_{pid}" if cond and pid else ""


def metadata_path_for(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.name.replace("_rdms.npz", "_metadata.tsv"))


def extract_one(npz_path: Path) -> tuple[pd.DataFrame, dict]:
    meta_path = metadata_path_for(npz_path)
    log = {"path": str(npz_path), "metadata_path": str(meta_path), "status": "ok"}
    if not meta_path.exists():
        log["status"] = "missing_metadata"
        return pd.DataFrame(), log
    meta = read_table(meta_path)
    needed_meta = {"condition", "pair_id"}
    if not needed_meta.issubset(meta.columns):
        log["status"] = f"missing_metadata_columns: {sorted(needed_meta - set(meta.columns))}"
        return pd.DataFrame(), log
    try:
        data = np.load(npz_path)
        pair_i = data["pair_i"].astype(int)
        pair_j = data["pair_j"].astype(int)
        # Existing audit files store neural RDM distances; 1 - distance is used
        # here as a relative global-similarity index within each RDM.
        sim = 1.0 - data["neural_rdm"].astype(float)
    except Exception as exc:
        log["status"] = f"npz_failed: {exc}"
        return pd.DataFrame(), log
    n_items = len(meta)
    if n_items == 0 or pair_i.max(initial=-1) >= n_items or pair_j.max(initial=-1) >= n_items:
        log["status"] = "metadata_matrix_mismatch"
        log["n_metadata"] = n_items
        return pd.DataFrame(), log
    sums = np.zeros(n_items, dtype=float)
    counts = np.zeros(n_items, dtype=float)
    valid = np.isfinite(sim)
    np.add.at(sums, pair_i[valid], sim[valid])
    np.add.at(sums, pair_j[valid], sim[valid])
    np.add.at(counts, pair_i[valid], 1)
    np.add.at(counts, pair_j[valid], 1)
    centrality = np.divide(sums, counts, out=np.full(n_items, np.nan), where=counts > 0)
    roi = npz_path.parent.name
    subject = npz_path.parent.parent.name
    phase = npz_path.name.split("_all_rdms.npz")[0]
    network = roi_to_network(roi)
    rows = meta.copy()
    rows["subject"] = subject
    rows["roi"] = roi
    rows["network"] = network
    rows["phase"] = phase
    rows["global_similarity"] = centrality
    rows["condition_item_id"] = [condition_item_id(c, p) for c, p in zip(rows["condition"], rows["pair_id"])]
    log["n_rows"] = len(rows)
    log["n_valid_items"] = int(np.isfinite(centrality).sum())
    return rows[["subject", "roi", "network", "phase", "condition", "condition_item_id", "global_similarity"]], log


def build_global_table(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    logs = []
    for path in paths:
        frame, log = extract_one(path)
        logs.append(log)
        if not frame.empty:
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame(logs)
    raw = raw[raw["network"].notna() & raw["condition_item_id"].astype(str).ne("")]
    # Aggregate ROI-level centrality to the network mean (raw units) per
    # subject x network x condition_item_id x phase.
    item = raw.groupby(["subject", "network", "phase", "condition", "condition_item_id"], dropna=False, as_index=False).agg(
        global_similarity=("global_similarity", "mean"),
        n_roi=("roi", "nunique"),
    )
    wide = item.pivot_table(index=["subject", "network", "condition", "condition_item_id"], columns="phase", values="global_similarity", aggfunc="mean").reset_index()
    wide.columns = [str(c).lower() for c in wide.columns]
    if {"pre", "post"}.issubset(wide.columns):
        wide["global_similarity_change"] = wide["post"] - wide["pre"]
        # Single global z-score; keeps condition means in their natural scale.
        wide["global_similarity_change_z"] = zscore(wide["global_similarity_change"])
    return wide, pd.DataFrame(logs)


def prepare_item_bridge(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_network_column(build_condition_item_id(frame))
    for col in ["memory", "memory_prop", "trained_edge_drop", "post_edge_specificity", *COVARIATES]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "memory_prop" not in out.columns and "memory" in out.columns:
        out["memory_prop"] = out["memory"]
    if "network" not in out.columns or "condition_item_id" not in out.columns:
        return out
    out = out[out["network"].notna() & out["condition_item_id"].astype(str).ne("")].copy()
    id_keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    metric_col = "post_edge_specificity" if "post_edge_specificity" in out.columns else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    if metric_col is None:
        return out
    value_cols = [metric_col]
    extra_cols = [c for c in ["memory_prop", *COVARIATES] if c in out.columns]
    agg_cols = list(dict.fromkeys(value_cols + extra_cols))
    network_raw = out.groupby(id_keys, dropna=False, as_index=False)[agg_cols].mean()
    network_raw = network_raw.rename(columns={metric_col: "post_edge_differentiation_raw"})
    # Single global z-score across the network-item table instead of within
    # each network, so condition main effects are not forced into mean-zero
    # within-network space.
    network_raw["post_edge_differentiation_z"] = zscore(network_raw["post_edge_differentiation_raw"])
    keep = [c for c in ["subject", "network", "condition", "condition_item_id", "memory_prop",
                        "post_edge_differentiation_raw", "post_edge_differentiation_z", *COVARIATES] if c in network_raw.columns]
    return network_raw[keep].drop_duplicates()


def usable_covariates(frame: pd.DataFrame) -> list[str]:
    return [c for c in COVARIATES if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().sum() >= 10]


def fit_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network, sub in data.groupby("network", dropna=False):
        covs = usable_covariates(sub)
        if {"global_similarity_change_z", "memory_prop"}.issubset(sub.columns):
            model_data = sub.dropna(subset=["global_similarity_change_z", "memory_prop", "condition", *covs])
            if len(model_data) >= 30:
                formula = "global_similarity_change_z ~ condition * memory_prop"
                if covs:
                    formula += " + " + " + ".join(covs)
                res = fit_formula(model_data, formula, family="gaussian")
                res.insert(0, "network", network)
                res.insert(1, "mechanism_model", "global_change_by_memory")
                rows.append(res)
        if {"post_edge_differentiation_z", "global_similarity_change_z"}.issubset(sub.columns):
            model_data = sub.dropna(subset=["post_edge_differentiation_z", "global_similarity_change_z", "condition", *covs])
            if len(model_data) >= 30:
                formula = "post_edge_differentiation_z ~ global_similarity_change_z * condition"
                if covs:
                    formula += " + " + " + ".join(covs)
                res = fit_formula(model_data, formula, family="gaussian")
                res.insert(0, "network", network)
                res.insert(1, "mechanism_model", "edge_differentiation_by_global_change")
                rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame([{"status": "no_models_fit"}])


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.rdm_input or []
    if not paths:
        for pattern in RDM_PATTERNS:
            paths.extend(cfg.paper_output_root.glob(pattern))
    paths = sorted(set(paths))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    global_table, manifest = build_global_table(paths)
    item_path = Path(args.item_input or cfg.paper_output_root / ITEM_INPUT)
    bridge = prepare_item_bridge(read_table(item_path))
    merged = global_table.merge(bridge, on=["subject", "network", "condition", "condition_item_id"], how="left") if not global_table.empty else pd.DataFrame()
    models = fit_models(merged) if not merged.empty else pd.DataFrame([{"status": "empty_global_table"}])
    write_outputs(cfg, MODULE, {
        "global_similarity_item.tsv": merged,
        "global_similarity_models.tsv": models,
        "global_similarity_manifest.tsv": manifest,
    })


if __name__ == "__main__":
    main()
