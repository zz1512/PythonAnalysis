#!/usr/bin/env python3
"""B2: compare Step5C correlation estimates with crossnobis/Mahalanobis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from scipy import stats
import numpy as np
import pandas as pd

from shared_nc import add_common_args, bh_fdr, default_config, discover_tables, first_existing_column, read_table, roi_to_network, write_outputs, q_for_ok_rows, zscore

MODULE = "b2_crossnobis_robustness"
CORR_PATTERNS = ["**/*step5c*.tsv", "**/*edge_specificity*.tsv", "**/network_composite_summary.tsv"]
CROSS_PATTERNS = ["**/b2a_mahalanobis_from_patterns/mahalanobis_step5c.tsv", "**/b2a_mahalanobis_from_patterns/mahalanobis_subject_delta.tsv", "**/*crossnobis*.tsv", "**/*mahalanobis*.tsv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--correlation-input", nargs="*", type=Path, default=None)
    parser.add_argument("--crossnobis-input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def load(paths: list[Path], label: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        frame["source_path"] = str(path)
        frame["source_metric_family"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def standardize(frame: pd.DataFrame, estimate_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    est = first_existing_column(out, ["estimate", "mean", "effect", "metric_value", "value", "beta"])
    if est:
        out[estimate_name] = pd.to_numeric(out[est], errors="coerce")
    keys = [c for c in ["network", "roi", "condition", "stage", "metric", "term"] if c in out.columns]
    if not keys:
        out["merge_key"] = out.index.astype(str)
    else:
        out["merge_key"] = out[keys].apply(lambda row: "|".join(str(value) for value in row.tolist()), axis=1)
    return out


def load_corr_delta(cfg) -> pd.DataFrame:
    item_path = cfg.paper_output_root / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv"
    if item_path.exists():
        frame = read_table(item_path)
        needed = {"subject", "roi", "condition", "condition_item_id"}
        value_col = first_existing_column(frame, ["post_minus_pre_pair_similarity", "delta_post_minus_pre"])
        if needed.issubset(frame.columns) and value_col:
            out = frame.copy()
            out["condition"] = out["condition"].map({"yy": "Metaphor", "kj": "Spatial"}).fillna(out["condition"])
            if "network" not in out.columns:
                roi_set = out["roi_set"] if "roi_set" in out.columns else pd.Series([None] * len(out), index=out.index)
                out["network"] = [roi_to_network(r, s) for r, s in zip(out["roi"], roi_set)]
            out["corr_delta_post_minus_pre"] = pd.to_numeric(out[value_col], errors="coerce")
            keep = [c for c in ["subject", "roi_set", "roi", "network", "condition", "condition_item_id", "corr_delta_post_minus_pre"] if c in out.columns]
            raw = out[keep].dropna(subset=["corr_delta_post_minus_pre"])
            if {"subject", "roi", "network", "condition", "condition_item_id"}.issubset(raw.columns):
                comp = raw.copy()
                comp["corr_delta_z"] = comp.groupby(["subject", "roi"], dropna=False)["corr_delta_post_minus_pre"].transform(zscore)
                comp = (
                    comp.groupby(["subject", "network", "condition", "condition_item_id"], dropna=False, as_index=False)
                    .agg(corr_delta_post_minus_pre=("corr_delta_z", "mean"))
                )
                comp["roi"] = "__network_composite__"
                comp["roi_set"] = "network_composite"
                raw = pd.concat([raw, comp], ignore_index=True, sort=False)
            return raw
    path = cfg.paper_output_root / "figures_main" / "table_step5c_roi_delta.tsv"
    if not path.exists():
        return pd.DataFrame()
    frame = read_table(path)
    needed = {"subject", "roi", "condition", "delta_post_minus_pre"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["corr_delta_post_minus_pre"] = pd.to_numeric(out["delta_post_minus_pre"], errors="coerce")
    return out[["subject", "roi", "condition", "corr_delta_post_minus_pre"]].dropna(subset=["corr_delta_post_minus_pre"])


def load_cross_delta(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.name != "mahalanobis_subject_delta.tsv":
            continue
        try:
            frame = read_table(path)
        except Exception:
            continue
        needed = {"subject", "roi", "condition", "delta_post_minus_pre"}
        if not needed.issubset(frame.columns):
            continue
        out = frame.copy()
        out["cross_delta_post_minus_pre"] = pd.to_numeric(out["delta_post_minus_pre"], errors="coerce")
        out["source_path"] = str(path)
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def summarize_delta_agreement(corr_delta: pd.DataFrame, cross_delta: pd.DataFrame) -> pd.DataFrame:
    if corr_delta.empty or cross_delta.empty:
        return pd.DataFrame()
    merge_cols = ["subject", "roi", "condition"]
    if "condition_item_id" in corr_delta.columns and "condition_item_id" in cross_delta.columns:
        merge_cols.append("condition_item_id")
    for optional_key in ["network", "roi_set"]:
        if optional_key in corr_delta.columns and optional_key in cross_delta.columns:
            merge_cols.append(optional_key)
    cross_keep = [
        c
        for c in [
            "subject",
            "roi",
            "condition",
            "condition_item_id",
            "network",
            "roi_set",
            "metric",
            "estimator",
            "cross_delta_post_minus_pre",
            "source_path",
        ]
        if c in cross_delta.columns
    ]
    merged = corr_delta.merge(cross_delta[cross_keep], on=merge_cols, how="inner")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["network", "roi_set", "roi", "condition", "metric", "estimator"] if c in merged.columns]
    for key, sub in merged.groupby(group_cols, dropna=False):
        corr_values = pd.to_numeric(sub["corr_delta_post_minus_pre"], errors="coerce")
        cross_values = pd.to_numeric(sub["cross_delta_post_minus_pre"], errors="coerce")
        valid = corr_values.notna() & cross_values.notna()
        corr_values = corr_values[valid]
        cross_values = cross_values[valid]
        if len(corr_values) < 2:
            r_value = p_spearman = t_value = p_cross = np.nan
            status = "too_few_rows"
        else:
            r_value, p_spearman = stats.spearmanr(corr_values, cross_values, nan_policy="omit")
            t_value, p_cross = stats.ttest_1samp(cross_values, 0.0, nan_policy="omit")
            status = "ok"
        sign_same = bool(np.sign(corr_values.mean()) == np.sign(cross_values.mean())) if len(corr_values) else False
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "term": "post_minus_pre_delta_agreement",
                "estimate": float(cross_values.mean()) if len(cross_values) else np.nan,
                "beta_corr": float(corr_values.mean()) if len(corr_values) else np.nan,
                "beta_crossnobis": float(cross_values.mean()) if len(cross_values) else np.nan,
                "p": p_cross,
                "spearman_r": r_value,
                "spearman_p": p_spearman,
                "n_obs": int(len(cross_values)),
                "n_subjects": int(sub.loc[valid, "subject"].nunique()) if "subject" in sub.columns else int(len(cross_values)),
                "same_direction": sign_same,
                "direction_consistent": sign_same and status == "ok",
                "status": "ok" if sign_same and status == "ok" else ("direction_mismatch" if status == "ok" else status),
                "comparison_family": "correlation_vs_mahalanobis_subject_delta",
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out["status"].eq("ok")
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    corr_paths = args.correlation_input or discover_tables(cfg.output_root, CORR_PATTERNS) or discover_tables(cfg.paper_output_root, CORR_PATTERNS)
    cross_paths = args.crossnobis_input or discover_tables(cfg.paper_output_root, CROSS_PATTERNS) or discover_tables(cfg.output_root, CROSS_PATTERNS)
    delta_agreement = summarize_delta_agreement(load_corr_delta(cfg), load_cross_delta(cross_paths))
    if not delta_agreement.empty:
        write_outputs(
            cfg,
            MODULE,
            {
                "crossnobis_step5c.tsv": delta_agreement,
                "crossnobis_input_manifest.tsv": pd.DataFrame({"path": [str(p) for p in corr_paths + cross_paths]}),
            },
        )
        return
    corr = standardize(load(corr_paths, "correlation"), "beta_corr")
    cross = standardize(load(cross_paths, "crossnobis"), "beta_crossnobis")
    if corr.empty and cross.empty and not args.allow_empty:
        raise FileNotFoundError("No correlation or crossnobis inputs found; pass explicit inputs or --allow-empty.")
    if corr.empty or cross.empty:
        table = pd.DataFrame([{
            "status": "missing_crossnobis_input" if cross.empty else "missing_correlation_input",
            "n_corr_tables": len(corr_paths),
            "n_crossnobis_tables": len(cross_paths),
        }])
    else:
        keep_corr = ["merge_key", "beta_corr"] + [c for c in ["network", "roi", "condition", "stage", "metric", "term", "p"] if c in corr.columns]
        keep_cross = ["merge_key", "beta_crossnobis"] + [c for c in ["p"] if c in cross.columns]
        table = corr[keep_corr].merge(cross[keep_cross], on="merge_key", how="outer", suffixes=("_corr", "_crossnobis"))
        table["same_direction"] = np.sign(table["beta_corr"]) == np.sign(table["beta_crossnobis"])
        table["relative_delta"] = (table["beta_corr"] - table["beta_crossnobis"]).abs() / table["beta_corr"].abs().replace(0, np.nan)
        table["direction_consistent"] = table["same_direction"] & (table["relative_delta"] < 0.5)
        table["status"] = np.where(table["direction_consistent"], "ok", "direction_or_magnitude_mismatch")
        if "p_corr" in table.columns:
            table = table.rename(columns={"p_corr": "p"})
        table = q_for_ok_rows(table)
    write_outputs(cfg, MODULE, {"crossnobis_step5c.tsv": table, "crossnobis_input_manifest.tsv": pd.DataFrame({"path": [str(p) for p in corr_paths + cross_paths]})})


if __name__ == "__main__":
    main()
