#!/usr/bin/env python3
"""Stage 3: semantic-to-edge model shift from pre/post word-level RDMs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


META_METAPHOR_ROIS = {
    "meta_L_IFG",
    "meta_R_IFG",
    "meta_L_temporal_pole",
    "meta_R_temporal_pole",
    "meta_L_AG",
    "meta_R_AG",
    "meta_L_pMTG_pSTS",
    "meta_R_pMTG_pSTS",
}
META_SPATIAL_ROIS = {
    "meta_L_hippocampus",
    "meta_R_hippocampus",
    "meta_L_PPA_PHG",
    "meta_R_PPA_PHG",
    "meta_L_RSC_PCC",
    "meta_R_RSC_PCC",
    "meta_L_PPC_SPL",
    "meta_R_PPC_SPL",
    "meta_L_precuneus",
    "meta_R_precuneus",
}


def _default_qc_root() -> Path:
    return Path("E:/python_metaphor/paper_outputs/qc")


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = numeric.notna() & np.isfinite(numeric)
    if ok.any():
        _, qvals, _, _ = multipletests(numeric.loc[ok], method="fdr_bh")
        out.loc[numeric.loc[ok].index] = qvals
    return out


def _rank_z(values: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(values, method="average")
    sd = ranked.std(ddof=1)
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        return np.full_like(ranked, np.nan, dtype=float)
    return (ranked - ranked.mean()) / sd


def _ols_r2(y: np.ndarray, columns: list[np.ndarray]) -> float:
    if not columns:
        return 0.0
    x = np.column_stack(columns)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if valid.sum() < x.shape[1] + 5:
        return np.nan
    yv = y[valid]
    xv = x[valid]
    keep = [idx for idx in range(xv.shape[1]) if np.nanstd(xv[:, idx], ddof=1) > 0]
    if not keep:
        return 0.0
    xv = xv[:, keep]
    design = np.column_stack([np.ones(len(yv)), xv])
    try:
        beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    pred = design @ beta
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    if np.isclose(ss_tot, 0.0):
        return np.nan
    r2 = 1.0 - ss_res / ss_tot
    return float(max(min(r2, 1.0), 0.0))


def _vif(target: np.ndarray, others: list[np.ndarray]) -> float:
    if not others:
        return np.nan
    r2 = _ols_r2(target, others)
    if not np.isfinite(r2):
        return np.nan
    if r2 >= 0.999999:
        return np.inf
    return float(1.0 / (1.0 - r2))


def _condition_number(columns: list[np.ndarray]) -> float:
    x = np.column_stack(columns)
    valid = np.all(np.isfinite(x), axis=1)
    if valid.sum() < x.shape[1] + 5:
        return np.nan
    xv = x[valid]
    keep = [idx for idx in range(xv.shape[1]) if np.nanstd(xv[:, idx], ddof=1) > 0]
    if not keep:
        return np.nan
    try:
        return float(np.linalg.cond(xv[:, keep]))
    except np.linalg.LinAlgError:
        return np.nan


def _roi_set(roi: str) -> str:
    if roi in META_METAPHOR_ROIS:
        return "meta_metaphor"
    if roi in META_SPATIAL_ROIS:
        return "meta_spatial"
    return "unknown"


def _network(roi: str) -> str:
    if roi in META_METAPHOR_ROIS:
        return "semantic"
    if roi in META_SPATIAL_ROIS:
        return "hpc_spatial"
    return "unknown"


def _iter_npz(audit_roots: list[Path], *, condition_group: str | None = None) -> list[Path]:
    paths: list[Path] = []
    pattern = "*_rdms.npz" if condition_group is None else f"*_{condition_group}_rdms.npz"
    for root in audit_roots:
        if root.exists():
            paths.extend(sorted(root.glob(f"sub-*/*/{pattern}")))
    return paths


def _cell_from_path(path: Path) -> dict[str, str]:
    subject = path.parent.parent.name
    roi = path.parent.name
    stem = path.name.replace("_rdms.npz", "")
    parts = stem.split("_", 1)
    time = parts[0]
    condition_group = parts[1] if len(parts) > 1 else "unknown"
    return {
        "subject": subject,
        "roi": roi,
        "roi_set": _roi_set(roi),
        "network": _network(roi),
        "time": time,
        "condition_group": condition_group,
    }


def _load_cell(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    meta = _cell_from_path(path)
    z = np.load(path, allow_pickle=False)
    is_all = meta["condition_group"] == "all"
    required = ["neural_rdm", "M3_embedding", "M8_reverse_pair"]
    if is_all:
        required.append("M1_condition")
    if any(key not in z.files for key in required):
        row = {
            **meta,
            "status": "missing_required_model",
            "message": ",".join([key for key in required if key not in z.files]),
        }
        return row, []

    raw = {key: np.asarray(z[key], dtype=float) for key in required}
    valid = np.ones_like(raw["neural_rdm"], dtype=bool)
    for vec in raw.values():
        valid &= np.isfinite(vec)
    n_valid = int(valid.sum())
    if n_valid < 20:
        row = {**meta, "status": "too_few_valid_pairs", "message": "", "n_pairs": n_valid}
        return row, []

    ranked = {key: _rank_z(vec[valid]) for key, vec in raw.items()}
    if any(np.isnan(vec).all() for vec in ranked.values()):
        row = {**meta, "status": "constant_ranked_vector", "message": "", "n_pairs": n_valid}
        return row, []

    y = ranked["neural_rdm"]
    emb = ranked["M3_embedding"]
    edge = ranked["M8_reverse_pair"]
    cond = ranked["M1_condition"] if "M1_condition" in ranked else None

    r2_embedding = _ols_r2(y, [emb])
    r2_edge = _ols_r2(y, [edge])
    r2_condition = _ols_r2(y, [cond]) if cond is not None else np.nan
    controls = [emb, edge, cond] if cond is not None else [emb, edge]
    r2_full = _ols_r2(y, controls)
    r2_no_embedding = _ols_r2(y, [vec for vec in [edge, cond] if vec is not None])
    r2_no_edge = _ols_r2(y, [vec for vec in [emb, cond] if vec is not None])
    r2_no_condition = _ols_r2(y, [emb, edge]) if cond is not None else np.nan
    unique_embedding = r2_full - r2_no_embedding if np.isfinite(r2_full) and np.isfinite(r2_no_embedding) else np.nan
    unique_edge = r2_full - r2_no_edge if np.isfinite(r2_full) and np.isfinite(r2_no_edge) else np.nan
    unique_condition = r2_full - r2_no_condition if np.isfinite(r2_full) and np.isfinite(r2_no_condition) else np.nan

    model_vecs = {
        "embedding": emb,
        "edge": edge,
    }
    if cond is not None:
        model_vecs["condition"] = cond
    condition_columns = list(model_vecs.values())
    coll_rows = []
    for a_name, a_vec in model_vecs.items():
        for b_name, b_vec in model_vecs.items():
            if a_name >= b_name:
                continue
            rho, p = stats.spearmanr(a_vec, b_vec)
            coll_rows.append(
                {
                    **meta,
                    "model_a": a_name,
                    "model_b": b_name,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "abs_rho": float(abs(rho)),
                    "flag_high_collinearity": bool(abs(rho) > 0.7),
                    "condition_number": _condition_number(condition_columns),
                }
            )
    for name, vec in model_vecs.items():
        others = [other for other_name, other in model_vecs.items() if other_name != name]
        coll_rows.append(
            {
                **meta,
                "model_a": name,
                "model_b": "__vif__",
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "abs_rho": np.nan,
                "flag_high_collinearity": False,
                "vif": _vif(vec, others),
                "condition_number": _condition_number(condition_columns),
            }
        )

    row = {
        **meta,
        "status": "ok",
        "message": "",
        "n_pairs": n_valid,
        "r2_embedding_only": r2_embedding,
        "r2_edge_only": r2_edge,
        "r2_condition_only": r2_condition,
        "r2_full": r2_full,
        "r2_no_embedding": r2_no_embedding,
        "r2_no_edge": r2_no_edge,
        "r2_no_condition": r2_no_condition,
        "unique_embedding_r2": unique_embedding,
        "unique_edge_r2": unique_edge,
        "unique_condition_r2": unique_condition,
        "edge_minus_embedding_unique_r2": unique_edge - unique_embedding,
    }
    return row, coll_rows


def _build_shift_table(cell: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "r2_embedding_only",
        "r2_edge_only",
        "r2_condition_only",
        "r2_full",
        "unique_embedding_r2",
        "unique_edge_r2",
        "unique_condition_r2",
        "edge_minus_embedding_unique_r2",
    ]
    idx_cols = ["subject", "roi_set", "roi", "network", "condition_group"]
    wide = cell[cell["status"].eq("ok")].pivot_table(index=idx_cols, columns="time", values=metric_cols, aggfunc="mean")
    rows = []
    for index, vals in wide.iterrows():
        record = dict(zip(idx_cols, index))
        for metric in metric_cols:
            pre = vals.get((metric, "pre"), np.nan)
            post = vals.get((metric, "post"), np.nan)
            record[f"pre_{metric}"] = pre
            record[f"post_{metric}"] = post
            record[f"delta_{metric}"] = post - pre if pd.notna(pre) and pd.notna(post) else np.nan
        record["semantic_to_edge_shift"] = (
            record["delta_unique_edge_r2"] - record["delta_unique_embedding_r2"]
            if pd.notna(record.get("delta_unique_edge_r2")) and pd.notna(record.get("delta_unique_embedding_r2"))
            else np.nan
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _make_network_shift(shift: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in shift.columns if c.startswith(("pre_", "post_", "delta_")) or c == "semantic_to_edge_shift"]
    key = ["subject", "network", "condition_group"]
    out = shift[shift["network"].isin(["semantic", "hpc_spatial"])].groupby(key, as_index=False, observed=True)[metric_cols].mean()
    out["roi_set"] = "network"
    out["roi"] = out["network"]
    return out[["subject", "roi_set", "roi", "network", "condition_group", *metric_cols]]


def _one_sample_models(shift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shift = shift[shift["condition_group"].eq("all")].copy()
    metrics = [
        "delta_unique_embedding_r2",
        "delta_unique_edge_r2",
        "semantic_to_edge_shift",
        "delta_edge_minus_embedding_unique_r2",
    ]
    for level_name, frame in [("roi", shift[shift["roi_set"].ne("network")]), ("network", shift[shift["roi_set"].eq("network")])]:
        for metric in metrics:
            for (roi_set, roi, network), sub in frame.groupby(["roi_set", "roi", "network"], observed=True, sort=False):
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) >= 3:
                    test = stats.ttest_1samp(vals, 0.0)
                    estimate = float(vals.mean())
                    se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
                    stat = float(test.statistic)
                    p = float(test.pvalue)
                    status = "ok"
                else:
                    estimate = se = stat = p = np.nan
                    status = "skipped_too_few_subjects"
                rows.append(
                    {
                        "level": level_name,
                        "metric": metric,
                        "roi_set": roi_set,
                        "roi": roi,
                        "network": network,
                        "term": "intercept_gt_zero",
                        "estimate": estimate,
                        "se": se,
                        "stat": stat,
                        "p": p,
                        "n_subjects": int(len(vals)),
                        "status": status,
                    }
                )
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    ok = out["status"].eq("ok") & out["p"].notna()
    for _, idx in out[ok].groupby(["level", "metric", "roi_set"], dropna=False, observed=True).groups.items():
        out.loc[idx, "q"] = _bh(out.loc[idx, "p"])
    return out


def _condition_contrast_models(shift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "delta_unique_embedding_r2",
        "delta_unique_edge_r2",
        "semantic_to_edge_shift",
        "delta_edge_minus_embedding_unique_r2",
    ]
    condition_shift = shift[shift["condition_group"].isin(["yy", "kj"])].copy()
    for level_name, frame in [
        ("roi", condition_shift[condition_shift["roi_set"].ne("network")]),
        ("network", condition_shift[condition_shift["roi_set"].eq("network")]),
    ]:
        for metric in metrics:
            for (roi_set, roi, network), sub in frame.groupby(["roi_set", "roi", "network"], observed=True, sort=False):
                wide = sub.pivot_table(index="subject", columns="condition_group", values=metric, aggfunc="mean")
                if {"yy", "kj"}.issubset(wide.columns):
                    diff = (wide["yy"] - wide["kj"]).dropna().to_numpy(dtype=float)
                else:
                    diff = np.asarray([], dtype=float)
                if len(diff) >= 3:
                    test = stats.ttest_1samp(diff, 0.0)
                    estimate = float(diff.mean())
                    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
                    stat = float(test.statistic)
                    p = float(test.pvalue)
                    status = "ok"
                else:
                    estimate = se = stat = p = np.nan
                    status = "skipped_too_few_subjects"
                rows.append(
                    {
                        "level": level_name,
                        "metric": metric,
                        "roi_set": roi_set,
                        "roi": roi,
                        "network": network,
                        "term": "yy_minus_kj",
                        "estimate": estimate,
                        "se": se,
                        "stat": stat,
                        "p": p,
                        "n_subjects": int(len(diff)),
                        "status": status,
                    }
                )
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    ok = out["status"].eq("ok") & out["p"].notna()
    for _, idx in out[ok].groupby(["level", "metric", "roi_set"], dropna=False, observed=True).groups.items():
        out.loc[idx, "q"] = _bh(out.loc[idx, "p"])
    return out


def _md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return ""
    work = df[[c for c in cols if c in df.columns]].head(n).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            work[col] = work[col].astype("object").where(work[col].notna(), "").astype(str)
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in work.columns) + " |" for _, row in work.iterrows()]
    return "\n".join([header, sep, *body])


def _write_review(out_dir: Path, cell: pd.DataFrame, coll: pd.DataFrame, models: pd.DataFrame, condition_models: pd.DataFrame) -> None:
    network_focus = models[
        models["level"].eq("network")
        & models["metric"].isin(["delta_unique_edge_r2", "delta_unique_embedding_r2", "semantic_to_edge_shift"])
    ].sort_values(["metric", "q"], na_position="last")
    roi_focus = models[
        models["level"].eq("roi")
        & models["metric"].isin(["semantic_to_edge_shift", "delta_unique_edge_r2"])
    ].sort_values(["metric", "q"], na_position="last")
    high_coll = coll[coll.get("flag_high_collinearity", False).eq(True)] if not coll.empty else coll
    condition_network = condition_models[
        condition_models["level"].eq("network")
        & condition_models["metric"].isin(["delta_unique_edge_r2", "delta_unique_embedding_r2", "semantic_to_edge_shift"])
    ].sort_values(["metric", "q"], na_position="last")
    condition_roi = condition_models[
        condition_models["level"].eq("roi")
        & condition_models["metric"].isin(["semantic_to_edge_shift", "delta_unique_edge_r2"])
    ].sort_values(["metric", "q"], na_position="last")
    qc = pd.DataFrame(
        [
            {
                "table": "stage3_cell_rdm",
                "n_rows": len(cell),
                "n_ok": int(cell["status"].eq("ok").sum()),
                "n_subjects": int(cell["subject"].nunique()),
                "n_rois": int(cell["roi"].nunique()),
                "condition_groups": ",".join(sorted(cell["condition_group"].dropna().astype(str).unique())),
            },
            {
                "table": "stage3_collinearity",
                "n_rows": len(coll),
                "n_high_abs_rho_gt_0.7": int(high_coll.shape[0]) if not coll.empty else 0,
                "n_subjects": int(coll["subject"].nunique()) if not coll.empty else 0,
                "n_rois": int(coll["roi"].nunique()) if not coll.empty else 0,
            },
        ]
    )
    text = f"""# Stage 3 Semantic-to-Edge Model Shift Review

## Design

Stage 3 reuses existing all-condition pre/post word-level model-RSA audit vectors.
For each subject x ROI x time cell, rank-transformed neural RDMs are regressed on:

```text
M3_embedding + M8_reverse_pair + M1_condition
```

For condition-specific yy/kj RDMs, M1_condition is unavailable by design, so the condition-specific model is:

```text
M3_embedding + M8_reverse_pair
```

Unique R2 is computed by reduced-model subtraction. The primary index is:

```text
semantic_to_edge_shift =
  delta_unique_edge_r2 - delta_unique_embedding_r2
```

The all-condition analysis tests global RDM shift. The condition-specific analysis tests whether this shift is stronger in YY than KJ.

## QC

{_md_table(qc, list(qc.columns), 10)}

## Collinearity Check

High model collinearity cells (|rho| > 0.7): {int(high_coll.shape[0]) if not coll.empty else 0}.

## Global All-Condition Network-Level Tests

{_md_table(network_focus, ["metric", "roi", "estimate", "se", "p", "q", "n_subjects", "status"], 30)}

## Global All-Condition ROI-Level Top Tests

{_md_table(roi_focus, ["metric", "roi_set", "roi", "estimate", "se", "p", "q", "n_subjects", "status"], 40)}

## YY-KJ Condition-Specific Network Tests

{_md_table(condition_network, ["metric", "roi", "estimate", "se", "p", "q", "n_subjects", "status"], 30)}

## YY-KJ Condition-Specific ROI Top Tests

{_md_table(condition_roi, ["metric", "roi_set", "roi", "estimate", "se", "p", "q", "n_subjects", "status"], 40)}

## Interpretation

- Positive semantic_to_edge_shift would mean unique edge-model fit increased from pre to post more than unique embedding-model fit.
- If edge unique fit does not increase after controlling embedding and condition, post edge specificity is better described by direct pair-similarity contrasts than by a global RDM model shift.
- YY-KJ condition-specific tests ask whether the semantic-to-edge shift is stronger for YY than KJ.
"""
    (out_dir / "stage3_semantic_edge_shift_review.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    qc_root = _default_qc_root()
    parser.add_argument(
        "--audit-roots",
        nargs="*",
        type=Path,
        default=[
            qc_root / "model_rdm_results_meta_metaphor" / "model_rdm_audit",
            qc_root / "model_rdm_results_meta_spatial" / "model_rdm_audit",
        ],
    )
    parser.add_argument(
        "--condition-audit-roots",
        nargs="*",
        type=Path,
        default=[
            qc_root / "stagewise_mechanism" / "stage3_model_rdm_by_condition_meta_metaphor" / "model_rdm_audit",
            qc_root / "stagewise_mechanism" / "stage3_model_rdm_by_condition_meta_spatial" / "model_rdm_audit",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=qc_root / "stagewise_mechanism")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_npz(args.audit_roots, condition_group="all")
    condition_paths = _iter_npz(args.condition_audit_roots)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "n_all_audit_npz": len(paths),
                    "n_condition_audit_npz": len(condition_paths),
                    "audit_roots": [str(p) for p in args.audit_roots],
                    "condition_audit_roots": [str(p) for p in args.condition_audit_roots],
                },
                indent=2,
            )
        )
        return

    cell_rows = []
    coll_rows = []
    for path in [*paths, *condition_paths]:
        row, coll = _load_cell(path)
        cell_rows.append(row)
        coll_rows.extend(coll)
    cell = pd.DataFrame(cell_rows)
    coll = pd.DataFrame(coll_rows)
    roi_shift = _build_shift_table(cell)
    network_shift = _make_network_shift(roi_shift)
    shift = pd.concat([roi_shift, network_shift], ignore_index=True, sort=False)
    models = _one_sample_models(shift)
    condition_models = _condition_contrast_models(shift)

    cell.to_csv(args.out_dir / "stage3_semantic_edge_rdm_table.tsv", sep="\t", index=False)
    coll.to_csv(args.out_dir / "stage3_semantic_edge_model_collinearity.tsv", sep="\t", index=False)
    shift.to_csv(args.out_dir / "stage3_semantic_edge_shift.tsv", sep="\t", index=False)
    models.to_csv(args.out_dir / "stage3_semantic_edge_shift_models.tsv", sep="\t", index=False)
    condition_models.to_csv(args.out_dir / "stage3_semantic_edge_condition_models.tsv", sep="\t", index=False)
    _write_review(args.out_dir, cell, coll, models, condition_models)
    manifest = {
        "analysis_id": "stage3_semantic_edge_model_shift",
        "n_all_audit_npz": len(paths),
        "n_condition_audit_npz": len(condition_paths),
        "n_cell_rows": int(len(cell)),
        "n_shift_rows": int(len(shift)),
        "n_model_rows": int(len(models)),
        "n_condition_model_rows": int(len(condition_models)),
        "output_dir": str(args.out_dir),
        "model_mapping": {
            "embedding_model": "M3_embedding",
            "edge_model": "M8_reverse_pair",
            "condition_model": "M1_condition",
        },
        "scope": "condition_group=all global RDM shift plus yy-vs-kj condition-specific shift",
    }
    (args.out_dir / "stage3_semantic_edge_shift_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
