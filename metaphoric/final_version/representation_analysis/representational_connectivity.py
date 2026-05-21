"""
representational_connectivity.py

用途
- 做 ROI-to-ROI representational connectivity，优先服务主故事，而不是直接走 gPPI。
- 当前定义：两个 ROI 在同一批 item 上的神经变化轮廓（post - pre 的 item-wise delta similarity）有多一致。
- 主分析默认使用：
  - main_functional（Run3/4 定义 ROI，但结果基于 pre/post，所以可作为主分析）
  - 且必须按 family-aware 方式汇总：
    - within_Metaphor_gt_Spatial
    - within_Spatial_gt_Metaphor
    - cross_main_functional_family
- 确认性分析默认使用：
  - literature + literature_spatial 的独立 ROI 池
  - 汇总为：
    - within_literature
    - within_literature_spatial
    - cross_independent_sets

主指标
- repr_connectivity_rho:
  corr(delta_similarity_source, delta_similarity_target) across shared items
- repr_connectivity_z:
  fisher-z transformed rho，用于组水平统计

输出（paper_outputs）
- tables_main/table_repr_connectivity.tsv
- figures_main/fig_repr_connectivity.png
- tables_si/table_repr_connectivity_subject.tsv
- tables_si/table_repr_connectivity_edges.tsv
- qc/repr_connectivity_summary_group.tsv
- qc/repr_connectivity_edge_group.tsv
- qc/repr_connectivity_meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import (  # noqa: E402
    ensure_dir,
    one_sample_t_summary,
    paired_t_summary,
    percentile_bootstrap_ci,
    save_json,
    write_table,
)


CONDITION_MAP = {
    "metaphor": "yy",
    "yy": "yy",
    "spatial": "kj",
    "kj": "kj",
    "baseline": "baseline",
}


def _default_base_dir() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402
    return Path(BASE_DIR)


def _default_rsa_files(base_dir: Path) -> list[Path]:
    return [
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_main_functional" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature_spatial" / "rsa_itemwise_details.csv",
    ]


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    sep = "\t" if suffixes.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _canonical_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "time" not in out.columns and "stage" in out.columns:
        out["time"] = out["stage"]
    out["time"] = out["time"].astype(str).str.strip().str.lower().map(
        {"pre": "pre", "post": "post", "1": "pre", "2": "post"}
    )
    return out


def _infer_roi_set(path: Path) -> str:
    text = path.parent.name.lower()
    prefix = "rsa_results_optimized_"
    if prefix in text:
        return text.split(prefix, 1)[1]
    return path.parent.name


def _infer_roi_pool(roi_set: str) -> str:
    roi_set = str(roi_set).strip().lower()
    if roi_set == "main_functional":
        return "main_functional"
    if roi_set in {"literature", "literature_spatial"}:
        return "independent"
    return roi_set


def _infer_roi_family(roi_name: str, roi_set: str) -> str:
    roi_set_norm = str(roi_set).strip().lower()
    roi_name_norm = str(roi_name)
    if roi_set_norm == "main_functional":
        if "Metaphor_gt_Spatial" in roi_name_norm:
            return "Metaphor_gt_Spatial"
        if "Spatial_gt_Metaphor" in roi_name_norm:
            return "Spatial_gt_Metaphor"
        return "main_functional_other"
    if roi_set_norm == "literature":
        return "literature"
    if roi_set_norm == "literature_spatial":
        return "literature_spatial"
    return roi_set_norm or "unknown"


def _pair_scope(roi_pool: str, family_a: str, family_b: str) -> str:
    if roi_pool == "main_functional":
        if family_a == family_b == "Metaphor_gt_Spatial":
            return "within_Metaphor_gt_Spatial"
        if family_a == family_b == "Spatial_gt_Metaphor":
            return "within_Spatial_gt_Metaphor"
        return "cross_main_functional_family"
    if roi_pool == "independent":
        if family_a == family_b == "literature":
            return "within_literature"
        if family_a == family_b == "literature_spatial":
            return "within_literature_spatial"
        return "cross_independent_sets"
    if family_a == family_b:
        return f"within_{family_a}"
    return f"cross_{family_a}_x_{family_b}"


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    ci = percentile_bootstrap_ci(values)
    return float(ci["low"]), float(ci["high"])


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a, dtype=float)[valid]
    b = np.asarray(b, dtype=float)[valid]
    if a.size < 3:
        return float("nan")
    if np.isclose(np.nanstd(a, ddof=1), 0.0) or np.isclose(np.nanstd(b, ddof=1), 0.0):
        return float("nan")
    if method == "pearson":
        rho = stats.pearsonr(a, b).statistic
    else:
        rho = stats.spearmanr(a, b, nan_policy="omit").correlation
    return float(rho) if np.isfinite(rho) else float("nan")


def _load_delta_frame(rsa_files: list[Path], *, conditions: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for priority, path in enumerate(rsa_files):
        if not path.exists():
            continue
        frame = _canonical_time(_read_table(path))
        if frame.empty:
            continue
        frame["roi_set"] = _infer_roi_set(path)
        frame["roi_pool"] = frame["roi_set"].map(_infer_roi_pool)
        frame["roi_family"] = [
            _infer_roi_family(roi_name, roi_set)
            for roi_name, roi_set in zip(frame["roi"].astype(str), frame["roi_set"].astype(str))
        ]
        frame["source_priority"] = int(priority)
        frame["condition"] = frame["condition"].astype(str).str.strip().str.lower().map(CONDITION_MAP)
        frame = frame[frame["condition"].isin(conditions)].copy()
        frame["word_label"] = frame["word_label"].astype(str).str.strip()
        frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
        frame = frame.dropna(subset=["subject", "roi", "time", "condition", "word_label", "similarity"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["source_priority", "subject", "roi", "condition", "time", "word_label"])
    merged = merged.drop_duplicates(subset=["subject", "roi", "condition", "time", "word_label"], keep="first")

    pivot = (
        merged.pivot_table(
            index=["subject", "roi", "roi_set", "roi_pool", "roi_family", "condition", "word_label"],
            columns="time",
            values="similarity",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["pre", "post"]).copy()
    pivot["delta_similarity"] = pivot["post"] - pivot["pre"]
    return pivot


def _subject_metrics(delta_frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
    if delta_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (subject, condition, roi_pool), cond_frame in delta_frame.groupby(["subject", "condition", "roi_pool"]):
        roi_names = sorted(cond_frame["roi"].astype(str).unique().tolist())
        for idx, roi_a in enumerate(roi_names):
            frame_a = cond_frame.loc[
                cond_frame["roi"] == roi_a,
                ["word_label", "delta_similarity", "roi_set", "roi_family"],
            ].rename(columns={"delta_similarity": "delta_a"})
            if frame_a.empty:
                continue
            for roi_b in roi_names[idx + 1:]:
                frame_b = cond_frame.loc[
                    cond_frame["roi"] == roi_b,
                    ["word_label", "delta_similarity", "roi_set", "roi_family"],
                ].rename(columns={"delta_similarity": "delta_b"})
                if frame_b.empty:
                    continue
                joined = frame_a.merge(frame_b, how="inner", on="word_label", suffixes=("_a", "_b"))
                if len(joined) < 3:
                    continue
                rho = _safe_corr(
                    joined["delta_a"].to_numpy(dtype=float),
                    joined["delta_b"].to_numpy(dtype=float),
                    method,
                )
                z = float(np.arctanh(np.clip(rho, -0.999999, 0.999999))) if np.isfinite(rho) else float("nan")
                family_a = str(joined["roi_family_a"].iloc[0])
                family_b = str(joined["roi_family_b"].iloc[0])
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "roi_pool": roi_pool,
                        "roi_set_a": str(joined["roi_set_a"].iloc[0]),
                        "roi_set_b": str(joined["roi_set_b"].iloc[0]),
                        "roi_family_a": family_a,
                        "roi_family_b": family_b,
                        "pair_scope": _pair_scope(roi_pool, family_a, family_b),
                        "roi_a": roi_a,
                        "roi_b": roi_b,
                        "roi_pair": f"{roi_a} <-> {roi_b}",
                        "n_items": int(len(joined)),
                        "repr_connectivity_rho": rho,
                        "repr_connectivity_z": z,
                        "roi_a_mean_delta": float(joined["delta_a"].mean()),
                        "roi_b_mean_delta": float(joined["delta_b"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _summary_subject(subject_frame: pd.DataFrame) -> pd.DataFrame:
    if subject_frame.empty:
        return pd.DataFrame()
    return (
        subject_frame.groupby(["subject", "condition", "roi_pool", "pair_scope"], as_index=False)
        .agg(
            repr_connectivity_z=("repr_connectivity_z", "mean"),
            repr_connectivity_rho=("repr_connectivity_rho", "mean"),
            n_edges=("roi_pair", "nunique"),
            mean_items_per_edge=("n_items", "mean"),
        )
    )


def _group_summary(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_frame.empty:
        return pd.DataFrame()
    for (roi_pool, pair_scope), pair_frame in summary_frame.groupby(["roi_pool", "pair_scope"]):
        yy = pair_frame[pair_frame["condition"] == "yy"].set_index("subject")
        kj = pair_frame[pair_frame["condition"] == "kj"].set_index("subject")
        common_subjects = sorted(set(yy.index) & set(kj.index))
        if common_subjects:
            yy_vals = yy.loc[common_subjects, "repr_connectivity_z"].to_numpy(dtype=float)
            kj_vals = kj.loc[common_subjects, "repr_connectivity_z"].to_numpy(dtype=float)
            paired = paired_t_summary(yy_vals, kj_vals)
            yy_low, yy_high = _bootstrap_mean_ci(yy_vals)
            kj_low, kj_high = _bootstrap_mean_ci(kj_vals)
            rows.append(
                {
                    "roi_pool": roi_pool,
                    "pair_scope": pair_scope,
                    "test_type": "yy_vs_kj",
                    "n": paired["n"],
                    "mean_yy": paired["mean_a"],
                    "mean_kj": paired["mean_b"],
                    "t": paired["t"],
                    "p": paired["p"],
                    "cohens_dz": paired["cohens_dz"],
                    "yy_ci_low": yy_low,
                    "yy_ci_high": yy_high,
                    "kj_ci_low": kj_low,
                    "kj_ci_high": kj_high,
                }
            )
        for condition in ["yy", "kj"]:
            cond_values = pair_frame.loc[pair_frame["condition"] == condition, "repr_connectivity_z"].to_numpy(dtype=float)
            summary = one_sample_t_summary(cond_values, popmean=0.0)
            ci_low, ci_high = _bootstrap_mean_ci(cond_values) if np.isfinite(cond_values).any() else (float("nan"), float("nan"))
            rows.append(
                {
                    "roi_pool": roi_pool,
                    "pair_scope": pair_scope,
                    "test_type": f"{condition}_vs_zero",
                    "condition": condition,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "t": summary["t"],
                    "p": summary["p"],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _edge_group_summary(subject_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if subject_frame.empty:
        return pd.DataFrame()
    for (roi_pool, roi_pair), pair_frame in subject_frame.groupby(["roi_pool", "roi_pair"]):
        yy = pair_frame[pair_frame["condition"] == "yy"].set_index("subject")
        kj = pair_frame[pair_frame["condition"] == "kj"].set_index("subject")
        common_subjects = sorted(set(yy.index) & set(kj.index))
        if not common_subjects:
            continue
        yy_vals = yy.loc[common_subjects, "repr_connectivity_z"].to_numpy(dtype=float)
        kj_vals = kj.loc[common_subjects, "repr_connectivity_z"].to_numpy(dtype=float)
        paired = paired_t_summary(yy_vals, kj_vals)
        rows.append(
            {
                "roi_pool": roi_pool,
                "pair_scope": pair_frame["pair_scope"].iloc[0],
                "roi_pair": roi_pair,
                "roi_a": pair_frame["roi_a"].iloc[0],
                "roi_b": pair_frame["roi_b"].iloc[0],
                "n": paired["n"],
                "mean_yy": paired["mean_a"],
                "mean_kj": paired["mean_b"],
                "t": paired["t"],
                "p": paired["p"],
                "cohens_dz": paired["cohens_dz"],
            }
        )
    return pd.DataFrame(rows)


def _plot_subject_means(summary_frame: pd.DataFrame, figure_path: Path) -> None:
    if summary_frame.empty:
        return
    order = (
        summary_frame.groupby(["roi_pool", "pair_scope"])["subject"]
        .nunique()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not order:
        return
    labels = [f"{pool}: {scope}" for pool, scope in order]
    plot_frame = summary_frame.copy()
    fig, ax = plt.subplots(figsize=(13.5, max(4.5, 0.72 * len(order) + 1.5)))
    colors = {"yy": "#b23a48", "kj": "#3574b7"}
    positions = np.arange(len(order))
    for offset, condition in [(-0.12, "yy"), (0.12, "kj")]:
        means = []
        lows = []
        highs = []
        for roi_pool, pair_scope in order:
            vec = plot_frame.loc[
                (plot_frame["roi_pool"] == roi_pool)
                & (plot_frame["pair_scope"] == pair_scope)
                & (plot_frame["condition"] == condition),
                "repr_connectivity_z",
            ].to_numpy(dtype=float)
            means.append(float(np.nanmean(vec)))
            if np.isfinite(vec).sum() > 0:
                ci_low, ci_high = _bootstrap_mean_ci(vec)
                lows.append(means[-1] - ci_low)
                highs.append(ci_high - means[-1])
            else:
                lows.append(np.nan)
                highs.append(np.nan)
        ax.errorbar(
            means,
            positions + offset,
            xerr=np.vstack([lows, highs]),
            fmt="o",
            color=colors[condition],
            ecolor=colors[condition],
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            label="Metaphor" if condition == "yy" else "Spatial",
        )
    ax.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Representational connectivity (Fisher z)")
    ax.set_title("All-by-all ROI representational connectivity from item-wise neural change", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROI-to-ROI representational connectivity from item-wise post-pre neural change."
    )
    parser.add_argument("--rsa-details", nargs="*", type=Path, default=None,
                        help="One or more rsa_itemwise_details files. Default: literature + literature_spatial under paper_outputs/qc.")
    parser.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--paper-output-root", type=Path, default=None)
    args = parser.parse_args()

    base_dir = _default_base_dir()
    conditions = {"yy", "kj", "baseline"} if args.include_baseline else {"yy", "kj"}
    rsa_files = args.rsa_details or _default_rsa_files(base_dir)
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")
    figures_main = ensure_dir(paper_root / "figures_main")
    qc_dir = ensure_dir(paper_root / "qc")

    delta_frame = _load_delta_frame(rsa_files, conditions=conditions)
    edge_subject_frame = _subject_metrics(delta_frame, method=args.method)
    summary_subject_frame = _summary_subject(edge_subject_frame)
    group_frame = _group_summary(summary_subject_frame)
    edge_group_frame = _edge_group_summary(edge_subject_frame)

    main_table = group_frame[group_frame["test_type"] == "yy_vs_kj"].copy() if not group_frame.empty else pd.DataFrame()
    write_table(main_table, tables_main / "table_repr_connectivity.tsv")
    write_table(summary_subject_frame, tables_si / "table_repr_connectivity_subject.tsv")
    write_table(edge_subject_frame, tables_si / "table_repr_connectivity_edges.tsv")
    write_table(group_frame, qc_dir / "repr_connectivity_summary_group.tsv")
    write_table(edge_group_frame, qc_dir / "repr_connectivity_edge_group.tsv")
    _plot_subject_means(summary_subject_frame, figures_main / "fig_repr_connectivity.png")
    save_json(
        {
            "rsa_details_files": [str(path) for path in rsa_files],
            "method": args.method,
            "include_baseline": bool(args.include_baseline),
            "n_delta_rows": int(len(delta_frame)),
            "n_edge_subject_rows": int(len(edge_subject_frame)),
            "n_summary_subject_rows": int(len(summary_subject_frame)),
            "n_group_rows": int(len(group_frame)),
            "n_edge_group_rows": int(len(edge_group_frame)),
            "primary_analysis_note": "main_functional is primary for B2, but must be interpreted with family-aware summary instead of pooled union.",
            "confirmation_analysis_note": "literature + literature_spatial are used as independent confirmation pool.",
        },
        qc_dir / "repr_connectivity_meta.json",
    )


if __name__ == "__main__":
    main()
