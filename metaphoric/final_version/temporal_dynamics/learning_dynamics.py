"""
learning_dynamics.py

用途
- 在学习阶段（learn_yy / learn_kj）做 repeated-exposure analysis。
- 设计前提：Run3 与 Run4 使用同一批句子，每句各出现一次，但顺序不同。
- 主分析（A2）：pair-based repeated-exposure discrimination
  - same_pair_cross_run_similarity = sim(run3_i, run4_i)
  - different_pair_cross_run_similarity = mean(sim(run3_i, run4_j), j != i)
  - pair_discrimination = same_pair - different_pair
- 补充分析（A1）：same_sentence_cross_run_similarity（即 raw same-pair cross-run similarity）

输出（paper_outputs）
- `tables_main/table_learning_repeated_exposure.tsv`：主表（A2 主指标 + A1 补充指标）
- `tables_si/table_learning_repeated_exposure_subject.tsv`：被试级汇总
- `figures_main/fig_learning_dynamics.png`：主图（A2 + A1）
- `qc/learning_repeated_exposure_itemwise.tsv`：item 级明细
- `qc/learning_repeated_exposure_group.tsv`：完整组水平统计
- `qc/learning_dynamics_meta.json`：元信息
"""

from __future__ import annotations



import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
from common.roi_library import sanitize_roi_tag, select_roi_masks  # noqa: E402
from common.pattern_metrics import load_masked_samples  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    sep = "\t" if suffixes.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _metadata_path(pattern_path: Path) -> Path:
    name = pattern_path.name
    if name.endswith(".nii.gz"):
        meta_name = name[:-7] + "_metadata.tsv"
    elif name.endswith(".nii"):
        meta_name = name[:-4] + "_metadata.tsv"
    else:
        meta_name = name + "_metadata.tsv"
    return pattern_path.with_name(meta_name)


def _default_pattern_root() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402
    return Path(BASE_DIR) / "pattern_root"


def _default_roi_manifest() -> Path:
    from rsa_analysis.rsa_config import ROI_MANIFEST  # noqa: E402
    return Path(ROI_MANIFEST)


def _infer_roi_family(roi_name: str, roi_set: str) -> str:
    if str(roi_set).strip().lower() != "main_functional":
        return str(roi_set)
    if "Metaphor_gt_Spatial" in roi_name:
        return "Metaphor_gt_Spatial"
    if "Spatial_gt_Metaphor" in roi_name:
        return "Spatial_gt_Metaphor"
    return "main_functional_other"


def _normalize_run_series(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(as_str, errors="coerce")


def _rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    a_norm = np.linalg.norm(a_centered, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_centered, axis=1, keepdims=True)
    a_norm[a_norm == 0] = np.nan
    b_norm[b_norm == 0] = np.nan
    a_unit = a_centered / a_norm
    b_unit = b_centered / b_norm
    return a_unit @ b_unit.T


def _aggregate_by_label(samples: np.ndarray, meta: pd.DataFrame, key_col: str) -> tuple[np.ndarray, pd.DataFrame]:
    if len(samples) != len(meta):
        raise ValueError("Pattern rows and metadata rows are misaligned.")
    rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    work = meta.copy().reset_index(drop=True)
    work["_row_index"] = np.arange(len(work))
    for label, group in work.groupby(key_col, sort=True):
        idx = group["_row_index"].to_numpy(dtype=int)
        rows.append(np.nanmean(samples[idx, :], axis=0))
        first = group.iloc[0]
        meta_rows.append(
            {
                key_col: label,
                "pair_id": first.get("pair_id", label),
                "run": first.get("run", np.nan),
            }
        )
    return np.vstack(rows), pd.DataFrame(meta_rows)


def _extract_condition_frame(
    samples: np.ndarray,
    meta: pd.DataFrame,
    *,
    subject: str,
    roi: str,
    condition: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    meta = meta.copy().reset_index(drop=True)
    if "run" not in meta.columns:
        raise ValueError(f"{subject}|{condition}|{roi}: metadata is missing 'run'")
    if "unique_label" not in meta.columns:
        raise ValueError(f"{subject}|{condition}|{roi}: metadata is missing 'unique_label'")
    if "pair_id" not in meta.columns:
        meta["pair_id"] = meta["unique_label"]

    meta["run_num"] = _normalize_run_series(meta["run"])
    run3_mask = meta["run_num"].eq(3)
    run4_mask = meta["run_num"].eq(4)
    if not run3_mask.any() or not run4_mask.any():
        return pd.DataFrame(), None

    samples3, meta3 = _aggregate_by_label(samples[run3_mask.to_numpy(), :], meta.loc[run3_mask], "unique_label")
    samples4, meta4 = _aggregate_by_label(samples[run4_mask.to_numpy(), :], meta.loc[run4_mask], "unique_label")

    labels3 = meta3["unique_label"].astype(str)
    labels4 = meta4["unique_label"].astype(str)
    common_labels = sorted(set(labels3) & set(labels4))
    if len(common_labels) < 2:
        return pd.DataFrame(), {
            "subject": subject,
            "roi": roi,
            "condition": condition,
            "n_run3_items": int(len(meta3)),
            "n_run4_items": int(len(meta4)),
            "n_common_items": int(len(common_labels)),
        }

    idx3 = [int(labels3[labels3 == label].index[0]) for label in common_labels]
    idx4 = [int(labels4[labels4 == label].index[0]) for label in common_labels]
    corr = _rowwise_correlation(samples3[idx3, :], samples4[idx4, :])

    item_rows: list[dict[str, Any]] = []
    for pos, label in enumerate(common_labels):
        same = float(corr[pos, pos])
        different_vec = np.delete(corr[pos, :], pos)
        diff = float(np.nanmean(different_vec)) if different_vec.size > 0 else float("nan")
        pair_id = meta3.loc[idx3[pos], "pair_id"]
        item_rows.append(
            {
                "subject": subject,
                "roi": roi,
                "condition": condition,
                "unique_label": str(label),
                "pair_id": pair_id,
                "same_sentence_cross_run_similarity": same,
                "same_pair_cross_run_similarity": same,
                "different_pair_cross_run_similarity": diff,
                "pair_discrimination": float(same - diff) if np.isfinite(same) and np.isfinite(diff) else float("nan"),
            }
        )
    return pd.DataFrame(item_rows), {
        "subject": subject,
        "roi": roi,
        "condition": condition,
        "n_run3_items": int(len(meta3)),
        "n_run4_items": int(len(meta4)),
        "n_common_items": int(len(common_labels)),
    }


def _subject_level(item_frame: pd.DataFrame) -> pd.DataFrame:
    if item_frame.empty:
        return pd.DataFrame()
    agg = (
        item_frame.groupby(["subject", "roi", "roi_family", "condition"], as_index=False)
        .agg(
            same_sentence_cross_run_similarity=("same_sentence_cross_run_similarity", "mean"),
            same_pair_cross_run_similarity=("same_pair_cross_run_similarity", "mean"),
            different_pair_cross_run_similarity=("different_pair_cross_run_similarity", "mean"),
            pair_discrimination=("pair_discrimination", "mean"),
            n_common_items=("unique_label", "nunique"),
        )
    )
    return agg


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    ci = percentile_bootstrap_ci(values)
    return float(ci["low"]), float(ci["high"])


def _group_summary(subject_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if subject_frame.empty:
        return pd.DataFrame()

    metrics = [
        ("pair_discrimination", "primary"),
        ("same_sentence_cross_run_similarity", "supplementary"),
        ("different_pair_cross_run_similarity", "context"),
    ]
    for roi, roi_frame in subject_frame.groupby("roi"):
        roi_family = roi_frame["roi_family"].iloc[0] if "roi_family" in roi_frame.columns and not roi_frame.empty else ""
        yy = roi_frame[roi_frame["condition"] == "yy"].set_index("subject")
        kj = roi_frame[roi_frame["condition"] == "kj"].set_index("subject")
        common_subjects = sorted(set(yy.index) & set(kj.index))
        for metric, role in metrics:
            if common_subjects:
                yy_vals = yy.loc[common_subjects, metric].to_numpy(dtype=float)
                kj_vals = kj.loc[common_subjects, metric].to_numpy(dtype=float)
                paired = paired_t_summary(yy_vals, kj_vals)
                yy_low, yy_high = _bootstrap_mean_ci(yy_vals)
                kj_low, kj_high = _bootstrap_mean_ci(kj_vals)
                rows.append(
                    {
                        "roi": roi,
                        "roi_family": roi_family,
                        "metric": metric,
                        "analysis_role": role,
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
            for condition_label, cond_frame in [("yy", yy), ("kj", kj)]:
                values = cond_frame[metric].to_numpy(dtype=float)
                one_sample = one_sample_t_summary(values, popmean=0.0)
                ci_low, ci_high = _bootstrap_mean_ci(values)
                rows.append(
                    {
                        "roi": roi,
                        "roi_family": roi_family,
                        "metric": metric,
                        "analysis_role": role,
                        "test_type": f"{condition_label}_vs_zero",
                        "condition": condition_label,
                        "n": one_sample["n"],
                        "mean": one_sample["mean"],
                        "t": one_sample["t"],
                        "p": one_sample["p"],
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def _family_group_summary(subject_frame: pd.DataFrame) -> pd.DataFrame:
    if subject_frame.empty or "roi_family" not in subject_frame.columns:
        return pd.DataFrame()
    family_subject = (
        subject_frame.groupby(["subject", "roi_family", "condition"], as_index=False)
        .agg(
            pair_discrimination=("pair_discrimination", "mean"),
            same_sentence_cross_run_similarity=("same_sentence_cross_run_similarity", "mean"),
            same_pair_cross_run_similarity=("same_pair_cross_run_similarity", "mean"),
            different_pair_cross_run_similarity=("different_pair_cross_run_similarity", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for roi_family, fam_frame in family_subject.groupby("roi_family"):
        yy = fam_frame[fam_frame["condition"] == "yy"].set_index("subject")
        kj = fam_frame[fam_frame["condition"] == "kj"].set_index("subject")
        common_subjects = sorted(set(yy.index) & set(kj.index))
        for metric in ["pair_discrimination", "same_sentence_cross_run_similarity"]:
            if not common_subjects:
                continue
            yy_vals = yy.loc[common_subjects, metric].to_numpy(dtype=float)
            kj_vals = kj.loc[common_subjects, metric].to_numpy(dtype=float)
            paired = paired_t_summary(yy_vals, kj_vals)
            rows.append(
                {
                    "roi_family": roi_family,
                    "metric": metric,
                    "test_type": "yy_vs_kj",
                    "n": paired["n"],
                    "mean_yy": paired["mean_a"],
                    "mean_kj": paired["mean_b"],
                    "t": paired["t"],
                    "p": paired["p"],
                    "cohens_dz": paired["cohens_dz"],
                }
            )
    return pd.DataFrame(rows)


def _plot_learning_dynamics(subject_frame: pd.DataFrame, figure_path: Path) -> None:
    if subject_frame.empty:
        return
    roi_order = (
        subject_frame.groupby("roi")["subject"]
        .nunique()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    plot_frame = subject_frame[subject_frame["roi"].isin(roi_order)].copy()
    if plot_frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.5, 0.75 * len(roi_order) + 1.5)), sharey=True)
    colors = {"yy": "#b23a48", "kj": "#3574b7"}
    metrics = [
        ("pair_discrimination", "A2: same - different"),
        ("same_sentence_cross_run_similarity", "A1: same-sentence cross-run similarity"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        positions = np.arange(len(roi_order))
        for offset, condition in [(-0.12, "yy"), (0.12, "kj")]:
            values = []
            lows = []
            highs = []
            for roi in roi_order:
                vec = plot_frame.loc[
                    (plot_frame["roi"] == roi) & (plot_frame["condition"] == condition),
                    metric,
                ].to_numpy(dtype=float)
                values.append(float(np.nanmean(vec)))
                if np.isfinite(vec).sum() > 0:
                    ci_low, ci_high = _bootstrap_mean_ci(vec)
                else:
                    ci_low, ci_high = float("nan"), float("nan")
                lows.append(values[-1] - ci_low if np.isfinite(ci_low) else np.nan)
                highs.append(ci_high - values[-1] if np.isfinite(ci_high) else np.nan)
            ax.errorbar(
                values,
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
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mean similarity")
        ax.set_yticks(positions)
        ax.set_yticklabels(roi_order, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learning-stage repeated-exposure analysis: A2 main + A1 supplementary."
    )
    parser.add_argument("pattern_root", type=Path, nargs="?", default=None, help="Pattern root with per-subject folders.")
    parser.add_argument("roi_dir", type=Path, nargs="?", default=None, help="Optional ROI mask directory override.")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="Optional output root.")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--time", default="learn", help="Time label to use (default: learn).")
    parser.add_argument("--roi-set", default="literature",
                        help="ROI scope for this analysis. Recommended: literature (default). "
                             "Use main_functional only as supplementary family-split analysis.")
    parser.add_argument("--roi-manifest", type=Path, default=None,
                        help="Optional ROI manifest path. Defaults to rsa_config.ROI_MANIFEST.")
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=None,
        help="Unified paper output root. Default: ${PYTHON_METAPHOR_ROOT}/paper_outputs",
    )
    args = parser.parse_args()

    pattern_root = args.pattern_root or _default_pattern_root()
    roi_manifest = args.roi_manifest or _default_roi_manifest()
    if args.roi_dir is not None:
        roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    elif roi_manifest.exists():
        roi_masks = select_roi_masks(roi_manifest, roi_set=args.roi_set, include_flag="include_in_rsa")
        roi_paths = [Path(path) for path in roi_masks.values()]
    else:
        raise FileNotFoundError(f"ROI manifest not found and roi_dir not provided: {roi_manifest}")

    paper_root = args.paper_output_root or (pattern_root.parents[0] / "paper_outputs")
    output_root = ensure_dir(args.output_dir or paper_root)
    qc_dir = ensure_dir(output_root / "qc")
    tables_main = ensure_dir(output_root / "tables_main")
    tables_si = ensure_dir(output_root / "tables_si")
    figures_main = ensure_dir(output_root / "figures_main")
    subject_dirs = sorted([p for p in pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    item_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for condition in ["yy", "kj"]:
                pattern_path = subject_dir / args.filename_template.format(time=args.time, condition=condition)
                metadata_path = _metadata_path(pattern_path)
                if not pattern_path.exists() or not metadata_path.exists():
                    continue
                samples = load_masked_samples(pattern_path, roi_path)
                metadata = _read_table(metadata_path)
                item_frame, qc_row = _extract_condition_frame(
                    samples,
                    metadata,
                    subject=subject_dir.name,
                    roi=roi_name,
                    condition=condition,
                )
                if not item_frame.empty:
                    item_frame["roi_family"] = _infer_roi_family(roi_name, args.roi_set)
                    item_frame["roi_set"] = sanitize_roi_tag(args.roi_set)
                    item_frames.append(item_frame)
                if qc_row is not None:
                    qc_row["roi_family"] = _infer_roi_family(roi_name, args.roi_set)
                    qc_row["roi_set"] = sanitize_roi_tag(args.roi_set)
                    qc_rows.append(qc_row)

    item_frame = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    subject_frame = _subject_level(item_frame)
    group_frame = _group_summary(subject_frame)
    family_frame = _family_group_summary(subject_frame)

    main_table = group_frame[
        (
            group_frame["metric"].isin(["pair_discrimination", "same_sentence_cross_run_similarity"])
            & group_frame["test_type"].eq("yy_vs_kj")
        )
    ].copy() if not group_frame.empty else pd.DataFrame()

    write_table(main_table, tables_main / "table_learning_repeated_exposure.tsv")
    write_table(subject_frame, tables_si / "table_learning_repeated_exposure_subject.tsv")
    write_table(family_frame, tables_si / "table_learning_repeated_exposure_family.tsv")
    write_table(item_frame, qc_dir / "learning_repeated_exposure_itemwise.tsv")
    write_table(group_frame, qc_dir / "learning_repeated_exposure_group.tsv")
    write_table(pd.DataFrame(qc_rows), qc_dir / "learning_repeated_exposure_qc.tsv")
    _plot_learning_dynamics(subject_frame, figures_main / "fig_learning_dynamics.png")
    save_json(
        {
            "analysis_definition": "A2 primary + A1 supplementary repeated-exposure analysis",
            "roi_set": sanitize_roi_tag(args.roi_set),
            "roi_scope_note": "Primary analysis should use literature ROI. main_functional is supplementary and must be interpreted as family-split, not pooled union.",
            "n_item_rows": int(len(item_frame)),
            "n_subject_rows": int(len(subject_frame)),
            "n_group_rows": int(len(group_frame)),
            "n_family_rows": int(len(family_frame)),
            "n_qc_rows": int(len(qc_rows)),
            "output_root": str(output_root),
        },
        qc_dir / "learning_dynamics_meta.json",
    )


if __name__ == "__main__":
    main()
