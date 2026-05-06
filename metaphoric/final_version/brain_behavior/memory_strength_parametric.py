#!/usr/bin/env python3
"""Memory-strength parametric RSA (C1, gated by B1 G1).

对每个 subject × ROI × condition，把 B1 产出的 item-level ERS 作为"神经
印迹强度"回归到 retrieval 阶段的 item-level memory / RT：

  logit(memory_i)     = β0 + β1 · ERS_i + β_c · condition + β_x · ERS_i×condition
  log(rt_correct_i)   = β0 + β1 · ERS_i + β_c · condition + β_x · ERS_i×condition

与 B1 内的 GEE 不同，这里按每个 ROI × variant 分别拟合 Mixed LMM（lme4-style
random intercept for subject），更贴近传统的 memory-strength parametric 分析；
同时生成每 subject × ROI 的 memory-weighted ERS（memory=1 mean − memory=0 mean）
作为独立效应量。

实现说明（重要）：
- 二元 ``memory``：使用 **Binomial GEE**（cluster=subject, Exchangeable），对应上式 logit。
- 连续 ``log_rt_correct``：使用 **Gaussian MixedLM**（random intercept for subject）。

Gate G1 条件（由 B1 的组水平表触发）：
- 存在至少一个 ROI ∈ {meta_metaphor, meta_spatial} 在
  ``run3_to_retrieval`` 或 ``run4_to_retrieval`` 的 one-sample ERS
  ``q_bh_within_family < 0.1``。
不满足则脚本只写 ``gate_status.tsv``（gate_pass=False）并退出，不做昂贵回归。

输出：``$paper/qc/memory_strength_parametric_{roi_tag}/`` 与
``$paper/tables_main/table_memory_strength_parametric_{roi_tag}.tsv``。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

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

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402


GATE_VARIANTS = ("run3_to_retrieval", "run4_to_retrieval")
GATE_Q_THRESHOLD = 0.1


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _dedupe_ers_items(items: pd.DataFrame) -> pd.DataFrame:
    """防止跨 roi_tag 目录拼接时双计数（相同 ERS 行在多个目录都存在）。"""
    if items.empty:
        return items
    key_candidates = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "pair_id",
        "variant",
        # behavior columns (if present) help disambiguate repeated retrieval attempts
        "memory",
        "action_time",
        "word_label",
    ]
    subset = [c for c in key_candidates if c in items.columns]
    if not subset:
        return items.drop_duplicates()
    return items.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def _dedupe_ers_group(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return group
    key_candidates = ["analysis_type", "roi_set", "roi", "condition", "variant"]
    subset = [c for c in key_candidates if c in group.columns]
    if not subset:
        return group.drop_duplicates().reset_index(drop=True)
    return group.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def _check_gate_g1(ers_group: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    if ers_group is None or ers_group.empty:
        return False, pd.DataFrame(
            [
                {
                    "gate": "G1_ers_one_sample",
                    "gate_pass": False,
                    "reason": "ers_group_one_sample.tsv is missing or empty",
                    "q_threshold": GATE_Q_THRESHOLD,
                }
            ]
        )
    mask = (
        ers_group["analysis_type"].eq("ers_one_sample_gt_zero")
        & ers_group["variant"].isin(GATE_VARIANTS)
        & ers_group["roi_set"].isin(["meta_metaphor", "meta_spatial"])
        & pd.to_numeric(ers_group["q_bh_within_family"], errors="coerce").lt(
            GATE_Q_THRESHOLD
        )
    )
    passing = ers_group[mask].copy()
    report = pd.DataFrame(
        [
            {
                "gate": "G1_ers_one_sample",
                "gate_pass": bool(not passing.empty),
                "q_threshold": GATE_Q_THRESHOLD,
                "n_passing_rows": int(len(passing)),
                "variants_considered": ",".join(GATE_VARIANTS),
                "roi_sets_considered": "meta_metaphor,meta_spatial",
                "passing_roi_variants": ",".join(
                    passing.apply(
                        lambda r: f"{r['roi_set']}/{r.get('roi','?')}[{r['variant']}]",
                        axis=1,
                    ).tolist()
                ),
            }
        ]
    )
    return bool(not passing.empty), report


def _fit_mixed_gaussian(
    frame: pd.DataFrame, response: str
) -> dict[str, object] | None:
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "error": f"statsmodels missing: {exc}"}

    work = frame.dropna(subset=["ers_z", response, "subject", "condition_yy"]).copy()
    if work["subject"].nunique() < 8 or len(work) < 30 or work[response].nunique() < 2:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixedLM.from_formula(
                f"{response} ~ ers_z * condition_yy",
                groups="subject",
                data=work,
            )
            fit = model.fit(method="lbfgs", reml=True)
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "n_trials": int(len(work)),
            "n_subjects": int(work["subject"].nunique()),
        }

    def _get(name: str) -> tuple[float, float, float]:
        if name not in fit.params.index:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(fit.params[name]),
            float(fit.bse[name]) if name in fit.bse.index else float("nan"),
            float(fit.pvalues[name]) if name in fit.pvalues.index else float("nan"),
        )

    ers_b, ers_se, ers_p = _get("ers_z")
    cond_b, cond_se, cond_p = _get("condition_yy")
    inter_b, inter_se, inter_p = _get("ers_z:condition_yy")
    return {
        "status": "ok",
        "model": "mixedlm_gaussian",
        "n_trials": int(len(work)),
        "n_subjects": int(work["subject"].nunique()),
        "ers_beta": ers_b,
        "ers_se": ers_se,
        "ers_p": ers_p,
        "condition_beta": cond_b,
        "condition_se": cond_se,
        "condition_p": cond_p,
        "interaction_beta": inter_b,
        "interaction_se": inter_se,
        "interaction_p": inter_p,
        "loglik": float(fit.llf) if hasattr(fit, "llf") else float("nan"),
        "converged": bool(getattr(fit, "converged", False)),
    }


def _fit_gee_binomial(frame: pd.DataFrame) -> dict[str, object] | None:
    """Binary memory model: GEE Binomial with subject clusters."""
    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "error": f"statsmodels missing: {exc}"}

    work = frame.dropna(subset=["ers_z", "memory", "subject", "condition_yy"]).copy()
    if work["subject"].nunique() < 8 or len(work) < 30 or work["memory"].nunique() < 2:
        return None

    try:
        model = GEE.from_formula(
            "memory ~ ers_z * condition_yy",
            groups="subject",
            data=work,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        return {
            "status": "failed",
            "model": "gee_binomial",
            "error": str(exc),
            "n_trials": int(len(work)),
            "n_subjects": int(work["subject"].nunique()),
        }

    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    return {
        "status": "ok",
        "model": "gee_binomial",
        "n_trials": int(len(work)),
        "n_subjects": int(work["subject"].nunique()),
        "ers_beta": float(params.get("ers_z", np.nan)),
        "ers_se": float(bse.get("ers_z", np.nan)),
        "ers_p": float(pvalues.get("ers_z", np.nan)),
        "condition_beta": float(params.get("condition_yy", np.nan)),
        "condition_se": float(bse.get("condition_yy", np.nan)),
        "condition_p": float(pvalues.get("condition_yy", np.nan)),
        "interaction_beta": float(params.get("ers_z:condition_yy", np.nan)),
        "interaction_se": float(bse.get("ers_z:condition_yy", np.nan)),
        "interaction_p": float(pvalues.get("ers_z:condition_yy", np.nan)),
        "converged": True,
    }



    if items.empty:
        return pd.DataFrame()
    work = items.dropna(subset=["ers", "memory"]).copy()
    if work.empty:
        return pd.DataFrame()
    per_subject = (
        work.groupby(
            ["subject", "roi_set", "roi", "condition", "variant", "memory"],
            dropna=False,
        )["ers"]
        .mean()
        .reset_index()
    )
    pivot = per_subject.pivot_table(
        index=["subject", "roi_set", "roi", "condition", "variant"],
        columns="memory",
        values="ers",
        aggfunc="mean",
    ).reset_index()
    pivot.columns = [
        "memory_1" if col == 1 or col == 1.0
        else "memory_0" if col == 0 or col == 0.0
        else col
        for col in pivot.columns
    ]
    if "memory_1" not in pivot.columns or "memory_0" not in pivot.columns:
        return pd.DataFrame()
    pivot["delta_remembered_minus_forgot"] = pivot["memory_1"] - pivot["memory_0"]

    group_rows: list[dict[str, object]] = []
    for keys, subset in pivot.groupby(
        ["roi_set", "roi", "condition", "variant"], sort=False
    ):
        roi_set, roi, condition, variant = keys
        values = pd.to_numeric(
            subset["delta_remembered_minus_forgot"], errors="coerce"
        ).dropna()
        if values.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(
            values.to_numpy(dtype=float), 0.0, nan_policy="omit"
        )
        group_rows.append(
            {
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "variant": variant,
                "n_subjects_with_both_outcomes": int(values.size),
                "mean_delta": float(values.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
            }
        )
    group = pd.DataFrame(group_rows)
    if not group.empty:
        group["q_bh_within_family"] = np.nan
        for _, idx in group.groupby(
            ["roi_set", "variant"], dropna=False
        ).groups.items():
            idx = list(idx)
            group.loc[idx, "q_bh_within_family"] = _bh_fdr(group.loc[idx, "p"])
    return group


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Memory-strength parametric Mixed LMM on item-level ERS (gated by G1).",
    )
    parser.add_argument(
        "--paper-output-root", type=Path, default=base_dir / "paper_outputs"
    )
    parser.add_argument(
        "--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"]
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="忽略 Gate G1，无论 B1 结果如何都运行回归（不建议）。",
    )
    args = parser.parse_args()

    tables_main = ensure_dir(args.paper_output_root / "tables_main")

    # 收集 B1 item-level 与 group-level 表
    item_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    for roi_set in args.roi_sets:
        roi_tag = sanitize_roi_tag(roi_set)
        ers_dir = (
            args.paper_output_root
            / "qc"
            / f"encoding_retrieval_similarity_{roi_tag}"
        )
        item_path = ers_dir / "ers_item.tsv"
        group_path = ers_dir / "ers_group_one_sample.tsv"
        if item_path.exists():
            frame = read_table(item_path).copy()
            frame["_source_roi_tag"] = roi_tag
            item_frames.append(frame)
        if group_path.exists():
            frame = read_table(group_path).copy()
            frame["_source_roi_tag"] = roi_tag
            group_frames.append(frame)
    items = (
        pd.concat(item_frames, ignore_index=True, sort=False)
        if item_frames
        else pd.DataFrame()
    )
    group = (
        pd.concat(group_frames, ignore_index=True, sort=False)
        if group_frames
        else pd.DataFrame()
    )
    items = _dedupe_ers_items(items)
    group = _dedupe_ers_group(group)

    # Gate G1 check
    gate_pass, gate_report = _check_gate_g1(group)
    combined_tag = sanitize_roi_tag("meta_mem_strength")
    out_dir = ensure_dir(
        args.paper_output_root / "qc" / f"memory_strength_parametric_{combined_tag}"
    )
    write_table(gate_report, out_dir / "gate_status.tsv")

    if not gate_pass and not args.force_run:
        save_json(
            {
                "gate": "G1_ers_one_sample",
                "gate_pass": False,
                "message": (
                    "Gate G1 not met (no ROI in meta_metaphor/meta_spatial with "
                    "run3_to_retrieval or run4_to_retrieval q < 0.1). Skipped regression. "
                    "Rerun with --force-run to override."
                ),
                "output_dir": str(out_dir),
            },
            out_dir / "manifest.json",
        )
        return

    # Memory-weighted ERS (subject-level delta)
    delta_group = _memory_weighted_ers(items)
    write_table(delta_group, out_dir / "memory_weighted_ers_group.tsv")

    # Item-level MixedLM
    if items.empty:
        save_json(
            {"gate_pass": True, "message": "Items are empty, skipping MixedLM."},
            out_dir / "manifest.json",
        )
        return

    items = items.copy()
    items["condition_yy"] = (items["condition"].astype(str) == "yy").astype(int)
    items["log_rt_correct"] = np.where(
        items["memory"].eq(1) & items["action_time"].gt(0),
        np.log(items["action_time"]),
        np.nan,
    )

    rows: list[dict[str, object]] = []
    for keys, subset in items.groupby(["roi_set", "roi", "variant"], sort=False):
        roi_set, roi, variant = keys
        work = subset.copy()
        work["ers_z"] = _zscore(work["ers"])
        # 1) Binary memory: binomial GEE (logit link)
        fit_mem = _fit_gee_binomial(work)
        if fit_mem is not None:
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "variant": variant,
                    "response": "memory",
                    **fit_mem,
                }
            )

        # 2) Continuous RT (correct only): gaussian MixedLM with random intercept
        fit_rt = _fit_mixed_gaussian(work, "log_rt_correct")
        if fit_rt is not None:
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "variant": variant,
                    "response": "log_rt_correct",
                    **fit_rt,
                }
            )
    parametric = pd.DataFrame(rows)
    if not parametric.empty:
        ok = parametric["status"].eq("ok")
        parametric["q_bh_within_family"] = np.nan
        for _, idx in parametric[ok].groupby(
            ["roi_set", "variant", "response"], dropna=False
        ).groups.items():
            idx = list(idx)
            parametric.loc[idx, "q_bh_within_family"] = _bh_fdr(
                parametric.loc[idx, "ers_p"]
            )
        parametric = parametric.sort_values(
            ["q_bh_within_family", "ers_p"], na_position="last"
        )
    write_table(parametric, out_dir / "memory_strength_parametric.tsv")
    write_table(
        parametric,
        tables_main / f"table_memory_strength_parametric_{combined_tag}.tsv",
    )

    save_json(
        {
            "gate": "G1_ers_one_sample",
            "gate_pass": True,
            "roi_sets": list(args.roi_sets),
            "n_parametric_rows": int(len(parametric)),
            "n_delta_rows": int(len(delta_group)),
            "output_dir": str(out_dir),
            "force_run": bool(args.force_run),
        },
        out_dir / "manifest.json",
    )


if __name__ == "__main__":
    main()
