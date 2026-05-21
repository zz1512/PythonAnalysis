#!/usr/bin/env python3
"""D1c: 个体级 understand/like → memory_strict 多元回归 + 跨被试聚合.

Goal
----
回答："对于同一个句子，run3 理解判断与 run4 喜欢判断如何共同预测 run7 的词对联想
记忆？这一行为链是否区分 yy / kj 条件？"

设计 (per-subject 两阶段) :

Stage 1 — 对每个被试在三个 subset 下分别拟合多元回归：
    memory_strict ~ run3_understand_yes + run4_like_yes

* subset ∈ {yy, kj, pooled}; pooled 在被试内不做条件约束;
* 主引擎 statsmodels Logit; quasi-separation / 拟合失败时 fallback 到 OLS
  (memory_strict 当 0/1 数值, β 解释为 linear probability);
* 每被试每 subset 仅在同时具备 run3_understand_yes / run4_like_yes /
  memory_strict 三列且 outcome 至少有 2 个 class、≥ MIN_TRIALS 行时才入模型;
* 不在被试-subset 内剔除 covariates (按 spec 要求纯行为, 无 covariates / 无神经
  mediator).

Stage 2 — 跨被试聚合：
* per (subset, predictor) 做 one-sample t-test (β vs 0), Wilcoxon signed-rank
  作为非参 sanity;
* paired t-test (β_yy vs β_kj) 在每被试都有两条件 β 的子样本上做.

附带产出 (本批顺带补缺) :

* condition × run3_understand_yes / run4_like_yes 的 2×2 cell 命中率描述表;
* 行为可用性筛选: 用户经验上 run3+run4+run7 全数据的被试约 26 / 28; 在 manifest
  中记录 dropped subject id 与丢弃理由.

设计与既有 d1 的区别:

* d1 是 item-level GLMM 群体平均 (含 pre_pair_similarity_z 与材料 covariates,
  按 network 分块跑); d1c 是被试-level 多元回归 (不含任何 covariate / mediator),
  在被试维度做斜率分布与 paired 推断, 互补不冲突.

输出沙箱: ``paper_outputs/qc/nc_converge/d1c_per_subject_multivariate_dm/``.
不修改任何已有脚本或既有输出.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from shared_nc import (
    add_common_args,
    bh_fdr,
    build_condition_item_id,
    default_config,
    read_table,
    write_outputs,
)

MODULE = "d1c_per_subject_multivariate_dm"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_LEARNING_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
PREDICTORS = ("run3_understand_yes", "run4_like_yes")
INTERACTION_TERM = "run3_understand_x_run4_like"
SUBSETS = ("yy", "kj", "pooled")
MIN_TRIALS = 12  # 每被试每 subset 至少 12 行才拟合 (避免 2 个 predictor 的过拟合)
MEMORY_COL = "memory_strict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--learning-behavior", type=Path, default=None)
    parser.add_argument(
        "--min-trials",
        type=int,
        default=MIN_TRIALS,
        help="每被试每 subset 最少 trial 数; 低于此值跳过拟合.",
    )
    parser.add_argument(
        "--include-interaction",
        action="store_true",
        help="额外跑一轮 understand × like 交互模型 (3 predictor: U + L + U:L); "
        "结果以独立 model='with_interaction' 行写入 per_subject_betas / group_inference.",
    )
    return parser.parse_args()


def _coerce_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {"true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0,
         "1": 1.0, "1.0": 1.0,
         "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0,
         "0": 0.0, "0.0": 0.0}
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def _derive_memory_strict(frame: pd.DataFrame) -> pd.Series:
    if "remembered_strict" in frame.columns:
        return pd.to_numeric(frame["remembered_strict"], errors="coerce")
    if "memory_score" in frame.columns:
        ms = pd.to_numeric(frame["memory_score"], errors="coerce")
        return pd.Series(np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    if "memory" in frame.columns:
        ms = pd.to_numeric(frame["memory"], errors="coerce")
        return pd.Series(np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    if "memory_successes" in frame.columns:
        ms = pd.to_numeric(frame["memory_successes"], errors="coerce")
        return pd.Series(np.where(ms.ge(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _load_and_merge(item_path: Path, behavior_path: Path) -> tuple[pd.DataFrame, dict]:
    """读 item_mechanism_table.tsv + learning_behavior_item.tsv, item 级 merge."""
    audit: dict[str, object] = {
        "item_path": str(item_path),
        "behavior_path": str(behavior_path),
        "item_exists": item_path.exists(),
        "behavior_exists": behavior_path.exists(),
    }
    if not item_path.exists() or not behavior_path.exists():
        audit["status"] = "missing_input"
        return pd.DataFrame(), audit

    item = build_condition_item_id(read_table(item_path))
    behav = build_condition_item_id(read_table(behavior_path))

    # item 级而非 network 平均: drop network 重复
    if "network" in item.columns:
        item = item.drop(columns=[c for c in ("network",) if c in item.columns])
    item = item.drop_duplicates(subset=[c for c in ("subject", "condition_item_id") if c in item.columns])

    keys = [c for c in ("subject", "condition_item_id") if c in item.columns and c in behav.columns]
    if len(keys) < 2:
        audit["status"] = f"missing_merge_keys: {keys}"
        return pd.DataFrame(), audit

    behav_cols = [*keys, *(c for c in PREDICTORS if c in behav.columns)]
    behav = behav[behav_cols].copy()
    for col in PREDICTORS:
        if col in behav.columns:
            behav[col] = _coerce_binary(behav[col])
    behav = behav.groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)

    merged = item.merge(behav, on=keys, how="left", validate="many_to_one")
    merged[MEMORY_COL] = _derive_memory_strict(merged)
    if "condition" in merged.columns:
        merged["condition"] = merged["condition"].astype(str).str.lower()

    audit.update(
        {
            "status": "ok",
            "item_rows": int(len(item)),
            "behavior_rows": int(len(behav)),
            "merged_rows": int(len(merged)),
            "n_subjects_raw": int(merged["subject"].nunique()) if "subject" in merged.columns else None,
            "n_items_raw": int(merged["condition_item_id"].nunique()) if "condition_item_id" in merged.columns else None,
        }
    )
    for col in (*PREDICTORS, MEMORY_COL):
        audit[f"{col}_n_nonnull"] = int(merged[col].notna().sum())
    return merged, audit


def _eligible_subjects(frame: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """筛选同时具备 run3+run4+run7 行为的被试 (经验上 ~26/28)."""
    drop_log: list[dict] = []
    eligible: list[str] = []
    if "subject" not in frame.columns:
        return eligible, drop_log
    needed = (*PREDICTORS, MEMORY_COL)
    for subj, sub in frame.groupby("subject", dropna=False):
        nonnull = {c: int(sub[c].notna().sum()) for c in needed if c in sub.columns}
        missing = [c for c, n in nonnull.items() if n == 0]
        if missing:
            drop_log.append({"subject": subj, "reason": f"all_null:{','.join(missing)}", **nonnull})
            continue
        eligible.append(subj)
    return eligible, drop_log


def _fit_one(model_data: pd.DataFrame, predictor_list: list[str]) -> dict:
    """主跑 Logit, fallback 到 OLS. predictor_list 决定列顺序; 返回 {col__beta/se/p, status, engine, n_obs}."""
    from numpy.linalg import LinAlgError

    out: dict = {"engine": None, "status": None, "n_obs": int(len(model_data))}

    # 检查 outcome 单 class
    y = pd.to_numeric(model_data[MEMORY_COL], errors="coerce")
    if y.nunique(dropna=True) < 2:
        out["status"] = "single_class_outcome"
        return out

    X = model_data[predictor_list].astype(float).to_numpy()
    y_vec = y.astype(float).to_numpy()
    X_const = np.column_stack([np.ones(len(X)), X])
    name_seq = ("__intercept__", *predictor_list)

    # design rank check (若交互列与主 predictor 在被试-subset 内完全共线则跳)
    if np.linalg.matrix_rank(X_const) < X_const.shape[1]:
        out["status"] = "design_rank_deficient"
        return out

    # Stage 1a: Logit (statsmodels)
    try:
        import statsmodels.api as sm  # type: ignore

        model = sm.Logit(y_vec, X_const)
        try:
            res = model.fit(disp=False, maxiter=200)
            params = res.params
            ses = res.bse
            if not np.all(np.isfinite(params)) or not np.all(np.isfinite(ses)):
                raise ValueError("logit_nonfinite")
            for i, name in enumerate(name_seq):
                out[f"{name}__beta"] = float(params[i])
                out[f"{name}__se"] = float(ses[i])
                if name != "__intercept__":
                    z = float(params[i] / ses[i]) if ses[i] > 0 else np.nan
                    out[f"{name}__p"] = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
            out["engine"] = "logit"
            out["status"] = "ok"
            return out
        except Exception as exc:  # quasi-separation / convergence failure
            logit_reason = str(exc)[:80]
    except Exception as exc:  # statsmodels unavailable
        logit_reason = f"statsmodels_unavailable: {exc}"

    # Stage 1b: OLS fallback (memory_strict 当 0/1 数值, β = linear probability)
    try:
        beta, *_ = np.linalg.lstsq(X_const, y_vec, rcond=None)
        residuals = y_vec - X_const @ beta
        df_resid = max(len(X) - X_const.shape[1], 1)
        sigma2 = float(residuals @ residuals / df_resid)
        XtX_inv = np.linalg.pinv(X_const.T @ X_const)
        cov = sigma2 * XtX_inv
        ses = np.sqrt(np.diag(cov))
        for i, name in enumerate(name_seq):
            out[f"{name}__beta"] = float(beta[i])
            out[f"{name}__se"] = float(ses[i])
            if name != "__intercept__":
                t = float(beta[i] / ses[i]) if ses[i] > 0 else np.nan
                out[f"{name}__p"] = (
                    float(2 * (1 - stats.t.cdf(abs(t), df_resid))) if np.isfinite(t) else np.nan
                )
        out["engine"] = "ols_fallback"
        out["status"] = f"ok_fallback: {logit_reason}"
    except (LinAlgError, ValueError) as exc:
        out["engine"] = "ols_fallback"
        out["status"] = f"ols_failed: {exc}"
    return out


def per_subject_betas(
    frame: pd.DataFrame,
    eligible: list[str],
    min_trials: int,
    *,
    include_interaction: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    model_specs: list[tuple[str, list[str]]] = [("main_effects", list(PREDICTORS))]
    if include_interaction:
        model_specs.append(("with_interaction", [*PREDICTORS, INTERACTION_TERM]))
    for subj in eligible:
        sub_all = frame.loc[frame["subject"].eq(subj)].copy()
        for subset in SUBSETS:
            if subset == "pooled":
                data = sub_all
            else:
                data = sub_all.loc[sub_all["condition"].eq(subset)] if "condition" in sub_all.columns else sub_all.iloc[:0]
            data = data.dropna(subset=[MEMORY_COL, *PREDICTORS]).copy()
            # 派生交互列 (在 subset 切完后再 build, 避免不同 subset 间污染)
            if include_interaction and not data.empty:
                data[INTERACTION_TERM] = (
                    pd.to_numeric(data["run3_understand_yes"], errors="coerce")
                    * pd.to_numeric(data["run4_like_yes"], errors="coerce")
                )
            for model_name, predictor_list in model_specs:
                base = {
                    "subject": subj,
                    "subset": subset,
                    "model": model_name,
                    "n_obs": int(len(data)),
                }
                if len(data) < min_trials:
                    rows.append({**base, "status": "too_few_rows", "engine": None})
                    continue
                fit_out = _fit_one(data, predictor_list)
                rows.append({**base, **fit_out})
    return pd.DataFrame(rows)


def group_inference(per_subject: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: 跨被试聚合 (one-sample t-test + Wilcoxon + paired yy vs kj).

    每 (model, subset, predictor) 一行 one-sample t-test;
    每 (model, predictor) 一行 paired_yy_minus_kj.
    """
    rows: list[dict] = []
    if per_subject.empty:
        return pd.DataFrame(rows)
    if "model" not in per_subject.columns:
        per_subject = per_subject.assign(model="main_effects")
    ok_mask = per_subject["status"].astype(str).str.startswith("ok")
    ok = per_subject.loc[ok_mask].copy()
    models_present = list(dict.fromkeys(ok["model"].astype(str)))

    for model_name in models_present:
        ok_m = ok.loc[ok["model"].eq(model_name)]
        # predictor 集合: main_effects 仅 PREDICTORS; with_interaction 多一个 INTERACTION_TERM
        predictor_set = list(PREDICTORS)
        if model_name == "with_interaction":
            predictor_set = [*PREDICTORS, INTERACTION_TERM]
        for subset in SUBSETS:
            sub = ok_m.loc[ok_m["subset"].eq(subset)]
            for predictor in predictor_set:
                col = f"{predictor}__beta"
                if col not in sub.columns:
                    continue
                betas = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
                if betas.size < 3:
                    rows.append(
                        {
                            "model": model_name,
                            "subset": subset,
                            "predictor": predictor,
                            "test": "one_sample_t",
                            "n": int(betas.size),
                            "status": "too_few_subjects",
                        }
                    )
                    continue
                t_stat, t_p = stats.ttest_1samp(betas, 0.0, nan_policy="omit")
                try:
                    w_stat, w_p = stats.wilcoxon(betas, zero_method="wilcox", alternative="two-sided")
                    wilcox_status = "ok"
                except ValueError as exc:
                    w_stat, w_p = np.nan, np.nan
                    wilcox_status = f"wilcox_failed: {exc}"
                rows.append(
                    {
                        "model": model_name,
                        "subset": subset,
                        "predictor": predictor,
                        "test": "one_sample_t",
                        "n": int(betas.size),
                        "mean_beta": float(np.mean(betas)),
                        "median_beta": float(np.median(betas)),
                        "sd_beta": float(np.std(betas, ddof=1)) if betas.size > 1 else np.nan,
                        "t": float(t_stat),
                        "p": float(t_p),
                        "wilcoxon_W": float(w_stat) if np.isfinite(w_stat) else np.nan,
                        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                        "wilcoxon_status": wilcox_status,
                        "status": "ok",
                    }
                )

        # Paired YY vs KJ in same subject (per model)
        for predictor in predictor_set:
            col = f"{predictor}__beta"
            if col not in ok_m.columns:
                continue
            wide = ok_m.loc[ok_m["subset"].isin(("yy", "kj")), ["subject", "subset", col]].pivot(
                index="subject", columns="subset", values=col
            )
            wide = wide.dropna(subset=["yy", "kj"]) if {"yy", "kj"}.issubset(wide.columns) else pd.DataFrame()
            if len(wide) < 3:
                rows.append(
                    {
                        "model": model_name,
                        "subset": "paired_yy_minus_kj",
                        "predictor": predictor,
                        "test": "paired_t",
                        "n": int(len(wide)),
                        "status": "too_few_paired_subjects",
                    }
                )
                continue
            diff = (wide["yy"] - wide["kj"]).to_numpy()
            t_stat, t_p = stats.ttest_rel(wide["yy"], wide["kj"], nan_policy="omit")
            try:
                w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                wilcox_status = "ok"
            except ValueError as exc:
                w_stat, w_p = np.nan, np.nan
                wilcox_status = f"wilcox_failed: {exc}"
            rows.append(
                {
                    "model": model_name,
                    "subset": "paired_yy_minus_kj",
                    "predictor": predictor,
                    "test": "paired_t",
                    "n": int(len(wide)),
                    "mean_beta": float(np.mean(diff)),
                    "median_beta": float(np.median(diff)),
                    "sd_beta": float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan,
                    "t": float(t_stat),
                    "p": float(t_p),
                    "wilcoxon_W": float(w_stat) if np.isfinite(w_stat) else np.nan,
                    "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                    "wilcoxon_status": wilcox_status,
                    "status": "ok",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty and "p" in out.columns:
        # BH-FDR family: 每 model 内单独做 (避免 main vs interaction 跨族膨胀)
        out["q_bh"] = np.nan
        for model_name in models_present:
            mask = out["model"].eq(model_name) & out["status"].astype(str).eq("ok")
            if mask.any():
                out.loc[mask, "q_bh"] = bh_fdr(out.loc[mask, "p"].to_numpy())
    return out


def cell_means_2x2(frame: pd.DataFrame, eligible: list[str]) -> pd.DataFrame:
    """condition × predictor 2×2 cell 的 memory_strict 命中率描述表."""
    rows: list[dict] = []
    if "condition" not in frame.columns:
        return pd.DataFrame(rows)
    base = frame.loc[frame["subject"].isin(eligible)].copy()
    for predictor in PREDICTORS:
        keep = ["subject", "condition", predictor, MEMORY_COL]
        keep = [c for c in keep if c in base.columns]
        data = base[keep].dropna(subset=[predictor, MEMORY_COL]).copy()
        if data.empty:
            continue
        # 转 0/1 标签
        data[predictor] = (pd.to_numeric(data[predictor], errors="coerce") >= 0.5).astype(int)
        for (cond, val), cell in data.groupby(["condition", predictor], dropna=False):
            n = int(len(cell))
            if n == 0:
                continue
            mem = pd.to_numeric(cell[MEMORY_COL], errors="coerce").dropna()
            mean = float(mem.mean()) if len(mem) else np.nan
            sd = float(mem.std(ddof=1)) if len(mem) > 1 else np.nan
            se = float(sd / np.sqrt(len(mem))) if len(mem) > 1 else np.nan
            rows.append(
                {
                    "predictor": predictor,
                    "condition": cond,
                    "predictor_value": int(val),
                    "n_items": n,
                    "n_subjects": int(cell["subject"].nunique()) if "subject" in cell.columns else None,
                    "memory_strict_mean": mean,
                    "memory_strict_sd": sd,
                    "memory_strict_se": se,
                }
            )
    return pd.DataFrame(rows)


def manifest(
    item_path: Path,
    behavior_path: Path,
    audit: dict,
    eligible: list[str],
    drop_log: list[dict],
    per_subject: pd.DataFrame,
    min_trials: int,
) -> pd.DataFrame:
    info: dict[str, object] = {
        "item_path": str(item_path),
        "behavior_path": str(behavior_path),
        "merge_status": audit.get("status"),
        "merged_rows": audit.get("merged_rows"),
        "n_subjects_raw": audit.get("n_subjects_raw"),
        "n_subjects_eligible": len(eligible),
        "n_subjects_dropped": len(drop_log),
        "min_trials": int(min_trials),
    }
    if not per_subject.empty and "engine" in per_subject.columns:
        info["fits_logit"] = int((per_subject["engine"].astype(str) == "logit").sum())
        info["fits_ols_fallback"] = int((per_subject["engine"].astype(str) == "ols_fallback").sum())
        info["fits_skipped"] = int(per_subject["engine"].isna().sum())
    out = pd.DataFrame([info])
    if drop_log:
        out["dropped_subjects"] = ";".join(d["subject"] for d in drop_log)
    return out


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    item_path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    behavior_path = Path(args.learning_behavior or cfg.paper_output_root / DEFAULT_LEARNING_BEHAVIOR)

    merged, audit = _load_and_merge(item_path, behavior_path)
    if merged.empty:
        if not args.allow_empty:
            raise FileNotFoundError(f"输入缺失: {audit}")
        write_outputs(
            cfg,
            MODULE,
            {
                "per_subject_betas.tsv": pd.DataFrame(),
                "group_inference.tsv": pd.DataFrame(),
                "cell_means_2x2.tsv": pd.DataFrame(),
                "subject_eligibility.tsv": pd.DataFrame(
                    columns=["subject", "reason", *PREDICTORS, MEMORY_COL]
                ),
                "manifest.tsv": pd.DataFrame([{**audit, "n_subjects_eligible": 0}]),
            },
        )
        return

    eligible, drop_log = _eligible_subjects(merged)
    drop_frame = pd.DataFrame(drop_log) if drop_log else pd.DataFrame(
        columns=["subject", "reason", *PREDICTORS, MEMORY_COL]
    )
    per_subject = per_subject_betas(
        merged, eligible, min_trials=args.min_trials, include_interaction=args.include_interaction
    )
    group = group_inference(per_subject)
    cells = cell_means_2x2(merged, eligible)

    manifest_frame = manifest(item_path, behavior_path, audit, eligible, drop_log, per_subject, args.min_trials)
    manifest_frame["include_interaction"] = bool(args.include_interaction)

    write_outputs(
        cfg,
        MODULE,
        {
            "per_subject_betas.tsv": per_subject,
            "group_inference.tsv": group,
            "cell_means_2x2.tsv": cells,
            "subject_eligibility.tsv": drop_frame,
            "manifest.tsv": manifest_frame,
        },
    )


if __name__ == "__main__":
    main()
