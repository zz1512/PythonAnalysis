#!/usr/bin/env python3
"""Encoding-Retrieval Similarity (ERS) — item-level learning→memory bridge (B1).

对每个 subject × ROI × pair_id，计算三种 ERS 变体：

- ``run3_to_retrieval`` : corr(L_i^run3, R_i)
- ``run4_to_retrieval`` : corr(L_i^run4, R_i)
- ``max_run_to_retrieval`` : max(run3_to_retrieval, run4_to_retrieval)

其中 ``L_i^run_k`` 是 learning 阶段 run=k 中 pair_id=i 的平均 pattern；
``R_i`` 是 retrieval 阶段 pair_id=i 的平均 pattern（yy/kj 分条件记录）。

组水平：
- 对每种 ERS 变体 × (yy / kj / yy−kj) 做 one-sample t，BH-FDR within ROI set。

Item-level GEE：
- ``memory_i ~ ERS_i * condition + (1|subject)``（Binomial for memory accuracy）。
- ``log_rt_correct_i ~ ERS_i * condition + (1|subject)``（Gaussian，仅 memory=1）。

输入：``pattern_root/sub-XX/{learn,retrieval}_{yy,kj}.nii.gz + *_metadata.tsv``
（learn 含 ``run ∈ {3,4}``、``pair_id``；retrieval 含 ``pair_id``、``memory``、``action_time``）。

输出：``$paper/qc/encoding_retrieval_similarity_{roi_tag}/``
- ers_item.tsv
- ers_group_one_sample.tsv
- ers_itemlevel_gee.tsv
- ers_qc.tsv
- ers_manifest.json
以及 ``$paper/tables_main/table_encoding_retrieval_similarity_{roi_tag}.tsv``。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
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

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag, select_roi_masks  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
LEARN_RUNS = [3, 4]
ERS_VARIANTS = ["run3_to_retrieval", "run4_to_retrieval", "max_run_to_retrieval"]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


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


def _cohens_dz(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    a = a[valid] - np.nanmean(a[valid])
    b = b[valid] - np.nanmean(b[valid])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if math.isclose(denom, 0.0):
        return float("nan")
    return float(np.dot(a, b) / denom)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def _masked_samples(
    data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path
) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _load_stage(
    subject_dir: Path, stage: str, condition: str
) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"{stage}_{condition}.nii.gz"
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {stage}_{condition} image or metadata for {subject_dir.name}"
        )
    meta = _read_any(meta_path).reset_index(drop=True)
    _, data = load_4d_data(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(
            f"Volume/metadata mismatch: {image_path} has {data.shape[3]} vols, meta has {len(meta)}"
        )
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["stage"] = stage
    if "pair_id" in meta.columns:
        meta["pair_id"] = meta["pair_id"].astype(str).str.strip()
    if "word_label" in meta.columns:
        meta["word_label"] = meta["word_label"].astype(str).str.strip()
    if stage == "learn" and "run" in meta.columns:
        meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    if stage == "retrieval":
        for col in ("memory", "action_time"):
            if col in meta.columns:
                meta[col] = pd.to_numeric(meta[col], errors="coerce")
    return meta, data, image_path


def _mean_pattern(samples: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.full(samples.shape[1], np.nan, dtype=np.float64)
    return np.nanmean(samples[indices, :], axis=0)


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
    retrieval_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject = subject_dir.name
    _, mask = load_mask(mask_path)
    item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        try:
            l_meta, l_data, l_image = learn_cache[condition]
            learn_samples = _masked_samples(l_data, mask, l_image, mask_path)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": f"learn: {exc}",
                    "n_pairs_with_all_three": 0,
                }
            )
            continue

        try:
            r_meta, r_data, r_image = retrieval_cache[condition]
            retrieval_samples = _masked_samples(r_data, mask, r_image, mask_path)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": f"retrieval: {exc}",
                    "n_pairs_with_all_three": 0,
                }
            )
            continue

        learn_pair_ids = l_meta["pair_id"].astype(str)
        retrieval_pair_ids = r_meta["pair_id"].astype(str)
        shared_pairs = sorted(
            set(learn_pair_ids.unique())
            & set(retrieval_pair_ids.unique())
            - {"", "nan"}
        )

        n_all_three = 0
        for pair_id in shared_pairs:
            if not pair_id or pair_id.lower() == "nan":
                continue

            run3_idx = np.flatnonzero(
                (learn_pair_ids == pair_id).to_numpy(dtype=bool)
                & (l_meta["run"].fillna(-1).astype(int) == 3).to_numpy(dtype=bool)
            )
            run4_idx = np.flatnonzero(
                (learn_pair_ids == pair_id).to_numpy(dtype=bool)
                & (l_meta["run"].fillna(-1).astype(int) == 4).to_numpy(dtype=bool)
            )
            ret_idx = np.flatnonzero(
                (retrieval_pair_ids == pair_id).to_numpy(dtype=bool)
            )

            if ret_idx.size == 0 or (run3_idx.size == 0 and run4_idx.size == 0):
                continue

            retrieval_pattern = _mean_pattern(retrieval_samples, ret_idx)
            run3_pattern = _mean_pattern(learn_samples, run3_idx)
            run4_pattern = _mean_pattern(learn_samples, run4_idx)

            ers_run3 = (
                _corr(run3_pattern, retrieval_pattern)
                if run3_idx.size > 0
                else float("nan")
            )
            ers_run4 = (
                _corr(run4_pattern, retrieval_pattern)
                if run4_idx.size > 0
                else float("nan")
            )
            finite_variants = [
                v for v in (ers_run3, ers_run4) if np.isfinite(v)
            ]
            ers_max = float(max(finite_variants)) if finite_variants else float("nan")

            if run3_idx.size > 0 and run4_idx.size > 0 and ret_idx.size > 0:
                n_all_three += 1

            # retrieval-level behavior: take first retrieval trial's behavior.
            ret_first = int(ret_idx[0])
            memory = r_meta.iloc[ret_first].get("memory", np.nan)
            rt = r_meta.iloc[ret_first].get("action_time", np.nan)
            word_label = str(r_meta.iloc[ret_first].get("word_label", "")).strip()

            for variant_name, value in (
                ("run3_to_retrieval", ers_run3),
                ("run4_to_retrieval", ers_run4),
                ("max_run_to_retrieval", ers_max),
            ):
                item_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "pair_id": pair_id,
                        "word_label": word_label,
                        "variant": variant_name,
                        "ers": float(value) if np.isfinite(value) else float("nan"),
                        "memory": float(memory) if pd.notna(memory) else float("nan"),
                        "action_time": float(rt) if pd.notna(rt) else float("nan"),
                        "n_run3_trials": int(run3_idx.size),
                        "n_run4_trials": int(run4_idx.size),
                        "n_retrieval_trials": int(ret_idx.size),
                    }
                )

        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi_name,
                "condition": condition,
                "ok": True,
                "fail_reason": "",
                "n_pairs_with_all_three": int(n_all_three),
                "n_shared_pairs": int(len(shared_pairs)),
            }
        )

    return item_rows, qc_rows


def summarize_group_one_sample(item_metrics: pd.DataFrame) -> pd.DataFrame:
    """每 ROI × condition × variant 做 one-sample t；并额外计算 yy−kj 对比。"""
    if item_metrics.empty:
        return pd.DataFrame()

    # subject-level mean ERS per (subject, roi_set, roi, condition, variant).
    subject_mean = (
        item_metrics.dropna(subset=["ers"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "variant"],
            dropna=False,
        )["ers"]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for keys, subset in subject_mean.groupby(
        ["roi_set", "roi", "condition", "variant"], sort=False
    ):
        roi_set, roi, condition, variant = keys
        values = pd.to_numeric(subset["ers"], errors="coerce").dropna()
        if values.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(values.to_numpy(dtype=float), 0.0, nan_policy="omit")
        rows.append(
            {
                "analysis_type": "ers_one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "variant": variant,
                "n_subjects": int(values.size),
                "mean_ers": float(values.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(values),
            }
        )

    # yy − kj 差异（paired per subject）
    if not subject_mean.empty:
        pivot = subject_mean.pivot_table(
            index=["subject", "roi_set", "roi", "variant"],
            columns="condition",
            values="ers",
            aggfunc="mean",
        ).reset_index()
        if "yy" in pivot.columns and "kj" in pivot.columns:
            pivot["diff_yy_minus_kj"] = pivot["yy"] - pivot["kj"]
            for keys, subset in pivot.groupby(
                ["roi_set", "roi", "variant"], sort=False
            ):
                roi_set, roi, variant = keys
                values = pd.to_numeric(
                    subset["diff_yy_minus_kj"], errors="coerce"
                ).dropna()
                if values.size < 2:
                    continue
                t_val, p_val = stats.ttest_1samp(
                    values.to_numpy(dtype=float), 0.0, nan_policy="omit"
                )
                rows.append(
                    {
                        "analysis_type": "ers_yy_minus_kj",
                        "roi_set": roi_set,
                        "roi": roi,
                        "condition": "yy_minus_kj",
                        "condition_label": "YY-KJ",
                        "variant": variant,
                        "n_subjects": int(values.size),
                        "mean_ers": float(values.mean()),
                        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                        "cohens_dz": _cohens_dz(values),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(
        ["roi_set", "variant", "analysis_type"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_within_family", "p"], na_position="last")


def _fit_gee(
    frame: pd.DataFrame, response: str
) -> dict[str, object] | None:
    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial, Gaussian
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "error": f"statsmodels missing: {exc}"}

    work = frame.dropna(subset=["ers", response, "subject", "condition"]).copy()
    if work["subject"].nunique() < 8 or work[response].nunique() < 2 or len(work) < 30:
        return None
    work["ers_z"] = _zscore(work["ers"])
    work["condition_yy"] = (work["condition"].astype(str) == "yy").astype(int)
    family = Binomial() if response == "memory" else Gaussian()
    try:
        model = GEE.from_formula(
            f"{response} ~ ers_z * condition_yy",
            groups="subject",
            data=work,
            family=family,
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "n_trials": int(len(work)),
            "n_subjects": int(work["subject"].nunique()),
        }
    params = fit.params
    pvalues = fit.pvalues
    bse = fit.bse
    return {
        "status": "ok",
        "n_trials": int(len(work)),
        "n_subjects": int(work["subject"].nunique()),
        "ers_beta": float(params.get("ers_z", np.nan)),
        "ers_se": float(bse.get("ers_z", np.nan)),
        "ers_p": float(pvalues.get("ers_z", np.nan)),
        "condition_beta": float(params.get("condition_yy", np.nan)),
        "condition_p": float(pvalues.get("condition_yy", np.nan)),
        "interaction_beta": float(params.get("ers_z:condition_yy", np.nan)),
        "interaction_p": float(pvalues.get("ers_z:condition_yy", np.nan)),
        "family": "binomial" if response == "memory" else "gaussian",
    }


def summarize_itemlevel_gee(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty:
        return pd.DataFrame()
    frame = item_metrics.copy()
    frame["log_rt_correct"] = np.where(
        frame["memory"].eq(1) & frame["action_time"].gt(0),
        np.log(frame["action_time"]),
        np.nan,
    )

    rows: list[dict[str, object]] = []
    for keys, subset in frame.groupby(["roi_set", "roi", "variant"], sort=False):
        roi_set, roi, variant = keys
        for response in ("memory", "log_rt_correct"):
            fit = _fit_gee(subset, response)
            if fit is None:
                continue
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "variant": variant,
                    "response": response,
                    **fit,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ok = out["status"].eq("ok")
    out["q_bh_within_family"] = np.nan
    for _, idx in out[ok].groupby(
        ["roi_set", "variant", "response"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "ers_p"])
    return out.sort_values(
        ["q_bh_within_family", "ers_p"], na_position="last"
    )


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Item-level Encoding-Retrieval Similarity (ERS) + memory bridge.",
    )
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument(
        "--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv"
    )
    parser.add_argument(
        "--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"]
    )
    parser.add_argument(
        "--paper-output-root", type=Path, default=base_dir / "paper_outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. If not set, uses default ROI-tagged path.",
    )
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(current_roi_set())
    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
    else:
        output_dir = ensure_dir(
            args.paper_output_root / "qc" / f"encoding_retrieval_similarity_{roi_tag}"
        )
    tables_main = ensure_dir(args.paper_output_root / "tables_main")

    roi_masks_by_set = {
        roi_set: select_roi_masks(
            args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa"
        )
        for roi_set in args.roi_sets
    }

    subjects = sorted(
        [
            path
            for path in args.pattern_root.glob("sub-*")
            if path.is_dir()
            and all((path / f"learn_{c}.nii.gz").exists() for c in CONDITIONS)
            and all((path / f"retrieval_{c}.nii.gz").exists() for c in CONDITIONS)
        ]
    )

    item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        retrieval_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        load_error: str | None = None
        try:
            for condition in CONDITIONS:
                learn_cache[condition] = _load_stage(subject_dir, "learn", condition)
                retrieval_cache[condition] = _load_stage(
                    subject_dir, "retrieval", condition
                )
        except Exception as exc:
            load_error = str(exc)
            failure_rows.append(
                {
                    "subject": subject_dir.name,
                    "roi_set": "all",
                    "roi": "all",
                    "error": load_error,
                    "traceback": traceback.format_exc(),
                }
            )
        if load_error is not None:
            continue

        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                try:
                    items, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        learn_cache,
                        retrieval_cache,
                    )
                    item_rows.extend(items)
                    qc_rows.extend(qc)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

    item_metrics = pd.DataFrame(item_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group_one_sample = summarize_group_one_sample(item_metrics)
    group_gee = summarize_itemlevel_gee(item_metrics)

    write_table(item_metrics, output_dir / "ers_item.tsv")
    write_table(group_one_sample, output_dir / "ers_group_one_sample.tsv")
    write_table(group_gee, output_dir / "ers_itemlevel_gee.tsv")
    write_table(qc, output_dir / "ers_qc.tsv")
    write_table(failures, output_dir / "ers_failures.tsv")
    write_table(
        group_one_sample,
        tables_main / f"table_encoding_retrieval_similarity_{roi_tag}.tsv",
    )

    save_json(
        {
            "roi_sets": args.roi_sets,
            "roi_tag": roi_tag,
            "variants": ERS_VARIANTS,
            "learn_runs": LEARN_RUNS,
            "n_subjects": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_item_rows": int(len(item_metrics)),
            "n_gee_rows": int(len(group_gee)),
            "n_failures": int(len(failures)),
            "output_dir": str(output_dir),
            "gate_g1_description": (
                "Gate G1 passes if at least one ROI in meta_metaphor/meta_spatial has "
                "group-level q_bh_within_family < 0.1 for run3_to_retrieval or "
                "run4_to_retrieval ERS one-sample t (see ers_group_one_sample.tsv)."
            ),
        },
        output_dir / "ers_manifest.json",
    )


if __name__ == "__main__":
    main()
