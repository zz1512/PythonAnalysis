#!/usr/bin/env python3
"""Learning-stage subsequent-memory analysis.

对每个 subject x ROI x pair_id:
- 使用 learning 阶段 `learn_{yy,kj}.nii.gz` 的 metadata 中 `run` / `pair_id`
  对齐 run3 与 run4；
- 计算 `self_similarity_run3_run4 = corr(L_i^run3, L_i^run4)`；
- 将该 pair-level fidelity 与 run7 行为表对齐；
- 输出 item / group one-sample / item-level GEE / QC。

默认输出目录：
`$paper_outputs/qc/learning_subsequent_memory_{roi_tag}/`

核心输出：
- `learning_subsequent_memory_item.tsv`
- `learning_subsequent_memory_group_one_sample.tsv`
- `learning_subsequent_memory_itemlevel_gee.tsv`
- `learning_subsequent_memory_qc.tsv`
- `learning_subsequent_memory_failures.tsv`
- `learning_subsequent_memory_manifest.json`
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback

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
from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag, select_roi_masks  # noqa: E402
from common.stimulus_text_mapping import normalize_stimulus_label  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
BEHAVIOR_CONDITION_MAP = {
    "YY": "yy",
    "KJ": "kj",
    "yy": "yy",
    "kj": "kj",
    "Metaphor": "yy",
    "Spatial": "kj",
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


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


def _normalize_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        numeric = float(text)
    except Exception:
        return text
    if math.isnan(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _masked_samples(
    data: np.ndarray,
    mask: np.ndarray,
    image_path: Path,
    mask_path: Path,
) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _load_learn_condition(
    subject_dir: Path,
    condition: str,
) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"learn_{condition}.nii.gz"
    meta_path = subject_dir / f"learn_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing learn_{condition} image or metadata for {subject_dir.name}"
        )
    meta = read_table(meta_path).reset_index(drop=True)
    if "run" not in meta.columns or "pair_id" not in meta.columns:
        raise ValueError(
            f"{meta_path.name} must contain both 'run' and 'pair_id' columns."
        )
    _, data = load_4d_data(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(
            f"Volume/metadata mismatch: {image_path} has {data.shape[3]} vols, "
            f"metadata has {len(meta)} rows."
        )
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    meta["pair_id"] = meta["pair_id"].map(_normalize_pair_id)
    if "word_label" in meta.columns:
        meta["word_label"] = meta["word_label"].astype(str).str.strip()
    return meta, data, image_path


def _aggregate_run_patterns(
    samples: np.ndarray,
    meta: pd.DataFrame,
    run: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], int]:
    subset = meta.loc[meta["run"].eq(run)].copy()
    subset = subset[subset["pair_id"].astype(str).str.strip().ne("")]
    pair_patterns: dict[str, np.ndarray] = {}
    pair_info: dict[str, dict[str, object]] = {}
    if subset.empty:
        return pair_patterns, pair_info, 0

    row_indices = subset.index.to_numpy(dtype=int)
    work = subset.copy()
    work["_row_index"] = row_indices
    duplicate_pair_count = int(work["pair_id"].duplicated(keep=False).sum())

    for pair_id, group in work.groupby("pair_id", sort=True):
        idx = group["_row_index"].to_numpy(dtype=int)
        pair_patterns[pair_id] = np.nanmean(samples[idx, :], axis=0)
        first = group.iloc[0]
        pair_info[pair_id] = {
            "pair_id": pair_id,
            "n_rows": int(len(group)),
            "word_label_example": str(first.get("word_label", "")).strip(),
        }
    return pair_patterns, pair_info, duplicate_pair_count


def _prepare_template(template_path: Path | None) -> pd.DataFrame:
    if template_path is None or not template_path.exists():
        return pd.DataFrame(
            columns=["condition", "pair_id", "word_label", "normalized_word_label"]
        )
    frame = read_table(template_path).copy()
    required = {"condition", "pair_id", "word_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=["condition", "pair_id", "word_label", "normalized_word_label"]
        )
    frame["condition"] = frame["condition"].map(BEHAVIOR_CONDITION_MAP).fillna("")
    frame = frame[frame["condition"].isin(CONDITIONS)].copy()
    frame["pair_id"] = frame["pair_id"].map(_normalize_pair_id)
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    frame["normalized_word_label"] = frame["word_label"].map(normalize_stimulus_label)
    frame = frame[
        frame["pair_id"].ne("") & frame["normalized_word_label"].ne("")
    ].copy()
    # Drop ambiguous label -> pair mappings within the same condition.
    counts = frame.groupby(["condition", "normalized_word_label"])["pair_id"].nunique()
    unique_keys = counts[counts == 1].index
    if len(unique_keys) == 0:
        return pd.DataFrame(
            columns=["condition", "pair_id", "word_label", "normalized_word_label"]
        )
    key_frame = pd.DataFrame(unique_keys.tolist(), columns=["condition", "normalized_word_label"])
    frame = frame.merge(
        key_frame,
        on=["condition", "normalized_word_label"],
        how="inner",
        validate="many_to_many",
    )
    frame = frame.drop_duplicates(
        subset=["condition", "normalized_word_label"], keep="first"
    )
    return frame[
        ["condition", "pair_id", "word_label", "normalized_word_label"]
    ].reset_index(drop=True)


def _prepare_behavior_trials(
    behavior_trials_path: Path,
    template: pd.DataFrame,
) -> pd.DataFrame:
    frame = read_table(behavior_trials_path).copy()
    if "condition" not in frame.columns and "trial_type" in frame.columns:
        frame["condition"] = frame["trial_type"]
    frame["condition"] = frame["condition"].map(BEHAVIOR_CONDITION_MAP)
    frame = frame[frame["condition"].isin(CONDITIONS)].copy()
    if frame.empty:
        return frame

    frame["subject"] = frame["subject"].astype(str).str.strip()
    frame["memory"] = pd.to_numeric(frame.get("memory"), errors="coerce")
    frame["action_time"] = pd.to_numeric(frame.get("action_time"), errors="coerce")

    if "log_rt_correct" in frame.columns:
        frame["log_rt_correct"] = pd.to_numeric(frame["log_rt_correct"], errors="coerce")
    else:
        frame["log_rt_correct"] = np.where(
            frame["memory"].eq(1) & frame["action_time"].gt(0),
            np.log(frame["action_time"]),
            np.nan,
        )

    frame["behavior_item_label"] = (
        frame.get("pic_out", pd.Series(index=frame.index, dtype=object))
        .fillna(frame.get("item", pd.Series(index=frame.index, dtype=object)))
        .astype(str)
        .str.strip()
    )
    frame["normalized_word_label"] = frame["behavior_item_label"].map(
        normalize_stimulus_label
    )

    native_pair = (
        frame["pair_id"].map(_normalize_pair_id)
        if "pair_id" in frame.columns
        else pd.Series("", index=frame.index, dtype=object)
    )
    frame["pair_id_native"] = native_pair

    if not template.empty:
        tpl = template.rename(
            columns={
                "pair_id": "pair_id_template",
                "word_label": "template_word_label",
            }
        )
        frame = frame.merge(
            tpl,
            on=["condition", "normalized_word_label"],
            how="left",
            validate="many_to_one",
        )
    else:
        frame["pair_id_template"] = ""
        frame["template_word_label"] = ""

    frame["pair_id_template"] = frame["pair_id_template"].map(_normalize_pair_id)
    frame["pair_id"] = np.where(
        frame["pair_id_native"].astype(str).str.strip().ne(""),
        frame["pair_id_native"],
        frame["pair_id_template"],
    )
    frame["pair_id"] = pd.Series(frame["pair_id"], index=frame.index).map(
        _normalize_pair_id
    )
    frame["pair_id_source"] = np.where(
        frame["pair_id_native"].astype(str).str.strip().ne(""),
        "behavior_native",
        np.where(
            frame["pair_id_template"].astype(str).str.strip().ne(""),
            "stimuli_template",
            "missing",
        ),
    )
    frame["condition_label"] = frame["condition"].map(CONDITION_LABELS)
    frame["_behavior_row_id"] = np.arange(len(frame), dtype=int)
    return frame.reset_index(drop=True)


def _build_behavior_pair_lookup(behavior: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    if behavior.empty:
        return {}
    valid = behavior[behavior["pair_id"].astype(str).str.strip().ne("")].copy()
    if valid.empty:
        return {}
    summary = (
        valid.groupby(["subject", "condition", "pair_id"], dropna=False)
        .agg(
            n_behavior_trials=("_behavior_row_id", "count"),
            n_behavior_missing_memory=("memory", lambda x: int(pd.isna(x).sum())),
            n_behavior_nonmissing_memory=("memory", lambda x: int(pd.notna(x).sum())),
            n_behavior_nonmissing_log_rt_correct=("log_rt_correct", lambda x: int(pd.notna(x).sum())),
        )
        .reset_index()
    )
    lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for (subject, condition), subset in summary.groupby(["subject", "condition"], sort=False):
        lookup[(str(subject), str(condition))] = subset.reset_index(drop=True)
    return lookup


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
    behavior_lookup: dict[tuple[str, str], pd.DataFrame],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject = subject_dir.name
    _, mask = load_mask(mask_path)
    item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        meta, data, image_path = learn_cache[condition]
        try:
            samples = _masked_samples(data, mask, image_path, mask_path)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "ok": False,
                    "fail_reason": f"masking failed: {exc}",
                    "n_run3_rows": 0,
                    "n_run4_rows": 0,
                    "n_run3_pairs": 0,
                    "n_run4_pairs": 0,
                    "n_shared_pairs": 0,
                    "n_duplicate_rows_run3": 0,
                    "n_duplicate_rows_run4": 0,
                    "n_behavior_pairs": 0,
                    "n_behavior_trials": 0,
                    "n_behavior_pairs_joined": 0,
                    "n_behavior_trials_joined": 0,
                    "n_neural_pairs_without_behavior": 0,
                    "n_behavior_pairs_without_neural": 0,
                    "n_behavior_missing_memory": 0,
                    "n_behavior_nonmissing_log_rt_correct": 0,
                }
            )
            continue

        run3_patterns, run3_info, dup3 = _aggregate_run_patterns(samples, meta, 3)
        run4_patterns, run4_info, dup4 = _aggregate_run_patterns(samples, meta, 4)
        shared_pairs = sorted(set(run3_patterns) & set(run4_patterns))

        for pair_id in shared_pairs:
            sim = _corr(run3_patterns[pair_id], run4_patterns[pair_id])
            row_info = run3_info.get(pair_id, {})
            item_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "pair_id": pair_id,
                    "word_label_example": str(row_info.get("word_label_example", "")).strip(),
                    "n_run3_rows_for_pair": int(row_info.get("n_rows", 0)),
                    "n_run4_rows_for_pair": int(run4_info.get(pair_id, {}).get("n_rows", 0)),
                    "self_similarity_run3_run4": float(sim)
                    if np.isfinite(sim)
                    else float("nan"),
                }
            )

        behavior_pairs = behavior_lookup.get((subject, condition), pd.DataFrame())
        behavior_pair_ids = set(
            behavior_pairs["pair_id"].astype(str).tolist()
        ) if not behavior_pairs.empty else set()
        shared_pair_set = set(shared_pairs)
        matched_pairs = shared_pair_set & behavior_pair_ids

        n_behavior_trials_joined = 0
        n_behavior_missing_memory = 0
        n_behavior_nonmissing_log_rt_correct = 0
        if not behavior_pairs.empty and matched_pairs:
            subset = behavior_pairs[behavior_pairs["pair_id"].isin(sorted(matched_pairs))]
            n_behavior_trials_joined = int(subset["n_behavior_trials"].sum())
            n_behavior_missing_memory = int(subset["n_behavior_missing_memory"].sum())
            n_behavior_nonmissing_log_rt_correct = int(
                subset["n_behavior_nonmissing_log_rt_correct"].sum()
            )

        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi_name,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "ok": True,
                "fail_reason": "",
                "n_run3_rows": int(meta["run"].eq(3).sum()),
                "n_run4_rows": int(meta["run"].eq(4).sum()),
                "n_run3_pairs": int(len(run3_patterns)),
                "n_run4_pairs": int(len(run4_patterns)),
                "n_shared_pairs": int(len(shared_pairs)),
                "n_duplicate_rows_run3": int(dup3),
                "n_duplicate_rows_run4": int(dup4),
                "n_behavior_pairs": int(len(behavior_pair_ids)),
                "n_behavior_trials": int(behavior_pairs["n_behavior_trials"].sum())
                if not behavior_pairs.empty
                else 0,
                "n_behavior_pairs_joined": int(len(matched_pairs)),
                "n_behavior_trials_joined": int(n_behavior_trials_joined),
                "n_neural_pairs_without_behavior": int(len(shared_pair_set - behavior_pair_ids)),
                "n_behavior_pairs_without_neural": int(len(behavior_pair_ids - shared_pair_set)),
                "n_behavior_missing_memory": int(n_behavior_missing_memory),
                "n_behavior_nonmissing_log_rt_correct": int(
                    n_behavior_nonmissing_log_rt_correct
                ),
            }
        )

    return item_rows, qc_rows


def summarize_group_one_sample(neural_item: pd.DataFrame) -> pd.DataFrame:
    if neural_item.empty:
        return pd.DataFrame()

    subject_mean = (
        neural_item.dropna(subset=["self_similarity_run3_run4"])
        .groupby(["subject", "roi_set", "roi", "condition", "condition_label"], dropna=False)[
            "self_similarity_run3_run4"
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, subset in subject_mean.groupby(
        ["roi_set", "roi", "condition", "condition_label"], sort=False
    ):
        roi_set, roi, condition, condition_label = keys
        values = pd.to_numeric(
            subset["self_similarity_run3_run4"], errors="coerce"
        ).dropna()
        if values.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(
            values.to_numpy(dtype=float),
            0.0,
            nan_policy="omit",
        )
        rows.append(
            {
                "analysis_type": "fidelity_one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "predictor": "self_similarity_run3_run4",
                "n_subjects": int(values.size),
                "mean_fidelity": float(values.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "condition"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_within_family", "p"], na_position="last")


def _fit_gee(frame: pd.DataFrame, response: str) -> dict[str, object] | None:
    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial, Gaussian
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "error": f"statsmodels missing: {exc}"}

    work = frame.dropna(
        subset=["self_similarity_run3_run4", response, "subject", "condition"]
    ).copy()
    if work["subject"].nunique() < 8 or len(work) < 30 or work[response].nunique() < 2:
        return None

    work["self_similarity_z"] = _zscore(work["self_similarity_run3_run4"])
    work["condition_yy"] = (work["condition"].astype(str) == "yy").astype(int)
    family = Binomial() if response == "memory" else Gaussian()

    try:
        model = GEE.from_formula(
            f"{response} ~ self_similarity_z * condition_yy",
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
    bse = fit.bse
    pvalues = fit.pvalues
    return {
        "status": "ok",
        "family": "binomial" if response == "memory" else "gaussian",
        "n_trials": int(len(work)),
        "n_subjects": int(work["subject"].nunique()),
        "self_similarity_beta": float(params.get("self_similarity_z", np.nan)),
        "self_similarity_se": float(bse.get("self_similarity_z", np.nan)),
        "self_similarity_p": float(pvalues.get("self_similarity_z", np.nan)),
        "condition_beta": float(params.get("condition_yy", np.nan)),
        "condition_se": float(bse.get("condition_yy", np.nan)),
        "condition_p": float(pvalues.get("condition_yy", np.nan)),
        "interaction_beta": float(params.get("self_similarity_z:condition_yy", np.nan)),
        "interaction_se": float(bse.get("self_similarity_z:condition_yy", np.nan)),
        "interaction_p": float(pvalues.get("self_similarity_z:condition_yy", np.nan)),
    }


def summarize_itemlevel_gee(item_with_behavior: pd.DataFrame) -> pd.DataFrame:
    if item_with_behavior.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in item_with_behavior.groupby(["roi_set", "roi"], sort=False):
        roi_set, roi = keys
        for response in ("memory", "log_rt_correct"):
            fit = _fit_gee(subset, response)
            if fit is None:
                continue
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "predictor": "self_similarity_run3_run4",
                    "response": response,
                    **fit,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ok = out["status"].eq("ok")
    out["q_bh_within_family"] = np.nan
    out["interaction_q_bh_within_family"] = np.nan
    for _, idx in out[ok].groupby(["roi_set", "response"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(
            out.loc[idx, "self_similarity_p"]
        )
        out.loc[idx, "interaction_q_bh_within_family"] = _bh_fdr(
            out.loc[idx, "interaction_p"]
        )
    return out.sort_values(
        ["q_bh_within_family", "self_similarity_p"],
        na_position="last",
    )


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning-stage run3/run4 repetition fidelity -> run7 behavior.",
    )
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=base_dir / "roi_library" / "manifest.tsv",
    )
    parser.add_argument(
        "--roi-sets",
        nargs="+",
        default=["meta_metaphor", "meta_spatial"],
    )
    parser.add_argument(
        "--behavior-trials",
        type=Path,
        default=base_dir
        / "paper_outputs"
        / "qc"
        / "behavior_results"
        / "refined"
        / "behavior_trials.tsv",
    )
    parser.add_argument(
        "--stimuli-template",
        type=Path,
        default=base_dir / "stimuli_template.csv",
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=base_dir / "paper_outputs",
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
            args.paper_output_root / "qc" / f"learning_subsequent_memory_{roi_tag}"
        )

    template = _prepare_template(args.stimuli_template)
    behavior = _prepare_behavior_trials(args.behavior_trials, template)
    behavior_lookup = _build_behavior_pair_lookup(behavior)

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
            and all((path / f"learn_{condition}.nii.gz").exists() for condition in CONDITIONS)
            and all(
                (path / f"learn_{condition}_metadata.tsv").exists()
                for condition in CONDITIONS
            )
        ]
    )

    neural_item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        load_error: str | None = None
        try:
            for condition in CONDITIONS:
                learn_cache[condition] = _load_learn_condition(subject_dir, condition)
        except Exception as exc:
            load_error = str(exc)
            failure_rows.append(
                {
                    "subject": subject_dir.name,
                    "roi_set": "all",
                    "roi": "all",
                    "condition": "",
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
                        behavior_lookup,
                    )
                    neural_item_rows.extend(items)
                    qc_rows.extend(qc)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": "",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

    neural_item = pd.DataFrame(neural_item_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)

    if not behavior.empty and not neural_item.empty:
        item_with_behavior = behavior.merge(
            neural_item,
            on=["subject", "condition", "condition_label", "pair_id"],
            how="inner",
            validate="many_to_many",
        )
    else:
        item_with_behavior = pd.DataFrame()

    group_one_sample = summarize_group_one_sample(neural_item)
    itemlevel_gee = summarize_itemlevel_gee(item_with_behavior)

    write_table(item_with_behavior, output_dir / "learning_subsequent_memory_item.tsv")
    write_table(
        group_one_sample,
        output_dir / "learning_subsequent_memory_group_one_sample.tsv",
    )
    write_table(
        itemlevel_gee,
        output_dir / "learning_subsequent_memory_itemlevel_gee.tsv",
    )
    write_table(qc, output_dir / "learning_subsequent_memory_qc.tsv")
    write_table(failures, output_dir / "learning_subsequent_memory_failures.tsv")

    save_json(
        {
            "roi_sets": args.roi_sets,
            "roi_tag": roi_tag,
            "n_subjects_with_learning_patterns": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_behavior_rows": int(len(behavior)),
            "n_behavior_rows_with_pair_id": int(
                behavior["pair_id"].astype(str).str.strip().ne("").sum()
            )
            if not behavior.empty
            else 0,
            "n_neural_item_rows": int(len(neural_item)),
            "n_item_rows_after_behavior_join": int(len(item_with_behavior)),
            "n_group_rows": int(len(group_one_sample)),
            "n_gee_rows": int(len(itemlevel_gee)),
            "n_qc_rows": int(len(qc)),
            "n_failures": int(len(failures)),
            "pattern_root": str(args.pattern_root),
            "behavior_trials": str(args.behavior_trials),
            "stimuli_template": str(args.stimuli_template),
            "roi_manifest": str(args.roi_manifest),
            "output_dir": str(output_dir),
            "outputs": {
                "item": str(output_dir / "learning_subsequent_memory_item.tsv"),
                "group": str(
                    output_dir / "learning_subsequent_memory_group_one_sample.tsv"
                ),
                "gee": str(
                    output_dir / "learning_subsequent_memory_itemlevel_gee.tsv"
                ),
                "qc": str(output_dir / "learning_subsequent_memory_qc.tsv"),
            },
        },
        output_dir / "learning_subsequent_memory_manifest.json",
    )


if __name__ == "__main__":
    main()
