#!/usr/bin/env python3
"""Shared utilities for reviewer supplementary analyses.

Safety policy:
- read upstream data only;
- write only under ``paper_outputs/qc/reviewer_supp``;
- never overwrite an existing file unless a caller explicitly asks for a
  unique suffixed file through ``unique=True``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception:  # pragma: no cover - dependency is available on analysis host
    smf = None
    sm = None


def final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import select_roi_masks  # noqa: E402

CONDITIONS = ("yy", "kj")
PHASE_TO_STAGE = {"pre": "pre", "learning": "learn", "post": "post", "retrieval": "retrieval"}
SAFE_RELATIVE_ROOT = Path("paper_outputs") / "qc" / "reviewer_supp"


@dataclass(frozen=True)
class AnalysisConfig:
    base_dir: Path
    pattern_root: Path
    paper_output_root: Path
    output_root: Path
    roi_manifest: Path
    roi_sets: tuple[str, ...]


def default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def default_config(args: argparse.Namespace) -> AnalysisConfig:
    base_dir = Path(args.base_dir)
    paper_output_root = Path(args.paper_output_root or base_dir / "paper_outputs")
    output_root = Path(args.output_root or paper_output_root / "qc" / "reviewer_supp")
    cfg = AnalysisConfig(
        base_dir=base_dir,
        pattern_root=Path(args.pattern_root or base_dir / "pattern_root"),
        paper_output_root=paper_output_root,
        output_root=output_root,
        roi_manifest=Path(args.roi_manifest or base_dir / "roi_library" / "manifest.tsv"),
        roi_sets=tuple(args.roi_sets),
    )
    ensure_reviewer_root(cfg)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> None:
    base = default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_reviewer_root(cfg: AnalysisConfig) -> Path:
    root = _resolved(cfg.output_root)
    expected = _resolved(cfg.paper_output_root / "qc" / "reviewer_supp")
    if root != expected and expected not in root.parents:
        raise ValueError(
            f"Refusing output outside reviewer_supp sandbox: {root}; expected under {expected}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def module_dir(cfg: AnalysisConfig, module_name: str) -> Path:
    target = _resolved(cfg.output_root / module_name)
    sandbox = _resolved(cfg.output_root)
    if target != sandbox and sandbox not in target.parents:
        raise ValueError(f"Refusing module output outside sandbox: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_output_path(cfg: AnalysisConfig, module_name: str, filename: str, *, unique: bool = False) -> Path:
    directory = module_dir(cfg, module_name)
    path = _resolved(directory / filename)
    if directory not in path.parents:
        raise ValueError(f"Refusing path outside module directory: {path}")
    if path.exists():
        if not unique:
            raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
        stem, suffix = path.stem, path.suffix
        index = 1
        while path.exists():
            path = directory / f"{stem}_{index:03d}{suffix}"
            index += 1
    return path


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame.to_csv(path, sep=sep, index=False)


def write_text(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required upstream table not found: {path}")
    return read_table(path)


def normalize_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        val = float(text)
    except Exception:
        return text
    if math.isnan(val):
        return ""
    return str(int(val)) if val.is_integer() else str(val)


def make_condition_item_id(condition: object, pair_id: object) -> str:
    pair = normalize_pair_id(pair_id)
    cond = str(condition).strip().lower()
    if not cond or not pair:
        return ""
    return f"{cond}_{pair}"


def ensure_condition_item_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "condition_item_id" in out.columns:
        out["condition_item_id"] = out["condition_item_id"].astype(str)
        return out
    if {"condition", "pair_id"}.issubset(out.columns):
        out["condition_item_id"] = [
            make_condition_item_id(condition, pair_id)
            for condition, pair_id in zip(out["condition"], out["pair_id"])
        ]
    elif {"condition", "original_pair_id"}.issubset(out.columns):
        out["condition_item_id"] = [
            make_condition_item_id(condition, pair_id)
            for condition, pair_id in zip(out["condition"], out["original_pair_id"])
        ]
    return out


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    values = pd.to_numeric(pd.Series(pvalues), errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out_indices = np.flatnonzero(valid)[order]
    out[out_indices] = np.clip(adjusted, 0.0, 1.0)
    return out


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    av = a[valid] - np.nanmean(a[valid])
    bv = b[valid] - np.nanmean(b[valid])
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if math.isclose(denom, 0.0):
        return float("nan")
    return float(np.dot(av, bv) / denom)


def subject_dirs(pattern_root: Path, required_stages: Iterable[str]) -> list[Path]:
    if not pattern_root.exists():
        return []
    stages = tuple(required_stages)
    out = []
    for path in sorted(pattern_root.glob("sub-*")):
        if not path.is_dir():
            continue
        ok = True
        for stage in stages:
            for cond in CONDITIONS:
                if not (path / f"{stage}_{cond}.nii.gz").exists() or not (path / f"{stage}_{cond}_metadata.tsv").exists():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(path)
    return out


def load_stage(subject_dir: Path, stage: str, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"{stage}_{condition}.nii.gz"
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {stage}_{condition} image or metadata for {subject_dir.name}")
    meta = read_table(meta_path).reset_index(drop=True)
    _, data = load_4d_data(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(f"Volume/metadata mismatch: {image_path} has {data.shape[3]}, metadata has {len(meta)}")
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["stage"] = stage
    if "pair_id" in meta.columns:
        meta["pair_id"] = meta["pair_id"].map(normalize_pair_id)
        meta["condition_item_id"] = [
            make_condition_item_id(condition, pair_id)
            for pair_id in meta["pair_id"]
        ]
    if "run" in meta.columns:
        meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    for col in ["memory", "action_time"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    return meta, data, image_path


def masked_samples(data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def roi_masks(cfg: AnalysisConfig) -> dict[str, dict[str, Path]]:
    if not cfg.roi_manifest.exists():
        raise FileNotFoundError(f"ROI manifest not found: {cfg.roi_manifest}")
    return {
        roi_set: select_roi_masks(cfg.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa")
        for roi_set in cfg.roi_sets
    }


def one_sample_summary(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in frame.groupby(group_cols, dropna=False):
        values = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size < 2:
            t, p, mean, sd, dz = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            t, p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            dz = mean / sd if sd and np.isfinite(sd) else np.nan
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(group_cols, key)), "n": int(values.size), "mean": mean, "sd": sd, "t": t, "p": p, "cohens_dz": dz})
    out = pd.DataFrame(rows)
    if not out.empty and "p" in out:
        out["q_bh"] = bh_fdr(out["p"])
    return out


def _normal_two_sided_p(stat_value: float) -> float:
    if not np.isfinite(stat_value):
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(stat_value)))


def _pack_frequentist_result(model, formula: str, *, status: str, model_type: str) -> pd.DataFrame:
    rows = []
    params = model.params
    for term in params.index:
        stat_value = (
            float(model.tvalues[term])
            if hasattr(model, "tvalues") and term in model.tvalues.index
            else float(model.params[term] / model.bse[term])
        )
        rows.append(
            {
                "term": term,
                "estimate": float(params[term]),
                "se": float(model.bse[term]),
                "stat": stat_value,
                "p": float(model.pvalues[term]) if term in model.pvalues.index else _normal_two_sided_p(stat_value),
                "status": status,
                "model_type": model_type,
                "formula": formula,
                "n_obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def _detect_mixedlm_status(result, default_status: str) -> str:
    converged = getattr(result, "converged", True)
    if converged is False:
        return "mixed_not_converged"
    cov_re = getattr(result, "cov_re", None)
    if cov_re is not None:
        try:
            diag = np.diag(np.asarray(cov_re, dtype=float))
            if diag.size and (not np.all(np.isfinite(diag)) or np.any(diag <= 1e-12)):
                return "mixed_singular"
        except Exception:
            pass
    bse_fe = getattr(result, "bse_fe", None)
    if bse_fe is not None and not np.all(np.isfinite(np.asarray(bse_fe, dtype=float))):
        return "mixed_nonfinite_se"
    return default_status


def _pack_mixedlm_result(result, formula: str, *, status: str) -> pd.DataFrame:
    rows = []
    fe_params = result.fe_params
    bse_fe = result.bse_fe
    effective_status = _detect_mixedlm_status(result, status)
    for term in fe_params.index:
        estimate = float(fe_params[term])
        se = float(bse_fe[term]) if term in bse_fe.index else float("nan")
        stat_value = estimate / se if np.isfinite(se) and not math.isclose(se, 0.0) else float("nan")
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "se": se,
                "stat": stat_value,
                "p": _normal_two_sided_p(stat_value),
                "status": effective_status,
                "model_type": "mixedlm",
                "formula": formula,
                "n_obs": int(result.nobs),
            }
        )
    return pd.DataFrame(rows)


def _pack_bayes_binomial_mixed_result(result, model, formula: str, *, status: str, n_obs: int) -> pd.DataFrame:
    rows = []
    for term, estimate, se in zip(model.exog_names, result.fe_mean, result.fe_sd):
        estimate = float(estimate)
        se = float(se)
        stat_value = estimate / se if np.isfinite(se) and not math.isclose(se, 0.0) else float("nan")
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "se": se,
                "stat": stat_value,
                "p": _normal_two_sided_p(stat_value),
                "status": status,
                "model_type": "binomial_bayes_mixed_glm",
                "formula": formula,
                "n_obs": int(n_obs),
            }
        )
    return pd.DataFrame(rows)


def fit_formula(
    frame: pd.DataFrame,
    formula: str,
    *,
    family: str = "gaussian",
    group_col: str | None = None,
    item_col: str | None = None,
    prefer_mixed: bool = False,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    if smf is None or sm is None:
        return pd.DataFrame([{"term": "__model__", "status": "statsmodels_unavailable", "formula": formula}])
    data = frame.copy()
    mixed_error: str | None = None

    if prefer_mixed and group_col and group_col in data.columns:
        try:
            if family == "binomial":
                vc_formulas = {str(group_col): f"0 + C({group_col})"}
                if item_col and item_col in data.columns:
                    vc_formulas[str(item_col)] = f"0 + C({item_col})"
                model = sm.BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data)
                result = model.fit_vb()
                return _pack_bayes_binomial_mixed_result(
                    result,
                    model,
                    formula,
                    status="ok",
                    n_obs=len(data),
                )

            vc_formula = None
            if item_col and item_col in data.columns:
                vc_formula = {str(item_col): f"0 + C({item_col})"}
            model = smf.mixedlm(
                formula,
                data=data,
                groups=data[group_col],
                re_formula="1",
                vc_formula=vc_formula,
            )
            result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            return _pack_mixedlm_result(result, formula, status="ok")
        except Exception as exc:
            mixed_error = str(exc)

        if mixed_error and not allow_fallback:
            return pd.DataFrame(
                [
                    {
                        "term": "__model__",
                        "status": f"mixed_failed_no_fallback: {mixed_error}",
                        "model_type": "mixed_preferred",
                        "formula": formula,
                        "n_obs": int(len(data)),
                    }
                ]
            )

    try:
        if family == "binomial":
            model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()
            status = "fallback_nonmixed" if prefer_mixed else "ok"
            if mixed_error:
                status = f"fallback_after_mixed_failure: {mixed_error}"
            return _pack_frequentist_result(
                model,
                formula,
                status=status,
                model_type="glm_binomial",
            )
        else:
            model = smf.ols(formula, data=data).fit(cov_type="HC3")
            status = "fallback_nonmixed" if prefer_mixed else "ok"
            if mixed_error:
                status = f"fallback_after_mixed_failure: {mixed_error}"
            return _pack_frequentist_result(
                model,
                formula,
                status=status,
                model_type="ols_hc3",
            )
    except Exception as exc:
        return pd.DataFrame([{"term": "__model__", "status": f"failed: {exc}", "formula": formula, "n_obs": int(len(data))}])


def write_standard_outputs(cfg: AnalysisConfig, module_name: str, outputs: dict[str, pd.DataFrame | str | dict]) -> list[str]:
    written = []
    for filename, payload in outputs.items():
        path = safe_output_path(cfg, module_name, filename)
        if isinstance(payload, pd.DataFrame):
            write_table(payload, path)
        elif isinstance(payload, dict):
            save_json(payload, path)
        else:
            write_text(str(payload), path)
        written.append(str(path))
    return written
