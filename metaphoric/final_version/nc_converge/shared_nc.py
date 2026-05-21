#!/usr/bin/env python3
"""Shared helpers for NC convergence analyses.

Safety policy:
- read upstream tables only;
- write only under paper_outputs/qc/nc_converge;
- never overwrite existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

FINAL_ROOT = Path(__file__).resolve().parents[1]
REVIEWER_SUPP = FINAL_ROOT / "reviewer_supp"
if str(REVIEWER_SUPP) not in sys.path:
    sys.path.insert(0, str(REVIEWER_SUPP))

try:
    import shared as reviewer_shared  # type: ignore
except Exception:  # pragma: no cover
    reviewer_shared = None

SAFE_RELATIVE_ROOT = Path("paper_outputs") / "qc" / "nc_converge"
RUN_MAP = {
    "pre": ["run1", "run2"],
    "learning": ["run3", "run4"],
    "post": ["run5", "run6"],
    "retrieval": ["run7"],
}
SEMANTIC_ROIS = ("temporal_pole", "IFG", "AG", "pMTG_pSTS")
HPC_SPATIAL_ROIS = ("hippocampus", "PPA_PHG", "RSC_PCC", "PPC_SPL", "precuneus")


@dataclass(frozen=True)
class NCConfig:
    base_dir: Path
    paper_output_root: Path
    output_root: Path
    final_root: Path


def default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    base = default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--allow-empty", action="store_true", help="Write empty audit outputs instead of failing when optional inputs are missing.")


def default_config(args: argparse.Namespace) -> NCConfig:
    base_dir = Path(args.base_dir)
    paper_output_root = Path(args.paper_output_root or base_dir / "paper_outputs")
    output_root = Path(args.output_root or paper_output_root / "qc" / "nc_converge")
    cfg = NCConfig(base_dir=base_dir, paper_output_root=paper_output_root, output_root=output_root, final_root=Path(args.final_root))
    ensure_nc_root(cfg)
    return cfg


def resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_nc_root(cfg: NCConfig) -> Path:
    root = resolved(cfg.output_root)
    expected = resolved(cfg.paper_output_root / "qc" / "nc_converge")
    if root != expected and expected not in root.parents:
        raise ValueError(f"Refusing output outside nc_converge sandbox: {root}; expected under {expected}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def module_dir(cfg: NCConfig, module_name: str) -> Path:
    target = resolved(cfg.output_root / module_name)
    sandbox = resolved(cfg.output_root)
    if target != sandbox and sandbox not in target.parents:
        raise ValueError(f"Refusing module output outside sandbox: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_output_path(cfg: NCConfig, module_name: str, filename: str) -> Path:
    directory = module_dir(cfg, module_name)
    path = resolved(directory / filename)
    if directory not in path.parents:
        raise ValueError(f"Refusing path outside module directory: {path}")
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    return path


def read_table(path: Path) -> pd.DataFrame:
    if reviewer_shared is not None:
        return reviewer_shared.read_table(Path(path))
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


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


def write_outputs(cfg: NCConfig, module_name: str, outputs: dict[str, pd.DataFrame | str | dict]) -> list[str]:
    written: list[str] = []
    for name, payload in outputs.items():
        path = safe_output_path(cfg, module_name, name)
        if isinstance(payload, pd.DataFrame):
            write_table(payload, path)
        elif isinstance(payload, dict):
            save_json(payload, path)
        else:
            write_text(str(payload), path)
        written.append(str(path))
    return written


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    if reviewer_shared is not None:
        return reviewer_shared.bh_fdr(pvalues)
    values = pd.to_numeric(pd.Series(pvalues), errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[np.flatnonzero(valid)[order]] = np.clip(adjusted, 0, 1)
    return out


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if not np.isfinite(sd) or math.isclose(sd, 0.0):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def fit_formula(frame: pd.DataFrame, formula: str, *, family: str = "gaussian", group_col: str | None = "subject", item_col: str | None = "condition_item_id") -> pd.DataFrame:
    if reviewer_shared is None:
        return _fit_formula_numpy(frame, formula, family=family, status_prefix="reviewer_shared_unavailable")
    result = reviewer_shared.fit_formula(frame, formula, family=family, group_col=group_col, item_col=item_col, prefer_mixed=True, allow_fallback=True)
    if (
        isinstance(result, pd.DataFrame)
        and not result.empty
        and "status" in result.columns
        and result["status"].astype(str).str.contains("statsmodels_unavailable", na=False).any()
    ):
        return _fit_formula_numpy(frame, formula, family=family, status_prefix="statsmodels_unavailable")
    if _mixed_result_needs_fallback(result):
        fallback = reviewer_shared.fit_formula(frame, formula, family=family, group_col=group_col, item_col=item_col, prefer_mixed=False, allow_fallback=True)
        if isinstance(fallback, pd.DataFrame) and not fallback.empty:
            fallback = fallback.copy()
            fallback["fallback_reason"] = ";".join(sorted(result["status"].dropna().astype(str).unique()))
            return fallback
    return result


def _mixed_result_needs_fallback(result: pd.DataFrame) -> bool:
    if not isinstance(result, pd.DataFrame) or result.empty or "status" not in result.columns:
        return False
    statuses = result["status"].dropna().astype(str)
    if statuses.empty:
        return False
    if statuses.eq("ok").all():
        return False
    bad_tokens = (
        "mixed_nonfinite_se",
        "mixed_not_converged",
        "mixed_singular",
        "mixed_failed_no_fallback",
    )
    return statuses.str.contains("|".join(bad_tokens), regex=True).any()


def _split_top_level_plus(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for idx, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "+" and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _expand_formula_rhs(rhs: str) -> list[str]:
    rhs = " ".join(str(rhs).replace("\n", " ").split())
    terms: list[str] = []
    for term in _split_top_level_plus(rhs):
        if "*" not in term:
            terms.append(term)
            continue
        left, right = [part.strip() for part in term.split("*", 1)]
        if right.startswith("(") and right.endswith(")"):
            right_terms = _split_top_level_plus(right[1:-1])
        else:
            right_terms = [right]
        terms.append(left)
        terms.extend(right_terms)
        terms.extend([f"{left}:{rt}" for rt in right_terms])
    out: list[str] = []
    for term in terms:
        term = term.strip()
        if term and term not in out:
            out.append(term)
    return out


def _categorical_spec(term: str) -> tuple[str, str | None] | None:
    if not term.strip().startswith("C("):
        return None
    match = re.match(r"C\(\s*([^,\)]+)", term.strip())
    if not match:
        return None
    var = match.group(1).strip()
    ref_match = re.search(r"Treatment\(\s*(?:reference\s*=\s*)?[\"']?([^\"'\)]+)[\"']?", term)
    reference = ref_match.group(1).strip() if ref_match else None
    return var, reference


def _factor_design(data: pd.DataFrame, term: str) -> dict[str, pd.Series]:
    term = term.strip()
    cat = _categorical_spec(term)
    if cat is not None:
        var, reference = cat
        if var not in data.columns:
            return {}
        values = data[var].astype(str).str.lower()
        levels = sorted(v for v in values.dropna().unique() if v and v != "nan")
        if len(levels) <= 1:
            return {}
        ref = str(reference).lower() if reference is not None else levels[0]
        if ref not in levels:
            ref = levels[0]
        prefix = term
        return {f"{prefix}[T.{level}]": values.eq(level).astype(float) for level in levels if level != ref}

    square = re.fullmatch(r"I\(\s*([^*\s]+)\s*\*\*\s*2\s*\)", term)
    if square:
        var = square.group(1)
        if var not in data.columns:
            return {}
        values = pd.to_numeric(data[var], errors="coerce")
        return {term: values * values}

    if term in data.columns:
        return {term: pd.to_numeric(data[term], errors="coerce")}
    return {}


def _term_design(data: pd.DataFrame, term: str) -> dict[str, pd.Series]:
    factors = [part.strip() for part in term.split(":")]
    designs = [_factor_design(data, factor) for factor in factors]
    if any(not design for design in designs):
        return {}
    out = designs[0]
    for design in designs[1:]:
        merged: dict[str, pd.Series] = {}
        for name_a, values_a in out.items():
            for name_b, values_b in design.items():
                merged[f"{name_a}:{name_b}"] = values_a * values_b
        out = merged
    return out


def _design_from_formula(frame: pd.DataFrame, formula: str) -> tuple[str, pd.DataFrame]:
    lhs, rhs = [part.strip() for part in formula.split("~", 1)]
    columns: dict[str, pd.Series] = {"Intercept": pd.Series(1.0, index=frame.index)}
    for term in _expand_formula_rhs(rhs):
        columns.update(_term_design(frame, term))
    design = pd.DataFrame(columns, index=frame.index)
    return lhs, design


def _pack_numpy_result(
    names: Sequence[str],
    beta: np.ndarray,
    se: np.ndarray,
    formula: str,
    *,
    n_obs: int,
    status: str,
    model_type: str,
) -> pd.DataFrame:
    rows = []
    for name, estimate, std_err in zip(names, beta, se):
        stat = float(estimate / std_err) if np.isfinite(std_err) and not math.isclose(float(std_err), 0.0) else float("nan")
        p = float(2.0 * stats.norm.sf(abs(stat))) if np.isfinite(stat) else float("nan")
        rows.append(
            {
                "term": name,
                "estimate": float(estimate),
                "se": float(std_err),
                "stat": stat,
                "p": p,
                "status": status,
                "model_type": model_type,
                "formula": formula,
                "n_obs": int(n_obs),
            }
        )
    return pd.DataFrame(rows)


def _fit_formula_numpy(frame: pd.DataFrame, formula: str, *, family: str, status_prefix: str) -> pd.DataFrame:
    try:
        lhs, design = _design_from_formula(frame, formula)
        y = pd.to_numeric(frame[lhs], errors="coerce")
        model_data = pd.concat([y.rename(lhs), design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        names = list(design.columns)
        if len(model_data) <= max(3, len(names) + 1):
            return pd.DataFrame([{"term": "__model__", "status": f"{status_prefix}_numpy_failed: too_few_complete_rows", "formula": formula, "n_obs": int(len(model_data))}])
        x = model_data[names].to_numpy(dtype=float)
        yv = model_data[lhs].to_numpy(dtype=float)
        if family == "binomial":
            if set(np.unique(yv)) - {0.0, 1.0}:
                return pd.DataFrame([{"term": "__model__", "status": f"{status_prefix}_numpy_failed: non_binary_outcome", "formula": formula, "n_obs": int(len(model_data))}])

            def objective(beta: np.ndarray) -> float:
                eta = np.clip(x @ beta, -35.0, 35.0)
                return float(np.sum(np.logaddexp(0.0, eta) - yv * eta))

            result = optimize.minimize(objective, np.zeros(x.shape[1], dtype=float), method="BFGS")
            beta = np.asarray(result.x, dtype=float)
            eta = np.clip(x @ beta, -35.0, 35.0)
            prob = 1.0 / (1.0 + np.exp(-eta))
            weights = np.clip(prob * (1.0 - prob), 1e-8, None)
            cov = np.linalg.pinv(x.T @ (x * weights[:, None]))
            se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            status = f"{status_prefix}_numpy_logit"
            if not result.success:
                status += f"_not_converged: {result.message}"
            return _pack_numpy_result(names, beta, se, formula, n_obs=len(model_data), status=status, model_type="logit_numpy")

        xtx_inv = np.linalg.pinv(x.T @ x)
        beta = xtx_inv @ x.T @ yv
        fitted = x @ beta
        resid = yv - fitted
        hat = np.sum((x @ xtx_inv) * x, axis=1)
        denom = np.clip(1.0 - hat, 1e-8, None)
        scaled = (resid / denom) ** 2
        meat = x.T @ (x * scaled[:, None])
        cov = xtx_inv @ meat @ xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        return _pack_numpy_result(names, beta, se, formula, n_obs=len(model_data), status=f"{status_prefix}_numpy_ols_hc3", model_type="ols_hc3_numpy")
    except Exception as exc:
        return pd.DataFrame([{"term": "__model__", "status": f"{status_prefix}_numpy_failed: {exc}", "formula": formula, "n_obs": int(len(frame))}])


def normalize_condition(value: object) -> str:
    return re.sub(r"[^a-z]+", "", str(value).strip().lower())


def normalize_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        value_f = float(text)
    except Exception:
        return text
    if math.isnan(value_f):
        return ""
    return str(int(value_f)) if value_f.is_integer() else str(value_f)


def pair_id_column(frame: pd.DataFrame) -> str | None:
    for col in ("original_pair_id", "pair_id", "item_id", "stimulus_id", "condition_item_number"):
        if col in frame.columns:
            return col
    return None


def build_condition_item_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "condition_item_id" in out.columns:
        out["condition_item_id"] = out["condition_item_id"].astype(str)
        return out
    if "condition" not in out.columns:
        return out
    pid = pair_id_column(out)
    if pid is None:
        return out
    out["condition_item_id"] = [
        f"{normalize_condition(c)}_{normalize_pair_id(p)}" if normalize_condition(c) and normalize_pair_id(p) else ""
        for c, p in zip(out["condition"], out[pid])
    ]
    return out


def stage_from_table(frame: pd.DataFrame, path: Path) -> str:
    if "stage" in frame.columns and frame["stage"].notna().any():
        return str(frame["stage"].dropna().iloc[0])
    name = path.as_posix().lower()
    for stage in RUN_MAP:
        if stage in name:
            return stage
    return "unknown"


def discover_tables(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(root).glob(pattern))
    return sorted({p for p in paths if p.is_file() and p.suffix.lower() in {".tsv", ".csv", ".txt"}})


def audit_condition_items(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    required_keys = ["subject", "stage", "roi", "condition_item_id"]
    duplicate_disambiguators = [
        "run",
        "phase",
        "time",
        "role",
        "edge_status",
        "pairing_scheme",
        "component_name",
        "component_metric",
        "metric",
        "control_condition_item_id",
        "control_original_pair_id",
    ]
    for path in paths:
        try:
            frame = build_condition_item_id(read_table(path))
            stage = stage_from_table(frame, path)
            if "stage" not in frame.columns:
                frame = frame.assign(stage=stage)
            keys = [c for c in required_keys if c in frame.columns]
            missing_keys = [c for c in required_keys if c not in frame.columns]
            duplicates = 0
            if not missing_keys:
                counts = frame.groupby(keys, dropna=False).size()
                duplicates = int((counts > 1).sum())
                duplicate_key = "|".join(keys)
                extended_duplicate_key = ""
                extended_duplicates = np.nan
                if duplicates == 0:
                    status = "ok"
                else:
                    extra_keys = [c for c in duplicate_disambiguators if c in frame.columns and c not in keys]
                    extended_keys = keys + extra_keys
                    if extra_keys:
                        extended_counts = frame.groupby(extended_keys, dropna=False).size()
                        extended_duplicates = int((extended_counts > 1).sum())
                        extended_duplicate_key = "|".join(extended_keys)
                        status = "ok_repeated_measure_key" if extended_duplicates == 0 else "duplicate_condition_item_id"
                    else:
                        status = "duplicate_condition_item_id"
            else:
                # Do not fail long-form/model tables that lack the full spec key.
                # They remain audit-only so A2 does not block valid downstream data.
                status = f"audit_only_missing_keys: {','.join(missing_keys)}"
                duplicate_key = "|".join(keys)
                extended_duplicate_key = ""
                extended_duplicates = np.nan
            rows.append({
                "path": str(path),
                "stage": stage,
                "n_rows": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()) if "subject" in frame.columns else np.nan,
                "n_condition_items": int(frame["condition_item_id"].replace("", np.nan).nunique()) if "condition_item_id" in frame.columns else np.nan,
                "duplicates": duplicates,
                "duplicate_key": duplicate_key,
                "extended_duplicates": extended_duplicates,
                "extended_duplicate_key": extended_duplicate_key,
                "missing_duplicate_keys": ",".join(missing_keys),
                "dropped_missing_condition_item_id": int((frame.get("condition_item_id", pd.Series(dtype=str)).astype(str) == "").sum()) if "condition_item_id" in frame.columns else np.nan,
                "status": status,
            })
        except Exception as exc:
            rows.append({"path": str(path), "stage": "unknown", "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def subject_intersection(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    subject_sets: dict[str, set[str]] = {}
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        if "subject" not in frame.columns:
            continue
        label = stage_from_table(frame, path)
        if "retrieval" in path.as_posix().lower() or label == "retrieval" or "run7" in path.as_posix().lower():
            key = "run7_neural"
        elif "behavior" in path.as_posix().lower() or "memory" in path.as_posix().lower():
            key = "behavior_memory"
        else:
            key = label
        subject_sets.setdefault(key, set()).update(frame["subject"].dropna().astype(str))
    all_subjects = sorted(set().union(*subject_sets.values())) if subject_sets else []
    for sub in all_subjects:
        row = {"subject": sub}
        for key, values in subject_sets.items():
            row[key] = sub in values
        row["in_all_sets"] = all(sub in values for values in subject_sets.values()) if subject_sets else False
        rows.append(row)
    return pd.DataFrame(rows)


def roi_base(roi: object) -> str:
    text = str(roi)
    text = re.sub(r"^meta_[LR]_", "", text)
    text = re.sub(r"^meta_", "", text)
    text = text.replace("/", "_")
    return text


def roi_to_network(roi: object, roi_set: object | None = None) -> str | None:
    base = roi_base(roi).lower()
    for name in SEMANTIC_ROIS:
        if name.lower() in base:
            return "semantic"
    for name in HPC_SPATIAL_ROIS:
        if name.lower() in base:
            return "hpc_spatial"
    set_text = str(roi_set).lower() if roi_set is not None else ""
    if "metaphor" in set_text:
        return "semantic"
    if "spatial" in set_text:
        return "hpc_spatial"
    return None


def add_network_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    roi_set = out["roi_set"] if "roi_set" in out.columns else pd.Series([None] * len(out), index=out.index)
    inferred = pd.Series(
        [roi_to_network(r, s) for r, s in zip(out.get("roi", pd.Series(index=out.index, dtype=object)), roi_set)],
        index=out.index,
        dtype="object",
    )
    if "network" not in out.columns:
        out["network"] = inferred
    else:
        out["network"] = out["network"].where(out["network"].notna() & out["network"].astype(str).ne(""), inferred)
    return out


def first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def value_column(frame: pd.DataFrame) -> str | None:
    return first_existing_column(frame, ["value", "metric_value", "similarity", "pair_similarity", "estimate", "mean", "z", "rho", "score", "effect"])


def make_network_composite(frame: pd.DataFrame, value_col: str | None = None, metric_name: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = add_network_column(build_condition_item_id(frame))
    value_col = value_col or value_column(out)
    if value_col is None or "network" not in out.columns:
        return pd.DataFrame()
    out = out[out["network"].notna()].copy()
    out["metric_value_raw"] = pd.to_numeric(out[value_col], errors="coerce")
    group_for_z = [c for c in ["subject", "roi", "metric"] if c in out.columns]
    if not group_for_z:
        group_for_z = ["roi"] if "roi" in out.columns else []
    if group_for_z:
        out["metric_value_z"] = out.groupby(group_for_z, dropna=False)["metric_value_raw"].transform(zscore)
    else:
        out["metric_value_z"] = zscore(out["metric_value_raw"])
    group_cols = [c for c in ["subject", "condition_item_id", "stage", "phase", "condition", "network", "metric"] if c in out.columns]
    if "metric" not in group_cols:
        out["metric"] = metric_name or value_col
        group_cols.append("metric")
    comp = out.groupby(group_cols, dropna=False, as_index=False).agg(
        metric_value=("metric_value_z", "mean"),
        n_roi=("roi", "nunique") if "roi" in out.columns else ("metric_value_z", "size"),
    )
    return comp


def combine_metric_tables(paths: Sequence[Path], metric_hint: str | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_path"] = str(path)
        if "metric" not in frame.columns:
            frame["metric"] = metric_hint or path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def q_for_ok_rows(frame: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    out = frame.copy()
    if p_col not in out.columns:
        return out
    if "status" in out.columns:
        status = out["status"].astype(str)
        ok = status.eq("ok") | status.str.startswith("fallback_after_mixed_failure")
    else:
        ok = pd.Series(True, index=out.index)
    out["q_bh"] = np.nan
    out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, p_col])
    return out


def markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    """Render a small markdown table without requiring pandas' tabulate extra."""
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    if index:
        text = text.reset_index()
    text = text.fillna("")
    columns = [str(c) for c in text.columns]
    rows = [[str(v) for v in row] for row in text.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_status(path: Path) -> str:
    try:
        result = subprocess.run(["git", "status", "--short", str(path)], cwd=FINAL_ROOT.parents[1], text=True, capture_output=True, check=False)
        return result.stdout.strip()
    except Exception as exc:
        return f"git_status_failed: {exc}"


def log_text(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines)
    return f"# {title}\n\n{body}\n"
