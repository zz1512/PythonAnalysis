"""
delta_rho_lmm.py

用途
- 在 Model-RSA 产出的 `model_rdm_subject_metrics.tsv` 基础上，对每个被试×ROI×模型
  计算 Δρ = ρ_post − ρ_pre，然后用线性混合模型检验：
    delta_rho ~ C(condition_group) * C(model) + (1|subject) + (1|roi)
- 如果原始表没有 `condition_group` 列（即多条件被合成一个 neural RDM），那么
  交互项默认只在 model 上展开：delta_rho ~ C(model) + (1|subject) + (1|roi)。

输入
- `--metrics-file`: `model_rdm_subject_metrics.tsv`（model_rdm_comparison.py 产出）
- `--partial-metrics-file`（可选）：`model_rdm_partial_metrics.tsv`，用 partial_rho 替代 rho

输出（--output-dir）
- `delta_rho_lmm_summary.txt`
- `delta_rho_lmm_params.tsv`
- `delta_rho_lmm_model.json`

常见坑
- 必须同一 subject/roi/model 上 pre 与 post 都存在才能形成 Δρ。
- 若 condition 列存在（例如先按 condition 独立跑 RSA 再合并），请用 `--condition-col` 指定。

随机效应结构说明（重要）
- 目标模型是 crossed random effects：`(1|subject) + (1|roi)`，因为我们希望同时泛化到被试与 ROI。
- 但 statsmodels 的 MixedLM 对 fully-crossed 的多组随机截距支持有限，这里采用：
  - `groups=subject` 作为主要随机截距
  - `vc_formula={"roi": "0 + C(roi)"}` 把 ROI 作为方差分量加入（实践中常用的近似写法）
- 这在语义上更接近 crossed，而不是把 ROI 真的当作 nested 在 subject 之内；同时请在结果解读中报告该建模选择。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

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
RSA_ROOT = Path(__file__).resolve().parent
if str(RSA_ROOT) not in sys.path:
    sys.path.append(str(RSA_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402
import rsa_config as cfg  # noqa: E402


def _default_model_rdm_dir() -> Path:
    roi_tag = sanitize_roi_tag(getattr(cfg, "ROI_SET", ""))
    return Path(cfg.BASE_DIR) / f"model_rdm_results_{roi_tag}"


def _resolve_metrics_file(path_arg: Path | None) -> Path:
    if path_arg is None:
        return _default_model_rdm_dir() / "model_rdm_subject_metrics.tsv"
    if path_arg.is_dir():
        return path_arg / "model_rdm_subject_metrics.tsv"
    return path_arg


def _resolve_partial_metrics_file(path_arg: Path | None, *, metrics_file: Path, use_partial: bool) -> Path | None:
    if path_arg is not None:
        if path_arg.is_dir():
            return path_arg / "model_rdm_partial_metrics.tsv"
        return path_arg
    if use_partial:
        return metrics_file.parent / "model_rdm_partial_metrics.tsv"
    return None


def _resolve_output_dir(path_arg: Path | None, *, metrics_file: Path) -> Path:
    if path_arg is not None:
        return path_arg
    override = os.environ.get("METAPHOR_DELTA_RHO_OUT_DIR", "").strip()
    if override:
        return Path(override)
    return metrics_file.parent / "lmm"


def _slug(text: object) -> str:
    out = str(text).strip()
    for src, dst in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("*", "_"),
        ("?", "_"),
        ('"', "_"),
        ("<", "_"),
        (">", "_"),
        ("|", "_"),
    ]:
        out = out.replace(src, dst)
    return out or "unknown"


def _derive_base_contrast(roi_name: object) -> str | None:
    text = str(roi_name).strip()
    if not text:
        return None
    match = re.match(r"^(.*?)(_c\d+_.*)$", text)
    if match:
        return match.group(1)
    return None


def _ensure_family_column(frame: pd.DataFrame, family_col: str) -> tuple[pd.DataFrame, str | None]:
    if family_col in frame.columns:
        return frame, family_col
    if family_col == "base_contrast":
        out = frame.copy()
        out["base_contrast"] = out["roi"].map(_derive_base_contrast)
        if out["base_contrast"].notna().any():
            return out, "base_contrast"
    return frame, None


def compute_delta(frame: pd.DataFrame, value_col: str, condition_col: str | None) -> pd.DataFrame:
    required = {"subject", "roi", "time", "model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    work = frame.copy()
    work = work[work["model"].notna()]
    if "skip_reason" in work.columns:
        work = work[work["skip_reason"].fillna("") == ""]
    index_cols = ["subject", "roi", "model"]
    if condition_col and condition_col in work.columns:
        index_cols.append(condition_col)
    duplicate_keys = work.duplicated(subset=index_cols + ["time"]).sum()
    if duplicate_keys:
        print(f"[delta_rho_lmm] warning: {duplicate_keys} duplicate rows on "
              f"{index_cols + ['time']} will be averaged before Δρ.")
    pivot = work.pivot_table(index=index_cols, columns="time", values=value_col, aggfunc="mean")
    n_before = len(pivot)
    pivot = pivot.dropna(subset=["pre", "post"])
    n_after = len(pivot)
    if n_before - n_after:
        print(f"[delta_rho_lmm] dropped {n_before - n_after} rows missing pre or post.")
    pivot["delta_rho"] = pivot["post"] - pivot["pre"]
    return pivot.reset_index()


def fit_lmm(delta_frame: pd.DataFrame, condition_col: str | None):
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        raise RuntimeError(f"statsmodels is required: {exc}")

    formula = "delta_rho ~ C(model)"
    if condition_col and condition_col in delta_frame.columns:
        formula = f"delta_rho ~ C({condition_col}) * C(model)"

    model = smf.mixedlm(
        formula,
        data=delta_frame,
        groups=delta_frame["subject"],
        vc_formula={"roi": "0 + C(roi)"},
        re_formula="1",
    )
    return model.fit(method="lbfgs", maxiter=200, disp=False), formula


def _write_fit_outputs(
    fit,
    *,
    formula: str,
    delta_frame: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    source: Path,
    condition_col: str | None,
    extra_json: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    summary_text = str(fit.summary())
    (output_dir / "delta_rho_lmm_summary.txt").write_text(summary_text, encoding="utf-8")
    params = pd.DataFrame({"term": fit.params.index, "estimate": fit.params.values})
    write_table(params, output_dir / "delta_rho_lmm_params.tsv")
    model_info = {
        "formula": formula,
        "value_col": value_col,
        "source_file": str(source),
        "condition_col": condition_col,
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "n_obs": int(fit.nobs),
        "n_subjects": int(delta_frame["subject"].nunique()),
        "n_rois": int(delta_frame["roi"].nunique()),
        "n_models": int(delta_frame["model"].nunique()),
        "converged": bool(getattr(fit, "converged", True)),
        "random_effects_note": "groups=subject; roi modeled via vc_formula (nested within subject "
                                "as a pragmatic approximation of crossed RE given statsmodels limits).",
    }
    if extra_json:
        model_info.update(extra_json)
    save_json(model_info, output_dir / "delta_rho_lmm_model.json")
    return params, model_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Δρ LMM on Model-RSA outputs.")
    parser.add_argument("--metrics-file", type=Path, default=None,
                        help="model_rdm_subject_metrics.tsv or a model_rdm_results_* directory. "
                             "If omitted, auto-detects the current ROI-set result directory.")
    parser.add_argument("--partial-metrics-file", type=Path, default=None,
                        help="Optional partial-correlation table or containing directory.")
    parser.add_argument("--use-partial", action="store_true",
                        help="Use model_rdm_partial_metrics.tsv from the resolved result directory automatically.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional output directory. Default: <model_rdm_results_dir>/lmm")
    parser.add_argument("--condition-col", default=None,
                        help="Optional condition-group column for the interaction.")
    parser.add_argument("--family-split", action="store_true",
                        help="Also fit separate LMMs within ROI families (e.g. main_functional base_contrast families).")
    parser.add_argument("--family-col", default="base_contrast",
                        help="Family column used with --family-split. "
                             "If absent, base_contrast will be auto-derived from ROI names when possible.")
    args = parser.parse_args()

    metrics_file = _resolve_metrics_file(args.metrics_file)
    if not metrics_file.exists():
        raise FileNotFoundError(
            "Model-RSA metrics file not found.\n"
            f"Resolved path: {metrics_file}\n"
            f"Default search root: {_default_model_rdm_dir()}"
        )

    partial_metrics_file = _resolve_partial_metrics_file(
        args.partial_metrics_file,
        metrics_file=metrics_file,
        use_partial=args.use_partial,
    )
    if partial_metrics_file is not None:
        if not partial_metrics_file.exists():
            raise FileNotFoundError(f"Partial metrics file not found: {partial_metrics_file}")
        source = partial_metrics_file
        value_col = "partial_rho"
    else:
        source = metrics_file
        value_col = "rho"

    output_dir = ensure_dir(_resolve_output_dir(args.output_dir, metrics_file=metrics_file))

    sep = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(source, sep=sep)

    condition_col = args.condition_col

    delta_frame = compute_delta(frame, value_col=value_col, condition_col=condition_col)
    write_table(delta_frame, output_dir / "delta_rho_long.tsv")

    fit, formula = fit_lmm(delta_frame, condition_col)
    _write_fit_outputs(
        fit,
        formula=formula,
        delta_frame=delta_frame,
        output_dir=output_dir,
        value_col=value_col,
        source=source,
        condition_col=condition_col,
    )

    if args.family_split:
        family_ready, resolved_family_col = _ensure_family_column(delta_frame, args.family_col)
        if resolved_family_col is None:
            raise ValueError(
                f"--family-split requested, but could not resolve family column '{args.family_col}'."
            )
        family_ready = family_ready[family_ready[resolved_family_col].notna()].copy()
        family_rows = []
        family_param_rows = []
        family_root = ensure_dir(output_dir / "family_split")
        for family_value, family_frame in family_ready.groupby(resolved_family_col):
            family_frame = family_frame.copy()
            family_out = ensure_dir(family_root / f"{resolved_family_col}__{_slug(family_value)}")
            write_table(family_frame, family_out / "delta_rho_long.tsv")
            summary_row = {
                "family_col": resolved_family_col,
                "family_value": family_value,
                "n_obs_input": int(len(family_frame)),
                "n_subjects": int(family_frame["subject"].nunique()),
                "n_rois": int(family_frame["roi"].nunique()),
                "n_models": int(family_frame["model"].nunique()),
                "skip_reason": "",
            }
            if family_frame["subject"].nunique() < 2 or family_frame["roi"].nunique() < 1 or family_frame["model"].nunique() < 2:
                summary_row["skip_reason"] = "insufficient_data"
                family_rows.append(summary_row)
                continue
            try:
                family_fit, family_formula = fit_lmm(family_frame, condition_col)
                family_params, family_info = _write_fit_outputs(
                    family_fit,
                    formula=family_formula,
                    delta_frame=family_frame,
                    output_dir=family_out,
                    value_col=value_col,
                    source=source,
                    condition_col=condition_col,
                    extra_json={"family_col": resolved_family_col, "family_value": family_value},
                )
                family_rows.append({
                    **summary_row,
                    "formula": family_formula,
                    "aic": family_info["aic"],
                    "bic": family_info["bic"],
                    "n_obs_model": family_info["n_obs"],
                    "converged": family_info["converged"],
                    "output_dir": str(family_out),
                })
                family_params = family_params.assign(
                    family_col=resolved_family_col,
                    family_value=family_value,
                    output_dir=str(family_out),
                )
                family_param_rows.append(family_params)
            except Exception as exc:
                summary_row["skip_reason"] = f"error: {exc}"
                family_rows.append(summary_row)

        if family_rows:
            write_table(pd.DataFrame(family_rows), output_dir / "delta_rho_family_split.tsv")
        if family_param_rows:
            write_table(pd.concat(family_param_rows, ignore_index=True), output_dir / "delta_rho_family_split_params.tsv")
    print(f"[delta_rho_lmm] fit ({formula}) -> {output_dir}")


if __name__ == "__main__":
    main()
