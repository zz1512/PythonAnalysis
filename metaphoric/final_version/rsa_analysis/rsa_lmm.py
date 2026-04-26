"""
rsa_lmm.py

用途
- 对 `run_rsa_optimized.py` 输出的 item-wise RSA 明细做线性混合模型（LMM）：
  `similarity ~ condition * time + (1|subject) + (1|item)`（item 作为方差成分）。
- 当前版本默认避免把不同来源/不同对比定义的 ROI 粗暴混在一个模型里。
- 默认纳入 `rsa_itemwise_details.csv` 中存在的全部条件；当前主线应包含 `Metaphor / Spatial / Baseline`。

输入
- input_path: TSV/CSV（默认读取 `rsa_config.py` 中 `OUTPUT_DIR / rsa_itemwise_details.csv`）
  必须包含列：`subject`, `condition`, `time`(或 stage), `similarity`，以及 item 列（pair_id/word_label 等）

输出（output_dir）
- 默认写入 `rsa_config.py` 中 `OUTPUT_DIR / lmm_{ROI_SET}`
- `by_base_contrast/`：按 ROI 所属功能对比分开跑的 LMM
- `by_roi/`：每个 ROI 单独跑的 LMM
- `lmm_by_base_contrast.tsv`：按 contrast 汇总的参数表
- `lmm_by_roi.tsv`：按 ROI 汇总的参数表
- `analysis_manifest.json`：记录本次分析如何分层
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

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

from common.final_utils import ensure_dir, read_table, save_json, write_table
from common.roi_library import sanitize_roi_tag
import rsa_config as cfg


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "time" not in working.columns and "stage" in working.columns:
        working["time"] = working["stage"]
    if "item" not in working.columns:
        if "pair_id" in working.columns:
            working["item"] = working["pair_id"]
        elif "word_label" in working.columns:
            working["item"] = working["word_label"]
        else:
            working["item"] = range(1, len(working) + 1)
    return working


def attach_roi_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if "roi" not in frame.columns:
        return frame
    manifest_path = getattr(cfg, "ROI_MANIFEST", None)
    if not manifest_path or not Path(manifest_path).exists():
        return frame

    manifest = read_table(manifest_path).copy()
    if "roi_name" not in manifest.columns:
        return frame

    merge_cols = ["roi_name", "roi_set", "base_contrast", "hemisphere", "source_type", "theory_role"]
    available = [column for column in merge_cols if column in manifest.columns]
    meta = manifest[available].drop_duplicates("roi_name")
    merged = frame.merge(meta, how="left", left_on="roi", right_on="roi_name")
    if "roi_name" in merged.columns:
        merged = merged.drop(columns=["roi_name"])
    return merged


def _default_input_candidates() -> list[Path]:
    roi_tag = sanitize_roi_tag(getattr(cfg, "ROI_SET", ""))
    candidates = [
        Path(cfg.OUTPUT_DIR) / "rsa_itemwise_details.csv",
        Path(cfg.BASE_DIR) / f"rsa_results_optimized_{roi_tag}" / "rsa_itemwise_details.csv",
        Path(cfg.BASE_DIR) / "rsa_results_optimized" / "rsa_itemwise_details.csv",
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _default_input_path() -> Path:
    candidates = _default_input_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_input_path(input_arg: Path | None) -> Path:
    if input_arg is None:
        return _default_input_path()
    if input_arg.is_dir():
        return input_arg / "rsa_itemwise_details.csv"
    return input_arg


def _infer_roi_tag_from_input(input_path: Path) -> str:
    parent_name = input_path.parent.name.strip()
    prefix = "rsa_results_optimized_"
    if parent_name.startswith(prefix):
        suffix = parent_name[len(prefix):].strip()
        if suffix:
            return sanitize_roi_tag(suffix)
    return sanitize_roi_tag(getattr(cfg, "ROI_SET", ""))


def _resolve_output_dir(output_arg: Path | None, *, input_path: Path) -> Path:
    if output_arg is not None:
        return output_arg
    override = os.environ.get("METAPHOR_RSA_LMM_OUT_DIR", "").strip()
    if override:
        return Path(override)
    roi_tag = _infer_roi_tag_from_input(input_path)
    return input_path.parent / f"lmm_{roi_tag}"


def fit_model(frame: pd.DataFrame, *, formula: str):
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        raise RuntimeError(f"statsmodels is required: {exc}")

    working = prepare_frame(frame)
    model = smf.mixedlm(
        formula,
        data=working,
        groups=working["subject"],
        vc_formula={"item": "0 + C(item)"},
        re_formula="1",
    )
    return model.fit(method="lbfgs", maxiter=200, disp=False)


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(result):
        return result
    return float("nan")


def summarize_fit(
    fit,
    *,
    analysis_level: str,
    analysis_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    params = fit.params
    bse = getattr(fit, "bse", pd.Series(index=params.index, dtype=float))
    pvalues = getattr(fit, "pvalues", pd.Series(index=params.index, dtype=float))
    conf_int = fit.conf_int()
    for term in params.index:
        rows.append(
            {
                "analysis_level": analysis_level,
                "analysis_name": analysis_name,
                "term": term,
                "estimate": _safe_float(params[term]),
                "std_err": _safe_float(bse.get(term)),
                "p_value": _safe_float(pvalues.get(term)),
                "ci_low": _safe_float(conf_int.loc[term, 0]) if term in conf_int.index else float("nan"),
                "ci_high": _safe_float(conf_int.loc[term, 1]) if term in conf_int.index else float("nan"),
                "n_obs": int(fit.nobs),
                "n_subjects": int(frame["subject"].nunique()) if "subject" in frame.columns else 0,
                "n_items": int(frame["item"].nunique()) if "item" in frame.columns else 0,
                "n_roi": int(frame["roi"].nunique()) if "roi" in frame.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def write_single_model(output_dir: Path, fit, frame: pd.DataFrame, analysis_meta: dict[str, object]) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    summary_text = str(fit.summary())
    (output_dir / "rsa_lmm_summary.txt").write_text(summary_text, encoding="utf-8")
    params = summarize_fit(
        fit,
        analysis_level=str(analysis_meta["analysis_level"]),
        analysis_name=str(analysis_meta["analysis_name"]),
        frame=frame,
    )
    write_table(params, output_dir / "rsa_lmm_params.tsv")
    save_json(
        {
            "analysis_level": analysis_meta["analysis_level"],
            "analysis_name": analysis_meta["analysis_name"],
            "aic": _safe_float(getattr(fit, "aic", float("nan"))),
            "bic": _safe_float(getattr(fit, "bic", float("nan"))),
            "n_obs": int(fit.nobs),
            "n_subjects": int(frame["subject"].nunique()) if "subject" in frame.columns else 0,
            "n_items": int(frame["item"].nunique()) if "item" in frame.columns else 0,
            "n_roi": int(frame["roi"].nunique()) if "roi" in frame.columns else 0,
        },
        output_dir / "rsa_lmm_model.json",
    )
    return params


def run_models_by_group(
    frame: pd.DataFrame,
    *,
    group_col: str,
    base_output_dir: Path,
    analysis_level: str,
    formula: str,
) -> pd.DataFrame:
    all_params = []
    for group_value, group_frame in frame.groupby(group_col):
        working = prepare_frame(group_frame)
        fit = fit_model(working, formula=formula)
        safe_name = str(group_value).replace("/", "_").replace("\\", "_").replace(" ", "_")
        params = write_single_model(
            base_output_dir / safe_name,
            fit,
            working,
            {"analysis_level": analysis_level, "analysis_name": str(group_value)},
        )
        all_params.append(params)
    if not all_params:
        return pd.DataFrame()
    return pd.concat(all_params, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-effects models for item-wise RSA results.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=None,
        help="Item-wise RSA table or RSA result directory. If omitted, the script auto-detects the current "
             "ROI-set result under E:/python_metaphor.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, writes to <input_dir>/lmm_<roi_set>.",
    )
    parser.add_argument(
        "--allow-pooled",
        action="store_true",
        help="Also fit one pooled model across all ROI, even when multiple base_contrast families are mixed.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition filter (e.g., baseline). When only one condition remains, the model uses "
             "`similarity ~ C(time)` instead of the full interaction.",
    )
    parser.add_argument(
        "--formula",
        default=None,
        help="Optional statsmodels formula override. If provided, takes precedence over auto-selected formula.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input_path)
    if not input_path.exists():
        tried = "\n".join(f"- {path}" for path in _default_input_candidates())
        raise FileNotFoundError(
            "RSA item-wise input not found.\n"
            f"Resolved path: {input_path}\n"
            "Auto-detect candidates were:\n"
            f"{tried}"
        )

    output_dir = ensure_dir(_resolve_output_dir(args.output_dir, input_path=input_path))
    frame = attach_roi_metadata(read_table(input_path))
    if args.conditions:
        allowed = {str(item) for item in args.conditions}
        frame = frame[frame["condition"].astype(str).isin(allowed)].copy()
    frame = prepare_frame(frame)

    if args.formula:
        formula = str(args.formula)
    else:
        n_cond = int(frame["condition"].nunique()) if "condition" in frame.columns else 0
        formula = "similarity ~ C(time)" if n_cond <= 1 else "similarity ~ C(condition) * C(time)"

    by_roi = run_models_by_group(
        frame,
        group_col="roi",
        base_output_dir=output_dir / "by_roi",
        analysis_level="roi",
        formula=formula,
    )
    if not by_roi.empty:
        write_table(by_roi, output_dir / "lmm_by_roi.tsv")

    by_base_contrast = pd.DataFrame()
    valid_base = "base_contrast" in frame.columns and frame["base_contrast"].notna().any()
    if valid_base:
        contrast_frame = frame[frame["base_contrast"].astype(str).str.strip() != ""].copy()
        if not contrast_frame.empty and contrast_frame["base_contrast"].nunique() > 0:
            by_base_contrast = run_models_by_group(
                contrast_frame,
                group_col="base_contrast",
                base_output_dir=output_dir / "by_base_contrast",
                analysis_level="base_contrast",
                formula=formula,
            )
            if not by_base_contrast.empty:
                write_table(by_base_contrast, output_dir / "lmm_by_base_contrast.tsv")

    should_run_pooled = args.allow_pooled
    if valid_base:
        unique_contrasts = sorted({str(item) for item in frame["base_contrast"].dropna().unique() if str(item).strip()})
        should_run_pooled = should_run_pooled or len(unique_contrasts) <= 1
    pooled_note = "skipped_due_to_mixed_roi_families"
    if should_run_pooled:
        pooled_fit = fit_model(frame, formula=formula)
        pooled_params = write_single_model(
            output_dir / "overall",
            pooled_fit,
            frame,
            {"analysis_level": "overall", "analysis_name": "overall"},
        )
        write_table(pooled_params, output_dir / "lmm_overall.tsv")
        pooled_note = "included"

    save_json(
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "roi_set": getattr(cfg, "ROI_SET", ""),
            "n_rows": int(len(frame)),
            "n_roi": int(frame["roi"].nunique()) if "roi" in frame.columns else 0,
            "n_base_contrast": int(frame["base_contrast"].nunique()) if "base_contrast" in frame.columns else 0,
            "pooled_model": pooled_note,
            "generated_outputs": {
                "by_roi": (output_dir / "lmm_by_roi.tsv").exists(),
                "by_base_contrast": (output_dir / "lmm_by_base_contrast.tsv").exists(),
                "overall": (output_dir / "lmm_overall.tsv").exists(),
            },
        },
        output_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
