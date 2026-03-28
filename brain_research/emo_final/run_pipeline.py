#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="emo_final pipeline runner (config-driven)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=str, default=None, help="可选：逗号分隔步骤名，仅执行这些步骤")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("config 必须是 JSON object")
    return obj


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    return bool(x)


def _as_str(x: Any, default: Optional[str] = None) -> Optional[str]:
    if x is None:
        return default
    return str(x)


def _as_int(x: Any, default: int) -> int:
    if x is None:
        return int(default)
    return int(x)


def _as_float(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    return float(x)


def _as_path(x: Any, default: Optional[Path] = None) -> Optional[Path]:
    if x is None:
        return default
    return Path(str(x))


def resolve_steps(cfg: Dict[str, Any], steps_arg: Optional[str]) -> List[str]:
    if steps_arg is not None:
        # Allow running a subset of steps from CLI without editing the JSON config.
        return [s.strip() for s in str(steps_arg).split(",") if s.strip()]
    steps = cfg.get("steps")
    if isinstance(steps, list):
        return [str(s) for s in steps]
    return [
        "lss",
        "repr_trial",
        "repr_emotion",
        "beh_repr_trial",
        "beh_repr_emotion",
        "isc_trial",
        "isc_emotion",
        "beh_isc_trial",
        "beh_isc_emotion",
        "perm_trial",
        "perm_emotion",
        "brain_beh_trial",
        "brain_beh_emotion",
        "plot_sig_trial",
        "plot_sig_emotion",
        "plot_brain_trial",
        "plot_brain_emotion",
        "plot_traj_trial",
        "plot_traj_emotion",
    ]


def fisher_z_policy(isc_method: str, fisher_z_cfg: Optional[Any]) -> bool:
    # Default Fisher-Z policy:
    # - Correlation-based ISC: Fisher-Z is meaningful (stabilize variance).
    # - Distance-based ISC (euclidean/mahalanobis): Fisher-Z is not applicable by default.
    if fisher_z_cfg is not None:
        return bool(fisher_z_cfg)
    m = str(isc_method).strip().lower()
    if m in {"pearson", "spearman"}:
        return True
    if m in {"euclidean", "mahalanobis"}:
        return False
    raise ValueError(f"未知 isc_method: {isc_method}")


def main() -> None:
    # Make project modules importable when running as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    args = parse_args()
    cfg = read_json(Path(args.config))
    steps = resolve_steps(cfg, args.steps)

    if "lss" in steps and _as_bool(cfg.get("lss", {}).get("run"), default=False):
        from lss_main import run_lss

        lss_cfg = cfg.get("lss", {})
        subject_file = _as_path(lss_cfg.get("subject_file"), default=None)
        write_mode = _as_str(lss_cfg.get("write_mode"), default="overwrite")
        # Step: LSS single-trial GLM to produce per-trial betas + aligned index CSVs.
        run_lss(subject_file=subject_file, write_mode=str(write_mode))

    if "repr_trial" in steps and _as_bool(cfg.get("repr_trial", {}).get("run"), default=False):
        from build_roi_repr_matrix import run as run_repr

        s1 = cfg.get("repr_trial", {})
        # Step: build trial-level ROI RSMs per stimulus_type (by_stimulus/<stimulus_type>/).
        run_repr(
            lss_root=_as_path(s1.get("lss_root")),
            out_dir=_as_path(s1.get("out_dir")),
            roi_set=_as_int(s1.get("roi_set"), 232),
            stim_type_col=_as_str(s1.get("stim_type_col"), default="raw_emotion"),
            stim_id_col=_as_str(s1.get("stim_id_col"), default="stimulus_content"),
            threshold_ratio=_as_float(s1.get("threshold_ratio"), 0.80),
            rsm_method=_as_str(s1.get("rsm_method"), default="spearman"),
            cocktail_blank=_as_bool(s1.get("cocktail_blank"), default=True),
            save_pattern=_as_bool(s1.get("save_pattern"), default=True),
            pattern_prefix=_as_str(s1.get("pattern_prefix"), default=None),
        )

    if "repr_emotion" in steps and _as_bool(cfg.get("repr_emotion", {}).get("run"), default=False):
        from build_roi_emotion_repr_matrix import run as run_emo

        em = cfg.get("repr_emotion", {})
        roi_set = _as_int(em.get("roi_set"), 232)
        emotions = em.get("emotions", None)
        if not isinstance(emotions, list) or len(emotions) == 0:
            emotions = ["Anger", "Disgust", "Fear", "Sad"]
        out_prefix = _as_str(em.get("out_prefix"), default=None)
        if out_prefix is None:
            out_prefix = f"roi_repr_matrix_{roi_set}_emotion{len(emotions)}"
        # Step: aggregate trial-level patterns to emotion-level patterns, then recompute emotion-level RSMs.
        run_emo(
            matrix_dir=_as_path(em.get("matrix_dir")),
            stimulus_dir_name=_as_str(em.get("stimulus_dir_name"), default="by_stimulus"),
            roi_set=int(roi_set),
            pattern_prefix=_as_str(em.get("pattern_prefix"), default=f"roi_beta_patterns_{roi_set}"),
            emotions=[str(x) for x in emotions],
            out_stimulus_dir_name=_as_str(em.get("out_stimulus_dir_name"), default="by_emotion"),
            out_prefix=str(out_prefix),
            rsm_method=_as_str(em.get("rsm_method"), default="spearman"),
            cocktail_blank=_as_bool(em.get("cocktail_blank"), default=True),
            save_pattern=_as_bool(em.get("save_pattern"), default=True),
        )

    def run_beh_repr(block_name: str) -> None:
        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return

        if block_name == "beh_repr_trial":
            from behavior.build_behavior_trial_repr_matrix import run as run_beh_repr_trial

            feature_cols = s.get("feature_cols", None)
            if not isinstance(feature_cols, list) or len(feature_cols) == 0:
                feature_cols = ["emot_rating"]
            run_beh_repr_trial(
                matrix_dir=_as_path(s.get("matrix_dir")),
                stimulus_dir_name=_as_str(s.get("stimulus_dir_name"), default="by_stimulus"),
                brain_repr_prefix=_as_str(s.get("brain_repr_prefix"), default="roi_repr_matrix_232"),
                beh_data_dir=_as_path(s.get("beh_data_dir"), default=Path("/public/home/dingrui/fmri_analysis/data/beh")),
                fmri_data_dir=_as_path(s.get("fmri_data_dir"), default=Path("/public/home/dingrui/BIDS_DATA")),
                participants_file=_as_path(s.get("participants_file"), default=None),
                valid_er_file=_as_path(s.get("valid_er_file"), default=None),
                valid_tg_file=_as_path(s.get("valid_tg_file"), default=None),
                feature_cols=[str(x) for x in feature_cols],
                agg_func=_as_str(s.get("agg_func"), default="mean"),
                diff_method=_as_str(s.get("diff_method"), default="euclidean"),
                score_file_name=_as_str(s.get("score_file_name"), default="behavior_subject_stimulus_scores.csv"),
                pattern_prefix=_as_str(s.get("pattern_prefix"), default="behavior_patterns_trial"),
                diff_prefix=_as_str(s.get("diff_prefix"), default="behavior_diff_matrix_trial"),
                repr_prefix=_as_str(s.get("repr_prefix"), default="behavior_repr_matrix_trial"),
            )
            return

        if block_name == "beh_repr_emotion":
            from behavior.build_behavior_emotion_repr_matrix import run as run_beh_repr_emotion

            emotions = s.get("emotions", None)
            if not isinstance(emotions, list) or len(emotions) == 0:
                emotions = ["Anger", "Disgust", "Fear", "Sad"]
            run_beh_repr_emotion(
                matrix_dir=_as_path(s.get("matrix_dir")),
                stimulus_dir_name=_as_str(s.get("stimulus_dir_name"), default="by_stimulus"),
                pattern_prefix=_as_str(s.get("pattern_prefix"), default="behavior_patterns_trial"),
                emotions=[str(x) for x in emotions],
                diff_method=_as_str(s.get("diff_method"), default="euclidean"),
                out_stimulus_dir_name=_as_str(s.get("out_stimulus_dir_name"), default="by_emotion"),
                score_file_name=_as_str(s.get("score_file_name"), default="behavior_subject_emotion_scores.csv"),
                out_pattern_prefix=_as_str(s.get("out_pattern_prefix"), default="behavior_patterns_emotion4"),
                out_diff_prefix=_as_str(s.get("out_diff_prefix"), default="behavior_diff_matrix_emotion4"),
                out_repr_prefix=_as_str(s.get("out_repr_prefix"), default="behavior_repr_matrix_emotion4"),
            )
            return

    def run_isc(block_name: str) -> None:
        from calc_roi_isc_by_age import run as run_isc_inner

        s2 = cfg.get(block_name, {})
        if not _as_bool(s2.get("run"), default=False):
            return
        matrix_dir = _as_path(s2.get("matrix_dir"))
        stimulus_dir_name = _as_str(s2.get("stimulus_dir_name"), default="by_stimulus")
        subject_info = _as_path(s2.get("subject_info"))
        repr_prefix = _as_str(s2.get("repr_prefix"), default="roi_repr_matrix_232")
        isc_method = _as_str(s2.get("isc_method"), default="mahalanobis")
        isc_prefix = _as_str(s2.get("isc_prefix"), default=None)
        if isc_prefix is None:
            # Convention: roi_isc_<method>_by_age is the canonical output prefix.
            isc_prefix = f"roi_isc_{isc_method}_by_age"
        run_isc_inner(
            matrix_dir=matrix_dir,
            stimulus_dir_name=stimulus_dir_name,
            subject_info=subject_info,
            repr_prefix=repr_prefix,
            isc_method=isc_method,
            isc_prefix=isc_prefix,
        )

    def run_beh_isc(block_name: str) -> None:
        from behavior.calc_behavior_isc_by_age import run as run_beh_isc_inner

        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return
        matrix_dir = _as_path(s.get("matrix_dir"))
        stimulus_dir_name = _as_str(s.get("stimulus_dir_name"), default="by_stimulus")
        subject_info = _as_path(s.get("subject_info"))
        repr_prefix = _as_str(s.get("repr_prefix"), default="behavior_repr_matrix_trial")
        isc_method = _as_str(s.get("isc_method"), default="mahalanobis")
        isc_prefix = _as_str(s.get("isc_prefix"), default=None)
        if isc_prefix is None:
            isc_prefix = f"behavior_isc_{isc_method}_by_age"
        run_beh_isc_inner(
            matrix_dir=matrix_dir,
            stimulus_dir_name=stimulus_dir_name,
            subject_info=subject_info,
            repr_prefix=repr_prefix,
            isc_method=isc_method,
            isc_prefix=isc_prefix,
        )

    def run_perm(block_name: str) -> None:
        from joint_analysis_roi_isc_dev_models import run as run_perm_inner

        s3 = cfg.get(block_name, {})
        if not _as_bool(s3.get("run"), default=False):
            return
        matrix_dir = _as_path(s3.get("matrix_dir"))
        stimulus_dir_name = _as_str(s3.get("stimulus_dir_name"), default="by_stimulus")
        isc_prefix = _as_str(s3.get("isc_prefix"), default=None)
        isc_method = _as_str(s3.get("isc_method"), default=None)
        assoc_method = _as_str(s3.get("assoc_method"), default="spearman")
        correction_mode = _as_str(s3.get("correction_mode"), default="fdr_only")
        normalize_models = _as_bool(s3.get("normalize_models"), default=True)
        n_perm = _as_int(s3.get("n_perm"), 5000)
        seed = _as_int(s3.get("seed"), 42)
        repr_prefix = _as_str(s3.get("repr_prefix"), default=None)

        isc_method_for_policy = _as_str(s3.get("isc_method_for_policy"), default=None)
        if isc_method_for_policy is None:
            isc_method_for_policy = _as_str(s3.get("isc_method"), default=None) or "mahalanobis"
        fisher_z_enabled = fisher_z_policy(isc_method=str(isc_method_for_policy), fisher_z_cfg=s3.get("fisher_z"))

        if str(isc_method_for_policy).strip().lower() in {"euclidean", "mahalanobis"}:
            # Distance-based ISC outputs are identified by isc_prefix; isc_method is irrelevant for auto-detection.
            isc_method = None
            if isc_prefix is None:
                isc_prefix = f"roi_isc_{str(isc_method_for_policy).strip().lower()}_by_age"

        # Step: permutation test against developmental models (one-tailed) with model-wise maxT FWER.
        run_perm_inner(
            matrix_dir=matrix_dir,
            stimulus_dir_name=stimulus_dir_name,
            n_perm=int(n_perm),
            seed=int(seed),
            isc_prefix=isc_prefix,
            isc_method=isc_method,
            normalize_models=bool(normalize_models),
            fisher_z_enabled=bool(fisher_z_enabled),
            assoc_method=str(assoc_method),
            repr_prefix=repr_prefix,
            correction_mode=str(correction_mode),
        )

    def run_brain_beh(block_name: str) -> None:
        from behavior.joint_analysis_roi_isc_behavior import run as run_brain_beh_inner

        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return
        run_brain_beh_inner(
            matrix_dir=_as_path(s.get("matrix_dir")),
            stimulus_dir_name=_as_str(s.get("stimulus_dir_name"), default="by_stimulus"),
            brain_repr_prefix=_as_str(s.get("brain_repr_prefix"), default=None),
            brain_isc_method=_as_str(s.get("brain_isc_method"), default="mahalanobis"),
            brain_isc_prefix=_as_str(s.get("brain_isc_prefix"), default=None),
            behavior_repr_prefix=_as_str(s.get("behavior_repr_prefix"), default=None),
            behavior_isc_method=_as_str(s.get("behavior_isc_method"), default="mahalanobis"),
            behavior_isc_prefix=_as_str(s.get("behavior_isc_prefix"), default=None),
            assoc_method=_as_str(s.get("assoc_method"), default="spearman"),
            correction_mode=_as_str(s.get("correction_mode"), default="perm_fwer_fdr"),
            n_perm=_as_int(s.get("n_perm"), 5000),
            seed=_as_int(s.get("seed"), 42),
            fisher_z_brain=_as_str(s.get("fisher_z_brain"), default=None),
            fisher_z_behavior=_as_str(s.get("fisher_z_behavior"), default=None),
        )

    def _iter_stimulus_types(matrix_dir: Path, stimulus_dir_name: str, stimulus_types: Optional[Any]) -> List[str]:
        by_stim = Path(matrix_dir) / str(stimulus_dir_name)
        if not by_stim.exists():
            return []
        if isinstance(stimulus_types, list) and len(stimulus_types) > 0:
            return [str(x) for x in stimulus_types]
        return sorted([p.name for p in by_stim.iterdir() if p.is_dir()])

    def run_plot_sig(block_name: str) -> None:
        from plot_roi_isc_dev_models_perm_sig import collect_results, draw, resolve_sig_method

        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return
        matrix_dir = _as_path(s.get("matrix_dir"))
        stimulus_dir_name = _as_str(s.get("stimulus_dir_name"), default="by_stimulus")
        repr_prefix = _as_str(s.get("repr_prefix"), default=None)
        alpha = _as_float(s.get("alpha"), 0.05)
        dpi = _as_int(s.get("dpi"), 300)
        sig_method = _as_str(s.get("sig_method"), default=None)
        fdr_mode = _as_str(s.get("fdr_mode"), default="model_wise")
        positive_only = _as_bool(s.get("positive_only"), default=False)
        sig_method_resolved = resolve_sig_method(sig_method=sig_method, fdr_mode=str(fdr_mode))
        df = collect_results(
            matrix_dir=Path(matrix_dir),
            stimulus_dir_name=str(stimulus_dir_name),
            alpha=float(alpha),
            sig_method=str(sig_method_resolved),
            positive_only=bool(positive_only),
            repr_prefix=repr_prefix,
        )
        out_dir = Path(matrix_dir) / "figures" / str(stimulus_dir_name)
        draw(
            df=df,
            out_dir=out_dir,
            alpha=float(alpha),
            dpi=int(dpi),
            sig_method=str(sig_method_resolved),
            positive_only=bool(positive_only),
        )

    def run_plot_brain(block_name: str) -> None:
        import pandas as pd

        from plot_brain_surface_vol import apply_significance, has_repr_files, load_atlas, map_values_to_brain, save_and_plot_maps

        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return
        matrix_dir = Path(_as_path(s.get("matrix_dir")))
        stimulus_dir_name = _as_str(s.get("stimulus_dir_name"), default="by_stimulus")
        repr_prefix = _as_str(s.get("repr_prefix"), default=None)
        models = s.get("models", None)
        if not isinstance(models, list) or len(models) == 0:
            models = ["M_nn", "M_conv", "M_div"]
        stimulus_types = _iter_stimulus_types(matrix_dir, stimulus_dir_name=str(stimulus_dir_name), stimulus_types=s.get("stimulus_types"))
        sig_method = _as_str(s.get("sig_method"), default="fwer")
        sig_col = _as_str(s.get("sig_col"), default="p_perm_one_tailed")
        alpha = _as_float(s.get("alpha"), 0.05)
        no_preview = _as_bool(s.get("no_preview"), default=True)

        surf_l_data, surf_r_data, vol_img, vol_data = load_atlas()
        out_dir = matrix_dir / "figures" / f"{str(stimulus_dir_name)}_brain_maps"

        for stim_type in stimulus_types:
            stim_dir = matrix_dir / str(stimulus_dir_name) / str(stim_type)
            if repr_prefix is not None and not has_repr_files(stim_dir, repr_prefix=str(repr_prefix)):
                print(f"[SKIP] {stim_dir.name}: 缺少 {repr_prefix} 对应输入文件，跳过。")
                continue
            result_csv = stim_dir / "roi_isc_dev_models_perm_fwer.csv"
            if not result_csv.exists():
                print(f"[SKIP] {stim_dir.name}: 缺少置换结果 roi_isc_dev_models_perm_fwer.csv，跳过。")
                continue
            df = pd.read_csv(result_csv)
            for m in models:
                df_sig = apply_significance(df=df, model=str(m), method=str(sig_method), sig_col=str(sig_col), alpha=float(alpha))
                l_arr, r_arr, vol_arr = map_values_to_brain(df_sig, surf_l_data, surf_r_data, vol_data)
                vmax = max(0.05, float(df_sig["r_obs"].abs().quantile(0.99))) if not df_sig.empty else 0.05
                prefix = f"brain_map_{stimulus_dir_name}_{stim_type}_{m}_{sig_method}_a{alpha:g}"
                save_and_plot_maps(out_dir, prefix, l_arr, r_arr, vol_img, vol_arr, vmax, preview=not bool(no_preview))

    def run_plot_traj(block_name: str) -> None:
        from plot_roi_isc_age_trajectory import (
            has_repr_files,
            load_pair_data,
            plot_one,
            resolve_isc_prefix,
            select_top_rois,
        )

        s = cfg.get(block_name, {})
        if not _as_bool(s.get("run"), default=False):
            return
        matrix_dir = Path(_as_path(s.get("matrix_dir")))
        stimulus_dir_name = _as_str(s.get("stimulus_dir_name"), default="by_stimulus")
        repr_prefix = _as_str(s.get("repr_prefix"), default=None)
        models = s.get("models", None)
        if not isinstance(models, list) or len(models) == 0:
            models = ["M_nn", "M_conv", "M_div"]
        stimulus_types = _iter_stimulus_types(matrix_dir, stimulus_dir_name=str(stimulus_dir_name), stimulus_types=s.get("stimulus_types"))

        isc_method = _as_str(s.get("isc_method"), default="mahalanobis")
        isc_prefix = _as_str(s.get("isc_prefix"), default=None)
        alpha = _as_float(s.get("alpha"), 0.05)
        method = _as_str(s.get("method"), default="fdr_model_wise")
        top_k = _as_int(s.get("top_k"), 5)
        positive_only = _as_bool(s.get("positive_only"), default=False)
        fit = _as_str(s.get("fit"), default="lowess")
        lowess = _as_bool(s.get("lowess"), default=False)
        plot_mode = _as_str(s.get("plot_mode"), default="hexbin")
        max_points = _as_int(s.get("max_points"), 40000)
        fit_max_points = _as_int(s.get("fit_max_points"), 5000)
        hexbin_gridsize = _as_int(s.get("hexbin_gridsize"), 55)
        seed = _as_int(s.get("seed"), 42)
        dpi = _as_int(s.get("dpi"), 300)

        out_dir = matrix_dir / "figures" / f"{str(stimulus_dir_name)}_pair_age_traj"
        out_dir.mkdir(parents=True, exist_ok=True)

        for stim_type in stimulus_types:
            stim_dir = matrix_dir / str(stimulus_dir_name) / str(stim_type)
            if not stim_dir.exists():
                continue
            if repr_prefix is not None and not has_repr_files(stim_dir, repr_prefix=str(repr_prefix)):
                print(f"[SKIP] {stim_dir.name}: 缺少 {repr_prefix} 对应输入文件，跳过。")
                continue
            result_csv = stim_dir / "roi_isc_dev_models_perm_fwer.csv"
            if not result_csv.exists():
                print(f"[SKIP] {stim_dir.name}: 缺少置换结果 roi_isc_dev_models_perm_fwer.csv，跳过。")
                continue
            prefix_resolved = resolve_isc_prefix(stim_dir, isc_prefix=isc_prefix, isc_method=isc_method)
            for m in models:
                try:
                    top = select_top_rois(
                        result_csv=result_csv,
                        model=str(m),
                        method=str(method),
                        alpha=float(alpha),
                        top_k=int(top_k),
                        positive_only=bool(positive_only),
                    )
                except Exception as e:
                    print(f"[SKIP] {stim_dir.name} {m}: {e}")
                    continue
                fit_eff = "lowess" if bool(lowess) else str(fit)
                method_tag = str(method) + ("_pos" if bool(positive_only) else "")
                for _, row in top.iterrows():
                    roi = str(row["roi"])
                    x, y = load_pair_data(stim_dir, roi, isc_prefix=str(prefix_resolved))
                    if x.size <= 10:
                        continue
                    out_path = out_dir / f"emo_final_pair_age_traj__{stimulus_dir_name}__{stim_type}__{m}__{roi}__{method_tag}_a{alpha:g}.png"
                    title = f"{stim_type} | {m} | {roi}"
                    plot_one(
                        out_path=out_path,
                        x=x,
                        y=y,
                        title=title,
                        y_label="ISC",
                        fit=fit_eff,
                        plot_mode=str(plot_mode),
                        max_points=int(max_points),
                        fit_max_points=int(fit_max_points),
                        hexbin_gridsize=int(hexbin_gridsize),
                        seed=int(seed),
                        dpi=int(dpi),
                    )

    if "beh_repr_trial" in steps:
        run_beh_repr("beh_repr_trial")
    if "beh_repr_emotion" in steps:
        run_beh_repr("beh_repr_emotion")
    if "isc_trial" in steps:
        run_isc("isc_trial")
    if "isc_emotion" in steps:
        run_isc("isc_emotion")
    if "beh_isc_trial" in steps:
        run_beh_isc("beh_isc_trial")
    if "beh_isc_emotion" in steps:
        run_beh_isc("beh_isc_emotion")
    if "perm_trial" in steps:
        run_perm("perm_trial")
    if "perm_emotion" in steps:
        run_perm("perm_emotion")
    if "brain_beh_trial" in steps:
        run_brain_beh("brain_beh_trial")
    if "brain_beh_emotion" in steps:
        run_brain_beh("brain_beh_emotion")
    if "plot_sig_trial" in steps:
        run_plot_sig("plot_sig_trial")
    if "plot_sig_emotion" in steps:
        run_plot_sig("plot_sig_emotion")
    if "plot_brain_trial" in steps:
        run_plot_brain("plot_brain_trial")
    if "plot_brain_emotion" in steps:
        run_plot_brain("plot_brain_emotion")
    if "plot_traj_trial" in steps:
        run_plot_traj("plot_traj_trial")
    if "plot_traj_emotion" in steps:
        run_plot_traj("plot_traj_emotion")


if __name__ == "__main__":
    main()
