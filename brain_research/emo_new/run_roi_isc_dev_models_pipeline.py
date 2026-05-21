#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emo_new 全局串联脚本：从平均 beta 到统计图的一键流程。

本脚本把 `brain_research/emo_new` 目录中的核心分析脚本统一为一个可控流水线，
目标是：
- 统一执行入口（避免手工逐条运行导致参数不一致）
- 把“统计结果计算”与“显著性可视化”纳入同一套 stage 管理
- 通过详细注释说明每一步输入/输出与依赖关系，便于后续维护

========================
一、emo_new 全局代码结构（建议执行顺序）
========================

Step 0（可选，预处理）
----------------------
`calc_mean_beta_by_stimulus_type.py`
- 输入：LSS 已计算好的 trial beta（`lss_index_*_aligned.csv`）
- 输出：按 stimulus_type 聚合后的 mean_beta 图，以及
  `avg_beta_index_surface_L.csv / avg_beta_index_surface_R.csv`
- 何时需要：当平均 beta 索引文件不存在，或者你更换了 stimulus_type 定义时

Step 1（ROI 特征构建）
----------------------
`build_roi_beta_matrix_200.py`
- 输入：Step0 输出的 avg_beta_index_surface_L/R.csv
- 输出：`<roi_results>/by_stimulus/<stim>/roi_beta_matrix_200.npz`
- 核心：每个 ROI 不是单一均值，而是 ROI 顶点向量（保留空间模式）

Step 2（ROI ISC）
-----------------
`calc_roi_isc_by_age.py`
- 输入：Step1 的 roi_beta_matrix_200.npz + 被试年龄表
- 输出：`roi_isc_spearman_by_age.npy` 与对应 subjects/rois csv
- 核心：对每个 ROI 计算被试×被试 Spearman 相似性矩阵，并按年龄排序

Step 3（发展模型统计）
----------------------
`joint_analysis_roi_isc_dev_models_perm_fwer.py`
- 输入：Step2 ISC 结果
- 输出：`roi_isc_dev_models_perm_fwer.csv`（每个 stimulus_type 一份）
- 核心：将 ROI ISC 上三角向量与 3 个发展模型（M_nn / M_conv / M_div）做置换检验
  并进行 model-wise FWER（单尾正效应）

Step 4（可选，绘图与结果整理）
-------------------------------
- `plot_roi_isc_dev_models_perm_fwer_sig.py`：基于 FWER 显著结果绘制热图
- `plot_roi_isc_dev_models_perm_fdr_sig.py`：基于置换原始 p 重新做 BH-FDR 并绘图

备注：
- Step4 不影响统计主结果（Step3），但对论文图与结果汇报非常关键。
- 若仅需数值结果，可关闭 plot stage 提高速度。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_roi_beta_matrix_200 import run as run_build
from calc_roi_isc_by_age import run as run_isc
from joint_analysis_roi_isc_dev_models_perm_fwer import run as run_joint
from plot_roi_isc_dev_models_perm_fdr_sig import plot_each_condition as plot_fdr_each_condition
from plot_roi_isc_dev_models_perm_fdr_sig import plot_three_conditions as plot_fdr_three_conditions
from plot_roi_isc_dev_models_perm_fwer_sig import plot_each_condition as plot_fwer_each_condition
from plot_roi_isc_dev_models_perm_fwer_sig import plot_three_conditions as plot_fwer_three_conditions


DEFAULT_AVG_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/averaged_betas_by_stimulus")
DEFAULT_ROI_RESULTS_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
DEFAULT_SUBJECT_INFO = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "emo_new 全流程串联：Step1 ROI矩阵 → Step2 ROI ISC → Step3 发展模型置换+FWER "
            "→ Step4 FWER/FDR显著热图"
        )
    )

    # 数据路径
    p.add_argument("--avg-dir", type=Path, default=DEFAULT_AVG_DIR)
    p.add_argument("--roi-results-dir", type=Path, default=DEFAULT_ROI_RESULTS_DIR)
    p.add_argument("--subject-info", type=Path, default=DEFAULT_SUBJECT_INFO)

    # Step1 参数
    p.add_argument("--threshold-ratio", type=float, default=0.80)

    # Step3 参数
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)

    # Step4 参数（绘图）
    p.add_argument("--plot-alpha", type=float, default=0.05, help="FWER/FDR 显著性阈值")
    p.add_argument(
        "--plot-stimuli",
        type=str,
        default=None,
        help=(
            "指定绘图条件，逗号分隔（例如 fear,happy,sad）。默认自动选择。"
            "若使用 --plot-three-only，则必须恰好 3 个。"
        ),
    )
    p.add_argument("--plot-three-only", action="store_true", help="仅绘制三条件并排图")
    p.add_argument("--plot-dpi", type=int, default=300)
    p.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"])

    # Stage 选择
    p.add_argument(
        "--stages",
        type=str,
        default="1,2,3,4",
        help=(
            "要执行的 stage，逗号分隔，可选 1/2/3/4。"
            "默认执行全部；例如 --stages 2,3 表示复用已有 Step1 输出。"
        ),
    )
    return p.parse_args()


def _parse_stages(text: str) -> set[int]:
    raw = [x.strip() for x in str(text).split(",") if x.strip()]
    if not raw:
        raise ValueError("--stages 不能为空")
    stages: set[int] = set()
    for token in raw:
        if token not in {"1", "2", "3", "4"}:
            raise ValueError(f"不支持的 stage: {token}（仅支持 1/2/3/4）")
        stages.add(int(token))
    return stages


def main() -> None:
    args = parse_args()
    stages = _parse_stages(args.stages)

    # Step 1：构建 ROI 顶点向量矩阵
    if 1 in stages:
        run_build(args.avg_dir, args.roi_results_dir, float(args.threshold_ratio))

    # Step 2：计算按龄排序 ROI ISC
    if 2 in stages:
        run_isc(args.roi_results_dir, args.subject_info)

    # Step 3：置换检验 + model-wise FWER
    if 3 in stages:
        run_joint(
            matrix_dir=args.roi_results_dir,
            out_dir=args.roi_results_dir,
            n_perm=int(args.n_perm),
            seed=int(args.seed),
            normalize_models=bool(args.normalize_models),
        )

    # Step 4：生成显著结果图（FWER 与 FDR 两套）
    if 4 in stages:
        fig_dir = args.roi_results_dir / "figures"
        if bool(args.plot_three_only):
            # 4A) FWER 三条件并排图
            plot_fwer_three_conditions(
                roi_root=args.roi_results_dir,
                out_dir=fig_dir,
                stimuli_arg=args.plot_stimuli,
                alpha=float(args.plot_alpha),
                positive_only=True,
                atlas_l=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                atlas_r=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                dpi=int(args.plot_dpi),
                fmt=args.plot_format,
            )
            # 4B) FDR 三条件并排图
            plot_fdr_three_conditions(
                roi_root=args.roi_results_dir,
                out_dir=fig_dir,
                stimuli_arg=args.plot_stimuli,
                alpha=float(args.plot_alpha),
                positive_only=True,
                fdr_mode="model_wise",
                atlas_l=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                atlas_r=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                dpi=int(args.plot_dpi),
                fmt=args.plot_format,
            )
        else:
            # 默认输出“全条件单图”结果，更适合完整汇总
            plot_fwer_each_condition(
                roi_root=args.roi_results_dir,
                out_dir=fig_dir,
                stimuli_arg=args.plot_stimuli,
                alpha=float(args.plot_alpha),
                positive_only=True,
                atlas_l=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                atlas_r=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                dpi=int(args.plot_dpi),
                fmt=args.plot_format,
            )
            plot_fdr_each_condition(
                roi_root=args.roi_results_dir,
                out_dir=fig_dir,
                stimuli_arg=args.plot_stimuli,
                alpha=float(args.plot_alpha),
                positive_only=True,
                fdr_mode="model_wise",
                atlas_l=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                atlas_r=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                dpi=int(args.plot_dpi),
                fmt=args.plot_format,
            )


if __name__ == "__main__":
    main()
