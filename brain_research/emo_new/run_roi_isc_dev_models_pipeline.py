#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emo_new 新链路串联总脚本（Step 1/3 → Step 2/3 → Step 3/3）

这份脚本用于把当前新链路的 3 个分析步骤串联起来，一条命令完成：
1) build_roi_beta_matrix_200.py
2) calc_roi_isc_by_age.py
3) joint_analysis_roi_isc_dev_models_perm_fwer.py

核心定义（与旧链路不同）：
- 输入来自 avg_beta_index_surface_L/R.csv，文件中每条记录指向 stimulus_type 级“全脑平均 beta 图”（保留空间结构的 GIFTI 向量）
- ROI 特征不是 ROI 均值标量，而是 ROI 内顶点向量（不同 ROI 顶点数不同；同 ROI 对所有被试一致）
- 每个 ROI 的被试特征向量为 concat(所有 stimulus_type 的 ROI 顶点向量)
- Step2 对每个 ROI 做 Spearman 被试×被试相似性矩阵（按年龄升序）
- Step3 将 ROI 的上三角相似性向量与 3 个发育模型（M_nn/M_conv/M_div）做置换检验：
  - 仅检验正效应（one-tailed, positive）
  - 每个模型内部做 FWER（model-wise max-stat）
  - 默认置换次数 5000

可选的 Step0（不在本脚本内自动执行）：
- calc_mean_beta_by_stimulus_type.py
  从 lss_index_*_aligned.csv 与 LSS trial beta 出发，按 stimulus_type（默认 raw_emotion）聚合生成 mean_beta.gii，
  并输出 avg_beta_index_surface_L/R.csv，供本脚本 Step1 使用。

推荐执行顺序（人工分步 / 或一键串联）：
- Step0: calc_mean_beta_by_stimulus_type.py（如 avg_beta_index_* 尚未生成）
- Step1: build_roi_beta_matrix_200.py
- Step2: calc_roi_isc_by_age.py
- Step3: joint_analysis_roi_isc_dev_models_perm_fwer.py

默认目录（与 emo 对齐）：
- roi_results: /public/home/dingrui/fmri_analysis/zz_analysis/roi_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_roi_beta_matrix_200 import run as run_build
from calc_roi_isc_by_age import run as run_isc
from joint_analysis_roi_isc_dev_models_perm_fwer import run as run_joint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="emo_new 新链路串联：ROI beta matrix → Spearman ISC → dev-model perm+FWER")
    p.add_argument(
        "--avg-dir",
        type=Path,
        default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/averaged_betas_by_stimulus"),
    )
    p.add_argument(
        "--roi-results-dir",
        type=Path,
        default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"),
    )
    p.add_argument("--threshold-ratio", type=float, default=0.80)
    p.add_argument(
        "--subject-info",
        type=Path,
        default=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
    )
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_build(args.avg_dir, args.roi_results_dir, float(args.threshold_ratio))
    run_isc(args.roi_results_dir, args.subject_info)
    run_joint(
        matrix_dir=args.roi_results_dir,
        out_dir=args.roi_results_dir,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        normalize_models=bool(args.normalize_models),
    )


if __name__ == "__main__":
    main()

