#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_fslr_alignment.py

用途
-
审计 surface(fsLR) 全流程的“顶点级对齐”是否一致，避免出现：
- 左右半球 label.gii 与 beta / timeseries 顶点数不匹配
- 输入 time series 被意外读成 1D（只有 1 个 timepoint）
- beta 文件路径解析错误导致加载到非本被试/本任务的文件

检查内容（抽样若干被试）
-
对每个 surface 场景（surface_L / surface_R）：
1) 从 cfg.get_paths() 找到对应 half 的 time series（*_space-fsLR_*.shape.gii）
2) 读取 time series，拿到 (n_vertices, n_time)
3) 读取对应半球 Schaefer label.gii，拿到 n_vertices_atlas
4) 从 aligned.csv 抽一条 beta 记录，读取 beta 的 n_vertices_beta
5) 要求：n_vertices_time_series == n_vertices_beta == n_vertices_atlas

使用
-
把顶部 ATLAS_SURF_L/ATLAS_SURF_R 改成你实际 atlas 路径后，直接运行：
python brain_research/emo/audit_fslr_alignment.py

返回码
-
若任何场景存在 mismatch，会以退出码 2 退出（便于批处理脚本捕捉失败）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import surface

from glm_config import get_scenario_configs


ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")

N_SAMPLE_SUBJECTS = 5


@dataclass(frozen=True)
class AuditResult:
    scenario: str
    ok: bool
    msg: str


def _load_label_len(path: Path) -> int:
    # label.gii: 每个顶点一个整数 label（parcel 编号）
    g = nib.load(str(path))
    return int(np.asarray(g.darrays[0].data).reshape(-1).shape[0])


def _resolve_beta_path(output_root: Path, row: pd.Series) -> Path:
    # aligned.csv 的 row 里 file 通常是相对文件名（例如 beta_xxx_L.gii）
    # 这里按 OUTPUT_ROOT/{emo|soc}/{sub}/{file} 拼出路径
    task = str(row.get("task", "")).upper()
    sub = str(row.get("subject", ""))
    fname = str(row.get("file", ""))
    folder = "emo" if task == "EMO" else "soc"
    return output_root / folder / sub / fname


def _summarize_time_series(fmri_path: Path) -> Tuple[int, int]:
    # 期望 time series 为 2D： (n_vertices, n_time)
    # 若读到 1D，则说明该文件可能是静态 map（或只有 1 个 timepoint）
    x = surface.load_surf_data(str(fmri_path))
    x = np.asarray(x)
    if x.ndim == 1:
        return int(x.shape[0]), 1
    return int(x.shape[0]), int(x.shape[1])


def _summarize_beta(beta_path: Path) -> int:
    # beta gii 默认只写一个 darray，长度应为 n_vertices
    g = nib.load(str(beta_path))
    v = np.asarray(g.darrays[0].data).reshape(-1)
    return int(v.shape[0])


def audit_one_scenario(cfg) -> AuditResult:
    # 只审计 surface_L / surface_R；volume 直接跳过
    scenario = f"{cfg.DATA_SPACE}_{cfg.HEMI}" if cfg.DATA_SPACE == "surface" else "volume"
    if cfg.DATA_SPACE != "surface":
        return AuditResult(scenario=scenario, ok=True, msg="skip(volume)")

    aligned_csv = cfg.OUTPUT_ROOT / f"lss_index_{cfg.SCENARIO_ID}_aligned.csv"
    if not aligned_csv.exists():
        return AuditResult(scenario=scenario, ok=False, msg=f"aligned csv not found: {aligned_csv}")

    atlas_path = ATLAS_SURF_L if cfg.HEMI == "L" else ATLAS_SURF_R
    if not atlas_path.exists():
        return AuditResult(scenario=scenario, ok=False, msg=f"atlas not found: {atlas_path}")

    atlas_n = _load_label_len(atlas_path)
    df = pd.read_csv(aligned_csv)
    df = df[df["task"].isin(["EMO", "SOC"])].copy()
    if df.empty:
        return AuditResult(scenario=scenario, ok=False, msg="aligned csv empty after task filter")

    subs = sorted(df["subject"].astype(str).unique().tolist())[: int(N_SAMPLE_SUBJECTS)]
    if not subs:
        return AuditResult(scenario=scenario, ok=False, msg="no subjects in aligned csv")

    problems: List[str] = []
    for sub in subs:
        try:
            # 用 cfg.get_paths(sub, run=1) 抽取一条半球对应的 time series，检查顶点数与 time 维度
            fmri_path_s, _, _ = cfg.get_paths(sub, 1)
            fmri_path = Path(fmri_path_s)
            if not fmri_path.exists():
                problems.append(f"{sub}: fmri missing: {fmri_path}")
                continue
            n_vert_ts, n_time = _summarize_time_series(fmri_path)
            if n_vert_ts != atlas_n:
                problems.append(f"{sub}: vertices mismatch (timeseries={n_vert_ts}, atlas={atlas_n}), time={n_time}")

            # 从 aligned.csv 抽一条 beta 记录，检查 beta 顶点数是否与 atlas 一致
            sub_rows = df[df["subject"].astype(str) == sub].head(1)
            if sub_rows.empty:
                continue
            beta_path = _resolve_beta_path(cfg.OUTPUT_ROOT, sub_rows.iloc[0])
            if not beta_path.exists():
                problems.append(f"{sub}: beta missing: {beta_path}")
                continue
            n_vert_beta = _summarize_beta(beta_path)
            if n_vert_beta != atlas_n:
                problems.append(f"{sub}: vertices mismatch (beta={n_vert_beta}, atlas={atlas_n})")
        except Exception as e:
            problems.append(f"{sub}: exception: {e}")

    if problems:
        msg = " | ".join(problems[:8])
        if len(problems) > 8:
            msg += f" | ...(+{len(problems) - 8})"
        return AuditResult(scenario=scenario, ok=False, msg=msg)

    return AuditResult(scenario=scenario, ok=True, msg=f"OK (atlas_n={atlas_n})")


def main() -> None:
    # get_scenario_configs() 会返回 surface_L, surface_R, volume 三个场景
    configs = get_scenario_configs()
    results: List[AuditResult] = []
    for cfg in configs:
        results.append(audit_one_scenario(cfg))

    ok_all = True
    for r in results:
        print(f"[{r.scenario}] ok={r.ok} | {r.msg}")
        ok_all = ok_all and r.ok
    if not ok_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
