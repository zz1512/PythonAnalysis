#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emo_new: 基于已有 LSS 输出，按 trial type 做全脑平均与相似性分析。

满足需求：
1) trial type 固定为: Passive_Emo / Reappraisal / Passive_Neutral
2) 仅保留 surface_L 与 surface_R 都有可用数据的被试交集
3) 每种 trial type 保存每个被试的全脑平均结果（L/R + 拼接向量）
4) 构建 (surface_L + surface_R) × 被试 × 被试 相似性矩阵（按年龄升序）
   并输出同排序下的 3 个发育模型矩阵
5) 参数配置写在脚本顶部，默认无需命令行参数
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import surface

# ===================== 配置区（无需每次手动传参） =====================
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
INDEX_L = LSS_ROOT / "lss_index_surface_L_aligned.csv"
INDEX_R = LSS_ROOT / "lss_index_surface_R_aligned.csv"

SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")
COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

OUT_ROOT = LSS_ROOT / "emo_new_trialtype_wholebrain"

TASK_FILTER = "EMO"
TRIAL_TYPES = ("Passive_Emo", "Reappraisal", "Passive_Neutral")

# =====================================================================


def parse_age(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass

    y = re.search(r"(\d+)\s*岁", s)
    m = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d = re.search(r"(\d+)\s*天", s)
    yy = int(y.group(1)) if y else 0
    mm = int(m.group(1)) if m else 0
    dd = int(d.group(1)) if d else 0
    return yy + mm / 12.0 + dd / 365.0


def load_age_map(path: Path) -> Dict[str, float]:
    if str(path).endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    elif str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if COL_SUB_ID in df.columns and COL_AGE in df.columns:
        sid_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in df.columns and LEGACY_COL_AGE in df.columns:
        sid_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError("年龄表缺少必要列")

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        sid = str(row[sid_col]).strip()
        sid = sid if sid.startswith("sub-") else f"sub-{sid}"
        out[sid] = parse_age(row[age_col])
    return out


def infer_trial_type(row: pd.Series) -> Optional[str]:
    # 优先：从 label 解析（lss_main 输出 label 形如 Passive_Emo_tr12）
    label = str(row.get("label", "")).strip()
    m = re.match(r"^(Passive_Emo|Reappraisal|Passive_Neutral)_tr\d+$", label)
    if m:
        return m.group(1)

    # 其次：raw_condition
    cond = str(row.get("raw_condition", "")).strip()
    for t in TRIAL_TYPES:
        if cond == t:
            return t

    # 再次：trial_type 列（如果存在）
    tt = str(row.get("trial_type", "")).strip()
    if tt in TRIAL_TYPES:
        return tt

    return None


def resolve_beta_path(row: pd.Series, hemi: str) -> Path:
    raw_file = str(row["file"])
    if raw_file.startswith("/"):
        return Path(raw_file)

    task_folder = str(row.get("task", "")).lower() or "emo"
    p1 = LSS_ROOT / task_folder / str(row["subject"]) / raw_file
    if p1.exists():
        return p1

    run = row.get("run", None)
    if pd.notna(run):
        p2 = LSS_ROOT / task_folder / str(row["subject"]) / f"run-{run}" / raw_file
        if p2.exists():
            return p2

    # 保底路径（供 exists() 判断）
    return p1


def prepare_merged_index() -> pd.DataFrame:
    if not INDEX_L.exists() or not INDEX_R.exists():
        raise FileNotFoundError(f"缺少索引文件: {INDEX_L} 或 {INDEX_R}")

    df_l = pd.read_csv(INDEX_L)
    df_r = pd.read_csv(INDEX_R)

    for df in (df_l, df_r):
        if TASK_FILTER:
            df = df[df["task"].astype(str).str.upper() == TASK_FILTER.upper()]
        # 注意：这里不能直接覆盖局部变量，需要外面分别处理

    if TASK_FILTER:
        df_l = df_l[df_l["task"].astype(str).str.upper() == TASK_FILTER.upper()].copy()
        df_r = df_r[df_r["task"].astype(str).str.upper() == TASK_FILTER.upper()].copy()

    df_l["trial_type"] = df_l.apply(infer_trial_type, axis=1)
    df_r["trial_type"] = df_r.apply(infer_trial_type, axis=1)

    df_l = df_l[df_l["trial_type"].isin(TRIAL_TYPES)].copy()
    df_r = df_r[df_r["trial_type"].isin(TRIAL_TYPES)].copy()

    merge_keys = [k for k in ["subject", "run", "label", "stimulus_content", "trial_type"] if k in df_l.columns and k in df_r.columns]
    if not merge_keys:
        raise ValueError("L/R 索引缺少可用于对齐的共同键")

    merged = pd.merge(
        df_l,
        df_r,
        on=merge_keys,
        suffixes=("_L", "_R"),
        how="inner",
    )

    merged["path_L"] = merged.apply(lambda r: resolve_beta_path(pd.Series({
        "file": r["file_L"], "task": r.get("task_L", "EMO"), "subject": r["subject"], "run": r.get("run", np.nan)
    }), "L"), axis=1)
    merged["path_R"] = merged.apply(lambda r: resolve_beta_path(pd.Series({
        "file": r["file_R"], "task": r.get("task_R", "EMO"), "subject": r["subject"], "run": r.get("run", np.nan)
    }), "R"), axis=1)

    merged = merged[merged["path_L"].apply(lambda p: p.exists()) & merged["path_R"].apply(lambda p: p.exists())].copy()
    return merged


def save_gifti(vec: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = nib.gifti.GiftiDataArray(vec.astype(np.float32), intent="NIFTI_INTENT_ESTIMATE")
    nib.save(nib.gifti.GiftiImage(darrays=[arr]), str(out_path))


def build_dev_model_mats(ages: np.ndarray) -> Dict[str, np.ndarray]:
    ai = ages[:, None]
    aj = ages[None, :]
    amax = float(np.nanmax(ages))

    m_nn = amax - np.abs(ai - aj)
    m_conv = np.minimum(ai, aj)
    m_div = amax - 0.5 * (ai + aj)
    return {"M_nn": m_nn, "M_conv": m_conv, "M_div": m_div}


def process_trial_type(merged: pd.DataFrame, trial_type: str, age_map: Dict[str, float]) -> None:
    df_t = merged[merged["trial_type"] == trial_type].copy()
    if df_t.empty:
        print(f"[跳过] {trial_type}: 无可用记录")
        return

    subject_all = sorted(set(df_t["subject"].astype(str)))
    # 要求 L/R 都可用的被试交集 + 有年龄
    subject_all = [s for s in subject_all if np.isfinite(age_map.get(s, np.nan))]
    subject_all.sort(key=lambda s: (age_map[s], s))

    if len(subject_all) < 3:
        print(f"[跳过] {trial_type}: 交集被试不足")
        return

    out_mean_dir = OUT_ROOT / "mean_betas" / trial_type
    out_sim_dir = OUT_ROOT / "similarity"
    out_model_dir = OUT_ROOT / "dev_models"
    out_meta_dir = OUT_ROOT / "meta"
    for d in (out_mean_dir, out_sim_dir, out_model_dir, out_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    vectors = []
    final_subs = []
    trial_count_rows = []

    for sub in subject_all:
        sub_rows = df_t[df_t["subject"].astype(str) == sub]
        if sub_rows.empty:
            continue

        arr_l = []
        arr_r = []
        for _, row in sub_rows.iterrows():
            try:
                l = surface.load_surf_data(str(row["path_L"])).astype(np.float32).reshape(-1)
                r = surface.load_surf_data(str(row["path_R"])).astype(np.float32).reshape(-1)
                arr_l.append(l)
                arr_r.append(r)
            except Exception:
                continue

        if not arr_l or not arr_r:
            continue

        mean_l = np.mean(np.stack(arr_l, axis=0), axis=0)
        mean_r = np.mean(np.stack(arr_r, axis=0), axis=0)
        whole = np.concatenate([mean_l, mean_r], axis=0).astype(np.float32)

        # 保存被试全脑平均结果（L/R + wholebrain vector）
        save_gifti(mean_l, out_mean_dir / sub / "mean_beta_surface_L.gii")
        save_gifti(mean_r, out_mean_dir / sub / "mean_beta_surface_R.gii")
        np.save(out_mean_dir / sub / "mean_beta_wholebrain_lr_concat.npy", whole)

        vectors.append(whole)
        final_subs.append(sub)
        trial_count_rows.append({"subject": sub, "trial_type": trial_type, "n_trials_used": int(len(arr_l))})

    if len(final_subs) < 3:
        print(f"[跳过] {trial_type}: 计算后有效被试不足")
        return

    # 按年龄递增再排序一次，确保最终顺序严格一致
    order = sorted(range(len(final_subs)), key=lambda i: (age_map[final_subs[i]], final_subs[i]))
    final_subs = [final_subs[i] for i in order]
    vectors = [vectors[i] for i in order]

    X = np.stack(vectors, axis=0)
    sim = np.corrcoef(X)

    ages = np.array([age_map[s] for s in final_subs], dtype=np.float32)
    models = build_dev_model_mats(ages)

    # 输出：相似性矩阵 + 被试顺序
    pd.DataFrame(sim, index=final_subs, columns=final_subs).to_csv(
        out_sim_dir / f"similarity_wholebrain_LR_{trial_type}_AgeSorted.csv"
    )
    pd.DataFrame({"subject": final_subs, "age": ages}).to_csv(
        out_meta_dir / f"subject_order_{trial_type}_AgeSorted.csv", index=False
    )
    pd.DataFrame(trial_count_rows).to_csv(
        out_meta_dir / f"trial_count_{trial_type}.csv", index=False
    )

    # 输出：3个随龄发育矩阵（同一被试顺序）
    for model_name, m in models.items():
        pd.DataFrame(m, index=final_subs, columns=final_subs).to_csv(
            out_model_dir / f"{model_name}_{trial_type}_AgeSorted.csv"
        )

    print(f"[完成] {trial_type}: n_sub={len(final_subs)}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    age_map = load_age_map(SUBJECT_INFO_PATH)
    merged = prepare_merged_index()

    if merged.empty:
        raise RuntimeError("L/R 联合后无可用数据，请检查索引与 beta 文件路径")

    # 全局交集被试（L/R都有）
    subj_intersection = sorted(set(merged["subject"].astype(str)))
    (OUT_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": subj_intersection}).to_csv(OUT_ROOT / "meta" / "subject_intersection_LR.csv", index=False)

    for tt in TRIAL_TYPES:
        process_trial_type(merged, tt, age_map)

    print(f"全部完成，输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
