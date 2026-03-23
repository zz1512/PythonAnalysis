#!/usr/bin/env python3
"""
build_roi_repr_matrix.py

Step 1:
按刺激类型(stimulus_type)分别构建 ROI 表征结构矩阵：
- 输入：LSS index + beta 文件
- 输出：每个 stimulus_type 一个 npz，键为 ROI 名，值为 [n_sub, n_stim, n_stim]

关键约束：
1) 被试必须同时具备 EMO 与 SOC 任务；
2) ROI 可选：
   - 232：surface(L/R) + volume
   - 200：仅 surface(L/R)
3) 每个 stimulus_type 内，所有被试使用完全一致的 stimulus_content 排序。
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
DEFAULT_OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")

ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
ATLAS_VOL = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")

def save_npz_named(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            k = str(name).replace("/", "_").replace("\\", "_")
            bio = io.BytesIO()
            np.save(bio, np.asarray(arr), allow_pickle=False)
            zf.writestr(f"{k}.npy", bio.getvalue())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建每个刺激类型的 ROI 表征结构矩阵（200/232 ROI）")
    p.add_argument("--lss-root", type=Path, default=DEFAULT_LSS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--roi-set", type=int, default=232, choices=(200, 232))
    p.add_argument("--stim-type-col", type=str, default="raw_emotion")
    p.add_argument("--stim-id-col", type=str, default="stimulus_content")
    p.add_argument("--threshold-ratio", type=float, default=0.80)
    p.add_argument("--rsm-method", type=str, default="spearman", choices=("spearman", "pearson"))
    p.add_argument("--no-cocktail-blank", action="store_false", dest="cocktail_blank", default=True)
    p.add_argument("--no-save-pattern", action="store_false", dest="save_pattern", default=True)
    p.add_argument("--pattern-prefix", type=str, default=None)
    return p.parse_args()


def load_lss_index(lss_root: Path, stim_type_col: str, stim_id_col: str) -> pd.DataFrame:
    # Merge all aligned LSS index CSVs (surface/volume scenarios) into a single table.
    files = sorted(lss_root.glob("lss_index_*_aligned.csv"))
    if not files:
        raise FileNotFoundError(f"未找到索引文件: {lss_root}/lss_index_*_aligned.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        required = {"subject", "task", "file", stim_type_col, stim_id_col}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["scenario"] = f.stem.replace("lss_index_", "").replace("_aligned", "").strip().lower()
        dfs.append(df)

    if not dfs:
        raise ValueError("没有可用的 aligned 索引文件（缺必要列）")

    out = pd.concat(dfs, axis=0, ignore_index=True)

    audit_file = lss_root / "lss_audit_surface_L.csv"
    if not audit_file.exists():
        raise FileNotFoundError(f"未找到条件文件: {audit_file}")
    audit_df = pd.read_csv(audit_file)
    condition_col = "trial_type"
    if condition_col not in audit_df.columns:
        raise ValueError(f"条件文件缺少列: {condition_col} ({audit_file})")
    condition_series = audit_df[condition_col].astype("string").str.strip()
    valid_conditions = sorted({c for c in condition_series.dropna().tolist() if c})
    if not valid_conditions:
        raise ValueError(f"条件文件无有效条件: {audit_file}")

    if "label" in out.columns:
        # Prefer parsing stimulus_type from the per-trial label prefix (longest prefix wins).
        label_series = out["label"].astype("string").str.strip()
        matched_condition = pd.Series(pd.NA, index=out.index, dtype="string")
        for cond in sorted(valid_conditions, key=len, reverse=True):
            starts_with_cond = label_series.str.startswith(cond, na=False)
            matched_condition = matched_condition.mask(matched_condition.isna() & starts_with_cond, cond)
        out = out.assign(_matched_condition=matched_condition)
        out = out[out["_matched_condition"].notna()].copy()
        out["stimulus_type"] = out["_matched_condition"].astype("string").str.strip()
    else:
        # Fallback: take stimulus_type from a dedicated column and filter to valid trial types.
        stim_values = out[stim_type_col].astype("string").str.strip()
        out = out[stim_values.isin(valid_conditions)].copy()
        out["stimulus_type"] = out[stim_type_col].astype("string").str.strip()

    out["subject"] = out["subject"].astype("string").str.strip()
    out["task"] = out["task"].astype("string").str.upper().str.strip()
    out["stimulus_id"] = out[stim_id_col].astype("string").str.strip()
    out = out[
        out["subject"].notna()
        & out["stimulus_type"].notna()
        & out["stimulus_id"].notna()
        & (out["subject"].str.len() > 0)
        & (out["stimulus_type"].str.len() > 0)
        & (out["stimulus_id"].str.len() > 0)
    ].copy()
    out = out[
        (out["subject"].str.lower() != "nan")
        & (out["stimulus_type"].str.lower() != "nan")
        & (out["stimulus_id"].str.lower() != "nan")
    ].copy()

    task_sets = out.groupby("subject")["task"].agg(lambda x: set(x.dropna().tolist()))
    eligible = task_sets[task_sets.apply(lambda s: {"EMO", "SOC"}.issubset(s))].index
    # Hard constraint: only keep subjects that have both EMO and SOC tasks.
    out = out[out["subject"].isin(eligible)].copy()
    if out.empty:
        raise ValueError("无同时具备 EMO+SOC 的被试")
    return out


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"):
        # Absolute paths are kept as-is (useful when index was generated elsewhere).
        return Path(f)

    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))

    p1 = lss_root / task / sub / f
    if p1.exists():
        return p1

    run = str(row.get("run", ""))
    # Some LSS layouts nest betas under run-<id>/.
    return lss_root / task / sub / f"run-{run}" / f


def load_surface_atlas(path: Path) -> Tuple[List[int], List[np.ndarray], int]:
    ext = "".join(path.suffixes).lower()
    if ext.endswith(".label.gii") or path.suffix.lower() == ".gii":
        # GIFTI label file: labels are stored in the first darray.
        gii = nib.load(str(path))
        if not hasattr(gii, "darrays") or len(gii.darrays) == 0:
            raise ValueError(f"表面图谱不包含 darrays: {path}")
        labels = np.asarray(gii.darrays[0].data).reshape(-1)
    else:
        # FreeSurfer annotation file: read_annot returns per-vertex labels.
        labels, _, _ = nib.freesurfer.read_annot(str(path))
        labels = np.asarray(labels).reshape(-1)

    roi_ids = sorted(np.unique(labels[labels > 0]).astype(int).tolist())
    roi_indices = [np.where(labels == rid)[0] for rid in roi_ids]
    return roi_ids, roi_indices, int(labels.shape[0])


def load_volume_atlas(path: Path) -> Tuple[object, List[int], List[np.ndarray]]:
    from nilearn import image

    # Tian atlas is treated as integer labels in voxel space.
    img = image.load_img(str(path))
    data = np.rint(img.get_fdata()).astype(np.int32).reshape(-1)
    roi_ids = sorted(np.unique(data[data > 0]).astype(int).tolist())
    roi_indices = [np.where(data == rid)[0] for rid in roi_ids]
    return img, roi_ids, roi_indices


def spearman_rsm(X: np.ndarray, cocktail_blank: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if bool(cocktail_blank):
        X = X - X.mean(axis=0, keepdims=True)
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = ranks.shape[1]
    C = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def pearson_rsm(X: np.ndarray, cocktail_blank: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if bool(cocktail_blank):
        X = X - X.mean(axis=0, keepdims=True)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (X - mean) / std
    p = X.shape[1]
    C = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def compute_rsm(X: np.ndarray, cocktail_blank: bool, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    if m == "spearman":
        return spearman_rsm(X, cocktail_blank=bool(cocktail_blank))
    if m == "pearson":
        return pearson_rsm(X, cocktail_blank=bool(cocktail_blank))
    raise ValueError(f"未知 rsm-method: {method}")


def pick_subjects_stimuli(df: pd.DataFrame, threshold_ratio: float) -> Tuple[List[str], Dict[str, List[str]]]:
    subjects = sorted(df["subject"].unique().tolist())
    n_sub = len(subjects)
    by_type: Dict[str, List[str]] = {}

    for stim_type, g in df.groupby("stimulus_type"):
        # Two-stage coverage filtering:
        # 1) keep stimuli seen in enough subjects;
        # 2) keep only subjects that fully cover the kept stimuli set.
        pairs = g[["subject", "stimulus_id"]].drop_duplicates()
        counts = pairs["stimulus_id"].value_counts()
        valid_ids = sorted(counts[counts >= n_sub * float(threshold_ratio)].index.tolist())
        if not valid_ids:
            continue
        sub_counts = pairs[pairs["stimulus_id"].isin(valid_ids)].groupby("subject")["stimulus_id"].nunique()
        full_sub = sorted(sub_counts[sub_counts == len(valid_ids)].index.tolist())
        if len(full_sub) >= 3:
            by_type[str(stim_type)] = valid_ids

    if not by_type:
        raise ValueError("没有任何 stimulus_type 通过覆盖率筛选")
    return subjects, by_type


def build_lookup(df: pd.DataFrame, lss_root: Path) -> Dict[Tuple[str, str, str, str], Path]:
    out: Dict[Tuple[str, str, str, str], Path] = {}
    for _, row in df.iterrows():
        key = (str(row["scenario"]), str(row["subject"]), str(row["stimulus_type"]), str(row["stimulus_id"]))
        if key not in out:
            out[key] = resolve_beta_path(lss_root, row)
    return out


def _lookup_beta(
    lookup: Dict[Tuple[str, str, str, str], Path],
    subject: str,
    stim_type: str,
    stim_id: str,
    scenario_candidates: List[str],
) -> Optional[Path]:
    for sc in scenario_candidates:
        p = lookup.get((sc, subject, stim_type, stim_id))
        if p is not None:
            return p
    return None


def run(
    lss_root: Path,
    out_dir: Path,
    roi_set: int,
    stim_type_col: str,
    stim_id_col: str,
    threshold_ratio: float,
    rsm_method: str,
    cocktail_blank: bool,
    save_pattern: bool,
    pattern_prefix: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_lss_index(lss_root, stim_type_col=stim_type_col, stim_id_col=stim_id_col)
    all_subjects, stim2ids = pick_subjects_stimuli(df, threshold_ratio=float(threshold_ratio))
    lookup = build_lookup(df, lss_root)

    include_volume = int(roi_set) == 232

    roi_ids_l, roi_idx_l, n_vert_l = load_surface_atlas(ATLAS_SURF_L)
    roi_ids_r, roi_idx_r, n_vert_r = load_surface_atlas(ATLAS_SURF_R)

    vol_img = None
    roi_ids_v: List[int] = []
    roi_idx_v: List[np.ndarray] = []
    if bool(include_volume):
        # Volume ROIs are appended after surface ROIs (V_<id>).
        vol_img, roi_ids_v, roi_idx_v = load_volume_atlas(ATLAS_VOL)

    rois = [f"L_{x}" for x in roi_ids_l] + [f"R_{x}" for x in roi_ids_r]
    if bool(include_volume):
        rois += [f"V_{x}" for x in roi_ids_v]

    out_prefix = f"roi_repr_matrix_{int(roi_set)}"
    pat_prefix = str(pattern_prefix) if pattern_prefix is not None else f"roi_beta_patterns_{int(roi_set)}"
    roi_feat_sizes = [int(idx.size) for idx in roi_idx_l] + [int(idx.size) for idx in roi_idx_r] + ([int(idx.size) for idx in roi_idx_v] if bool(include_volume) else [])
    max_feat = int(max(roi_feat_sizes)) if roi_feat_sizes else 0
    summary_rows = []
    by_stim_dir = out_dir / "by_stimulus"

    for stim_type in sorted(stim2ids.keys()):
        stim_ids = sorted(stim2ids[stim_type])
        pairs = df[df["stimulus_type"] == stim_type][["subject", "stimulus_id"]].drop_duplicates()
        sub_counts = pairs[pairs["stimulus_id"].isin(stim_ids)].groupby("subject")["stimulus_id"].nunique()
        subjects = sorted(sub_counts[sub_counts == len(stim_ids)].index.tolist())
        if len(subjects) < 3:
            continue

        n_sub, n_stim = len(subjects), len(stim_ids)
        out_roi: Dict[str, np.ndarray] = {r: np.zeros((n_sub, n_stim, n_stim), dtype=np.float32) for r in rois}
        out_pat_by_sub: Dict[str, np.ndarray] = {}

        for si, sub in enumerate(subjects):
            feats_l = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_l]
            feats_r = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_r]
            feats_v = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_v] if bool(include_volume) else []

            for ki, stim_id in enumerate(stim_ids):
                p_l = _lookup_beta(lookup, sub, stim_type, stim_id, ["surface_l", "surface_left", "lh", "left", "surface-l"])
                p_r = _lookup_beta(lookup, sub, stim_type, stim_id, ["surface_r", "surface_right", "rh", "right", "surface-r"])
                p_v = _lookup_beta(lookup, sub, stim_type, stim_id, ["volume", "vol"]) if bool(include_volume) else None

                if p_l is None or p_r is None or (bool(include_volume) and p_v is None):
                    raise FileNotFoundError(f"缺少 beta：{sub} {stim_type} {stim_id}")
                if (not p_l.exists()) or (not p_r.exists()) or (bool(include_volume) and (not p_v.exists())):
                    raise FileNotFoundError(f"beta 文件不存在：{sub} {stim_type} {stim_id}")

                vec_l = nib.load(str(p_l)).darrays[0].data.astype(np.float32).reshape(-1)
                vec_r = nib.load(str(p_r)).darrays[0].data.astype(np.float32).reshape(-1)
                if int(vec_l.shape[0]) != int(n_vert_l):
                    raise ValueError(f"surface_L 顶点数不匹配：{sub} {stim_type} {stim_id} beta={int(vec_l.shape[0])} atlas={int(n_vert_l)}")
                if int(vec_r.shape[0]) != int(n_vert_r):
                    raise ValueError(f"surface_R 顶点数不匹配：{sub} {stim_type} {stim_id} beta={int(vec_r.shape[0])} atlas={int(n_vert_r)}")

                for ri, idx in enumerate(roi_idx_l):
                    feats_l[ri][ki, :] = vec_l[idx]
                for ri, idx in enumerate(roi_idx_r):
                    feats_r[ri][ki, :] = vec_r[idx]

                if bool(include_volume):
                    from nilearn import image

                    img_v = image.load_img(str(p_v))
                    if img_v.shape != vol_img.shape or not np.allclose(img_v.affine, vol_img.affine):
                        # Resample beta into atlas space to ensure consistent voxel indexing.
                        img_v = image.resample_to_img(img_v, vol_img, interpolation="continuous")
                    vec_v = img_v.get_fdata().reshape(-1).astype(np.float32)
                    for ri, idx in enumerate(roi_idx_v):
                        feats_v[ri][ki, :] = vec_v[idx]

            for ri, rid in enumerate(roi_ids_l):
                out_roi[f"L_{rid}"][si] = compute_rsm(feats_l[ri], cocktail_blank=bool(cocktail_blank), method=str(rsm_method))
            for ri, rid in enumerate(roi_ids_r):
                out_roi[f"R_{rid}"][si] = compute_rsm(feats_r[ri], cocktail_blank=bool(cocktail_blank), method=str(rsm_method))
            if bool(include_volume):
                for ri, rid in enumerate(roi_ids_v):
                    out_roi[f"V_{rid}"][si] = compute_rsm(feats_v[ri], cocktail_blank=bool(cocktail_blank), method=str(rsm_method))
            if bool(save_pattern):
                # Persist ROI patterns (subject -> [roi, stim, max_feat]) for downstream re-aggregation.
                pat = np.full((len(rois), n_stim, int(max_feat)), np.nan, dtype=np.float32)
                off_r = int(len(roi_ids_l))
                off_v = int(len(roi_ids_l) + len(roi_ids_r))
                for ri in range(len(roi_ids_l)):
                    nf = int(feats_l[ri].shape[1])
                    pat[ri, :, :nf] = feats_l[ri]
                for ri in range(len(roi_ids_r)):
                    nf = int(feats_r[ri].shape[1])
                    pat[off_r + ri, :, :nf] = feats_r[ri]
                if bool(include_volume):
                    for ri in range(len(roi_ids_v)):
                        nf = int(feats_v[ri].shape[1])
                        pat[off_v + ri, :, :nf] = feats_v[ri]
                out_pat_by_sub[str(sub)] = pat

        stim_dir = by_stim_dir / stim_type
        stim_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stim_dir / f"{out_prefix}.npz", **out_roi)
        if bool(save_pattern):
            save_npz_named(stim_dir / f"{pat_prefix}.npz", out_pat_by_sub)
        pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
        pd.DataFrame({"subject": subjects}).to_csv(stim_dir / f"{out_prefix}_subjects.csv", index=False)
        if bool(save_pattern):
            pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{pat_prefix}_rois.csv", index=False)
            pd.DataFrame({"subject": subjects}).to_csv(stim_dir / f"{pat_prefix}_subjects.csv", index=False)
            pd.DataFrame({"roi": rois, "n_feat": roi_feat_sizes}).to_csv(stim_dir / f"{pat_prefix}_roi_feat_sizes.csv", index=False)
        pd.DataFrame(
            {
                "stimulus_type": [stim_type],
                "n_stimuli": [n_stim],
                "roi_set": [int(roi_set)],
                "rsm_method": [str(rsm_method)],
                "cocktail_blank": [bool(cocktail_blank)],
            }
        ).to_csv(stim_dir / f"{out_prefix}_meta.csv", index=False)
        if bool(save_pattern):
            pd.DataFrame(
                {
                    "stimulus_type": [stim_type],
                    "n_stimuli": [n_stim],
                    "roi_set": [int(roi_set)],
                    "pattern_prefix": [str(pat_prefix)],
                    "pattern_key": ["subject"],
                    "pattern_max_feat": [int(max_feat)],
                }
            ).to_csv(stim_dir / f"{pat_prefix}_meta.csv", index=False)
        pd.DataFrame({"stimulus_order": stim_ids}).to_csv(stim_dir / "stimulus_order.csv", index=False)
        pd.DataFrame([{"subject": s, "order_id": "|".join(stim_ids)} for s in subjects]).to_csv(
            stim_dir / "stimulus_order_by_subject.csv", index=False
        )

        summary_rows.append({"stimulus_type": stim_type, "n_subjects": n_sub, "n_stimuli": n_stim, "n_rois": len(rois)})

    if not summary_rows:
        raise ValueError("没有写出任何刺激类型结果")

    pd.DataFrame(summary_rows).sort_values("stimulus_type").to_csv(out_dir / f"{out_prefix}_summary.csv", index=False)
    pd.DataFrame({"subject": all_subjects}).to_csv(out_dir / "eligible_subjects_emo_soc.csv", index=False)


def main() -> None:
    args = parse_args()
    run(
        args.lss_root,
        args.out_dir,
        int(args.roi_set),
        args.stim_type_col,
        args.stim_id_col,
        float(args.threshold_ratio),
        str(args.rsm_method),
        bool(args.cocktail_blank),
        bool(args.save_pattern),
        args.pattern_prefix,
    )


if __name__ == "__main__":
    main()
