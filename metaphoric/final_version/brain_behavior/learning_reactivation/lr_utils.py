from __future__ import annotations

import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
MISSING_ORIGINAL_IDS = {
    "yy": {20, 25, 30, 31, 34},
    "kj": {4, 18, 32, 35, 40},
}
VALID_ORIGINAL_IDS = {
    condition: [idx for idx in range(1, 41) if idx not in missing]
    for condition, missing in MISSING_ORIGINAL_IDS.items()
}

SEMANTIC_ROIS = {
    "meta_L_temporal_pole",
    "meta_R_temporal_pole",
    "meta_L_IFG",
    "meta_R_IFG",
    "meta_L_AG",
    "meta_R_AG",
    "meta_L_pMTG_pSTS",
    "meta_R_pMTG_pSTS",
}

HPC_SPATIAL_ROIS = {
    "meta_L_hippocampus",
    "meta_R_hippocampus",
    "meta_L_PPA_PHG",
    "meta_R_PPA_PHG",
    "meta_L_RSC_PCC",
    "meta_R_RSC_PCC",
    "meta_L_PPC_SPL",
    "meta_R_PPC_SPL",
    "meta_L_precuneus",
    "meta_R_precuneus",
}


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"metaphor", "yy", "yyw", "yyew"}:
        return "yy"
    if text in {"spatial", "kj", "kjw", "kjew"}:
        return "kj"
    return text


def extract_original_id(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    stem = Path(text).stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return float("nan")
    return float(int(match.group(1)))


def condition_item_id(condition: str, original_pair_id: object) -> str:
    try:
        return f"{condition}_{int(float(original_pair_id))}"
    except Exception:
        return f"{condition}_nan"


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def load_template_map(template_path: Path) -> pd.DataFrame:
    template = read_any(template_path).copy()
    template["condition"] = template["condition"].map(normalize_condition)
    template = template[template["condition"].isin(CONDITIONS)].copy()
    template["template_pair_id"] = pd.to_numeric(template["pair_id"], errors="coerce").astype("Int64")
    template["word_label"] = template["word_label"].astype(str).str.strip()
    template["original_pair_id"] = template["word_label"].map(extract_original_id).astype("Int64")

    role_a = {"yy": "yyw", "kj": "kjw"}
    role_b = {"yy": "yyew", "kj": "kjew"}
    rows: list[dict[str, object]] = []
    for (condition, template_pair_id), group in template.groupby(["condition", "template_pair_id"], dropna=True):
        originals = sorted(set(int(v) for v in group["original_pair_id"].dropna().tolist()))
        if len(originals) != 1:
            raise ValueError(f"{condition}/{template_pair_id} maps to original IDs {originals}")
        a = group[group["type"].astype(str).str.lower().eq(role_a[condition])]
        b = group[group["type"].astype(str).str.lower().eq(role_b[condition])]
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f"{condition}/{template_pair_id} does not have one role A and one role B")
        original = originals[0]
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "template_pair_id": int(template_pair_id),
                "original_pair_id": int(original),
                "condition_item_id": condition_item_id(condition, original),
                "role_a_label": str(a.iloc[0]["word_label"]).strip(),
                "role_b_label": str(b.iloc[0]["word_label"]).strip(),
            }
        )
    return pd.DataFrame(rows)


def load_4d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def masked_samples(image_path: Path, mask: np.ndarray, mask_path: Path) -> np.ndarray:
    data = load_4d(image_path)
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def fisher_corr_one_to_many(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    a = np.asarray(one, dtype=float).reshape(1, -1)
    b = np.asarray(many, dtype=float)
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (a @ b.T / denom).reshape(-1)
    corr = np.clip(corr, -0.999999, 0.999999)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(corr)
    z[~np.isfinite(z)] = np.nan
    return z


def fisher_corr(a: np.ndarray, b: np.ndarray) -> float:
    vals = fisher_corr_one_to_many(a, np.asarray(b).reshape(1, -1))
    return float(vals[0]) if vals.size else float("nan")


def zscore_grouped(frame: pd.DataFrame, value_col: str, group_cols: list[str], out_col: str) -> pd.DataFrame:
    out = frame.copy()
    out[out_col] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        vals = pd.to_numeric(out.loc[idx, value_col], errors="coerce")
        sd = vals.std(ddof=1)
        if not np.isfinite(sd) or math.isclose(float(sd), 0.0):
            z = vals * np.nan
        else:
            z = (vals - vals.mean()) / sd
        out.loc[idx, out_col] = z
    return out


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def available_covariates(frame: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    candidates = [
        "sentence_char_len_z",
        "word_frequency_mean_z",
        "stroke_count_mean_z",
        "valence_mean_z",
        "arousal_mean_z",
    ]
    covs: list[str] = []
    for col in candidates:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().mean() >= min_coverage and vals.nunique(dropna=True) > 1:
            covs.append(col)
    return covs


def fit_gee_by_roi(
    frame: pd.DataFrame,
    *,
    formula: str,
    outcome: str,
    model_name: str,
    family: str = "gaussian",
    min_rows: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    for (roi_set, roi), sub in frame.groupby(["roi_set", "roi"], sort=False):
        data = sub.copy()
        if outcome not in data.columns:
            continue
        data = data.dropna(subset=[outcome, "subject"])
        if "condition" in data.columns:
            data["condition"] = pd.Categorical(
                data["condition"].astype(str).str.lower(),
                categories=["kj", "yy"],
            )
        if len(data) < min_rows or data["subject"].nunique() < 5:
            rows.append(
                {
                    "model": model_name,
                    "roi_set": roi_set,
                    "roi": roi,
                    "outcome": outcome,
                    "term": "__model__",
                    "estimate": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "n_rows": len(data),
                    "n_subjects": int(data["subject"].nunique()),
                    "status": "skipped_insufficient_data",
                    "formula": formula,
                }
            )
            continue
        try:
            result = smf.gee(
                formula=formula,
                groups="subject",
                data=data,
                family=fam,
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
            for term in result.params.index:
                rows.append(
                    {
                        "model": model_name,
                        "roi_set": roi_set,
                        "roi": roi,
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(result.params[term]),
                        "se": float(result.bse[term]),
                        "z": float(result.tvalues[term]),
                        "p": float(result.pvalues[term]),
                        "n_rows": len(data),
                        "n_subjects": int(data["subject"].nunique()),
                        "status": "ok",
                        "formula": formula,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "roi_set": roi_set,
                    "roi": roi,
                    "outcome": outcome,
                    "term": "__model__",
                    "estimate": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "n_rows": len(data),
                    "n_subjects": int(data["subject"].nunique()),
                    "status": f"failed: {exc}",
                    "formula": formula,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        ok = out["status"].eq("ok") & out["p"].notna()
        for (_, _, _), idx in out[ok].groupby(["model", "roi_set", "term"]).groups.items():
            out.loc[idx, "q"] = bh_fdr(out.loc[idx, "p"])
    return out
