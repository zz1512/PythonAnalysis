"""
model_rdm_comparison.py

用途
- 模型 RDM 比较（Model RSA）：把神经 RDM 与若干"理论/行为模型 RDM"做相关，
  检验哪些模型最能解释表征结构。
- 升级后支持五大核心模型：
  - M1: 条件类别（多水平，默认 yy/kj/baseline）
  - M2: 配对身份（同 pair_id=0，其余=1）
  - M8: 反向配对身份（同 pair_id=1，其余=0；分化探针）
  - M3: 预训练语义（来自 BERT/word2vec 的 cosine distance；通过外部 embedding 表接入）
  - M7: 被试回忆强度（per-subject |recall_i - recall_j|）
- 同时保留 numeric 模型（novelty/familiarity 等连续变量的差的绝对值）。

核心统计
- 单模型 Spearman ρ
- `--partial-correlation`：多模型 Spearman 偏相关（残差法），区分"预存语义"和"学习配对"
- `--partial-correlation` 开启后自动生成模型共线性报告（|ρ|>0.7 标注 high）

输入
- pattern_root: `${PATTERN_ROOT}`（4D patterns）
- roi_dir: ROI masks
- metadata：来自 `stack_patterns.py` 的 `*_metadata.tsv`
- `--embedding-file`：`build_stimulus_embeddings.py` 产出的 TSV（word_label + dim_*）
- `--memory-strength-dir`：`build_memory_strength_table.py` 产出的 per-subject TSV 目录
- `--conditions`：默认 yy/kj/baseline，可显式覆盖
- `--pair-id-col`：默认 `pair_id`，回退 `pic_num`

输出（自动加 ROI_SET 后缀，可用 METAPHOR_MODEL_RDM_OUT_DIR 覆盖）
- `model_rdm_subject_metrics.tsv`：每个 subject/roi/time/model 的 Spearman ρ
- `model_rdm_partial_metrics.tsv`（可选）：多模型偏相关
- `model_collinearity.tsv`（可选）：模型 RDM 两两 Spearman ρ，flag=high 当 |ρ|>0.7
- `model_rdm_group_summary.tsv`：ROI×模型水平的 pre vs post 配对 t 检验
- `<roi>_<model>_rdm_summary.json`：单个 ROI×模型的配对 t 摘要
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


M7_MODEL_FILE_VARIANTS: dict[str, str] = {
    "M7_memory": "binary",
    "M7_binary": "binary",
    "M7_continuous_confidence": "continuous_confidence",
    "M7_learning_weighted": "learning_weighted",
    "M7_irt": "irt",
}


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, paired_t_summary, read_table, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_masked_samples, neural_rdm_vector  # noqa: E402
from common.roi_library import default_roi_tagged_out_dir, select_roi_masks  # noqa: E402
from common.stimulus_text_mapping import attach_real_word_columns  # noqa: E402


# ---------- 模型 RDM 构造 ----------


def model_from_condition_multi(metadata: pd.DataFrame) -> np.ndarray:
    """M1: 多水平条件 RDM。同条件=0，跨条件=1。"""
    labels = metadata["condition"].astype(str).to_numpy()
    matrix = (labels[:, None] != labels[None, :]).astype(float)
    return matrix[np.triu_indices_from(matrix, k=1)]


def model_from_pair_identity(metadata: pd.DataFrame, pair_col: str = "pair_id") -> np.ndarray:
    """M2: 配对身份 RDM。同 pair_id=0，其余=1。"""
    col = pair_col if pair_col in metadata.columns else None
    if col is None and "pic_num" in metadata.columns:
        col = "pic_num"
    if col is None:
        raise ValueError("M2 requires pair_id or pic_num column in metadata.")
    values = metadata[col].astype(str).to_numpy()
    matrix = (values[:, None] != values[None, :]).astype(float)
    return matrix[np.triu_indices_from(matrix, k=1)]


def model_from_reverse_pair_identity(metadata: pd.DataFrame, pair_col: str = "pair_id") -> np.ndarray:
    """
    M8: 反向配对身份（分化探针）。

    - 同 pair_id=1，非同 pair_id=0
    - 直觉：若学习后同 pair 的神经距离更大（更分化），则该模型与神经 RDM 的相关可能上升。
    """
    col = pair_col if pair_col in metadata.columns else None
    if col is None and "pic_num" in metadata.columns:
        col = "pic_num"
    if col is None:
        raise ValueError("M8 requires pair_id or pic_num column in metadata.")
    values = metadata[col].astype(str).to_numpy()
    matrix = (values[:, None] == values[None, :]).astype(float)
    np.fill_diagonal(matrix, 0.0)
    return matrix[np.triu_indices_from(matrix, k=1)]


def model_from_embedding(metadata: pd.DataFrame, embeddings: pd.DataFrame,
                         word_col: str = "word_label") -> np.ndarray:
    """M3: 预训练语义 cosine distance RDM。缺词对应行列置 NaN。"""
    if word_col not in metadata.columns:
        raise ValueError(f"Metadata missing '{word_col}' column required for M3 embedding lookup.")
    if word_col not in embeddings.columns:
        raise ValueError(f"Embedding file missing '{word_col}' column required for M3 lookup.")
    if embeddings[word_col].duplicated().any():
        embeddings = embeddings.drop_duplicates(subset=word_col, keep="first")
    emb_map = embeddings.set_index(word_col)
    labels = metadata[word_col].astype(str).to_list()
    dim_cols = [c for c in emb_map.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("Embedding file must contain columns prefixed with 'dim_'.")
    n_trials = len(labels)
    vectors = np.full((n_trials, len(dim_cols)), np.nan, dtype=float)
    for idx, lab in enumerate(labels):
        if lab in emb_map.index:
            row = emb_map.loc[lab, dim_cols]
            vectors[idx] = np.asarray(row, dtype=float).ravel()[: len(dim_cols)]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = vectors / norms
    cosine_sim = normalized @ normalized.T
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    rdm = 1.0 - cosine_sim
    missing = np.isnan(vectors).any(axis=1) | (norms.squeeze(axis=1) == 0)
    if missing.any():
        rdm[missing, :] = np.nan
        rdm[:, missing] = np.nan
    return rdm[np.triu_indices_from(rdm, k=1)]


def model_from_subject_numeric(metadata: pd.DataFrame, table: pd.DataFrame,
                               value_col: str = "recall_score",
                               word_col: str = "word_label") -> np.ndarray:
    """M7: 被试维度 numeric 差 RDM（例如回忆强度）。"""
    if word_col not in metadata.columns:
        raise ValueError(f"Metadata missing '{word_col}' column required for subject numeric RDM.")
    if word_col not in table.columns:
        raise ValueError(f"Subject numeric table missing '{word_col}' column required for M7 lookup.")
    lookup = table.set_index(word_col)[value_col]
    values = np.asarray([lookup.get(lab, np.nan) for lab in metadata[word_col].astype(str)], dtype=float)
    diff = np.abs(values[:, None] - values[None, :])
    return diff[np.triu_indices_from(diff, k=1)]


def model_from_numeric(metadata: pd.DataFrame, column: str) -> np.ndarray:
    """保留的通用 numeric 模型（novelty/familiarity 等）。"""
    values = pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=float)
    matrix = np.abs(values[:, None] - values[None, :])
    return matrix[np.triu_indices_from(matrix, k=1)]


# ---------- 统计工具 ----------


def _safe_spearman(neural: np.ndarray, model_vec: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(neural) & np.isfinite(model_vec)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rho, pvalue = stats.spearmanr(neural[mask], model_vec[mask])
    return float(rho), float(pvalue), int(mask.sum())


def _rank_ignore_nan(vec: np.ndarray) -> np.ndarray:
    """Rank with NaN preserved (utility helper, reserved for future NaN-aware paths)."""
    ranks = np.full_like(vec, np.nan, dtype=float)
    mask = np.isfinite(vec)
    if mask.any():
        ranks[mask] = stats.rankdata(vec[mask])
    return ranks


def _partial_spearman(neural: np.ndarray, target: np.ndarray,
                      controls: list[np.ndarray]) -> tuple[float, float, int]:
    stacked = np.column_stack([target] + controls + [neural])
    valid = np.all(np.isfinite(stacked), axis=1)
    if valid.sum() < 5:
        return float("nan"), float("nan"), int(valid.sum())
    subset = stacked[valid]
    ranked = np.apply_along_axis(stats.rankdata, 0, subset)
    target_r = ranked[:, 0]
    controls_r = ranked[:, 1:1 + len(controls)]
    neural_r = ranked[:, -1]

    def _residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 0:
            return y - y.mean()
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        return y - X_design @ beta

    target_res = _residual(target_r, controls_r)
    neural_res = _residual(neural_r, controls_r)
    if np.allclose(target_res.std(), 0) or np.allclose(neural_res.std(), 0):
        return float("nan"), float("nan"), int(valid.sum())
    rho, pvalue = stats.pearsonr(target_res, neural_res)
    return float(rho), float(pvalue), int(valid.sum())


def _build_collinearity_report(model_vectors: dict[str, np.ndarray],
                               threshold: float = 0.7) -> pd.DataFrame:
    rows = []
    names = list(model_vectors.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            vec_a = model_vectors[name_a]
            vec_b = model_vectors[name_b]
            mask = np.isfinite(vec_a) & np.isfinite(vec_b)
            if mask.sum() < 3:
                rho = float("nan")
            else:
                rho, _ = stats.spearmanr(vec_a[mask], vec_b[mask])
            flag = "high" if np.isfinite(rho) and abs(rho) > threshold else "ok"
            rows.append({"model_a": name_a, "model_b": name_b,
                         "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
                         "flag": flag})
    return pd.DataFrame(rows)


def _partial_control_names(names: list[str], target_name: str) -> tuple[list[str], list[str]]:
    """
    Exclude perfectly complementary model pairs from the same partial-correlation control set.

    M2_pair and M8_reverse_pair are exact complements after vectorization:
    including one as the other's control makes the residualization step rank-deficient
    and the resulting partial rho non-interpretable.
    """
    controls = [name for name in names if name != target_name]
    excluded = []
    complementary = {
        "M2_pair": "M8_reverse_pair",
        "M8_reverse_pair": "M2_pair",
    }
    counterpart = complementary.get(target_name)
    if counterpart in controls:
        controls.remove(counterpart)
        excluded.append(counterpart)
    return controls, excluded


# ---------- 主流程 ----------


def _normalize_enabled_models(model_names: list[str] | None) -> list[str]:
    names = list(model_names or [])
    normalized: list[str] = []
    for name in names:
        canonical = "M7_binary" if name == "M7_memory" else name
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _load_memory_table(memory_dir: Path | None, subject: str, variant: str = "binary") -> pd.DataFrame | None:
    if memory_dir is None:
        return None
    candidates: list[Path] = []
    if variant == "binary":
        candidates.extend([
            memory_dir / f"memory_strength_binary_{subject}.tsv",
            memory_dir / f"memory_strength_{subject}.tsv",
        ])
    else:
        candidates.append(memory_dir / f"memory_strength_{variant}_{subject}.tsv")

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate, sep="\t")
    return None


def _default_output_dir() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR, ROI_SET  # type: ignore
    return default_roi_tagged_out_dir(
        BASE_DIR,
        "model_rdm_results",
        override_env="METAPHOR_MODEL_RDM_OUT_DIR",
        roi_set=ROI_SET,
    )


def _default_stimuli_template() -> Path | None:
    try:
        from rsa_analysis.rsa_config import STIMULI_TEMPLATE  # type: ignore
    except Exception:
        return None
    return Path(STIMULI_TEMPLATE)


def _default_pattern_root() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # type: ignore
    return Path(BASE_DIR) / "pattern_root"


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


def _save_rdm_audit(
    audit_root: Path,
    *,
    subject: str,
    roi_name: str,
    time: str,
    condition_group: str,
    metadata: pd.DataFrame,
    neural: np.ndarray,
    model_vectors: dict[str, np.ndarray],
    model_skip: dict[str, str],
) -> None:
    cell_dir = ensure_dir(audit_root / subject / _slug(roi_name))
    cell_stem = f"{_slug(time)}_{_slug(condition_group)}"

    metadata_out = metadata.reset_index(drop=True).copy()
    if "trial_index" in metadata_out.columns:
        metadata_out = metadata_out.drop(columns=["trial_index"])
    metadata_out.insert(0, "trial_index", np.arange(len(metadata_out), dtype=int))
    metadata_path = cell_dir / f"{cell_stem}_metadata.tsv"
    write_table(metadata_out, metadata_path)

    pair_i, pair_j = np.triu_indices(len(metadata_out), k=1)
    npz_payload: dict[str, np.ndarray] = {
        "pair_i": pair_i.astype(np.int32),
        "pair_j": pair_j.astype(np.int32),
        "neural_rdm": np.asarray(neural, dtype=float),
    }
    for model_name, model_vec in model_vectors.items():
        npz_payload[model_name] = np.asarray(model_vec, dtype=float)

    rdm_path = cell_dir / f"{cell_stem}_rdms.npz"
    np.savez_compressed(rdm_path, **npz_payload)

    save_json(
        {
            "subject": subject,
            "roi": roi_name,
            "time": time,
            "condition_group": condition_group,
            "n_trials": int(len(metadata_out)),
            "n_pairs": int(len(pair_i)),
            "metadata_file": str(metadata_path),
            "rdm_file": str(rdm_path),
            "model_names": sorted(model_vectors.keys()),
            "skipped_models": model_skip,
        },
        cell_dir / f"{cell_stem}_audit_manifest.json",
    )


def _resolve_roi_masks(roi_dir: Path | None) -> dict[str, Path]:
    if roi_dir is not None:
        if roi_dir.is_file():
            return {roi_dir.stem.replace(".nii", ""): roi_dir}
        roi_paths = sorted(roi_dir.glob("*.nii*"))
        return {path.stem.replace(".nii", ""): path for path in roi_paths}

    from rsa_analysis.rsa_config import ROI_MANIFEST, ROI_MASKS, ROI_SET  # type: ignore

    manifest_path = Path(ROI_MANIFEST) if ROI_MANIFEST is not None else None
    if manifest_path is not None and manifest_path.exists():
        return select_roi_masks(manifest_path, roi_set=ROI_SET, include_flag="include_in_rsa")
    return {str(name): Path(path) for name, path in dict(ROI_MASKS).items()}


def augment_metadata_for_models(
    metadata: pd.DataFrame,
    template_df: pd.DataFrame | None,
    *,
    word_col: str,
    pair_col: str,
) -> pd.DataFrame:
    out = metadata.copy()
    out = attach_real_word_columns(out, column_map={"unique_label": "real_word", "word_label": "real_word"})

    if word_col not in out.columns:
        if "unique_label" in out.columns:
            out[word_col] = out["unique_label"].astype(str).str.strip()
        elif "word_label" in out.columns:
            out[word_col] = out["word_label"].astype(str).str.strip()

    if template_df is not None and not template_df.empty:
        tpl = attach_real_word_columns(template_df, column_map={"word_label": "real_word"})
        if "word_label" in tpl.columns:
            tpl["word_label"] = tpl["word_label"].astype(str).str.strip()
        if "real_word" in tpl.columns:
            tpl["real_word"] = tpl["real_word"].astype(str).str.strip()
        if "word_label" in out.columns and "word_label" in tpl.columns:
            wanted = ["word_label"]
            if "real_word" in tpl.columns:
                wanted.append("real_word")
            if "pair_id" in tpl.columns:
                wanted.append("pair_id")
            lookup = tpl[wanted].drop_duplicates(subset=["word_label"], keep="first")
            out = out.merge(lookup, how="left", on="word_label", suffixes=("", "_tpl"))
            if "real_word_tpl" in out.columns:
                if "real_word" not in out.columns:
                    out["real_word"] = out["real_word_tpl"]
                else:
                    current = out["real_word"].astype(str).str.strip()
                    mask = out["real_word"].isna() | current.eq("") | current.eq("nan") | current.eq("<NA>")
                    out.loc[mask, "real_word"] = out.loc[mask, "real_word_tpl"]
                out = out.drop(columns=["real_word_tpl"])

    if pair_col not in out.columns:
        if "pair_id" in out.columns:
            out[pair_col] = out["pair_id"]
        elif "pic_num" in out.columns:
            out[pair_col] = out["pic_num"]

    if word_col not in out.columns and word_col == "real_word":
        out = attach_real_word_columns(out, column_map={"word_label": "real_word", "unique_label": "real_word"})

    return out


def _condition_group_label(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "yy": "yy",
        "metaphor": "yy",
        "kj": "kj",
        "spatial": "kj",
        "baseline": "baseline",
        "base": "baseline",
        "jx": "baseline",
    }
    return mapping.get(text, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare neural RDMs with candidate model RDMs (M1/M2/M3/M7/M8)."
    )
    parser.add_argument(
        "pattern_root",
        type=Path,
        nargs="?",
        default=None,
        help="Defaults to {BASE_DIR}/pattern_root.",
    )
    parser.add_argument(
        "roi_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Optional ROI directory or single ROI mask. Defaults to current ROI_SET masks from rsa_config.",
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Defaults to {BASE_DIR}/model_rdm_results_{ROI_SET}; "
                             "override via METAPHOR_MODEL_RDM_OUT_DIR env var.")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--metadata-template", default="{time}_{condition}_metadata.tsv")
    parser.add_argument("--conditions", nargs="*", default=["yy", "kj", "baseline"])
    parser.add_argument(
        "--analysis-mode",
        choices=["by_condition", "pooled", "both"],
        default="pooled",
        help="Whether to build neural RDMs within each condition, across pooled conditions, or both.",
    )
    parser.add_argument("--pair-id-col", default="pair_id")
    parser.add_argument("--word-col", default="real_word")
    parser.add_argument("--stimuli-template", type=Path, default=None,
                        help="Optional stimuli template used to backfill word_label/pair_id into pattern metadata.")
    parser.add_argument("--embedding-file", type=Path, default=None,
                        help="stimulus_embeddings TSV for M3 (cosine distance).")
    parser.add_argument("--memory-strength-dir", type=Path, default=None,
                        help="Directory with per-subject M7 tables, e.g. "
                             "memory_strength_binary_{subject}.tsv and "
                             "memory_strength_continuous_confidence_{subject}.tsv.")
    parser.add_argument("--memory-score-col", default="recall_score")
    parser.add_argument("--numeric-model-cols", nargs="*", default=[])
    parser.add_argument("--enabled-models", nargs="*",
                        default=[
                            "M1_condition",
                            "M2_pair",
                            "M3_embedding",
                            "M7_binary",
                            "M7_continuous_confidence",
                            "M8_reverse_pair",
                        ],
                        help="Subset of models to include: "
                             "M1_condition M2_pair M3_embedding "
                             "M7_binary M7_continuous_confidence "
                             "M7_learning_weighted M7_irt M8_reverse_pair "
                             "(legacy alias: M7_memory -> M7_binary)")
    # Alias: some docs may refer to `--models`
    parser.add_argument("--models", dest="enabled_models", nargs="*",
                        help="Alias of --enabled-models.")
    parser.add_argument("--partial-correlation", action="store_true",
                        help="Compute partial Spearman correlations between each model and neural RDM.")
    parser.add_argument("--collinearity-threshold", type=float, default=0.7)
    parser.add_argument("--allow-missing-models", action="store_true",
                        help="Allow missing prerequisites to be recorded as skip_reason instead of failing fast.")
    parser.add_argument("--save-rdm-audit", action="store_true", default=True,
                        help="Save per-cell metadata and compressed neural/model RDM vectors for audit (default: on).")
    parser.add_argument("--no-save-rdm-audit", dest="save_rdm_audit", action="store_false",
                        help="Disable per-cell model-RDM audit export.")
    parser.add_argument("--rdm-audit-dir", type=Path, default=None,
                        help="Optional directory for model-RDM audit exports. Default: <output_dir>/model_rdm_audit")
    args = parser.parse_args()
    args.enabled_models = _normalize_enabled_models(args.enabled_models)

    args.pattern_root = args.pattern_root or _default_pattern_root()
    output_dir = ensure_dir(args.output_dir if args.output_dir is not None else _default_output_dir())
    audit_root = None
    if args.save_rdm_audit:
        audit_root = ensure_dir(
            args.rdm_audit_dir if args.rdm_audit_dir is not None else (output_dir / "model_rdm_audit")
        )

    embedding_table: pd.DataFrame | None = None
    if args.embedding_file is not None:
        if not args.embedding_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {args.embedding_file}")
        embedding_table = pd.read_csv(args.embedding_file, sep="\t")

    stimuli_template = args.stimuli_template or _default_stimuli_template()
    template_df: pd.DataFrame | None = None
    if stimuli_template is not None and Path(stimuli_template).exists():
        template_df = read_table(Path(stimuli_template))

    rows = []
    partial_rows = []
    collinearity_snapshots: list[pd.DataFrame] = []

    roi_masks = _resolve_roi_masks(args.roi_dir)
    if not roi_masks:
        raise ValueError("No ROI masks found. Pass roi_dir explicitly or check rsa_config ROI settings.")

    subject_dirs = sorted([p for p in args.pattern_root.iterdir()
                           if p.is_dir() and p.name.startswith("sub-")])
    memory_table_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    for roi_name, roi_path in roi_masks.items():
        for subject_dir in subject_dirs:
            subject = subject_dir.name
            for time in ["pre", "post"]:
                image_paths: list[tuple[str, Path]] = []
                meta_frames: list[pd.DataFrame] = []
                for condition in args.conditions:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    meta_path = subject_dir / args.metadata_template.format(time=time, condition=condition)
                    if image_path.exists() and meta_path.exists():
                        image_paths.append((condition, image_path))
                        meta = pd.read_csv(meta_path, sep="\t")
                        meta["condition"] = condition
                        meta_frames.append(meta)
                if not image_paths:
                    rows.append({
                        "subject": subject, "roi": roi_name, "time": time,
                        "condition_group": None,
                        "model": None, "rho": float("nan"), "p_value": float("nan"),
                        "n_pairs": 0, "n_conditions_available": len(image_paths),
                        "skip_reason": "fewer_than_two_conditions_available",
                    })
                    continue
                sample_blocks = []
                checked_meta_frames = []
                per_condition_cells: list[tuple[str, np.ndarray, pd.DataFrame]] = []
                for (cond, img_path), meta in zip(image_paths, meta_frames):
                    block = load_masked_samples(img_path, roi_path)
                    if block.shape[0] != len(meta):
                        raise AssertionError(
                            f"samples/metadata row mismatch for {subject} {roi_name} {time} {cond}: "
                            f"samples={block.shape[0]} vs meta_rows={len(meta)} "
                            f"(check stack_patterns.py ordering and *_metadata.tsv)."
                        )
                    checked_meta = meta.reset_index(drop=True)
                    sample_blocks.append(block)
                    checked_meta_frames.append(checked_meta)
                    per_condition_cells.append((_condition_group_label(cond), block, checked_meta))

                analysis_cells: list[tuple[str, np.ndarray, pd.DataFrame]] = []
                if args.analysis_mode in {"pooled", "both"} and len(sample_blocks) >= 2:
                    pooled_samples = np.vstack(sample_blocks)
                    pooled_meta = pd.concat(checked_meta_frames, ignore_index=True)
                    if pooled_samples.shape[0] != len(pooled_meta):
                        raise AssertionError(
                            f"stacked samples/metadata mismatch for {subject} {roi_name} {time}: "
                            f"samples={pooled_samples.shape[0]} vs meta_rows={len(pooled_meta)}."
                        )
                    analysis_cells.append(("all", pooled_samples, pooled_meta))
                if args.analysis_mode in {"by_condition", "both"}:
                    analysis_cells.extend(per_condition_cells)

                for condition_group, samples, metadata in analysis_cells:
                    if samples.shape[0] < 2:
                        rows.append({
                            "subject": subject, "roi": roi_name, "time": time,
                            "condition_group": condition_group,
                            "model": None, "rho": float("nan"), "p_value": float("nan"),
                            "n_pairs": 0, "n_conditions_available": len(image_paths),
                            "skip_reason": "fewer_than_two_trials",
                        })
                        continue

                    metadata = augment_metadata_for_models(
                        metadata,
                        template_df,
                        word_col=args.word_col,
                        pair_col=args.pair_id_col,
                    )
                    neural = neural_rdm_vector(samples, metric="correlation")

                    model_vectors: dict[str, np.ndarray] = {}
                    model_skip: dict[str, str] = {}

                    def _handle_model_failure(model_name: str, reason: str, *, allow_skip: bool = False) -> None:
                        if allow_skip or args.allow_missing_models:
                            model_skip[model_name] = reason
                            return
                        raise RuntimeError(
                            f"{model_name} failed for {subject} {roi_name} {time} {condition_group}: {reason}"
                        )

                    single_condition_group = metadata["condition"].astype(str).nunique() <= 1
                    baseline_group = condition_group == "baseline"

                    if "M1_condition" in args.enabled_models:
                        if single_condition_group:
                            model_skip["M1_condition"] = "not_applicable_single_condition_group"
                        else:
                            try:
                                model_vectors["M1_condition"] = model_from_condition_multi(metadata)
                            except Exception as exc:
                                _handle_model_failure("M1_condition", f"error: {exc}")

                    if "M2_pair" in args.enabled_models:
                        if baseline_group:
                            model_skip["M2_pair"] = "not_applicable_nonpaired_condition"
                        elif args.pair_id_col not in metadata.columns and "pic_num" not in metadata.columns:
                            _handle_model_failure("M2_pair", "missing_pair_id")
                        else:
                            try:
                                model_vectors["M2_pair"] = model_from_pair_identity(metadata, args.pair_id_col)
                            except Exception as exc:
                                _handle_model_failure("M2_pair", f"error: {exc}")

                    if "M3_embedding" in args.enabled_models:
                        if embedding_table is None:
                            _handle_model_failure("M3_embedding", "missing_embedding_file")
                        elif args.word_col not in metadata.columns:
                            _handle_model_failure("M3_embedding", "missing_word_label")
                        else:
                            try:
                                model_vectors["M3_embedding"] = model_from_embedding(
                                    metadata, embedding_table, args.word_col
                                )
                            except Exception as exc:
                                _handle_model_failure("M3_embedding", f"error: {exc}")

                    requested_m7_models = [
                        model_name for model_name in args.enabled_models
                        if model_name in M7_MODEL_FILE_VARIANTS
                    ]
                    if requested_m7_models:
                        for model_name in requested_m7_models:
                            variant = M7_MODEL_FILE_VARIANTS[model_name]
                            cache_key = (subject, variant)
                            if cache_key not in memory_table_cache:
                                memory_table_cache[cache_key] = _load_memory_table(
                                    args.memory_strength_dir, subject, variant
                                )
                            mem_table = memory_table_cache[cache_key]
                            if mem_table is None:
                                _handle_model_failure(model_name, f"missing_memory_table:{variant}", allow_skip=True)
                            elif args.word_col not in metadata.columns:
                                _handle_model_failure(model_name, "missing_word_label")
                            else:
                                try:
                                    model_vectors[model_name] = model_from_subject_numeric(
                                        metadata, mem_table, args.memory_score_col, args.word_col
                                    )
                                except Exception as exc:
                                    _handle_model_failure(model_name, f"error: {exc}")

                    if "M8_reverse_pair" in args.enabled_models:
                        if baseline_group:
                            model_skip["M8_reverse_pair"] = "not_applicable_nonpaired_condition"
                        elif args.pair_id_col not in metadata.columns and "pic_num" not in metadata.columns:
                            _handle_model_failure("M8_reverse_pair", "missing_pair_id")
                        else:
                            try:
                                model_vectors["M8_reverse_pair"] = model_from_reverse_pair_identity(
                                    metadata, args.pair_id_col
                                )
                            except Exception as exc:
                                _handle_model_failure("M8_reverse_pair", f"error: {exc}")

                    for col in args.numeric_model_cols:
                        if col in metadata.columns:
                            try:
                                model_vectors[f"numeric_{col}"] = model_from_numeric(metadata, col)
                            except Exception as exc:
                                _handle_model_failure(f"numeric_{col}", f"error: {exc}", allow_skip=True)

                    if not model_vectors and not model_skip:
                        raise RuntimeError(
                            f"No model vectors were built for {subject} {roi_name} {time} {condition_group}."
                        )

                    if audit_root is not None:
                        _save_rdm_audit(
                            audit_root,
                            subject=subject,
                            roi_name=roi_name,
                            time=time,
                            condition_group=condition_group,
                            metadata=metadata,
                            neural=neural,
                            model_vectors=model_vectors,
                            model_skip=model_skip,
                        )

                    for model_name, model_vector in model_vectors.items():
                        rho, pvalue, n_pairs = _safe_spearman(neural, model_vector)
                        rows.append({
                            "subject": subject, "roi": roi_name, "time": time,
                            "condition_group": condition_group,
                            "model": model_name, "rho": rho, "p_value": pvalue,
                            "n_pairs": n_pairs,
                            "n_conditions_available": len(image_paths),
                            "skip_reason": "",
                        })

                    for model_name, reason in model_skip.items():
                        rows.append({
                            "subject": subject, "roi": roi_name, "time": time,
                            "condition_group": condition_group,
                            "model": model_name, "rho": float("nan"), "p_value": float("nan"),
                            "n_pairs": 0,
                            "n_conditions_available": len(image_paths),
                            "skip_reason": reason,
                        })

                    if args.partial_correlation and len(model_vectors) >= 2:
                        names = list(model_vectors.keys())
                        for name in names:
                            control_names, excluded_controls = _partial_control_names(names, name)
                            controls = [model_vectors[other] for other in control_names]
                            rho, pvalue, n_pairs = _partial_spearman(neural, model_vectors[name], controls)
                            partial_rows.append({
                                "subject": subject, "roi": roi_name, "time": time,
                                "condition_group": condition_group,
                                "model": name, "partial_rho": rho, "p_value": pvalue,
                                "n_pairs": n_pairs,
                                "controls": ",".join(control_names),
                                "excluded_controls": ",".join(excluded_controls),
                            })
                        collinearity_snapshots.append(
                            _build_collinearity_report(model_vectors, args.collinearity_threshold)
                            .assign(subject=subject, roi=roi_name, time=time, condition_group=condition_group)
                        )

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "model_rdm_subject_metrics.tsv")

    if partial_rows:
        write_table(pd.DataFrame(partial_rows), output_dir / "model_rdm_partial_metrics.tsv")

    if collinearity_snapshots:
        collinearity = pd.concat(collinearity_snapshots, ignore_index=True)
        write_table(collinearity, output_dir / "model_collinearity.tsv")

    summaries = []
    eligible = frame[(frame["model"].notna()) & (frame["skip_reason"] == "")]
    group_cols = ["roi", "model"]
    if "condition_group" in eligible.columns and eligible["condition_group"].notna().any():
        group_cols.insert(1, "condition_group")
    for keys, sub_frame in eligible.groupby(group_cols):
        pivot = sub_frame.pivot_table(index="subject", columns="time", values="rho").dropna()
        if "pre" not in pivot.columns or "post" not in pivot.columns or pivot.empty:
            continue
        if len(group_cols) == 3:
            roi_name, condition_group, model_name = keys
            summary = {
                "roi": roi_name,
                "condition_group": condition_group,
                "model": model_name,
                **paired_t_summary(pivot["post"], pivot["pre"]),
            }
            summary_name = f"{roi_name}_{condition_group}_{model_name}_rdm_summary.json"
        else:
            roi_name, model_name = keys
            summary = {"roi": roi_name, "model": model_name,
                       **paired_t_summary(pivot["post"], pivot["pre"])}
            summary_name = f"{roi_name}_{model_name}_rdm_summary.json"
        summaries.append(summary)
        save_json(summary, output_dir / summary_name)

    if summaries:
        write_table(pd.DataFrame(summaries), output_dir / "model_rdm_group_summary.tsv")

    if audit_root is not None:
        save_json(
            {
                "audit_root": str(audit_root),
                "note": "Each subject/roi/time/condition cell stores trial-order metadata and compressed RDM vectors.",
            },
            output_dir / "model_rdm_audit_manifest.json",
        )

    print(f"[model_rdm_comparison] wrote results to {output_dir}")


if __name__ == "__main__":
    main()
