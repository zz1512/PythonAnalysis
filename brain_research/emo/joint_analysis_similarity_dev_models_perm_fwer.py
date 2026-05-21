"""
对8个被试×被试相似性矩阵（全脑L/R + 3 ROI×L/R）做联合分析：
- 统计量：相似性矩阵上三角向量 与 3个发育模型上三角向量 的 Pearson r
- 显著性：对年龄标签做置换（label permutation）得到经验 p 值
- 多重比较：Westfall–Young max-statistic (按模型分组校正)
    - 修改点：FWER 校正在**每个模型内部**独立进行 (Model-wise correction)。
    - 即：NN 模型内部校正 8 个矩阵，Conv 内部校正 8 个矩阵...
- 其他特性：
    1. 自动寻找 subject_order 文件。
    2. 只关注正效应（单侧检验）。
    3. 默认置换次数 5000。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd


COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

DEFAULT_FINAL_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
DEFAULT_ROI_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi_age_sorted")
DEFAULT_SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")

DEFAULT_ROIS = ("Visual", "Salience_Insula", "Control_PFC")


@dataclass(frozen=True)
class SimilarityInput:
    name: str
    path: Path


@dataclass(frozen=True)
class DevModelVectors:
    nn: np.ndarray
    conv: np.ndarray
    div: np.ndarray


def parse_chinese_age_exact(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d_match = re.search(r"(\d+)\s*天", s)
    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    path_str = str(subject_info_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(
            "年龄表缺少必要列。需要以下任意一组列名：\n"
            f"- {COL_SUB_ID}, {COL_AGE}\n"
            f"- {LEGACY_COL_SUB_ID}, {LEGACY_COL_AGE}\n"
            f"实际列名: {list(info_df.columns)}"
        )

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = parse_chinese_age_exact(row[age_col])
    return age_map


def load_subject_order_list(order_path: Path) -> List[str]:
    """读取指定顺序的被试列表"""
    if not order_path.exists():
        raise FileNotFoundError(f"指定的 subject_order 文件不存在: {order_path}")

    order_df = pd.read_csv(order_path)
    # 兼容不同的列名
    if "subject" in order_df.columns:
        return order_df["subject"].astype(str).tolist()
    elif "sub_id" in order_df.columns:
        return order_df["sub_id"].astype(str).tolist()
    elif "被试编号" in order_df.columns:
        return order_df["被试编号"].astype(str).tolist()
    else:
        # 如果只有一列，直接取第一列
        return order_df.iloc[:, 0].astype(str).tolist()


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[mask].mean())
    sd = float(x[mask].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out.astype(np.float32, copy=False)


def build_dev_model_vectors(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool = True) -> DevModelVectors:
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages 包含 NaN/Inf，无法构建模型向量。")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32, copy=False)
    conv = np.minimum(ai, aj).astype(np.float32, copy=False)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32, copy=False)
    if normalize:
        return DevModelVectors(zscore_1d(nn), zscore_1d(conv), zscore_1d(div))
    return DevModelVectors(nn, conv, div)


def expected_similarity_inputs(final_dir: Path, roi_dir: Path, rois: Sequence[str]) -> List[SimilarityInput]:
    items: List[SimilarityInput] = []
    items.append(SimilarityInput("surface_L", final_dir / "similarity_surface_L_AgeSorted.csv"))
    items.append(SimilarityInput("surface_R", final_dir / "similarity_surface_R_AgeSorted.csv"))
    for roi in rois:
        items.append(SimilarityInput(f"surface_L_{roi}", roi_dir / f"similarity_surface_L_{roi}_AgeSorted.csv"))
        items.append(SimilarityInput(f"surface_R_{roi}", roi_dir / f"similarity_surface_R_{roi}_AgeSorted.csv"))
    missing = [str(x.path) for x in items if not x.path.exists()]
    if missing:
        raise FileNotFoundError("未找到以下相似性矩阵文件:\n" + "\n".join(missing))
    return items


def load_similarity_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def intersect_subjects(dfs: Sequence[pd.DataFrame]) -> List[str]:
    if not dfs:
        return []
    s = set(dfs[0].index.tolist())
    for df in dfs[1:]:
        s &= set(df.index.tolist())
    return sorted(s)


def resolve_automatic_subject_list(
    final_dir: Path,
    similarity_dfs: Sequence[pd.DataFrame],
    input_order_path: Optional[Path]
) -> List[str]:
    # 1. 确定要使用的 Order 文件路径
    target_order_path: Optional[Path] = None

    if input_order_path is not None:
        target_order_path = input_order_path
    else:
        candidate_l = final_dir / "subject_order_surface_L.csv"
        candidate_r = final_dir / "subject_order_surface_R.csv"

        if candidate_l.exists():
            print(f"自动检测到标准被试顺序文件: {candidate_l}")
            target_order_path = candidate_l
        elif candidate_r.exists():
            print(f"自动检测到标准被试顺序文件: {candidate_r}")
            target_order_path = candidate_r
        else:
            print("未在输出目录找到 subject_order_surface_L.csv 或 R，将使用相似性矩阵的索引作为基准。")

    # 2. 获取基准被试列表
    base_subjects: List[str] = []

    if target_order_path and target_order_path.exists():
        base_subjects = load_subject_order_list(target_order_path)
    else:
        print("使用 surface_L 相似性矩阵的索引作为被试基准。")
        base_subjects = similarity_dfs[0].index.astype(str).tolist()

    # 3. 与所有矩阵的实际存在被试取交集
    available_in_all = set(intersect_subjects(similarity_dfs))
    final_subjects = [s for s in base_subjects if s in available_in_all]

    removed_count = len(base_subjects) - len(final_subjects)
    if removed_count > 0:
        print(f"注意: 基准列表中的 {removed_count} 个被试在部分 ROI 矩阵中缺失，已被剔除。")

    return final_subjects


def prepare_similarity_vectors(
    inputs: Sequence[SimilarityInput],
    subjects: Sequence[str],
) -> Tuple[np.ndarray, List[str], int]:
    n = len(subjects)
    iu, ju = triu_indices(n)
    vecs = []
    names = []
    for item in inputs:
        df = load_similarity_df(item.path)
        df = df.loc[subjects, subjects]
        mat = df.to_numpy(dtype=np.float32, copy=False)
        v = mat[iu, ju]
        v_z = zscore_1d(v)
        if not np.isfinite(v_z).any():
            raise ValueError(f"相似性矩阵向量无有效值: {item.path}")
        vecs.append(v_z)
        names.append(item.name)
    S = np.stack(vecs, axis=0).astype(np.float32, copy=False)
    return S, names, int(iu.size)


def run_fwer_permutation(
    inputs: Sequence[SimilarityInput],
    subject_info_path: Path,
    out_dir: Path,
    final_dir_for_auto_order: Path,
    subject_order_path: Optional[Path] = None,
    n_perm: int = 5000,
    seed: int = 42,
    normalize_models: bool = True,
) -> Path:
    # 加载所有矩阵
    print("正在加载相似性矩阵...")
    similarity_dfs = [load_similarity_df(x.path) for x in inputs]

    # 加载年龄
    age_map = load_subject_ages_map(subject_info_path)

    # 确定被试列表
    subjects = resolve_automatic_subject_list(
        final_dir_for_auto_order,
        similarity_dfs,
        subject_order_path
    )

    subjects = [s for s in subjects if s in age_map and np.isfinite(age_map[s])]
    subjects.sort(key=lambda sid: (age_map.get(sid, np.nan), sid))

    if len(subjects) < 10:
        raise ValueError(f"有效被试数量过少 ({len(subjects)})，无法进行分析。")

    print(f"最终纳入分析的被试数量: {len(subjects)}")
    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_dir / "subjects_perm_analysis.csv", index=False)

    # 提取相似性向量 (n_mats, n_pairs)
    S_z, matrix_names, n_pairs = prepare_similarity_vectors(inputs, subjects)
    n_mats = S_z.shape[0]

    # 构建观测数据的发育模型
    iu, ju = triu_indices(len(subjects))
    models_obs = build_dev_model_vectors(ages, iu, ju, normalize=normalize_models)
    model_names = ("M_nn", "M_conv", "M_div")
    model_vecs_obs = (models_obs.nn, models_obs.conv, models_obs.div)

    # -------------------------------------------------------------------------
    # 计算观测值相关系数 (r_obs)
    # 注意：这里的 r_obs 列表结构是 [Model1_Mat1, Model1_Mat2... Model1_MatN, Model2_Mat1...]
    # 这对后续按模型分组校正至关重要
    # -------------------------------------------------------------------------
    r_obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        mvec = model_vecs_obs[mi]
        # Pearson r
        r_mats = (S_z @ mvec) / float(n_pairs)
        for si in range(n_mats):
            tests.append({"matrix": matrix_names[si], "model": mname})
            r_obs.append(float(r_mats[si]))

    r_obs_arr = np.asarray(r_obs, dtype=np.float32)
    n_tests = int(r_obs_arr.size)

    # --- Permutation Test (One-sided, Positive Effect) ---
    print(f"开始置换检验 (N={n_perm}, 单侧-正效应)...")
    print(f"校正策略: Model-wise FWER (在每个发育模型内部独立校正)")

    count_raw = np.zeros(n_tests, dtype=np.int64)
    count_fwer = np.zeros(n_tests, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for i in range(int(n_perm)):
        if (i + 1) % 1000 == 0:
            print(f"  Permutation {i + 1} / {n_perm}")

        perm = rng.permutation(len(subjects))
        ages_p = ages[perm]

        # 重构置换后的模型
        models_p = build_dev_model_vectors(ages_p, iu, ju, normalize=normalize_models)
        model_vecs_p = (models_p.nn, models_p.conv, models_p.div)

        # 计算当前置换下所有 Matrix x Model 的相关值
        r_perm_all = np.empty(n_tests, dtype=np.float32)
        offset = 0
        for mvec in model_vecs_p:
            r_mats = (S_z @ mvec) / float(n_pairs)
            r_perm_all[offset:offset + n_mats] = r_mats.astype(np.float32, copy=False)
            offset += n_mats

        # 1. 原始 P 值 (无校正)
        count_raw += (r_perm_all >= r_obs_arr)

        # 2. FWER 校正 (Model-wise Max Statistic)
        # 针对每个模型（包含 n_mats 个矩阵），计算该模型下的最大统计量
        # 然后只用这个最大值去比较该模型下的观测值
        offset = 0
        for mi in range(len(model_names)):
            start_idx = offset
            end_idx = offset + n_mats

            # 当前模型下的置换统计量 (长度为 n_mats)
            r_perm_subset = r_perm_all[start_idx:end_idx]

            # 当前模型下的观测统计量
            r_obs_subset = r_obs_arr[start_idx:end_idx]

            # 计算该模型内部的最大统计量
            max_stat_model = float(r_perm_subset.max(initial=-1.0))

            # 更新该模型对应测试的 FWER 计数
            # (如果该模型下的最大噪音 >= 观测值，则计数)
            count_fwer[start_idx:end_idx] += (max_stat_model >= r_obs_subset)

            offset += n_mats

    # 计算 P 值
    p_perm = (count_raw + 1.0) / (float(n_perm) + 1.0)
    p_fwer = (count_fwer + 1.0) / (float(n_perm) + 1.0)

    rows = []
    for i in range(n_tests):
        meta = tests[i]
        rows.append(
            {
                "matrix": meta["matrix"],
                "model": meta["model"],
                "r_obs": float(r_obs_arr[i]),
                "p_perm_one_tailed": float(p_perm[i]),
                "p_fwer_model_wise": float(p_fwer[i]), # 列名变更为 model_wise 以区分
                "n_subjects": int(len(subjects)),
                "n_pairs": int(n_pairs),
                "n_perm": int(n_perm),
                "seed": int(seed),
                "normalize_models": bool(normalize_models),
            }
        )

    res_df = pd.DataFrame(rows).sort_values(["model", "matrix"]) # 按模型排序更易读
    out_path = out_dir / "joint_analysis_dev_models_perm_fwer.csv"
    res_df.to_csv(out_path, index=False)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="相似性矩阵与发育模型的置换检验 + Model-wise FWER 校正")
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR, help="全脑相似性矩阵目录")
    p.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR, help="ROI 相似性矩阵目录")
    p.add_argument("--subject-info", type=Path, default=DEFAULT_SUBJECT_INFO_PATH, help="被试信息表路径")
    p.add_argument(
        "--subject-order",
        type=Path,
        default=None,
        help="[可选] 手动指定 subject_order 文件"
    )
    p.add_argument("--out-dir", type=Path, default=Path("./joint_perm_fwer_out"), help="输出目录")
    p.add_argument("--n-perm", type=int, default=5000, help="置换次数 (默认 5000)")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--no-normalize", action="store_true", help="不对模型向量做z-score")
    p.add_argument("--rois", type=str, default=",".join(DEFAULT_ROIS), help="ROI 名称，用逗号分隔")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print(f"设定置换次数: {args.n_perm}")
    print(f"设定检验方向: 单侧 (One-sided, Positive Effect)")

    rois = [x.strip() for x in str(args.rois).split(",") if x.strip()]
    inputs = expected_similarity_inputs(args.final_dir, args.roi_dir, rois)

    out_path = run_fwer_permutation(
        inputs=inputs,
        subject_info_path=args.subject_info,
        out_dir=args.out_dir,
        final_dir_for_auto_order=args.final_dir,
        subject_order_path=args.subject_order,
        n_perm=args.n_perm,
        seed=args.seed,
        normalize_models=not args.no_normalize,
    )
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()