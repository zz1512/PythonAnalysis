from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.decoding import SearchLight
from scipy import stats
from scipy.ndimage import generate_binary_structure, label
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from report_generator import generate_html_report, image_to_base64
from visualization import create_enhanced_visualizations


LOGGER = logging.getLogger("fmri_mvpa_searchlight")


def log(message: str, config: Optional["MVPAConfig"] = None, level: int = logging.INFO) -> None:
    """统一的日志入口，默认同时写入文件与控制台。"""

    LOGGER.log(level, message)


def memory_cleanup() -> None:
    """显式触发垃圾回收，避免长时间任务占用内存。"""

    gc.collect()


@dataclass
class MVPAConfig:
    """Searchlight MVPA 分析的核心配置。"""

    subjects: List[str] = field(default_factory=lambda: [f"sub-{i:02d}" for i in range(1, 29)])
    runs: List[int] = field(default_factory=lambda: [3, 4])
    lss_root: Path = field(default_factory=lambda: Path("../../../learn_LSS"))
    mask_dir: Path = field(default_factory=lambda: Path("../../../data/masks"))
    process_mask: str = "gray_matter_mask.nii.gz"
    results_dir: Path = field(default_factory=lambda: Path("../../../learn_mvpa/searchlight_mvpa"))

    searchlight_radius: int = 3
    min_region_size: int = 10
    svm_params: Dict[str, Any] = field(
        default_factory=lambda: {"kernel": "linear", "C": 1.0, "random_state": 42}
    )
    cv_folds: int = 5
    cv_random_state: int = 42
    within_subject_permutations: int = 50
    permutation_random_state: int = 42
    n_permutations: int = 1000
    max_exact_sign_flips_subjects: int = 12
    alpha_level: float = 0.05
    classification_significance_threshold: float = 0.55
    n_jobs: int = 1

    contrasts: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [("metaphor_vs_space", "yy", "kj")]
    )
    min_trials_per_condition: int = 5

    report_filename: str = "searchlight_mvpa_report.html"
    debug_mode: bool = False
    debug_subjects: List[str] = field(default_factory=lambda: [f"sub-{i:02d}" for i in range(1, 3)])

    log_level: int = logging.INFO
    log_file: Path | None = None

    available_subjects: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.lss_root = Path(self.lss_root)
        self.mask_dir = Path(self.mask_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "subject_results").mkdir(exist_ok=True)
        (self.results_dir / "group_results").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)

        self.log_file = self.results_dir / "logs" / "analysis.log"
        self._setup_logging()

        log("=======================================", self)
        log("启动 Searchlight MVPA 分析管线", self)
        log(f"结果目录: {self.results_dir.resolve()}", self)
        log(f"处理mask: {self.mask_dir / self.process_mask}", self)

    def _setup_logging(self) -> None:
        LOGGER.handlers.clear()
        LOGGER.setLevel(self.log_level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if self.log_file is not None:
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            LOGGER.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        LOGGER.addHandler(console_handler)

    def save_config(self, filepath: Path | str) -> None:
        config_dict: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (list, dict, str, int, float, bool)):
                config_dict[key] = value
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

def validate_config(config: MVPAConfig) -> bool:
    """检查基础路径和可用被试，确保后续分析有意义。"""

    log("验证配置参数...", config)
    issues: List[str] = []

    if not config.lss_root.exists():
        issues.append(f"LSS 根目录不存在: {config.lss_root}")
    if not config.mask_dir.exists():
        issues.append(f"Mask 目录不存在: {config.mask_dir}")

    mask_path = config.mask_dir / config.process_mask
    if not mask_path.exists():
        issues.append(f"处理 mask 缺失: {mask_path}")

    available = []
    for subject in config.subjects:
        if (config.lss_root / subject).exists():
            available.append(subject)
    config.available_subjects = available

    if len(available) < 3:
        issues.append(f"可用被试不足: 发现 {len(available)} 个")

    if issues:
        log("配置验证失败:", config, level=logging.ERROR)
        for issue in issues:
            log(f"  ⚠️ {issue}", config, level=logging.ERROR)
        return False

    log(f"配置验证通过: {len(available)} 名可用被试", config)
    return True


def load_lss_trial_data(subject: str, run: int, config: MVPAConfig) -> Tuple[Optional[List[str]], Optional[pd.DataFrame]]:
    """加载单个 run 的 LSS trial 数据及对应的 beta 图像。"""

    lss_dir = config.lss_root / subject / f"run-{run}_LSS"
    if not lss_dir.exists():
        log(f"LSS 目录不存在: {lss_dir}", config)
        return None, None

    trial_path = lss_dir / "trial_info.csv"
    if not trial_path.exists():
        log(f"trial_info.csv 缺失: {trial_path}", config)
        return None, None

    trial_info = pd.read_csv(trial_path)

    beta_images: List[str] = []
    valid_trials: List[pd.Series] = []
    missing: List[str] = []

    for _, trial in trial_info.iterrows():
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        if beta_path.exists():
            beta_images.append(str(beta_path))
            trial_copy = trial.copy()
            trial_copy["run"] = run
            valid_trials.append(trial_copy)
        else:
            missing.append(beta_path.name)

    if missing:
        log(f"run {run}: 缺失 {len(missing)} 个 beta 文件", config)
    log(f"run {run}: 成功加载 {len(beta_images)} 个 trial", config)

    if not beta_images:
        return None, None

    trial_df = pd.DataFrame(valid_trials)
    return beta_images, trial_df


def load_multi_run_lss_data(subject: str, runs: Sequence[int], config: MVPAConfig) -> Tuple[Optional[List[str]], Optional[pd.DataFrame]]:
    """整合多个 run 的 trial 数据并保持索引一致。"""

    log(f"加载被试 {subject} 的多 run LSS 数据: {list(runs)}", config)
    beta_all: List[str] = []
    trial_frames: List[pd.DataFrame] = []
    global_index = 0

    for run in runs:
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        if beta_images is None or trial_info is None:
            log(f"  run {run} 数据无效，跳过", config, level=logging.WARNING)
            continue

        trial_info = trial_info.copy()
        trial_info["combined_trial_index"] = range(global_index, global_index + len(trial_info))
        global_index += len(trial_info)

        beta_all.extend(beta_images)
        trial_frames.append(trial_info)

    if not trial_frames:
        log(f"被试 {subject} 没有可用的 LSS 数据", config, level=logging.ERROR)
        return None, None

    combined = pd.concat(trial_frames, ignore_index=True)

    condition_col = "original_condition" if "original_condition" in combined.columns else "condition"
    counts = combined[condition_col].value_counts().to_dict()
    log(f"  条件分布: {counts}", config)

    return beta_all, combined


def prepare_searchlight_data(
    trial_info: pd.DataFrame,
    beta_images: Sequence[str],
    cond1: str,
    cond2: str,
    config: MVPAConfig,
) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    """筛选目标条件并返回相应的 beta 图像及标签。"""

    condition_col = "original_condition" if "original_condition" in trial_info.columns else "condition"
    cond1_trials = trial_info[trial_info[condition_col] == cond1]
    cond2_trials = trial_info[trial_info[condition_col] == cond2]

    if len(cond1_trials) < config.min_trials_per_condition or len(cond2_trials) < config.min_trials_per_condition:
        log(
            f"条件样本不足: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)} (阈值 {config.min_trials_per_condition})",
            config,
        )
        return None, None, None, {}

    selected_paths: List[str] = []
    labels: List[int] = []
    groups: List[int] = []

    def _append_trials(trials: Iterable[pd.Series], label: int) -> None:
        for _, trial in trials:
            if "combined_trial_index" in trial:
                idx = int(trial["combined_trial_index"])
            else:
                idx = int(trial["trial_index"]) - 1
            if idx < 0 or idx >= len(beta_images):
                continue
            selected_paths.append(str(beta_images[idx]))
            labels.append(label)
            groups.append(int(trial.get("run", -1)))

    _append_trials(cond1_trials.iterrows(), 0)
    _append_trials(cond2_trials.iterrows(), 1)

    if len(selected_paths) < 6:
        log("有效 trial 数不足以进行稳定的交叉验证，跳过", config)
        return None, None, None, {}

    stats_summary = {
        "n_trials_cond1": int(len(cond1_trials)),
        "n_trials_cond2": int(len(cond2_trials)),
    }

    return selected_paths, np.asarray(labels, dtype=int), np.asarray(groups, dtype=int), stats_summary


def make_searchlight_estimator(config: MVPAConfig) -> Pipeline:
    """构建用于 Searchlight 的分类 pipeline。"""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svc", SVC(**config.svm_params)),
        ]
    )


def make_cv_splitter(labels: np.ndarray, groups: Optional[np.ndarray], config: MVPAConfig) -> Tuple[object, bool]:
    """根据标签与组信息选择合适的交叉验证策略。"""

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2 or counts.min() < 2:
        raise ValueError("每个类别至少需要两个样本用于交叉验证")

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) > 1:
            log(f"使用 Leave-One-Run-Out 交叉验证 (组数={len(unique_groups)})", config)
            return LeaveOneGroupOut(), True

    n_splits = max(2, min(config.cv_folds, counts.min()))
    log(f"使用 StratifiedKFold, n_splits={n_splits}", config)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.cv_random_state), False


def estimate_subject_chance_map(
    beta_img,
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    mask_img,
    config: MVPAConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """通过标签置换估计个体层面的机会水平地图。"""

    n_perm = int(config.within_subject_permutations)
    if n_perm <= 0:
        return np.full(mask_img.shape, 0.5, dtype=float), None

    rng = np.random.default_rng(config.permutation_random_state)
    null_maps: List[np.ndarray] = []

    for perm_idx in range(n_perm):
        permuted = labels.copy()
        if groups is not None and len(np.unique(groups)) > 1:
            for group_value in np.unique(groups):
                mask = groups == group_value
                rng.shuffle(permuted[mask])
        else:
            rng.shuffle(permuted)

        cv, require_groups = make_cv_splitter(permuted, groups, config)
        sl = SearchLight(
            mask_img=mask_img,
            radius=config.searchlight_radius,
            estimator=make_searchlight_estimator(config),
            n_jobs=config.n_jobs,
            scoring="accuracy",
            cv=cv,
            verbose=0,
        )
        sl.fit(beta_img, permuted, groups=groups if require_groups else None)
        null_maps.append(np.asarray(sl.scores_, dtype=float))

    null_array = np.stack(null_maps, axis=0)
    chance_data = np.mean(null_array, axis=0)
    return chance_data, null_array


def run_subject_searchlight(
    subject: str,
    contrast_name: str,
    beta_paths: Sequence[str],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    mask_img,
    mask_data: np.ndarray,
    config: MVPAConfig,
    trial_summary: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """执行单被试 Searchlight 分析并保存结果。"""

    log(f"开始被试 {subject} 的 Searchlight: {contrast_name}", config)

    beta_img = image.concat_imgs(beta_paths)
    cv, require_groups = make_cv_splitter(labels, groups, config)
    searchlight = SearchLight(
        mask_img=mask_img,
        radius=config.searchlight_radius,
        estimator=make_searchlight_estimator(config),
        n_jobs=config.n_jobs,
        scoring="accuracy",
        cv=cv,
        verbose=0,
    )
    searchlight.fit(beta_img, labels, groups=groups if require_groups else None)

    accuracy_data = np.asarray(searchlight.scores_, dtype=float)
    chance_data, null_array = estimate_subject_chance_map(beta_img, labels, groups, mask_img, config)
    delta_data = accuracy_data - chance_data

    mask_bool = mask_data.astype(bool)
    mean_accuracy = float(np.nanmean(accuracy_data[mask_bool]))
    mean_chance = float(np.nanmean(chance_data[mask_bool]))
    mean_delta = float(np.nanmean(delta_data[mask_bool]))

    subject_dir = config.results_dir / "subject_results" / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    contrast_tag = contrast_name.replace(" ", "_")

    accuracy_img = image.new_img_like(mask_img, accuracy_data)
    chance_img = image.new_img_like(mask_img, chance_data)
    delta_img = image.new_img_like(mask_img, delta_data)

    accuracy_path = subject_dir / f"{subject}_{contrast_tag}_accuracy.nii.gz"
    chance_path = subject_dir / f"{subject}_{contrast_tag}_chance.nii.gz"
    delta_path = subject_dir / f"{subject}_{contrast_tag}_delta.nii.gz"

    accuracy_img.to_filename(str(accuracy_path))
    chance_img.to_filename(str(chance_path))
    delta_img.to_filename(str(delta_path))

    log(
        f"  平均准确率={mean_accuracy:.3f}, 机会水平={mean_chance:.3f}, 差值={mean_delta:.3f}",
        config,
    )

    result = {
        "subject": subject,
        "contrast": contrast_name,
        "accuracy_map_path": accuracy_path,
        "chance_map_path": chance_path,
        "delta_map_path": delta_path,
        "mean_accuracy": mean_accuracy,
        "mean_chance_accuracy": mean_chance,
        "mean_delta_accuracy": mean_delta,
        "null_distribution": null_array,
        **trial_summary,
    }

    return {
        "summary": result,
        "accuracy_data": accuracy_data,
        "chance_data": chance_data,
        "delta_data": delta_data,
    }


def fdr_correction(p_values: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    """对 p 值执行 Benjamini-Hochberg FDR 校正。"""

    finite_mask = np.isfinite(p_values)
    p_valid = p_values[finite_mask]
    if p_valid.size == 0:
        return np.nan, np.zeros_like(p_values, dtype=bool)

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    n = len(ranked_p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    below = ranked_p <= thresholds
    if not np.any(below):
        return np.nan, np.zeros_like(p_values, dtype=bool)

    max_idx = np.where(below)[0].max()
    cutoff = ranked_p[max_idx]

    significant = np.zeros_like(p_values, dtype=bool)
    significant_indices = np.where(finite_mask)[0][order[: max_idx + 1]]
    significant[significant_indices] = True
    return float(cutoff), significant


def detect_significant_clusters(
    mask: np.ndarray,
    delta_map: np.ndarray,
    t_map: np.ndarray,
    p_map: np.ndarray,
    affine: np.ndarray,
    config: MVPAConfig,
    contrast_name: str,
) -> pd.DataFrame:
    """在显著掩模上识别空间簇并返回统计摘要。"""

    structure = generate_binary_structure(rank=3, connectivity=2)
    labeled, n_clusters = label(mask, structure=structure)

    if n_clusters == 0:
        return pd.DataFrame()

    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    rows: List[Dict[str, Any]] = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        n_voxels = int(cluster_mask.sum())
        if n_voxels < config.min_region_size:
            continue

        delta_vals = delta_map[cluster_mask]
        t_vals = t_map[cluster_mask]
        p_vals = p_map[cluster_mask]

        peak_idx = int(np.nanargmax(delta_vals))
        peak_voxel = np.argwhere(cluster_mask)[peak_idx]
        peak_world = nib.affines.apply_affine(affine, peak_voxel)

        rows.append(
            {
                "contrast": contrast_name,
                "cluster_id": cluster_id,
                "n_voxels": n_voxels,
                "cluster_volume_mm3": n_voxels * voxel_volume,
                "mean_delta_accuracy": float(np.nanmean(delta_vals)),
                "peak_delta_accuracy": float(np.nanmax(delta_vals)),
                "peak_t_value": float(t_vals.flat[peak_idx]),
                "peak_p_value": float(p_vals.flat[peak_idx]),
                "peak_x": float(peak_world[0]),
                "peak_y": float(peak_world[1]),
                "peak_z": float(peak_world[2]),
            }
        )

    return pd.DataFrame(rows)


def compute_sign_flip_pvalue(values: np.ndarray, config: MVPAConfig) -> Tuple[float, int]:
    """对组均值执行符号翻转检验。"""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), 0

    observed = float(np.mean(values))

    if n <= config.max_exact_sign_flips_subjects:
        import itertools

        flips = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        rng = np.random.default_rng(config.permutation_random_state)
        flips = rng.choice([-1, 1], size=(config.n_permutations, n))

    perm_means = flips * values
    perm_means = perm_means.mean(axis=1)
    greater = np.mean(np.abs(perm_means) >= abs(observed))
    return float(greater), perm_means.shape[0]


def compute_group_statistics(
    subject_summaries: pd.DataFrame,
    config: MVPAConfig,
) -> pd.DataFrame:
    """汇总每个对比条件的组水平统计指标。"""

    rows: List[Dict[str, Any]] = []

    for contrast_name, group in subject_summaries.groupby("contrast"):
        delta_values = group["mean_delta_accuracy"].to_numpy(dtype=float)
        chance_values = group["mean_chance_accuracy"].to_numpy(dtype=float)
        accuracy_values = group["mean_accuracy"].to_numpy(dtype=float)

        mean_delta = float(np.nanmean(delta_values))
        mean_accuracy = float(np.nanmean(accuracy_values))
        mean_chance = float(np.nanmean(chance_values))
        sem_delta = float(stats.sem(delta_values, nan_policy="omit")) if len(delta_values) > 1 else float("nan")

        t_stat, t_pvalue = stats.ttest_1samp(delta_values, popmean=0.0, nan_policy="omit")
        cohens_d = float(mean_delta / (np.nanstd(delta_values, ddof=1) + 1e-12))

        p_sign, n_perm = compute_sign_flip_pvalue(delta_values, config)

        ci_low, ci_high = float("nan"), float("nan")
        if len(delta_values) > 1:
            dof = len(delta_values) - 1
            interval = stats.t.interval(0.95, dof, loc=mean_delta, scale=stats.sem(delta_values, nan_policy="omit"))
            if isinstance(interval, tuple):
                ci_low, ci_high = map(float, interval)

        rows.append(
            {
                "contrast": contrast_name,
                "n_subjects": int(len(group)),
                "mean_accuracy": mean_accuracy,
                "mean_chance_accuracy": mean_chance,
                "mean_delta_accuracy": mean_delta,
                "sem_delta_accuracy": sem_delta,
                "ci_lower_delta": ci_low,
                "ci_upper_delta": ci_high,
                "t_statistic": float(t_stat),
                "t_pvalue": float(t_pvalue),
                "cohens_d": cohens_d,
                "sign_flip_pvalue": p_sign,
                "n_group_permutations": n_perm,
                "significant_t_test": bool(t_pvalue < config.alpha_level),
                "significant_permutation": bool(p_sign < config.alpha_level),
            }
        )

    return pd.DataFrame(rows)


def compute_group_level_maps(
    contrast_name: str,
    subject_maps: List[np.ndarray],
    chance_maps: List[np.ndarray],
    mask_img,
    mask_data: np.ndarray,
    config: MVPAConfig,
) -> Tuple[Dict[str, Path], pd.DataFrame, Dict[str, Any]]:
    """计算组水平地图并返回路径、显著簇与显著性摘要。"""

    group_dir = config.results_dir / "group_results"
    group_dir.mkdir(exist_ok=True)

    mask_bool = mask_data.astype(bool)
    accuracy_stack = np.stack(subject_maps, axis=0)
    chance_stack = np.stack(chance_maps, axis=0)
    delta_stack = accuracy_stack - chance_stack

    mean_accuracy_flat = np.nanmean(accuracy_stack[:, mask_bool], axis=0)
    mean_chance_flat = np.nanmean(chance_stack[:, mask_bool], axis=0)
    mean_delta_flat = mean_accuracy_flat - mean_chance_flat

    t_stat, t_pvalue = stats.ttest_1samp(delta_stack[:, mask_bool], popmean=0.0, axis=0, nan_policy="omit")
    t_stat = np.asarray(t_stat)
    t_pvalue = np.asarray(t_pvalue)

    mean_accuracy_data = np.zeros(mask_data.shape, dtype=float)
    mean_chance_data = np.zeros(mask_data.shape, dtype=float)
    mean_delta_data = np.zeros(mask_data.shape, dtype=float)
    t_map_data = np.zeros(mask_data.shape, dtype=float)
    p_map_data = np.ones(mask_data.shape, dtype=float)

    mean_accuracy_data[mask_bool] = mean_accuracy_flat
    mean_chance_data[mask_bool] = mean_chance_flat
    mean_delta_data[mask_bool] = mean_delta_flat
    t_map_data[mask_bool] = t_stat
    p_map_data[mask_bool] = t_pvalue

    classification_threshold = config.classification_significance_threshold
    classification_mask = (mean_accuracy_data >= classification_threshold) & mask_bool
    n_voxels_above_threshold = int(np.sum(classification_mask))

    fdr_threshold, significant_vector = fdr_correction(t_pvalue, config.alpha_level)
    significant_mask = np.zeros(mask_data.shape, dtype=bool)
    significant_mask[mask_bool] = significant_vector
    significant_mask &= classification_mask
    n_significant_voxels = int(np.sum(significant_mask))

    thresholded_delta = np.where(significant_mask, mean_delta_data, 0.0)

    accuracy_img = image.new_img_like(mask_img, mean_accuracy_data)
    chance_img = image.new_img_like(mask_img, mean_chance_data)
    delta_img = image.new_img_like(mask_img, mean_delta_data)
    t_img = image.new_img_like(mask_img, t_map_data)
    p_img = image.new_img_like(mask_img, p_map_data)
    classification_img = image.new_img_like(mask_img, classification_mask.astype(float))
    significant_img = image.new_img_like(mask_img, significant_mask.astype(float))
    thresholded_delta_img = image.new_img_like(mask_img, thresholded_delta)

    tag = contrast_name.replace(" ", "_")
    accuracy_path = group_dir / f"{tag}_mean_accuracy.nii.gz"
    chance_path = group_dir / f"{tag}_mean_chance.nii.gz"
    delta_path = group_dir / f"{tag}_mean_delta.nii.gz"
    t_path = group_dir / f"{tag}_t_map.nii.gz"
    p_path = group_dir / f"{tag}_p_map.nii.gz"
    class_mask_path = group_dir / f"{tag}_classification_mask.nii.gz"
    sig_mask_path = group_dir / f"{tag}_significant_mask.nii.gz"
    thresh_delta_path = group_dir / f"{tag}_thresholded_delta.nii.gz"

    accuracy_img.to_filename(str(accuracy_path))
    chance_img.to_filename(str(chance_path))
    delta_img.to_filename(str(delta_path))
    t_img.to_filename(str(t_path))
    p_img.to_filename(str(p_path))
    classification_img.to_filename(str(class_mask_path))
    significant_img.to_filename(str(sig_mask_path))
    thresholded_delta_img.to_filename(str(thresh_delta_path))

    log(
        f"组水平地图: {contrast_name} -> {accuracy_path.name}, {delta_path.name}, 显著体素 {n_significant_voxels}",
        config,
    )

    cluster_df = detect_significant_clusters(
        significant_mask,
        mean_delta_data,
        t_map_data,
        p_map_data,
        mask_img.affine,
        config,
        contrast_name,
    )

    cluster_csv_path = None
    if not cluster_df.empty:
        cluster_csv_path = group_dir / f"{tag}_significant_clusters.csv"
        cluster_df.to_csv(cluster_csv_path, index=False)
        log(f"显著簇汇总: {cluster_csv_path}", config)

    mean_significant_delta = float(np.nanmean(mean_delta_data[significant_mask])) if n_significant_voxels else float("nan")
    peak_significant_accuracy = (
        float(np.nanmax(mean_accuracy_data[significant_mask])) if n_significant_voxels else float("nan")
    )

    outputs = {
        "mean_accuracy_map": accuracy_path,
        "mean_chance_map": chance_path,
        "mean_delta_map": delta_path,
        "t_map": t_path,
        "p_map": p_path,
        "classification_mask": class_mask_path,
        "classification_threshold": classification_threshold,
        "n_voxels_above_threshold": n_voxels_above_threshold,
        "significant_mask": sig_mask_path,
        "thresholded_delta_map": thresh_delta_path,
        "cluster_table": cluster_csv_path,
        "fdr_threshold": fdr_threshold,
        "n_significant_voxels": n_significant_voxels,
    }

    significance_summary = {
        "contrast": contrast_name,
        "correction": "fdr",
        "classification_threshold": classification_threshold,
        "n_voxels_above_threshold": n_voxels_above_threshold,
        "n_significant_voxels": n_significant_voxels,
        "proportion_significant": (
            n_significant_voxels / n_voxels_above_threshold if n_voxels_above_threshold else 0.0
        ),
        "mean_significant_delta": mean_significant_delta,
        "peak_significant_accuracy": peak_significant_accuracy,
        "cluster_table": cluster_csv_path,
    }

    return outputs, cluster_df, significance_summary


def run_group_searchlight_analysis(config: MVPAConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行完整的 Searchlight 分析并生成报告。"""

    if not validate_config(config):
        raise RuntimeError("配置验证失败，终止分析")

    subjects = [s for s in config.available_subjects if s in config.subjects]
    if config.debug_mode:
        subjects = [s for s in subjects if s in config.debug_subjects]
        log(f"调试模式启用，仅分析: {subjects}", config)

    if not subjects:
        raise RuntimeError("没有可用被试用于分析")

    mask_path = config.mask_dir / config.process_mask
    mask_img = image.load_img(str(mask_path))
    mask_data = mask_img.get_fdata() > 0

    subject_rows: List[Dict[str, Any]] = []
    map_collections: Dict[str, List[np.ndarray]] = {}
    chance_collections: Dict[str, List[np.ndarray]] = {}
    cluster_rows: List[pd.DataFrame] = []
    significance_rows: List[Dict[str, Any]] = []

    for subject in subjects:
        beta_images, trial_info = load_multi_run_lss_data(subject, config.runs, config)
        if beta_images is None or trial_info is None:
            log(f"跳过被试 {subject}: 数据不足", config, level=logging.WARNING)
            continue

        for contrast_name, cond1, cond2 in config.contrasts:
            prepared = prepare_searchlight_data(trial_info, beta_images, cond1, cond2, config)
            beta_paths, labels, groups, trial_summary = prepared
            if beta_paths is None or labels is None:
                log(f"被试 {subject} 对比 {contrast_name} 无有效 trial", config)
                continue

            result = run_subject_searchlight(
                subject,
                contrast_name,
                beta_paths,
                labels,
                groups,
                mask_img,
                mask_data,
                config,
                trial_summary,
            )
            if result is None:
                continue

            subject_rows.append(result["summary"])
            map_collections.setdefault(contrast_name, []).append(result["accuracy_data"])
            chance_collections.setdefault(contrast_name, []).append(result["chance_data"])

    if not subject_rows:
        raise RuntimeError("没有成功的被试结果")

    subject_df = pd.DataFrame(subject_rows)
    stats_df = compute_group_statistics(subject_df, config)

    map_outputs: Dict[str, Dict[str, Path]] = {}

    for contrast_name in map_collections:
        if len(map_collections[contrast_name]) < 2:
            log(f"对比 {contrast_name} 的有效被试不足，跳过组水平地图", config, level=logging.WARNING)
            continue

        outputs, cluster_df, significance_summary = compute_group_level_maps(
            contrast_name,
            map_collections[contrast_name],
            chance_collections[contrast_name],
            mask_img,
            mask_data,
            config,
        )
        map_outputs[contrast_name] = outputs
        if not cluster_df.empty:
            cluster_rows.append(cluster_df)
        significance_rows.append(significance_summary)

    cluster_summary_df = pd.concat(cluster_rows, ignore_index=True) if cluster_rows else pd.DataFrame()
    cluster_summary_path = None
    if not cluster_summary_df.empty:
        cluster_summary_path = config.results_dir / "group_results" / "group_significant_clusters_summary.csv"
        cluster_summary_df.to_csv(cluster_summary_path, index=False)

    significance_summary_df = pd.DataFrame(significance_rows)
    significance_summary_path = None
    if not significance_summary_df.empty:
        significance_summary_path = config.results_dir / "group_results" / "group_significant_voxel_summary.csv"
        significance_summary_df.to_csv(significance_summary_path, index=False)

    main_viz_path, individual_viz_path, significance_viz_path = create_enhanced_visualizations(
        subject_df, stats_df, config, significance_summary_df
    )

    report_path = generate_html_report(
        subject_df,
        stats_df,
        config,
        main_viz_b64=image_to_base64(main_viz_path) if main_viz_path else None,
        individual_viz_b64=image_to_base64(individual_viz_path) if individual_viz_path else None,
        map_outputs=map_outputs,
        cluster_df=cluster_summary_df,
        cluster_summary_path=cluster_summary_path,
        significance_summary_df=significance_summary_df,
        significance_summary_path=significance_summary_path,
        significance_viz_b64=image_to_base64(significance_viz_path) if significance_viz_path else None,
    )

    log(f"报告生成完成: {report_path}", config)

    memory_cleanup()
    return subject_df, stats_df


def main() -> Optional[Path]:
    """命令行入口。"""

    config = MVPAConfig()
    try:
        subject_df, stats_df = run_group_searchlight_analysis(config)
        log(f"完成 Searchlight 分析，共 {len(subject_df)} 条个体结果", config)
        return config.results_dir
    except Exception as exc:
        log(f"分析失败: {exc}", config, level=logging.ERROR)
        raise


if __name__ == "__main__":
    main()
