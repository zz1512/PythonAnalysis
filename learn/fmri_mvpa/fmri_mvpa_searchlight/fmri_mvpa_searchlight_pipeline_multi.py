from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
import json
import logging
import time
import sys
import io
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from multiprocessing import cpu_count
from nilearn.decoding import SearchLight
from nilearn.maskers import NiftiMasker
from nilearn import image
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from report_generator import generate_html_report, image_to_base64
from sklearn.impute import SimpleImputer
from visualization import create_enhanced_visualizations


# ============================================================================
# 常用MVPA分析参数定义
# ============================================================================


@dataclass(frozen=True)
class CommonParamDefinition:
    """描述常用MVPA参数的默认值及其说明。"""

    description: str
    default: Any = None
    default_factory: Optional[Callable[["MVPAConfig"], Any]] = None

    def resolve(self, config: "MVPAConfig") -> Any:
        """根据配置解析默认值。"""

        if self.default_factory is not None:
            return self.default_factory(config)
        return self.default


COMMON_MVPA_ANALYSIS_PARAMS = OrderedDict([
    (
        "searchlight_radius",
        CommonParamDefinition(
            default=3,
            description="Searchlight球体半径（以体素为单位），控制局部分类的空间邻域大小。",
        ),
    ),
    (
        "process_mask",
        CommonParamDefinition(
            default="gray_matter_mask.nii.gz",
            description="用于限定分析区域的mask文件名称，确保只在目标脑区进行计算。",
        ),
    ),
    (
        "min_region_size",
        CommonParamDefinition(
            default=10,
            description="允许报告的最小体素簇大小，过滤噪声体素。",
        ),
    ),
    (
        "svm_params",
        CommonParamDefinition(
            default={"random_state": 42},
            description="Searchlight分类器的核心SVM配置，默认仅固定随机种子以确保可重复。",
        ),
    ),
    (
        "svm_param_grid",
        CommonParamDefinition(
            default={"C": [0.1, 1.0, 10.0], "kernel": ["linear"]},
            description="可选的SVM超参数网格，用于在调试或细化模型时进行交叉验证调优。",
        ),
    ),
    (
        "cv_folds",
        CommonParamDefinition(
            default=5,
            description="交叉验证折数，控制Searchlight在每个球体上的训练-验证划分。",
        ),
    ),
    (
        "cv_random_state",
        CommonParamDefinition(
            default=42,
            description="交叉验证拆分的随机种子，保证折叠划分的一致性。",
        ),
    ),
    (
        "n_permutations",
        CommonParamDefinition(
            default=500,
            description="组水平统计的置换次数，用于估计全局零分布。",
        ),
    ),
    (
        "permutation_random_state",
        CommonParamDefinition(
            default=42,
            description="置换测试使用的随机种子，以获得可重复的chance估计。",
        ),
    ),
    (
        "within_subject_permutations",
        CommonParamDefinition(
            default=100,
            description="单被试机会水平估计的置换次数。可根据探索性分析需求进行调整。",
        ),
    ),
    (
        "max_exact_sign_flips_subjects",
        CommonParamDefinition(
            default=12,
            description="当被试数量不超过该阈值时，组水平统计会尝试枚举全部符号翻转组合。",
        ),
    ),
    (
        "enable_two_sided_group_tests",
        CommonParamDefinition(
            default=True,
            description="是否在组水平分析中执行双侧检验。",
        ),
    ),
    (
        "permutation_strategy",
        CommonParamDefinition(
            default="analytic",
            description="机会水平估计策略，可选analytic/permutation/montecarlo。",
        ),
    ),
    (
        "enable_mask_cache",
        CommonParamDefinition(
            default=True,
            description="是否缓存Searchlight球体掩模以复用邻域结构。",
        ),
    ),
    (
        "monte_carlo_null_samples",
        CommonParamDefinition(
            default=500,
            description="当使用蒙特卡洛策略时抽样的零分布样本数量。",
        ),
    ),
    (
        "enable_beta_cache",
        CommonParamDefinition(
            default=True,
            description="是否缓存拼接后的beta图像，以减少重复磁盘读取。",
        ),
    ),
    (
        "alpha_level",
        CommonParamDefinition(
            default=0.05,
            description="显著性阈值，用于推断统计显著性。",
        ),
    ),
    (
        "n_jobs",
        CommonParamDefinition(
            default_factory=lambda _: max(1, min(4, cpu_count() - 1)),
            description="Searchlight分析并行处理使用的工作进程数量。",
        ),
    ),
    (
        "use_parallel",
        CommonParamDefinition(
            default=True,
            description="是否启用并行处理以加速Searchlight计算。",
        ),
    ),
])


def get_common_mvpa_analysis_params() -> OrderedDict[str, CommonParamDefinition]:
    """获取常用MVPA分析参数定义的拷贝。"""

    return OrderedDict(COMMON_MVPA_ANALYSIS_PARAMS)


def _resolve_common_param_default(config: "MVPAConfig", meta: Any) -> Any:
    """解析常用参数的默认值，支持default和default_factory。"""

    if isinstance(meta, CommonParamDefinition):
        return meta.resolve(config)

    if isinstance(meta, dict):
        if "default_factory" in meta:
            return meta["default_factory"](config)
        return meta.get("default")

    return None


def apply_common_mvpa_analysis_params(config: "MVPAConfig", overrides: Optional[Dict[str, Any]] = None,
                                      log_messages: bool = False) -> OrderedDict[str, Dict[str, Any]]:
    """为配置对象设置常用的MVPA分析参数，并附带文本描述。"""

    overrides = overrides or {}
    applied = OrderedDict()
    params = get_common_mvpa_analysis_params()

    if not hasattr(config, "param_descriptions"):
        config.param_descriptions = {}

    for name, meta in params.items():
        value = overrides.get(name, _resolve_common_param_default(config, meta))
        setattr(config, name, value)

        description = getattr(meta, "description", None)
        if description is None and isinstance(meta, dict):
            description = meta.get("description")
        config.param_descriptions[name] = description
        applied[name] = {"value": value, "description": description}

        if log_messages:
            log(f"参数 {name}={value}: {description}", config)

    return applied


def describe_common_mvpa_analysis_params(applied_params: OrderedDict[str, Dict[str, Any]]) -> list[str]:
    """将常用参数说明格式化为可读文本列表。"""

    if not applied_params:
        return []

    lines = ["常用MVPA分析参数设置:"]
    for name, meta in applied_params.items():
        lines.append(f"  - {name}: {meta['value']} -> {meta['description']}")

    return lines


# ============================================================================
# 受控缓存实现
# ============================================================================


class BoundedLRUCache(OrderedDict):
    """提供带上限的LRU缓存，用于控制内存增长。"""

    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max(1, int(max_items))

    def configure(self, max_items: int) -> None:
        """动态调整缓存容量。"""

        self.max_items = max(1, int(max_items))
        self._trim()

    def _trim(self) -> None:
        while len(self) > self.max_items:
            self.popitem(last=False)

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.move_to_end(key)
        self._trim()


# ============================================================================
# 配置管理模块 - 添加并行处理参数
# ============================================================================

class MVPAConfig:
    """MVPA Searchlight分析配置类 - 并行优化版本"""

    def __init__(self, overrides=None):
        overrides = overrides or {}

        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 4)]
        self.runs = [3, 4]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.mask_dir = Path(r"../../../data/masks")  # 使用通用mask目录
        self.results_dir = Path(r"../../../learn_mvpa/searchlight_mvpa")

        # 常用Searchlight参数：统一设置并附带描述
        self.param_descriptions = {}
        self.common_param_values = apply_common_mvpa_analysis_params(
            self,
            overrides=overrides,
            log_messages=False
        )

        # 缓存容量控制
        self.mask_cache_max_items = overrides.get('mask_cache_max_items', 8)
        self.beta_cache_max_items = overrides.get('beta_cache_max_items', 4)
        configure_cache_limits(self)

        # 并行处理参数（针对被试和批量任务）

        # 新增并行处理参数
        self.parallel_subjects = True  # 启用被试并行处理
        self.max_workers = 3  # 最大并行进程数
        self.chunk_size = 1  # 每个进程处理1个被试
        self.memory_aware_parallel = True  # 内存感知的并行处理
        self.min_memory_per_subject_gb = 24  # 每个被试预估需要的内存(GB)
        self.optimize_memory = True  # 内存优化
        self.max_chance_permutations = 5  # 限制机会水平置换次数
        self.enable_batch_processing = True  # 启用批量处理
        self.disable_chance_maps = False  # 可选：完全禁用机会水平地图计算

        # 显著性分析阈值
        self.classification_significance_threshold = overrides.get(
            'classification_significance_threshold',
            0.5,
        )

        # 内存优化参数
        self.memory_cache = 'nilearn_cache'
        self.memory_level = 1

        # 添加Searchlight特定配置：
        self.enable_memory_monitoring = True
        self.min_trials_per_condition = 5  # 每个条件最少trial数

        # 验证参数
        self.debug_mode = False  # 启用调试模式
        self.debug_subjects = [f"sub-{i:02d}" for i in range(1, 3)]  # 只对前2个被试进行详细调试
        self.n_permutations_debug = 100  # 调试用的置换测试次数

        # 对比条件
        self.contrasts = [
            ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间
        ]

        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.results_dir / "accuracy_maps").mkdir(exist_ok=True)
        (self.results_dir / "statistical_maps").mkdir(exist_ok=True)
        (self.results_dir / "subject_results").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "backups").mkdir(exist_ok=True)

        # 日志设置
        self.log_level = logging.INFO
        self.log_file = self.results_dir / "logs" / "analysis.log"

        # 组层面空间配准与多重比较设置
        group_ref = overrides.get('group_reference_image')
        self.group_reference_image = Path(group_ref) if group_ref else None
        self.group_interpolation = overrides.get('group_interpolation', 'continuous')
        self.primary_correction = overrides.get('primary_correction', 'fdr')
        chinese_font_path = overrides.get('chinese_font_path')
        self.chinese_font_path = Path(chinese_font_path) if chinese_font_path else None

        # 设置日志
        self._setup_logging()
        self.log_common_parameters()

    def log_common_parameters(self):
        """输出常用参数的说明，方便跟踪配置。"""

        for line in describe_common_mvpa_analysis_params(self.common_param_values):
            log(line, self)

    def _setup_logging(self):
        """设置日志配置"""
        file_handler, console_handler = _create_logging_handlers(self.log_file)
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler, console_handler],
            force=True,
        )

    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# ============================================================================
# 工具函数模块 - 添加并行处理支持
# ============================================================================

_CONSOLE_STREAM = None


def _get_console_stream():
    """返回UTF-8编码的输出流，确保控制台日志不会出现乱码。"""

    global _CONSOLE_STREAM

    if _CONSOLE_STREAM is not None:
        return _CONSOLE_STREAM

    stream = sys.stdout
    target_encoding = "utf-8"

    # 尝试直接重配置现有stdout
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=target_encoding, errors="replace")
            _CONSOLE_STREAM = stream
            return _CONSOLE_STREAM
    except Exception:
        pass

    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            stream = io.TextIOWrapper(
                buffer,
                encoding=target_encoding,
                errors="replace",
                line_buffering=True,
            )
        except Exception:
            stream = sys.stdout

    _CONSOLE_STREAM = stream

    if stream is not sys.stdout:
        sys.stdout = stream

    return _CONSOLE_STREAM


def _create_logging_handlers(log_file):
    """统一构建日志处理器，文件和控制台均使用UTF-8编码。"""

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler(_get_console_stream())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    return file_handler, console_handler


def log(message, config):
    """记录日志信息"""
    logging.info(message)


def log_safe(message, config):
    """安全的日志记录，同时保留中文等非ASCII字符"""
    # 将消息转换为字符串并替换掉换行符，避免打乱日志格式
    message_str = str(message).replace('\r', ' ').replace('\n', ' ')

    # 过滤掉不可打印的控制字符，其余字符（包括中文、emoji）保留
    safe_chars = []
    for ch in message_str:
        if ch.isprintable() or ch in '\t':
            safe_chars.append(ch)
        else:
            safe_chars.append(' ')

    safe_message = ''.join(safe_chars)
    logging.info(safe_message)


def memory_cleanup():
    """内存清理"""
    gc.collect()


def as_data_array(img):
    """将Niimg或数组安全地转换为numpy数组."""
    if hasattr(img, "get_fdata"):
        return img.get_fdata()
    return np.asarray(img)


def get_group_reference_image(config: "MVPAConfig", fallback_img):
    """获取组层面的参考图像，用于重采样对齐。"""

    reference_path = getattr(config, 'group_reference_image', None)
    if reference_path:
        try:
            reference_img = image.load_img(str(reference_path))
            log(f"使用自定义组参考图像: {reference_path}", config)
            return reference_img
        except Exception as exc:
            log(f"加载组参考图像失败 {reference_path}: {exc}，改用首个被试图像", config)

    return fallback_img


def resample_image_to_reference(img, reference_img, interpolation="continuous"):
    """将输入图像重采样到参考图像空间。"""

    if img.shape == reference_img.shape and np.allclose(img.affine, reference_img.affine):
        return img

    return image.resample_to_img(img, reference_img, interpolation=interpolation)


def compute_fdr_mask(p_map_data, alpha):
    """使用Benjamini-Hochberg程序计算FDR校正显著性掩模。"""

    flat_p = np.asarray(p_map_data, dtype=float).ravel()
    finite_mask = np.isfinite(flat_p)
    if not finite_mask.any():
        return np.zeros_like(p_map_data, dtype=bool), np.nan

    pvals = flat_p[finite_mask]
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = ranked.size
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds

    if not np.any(passed):
        fdr_mask = np.zeros_like(flat_p, dtype=bool)
        fdr_mask[finite_mask] = False
        return fdr_mask.reshape(p_map_data.shape), np.nan

    max_index = np.max(np.where(passed))
    critical_p = float(ranked[max_index])
    selected = pvals <= critical_p

    mask_flat = np.zeros_like(flat_p, dtype=bool)
    mask_flat[finite_mask] = selected
    return mask_flat.reshape(p_map_data.shape), critical_p


def compute_fwe_mask(p_map_data, alpha):
    """使用Bonferroni矫正计算FWE显著性掩模。"""

    flat_p = np.asarray(p_map_data, dtype=float).ravel()
    finite_mask = np.isfinite(flat_p)
    n_tests = int(np.sum(finite_mask))

    if n_tests == 0:
        return np.zeros_like(p_map_data, dtype=bool), np.nan

    threshold = alpha / n_tests
    mask_flat = np.zeros_like(flat_p, dtype=bool)
    mask_flat[finite_mask] = flat_p[finite_mask] <= threshold

    return mask_flat.reshape(p_map_data.shape), float(threshold)


def summarize_clusters(mask, mean_delta_data, t_map_data, p_map_data, affine, voxel_volume, config, contrast_name, correction_label):
    """根据显著性掩模提取聚类信息。"""

    structure = generate_binary_structure(3, 2)
    mask_bool = np.asarray(mask, dtype=bool)

    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=float), []

    labeled_map, n_clusters = label(mask_bool, structure=structure)
    filtered_mask = np.zeros_like(mask_bool, dtype=float)
    cluster_records = []

    min_region_size = int(getattr(config, 'min_region_size', 0))

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled_map == cluster_id
        if not np.any(cluster_mask):
            continue

        if cluster_mask.sum() < min_region_size:
            continue

        filtered_mask[cluster_mask] = 1.0

        cluster_delta = mean_delta_data[cluster_mask]
        cluster_t = t_map_data[cluster_mask]
        cluster_p = p_map_data[cluster_mask]

        cluster_coords = np.argwhere(cluster_mask)
        cluster_t_clean = np.nan_to_num(cluster_t, nan=-np.inf)

        if cluster_coords.size == 0:
            continue

        if np.all(np.isneginf(cluster_t_clean)):
            peak_idx = 0
        else:
            peak_idx = int(np.argmax(cluster_t_clean))

        peak_coord = cluster_coords[peak_idx]
        peak_mm = nib.affines.apply_affine(affine, peak_coord)

        peak_t_value = float(cluster_t[peak_idx]) if np.isfinite(cluster_t[peak_idx]) else float(np.nanmax(cluster_t))
        peak_delta_value = float(np.nanmax(cluster_delta)) if np.any(np.isfinite(cluster_delta)) else float('nan')
        mean_delta_value = float(np.nanmean(cluster_delta)) if np.any(np.isfinite(cluster_delta)) else float('nan')
        peak_p_candidates = cluster_p[np.isfinite(cluster_p)]
        peak_p_value = float(peak_p_candidates.min()) if peak_p_candidates.size > 0 else float('nan')

        cluster_records.append({
            'contrast': contrast_name,
            'correction': correction_label,
            'cluster_id': cluster_id,
            'n_voxels': int(cluster_mask.sum()),
            'cluster_volume_mm3': float(cluster_mask.sum() * voxel_volume),
            'mean_delta_accuracy': mean_delta_value,
            'peak_delta_accuracy': peak_delta_value,
            'peak_t_value': peak_t_value,
            'peak_p_value': peak_p_value,
            'peak_x': float(peak_mm[0]),
            'peak_y': float(peak_mm[1]),
            'peak_z': float(peak_mm[2])
        })

    return filtered_mask, cluster_records


SEARCHLIGHT_MASK_CACHE = BoundedLRUCache(max_items=8)
BETA_IMAGE_CACHE = BoundedLRUCache(max_items=4)


def configure_cache_limits(config: "MVPAConfig") -> None:
    """根据配置动态调整全局缓存容量。"""

    mask_limit = getattr(config, 'mask_cache_max_items', SEARCHLIGHT_MASK_CACHE.max_items)
    beta_limit = getattr(config, 'beta_cache_max_items', BETA_IMAGE_CACHE.max_items)

    SEARCHLIGHT_MASK_CACHE.configure(mask_limit)
    BETA_IMAGE_CACHE.configure(beta_limit)

    if not getattr(config, 'enable_mask_cache', True):
        SEARCHLIGHT_MASK_CACHE.clear()
    if not getattr(config, 'enable_beta_cache', True):
        BETA_IMAGE_CACHE.clear()


def clear_global_caches() -> None:
    """清空全局缓存，避免跨任务污染。"""

    SEARCHLIGHT_MASK_CACHE.clear()
    BETA_IMAGE_CACHE.clear()


def get_cached_beta_image(beta_images, use_cache=True):
    """根据文件列表缓存并返回拼接后的beta 4D图像。"""

    if not beta_images:
        return None

    cache_key = tuple(str(path) for path in beta_images)

    cached_img = BETA_IMAGE_CACHE.get(cache_key) if use_cache else None
    if cached_img is not None:
        return cached_img

    try:
        beta_img = image.concat_imgs(beta_images)
    except Exception as exc:
        logging.debug(f"beta图像拼接失败: {exc}")
        return None

    if use_cache:
        BETA_IMAGE_CACHE[cache_key] = beta_img

    return beta_img


def prepare_beta_data(beta_images, config):
    """为SearchLight准备输入数据，可复用缓存以避免重复IO。"""

    use_cache = getattr(config, 'enable_beta_cache', False)

    if not beta_images:
        return beta_images

    if use_cache:
        beta_img = get_cached_beta_image(beta_images, use_cache=True)
        if beta_img is not None:
            return beta_img
        log("beta图像缓存失败，回退至逐文件加载", config)

    return beta_images


def get_cached_mask_structures(process_mask_path, template_img=None, use_cache=True):
    """根据配置缓存并返回mask及相关结构。"""
    cache_key = str(process_mask_path)

    cached = SEARCHLIGHT_MASK_CACHE.get(cache_key) if use_cache else None
    if cached is not None:
        if template_img is not None:
            cached['reference_img'] = template_img
        return cached

    mask_img = image.load_img(process_mask_path)
    mask_data = as_data_array(mask_img) > 0
    reference_img = template_img if template_img is not None else mask_img

    cached = {
        'mask_img': mask_img,
        'mask_data': mask_data,
        'reference_img': reference_img
    }

    if use_cache:
        SEARCHLIGHT_MASK_CACHE[cache_key] = cached

    return cached


def permute_labels_within_groups(labels, groups, rng):
    """在保持组内结构的情况下随机置换标签"""
    labels = np.asarray(labels)

    if groups is None:
        return rng.permutation(labels)

    groups = np.asarray(groups)
    permuted = labels.copy()
    for group in np.unique(groups):
        mask = groups == group
        permuted[mask] = rng.permutation(labels[mask])

    return permuted


def generate_sign_flip_vectors(n_subjects, config, rng):
    """生成用于符号翻转置换的符号矩阵"""
    if n_subjects == 0:
        return [], 0, False

    exact_limit = getattr(config, 'max_exact_sign_flips_subjects', 0)
    two_sided = getattr(config, 'enable_two_sided_group_tests', True)

    if exact_limit and n_subjects <= exact_limit:
        import itertools

        sign_vectors = []
        for signs in itertools.product([-1, 1], repeat=n_subjects):
            if two_sided:
                sign_vectors.append(np.array(signs, dtype=float))
            else:
                if np.prod(signs) == 1:
                    sign_vectors.append(np.array(signs, dtype=float))

        return sign_vectors, len(sign_vectors), True

    n_permutations = getattr(config, 'n_permutations', 0)
    if n_permutations <= 0:
        return [], 0, False

    sign_vectors = [rng.choice([-1.0, 1.0], size=n_subjects) for _ in range(n_permutations)]
    return sign_vectors, len(sign_vectors), False


# ============================================================================
# 内存监控装饰器
# ============================================================================

def monitor_memory_usage(func):
    """监控内存使用的装饰器"""

    def wrapper(*args, **kwargs):
        try:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            # 记录大内存使用的函数
            if memory_used > 500:  # 如果使用超过500MB
                print(f"⚠️ 函数 {func.__name__} 内存使用: {memory_used:.1f}MB")

            return result
        except psutil.Error:
            return func(*args, **kwargs)

    return wrapper


# ============================================================================
# 并行处理函数
# ============================================================================

def adjust_parallel_config_based_on_memory(config):
    """根据系统内存调整并行配置"""
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)

        # 计算安全的最大并行数
        safe_max_workers = int(available_gb / config.min_memory_per_subject_gb)
        safe_max_workers = max(1, min(safe_max_workers, mp.cpu_count() - 1))

        if safe_max_workers < config.max_workers:
            log(f"内存限制，并行数从 {config.max_workers} 调整为 {safe_max_workers}", config)
            config.max_workers = safe_max_workers

        # 如果内存紧张，减少置换次数
        if available_gb < 8:
            config.within_subject_permutations = min(5, getattr(config, 'within_subject_permutations', 10))
            log(f"内存紧张，减少机会水平置换次数为: {config.within_subject_permutations}", config)

    except Exception as e:
        log(f"内存检测失败: {str(e)}，使用默认并行配置", config)

    return config


def config_to_dict(config):
    """将配置对象转换为可序列化的字典"""
    config_dict = {}

    # 只复制必要的配置参数
    attributes = [
        'subjects', 'runs', 'lss_root', 'mask_dir', 'results_dir',
        'searchlight_radius', 'process_mask', 'min_region_size',
        'svm_params', 'svm_param_grid', 'cv_folds', 'cv_random_state',
        'n_permutations', 'permutation_random_state', 'within_subject_permutations',
        'max_exact_sign_flips_subjects', 'enable_two_sided_group_tests', 'permutation_strategy',
        'enable_mask_cache', 'monte_carlo_null_samples', 'enable_beta_cache',
        'alpha_level', 'n_jobs', 'use_parallel', 'contrasts', 'max_workers',
        'parallel_subjects', 'memory_aware_parallel', 'min_memory_per_subject_gb',
        'optimize_memory', 'max_chance_permutations', 'enable_batch_processing',
        'disable_chance_maps', 'memory_cache', 'memory_level', 'enable_memory_monitoring',
        'min_trials_per_condition', 'debug_mode', 'debug_subjects', 'n_permutations_debug',
        'param_descriptions', 'common_param_values'
    ]

    for attr in attributes:
        if hasattr(config, attr):
            value = getattr(config, attr)
            # 转换Path对象为字符串
            if isinstance(value, Path):
                config_dict[attr] = str(value)
            else:
                config_dict[attr] = value

    return config_dict


def dict_to_config(config_dict):
    """从字典重建配置对象"""

    overrides = {
        key: config_dict[key]
        for key in COMMON_MVPA_ANALYSIS_PARAMS.keys()
        if key in config_dict
    }
    config = MVPAConfig(overrides=overrides)

    for key, value in config_dict.items():
        if key in COMMON_MVPA_ANALYSIS_PARAMS:
            continue

        # 将字符串路径转换回Path对象
        if key in ['lss_root', 'mask_dir', 'results_dir'] and isinstance(value, str):
            setattr(config, key, Path(value))
        else:
            setattr(config, key, value)

    configure_cache_limits(config)

    return config


# ============================================================================
# 验证和诊断模块 (保持原有功能)
# ============================================================================

def validate_config(config):
    """验证配置参数"""
    log("验证配置参数...", config)
    issues = []

    # 检查路径存在性
    if not config.lss_root.exists():
        issues.append(f"LSS根目录不存在: {config.lss_root}")

    if not config.mask_dir.exists():
        issues.append(f"Mask目录不存在: {config.mask_dir}")

    # 添加Searchlight特定检查
    if config.searchlight_radius < 1:
        issues.append("Searchlight半径必须大于0")

    if config.min_trials_per_condition < 3:
        issues.append("每个条件最少trial数建议至少为3")

    # 检查处理mask文件
    process_mask_path = config.mask_dir / config.process_mask
    if not process_mask_path.exists():
        issues.append(f"处理mask文件不存在: {process_mask_path}")

    # 检查被试数据可用性
    available_subjects = []
    for subject in config.subjects:
        subject_dir = config.lss_root / subject
        if subject_dir.exists():
            available_subjects.append(subject)

    if len(available_subjects) < 3:
        issues.append(f"可用被试数量不足: {len(available_subjects)}")

    config.available_subjects = available_subjects

    if issues:
        log("配置验证发现问题:", config)
        for issue in issues:
            log(f"  ⚠️ {issue}", config)
        return False

    log(f"配置验证通过: 找到{len(available_subjects)}个可用被试", config)
    return True


def check_data_leakage(beta_images, trial_info, config, subject_id):
    """检查数据泄露的潜在问题"""
    log(f"检查数据泄露问题 - {subject_id}:", config)
    issues = []

    # 1. 检查是否有重复的beta文件
    unique_betas = set(beta_images)
    if len(unique_betas) != len(beta_images):
        issues.append(f"发现重复的beta文件: 唯一文件{len(unique_betas)}个, 总引用{len(beta_images)}次")

    # 2. 检查trial索引是否正确
    condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'
    for condition in ['yy', 'kj']:  # 根据您的条件调整
        cond_trials = trial_info[trial_info[condition_col] == condition]
        log(f"  条件 '{condition}': {len(cond_trials)}个trial", config)

        # 检查索引范围
        if 'combined_trial_index' in cond_trials.columns:
            min_idx = cond_trials['combined_trial_index'].min()
            max_idx = cond_trials['combined_trial_index'].max()
            log(f"    索引范围: {min_idx} - {max_idx} (总beta文件: {len(beta_images)})", config)

            # 检查是否有索引越界
            out_of_bound = cond_trials[
                (cond_trials['combined_trial_index'] < 0) |
                (cond_trials['combined_trial_index'] >= len(beta_images))
                ]
            if len(out_of_bound) > 0:
                issues.append(f"发现{len(out_of_bound)}个越界索引!")

    # 3. 检查条件平衡
    if condition_col in trial_info.columns:
        condition_counts = trial_info[condition_col].value_counts()
        log(f"  条件分布: {dict(condition_counts)}", config)
        if len(condition_counts) < 2:
            issues.append("只有一个条件，无法进行分类")
        elif min(condition_counts.values) < 5:
            issues.append(f"条件样本不均衡，最小条件只有{min(condition_counts.values)}个样本")

    if issues:
        log_safe(f"  ⚠️ 数据泄露检查发现问题:", config)
        for issue in issues:
            log(f"    {issue}", config)
        return False

    log_safe(f"  ✅ 数据泄露检查通过", config)
    return True


def validate_searchlight_parameters(config, process_mask_path):
    """验证Searchlight参数合理性"""
    import nibabel as nib
    from nilearn import masking

    log("验证Searchlight参数...", config)

    try:
        # 检查mask体素数
        mask_img = nib.load(process_mask_path)
        mask_data = as_data_array(mask_img)
        n_voxels = np.sum(mask_data > 0)

        # 估算球体体积
        estimated_sphere_size = (4 / 3) * np.pi * (config.searchlight_radius ** 3)
        total_spheres = n_voxels  # 每个体素一个球体

        log(f"Mask信息:", config)
        log(f"  - 体素数: {n_voxels}", config)
        log(f"  - 球体半径: {config.searchlight_radius} 体素", config)
        log(f"  - 估计球体大小: ~{estimated_sphere_size:.0f} 体素", config)
        log(f"  - 总球体数: ~{total_spheres}", config)

        # 警告检查
        if estimated_sphere_size < 10:
            log("警告: Searchlight半径可能太小，考虑增加半径", config)
        if estimated_sphere_size > 500:
            log("警告: Searchlight半径可能太大，计算时间会很长", config)
        if n_voxels > 100000:
            log("警告: Mask体素过多，计算时间可能很长", config)

        return True

    except Exception as e:
        log(f"Searchlight参数验证失败: {str(e)}", config)
        return False


# ============================================================================
# 数据加载模块 (保持原有功能)
# ============================================================================

def load_process_mask(config):
    """加载处理mask文件"""
    log("加载处理mask", config)

    process_mask_path = config.mask_dir / config.process_mask

    if not process_mask_path.exists():
        raise FileNotFoundError(f"处理mask文件不存在: {process_mask_path}")

    log(f"加载处理mask: {config.process_mask}", config)
    return str(process_mask_path)


def enhanced_error_handling(operation_name):
    """增强的错误处理装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                print(f"{operation_name} - 文件未找到错误: {str(e)}")
                return None
            except MemoryError as e:
                print(f"{operation_name} - 内存错误: {str(e)}")
                memory_cleanup()
                return None
            except ValueError as e:
                print(f"{operation_name} - 数值错误: {str(e)}")
                return None
            except Exception as e:
                print(f"{operation_name} - 未知错误: {type(e).__name__}: {str(e)}")
                return None

        return wrapper

    return decorator


@enhanced_error_handling("LSS trial数据加载")
def load_lss_trial_data(subject, run, config):
    """加载LSS分析的trial数据和beta图像"""
    lss_dir = config.lss_root / subject / f"run-{run}_LSS"

    if not lss_dir.exists():
        log(f"LSS目录不存在: {lss_dir}", config)
        return None, None

    # 加载trial信息
    trial_map_path = lss_dir / "trial_info.csv"
    if not trial_map_path.exists():
        log(f"trial_info.csv不存在: {trial_map_path}", config)
        return None, None

    trial_info = pd.read_csv(trial_map_path)

    # 加载所有beta图像
    beta_images = []
    valid_trials = []
    missing_files = []

    for i, (_, trial) in enumerate(trial_info.iterrows()):
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        beta_filename = f"beta_trial_{trial['trial_index']:03d}.nii.gz"

        if beta_path.exists():
            beta_images.append(str(beta_path))
            # 添加run信息到trial数据中
            trial_with_run = trial.copy()
            trial_with_run['run'] = run
            trial_with_run['global_trial_index'] = f"run{run}_trial{trial['trial_index']}"
            valid_trials.append(trial_with_run)
        else:
            log(f"Beta图像缺失: {beta_path}", config)
            missing_files.append(beta_filename)

    # 报告缺失文件
    if missing_files:
        log(f"  run {run}: 发现 {len(missing_files)} 个缺失的beta文件", config)

    log(f"  run {run}: 加载了 {len(beta_images)} 个trial", config)

    if len(beta_images) == 0:
        log(f"没有找到有效的beta图像: {subject} run-{run}", config)
        return None, None

    valid_trials_df = pd.DataFrame(valid_trials)
    log(f"加载 {subject} run-{run}: {len(beta_images)}个trial", config)

    return beta_images, valid_trials_df


@enhanced_error_handling("多run LSS数据加载")
def load_multi_run_lss_data(subject, runs, config):
    """加载多个run的LSS数据"""
    log(f"开始加载被试 {subject} 的多run LSS数据, runs: {runs}", config)

    all_trial_info = []
    all_beta_images = []
    global_trial_index = 0

    for run in runs:
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        if beta_images is not None and trial_info is not None:
            # 为每个trial添加run信息和全局索引
            trial_info['run'] = run
            trial_info['global_trial_index'] = range(global_trial_index,
                                                     global_trial_index + len(trial_info))
            trial_info['combined_trial_index'] = trial_info['global_trial_index']

            # 记录索引映射关系
            log(f"  run {run} 索引映射: 原始trial_index {trial_info['trial_index'].min()}-{trial_info['trial_index'].max()}, "
                f"combined_trial_index {global_trial_index}-{global_trial_index + len(trial_info) - 1}", config)

            all_trial_info.append(trial_info)
            all_beta_images.extend(beta_images)
            global_trial_index += len(trial_info)
        else:
            log(f"  警告: run {run} 数据加载失败", config)

    if not all_trial_info:
        log(f"错误: 被试 {subject} 没有成功加载任何run的数据", config)
        return None, None

    # 合并所有run的数据
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)

    log(f"多run数据合并完成: 总共 {len(combined_trial_info)} 个trial, {len(all_beta_images)} 个beta文件", config)

    # 验证数据一致性
    if len(combined_trial_info) != len(all_beta_images):
        log(f"警告: trial数量({len(combined_trial_info)})与beta文件数量({len(all_beta_images)})不匹配!", config)

    # 打印每个条件的trial数量统计
    if 'original_condition' in combined_trial_info.columns:
        condition_counts = combined_trial_info['original_condition'].value_counts()
        log(f"  条件统计: {dict(condition_counts)}", config)

    return all_beta_images, combined_trial_info


# ============================================================================
# Searchlight核心功能 - 优化版本
# ============================================================================

def build_classification_pipeline(config, n_features):
    """构建Searchlight分类pipeline"""
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),  # 处理可能的缺失值
        ('variance_filter', VarianceThreshold(threshold=0.0)),  # 只移除零方差特征
        ('scaler', StandardScaler()),  # 标准化
        ('classifier', SVC(**config.svm_params))  # 分类器
    ]
    return Pipeline(steps)


def check_system_resources(config):
    """检查系统资源"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)

        log(f"系统资源:", config)
        log(f"  - 可用内存: {available_gb:.1f} GB", config)
        log(f"  - 使用并行数: {config.n_jobs}", config)

        # 根据内存调整并行数
        if available_gb < 4:
            original_jobs = config.n_jobs
            config.n_jobs = 1
            log(f"  - 内存不足，并行数从 {original_jobs} 调整为 1", config)
        elif available_gb < 8:
            original_jobs = config.n_jobs
            config.n_jobs = min(2, config.n_jobs)
            log(f"  - 内存有限，并行数从 {original_jobs} 调整为 {config.n_jobs}", config)

    except ImportError:
        log("无法检测系统资源，请安装psutil: pip install psutil", config)


def _fit_searchlight(beta_images, labels, process_mask_path, config, groups=None, verbose=1):
    """内部函数，用于拟合SearchLight估计器 - 增强错误处理"""
    try:
        classifier = SVC(**config.svm_params)

        if beta_images is None:
            log("错误: 未提供有效的beta数据", config)
            return None

        if groups is not None and len(np.unique(groups)) > 1:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(
                n_splits=min(config.cv_folds, len(labels)),
                random_state=config.cv_random_state,
                shuffle=True
            )

        searchlight = SearchLight(
            mask_img=process_mask_path,
            radius=config.searchlight_radius,
            estimator=classifier,
            cv=cv,
            scoring='accuracy',
            n_jobs=config.n_jobs,
            verbose=verbose
        )

        searchlight.fit(beta_images, labels, groups=groups)

        # 验证结果
        if not hasattr(searchlight, 'scores_'):
            log("错误: SearchLight拟合后没有scores_属性", config)
            return None

        # 确保scores_是有效的
        scores_data = as_data_array(searchlight.scores_)
        if scores_data is None or scores_data.size == 0:
            log("错误: SearchLight scores_数据为空", config)
            return None

        setattr(searchlight, '_cached_beta_source', beta_images)

        return searchlight

    except Exception as e:
        log(f"SearchLight拟合失败: {str(e)}", config)
        return None


@monitor_memory_usage
@enhanced_error_handling("Searchlight分类分析")
def run_searchlight_classification(beta_images, labels, process_mask_path, config, groups=None):
    """运行searchlight分类分析 - 改进版本"""
    try:
        check_system_resources(config)

        log(
            f"开始Searchlight分析: 半径={config.searchlight_radius}体素, 样本数={len(labels)}",
            config
        )
        start_time = time.time()
        beta_source = prepare_beta_data(beta_images, config)

        searchlight = _fit_searchlight(
            beta_source,
            labels,
            process_mask_path,
            config,
            groups=groups,
            verbose=1
        )

        duration = (time.time() - start_time) / 60
        log(f"Searchlight完成: 耗时{duration:.1f}分钟", config)

        return searchlight

    except Exception as e:
        log(f"Searchlight分析失败: {str(e)}", config)
        return None


def _estimate_chance_via_analytic(mask_data, reference_img, labels, config, strategy):
    """使用解析或快速蒙特卡洛方法估计机会水平。"""

    try:
        n_samples = max(len(labels), 1)
        rng = np.random.default_rng(config.permutation_random_state)

        if strategy == 'montecarlo':
            draws = getattr(config, 'monte_carlo_null_samples', 500)
            draws = max(int(draws), 1)
            # 对整体准确率进行快速蒙特卡洛抽样
            simulated = rng.binomial(n_samples, 0.5, size=draws) / float(n_samples)
            mean_val = float(np.mean(simulated))
            std_val = float(np.std(simulated, ddof=1)) if draws > 1 else 0.0
        else:
            # 解析估计: 假设二分类准确率服从0.5的二项分布
            mean_val = 0.5
            std_val = float(np.sqrt(0.25 / float(n_samples)))

        full_mean_data = np.zeros(mask_data.shape, dtype=float)
        full_std_data = np.zeros(mask_data.shape, dtype=float)

        full_mean_data[mask_data] = mean_val
        full_std_data[mask_data] = std_val

        chance_img = image.new_img_like(reference_img, full_mean_data)
        chance_std_img = image.new_img_like(reference_img, full_std_data)

        return chance_img, chance_std_img, {
            'n_subject_permutations': 0,
            'chance_mean_data': full_mean_data,
            'chance_std_data': full_std_data,
            'strategy': strategy,
            'effective_samples': int(n_samples)
        }

    except Exception as exc:
        log(f"解析机会水平估计失败: {exc}", config)
        return None, None, None


def estimate_subject_chance_map(beta_images, labels, process_mask_path, config, template_img=None,
                                groups=None, searchlight=None):
    """通过标签置换或快速近似估计单个被试的机会水平地图"""

    try:
        strategy = getattr(config, 'permutation_strategy', 'permutation')

        if searchlight is not None and hasattr(searchlight, 'mask_img_'):
            template_img = template_img or getattr(searchlight, 'scores_', None)
            mask_img_source = searchlight.mask_img_
        else:
            mask_img_source = None

        cached = get_cached_mask_structures(
            process_mask_path,
            template_img=template_img if template_img is not None else mask_img_source,
            use_cache=getattr(config, 'enable_mask_cache', True)
        )

        mask_data = cached['mask_data']
        reference_img = cached['reference_img']

        beta_source = None
        if searchlight is not None and hasattr(searchlight, '_cached_beta_source'):
            beta_source = getattr(searchlight, '_cached_beta_source')

        if beta_source is None:
            beta_source = prepare_beta_data(beta_images, config)

        if beta_source is None:
            log("无法准备用于置换的beta数据，跳过机会水平估计", config)
            return None, None, None

        if strategy in ('analytic', 'montecarlo'):
            log(f"使用{strategy}策略估计机会水平（跳过置换）", config)
            return _estimate_chance_via_analytic(mask_data, reference_img, labels, config, strategy)

        n_permutations = getattr(config, 'within_subject_permutations', 10)
        max_cap = getattr(config, 'max_chance_permutations', 0)
        if max_cap and max_cap > 0:
            n_permutations = min(n_permutations, max_cap)

        if n_permutations <= 0:
            log("置换次数为0，跳过机会水平估计", config)
            return None, None, None

        rng = np.random.default_rng(config.permutation_random_state)

        mean_accumulator = None
        m2_accumulator = None
        actual_permutations = 0

        for perm_index in range(n_permutations):
            permuted_labels = permute_labels_within_groups(labels, groups, rng)

            try:
                perm_searchlight = _fit_searchlight(
                    beta_source,
                    permuted_labels,
                    process_mask_path,
                    config,
                    groups=groups,
                    verbose=0
                )

                if (perm_searchlight is not None and
                        hasattr(perm_searchlight, 'scores_') and
                        hasattr(perm_searchlight.scores_, 'get_fdata')):

                    perm_scores = as_data_array(perm_searchlight.scores_)

                    if mask_data.shape == perm_scores.shape:
                        masked_scores = perm_scores[mask_data]
                        masked_scores = masked_scores.astype(float, copy=False)

                        if mean_accumulator is None:
                            mean_accumulator = np.zeros_like(masked_scores, dtype=float)
                            m2_accumulator = np.zeros_like(masked_scores, dtype=float)

                        actual_permutations += 1
                        delta = masked_scores - mean_accumulator
                        mean_accumulator += delta / actual_permutations
                        m2_accumulator += delta * (masked_scores - mean_accumulator)
                    else:
                        log("警告: mask形状不匹配，跳过当前置换结果", config)

            except Exception as exc:
                log(f"置换 {perm_index + 1}/{n_permutations} 失败: {exc}", config)
                continue

        if actual_permutations == 0:
            log("警告: 未能生成任何有效的置换结果", config)
            return None, None, None

        if mean_accumulator is None:
            log("未能计算任何有效的置换统计，跳过机会水平图", config)
            return None, None, None

        mean_data = mean_accumulator

        if actual_permutations > 1 and m2_accumulator is not None:
            variance = np.maximum(m2_accumulator / (actual_permutations - 1), 0)
            std_data = np.sqrt(variance)
        else:
            std_data = np.zeros_like(mean_data)

        full_mean_data = np.zeros(mask_data.shape)
        full_std_data = np.zeros(mask_data.shape)

        full_mean_data[mask_data] = mean_data
        full_std_data[mask_data] = std_data

        try:
            chance_img = image.new_img_like(reference_img, full_mean_data)
            chance_std_img = image.new_img_like(reference_img, full_std_data)
        except Exception as e:
            log(f"创建机会水平图像失败: {str(e)}", config)
            return None, None, None

        return chance_img, chance_std_img, {
            'n_subject_permutations': actual_permutations,
            'chance_mean_data': full_mean_data,
            'chance_std_data': full_std_data,
            'strategy': 'permutation'
        }

    except Exception as e:
        log(f"机会水平估计失败: {str(e)}", config)
        return None, None, None


def analyze_searchlight_results_optimized(searchlight, config, subject_id, contrast_name,
                                          accuracy_data=None, mask_data=None,
                                          chance_img=None, chance_info=None):
    """优化版的结果分析 - 修复数据验证"""

    try:
        # 使用预先计算的数据或重新计算
        if accuracy_data is None:
            if not hasattr(searchlight.scores_, 'get_fdata'):
                log(f"错误: searchlight.scores_ 不是有效的Niimg对象", config)
                return None
            accuracy_data = as_data_array(searchlight.scores_)

        if mask_data is None:
            if not hasattr(searchlight.mask_img_, 'get_fdata'):
                log(f"错误: searchlight.mask_img_ 不是有效的Niimg对象", config)
                return None
            mask_data = as_data_array(searchlight.mask_img_) > 0

        # 验证数据形状匹配
        if accuracy_data.shape != mask_data.shape:
            log(f"错误: 准确率数据形状{accuracy_data.shape}与mask形状{mask_data.shape}不匹配", config)
            return None

        masked_accuracies = accuracy_data[mask_data]

        if masked_accuracies.size == 0:
            log(f"警告: {subject_id} {contrast_name} 无有效体素", config)
            return None

        # 计算基本统计量
        mean_accuracy = float(np.mean(masked_accuracies))
        max_accuracy = float(np.max(masked_accuracies))
        min_accuracy = float(np.min(masked_accuracies))
        accuracy_std = float(np.std(masked_accuracies))

        results = {
            'subject': subject_id,
            'contrast': contrast_name,
            'accuracy': mean_accuracy,
            'mean_accuracy': mean_accuracy,
            'max_accuracy': max_accuracy,
            'min_accuracy': min_accuracy,
            'accuracy_std': accuracy_std,
            'above_chance_ratio': float(np.mean(masked_accuracies > 0.5)),
            'n_voxels_above_55': int(np.sum(masked_accuracies > 0.55)),
            'n_voxels_above_60': int(np.sum(masked_accuracies > 0.6)),
            'n_voxels_total': len(masked_accuracies)
        }

        # 机会水平相关计算
        if chance_img is not None:
            try:
                chance_data = as_data_array(chance_img)
                if chance_data.shape == mask_data.shape:
                    masked_chance = chance_data[mask_data]

                    chance_metrics = {
                        'mean_chance_accuracy': float(np.mean(masked_chance)),
                        'chance_accuracy_std': float(np.std(masked_chance)),
                        'chance_accuracy_min': float(np.min(masked_chance)),
                        'chance_accuracy_max': float(np.max(masked_chance)),
                        'chance_above_half_ratio': float(np.mean(masked_chance > 0.5))
                    }

                    if chance_info and 'n_subject_permutations' in chance_info:
                        chance_metrics['n_subject_permutations'] = chance_info['n_subject_permutations']

                    results.update(chance_metrics)

                    # 差值计算
                    delta_data = accuracy_data - chance_data
                    masked_delta = delta_data[mask_data]

                    delta_metrics = {
                        'mean_delta_accuracy': float(np.mean(masked_delta)),
                        'delta_accuracy_std': float(np.std(masked_delta, ddof=1)) if masked_delta.size > 1 else 0.0,
                        'delta_accuracy_min': float(np.min(masked_delta)),
                        'delta_accuracy_max': float(np.max(masked_delta)),
                        'delta_above_zero_ratio': float(np.mean(masked_delta > 0))
                    }

                    if chance_info and 'chance_std_data' in chance_info:
                        chance_std_masked = chance_info['chance_std_data'][mask_data]
                        delta_metrics['mean_chance_std'] = float(np.mean(chance_std_masked))

                    results.update(delta_metrics)

            except Exception as e:
                log(f"机会水平计算失败: {str(e)}", config)

        log(f"  {contrast_name} 结果:", config)
        log(f"    - 平均准确率: {results['mean_accuracy']:.4f}", config)
        log(f"    - 最高准确率: {results['max_accuracy']:.4f}", config)
        log(f"    - 体素数: {results['n_voxels_total']}", config)

        return results

    except Exception as e:
        log(f"结果分析失败: {str(e)}", config)
        import traceback
        log(f"详细错误: {traceback.format_exc()}", config)
        return None


@enhanced_error_handling("分类数据准备")
def prepare_classification_data(trial_info, beta_images, cond1, cond2, config):
    """准备分类数据"""
    # 确定条件列名
    condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'

    if condition_col not in trial_info.columns:
        log(f"错误: 未找到条件列 '{condition_col}'", config)
        return None, None, None

    # 筛选指定条件的trial
    cond1_trials = trial_info[trial_info[condition_col] == cond1]
    cond2_trials = trial_info[trial_info[condition_col] == cond2]

    log(f"条件 '{cond1}': {len(cond1_trials)} 个trial", config)
    log(f"条件 '{cond2}': {len(cond2_trials)} 个trial", config)

    # 检查是否有足够的trial
    if len(cond1_trials) < 5 or len(cond2_trials) < 5:
        log(f"警告: 条件样本不足 (需要至少5个), {cond1}: {len(cond1_trials)}, {cond2}: {len(cond2_trials)}", config)
        return None, None, None

    # 合并trial信息
    selected_trials = pd.concat([cond1_trials, cond2_trials], ignore_index=True)

    # 获取对应的beta图像
    selected_indices = selected_trials['combined_trial_index'].values

    # 验证索引范围
    if np.any(selected_indices >= len(beta_images)) or np.any(selected_indices < 0):
        log(f"错误: 索引越界，索引范围: {selected_indices.min()}-{selected_indices.max()}, beta文件数: {len(beta_images)}",
            config)
        return None, None, None

    selected_betas = [beta_images[i] for i in selected_indices]

    # 创建标签 (0: cond1, 1: cond2)
    labels = np.concatenate([
        np.zeros(len(cond1_trials), dtype=int),
        np.ones(len(cond2_trials), dtype=int)
    ])

    log(f"准备分类数据完成: {len(selected_betas)} 个trial, 标签分布: {np.bincount(labels)}", config)

    return selected_betas, labels, selected_trials


# ============================================================================
# 批量处理函数
# ============================================================================

def process_searchlight_results_batch(beta_images, labels, process_mask_path, config,
                                      searchlight, subject_id, contrast_name,
                                      groups=None, trial_details=None, cond1=None, cond2=None):
    """批量处理Searchlight结果 - 减少重复计算"""

    # 预先获取必要数据
    accuracy_data = as_data_array(searchlight.scores_)
    mask_data = as_data_array(searchlight.mask_img_) > 0

    # 并行处理机会水平估计和结果分析
    chance_img, chance_std_img, chance_info = estimate_subject_chance_map(
        beta_images, labels, process_mask_path, config,
        template_img=searchlight.scores_, groups=groups, searchlight=searchlight
    )

    # 分析结果
    subject_results = analyze_searchlight_results_optimized(
        searchlight, config, subject_id, contrast_name,
        accuracy_data, mask_data, chance_img, chance_info
    )

    if subject_results and trial_details is not None:
        # 添加trial数量信息
        condition_col = 'original_condition' if 'original_condition' in trial_details.columns else 'condition'
        if condition_col in trial_details.columns:
            subject_results['n_trials_cond1'] = int((trial_details[condition_col] == cond1).sum())
            subject_results['n_trials_cond2'] = int((trial_details[condition_col] == cond2).sum())

    return chance_img, chance_std_img, chance_info, subject_results


def save_maps_batch(accuracy_img, chance_img, chance_std_img, accuracy_data,
                    subject_result_dir, contrast_name, subject_results, config):
    """批量保存地图文件 - 减少IO操作"""

    subject_result_dir.mkdir(parents=True, exist_ok=True)

    # 保存准确率地图
    accuracy_map_path = subject_result_dir / f"{contrast_name}_accuracy_map.nii.gz"
    accuracy_img.to_filename(str(accuracy_map_path))
    subject_results['accuracy_map_path'] = str(accuracy_map_path)

    if chance_img is not None:
        # 保存机会水平地图
        chance_map_path = subject_result_dir / f"{contrast_name}_chance_map.nii.gz"
        chance_img.to_filename(str(chance_map_path))
        subject_results['chance_map_path'] = str(chance_map_path)

        # 保存机会水平标准差地图
        if chance_std_img is not None:
            chance_std_path = subject_result_dir / f"{contrast_name}_chance_std_map.nii.gz"
            chance_std_img.to_filename(str(chance_std_path))
            subject_results['chance_std_map_path'] = str(chance_std_path)

        # 计算并保存差值地图
        chance_data = as_data_array(chance_img)
        delta_data = accuracy_data - chance_data
        delta_img = image.new_img_like(accuracy_img, delta_data)
        delta_map_path = subject_result_dir / f"{contrast_name}_delta_map.nii.gz"
        delta_img.to_filename(str(delta_map_path))
        subject_results['delta_map_path'] = str(delta_map_path)


# ============================================================================
# 并行处理主流程
# ============================================================================

def process_single_subject(subject_id, config, process_mask_path):
    """处理单个被试 - 添加调试信息"""
    subject_results = []

    try:
        log(f"开始处理被试 {subject_id}", config)

        # 加载数据
        beta_images, trial_info = load_multi_run_lss_data(subject_id, config.runs, config)

        if beta_images is None or trial_info is None:
            log(f"被试 {subject_id} 的LSS数据加载失败", config)
            return subject_results

        log(f"成功加载 {len(beta_images)} 个beta图像", config)

        # 对每个对比条件进行分析
        for contrast_name, cond1, cond2 in config.contrasts:
            log(f"  分析对比: {contrast_name} ({cond1} vs {cond2})", config)

            # 准备分类数据
            selected_betas, labels, trial_details = prepare_classification_data(
                trial_info, beta_images, cond1, cond2, config
            )

            if selected_betas is None:
                log(f"跳过 {subject_id} {cond1} vs {cond2}: 数据准备失败", config)
                continue

            log(f"  准备完成: {len(selected_betas)} 个trial, 标签分布: {np.bincount(labels)}", config)

            # 检查样本数量
            if len(selected_betas) < 10:
                log(f"  样本数量不足({len(selected_betas)})，跳过", config)
                continue

            groups = trial_details['run'].fillna('unknown').astype(str).values

            # 运行searchlight分析
            searchlight = run_searchlight_classification(
                selected_betas, labels, process_mask_path, config, groups=groups
            )

            if searchlight is None:
                log(f"  Searchlight分析失败", config)
                continue

            if not hasattr(searchlight, 'scores_'):
                log(f"  Searchlight结果无效，没有scores_属性", config)
                continue

            result = process_searchlight_for_subject(
                searchlight, selected_betas, labels, process_mask_path, config,
                subject_id, contrast_name, groups, trial_details, cond1, cond2
            )

            if result:
                subject_results.append(result)
                log(f"  ✅ 成功完成 {contrast_name} 分析", config)
            else:
                log(f"  ❌ {contrast_name} 分析失败", config)

        log(f"完成被试 {subject_id} 处理，获得 {len(subject_results)} 个结果", config)
        return subject_results

    except Exception as e:
        log(f"被试 {subject_id} 处理失败: {str(e)}", config)
        import traceback
        log(f"详细错误: {traceback.format_exc()}", config)
        return subject_results


def process_searchlight_for_subject(searchlight, beta_images, labels, process_mask_path, config,
                                    subject_id, contrast_name, groups, trial_details, cond1, cond2):
    """处理单个被试的searchlight结果 - 修复版本"""

    try:
        # 检查searchlight结果的有效性
        if not hasattr(searchlight, 'scores_'):
            log(f"错误: searchlight对象没有scores_属性", config)
            return None

        # 确保scores_是有效的Niimg对象
        if not hasattr(searchlight.scores_, 'get_fdata'):
            log(f"警告: searchlight.scores_ 不是Niimg对象，尝试转换", config)
            try:
                # 如果是numpy数组，转换为Niimg
                if isinstance(searchlight.scores_, np.ndarray):
                    # 使用mask图像作为模板
                    reference_img = image.load_img(process_mask_path)
                    searchlight.scores_ = image.new_img_like(reference_img, searchlight.scores_)
                    log(f"已将numpy数组转换为Niimg对象", config)
                else:
                    log(f"错误: searchlight.scores_ 类型无法处理: {type(searchlight.scores_)}", config)
                    return None
            except Exception as e:
                log(f"转换searchlight.scores_失败: {str(e)}", config)
                return None

        # 预先获取数据，避免重复计算
        accuracy_data = as_data_array(searchlight.scores_)

        # 检查数据有效性
        if accuracy_data is None or accuracy_data.size == 0:
            log(f"错误: 准确率数据为空或无效", config)
            return None

        mask_img = searchlight.mask_img_

        # 批量处理机会水平估计和结果分析
        chance_img, chance_std_img, chance_info, subject_results = process_searchlight_results_batch(
            beta_images, labels, process_mask_path, config, searchlight,
            subject_id, contrast_name, groups=groups, trial_details=trial_details,
            cond1=cond1, cond2=cond2
        )

        if subject_results:
            # 创建被试结果目录
            subject_result_dir = config.results_dir / "subject_results" / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            # 批量保存所有地图文件
            save_maps_batch(
                searchlight.scores_, chance_img, chance_std_img, accuracy_data,
                subject_result_dir, contrast_name, subject_results, config
            )

            return subject_results
        else:
            log(f"被试 {subject_id} {contrast_name} 结果分析失败", config)
            return None

    except Exception as e:
        log(f"处理searchlight结果失败: {str(e)}", config)
        return None


def process_single_subject_wrapper(task_params):
    """单被试处理的包装函数，用于并行处理"""
    try:
        # 从字典重建配置对象（避免序列化问题）
        config = dict_to_config(task_params['config_dict'])
        subject_id = task_params['subject_id']
        process_mask_path = task_params['process_mask_path']

        # 设置子进程日志（避免日志冲突）
        log_file = config.results_dir / "logs" / f"subject_{subject_id}.log"
        file_handler, console_handler = _create_logging_handlers(log_file)
        logging.basicConfig(
            level=config.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler, console_handler],
            force=True,
        )

        return process_single_subject(subject_id, config, process_mask_path)

    except Exception as e:
        import traceback
        print(f"Wrapper error for {task_params.get('subject_id', 'unknown')}: {str(e)}")
        print(traceback.format_exc())
        return None


def process_subjects_parallel(config, process_mask_path):
    """并行处理所有被试"""
    all_results = []

    # 准备任务参数
    tasks = []
    for subject_id in config.subjects:
        task_params = {
            'subject_id': subject_id,
            'config_dict': config_to_dict(config),  # 将配置转换为可序列化的字典
            'process_mask_path': str(process_mask_path)
        }
        tasks.append(task_params)

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # 提交所有任务
        future_to_subject = {
            executor.submit(process_single_subject_wrapper, task): task['subject_id']
            for task in tasks
        }

        # 收集结果
        completed = 0
        total = len(tasks)

        for future in as_completed(future_to_subject):
            subject_id = future_to_subject[future]
            try:
                subject_results = future.result(timeout=3600)  # 1小时超时
                if subject_results:
                    all_results.extend(subject_results)
                    log(f"✅ 完成被试 {subject_id} ({completed + 1}/{total})", config)
                else:
                    log(f"❌ 被试 {subject_id} 处理失败", config)
            except Exception as e:
                log(f"❌ 被试 {subject_id} 处理异常: {str(e)}", config)

            completed += 1

    return all_results


def process_subjects_serial(config, process_mask_path):
    """串行处理所有被试（原逻辑）"""
    all_results = []
    processed_subjects = 0

    for i, subject_id in enumerate(config.subjects):
        log(f"\n处理被试 ({i + 1}/{len(config.subjects)}): {subject_id}", config)

        subject_results = process_single_subject(
            subject_id, config, str(process_mask_path)
        )

        if subject_results:
            all_results.extend(subject_results)
            processed_subjects += 1

        # 内存清理
        memory_cleanup()

    return all_results


# ============================================================================
# 组水平统计分析模块 (保持原有功能)
# ============================================================================

def permutation_test_group_level(deltas, config):
    """组水平符号翻转置换检验"""
    delta_values = np.asarray(deltas, dtype=float)

    if delta_values.size == 0:
        raise ValueError("group_accuracies must contain at least one value")

    rng = np.random.default_rng(config.permutation_random_state)
    observed_mean = float(np.mean(delta_values))
    observed_std = float(np.std(delta_values, ddof=1)) if delta_values.size > 1 else 0.0
    sem = observed_std / np.sqrt(len(delta_values)) if len(delta_values) > 0 else np.nan
    observed_t = observed_mean / sem if sem > 0 else 0.0

    sign_vectors, total_permutations, used_exact = generate_sign_flip_vectors(
        len(delta_values),
        config,
        rng
    )

    if total_permutations == 0:
        return {
            'observed_mean': observed_mean,
            'observed_t': observed_t,
            'p_value': 1.0,
            'null_distribution': np.array([], dtype=float),
            'used_exact': used_exact,
            'sem': sem,
            'observed_std': observed_std,
            'n_permutations': 0
        }

    permuted_means = []
    for signs in sign_vectors:
        permuted_means.append(np.mean(delta_values * signs))

    permuted_means = np.asarray(permuted_means, dtype=float)

    if getattr(config, 'enable_two_sided_group_tests', True):
        extreme = np.sum(np.abs(permuted_means) >= abs(observed_mean)) + 1
    else:
        extreme = np.sum(permuted_means >= observed_mean) + 1

    p_value = extreme / (total_permutations + 1)

    return {
        'observed_mean': observed_mean,
        'observed_t': observed_t,
        'p_value': float(p_value),
        'null_distribution': permuted_means,
        'used_exact': used_exact,
        'sem': sem,
        'observed_std': observed_std,
        'n_permutations': total_permutations
    }


def perform_group_statistics_corrected(group_df, config):
    """执行修正的组水平统计分析"""
    log("开始组水平统计分析", config)

    stats_results = []

    # 按对比条件分组进行分析
    for contrast_name in group_df['contrast'].unique():
        contrast_data = group_df[group_df['contrast'] == contrast_name]
        log(f"分析对比: {contrast_name}, 数据点: {len(contrast_data)}", config)

        if len(contrast_data) < 3:
            log(f"跳过 {contrast_name}: 数据点不足", config)
            continue

        accuracies = contrast_data['accuracy'].values
        chance_values = contrast_data.get('mean_chance_accuracy', pd.Series(np.full_like(accuracies, 0.5))).values
        if 'mean_delta_accuracy' in contrast_data.columns:
            deltas = contrast_data['mean_delta_accuracy'].values
        else:
            deltas = accuracies - chance_values

        mean_acc = np.mean(accuracies)
        mean_chance = np.mean(chance_values)
        mean_delta = np.mean(deltas)

        std_delta = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
        sem_delta = std_delta / np.sqrt(len(deltas)) if len(deltas) > 0 else np.nan

        t_stat, t_pval = stats.ttest_1samp(deltas, 0.0)

        perm_result = permutation_test_group_level(deltas, config)

        cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0

        ci_lower_delta = mean_delta - 1.96 * sem_delta if np.isfinite(sem_delta) else np.nan
        ci_upper_delta = mean_delta + 1.96 * sem_delta if np.isfinite(sem_delta) else np.nan

        ci_lower_acc = mean_chance + ci_lower_delta if np.isfinite(ci_lower_delta) else np.nan
        ci_upper_acc = mean_chance + ci_upper_delta if np.isfinite(ci_upper_delta) else np.nan

        is_significant_t = t_pval < config.alpha_level
        is_significant_perm = perm_result['p_value'] < config.alpha_level

        result = {
            'contrast': contrast_name,
            'n_subjects': len(accuracies),
            'mean_accuracy': mean_acc,
            'mean_chance_accuracy': mean_chance,
            'mean_delta_accuracy': mean_delta,
            'std_accuracy': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
            'std_delta_accuracy': std_delta,
            'sem_accuracy': sem_delta,
            'sem_delta_accuracy': sem_delta,
            'ci_lower': ci_lower_acc,
            'ci_upper': ci_upper_acc,
            'ci_lower_delta': ci_lower_delta,
            'ci_upper_delta': ci_upper_delta,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            't_like_statistic': perm_result['observed_t'],
            'permutation_pvalue': perm_result['p_value'],
            'sign_flip_pvalue': perm_result['p_value'],
            'cohens_d': cohens_d,
            'significant_t_test': is_significant_t,
            'significant_permutation': is_significant_perm,
            'used_exact_permutation': perm_result['used_exact'],
            'n_group_permutations': perm_result['n_permutations']
        }

        stats_results.append(result)

        # 记录结果
        sem_delta_str = f"{sem_delta:.4f}" if np.isfinite(sem_delta) else "nan"
        log(f"  {contrast_name} 结果:", config)
        log(f"    平均准确率: {mean_acc:.4f}", config)
        log(f"    平均机会水平: {mean_chance:.4f}", config)
        log(f"    平均差值: {mean_delta:.4f} ± {sem_delta_str}", config)
        log(
            f"    t检验(Δ): t={t_stat:.3f}, p={t_pval:.4f}"
            f" {'*' if is_significant_t else ''}",
            config
        )
        log(
            f"    符号翻转置换: p={perm_result['p_value']:.4f}"
            f" {'*' if is_significant_perm else ''}"
            f" | {'精确枚举' if perm_result['used_exact'] else '随机采样'}"
            f" {perm_result['n_permutations']} 次",
            config
        )
        log(f"    效应量(Δ): d={cohens_d:.3f}", config)

    stats_df = pd.DataFrame(stats_results)
    log(f"组水平统计分析完成，共 {len(stats_df)} 个对比条件", config)

    return stats_df


def perform_group_analysis(all_results, config):
    """执行组水平分析"""
    if all_results:
        log(f"\n开始组水平统计分析，共 {len(all_results)} 个结果", config)
        group_df = pd.DataFrame(all_results)
        group_df.to_csv(config.results_dir / "individual_searchlight_mvpa_results.csv", index=False)

        # 验证必要的字段存在
        if 'accuracy' not in group_df.columns:
            log("错误: 结果中缺少 'accuracy' 字段", config)
            return None, None

        # 执行组水平统计分析
        stats_df = perform_group_statistics_corrected(group_df, config)
        if stats_df is not None:
            stats_df.to_csv(config.results_dir / "group_searchlight_mvpa_statistics.csv", index=False)

        # 创建组水平图并获取汇总信息
        (
            map_outputs,
            cluster_summary,
            cluster_summary_path,
            significance_summary,
            significance_summary_path,
        ) = create_group_accuracy_maps(group_df, config)

        # 创建可视化并获取生成的图像路径
        main_viz_path, individual_viz_path, significance_viz_path = create_enhanced_visualizations(
            group_df,
            stats_df,
            config,
            significance_summary=significance_summary,
        )
        # 生成HTML报告
        main_viz_b64 = image_to_base64(main_viz_path) if main_viz_path else None
        individual_viz_b64 = image_to_base64(individual_viz_path) if individual_viz_path else None
        significance_viz_b64 = image_to_base64(significance_viz_path) if significance_viz_path else None
        generate_html_report(
            group_df,
            stats_df,
            config,
            main_viz_b64=main_viz_b64,
            individual_viz_b64=individual_viz_b64,
            map_outputs=map_outputs,
            cluster_df=cluster_summary,
            cluster_summary_path=cluster_summary_path,
            significance_summary_df=significance_summary,
            significance_summary_path=significance_summary_path,
            significance_viz_b64=significance_viz_b64,
        )

        log(f"\n分析完成！结果保存在: {config.results_dir}", config)
        log(f"分析被试数: {len(set(group_df['subject']))}", config)
        log(f"对比数量: {len(config.contrasts)}", config)

        return group_df, stats_df
    else:
        log("没有有效的分析结果", config)
        return None, None


# ============================================================================
# 主分析流程 - 并行版本
# ============================================================================

def run_group_searchlight_analysis_parallel(config):
    """并行版本的主分析流程"""
    log("开始并行组水平Searchlight MVPA分析", config)

    # 验证配置
    if not validate_config(config):
        log("配置验证失败，请检查配置参数", config)
        return None, None

    # 加载处理mask
    try:
        process_mask_path = load_process_mask(config)
    except FileNotFoundError as e:
        log(f"处理mask加载失败: {str(e)}", config)
        return None, None

    log(f"成功加载处理mask: {config.process_mask}", config)

    # 验证Searchlight参数
    if not validate_searchlight_parameters(config, process_mask_path):
        log("Searchlight参数验证失败", config)
        return None, None

    # 检查系统资源并调整并行参数
    if config.memory_aware_parallel:
        config = adjust_parallel_config_based_on_memory(config)

    # 并行处理所有被试
    if config.parallel_subjects and len(config.subjects) > 1:
        log(f"使用并行处理，最大进程数: {config.max_workers}", config)
        all_results = process_subjects_parallel(config, process_mask_path)
    else:
        log("使用串行处理", config)
        all_results = process_subjects_serial(config, process_mask_path)

    # 组水平分析
    return perform_group_analysis(all_results, config)


# ============================================================================
# 其他必要函数 (保持原有功能)
# ============================================================================

def create_group_accuracy_maps(group_df, config):
    """创建组水平准确率图及显著性结果, 返回路径信息和显著簇摘要。"""

    log("创建组水平准确率图", config)

    accuracy_dir = config.results_dir / "accuracy_maps"
    stats_dir = config.results_dir / "statistical_maps"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    map_outputs = {}
    cluster_rows = []
    significance_rows = []

    for contrast_name in group_df['contrast'].unique():
        contrast_data = group_df[group_df['contrast'] == contrast_name]

        if len(contrast_data) < 3:
            log(f"跳过 {contrast_name}: 数据点不足", config)
            continue

        accuracy_maps = []
        chance_maps = []

        for _, row in contrast_data.iterrows():
            try:
                accuracy_maps.append(image.load_img(row['accuracy_map_path']))
            except Exception as e:
                log(f"加载准确率图失败: {row.get('accuracy_map_path')}, {str(e)}", config)
                continue

            chance_path = row.get('chance_map_path')
            if chance_path:
                try:
                    chance_maps.append(image.load_img(chance_path))
                except Exception as e:
                    log(f"加载机会水平图失败: {chance_path}, {str(e)}", config)
                    chance_maps.append(None)
            else:
                chance_maps.append(None)

        if len(accuracy_maps) < 3:
            log(f"跳过 {contrast_name}: 有效准确率图不足", config)
            continue

        if any(ch is None for ch in chance_maps):
            log(f"跳过 {contrast_name}: 缺少机会水平图，无法执行两步校正", config)
            continue

        try:
            interpolation = getattr(config, 'group_interpolation', 'continuous')
            reference_img = get_group_reference_image(config, accuracy_maps[0])

            resampled_accuracy = [
                resample_image_to_reference(img, reference_img, interpolation=interpolation)
                for img in accuracy_maps
            ]
            resampled_chance = [
                resample_image_to_reference(img, reference_img, interpolation=interpolation)
                for img in chance_maps
            ]

            accuracy_data = np.stack([as_data_array(img) for img in resampled_accuracy], axis=0)
            chance_data = np.stack([as_data_array(img) for img in resampled_chance], axis=0)
            delta_data = accuracy_data - chance_data

            mean_accuracy_data = np.nanmean(accuracy_data, axis=0)
            mean_chance_data = np.nanmean(chance_data, axis=0)
            mean_delta_data = np.nanmean(delta_data, axis=0)

            threshold = float(getattr(config, 'classification_significance_threshold', 0.5))
            classification_mask = mean_accuracy_data > threshold
            classification_voxel_count = int(np.sum(classification_mask))

            if classification_voxel_count == 0:
                log(
                    f"跳过 {contrast_name}: 平均准确率超过 {threshold:.2f} 的体素为0，无法执行显著性分析",
                    config,
                )
                continue

            log(
                f"{contrast_name} 平均准确率超过 {threshold:.2f} 的体素数: {classification_voxel_count}",
                config,
            )

            mean_accuracy_img = image.new_img_like(reference_img, mean_accuracy_data)
            mean_chance_img = image.new_img_like(reference_img, mean_chance_data)
            mean_delta_img = image.new_img_like(reference_img, mean_delta_data)

            flat_delta = delta_data.reshape(delta_data.shape[0], -1)
            mask_flat = classification_mask.reshape(-1)
            masked_flat = flat_delta.copy()
            masked_flat[:, ~mask_flat] = np.nan

            t_vals, p_vals = stats.ttest_1samp(
                masked_flat,
                popmean=0.0,
                axis=0,
                nan_policy='omit',
            )
            t_map_data = t_vals.reshape(mean_delta_data.shape)
            p_map_data = p_vals.reshape(mean_delta_data.shape)

            t_map_data = np.where(classification_mask, t_map_data, 0.0)
            p_map_data = np.where(classification_mask, p_map_data, np.nan)

            positive_effect_mask = (mean_delta_data > 0) & classification_mask
            uncorrected_mask = (p_map_data < config.alpha_level) & positive_effect_mask
            fdr_mask, fdr_threshold = compute_fdr_mask(p_map_data, config.alpha_level)
            fdr_mask = fdr_mask & positive_effect_mask
            fwe_mask, fwe_threshold = compute_fwe_mask(p_map_data, config.alpha_level)
            fwe_mask = fwe_mask & positive_effect_mask

            slug = contrast_name.replace(' ', '_').replace('/', '-')
            accuracy_out = accuracy_dir / f"group_{slug}_mean_accuracy.nii.gz"
            chance_out = accuracy_dir / f"group_{slug}_mean_chance.nii.gz"
            delta_out = accuracy_dir / f"group_{slug}_mean_delta.nii.gz"
            t_out = stats_dir / f"group_{slug}_t_map.nii.gz"
            p_out = stats_dir / f"group_{slug}_p_map.nii.gz"
            threshold_label = f"{threshold:.2f}".replace('.', 'p')
            classification_mask_out = stats_dir / f"group_{slug}_classification_gt_{threshold_label}.nii.gz"

            classification_mask_img = image.new_img_like(
                reference_img,
                classification_mask.astype(np.uint8),
            )
            classification_mask_img.to_filename(str(classification_mask_out))
            log(
                f"保存分类阈值掩模(>{threshold:.2f}): {classification_mask_out}",
                config,
            )

            correction_masks = {
                'uncorrected': {'mask': uncorrected_mask, 'threshold': config.alpha_level},
                'fdr': {'mask': fdr_mask, 'threshold': fdr_threshold},
                'fwe': {'mask': fwe_mask, 'threshold': fwe_threshold},
            }

            affine = reference_img.affine
            voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))

            correction_outputs = {}

            for correction_label, info in correction_masks.items():
                mask_data = np.nan_to_num(info['mask'].astype(bool))
                filtered_mask, clusters = summarize_clusters(
                    mask_data,
                    mean_delta_data,
                    t_map_data,
                    p_map_data,
                    affine,
                    voxel_volume,
                    config,
                    contrast_name,
                    correction_label,
                )

                mask_img = image.new_img_like(reference_img, filtered_mask)
                thresh_delta = mean_delta_data * filtered_mask
                thresholded_img = image.new_img_like(reference_img, thresh_delta)

                mask_out = stats_dir / f"group_{slug}_significant_mask_{correction_label}.nii.gz"
                thresh_out = stats_dir / f"group_{slug}_thresholded_delta_{correction_label}.nii.gz"

                mask_img.to_filename(str(mask_out))
                thresholded_img.to_filename(str(thresh_out))

                log(
                    f"保存{correction_label.upper()}校正显著性掩模: {mask_out}",
                    config
                )
                log(
                    f"保存{correction_label.upper()}校正阈值化差值图: {thresh_out}",
                    config
                )

                cluster_csv = None
                n_significant_voxels = int(np.sum(filtered_mask))
                mean_significant_delta = (
                    float(np.nanmean(mean_delta_data[filtered_mask.astype(bool)]))
                    if n_significant_voxels > 0
                    else float('nan')
                )
                peak_significant_accuracy = (
                    float(np.nanmax(mean_accuracy_data[filtered_mask.astype(bool)]))
                    if n_significant_voxels > 0
                    else float('nan')
                )
                if clusters:
                    cluster_df = pd.DataFrame(clusters).sort_values('peak_t_value', ascending=False)
                    cluster_csv = stats_dir / f"group_{slug}_significant_clusters_{correction_label}.csv"
                    cluster_df.to_csv(cluster_csv, index=False)
                    cluster_rows.append(cluster_df)
                    log(f"保存{correction_label.upper()}校正显著脑区表: {cluster_csv}", config)
                else:
                    log(f"{contrast_name} 在 {correction_label.upper()} 校正下未检测到显著脑区", config)

                correction_outputs[correction_label] = {
                    'mask_path': mask_out,
                    'thresholded_delta_path': thresh_out,
                    'clusters': clusters,
                    'threshold': info['threshold'],
                    'cluster_table': cluster_csv,
                    'n_voxels': n_significant_voxels,
                    'proportion_of_classification_voxels': (
                        n_significant_voxels / classification_voxel_count
                        if classification_voxel_count > 0 else 0.0
                    ),
                    'mean_significant_delta': mean_significant_delta,
                    'peak_significant_accuracy': peak_significant_accuracy,
                }

                significance_rows.append({
                    'contrast': contrast_name,
                    'correction': correction_label,
                    'classification_threshold': threshold,
                    'n_voxels_above_threshold': classification_voxel_count,
                    'n_significant_voxels': n_significant_voxels,
                    'proportion_significant': (
                        n_significant_voxels / classification_voxel_count
                        if classification_voxel_count > 0 else 0.0
                    ),
                    'mean_significant_delta': mean_significant_delta,
                    'peak_significant_accuracy': peak_significant_accuracy,
                    'cluster_table': cluster_csv,
                })

            t_map_img = image.new_img_like(reference_img, t_map_data)
            p_map_img = image.new_img_like(reference_img, p_map_data)

            mean_accuracy_img.to_filename(str(accuracy_out))
            mean_chance_img.to_filename(str(chance_out))
            mean_delta_img.to_filename(str(delta_out))
            t_map_img.to_filename(str(t_out))
            p_map_img.to_filename(str(p_out))

            log(f"保存组水平平均准确率图: {accuracy_out}", config)
            log(f"保存组水平平均机会水平图: {chance_out}", config)
            log(f"保存组水平均差值图: {delta_out}", config)
            log(f"保存统计图: t图 {t_out}, p图 {p_out}", config)
            log(f"FDR 临界p值: {fdr_threshold if np.isfinite(fdr_threshold) else '无显著体素'}", config)
            log(f"FWE (Bonferroni) 阈值: {fwe_threshold if np.isfinite(fwe_threshold) else '无显著体素'}", config)

            primary_correction = getattr(config, 'primary_correction', 'fdr')
            primary_outputs = correction_outputs.get(primary_correction) or correction_outputs.get('uncorrected')

            map_outputs[contrast_name] = {
                'mean_accuracy_map': accuracy_out,
                'mean_chance_map': chance_out,
                'mean_delta_map': delta_out,
                't_map': t_out,
                'p_map': p_out,
                'classification_mask': classification_mask_out,
                'classification_threshold': threshold,
                'n_voxels_above_threshold': classification_voxel_count,
                'significant_mask': primary_outputs['mask_path'] if primary_outputs else None,
                'thresholded_delta_map': primary_outputs['thresholded_delta_path'] if primary_outputs else None,
                'cluster_table': primary_outputs['cluster_table'] if primary_outputs else None,
                'corrections': correction_outputs,
            }
        except Exception as e:
            log(f"创建 {contrast_name} 组水平图失败: {str(e)}", config)
            continue

    cluster_summary = pd.concat(cluster_rows, ignore_index=True) if cluster_rows else pd.DataFrame()
    summary_csv = None
    if not cluster_summary.empty:
        summary_csv = stats_dir / "group_significant_clusters_summary.csv"
        cluster_summary.to_csv(summary_csv, index=False)
        log(f"保存显著脑区汇总表: {summary_csv}", config)

    significance_summary = pd.DataFrame(significance_rows)
    significance_summary_csv = None
    if not significance_summary.empty:
        significance_summary_csv = stats_dir / "group_significant_voxel_summary.csv"
        significance_summary.to_csv(significance_summary_csv, index=False)
        log(f"保存分类阈值显著性汇总表: {significance_summary_csv}", config)

    return map_outputs, cluster_summary, summary_csv, significance_summary, significance_summary_csv


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()

    try:
        # 运行并行分析
        group_df, stats_df = run_group_searchlight_analysis_parallel(config)
        if group_df is not None:
            print(f"\n✅ 分析完成！结果保存在: {config.results_dir}")
            return config.results_dir
        else:
            print("\n❌ 分析失败：没有有效结果")
            return None
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()