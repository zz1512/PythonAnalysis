# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI MVPA Searchlight Pipeline - 完整版本

这是将所有模块合并到一个文件中的版本，包含以下功能：
- 配置管理
- 工具函数
- 数据加载
- Searchlight特征提取
- 特征选择
- 分类分析
- 组水平统计分析
- 可视化
- HTML报告生成
- 主分析流程
- 数据验证和诊断
"""

import gc
import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
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
# 配置管理模块 (config.py)
# ============================================================================

class MVPAConfig:
    """MVPA Searchlight分析配置类"""

    def __init__(self):
        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 5)]
        self.runs = [3, 4]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.mask_dir = Path(r"../../../data/masks")  # 使用通用mask目录
        self.results_dir = Path(r"../../../learn_mvpa/searchlight_mvpa")

        # Searchlight特定参数
        self.searchlight_radius = 3  # searchlight半径（体素）
        self.process_mask = "gray_matter_mask.nii.gz"  # 处理mask
        self.min_region_size = 10  # 最小区域大小（体素数）

        # 分类器参数
        self.svm_params = {
            'random_state': 42
        }

        # 参数网格搜索配置
        self.svm_param_grid = {
            'C': [0.1, 1.0, 10.0],  # 可选的C值
            'kernel': ['linear']  # 保持线性核
        }

        # 交叉验证参数
        self.cv_folds = 5  # searchlight通常使用较少的fold以节省计算时间
        self.cv_random_state = 42

        # 置换检验参数（仅用于组水平分析）
        self.n_permutations = 1000  # searchlight通常使用较少的置换次数
        self.permutation_random_state = 42
        self.within_subject_permutations = 1000  # 每个被试估计机会水平的置换次数
        self.max_exact_sign_flips_subjects = 12  # 被试数不超过该阈值时枚举全部符号翻转
        self.enable_two_sided_group_tests = True  # 组水平检测使用双侧检验

        # 显著性水平
        self.alpha_level = 0.05

        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)
        self.use_parallel = True

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

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
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
# 工具函数模块 (utils.py)
# ============================================================================

def log(message, config):
    """记录日志信息"""
    logging.info(message)

def log_safe(message, config):
    """安全的日志记录，完全禁用特殊字符"""
    try:
        logging.info(message)
    except UnicodeEncodeError:
        # 如果目标环境仍无法处理Unicode，则回退到移除不可编码字符
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        logging.info(safe_message)

def memory_cleanup():
    """内存清理"""
    gc.collect()


def as_data_array(img):
    """将Niimg或数组安全地转换为numpy数组."""
    if hasattr(img, "get_fdata"):
        return img.get_fdata()
    return np.asarray(img)


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
# 验证和诊断模块 (validation.py)
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


def check_feature_dimension_warning(X, y, sphere_info, config):
    """检查特征维度问题并给出建议"""
    n_samples, n_features = X.shape
    ratio = n_features / n_samples
    if ratio > 10:
        return False
    elif ratio > 5:
        return True
    else:
        return True


# ============================================================================
# 添加Searchlight参数验证
# ============================================================================

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


def build_classification_pipeline(config, n_features):
    """构建Searchlight分类pipeline"""

    steps = [
        ('imputer', SimpleImputer(strategy='mean')),  # 处理可能的缺失值
        ('variance_filter', VarianceThreshold(threshold=0.0)),  # 只移除零方差特征
        ('scaler', StandardScaler()),  # 标准化
        ('classifier', SVC(**config.svm_params))  # 分类器
    ]

    return Pipeline(steps)


# ============================================================================
# 添加缺失的函数
# ============================================================================

def analyze_searchlight_results(searchlight, config, subject_id, contrast_name,
                                chance_img=None, chance_info=None):
    """分析Searchlight结果"""
    try:
        accuracy_data = as_data_array(searchlight.scores_)
        mask_data = as_data_array(searchlight.mask_img_) > 0

        masked_accuracies = accuracy_data[mask_data]

        if masked_accuracies.size == 0:
            log(f"警告: {subject_id} {contrast_name} 无有效体素", config)
            return None

        chance_metrics = {}
        chance_data = None
        if chance_img is not None:
            chance_data = as_data_array(chance_img)
            masked_chance = chance_data[mask_data]
            chance_metrics['mean_chance_accuracy'] = float(np.mean(masked_chance))
            chance_metrics['chance_accuracy_std'] = float(np.std(masked_chance))
            chance_metrics['chance_accuracy_min'] = float(np.min(masked_chance))
            chance_metrics['chance_accuracy_max'] = float(np.max(masked_chance))
            chance_metrics['chance_above_half_ratio'] = float(np.mean(masked_chance > 0.5))

            if chance_info and 'n_subject_permutations' in chance_info:
                chance_metrics['n_subject_permutations'] = chance_info['n_subject_permutations']

        delta_metrics = {}
        if chance_data is not None:
            delta_data = accuracy_data - chance_data
            masked_delta = delta_data[mask_data]
            delta_metrics['mean_delta_accuracy'] = float(np.mean(masked_delta))
            delta_metrics['delta_accuracy_std'] = float(np.std(masked_delta, ddof=1)) if masked_delta.size > 1 else 0.0
            delta_metrics['delta_accuracy_min'] = float(np.min(masked_delta))
            delta_metrics['delta_accuracy_max'] = float(np.max(masked_delta))
            delta_metrics['delta_above_zero_ratio'] = float(np.mean(masked_delta > 0))

            if chance_info and 'chance_std_data' in chance_info:
                chance_std_masked = chance_info['chance_std_data'][mask_data]
                delta_metrics['mean_chance_std'] = float(np.mean(chance_std_masked))

        # 计算各种统计量
        results = {
            'subject': subject_id,
            'contrast': contrast_name,
            'accuracy': np.mean(masked_accuracies),  # 关键：这里需要accuracy字段
            'mean_accuracy': np.mean(masked_accuracies),
            'max_accuracy': np.max(masked_accuracies),
            'min_accuracy': np.min(masked_accuracies),
            'accuracy_std': np.std(masked_accuracies),
            'above_chance_ratio': np.mean(masked_accuracies > 0.5),
            'n_voxels_above_55': np.sum(masked_accuracies > 0.55),
            'n_voxels_above_60': np.sum(masked_accuracies > 0.6),
            'n_voxels_total': len(masked_accuracies)
        }

        results.update(chance_metrics)
        results.update(delta_metrics)

        log(f"  {contrast_name} 结果:", config)
        log(f"    - 平均准确率: {results['mean_accuracy']:.4f}", config)
        log(f"    - 最高准确率: {results['max_accuracy']:.4f}", config)
        log(f"    - 体素数: {results['n_voxels_total']}", config)
        log(f"    - 高于机会水平: {results['above_chance_ratio']:.1%}", config)

        return results

    except Exception as e:
        log(f"结果分析失败: {str(e)}", config)
        return None

def debug_data_pipeline(subject, config):
    """调试数据处理流程"""
    log(f"调试模式: 详细检查被试 {subject} 的数据处理流程", config)

    try:
        # 加载数据
        beta_images, trial_info = load_multi_run_lss_data(subject, config.runs, config)

        if beta_images is None or trial_info is None:
            log(f"调试: {subject} 数据加载失败", config)
            return None, None

        log(f"调试: {subject} 成功加载 {len(beta_images)} 个beta文件", config)
        log(f"调试: trial_info shape: {trial_info.shape}", config)
        log(f"调试: trial_info columns: {list(trial_info.columns)}", config)

        # 检查条件分布
        condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'
        if condition_col in trial_info.columns:
            condition_counts = trial_info[condition_col].value_counts()
            log(f"调试: 条件分布: {dict(condition_counts)}", config)

        return beta_images, trial_info

    except Exception as e:
        log(f"调试数据流程失败: {str(e)}", config)
        return None, None


def diagnose_dimension_issues(config):
    """诊断维度相关问题"""
    log("开始维度问题诊断...", config)

    # 检查一个示例被试的数据维度
    test_subject = config.subjects[0] if config.subjects else None
    if test_subject:
        try:
            beta_images, trial_info = load_multi_run_lss_data(test_subject, config.runs, config)
            if beta_images and trial_info is not None:
                log(f"示例被试 {test_subject}: {len(beta_images)} 个trial", config)

                # 检查处理mask
                process_mask_path = config.mask_dir / config.process_mask
                if process_mask_path.exists():
                    masker = NiftiMasker(mask_img=str(process_mask_path))
                    try:
                        X_sample = masker.fit_transform(beta_images[:2])  # 只测试前2个
                        log(f"处理mask特征维度: {X_sample.shape[1]} 个体素", config)
                    except Exception as e:
                        log(f"处理mask测试失败: {str(e)}", config)

        except Exception as e:
            log(f"维度诊断失败: {str(e)}", config)


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


# ============================================================================
# 数据加载模块 (data_loading.py)
# ============================================================================

def load_process_mask(config):
    """加载处理mask文件"""
    log("加载处理mask", config)

    process_mask_path = config.mask_dir / config.process_mask

    if not process_mask_path.exists():
        raise FileNotFoundError(f"处理mask文件不存在: {process_mask_path}")

    log(f"加载处理mask: {config.process_mask}", config)
    return str(process_mask_path)


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
# 改进内存管理
# ============================================================================

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
    """内部函数，用于拟合SearchLight估计器"""
    classifier = SVC(**config.svm_params)

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
    return searchlight


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
        searchlight = _fit_searchlight(
            beta_images,
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


def estimate_subject_chance_map(beta_images, labels, process_mask_path, config, template_img,
                                groups=None):
    """通过标签置换估计单个被试的机会水平地图"""

    n_permutations = getattr(config, 'within_subject_permutations', 0)
    if n_permutations is None or n_permutations <= 0:
        return None, None

    rng = np.random.default_rng(config.permutation_random_state)
    sum_scores = None
    sum_sq_scores = None
    actual_permutations = 0

    for perm_index in range(n_permutations):
        permuted_labels = permute_labels_within_groups(labels, groups, rng)

        try:
            perm_searchlight = _fit_searchlight(
                beta_images,
                permuted_labels,
                process_mask_path,
                config,
                groups=groups,
                verbose=0
            )
        except Exception as exc:  # pragma: no cover - 记录并继续
            log(f"置换 {perm_index + 1}/{n_permutations} 失败: {exc}", config)
            continue

        perm_scores = as_data_array(perm_searchlight.scores_)

        if sum_scores is None:
            sum_scores = perm_scores
            sum_sq_scores = perm_scores ** 2
        else:
            sum_scores += perm_scores
            sum_sq_scores += perm_scores ** 2

        actual_permutations += 1

    if actual_permutations == 0:
        log("警告: 未能生成任何有效的机会水平置换地图", config)
        return None, None

    mean_data = sum_scores / actual_permutations
    var_data = np.maximum(sum_sq_scores / actual_permutations - mean_data ** 2, 0)
    std_data = np.sqrt(var_data)

    chance_img = image.new_img_like(template_img, mean_data)
    chance_std_img = image.new_img_like(template_img, std_data)

    return chance_img, chance_std_img, {
        'n_subject_permutations': actual_permutations,
        'chance_mean_data': mean_data,
        'chance_std_data': std_data
    }


# ============================================================================
# Searchlight特征提取模块 (searchlight_extraction.py)
# ============================================================================


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
# 组水平统计分析模块 (group_statistics.py)
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


# ============================================================================
# 更新主分析流程
# ============================================================================
def run_group_searchlight_analysis(config):
    """运行组水平searchlight分析 - 更新版本"""
    log("开始组水平Searchlight MVPA分析", config)

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

    # 初始化结果存储
    all_results = []
    processed_subjects = 0

    # 对每个被试进行分析
    for i, subject_id in enumerate(config.subjects):
        log(f"\n处理被试 ({i + 1}/{len(config.subjects)}): {subject_id}", config)

        try:
            # 加载数据
            beta_images, trial_info = load_multi_run_lss_data(subject_id, config.runs, config)

            if beta_images is None or trial_info is None:
                log(f"被试 {subject_id} 的LSS数据加载失败，跳过", config)
                continue

            # 检查数据泄露
            if not check_data_leakage(beta_images, trial_info, config, subject_id):
                log(f"被试 {subject_id} 数据泄露检查失败，跳过", config)
                continue

            subject_has_results = False

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

                # 检查样本数量
                if len(selected_betas) < 10:
                    log(f"  样本数量不足({len(selected_betas)})，跳过", config)
                    continue

                groups = trial_details['run'].fillna('unknown').astype(str).values

                # 运行searchlight分析
                searchlight = run_searchlight_classification(
                    selected_betas, labels, process_mask_path, config, groups=groups
                )

                if searchlight is not None:
                    chance_img, chance_std_img, chance_info = estimate_subject_chance_map(
                        selected_betas,
                        labels,
                        process_mask_path,
                        config,
                        searchlight.scores_,
                        groups=groups
                    )

                    subject_results = analyze_searchlight_results(
                        searchlight,
                        config,
                        subject_id,
                        contrast_name,
                        chance_img=chance_img,
                        chance_info=chance_info
                    )

                    if subject_results:
                        condition_col = 'original_condition' if 'original_condition' in trial_details.columns else 'condition'
                        if condition_col in trial_details.columns:
                            subject_results['n_trials_cond1'] = int((trial_details[condition_col] == cond1).sum())
                            subject_results['n_trials_cond2'] = int((trial_details[condition_col] == cond2).sum())

                        subject_result_dir = config.results_dir / "subject_results" / subject_id
                        subject_result_dir.mkdir(parents=True, exist_ok=True)

                        accuracy_map_path = subject_result_dir / f"{contrast_name}_accuracy_map.nii.gz"
                        searchlight.scores_.to_filename(str(accuracy_map_path))
                        subject_results['accuracy_map_path'] = str(accuracy_map_path)

                        if chance_img is not None:
                            chance_map_path = subject_result_dir / f"{contrast_name}_chance_map.nii.gz"
                            chance_img.to_filename(str(chance_map_path))
                            subject_results['chance_map_path'] = str(chance_map_path)

                            if chance_info and chance_info.get('n_subject_permutations'):
                                log(
                                    f"    机会水平置换次数: {chance_info['n_subject_permutations']}",
                                    config
                                )

                            if chance_std_img is not None:
                                chance_std_path = subject_result_dir / f"{contrast_name}_chance_std_map.nii.gz"
                                chance_std_img.to_filename(str(chance_std_path))
                                subject_results['chance_std_map_path'] = str(chance_std_path)

                            delta_data = as_data_array(searchlight.scores_) - as_data_array(chance_img)
                            delta_img = image.new_img_like(searchlight.scores_, delta_data)
                            delta_map_path = subject_result_dir / f"{contrast_name}_delta_map.nii.gz"
                            delta_img.to_filename(str(delta_map_path))
                            subject_results['delta_map_path'] = str(delta_map_path)

                        all_results.append(subject_results)
                        subject_has_results = True

            if subject_has_results:
                processed_subjects += 1

            # 内存清理
            memory_cleanup()

        except Exception as e:
            log(f"被试 {subject_id} 处理失败: {str(e)}", config)
            continue

    # 组水平分析
    if all_results:
        log(f"\n开始组水平统计分析，共 {len(all_results)} 个结果，来自 {processed_subjects} 个被试", config)
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

        # 创建组水平图
        create_group_accuracy_maps(group_df, config)

        # 创建可视化并获取生成的图像路径
        main_viz_path, individual_viz_path = create_enhanced_visualizations(group_df, stats_df, config)
        # 生成HTML报告
        main_viz_b64 = image_to_base64(main_viz_path) if main_viz_path else None
        individual_viz_b64 = image_to_base64(individual_viz_path) if individual_viz_path else None
        generate_html_report(group_df, stats_df, config, main_viz_b64=main_viz_b64,
                             individual_viz_b64=individual_viz_b64)

        log(f"\n分析完成！结果保存在: {config.results_dir}", config)
        log(f"分析被试数: {len(set(group_df['subject']))}", config)
        log(f"对比数量: {len(config.contrasts)}", config)

        return group_df, stats_df
    else:
        log("没有有效的分析结果", config)
        return None, None


def create_group_accuracy_maps(group_df, config):
    """创建组水平准确率图"""
    log("创建组水平准确率图", config)

    for contrast_name in group_df['contrast'].unique():
        contrast_data = group_df[group_df['contrast'] == contrast_name]

        if len(contrast_data) < 3:
            log(f"跳过 {contrast_name}: 数据点不足", config)
            continue

        # 加载所有个体准确率图
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
            n_subjects = len(accuracy_maps)
            accuracy_data = np.stack([as_data_array(img) for img in accuracy_maps], axis=0)
            chance_data = np.stack([as_data_array(img) for img in chance_maps], axis=0)
            delta_data = accuracy_data - chance_data

            mean_accuracy_img = image.mean_img(accuracy_maps)
            mean_chance_img = image.mean_img(chance_maps)
            mean_delta_data = np.mean(delta_data, axis=0)

            mean_delta_img = image.new_img_like(accuracy_maps[0], mean_delta_data)

            group_dir = config.results_dir / "accuracy_maps"
            group_dir.mkdir(parents=True, exist_ok=True)

            accuracy_out = group_dir / f"group_{contrast_name}_mean_accuracy.nii.gz"
            chance_out = group_dir / f"group_{contrast_name}_mean_chance.nii.gz"
            delta_out = group_dir / f"group_{contrast_name}_mean_delta.nii.gz"

            mean_accuracy_img.to_filename(str(accuracy_out))
            mean_chance_img.to_filename(str(chance_out))
            mean_delta_img.to_filename(str(delta_out))

            log(f"保存组水平平均准确率图: {accuracy_out}", config)
            log(f"保存组水平平均机会水平图: {chance_out}", config)
            log(f"保存组水平平均差值图: {delta_out}", config)

            delta_flat = delta_data.reshape(n_subjects, -1)
            mean_delta_flat = mean_delta_data.reshape(-1)
            if n_subjects > 1:
                std_delta_flat = np.std(delta_flat, axis=0, ddof=1)
            else:
                std_delta_flat = np.zeros(delta_flat.shape[1], dtype=float)
            sem_flat = np.divide(std_delta_flat, np.sqrt(n_subjects), where=np.sqrt(n_subjects) > 0)

            t_flat = np.zeros_like(mean_delta_flat)
            valid_mask = sem_flat > 0
            t_flat[valid_mask] = mean_delta_flat[valid_mask] / sem_flat[valid_mask]

            rng = np.random.default_rng(config.permutation_random_state)
            sign_vectors, total_permutations, used_exact = generate_sign_flip_vectors(
                n_subjects,
                config,
                rng
            )

            abs_obs_t = np.abs(t_flat[valid_mask])
            extreme_counts = np.ones_like(abs_obs_t, dtype=np.int64)
            max_distribution = []

            if total_permutations > 0 and abs_obs_t.size > 0:
                for signs in sign_vectors:
                    signed = delta_flat * signs[:, None]
                    perm_mean = np.mean(signed, axis=0)
                    perm_t = np.zeros_like(perm_mean)
                    perm_t[valid_mask] = perm_mean[valid_mask] / sem_flat[valid_mask]
                    perm_abs = np.abs(perm_t[valid_mask])
                    extreme_counts += (perm_abs >= abs_obs_t).astype(np.int64)
                    max_distribution.append(float(np.max(perm_abs)))

                max_distribution = np.asarray(max_distribution, dtype=float)
                uncorrected_valid = extreme_counts / (total_permutations + 1)

                corrected_counts = (
                    np.sum(max_distribution[:, None] >= abs_obs_t[None, :], axis=0) + 1
                )
                corrected_valid = corrected_counts / (total_permutations + 1)
                max_t_threshold = np.quantile(
                    max_distribution,
                    1 - config.alpha_level
                ) if max_distribution.size > 0 else np.nan
            else:
                uncorrected_valid = np.ones_like(abs_obs_t, dtype=float)
                corrected_valid = np.ones_like(abs_obs_t, dtype=float)
                max_distribution = np.array([], dtype=float)
                max_t_threshold = np.nan

            p_uncorrected = np.ones_like(mean_delta_flat, dtype=float)
            p_corrected = np.ones_like(mean_delta_flat, dtype=float)
            p_uncorrected[valid_mask] = uncorrected_valid
            p_corrected[valid_mask] = corrected_valid

            t_map_img = image.new_img_like(accuracy_maps[0], t_flat.reshape(accuracy_maps[0].shape))
            p_uncorr_img = image.new_img_like(accuracy_maps[0], p_uncorrected.reshape(accuracy_maps[0].shape))
            p_corr_img = image.new_img_like(accuracy_maps[0], p_corrected.reshape(accuracy_maps[0].shape))

            stats_dir = config.results_dir / "statistical_maps"
            stats_dir.mkdir(parents=True, exist_ok=True)

            t_path = stats_dir / f"group_{contrast_name}_t_like_map.nii.gz"
            p_uncorr_path = stats_dir / f"group_{contrast_name}_p_uncorrected.nii.gz"
            p_corr_path = stats_dir / f"group_{contrast_name}_p_fwer_corrected.nii.gz"

            t_map_img.to_filename(str(t_path))
            p_uncorr_img.to_filename(str(p_uncorr_path))
            p_corr_img.to_filename(str(p_corr_path))

            neg_log_p_uncorr = np.zeros_like(p_uncorrected)
            valid_pos = p_uncorrected > 0
            neg_log_p_uncorr[valid_pos] = -np.log10(p_uncorrected[valid_pos])
            neg_log_img = image.new_img_like(accuracy_maps[0], neg_log_p_uncorr.reshape(accuracy_maps[0].shape))
            neg_log_path = stats_dir / f"group_{contrast_name}_neg_log10_p_uncorrected.nii.gz"
            neg_log_img.to_filename(str(neg_log_path))

            log(f"保存统计图: {t_path}", config)
            log(f"保存未校正p值图: {p_uncorr_path}", config)
            log(f"保存FWER校正p值图: {p_corr_path}", config)

            threshold_info = {
                'contrast': contrast_name,
                'n_subjects': n_subjects,
                'n_group_permutations': int(total_permutations),
                'used_exact': bool(used_exact),
                'max_t_threshold': float(max_t_threshold) if np.isfinite(max_t_threshold) else None
            }

            threshold_path = stats_dir / f"group_{contrast_name}_thresholds.json"
            with open(threshold_path, 'w', encoding='utf-8') as f:
                json.dump(threshold_info, f, ensure_ascii=False, indent=2)

            log(f"保存阈值信息: {threshold_path}", config)
            if np.isfinite(max_t_threshold):
                log(f"  max|t| FWER阈值 (@α={config.alpha_level}): {max_t_threshold:.3f}", config)

        except Exception as e:
            log(f"创建 {contrast_name} 组水平图失败: {str(e)}", config)
            continue


def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()

    try:
        # 运行分析
        group_df, stats_df = run_group_searchlight_analysis(config)
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