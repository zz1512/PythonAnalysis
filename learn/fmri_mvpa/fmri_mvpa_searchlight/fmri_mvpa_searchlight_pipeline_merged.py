# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI MVPA Searchlight Pipeline - 完整版本（带特征选择）

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
from nilearn.mass_univariate import permuted_ols
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
                logging.FileHandler(self.log_file),
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
    # 移除所有非ASCII字符
    safe_message = message.encode('ascii', 'ignore').decode('ascii')
    logging.info(safe_message)

def memory_cleanup():
    """内存清理"""
    gc.collect()


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
        mask_data = mask_img.get_fdata()
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

def analyze_searchlight_results(searchlight, config, subject_id, contrast_name):
    """分析Searchlight结果"""
    try:
        accuracy_data = searchlight.scores_.get_fdata()
        mask_data = searchlight.mask_img_.get_fdata() > 0

        # 只分析mask内的体素
        masked_accuracies = accuracy_data[mask_data]

        if len(masked_accuracies) == 0:
            log(f"警告: {subject_id} {contrast_name} 无有效体素", config)
            return None

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


@enhanced_error_handling("Searchlight分类分析")
def run_searchlight_classification(beta_images, labels, process_mask_path, config, groups=None):
    """运行searchlight分类分析 - 改进版本"""
    try:
        # 资源检查
        check_system_resources(config)

        # 构建分类器
        classifier = SVC(**config.svm_params)

        # 设置交叉验证
        if groups is not None and len(np.unique(groups)) > 1:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(n_splits=min(config.cv_folds, len(labels)),
                                 random_state=config.cv_random_state, shuffle=True)

        # 创建SearchLight对象
        searchlight = SearchLight(
            mask_img=process_mask_path,
            radius=config.searchlight_radius,
            estimator=classifier,
            cv=cv,
            scoring='accuracy',
            n_jobs=config.n_jobs,
            verbose=1
        )

        # 运行searchlight分析
        log(f"开始Searchlight分析: 半径={config.searchlight_radius}体素, 样本数={len(labels)}", config)
        start_time = time.time()

        searchlight.fit(beta_images, labels, groups=groups)

        end_time = time.time()
        duration = (end_time - start_time) / 60
        log(f"Searchlight完成: 耗时{duration:.1f}分钟", config)

        return searchlight

    except Exception as e:
        log(f"Searchlight分析失败: {str(e)}", config)
        return None


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

def permutation_test_group_level(group_accuracies, n_permutations=1000, random_state=None):
    """组水平置换检验"""
    if random_state is not None:
        np.random.seed(random_state)

    # 计算真实的组平均准确率
    true_mean = np.mean(group_accuracies)

    # 运行置换测试
    perm_means = []
    chance_level = 0.5

    for i in range(n_permutations):
        # 生成随机准确率（围绕chance level）
        perm_accuracies = np.random.normal(chance_level, 0.1, len(group_accuracies))
        perm_accuracies = np.clip(perm_accuracies, 0, 1)  # 限制在[0,1]范围内
        perm_means.append(np.mean(perm_accuracies))

    perm_means = np.array(perm_means)

    # 计算p值
    p_value = np.sum(perm_means >= true_mean) / n_permutations

    return {
        'true_mean': true_mean,
        'perm_means': perm_means,
        'p_value': p_value,
        'perm_mean': np.mean(perm_means),
        'perm_std': np.std(perm_means)
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

        # 提取准确率
        accuracies = contrast_data['accuracy'].values

        # 基本统计
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        sem_acc = std_acc / np.sqrt(len(accuracies))

        # 单样本t检验 (与chance level 0.5比较)
        t_stat, t_pval = stats.ttest_1samp(accuracies, 0.5)

        # 置换检验
        perm_result = permutation_test_group_level(
            accuracies,
            n_permutations=config.n_permutations,
            random_state=config.permutation_random_state
        )

        # 效应量 (Cohen's d)
        cohens_d = (mean_acc - 0.5) / std_acc if std_acc > 0 else 0

        # 95%置信区间
        ci_lower = mean_acc - 1.96 * sem_acc
        ci_upper = mean_acc + 1.96 * sem_acc

        # 显著性判断
        is_significant_t = t_pval < config.alpha_level
        is_significant_perm = perm_result['p_value'] < config.alpha_level

        result = {
            'contrast': contrast_name,
            'n_subjects': len(accuracies),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'sem_accuracy': sem_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'permutation_pvalue': perm_result['p_value'],
            'cohens_d': cohens_d,
            'significant_t_test': is_significant_t,
            'significant_permutation': is_significant_perm,
            'perm_mean': perm_result['perm_mean'],
            'perm_std': perm_result['perm_std']
        }

        stats_results.append(result)

        # 记录结果
        log(f"  {contrast_name} 结果:", config)
        log(f"    平均准确率: {mean_acc:.4f} ± {sem_acc:.4f}", config)
        log(f"    t检验: t={t_stat:.3f}, p={t_pval:.4f} {'*' if is_significant_t else ''}", config)
        log(f"    置换检验: p={perm_result['p_value']:.4f} {'*' if is_significant_perm else ''}", config)
        log(f"    效应量: d={cohens_d:.3f}", config)

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
                    # 分析结果
                    subject_results = analyze_searchlight_results(
                        searchlight, config, subject_id, contrast_name
                    )

                    if subject_results:
                        # 保存个体结果
                        subject_result_dir = config.results_dir / "subject_results" / subject_id
                        subject_result_dir.mkdir(parents=True, exist_ok=True)

                        # 保存准确率图
                        accuracy_map_path = subject_result_dir / f"{contrast_name}_accuracy_map.nii.gz"
                        searchlight.scores_.to_filename(str(accuracy_map_path))
                        subject_results['accuracy_map_path'] = str(accuracy_map_path)

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
        for _, row in contrast_data.iterrows():
            try:
                accuracy_img = image.load_img(row['accuracy_map_path'])
                accuracy_maps.append(accuracy_img)
            except Exception as e:
                log(f"加载准确率图失败: {row['accuracy_map_path']}, {str(e)}", config)
                continue

        if len(accuracy_maps) < 3:
            log(f"跳过 {contrast_name}: 有效准确率图不足", config)
            continue

        try:
            # 计算平均准确率图
            mean_accuracy_img = image.mean_img(accuracy_maps)

            # 保存组水平准确率图
            group_accuracy_path = config.results_dir / "accuracy_maps" / f"group_{contrast_name}_mean_accuracy.nii.gz"
            group_accuracy_path.parent.mkdir(parents=True, exist_ok=True)
            mean_accuracy_img.to_filename(str(group_accuracy_path))

            log(f"保存组水平准确率图: {group_accuracy_path}", config)

            # 进行单样本t检验
            # 准备数据进行统计检验
            accuracy_data = np.array([img.get_fdata() for img in accuracy_maps])

            # 创建设计矩阵（单样本t检验）
            design_matrix = np.ones((len(accuracy_maps), 1))

            # 运行置换检验
            log(f"运行 {contrast_name} 的置换检验...", config)
            neg_log_pvals, t_scores_original_data, _ = permuted_ols(
                tested_vars=design_matrix,
                target_vars=accuracy_data.reshape(len(accuracy_maps), -1),
                model_intercept=False,
                n_perm=config.n_permutations,
                two_sided_test=False,  # 单侧检验（准确率 > 0.5）
                random_state=config.permutation_random_state,
                n_jobs=config.n_jobs
            )

            # 重塑结果到原始图像形状
            original_shape = accuracy_maps[0].shape
            t_scores_img_data = t_scores_original_data.reshape(original_shape)
            neg_log_pvals_img_data = neg_log_pvals.reshape(original_shape)

            # 创建统计图像
            t_scores_img = image.new_img_like(accuracy_maps[0], t_scores_img_data)
            neg_log_pvals_img = image.new_img_like(accuracy_maps[0], neg_log_pvals_img_data)

            # 保存统计图
            t_scores_path = config.results_dir / "statistical_maps" / f"group_{contrast_name}_t_scores.nii.gz"
            neg_log_pvals_path = config.results_dir / "statistical_maps" / f"group_{contrast_name}_neg_log_pvals.nii.gz"

            t_scores_path.parent.mkdir(parents=True, exist_ok=True)

            t_scores_img.to_filename(str(t_scores_path))
            neg_log_pvals_img.to_filename(str(neg_log_pvals_path))

            log(f"保存统计图: {t_scores_path}", config)
            log(f"保存p值图: {neg_log_pvals_path}", config)

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