#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI MVPA ROI Pipeline - 完整版本（带特征选择）

这是将所有模块合并到一个文件中的版本，包含以下功能：
- 配置管理
- 工具函数
- 数据加载
- 特征提取
- 特征选择
- 分类分析
- 组水平统计分析
- 可视化
- HTML报告生成
- 主分析流程
- 数据验证和诊断

使用方法：
    python fmri_mvpa_roi_pipeline_merged.py

作者: AI Assistant
日期: 2024
"""

import gc
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from multiprocessing import cpu_count
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
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
    """MVPA分析配置类"""

    def __init__(self):
        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 29)]
        self.runs = [3, 4]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.roi_dir = Path(r"../../../learn_mvpa/full_roi_mask")
        self.results_dir = Path(r"../../../learn_mvpa/metaphor_ROI_MVPA_corrected")

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
        self.cv_folds = 10
        self.cv_random_state = 42

        # 置换检验参数（仅用于组水平分析）
        self.n_permutations = 10000
        self.permutation_random_state = 42

        # 显著性水平
        self.alpha_level = 0.05

        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)
        self.use_parallel = True

        # 内存优化参数
        self.memory_cache = 'nilearn_cache'
        self.memory_level = 1

        # 特征选择配置（新增）
        self.apply_feature_selection = True  # 启用特征选择
        self.feature_selection_method = 'univariate'  # 'univariate' 或 'pca'
        self.max_features = 500  # 最大特征数量
        self.feature_selection_threshold = 10  # 特征/样本比例阈值

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

        # 日志设置
        self.log_level = logging.INFO
        self.log_file = self.results_dir / "analysis.log"

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
    print(message)


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

    if not config.roi_dir.exists():
        issues.append(f"ROI目录不存在: {config.roi_dir}")

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
        log(f"  ⚠️ 数据泄露检查发现问题:", config)
        for issue in issues:
            log(f"    {issue}", config)
        return False

    log(f"  ✅ 数据泄露检查通过", config)
    return True


def check_feature_dimension_warning(X, y, roi_name, config):
    """检查特征维度问题并给出建议"""
    n_samples, n_features = X.shape
    ratio = n_features / n_samples
    if ratio > 10:
        return False
    elif ratio > 5:
        return True
    else:
        return True


class DynamicSelectKBest(SelectKBest):
    """在每次拟合时根据输入维度自动调整k值的SelectKBest"""

    def __init__(self, score_func=f_classif, k=500):
        super().__init__(score_func=score_func, k=k)
        self.original_k = k

    def fit(self, X, y=None, **fit_params):
        if isinstance(self.original_k, int):
            max_k = min(self.original_k, X.shape[1])
            max_k = max(1, max_k)
            self.k = max_k
        else:
            self.k = self.original_k
        return super().fit(X, y, **fit_params)


class DynamicPCA(PCA):
    """在每次拟合时根据输入维度自动调整主成分数量的PCA"""

    def __init__(self, n_components=500, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.original_n_components = n_components

    def fit(self, X, y=None, **fit_params):
        if isinstance(self.original_n_components, int):
            max_components = min(self.original_n_components, X.shape[1])
            if X.shape[0] > 1:
                max_components = min(max_components, X.shape[0] - 1)
            max_components = max(1, max_components)
            self.n_components = max_components
        else:
            self.n_components = self.original_n_components
        return super().fit(X, y, **fit_params)


def build_classification_pipeline(config, n_features):
    """构建包含必要预处理和可选特征选择的分类pipeline"""

    steps = [('imputer', SimpleImputer(strategy='mean'))]

    if config.apply_feature_selection:
        if config.feature_selection_method == 'univariate':
            selector = DynamicSelectKBest(score_func=f_classif, k=min(config.max_features, n_features))
            steps.append(('feature_selector', selector))
        elif config.feature_selection_method == 'pca':
            reducer = DynamicPCA(n_components=min(config.max_features, n_features))
            steps.append(('feature_selector', reducer))
        else:
            log(f"  ⚠️ 未知的特征选择方法: {config.feature_selection_method}，跳过特征选择", config)

    steps.append(('scaler', StandardScaler()))

    svm_params = config.svm_params.copy()
    steps.append(('svm', SVC(**svm_params)))

    return Pipeline(steps)


def apply_feature_selection(X, y, config, method='univariate', k=500):
    """应用特征选择解决维度问题"""
    n_samples, n_features = X.shape

    # 去除常数特征（方差为0的特征）
    def remove_constant_features(X):
        variance = np.var(X, axis=0)
        non_constant_features = variance > 0  # 保留方差不为0的特征
        return X[:, non_constant_features]

    # 去除常数特征
    X = remove_constant_features(X)
    n_samples, n_features = X.shape  # 更新特征数量

    if n_features <= k:
        log(f"特征选择: 特征数({n_features}) <= 目标数({k})，跳过特征选择", config)
        return X

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充NaN
    X_imputed = imputer.fit_transform(X)

    if method == 'univariate':
        # 基于单变量统计检验的特征选择
        selector = SelectKBest(score_func=f_classif, k=min(k, n_features))
        X_selected = selector.fit_transform(X_imputed, y)
        selected_features = selector.get_support()
        log(f"单变量特征选择: 从{n_features}个特征中选择{np.sum(selected_features)}个最重要特征", config)

    elif method == 'pca':
        # 主成分分析降维
        pca = PCA(n_components=min(k, n_features))
        X_selected = pca.fit_transform(X_imputed)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        log(f"PCA降维: 从{n_features}个特征降维到{X_selected.shape[1]}个成分，解释方差{explained_variance:.3f}", config)

    else:
        log(f"使用默认特征选择方法", config)
        return X

    # 检查特征选择后的数据维度
    if X_selected.shape[0] < 10:
        log(f"⚠️ 特征选择后样本数不足({X_selected.shape[0]})，建议增加样本量或减少特征选择", config)

    return X_selected



def validate_classifier_performance(X, y, groups, config, roi_name, contrast_name):
    """验证分类器性能的真实性"""

    log(f"验证分类器泛化能力 - {roi_name} {contrast_name}:", config)

    y = np.asarray(y, dtype=int)

    # 1. 检查特征数量与样本数量的比例
    n_samples, n_features = X.shape
    ratio = n_features / n_samples
    log(f"  特征维度: {n_features}特征 / {n_samples}样本 = 比例{ratio:.2f}", config)

    if ratio > 10:
        log(f"  ⚠️ 警告: 特征维度远大于样本数量，容易过拟合!", config)

    # 2. 使用更严格的交叉验证
    pipeline = build_classification_pipeline(config, X.shape[1])

    try:
        if groups is not None and len(np.unique(groups)) > 1:
            logo = LeaveOneGroupOut()
            cv_scores = cross_val_score(pipeline, X, y, cv=logo, groups=groups, scoring='accuracy')
            log(f"  按run分组交叉验证: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}", config)
        else:
            max_splits = min(5, np.min(np.bincount(y)))
            n_splits = max(2, max_splits)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            log(f"  常规交叉验证: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}", config)
    except ValueError as e:
        log(f"  ⚠️ 交叉验证失败: {e}", config)
        cv_scores = np.array([])

    return cv_scores, ratio


def run_permutation_test_debug(X, y, config, n_permutations=100):
    """运行置换测试验证分类显著性（调试用）"""
    log("运行置换测试验证结果真实性:", config)

    y = np.asarray(y, dtype=int)

    pipeline = build_classification_pipeline(config, X.shape[1])

    # 真实准确率
    cv = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
    true_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    true_mean = np.mean(true_scores)

    # 置换标签测试
    perm_scores = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_score = cross_val_score(pipeline, X, y_perm, cv=cv, scoring='accuracy')
        perm_scores.append(np.mean(perm_score))

    perm_scores = np.array(perm_scores)
    p_value = np.mean(perm_scores >= true_mean)

    log(f"  真实准确率: {true_mean:.3f}", config)
    log(f"  置换测试准确率分布: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}", config)
    log(f"  置换测试p值: {p_value:.4f}", config)

    return true_mean, perm_scores, p_value


def debug_data_pipeline(subject, config):
    """调试数据流水线"""

    log(f"\n=== 调试数据流水线 - {subject} ===", config)

    # 加载原始数据
    beta_images, trial_info = load_multi_run_lss_data(subject, config.runs, config)

    if beta_images is None:
        log("数据加载失败", config)
        return None, None

    # 详细检查数据
    log(f"Beta文件数量: {len(beta_images)}", config)
    log(f"Trial信息数量: {len(trial_info)}", config)

    # 检查条件分布
    condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'
    if condition_col in trial_info.columns:
        condition_counts = trial_info[condition_col].value_counts()
        log(f"条件分布: {dict(condition_counts)}", config)

    # 检查索引映射
    if 'combined_trial_index' in trial_info.columns:
        log(f"组合索引范围: {trial_info['combined_trial_index'].min()} - {trial_info['combined_trial_index'].max()}",
            config)

        # 检查是否有重复索引
        duplicate_indices = trial_info['combined_trial_index'].duplicated()
        if duplicate_indices.any():
            log(f"⚠️ 发现重复的组合索引!", config)
            log(f"重复的索引: {trial_info[duplicate_indices]['combined_trial_index'].values}", config)

    # 检查具体的几个trial
    log("前5个trial的详细信息:", config)
    for i in range(min(5, len(trial_info))):
        trial = trial_info.iloc[i]
        log(f"  Trial {i}: 条件={trial[condition_col]}, 索引={trial.get('combined_trial_index', 'N/A')}, run={trial.get('run', 'N/A')}",
            config)

    return beta_images, trial_info


def diagnose_dimension_issues(config):
    """诊断维度问题"""
    log("=== 维度问题诊断 ===", config)

    # 加载一个示例被试的数据进行检查
    example_subject = config.subjects[0]
    beta_images, trial_info = load_multi_run_lss_data(example_subject, config.runs, config)

    if beta_images is None:
        log("无法加载示例数据", config)
        return

    # 检查一个ROI的特征维度
    roi_masks = load_roi_masks(config)
    example_roi = list(roi_masks.keys())[0]
    roi_path = roi_masks[example_roi]

    # 准备示例分类数据
    contrast = config.contrasts[0]
    selected_betas, labels, _ = prepare_classification_data(trial_info, beta_images,
                                                            contrast[1], contrast[2], config)

    if selected_betas is not None:
        # 提取特征但不进行选择
        X = extract_roi_features(selected_betas, roi_path, config)

        if X is not None:
            n_samples, n_features = X.shape
            ratio = n_features / n_samples

            log(f"诊断结果:", config)
            log(f"  ROI: {example_roi}", config)
            log(f"  样本数: {n_samples}", config)
            log(f"  特征数: {n_features}", config)
            log(f"  特征/样本比例: {ratio:.1f}", config)

            if ratio > 10:
                log(f"  🚨 严重维度问题! 需要立即修复", config)
            elif ratio > 5:
                log(f"  ⚠️ 中度维度问题", config)
            else:
                log(f"  ✅ 维度正常", config)


def enhanced_error_handling(operation_name):
    """增强的错误处理装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            config = None
            # 尝试从参数中获取config
            for arg in args:
                if isinstance(arg, MVPAConfig):
                    config = arg
                    break
            for kwarg in kwargs.values():
                if isinstance(kwarg, MVPAConfig):
                    config = kwarg
                    break

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{operation_name}失败: {str(e)}"
                if config:
                    log(error_msg, config)
                    logging.error(f"参数: args={args}, kwargs={kwargs}")
                else:
                    print(error_msg)
                raise

        return wrapper

    return decorator


# ============================================================================
# 数据加载模块 (data_loader.py)
# ============================================================================

def load_roi_masks(config):
    """加载ROI mask文件"""
    log("加载ROI masks", config)

    roi_masks = {}
    roi_files = list(config.roi_dir.glob("*.nii.gz"))

    if not roi_files:
        raise FileNotFoundError(f"在{config.roi_dir}中未找到ROI mask文件")

    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        roi_masks[roi_name] = str(roi_file)
        log(f"  加载ROI: {roi_name}", config)

    log(f"总共加载了{len(roi_masks)}个ROI masks", config)
    return roi_masks


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
# 特征提取模块 (feature_extraction.py)
# ============================================================================

@enhanced_error_handling("ROI特征提取")
def extract_roi_features(beta_images, roi_mask_path, config):
    """从beta图像中提取ROI特征"""
    try:
        # 使用NiftiMasker进行特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )

        # 提取特征
        X = masker.fit_transform(beta_images)

        return X

    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        log(f"失败的文件列表: {beta_images}", config)
        return None


@enhanced_error_handling("ROI特征提取（带特征选择）")
def extract_roi_features_with_selection(beta_images, roi_mask_path, config, labels=None):
    """带特征选择的ROI特征提取"""
    try:
        # 原始特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )

        X = masker.fit_transform(beta_images)

        # 检查维度问题（仅用于记录，实际特征选择在交叉验证pipeline中进行）
        if labels is not None:
            roi_name = Path(roi_mask_path).stem
            check_feature_dimension_warning(X, labels, roi_name, config)

        return X

    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        return None


# ============================================================================
# 分类分析模块 (classification.py)
# ============================================================================

@enhanced_error_handling("ROI分类分析")
def run_roi_classification_enhanced(X, y, config, groups=None):
    """增强的ROI分类分析 - 修正版，避免数据泄露并支持分组交叉验证"""

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups) if groups is not None else None
    use_groups = groups is not None and len(np.unique(groups)) > 1

    class_counts = np.bincount(y)
    if class_counts.min() < 2:
        log("  ⚠️ 分类任务跳过: 至少有一个类别的样本数不足2，无法进行稳定的交叉验证", config)
        return None

    if use_groups:
        outer_splitter = LeaveOneGroupOut()
        outer_splits = list(outer_splitter.split(X, y, groups))
        if len(outer_splits) == 0:
            use_groups = False
    else:
        outer_splits = []

    if not use_groups:
        min_class = class_counts.min()
        n_splits = max(2, min(config.cv_folds, min_class))
        outer_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.cv_random_state
        )
        outer_splits = list(outer_splitter.split(X, y))

    param_grid = {'svm__' + key: value for key, value in config.svm_param_grid.items()}

    outer_scores = []

    for train_idx, test_idx in outer_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx] if use_groups else None

        pipeline = build_classification_pipeline(config, X.shape[1])

        if use_groups:
            unique_groups = np.unique(groups_train)
            if len(unique_groups) >= 2:
                n_splits = min(config.cv_folds, len(unique_groups))
                inner_splitter = GroupKFold(n_splits=n_splits)
                inner_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=inner_splitter,
                    scoring='accuracy',
                    n_jobs=1
                )
                inner_search.fit(X_train, y_train, groups=groups_train)
            else:
                # 如果训练集中只有一个group，退回到分层交叉验证
                min_class = np.min(np.bincount(y_train))
                if min_class < 2:
                    log("  ⚠️ 外层折跳过: 训练集中某类别样本不足2，无法执行内层交叉验证", config)
                    continue
                n_splits = max(2, min(config.cv_folds, min_class))
                inner_splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=config.cv_random_state + 1
                )
                inner_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=inner_splitter,
                    scoring='accuracy',
                    n_jobs=1
                )
                inner_search.fit(X_train, y_train)
        else:
            min_class = np.min(np.bincount(y_train))
            if min_class < 2:
                log("  ⚠️ 外层折跳过: 训练集中某类别样本不足2，无法执行内层交叉验证", config)
                continue
            n_splits = max(2, min(config.cv_folds, min_class))
            inner_splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.cv_random_state + 1
            )
            inner_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_splitter,
                scoring='accuracy',
                n_jobs=1
            )
            inner_search.fit(X_train, y_train)

        best_model = inner_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        outer_scores.append(test_score)

    if len(outer_scores) == 0:
        log("  ⚠️ 外层交叉验证未产生有效评分，跳过该ROI", config)
        return None

    cv_scores = np.array(outer_scores)

    return {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'p_value_permutation': np.nan,
        'null_distribution': np.array([]),
        'chance_level': 0.5,
        'n_features': X.shape[1],
        'n_samples': X.shape[0]
    }


@enhanced_error_handling("分类数据准备")
def prepare_classification_data(trial_info, beta_images, cond1, cond2, config):
    """准备分类数据"""
    # 筛选特定条件的trial - 使用original_condition列
    condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'
    cond1_trials = trial_info[trial_info[condition_col] == cond1]
    cond2_trials = trial_info[trial_info[condition_col] == cond2]

    if len(cond1_trials) == 0 or len(cond2_trials) == 0:
        log(f"条件数据不足: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)}", config)
        return None, None, None

    log(f"准备分类数据: {cond1} vs {cond2} (总beta文件: {len(beta_images)}, trial数: {len(trial_info)})", config)

    X_indices = []
    y_labels = []
    trial_details = []

    # 统一的索引获取方式
    def get_trial_index(trial):
        if 'combined_trial_index' in trial and 0 <= trial['combined_trial_index'] < len(beta_images):
            return trial['combined_trial_index']
        elif 'trial_index' in trial and 0 <= trial['trial_index'] - 1 < len(beta_images):
            return trial['trial_index'] - 1
        else:
            return None

    # 条件1的trial
    for i, (_, trial) in enumerate(cond1_trials.iterrows()):
        trial_idx = get_trial_index(trial)

        if trial_idx is not None and trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(0)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': cond1,
                'run': trial.get('run', 'unknown'),
                'beta_path': beta_images[trial_idx]
            }
            trial_details.append(trial_detail)

            # 只记录前3个trial的详细对应关系
            if i < 3:
                log(f"  {cond1}[{i + 1}]: trial_index={trial['trial_index']}, 映射索引={trial_idx}, "
                    f"run={trial.get('run', 'unknown')}", config)
        else:
            log(f"  警告: trial_idx={trial_idx} 超出beta_images范围 (len={len(beta_images)})", config)

    # 条件2的trial
    for i, (_, trial) in enumerate(cond2_trials.iterrows()):
        trial_idx = get_trial_index(trial)

        if trial_idx is not None and trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(1)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': cond2,
                'run': trial.get('run', 'unknown'),
                'beta_path': beta_images[trial_idx]
            }
            trial_details.append(trial_detail)

            # 只记录前3个trial的详细对应关系
            if i < 3:
                log(f"  {cond2}[{i + 1}]: trial_index={trial['trial_index']}, 映射索引={trial_idx}, "
                    f"run={trial.get('run', 'unknown')}", config)
        else:
            log(f"  警告: trial_idx={trial_idx} 超出beta_images范围 (len={len(beta_images)})", config)

    if len(X_indices) == 0:
        log(f"没有有效的trial数据用于分类", config)
        return None, None, None

    # 获取对应的beta图像路径
    selected_beta_images = [beta_images[i] for i in X_indices]

    # 汇总日志
    cond1_count = sum(1 for y in y_labels if y == 0)
    cond2_count = sum(1 for y in y_labels if y == 1)
    log(f"分类数据准备完成: {cond1}={cond1_count}, {cond2}={cond2_count}, 总计{len(selected_beta_images)}个beta文件",
        config)

    # 检查是否有重复的beta文件
    unique_betas = set(selected_beta_images)
    if len(unique_betas) != len(selected_beta_images):
        log(f"警告: 发现重复的beta文件! 唯一文件数: {len(unique_betas)}, 总选择数: {len(selected_beta_images)}", config)

    return selected_beta_images, np.array(y_labels), pd.DataFrame(trial_details)


# ============================================================================
# 组水平统计分析模块 (group_statistics.py)
# ============================================================================

def permutation_test_group_level(group_accuracies, n_permutations=1000, random_state=None):
    """组水平置换检验 - 修正版本，解决随机种子问题"""
    # 创建独立的随机状态对象，不影响全局
    rng = np.random.RandomState(random_state)

    # 观察到的组平均准确率
    observed_mean = np.mean(group_accuracies)
    n_subjects = len(group_accuracies)

    # 数据质量检查
    if np.any(group_accuracies > 1.0) or np.any(group_accuracies < 0.0):
        print(
            f"⚠️  警告: 准确率数据超出合理范围 [0,1]: min={np.min(group_accuracies):.3f}, max={np.max(group_accuracies):.3f}")

    if np.mean(group_accuracies) > 0.95:
        print(f"⚠️  警告: 平均准确率过高 ({np.mean(group_accuracies):.3f})，可能存在过拟合或数据泄露")

    # 修正的置换策略：通过对符号进行随机翻转来构建零分布
    null_distribution = []

    # 将数据以0.5为中心
    data_centered = group_accuracies - 0.5

    for _ in range(n_permutations):
        # 随机翻转符号 - 这是正确的置换方法
        random_signs = rng.choice([-1, 1], size=n_subjects, replace=True)
        permuted_accuracies = 0.5 + random_signs * data_centered
        null_distribution.append(np.mean(permuted_accuracies))

    null_distribution = np.array(null_distribution)

    # 计算p值（单尾检验，因为我们预期准确率高于随机水平）
    if observed_mean > 0.5:
        p_value = np.mean(null_distribution >= observed_mean)
    else:
        p_value = np.mean(null_distribution <= observed_mean)

    # 修正：如果p值为0，使用保守估计
    if p_value == 0:
        p_value = 1.0 / (n_permutations + 1)  # 保守估计
        print(f"📊 注意: 使用保守估计 p = 1/(n_permutations+1) = {p_value:.6f}")

    # 添加诊断信息
    extreme_count = np.sum(null_distribution >= observed_mean) if observed_mean > 0.5 else np.sum(
        null_distribution <= observed_mean)
    print(
        f"📈 置换检验诊断: 观察值={observed_mean:.4f}, 零分布范围=[{np.min(null_distribution):.4f}, {np.max(null_distribution):.4f}], 极端值数量={extreme_count}/{n_permutations}")

    return p_value, null_distribution


def perform_group_statistics_corrected(group_df, config):
    """执行修正的组水平统计分析"""
    log("开始组水平统计分析", config)

    stats_results = []

    # 按对比和ROI分组
    for (contrast, roi), group_data in group_df.groupby(['contrast', 'roi']):
        accuracies = group_data['mean_accuracy'].values
        n_subjects = len(accuracies)

        if n_subjects < 3:
            log(f"跳过 {contrast}-{roi}: 被试数量不足 (n={n_subjects})", config)
            continue

        # 基本统计量
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        sem_acc = std_acc / np.sqrt(n_subjects)

        # 数据质量检查
        if np.any(accuracies > 1.0) or np.any(accuracies < 0.0):
            log(f"警告 {contrast}-{roi}: 准确率数据超出合理范围 [0,1]", config)
            accuracies = np.clip(accuracies, 0.0, 1.0)  # 限制在合理范围内

        # 单样本t检验（vs 0.5随机水平）
        t_stat, t_p_value = stats.ttest_1samp(accuracies, 0.5)

        # 计算单尾t检验p值
        t_p_value_one_tailed = t_p_value / 2 if mean_acc > 0.5 else 1 - t_p_value / 2

        # 修正：如果t检验p值为0，使用更精确的计算
        if t_p_value == 0:
            # 使用t分布的累积分布函数
            from scipy.stats import t as t_dist
            t_p_value_precise = 2 * (1 - t_dist.cdf(abs(t_stat), n_subjects - 1))
            t_p_value_one_tailed_precise = 1 - t_dist.cdf(abs(t_stat),
                                                          n_subjects - 1) if mean_acc > 0.5 else t_dist.cdf(
                -abs(t_stat), n_subjects - 1)
            log(f"使用精确t检验计算: {contrast}-{roi}, 精确双尾p={t_p_value_precise:.2e}, 精确单尾p={t_p_value_one_tailed_precise:.2e}",
                config)
            t_p_value = t_p_value_precise
            t_p_value_one_tailed = t_p_value_one_tailed_precise

        # Cohen's d效应量
        cohens_d = (mean_acc - 0.5) / std_acc

        # 95%置信区间
        ci_margin = stats.t.ppf(0.975, n_subjects - 1) * sem_acc
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin

        # 为每个ROI-对比组合生成不同的随机种子
        roi_contrast_seed = hash(f"{contrast}_{roi}") % (2 ** 31) + config.permutation_random_state

        # 组水平置换检验（现在使用单尾检验）
        p_permutation, null_dist = permutation_test_group_level(
            accuracies,
            n_permutations=config.n_permutations,
            random_state=roi_contrast_seed
        )

        # 效应量分类
        effect_size_category = "小" if abs(cohens_d) < 0.2 else "中" if abs(cohens_d) < 0.8 else "大"

        result = {
            'contrast': contrast,
            'roi': roi,
            'n_subjects': n_subjects,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'sem_accuracy': sem_acc,
            't_statistic': t_stat,
            'p_value_ttest_two_tailed': t_p_value,  # 双尾t检验p值
            'p_value_ttest_one_tailed': t_p_value_one_tailed,  # 单尾t检验p值
            'p_value_permutation': p_permutation,  # 单尾置换检验p值
            'cohens_d': cohens_d,
            'effect_size_category': effect_size_category,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'null_distribution_mean': np.mean(null_dist),
            'null_distribution_std': np.std(null_dist),
            'null_distribution_range': f"[{np.min(null_dist):.4f}, {np.max(null_dist):.4f}]"
        }

        stats_results.append(result)

        # 判断显著性（基于置换检验）
        is_significant = p_permutation < config.alpha_level

        # 详细的结果报告
        log(f"\n=== 组统计结果 {contrast}-{roi} ===", config)
        log(f"📊 基本统计: 平均准确率={mean_acc:.4f} (±{std_acc:.4f}), n={n_subjects}", config)
        log(f"📈 t检验: t({n_subjects - 1})={t_stat:.3f}, 双尾p={t_p_value:.2e}, 单尾p={t_p_value_one_tailed:.2e}",
            config)
        log(f"🔄 置换检验: p={p_permutation:.6f} (基于{config.n_permutations}次置换)", config)
        log(f"📏 效应量: Cohen's d={cohens_d:.3f} ({effect_size_category}效应)", config)
        log(f"🎯 95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]", config)
        log(f"🔍 零分布: 均值={np.mean(null_dist):.4f}, 范围={result['null_distribution_range']}", config)
        log(f"✅ 显著性 (α={config.alpha_level}): {'显著' if is_significant else '不显著'}", config)

        # 质量诊断
        if mean_acc > 0.95:
            log(f"⚠️  质量警告: 准确率过高，建议检查是否存在过拟合或数据泄露", config)
        if p_permutation == 1.0 / (config.n_permutations + 1):
            log(f"📊 注意: 使用了保守p值估计，建议增加置换次数", config)

    # 转换为DataFrame
    stats_df = pd.DataFrame(stats_results)

    if stats_df.empty:
        log("警告：没有生成任何统计结果", config)
        return stats_df

    # 使用置换检验p值作为最终显著性判断（单尾检验）
    stats_df['significant'] = stats_df['p_value_permutation'] < config.alpha_level

    # 报告结果 - 明确说明使用单尾检验
    significant_results = stats_df[stats_df['significant']]

    log(f"\n组水平置换检验显著结果数(单尾检验): {len(significant_results)}/{len(stats_df)}", config)

    if not significant_results.empty:
        log("\n显著结果(单尾置换检验):", config)
        for _, result in significant_results.iterrows():
            log(f"  {result['contrast']} - {result['roi']}: "
                f"accuracy = {result['mean_accuracy']:.3f}, "
                f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                f"p_t(单尾) = {result['p_value_ttest_one_tailed']:.4f}, "
                f"p_permutation(单尾) = {result['p_value_permutation']:.4f}, "
                f"d = {result['cohens_d']:.3f}", config)
    else:
        log("\n无显著结果", config)

    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results_corrected.csv", index=False)

    return stats_df
# ============================================================================
# 主分析流程
# ============================================================================

def run_group_roi_analysis_corrected(config):
    """运行组水平ROI分析（修正版本）"""
    log("开始组水平ROI分析", config)

    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")

    # 验证配置
    if not validate_config(config):
        log("配置验证失败，请检查配置参数", config)
        return None, None

    # 运行维度问题诊断
    diagnose_dimension_issues(config)

    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    if not roi_masks:
        raise ValueError("未找到有效的ROI mask文件")

    log(f"成功加载 {len(roi_masks)} 个ROI masks", config)

    # 初始化结果存储
    all_results = []

    # 对每个被试进行分析
    for subject_id in config.subjects:
        log(f"\n处理被试: {subject_id}", config)

        try:
            # 调试模式：对前几个被试进行详细调试
            if config.debug_mode and subject_id in config.debug_subjects:
                beta_images, trial_info = debug_data_pipeline(subject_id, config)
            else:
                # 正常加载数据
                beta_images, trial_info = load_multi_run_lss_data(subject_id, config.runs, config)

            if beta_images is None or trial_info is None:
                log(f"被试 {subject_id} 的LSS数据加载失败，跳过", config)
                continue

            # 检查数据泄露
            if not check_data_leakage(beta_images, trial_info, config, subject_id):
                log(f"被试 {subject_id} 数据泄露检查失败，跳过", config)
                continue

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

                # 对每个ROI进行分类分析
                for roi_name, roi_mask_path in roi_masks.items():
                    try:
                        # 提取ROI特征（带特征选择）
                        X = extract_roi_features_with_selection(selected_betas, roi_mask_path, config, labels)

                        if X is None or X.shape[1] == 0:
                            log(f"  {roi_name}: 特征提取失败", config)
                            continue

                        # 如果特征选择后样本仍然不足，跳过
                        if X.shape[0] < 10:  # 至少需要10个样本
                            log(f"  {roi_name}: 样本数量不足({X.shape[0]})，跳过", config)
                            continue

                        groups = trial_details['run'].fillna('unknown').astype(str).values

                        # 验证分类器性能
                        cv_scores, feature_ratio = validate_classifier_performance(
                            X, labels, groups, config, roi_name, contrast_name
                        )

                        # 调试模式：运行置换测试验证
                        if config.debug_mode and subject_id in config.debug_subjects:
                            true_mean, perm_scores, p_value = run_permutation_test_debug(
                                X, labels, config, n_permutations=config.n_permutations_debug
                            )

                        # 运行正式分类分析
                        result = run_roi_classification_enhanced(X, labels, config, groups=groups)

                        if result is not None:
                            result.update({
                                'subject': subject_id,
                                'roi': roi_name,
                                'contrast': contrast_name,
                                'condition1': cond1,
                                'condition2': cond2,
                                'n_trials_cond1': np.sum(labels == 0),
                                'n_trials_cond2': np.sum(labels == 1),
                                'feature_ratio': feature_ratio,
                                'strict_cv_mean': np.mean(cv_scores) if 'cv_scores' in locals() else np.nan,
                                'strict_cv_std': np.std(cv_scores) if 'cv_scores' in locals() else np.nan
                            })
                            all_results.append(result)

                    except Exception as e:
                        log(f"      {roi_name} 分析出错: {str(e)}", config)
                        continue

            # 内存清理
            memory_cleanup()

        except Exception as e:
            log(f"被试 {subject_id} 处理失败: {str(e)}", config)
            continue

    # 如果有结果，进行组水平分析
    if all_results:
        log(f"\n开始组水平统计分析，共 {len(all_results)} 个结果", config)

        # 转换为DataFrame
        group_df = pd.DataFrame(all_results)

        # 保存个体结果
        group_df.to_csv(config.results_dir / "individual_roi_mvpa_results.csv", index=False)

        # 执行组水平统计分析
        stats_df = perform_group_statistics_corrected(group_df, config)

        # 保存组水平统计结果
        stats_df.to_csv(config.results_dir / "group_roi_mvpa_statistics.csv", index=False)

        # 创建可视化
        create_enhanced_visualizations(group_df, stats_df, config)

        # 生成HTML报告
        main_viz_b64 = image_to_base64(config.results_dir / "main_visualizations.png")
        individual_viz_b64 = image_to_base64(config.results_dir / "individual_roi_plots.png")
        generate_html_report(group_df, stats_df, config, main_viz_b64=main_viz_b64,
                             individual_viz_b64=individual_viz_b64)

        log(f"\n分析完成！结果保存在: {config.results_dir}", config)
        log(f"分析被试数: {len(set(group_df['subject']))}", config)
        log(f"ROI数量: {len(roi_masks)}", config)
        log(f"对比数量: {len(config.contrasts)}", config)

        return group_df, stats_df
    else:
        log("没有有效的分析结果", config)
        return None, None


def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()

    try:
        # 运行分析
        group_df, stats_df = run_group_roi_analysis_corrected(config)
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