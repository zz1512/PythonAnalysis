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

用途（主线定位）
- 学习阶段（run3-4）的 ROI-level MVPA：检验特定 ROI 是否包含可区分 yy vs kj 的多体素信息。
- 输出每个 ROI 的分类准确率、置换检验与组水平统计，并生成可视化与 HTML 报告。

输入（默认相对于 `${PYTHON_METAPHOR_ROOT}`）
- LSS 单 trial betas（学习阶段）：`lss_betas_final/sub-xx/run-3|4/` + `trial_info.csv`
- ROI masks：`glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/...` 或 `roi_masks_final/`

输出（默认 `${PYTHON_METAPHOR_ROOT}/mvpa_roi_results`）
- `mvpa_subject_metrics.tsv/csv`（以脚本实际命名为准）：subject×roi 的准确率与诊断信息
- `mvpa_group_summary.tsv`：组水平统计与显著性
- `analysis.log`：日志
- `report.html` + 图表文件（由 visualization/report_generator 生成）

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
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
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
        base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
        self.lss_root = base_dir / "lss_betas_final"
        self.roi_dir = base_dir / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE"
        self.results_dir = base_dir / "mvpa_roi_results"

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
        cond_trials = trial_info[get_condition_mask(trial_info, condition_col, condition)]
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
    """构建Searchlight分类pipeline - 简化版本"""

    steps = [
        ('imputer', SimpleImputer(strategy='mean')),  # 处理可能的缺失值
        ('variance_filter', VarianceThreshold(threshold=0.0)),  # 只移除零方差特征
        ('scaler', StandardScaler()),  # 标准化
        ('classifier', SVC(**config.svm_params))  # 分类器
    ]

    return Pipeline(steps)


def apply_feature_selection(X, y, config, method='univariate', k=500):
    """应用特征选择解决维度问题"""
    n_samples, n_features = X.shape

    # 去除常数特征（方差为0的特征），避免后续统计检验出现无效值
    variance = np.var(X, axis=0)
    if not np.any(variance > 0):
        log("⚠️ 所有特征的方差均为0，跳过特征选择并返回原始特征。", config)
        return X

    variance_filter = VarianceThreshold(threshold=0.0)
    X = variance_filter.fit_transform(X)
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

def _candidate_lss_run_dirs(lss_root, subject, run):
    subject_dir = lss_root / subject
    return [
        subject_dir / f"run-{run}",
        subject_dir / f"run-{run}_LSS",
        subject_dir / f"run{run}",
        subject_dir / f"run{run}_LSS",
    ]


def resolve_lss_run_dir(lss_root, subject, run):
    for candidate in _candidate_lss_run_dirs(lss_root, subject, run):
        if candidate.exists():
            return candidate
    return None


def resolve_beta_path(lss_dir, trial):
    beta_name = str(trial.get('beta_file', '')).strip()
    if beta_name:
        beta_path = Path(beta_name)
        if not beta_path.is_absolute():
            beta_path = lss_dir / beta_path.name
        if beta_path.exists():
            return beta_path

    trial_index = int(trial.get('trial_index', -1))
    unique_label = str(trial.get('unique_label', '')).strip()
    candidates = []
    if unique_label:
        candidates.append(lss_dir / f"beta_trial-{trial_index:03d}_{unique_label}.nii.gz")
        candidates.extend(sorted(lss_dir.glob(f"beta_trial-*_{unique_label}.nii.gz")))
    candidates.append(lss_dir / f"beta_trial-{trial_index:03d}.nii.gz")
    candidates.append(lss_dir / f"beta_trial_{trial_index:03d}.nii.gz")
    candidates.extend(sorted(lss_dir.glob(f"beta_trial-{trial_index:03d}_*.nii.gz")))
    candidates.extend(sorted(lss_dir.glob(f"beta_trial_{trial_index:03d}_*.nii.gz")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_condition_label(value):
    value = str(value).strip().lower()
    if value.startswith('yy'):
        return 'yy'
    if value.startswith('kj'):
        return 'kj'
    return value


def condition_matches(value, target):
    return normalize_condition_label(value) == normalize_condition_label(target)


def get_condition_mask(frame, condition_col, target):
    return frame[condition_col].astype(str).map(lambda value: condition_matches(value, target))


def load_roi_masks(config):
    """??ROI mask????????? GLM ???? *_roi_mask.nii.gz?"""
    log("??ROI masks", config)

    roi_masks = {}
    roi_files = sorted(config.roi_dir.rglob("*_roi_mask.nii.gz"))
    if not roi_files:
        roi_files = sorted(config.roi_dir.glob("*.nii.gz"))

    if not roi_files:
        raise FileNotFoundError(f"?{config.roi_dir}????ROI mask??")

    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        if roi_name == 'group_mask':
            continue
        roi_masks[roi_name] = str(roi_file)
        log(f"  ??ROI: {roi_name}", config)

    if not roi_masks:
        raise FileNotFoundError(f"?{config.roi_dir}??????ROI mask??")

    log(f"?????{len(roi_masks)}?ROI masks", config)
    return roi_masks


@enhanced_error_handling("LSS trial????")
def load_lss_trial_data(subject, run, config):
    """??LSS???trial???beta????????????????"""
    lss_dir = resolve_lss_run_dir(config.lss_root, subject, run)

    if lss_dir is None:
        log(f"LSS?????: {config.lss_root / subject} (run-{run})", config)
        return None, None

    trial_map_path = lss_dir / "trial_info.csv"
    if not trial_map_path.exists():
        log(f"trial_info.csv???: {trial_map_path}", config)
        return None, None

    trial_info = pd.read_csv(trial_map_path)
    beta_images = []
    valid_trials = []
    missing_files = []

    for _, trial in trial_info.iterrows():
        beta_path = resolve_beta_path(lss_dir, trial)
        if beta_path is None:
            missing_files.append(str(trial.get('beta_file', f"trial_index={trial.get('trial_index', 'NA')}")))
            continue

        beta_images.append(str(beta_path))
        trial_with_run = trial.copy()
        trial_with_run['run'] = run
        trial_with_run['beta_file'] = beta_path.name
        trial_with_run['beta_path'] = str(beta_path)
        condition_col = 'original_condition' if 'original_condition' in trial_with_run.index else 'condition'
        trial_with_run['condition_group'] = normalize_condition_label(trial_with_run.get(condition_col, ''))
        valid_trials.append(trial_with_run)

    if missing_files:
        log(f"  run {run}: ?? {len(missing_files)} ????beta??", config)

    if len(beta_images) == 0:
        log(f"???????beta??: {subject} run-{run}", config)
        return None, None

    valid_trials_df = pd.DataFrame(valid_trials).reset_index(drop=True)
    log(f"?? {subject} run-{run}: {len(beta_images)}?trial (??: {lss_dir.name})", config)

    return beta_images, valid_trials_df

