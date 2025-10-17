#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI Searchlight-based MVPA分析流水线 (增强版)

功能描述:
    基于searchlight方法的多体素模式分析(MVPA)，用于隐喻-空间认知的神经解码研究。
    相比ROI方法，searchlight在每个体素周围定义球形搜索区域，提供更精细的空间定位。

主要功能:
    1. LSS trial-wise beta系数加载和预处理
    2. Searchlight球形区域定义和特征提取
    3. SVM分类器训练和交叉验证
    4. 置换检验和统计显著性测试
    5. 全脑分类准确率图生成和可视化
    6. 多重比较校正和阈值处理

分析流程:
    数据加载 → Searchlight定义 → 特征提取 → SVM分类 → 置换检验 → 统计校正 → 结果可视化

输入数据:
    - LSS分析结果: {lss_root}/{subject}/run-{run}_LSS/beta_trial_*.nii.gz
    - Trial信息: {lss_root}/{subject}/run-{run}_LSS/trial_info.csv
    - 脑掩模: {mask_path} (可选，用于限制搜索范围)

输出结果:
    - 分类准确率图: {results_dir}/accuracy_maps/
    - 统计显著性图: {results_dir}/statistical_maps/
    - 组水平分析结果: {results_dir}/group_searchlight_results.csv
    - HTML分析报告: {results_dir}/searchlight_analysis_report.html

作者: AI Assistant
创建时间: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.decoding import SearchLight
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 配置参数类 ==========
class SearchlightMVPAConfig:
    """Searchlight MVPA分析配置参数"""
    
    def __init__(self):
        # ========== 被试和实验设置 ==========
        # 匹配fmri_lss_enhance.py的设置
        self.subjects = [f"sub-{i:02d}" for i in range(1, 29)]  # 被试范围: sub-01 到 sub-28
        self.runs = [3, 4]  # run编号列表，匹配LSS分析的runs
        
        # ========== 路径配置 ==========
        # 采用相对路径配置，匹配LSS输出结构
        self.lss_root = Path(r"../../learn_LSS")  # LSS分析结果目录，匹配fmri_lss_enhance.py的OUTPUT_ROOT
        self.results_dir = Path(r"../../../results/searchlight_mvpa")  # 结果输出目录
        self.mask_path = None  # 脑掩模路径(可选，None表示使用全脑)
        
        # ========== 常用语言处理脑区掩模路径 ==========
        self.mask_dir = Path(r"../../../data/masks")
        
        self.language_masks = {
            'broca': self.mask_dir / 'broca_area.nii.gz',  # 布洛卡区(语言产生)
            'wernicke': self.mask_dir / 'wernicke_area.nii.gz',  # 韦尼克区(语言理解)
            'angular_gyrus': self.mask_dir / 'angular_gyrus.nii.gz',  # 角回(语义处理)
            'inferior_frontal': self.mask_dir / 'inferior_frontal_gyrus.nii.gz',  # 下额回
            'superior_temporal': self.mask_dir / 'superior_temporal_gyrus.nii.gz',  # 上颞回
            'middle_temporal': self.mask_dir / 'middle_temporal_gyrus.nii.gz',  # 中颞回
            'precuneus': self.mask_dir / 'precuneus.nii.gz',  # 楔前叶(空间认知)
            'posterior_parietal': self.mask_dir / 'posterior_parietal_cortex.nii.gz',  # 后顶叶皮层
            'language_network': self.mask_dir / 'language_network_mask.nii.gz',  # 语言网络综合掩模
            'spatial_network': self.mask_dir / 'spatial_network_mask.nii.gz',  # 空间认知网络掩模
            'gray_matter': self.mask_dir / 'gray_matter_mask.nii.gz',  # 灰质层掩模
            'cortical_gray_matter': self.mask_dir / 'cortical_gray_matter_mask.nii.gz'  # 皮层灰质掩模
        }
        
        # ========== Searchlight参数配置 ==========
        self.searchlight_radius = 4.0  # searchlight球形半径(mm)
        self.min_voxels_in_sphere = 10  # 球形区域最小体素数
        self.mask_strategy = 'background'  # 掩模策略: 'background', 'epi', 'template'
        
        # ========== SVM分类器参数网格 ==========
        self.svm_param_grid = {
            'classifier__C': [0.1, 1, 10],  # 正则化参数
            'classifier__kernel': ['linear', 'rbf'],  # 核函数类型
            'classifier__gamma': ['scale', 'auto']  # RBF核参数(仅rbf核使用)
        }
        
        # ========== 交叉验证参数 ==========
        self.cv_folds = 5  # 交叉验证折数
        self.cv_random_state = 42  # 随机种子
        
        # ========== 置换检验参数 ==========
        self.n_permutations = 100  # 置换检验次数
        self.permutation_random_state = 42  # 置换随机种子
        
        # ========== 多重比较校正参数 ==========
        self.correction_method = 'fdr_bh'  # 校正方法: 'fdr_bh', 'bonferroni', 'holm'
        self.alpha_level = 0.05  # 显著性水平
        
        # ========== 并行处理参数 ==========
        self.n_jobs = min(4, cpu_count())  # 并行进程数
        self.use_parallel = True  # 是否使用并行处理
        
        # ========== 内存优化参数 ==========
        self.memory_cache = None  # 内存缓存目录(None表示不缓存)
        self.memory_level = 1  # 缓存级别
        self.batch_size = 50  # 批处理大小
        
        # ========== 分类对比设置 ==========
        # 格式: [(对比名称, 条件1, 条件2), ...]
        # 可根据实验设计扩展更多对比条件
        self.contrasts = [
            ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间条件对比
        ]
        
        # ========== 掩模选择策略 ==========
        self.mask_selection = 'whole_brain'  # 'whole_brain', 'language_network', 'spatial_network', 或具体掩模名称
        self.use_custom_mask = False  # 是否使用自定义掩模
        
        # ========== 多掩模分析支持 ==========
        self.mask_list = None  # mask列表，用于多mask分析，格式: ['gray_matter', 'language_network']
        self.use_mask_list = False  # 是否使用mask列表进行多mask分析
        
        # 创建输出目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "accuracy_maps").mkdir(exist_ok=True)
        (self.results_dir / "statistical_maps").mkdir(exist_ok=True)
        (self.results_dir / "subject_results").mkdir(exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
    
    def get_selected_mask(self):
        """根据配置返回选定的掩模路径或掩模路径列表"""
        # 如果使用mask列表
        if self.use_mask_list and self.mask_list:
            mask_paths = []
            for mask_name in self.mask_list:
                if mask_name == 'whole_brain':
                    mask_paths.append(None)  # None表示全脑
                elif mask_name in self.language_masks:
                    mask_path = self.language_masks[mask_name]
                    if mask_path.exists():
                        mask_paths.append(str(mask_path))
                    else:
                        log(f"警告: 掩模文件不存在 {mask_path}, 跳过该mask")
                else:
                    log(f"警告: 未知的掩模选择 {mask_name}, 跳过该mask")
            return mask_paths if mask_paths else [None]  # 如果没有有效mask，返回全脑
        
        # 单个mask的原有逻辑
        if not self.use_custom_mask or self.mask_selection == 'whole_brain':
            return None
        elif self.mask_selection in self.language_masks:
            mask_path = self.language_masks[self.mask_selection]
            if mask_path.exists():
                return str(mask_path)
            else:
                log(f"警告: 掩模文件不存在 {mask_path}, 使用全脑分析")
                return None
        else:
            log(f"警告: 未知的掩模选择 {self.mask_selection}, 使用全脑分析")
            return None
    
    def validate_mask_availability(self):
        """验证所选mask或mask列表的可用性"""
        # 如果使用mask列表
        if self.use_mask_list and self.mask_list:
            valid_masks = []
            invalid_masks = []
            
            for mask_name in self.mask_list:
                if mask_name == 'whole_brain':
                    valid_masks.append(f"{mask_name}(全脑分析)")
                elif mask_name in self.language_masks:
                    mask_path = self.language_masks[mask_name]
                    if mask_path.exists():
                        valid_masks.append(f"{mask_name}({mask_path})")
                    else:
                        invalid_masks.append(f"{mask_name}(文件不存在: {mask_path})")
                else:
                    invalid_masks.append(f"{mask_name}(未知mask)")
            
            if not valid_masks:
                return False, f"所有mask都无效: {invalid_masks}"
            elif invalid_masks:
                return True, f"部分mask有效: {valid_masks}, 无效: {invalid_masks}"
            else:
                return True, f"所有mask都有效: {valid_masks}"
        
        # 单个mask的原有逻辑
        if not self.use_custom_mask or self.mask_selection == 'whole_brain':
            return True, "全脑分析"
        
        if self.mask_selection not in self.language_masks:
            return False, f"未知的掩模选择: {self.mask_selection}"
        
        mask_path = self.language_masks[self.mask_selection]
        if not mask_path.exists():
            return False, f"掩模文件不存在: {mask_path}"
        
        return True, f"使用掩模: {mask_path}"
    
    def get_available_masks(self):
        """获取所有可用的mask列表"""
        available_masks = ['whole_brain']  # 全脑分析总是可用的
        
        for mask_name, mask_path in self.language_masks.items():
            if mask_path.exists():
                available_masks.append(mask_name)
        
        return available_masks
    
    def set_mask(self, mask_name_or_list):
        """设置mask选择的便捷方法，支持单个mask或mask列表"""
        # 如果输入是列表，使用mask列表模式
        if isinstance(mask_name_or_list, (list, tuple)):
            self.set_mask_list(mask_name_or_list)
        else:
            # 单个mask的原有逻辑
            mask_name = mask_name_or_list
            if mask_name == 'whole_brain':
                self.mask_selection = 'whole_brain'
                self.use_custom_mask = False
                self.use_mask_list = False
                self.mask_list = None
            elif mask_name in self.language_masks:
                self.mask_selection = mask_name
                self.use_custom_mask = True
                self.use_mask_list = False
                self.mask_list = None
            else:
                raise ValueError(f"未知的mask名称: {mask_name}. 可用选项: {self.get_available_masks()}")
    
    def set_mask_list(self, mask_list):
        """设置mask列表进行多mask分析"""
        if not isinstance(mask_list, (list, tuple)):
            raise ValueError("mask_list必须是列表或元组")
        
        if not mask_list:
            raise ValueError("mask_list不能为空")
        
        # 验证所有mask名称
        available_masks = self.get_available_masks()
        invalid_masks = [mask for mask in mask_list if mask not in available_masks]
        if invalid_masks:
            raise ValueError(f"无效的mask名称: {invalid_masks}. 可用选项: {available_masks}")
        
        # 设置mask列表配置
        self.mask_list = list(mask_list)
        self.use_mask_list = True
        self.use_custom_mask = True  # mask列表模式下也算使用自定义mask
        self.mask_selection = f"mask_list_{len(mask_list)}"  # 用于标识
    
    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

# ========== 核心工具函数 ==========
def log(message, config=None):
    """
    带时间戳的日志记录函数
    
    参数:
        message (str): 日志消息内容
        config (SearchlightMVPAConfig, optional): 配置对象
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    logging.info(message)

def memory_cleanup():
    """内存清理函数"""
    gc.collect()

def load_lss_trial_data(subject, run, config):
    """
    加载LSS分析的trial数据和beta图像
    
    参数:
        subject (str): 被试ID (如 'sub-01')
        run (int): run编号
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        tuple: (beta_images列表, trial_info数据框) 或 (None, None)
    
    注意:
        - 匹配fmri_lss_enhance.py的输出结构: {subject}/run-{run}_LSS/
        - 检查LSS目录和trial_info.csv文件是否存在
        - 验证beta图像文件完整性
        - 添加run信息到trial数据中
    
    示例:
        >>> beta_imgs, trial_df = load_lss_trial_data('sub-01', 3, config)
        >>> print(f"加载了 {len(beta_imgs)} 个trial")
    """
    # 匹配LSS输出目录结构: OUTPUT_ROOT / subject / f"run-{run}_LSS"
    lss_dir = config.lss_root / subject / f"run-{run}_LSS"
    
    if not lss_dir.exists():
        log(f"LSS目录不存在: {lss_dir}", config)
        return None, None
    
    # 加载trial信息文件
    trial_info_path = lss_dir / "trial_info.csv"
    if not trial_info_path.exists():
        log(f"trial_info.csv不存在: {trial_info_path}", config)
        return None, None
    
    try:
        trial_info = pd.read_csv(trial_info_path)
        log(f"成功读取trial_info.csv: {len(trial_info)}条记录", config)
    except Exception as e:
        log(f"读取trial_info.csv失败: {e}", config)
        return None, None
    
    # 加载所有beta图像，匹配LSS的命名格式: beta_trial_XXX.nii.gz
    beta_images = []
    valid_trials = []
    
    for _, trial in trial_info.iterrows():
        # LSS输出的beta文件命名: beta_trial_{trial_index:03d}.nii.gz
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        if beta_path.exists():
            beta_images.append(str(beta_path))
            # 添加run信息到trial数据中
            trial_with_run = trial.copy()
            trial_with_run['run'] = run
            trial_with_run['subject'] = subject
            trial_with_run['global_trial_index'] = f"{subject}_run{run}_trial{trial['trial_index']}"
            trial_with_run['beta_file_path'] = str(beta_path)
            valid_trials.append(trial_with_run)
        else:
            log(f"Beta图像缺失: {beta_path}", config)
    
    if len(beta_images) == 0:
        log(f"没有找到有效的beta图像: {subject} run-{run}", config)
        return None, None
    
    valid_trials_df = pd.DataFrame(valid_trials)
    log(f"成功加载 {subject} run-{run}: {len(beta_images)}个trial (条件: {valid_trials_df['original_condition'].unique()})", config)
    
    return beta_images, valid_trials_df

def load_multi_run_lss_data(subject, runs, config):
    """
    加载并合并多个run的LSS trial数据
    
    参数:
        subject (str): 被试ID
        runs (list): run编号列表
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        tuple: (合并的beta_images列表, 合并的trial_info数据框)
    """
    log(f"开始加载 {subject} 的多run数据: runs {runs}", config)
    
    all_beta_images = []
    all_trial_info = []
    
    for run in runs:
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        if beta_images is not None and trial_info is not None:
            all_beta_images.extend(beta_images)
            all_trial_info.append(trial_info)
            log(f"  成功加载 run-{run}: {len(beta_images)}个trial", config)
        else:
            log(f"  跳过 run-{run}: 数据加载失败", config)
    
    if not all_trial_info:
        log(f"错误: {subject} 没有成功加载任何run数据", config)
        return None, None
    
    # 合并所有trial信息
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    
    log(f"{subject} 多run数据加载完成: 总计 {len(all_beta_images)}个trial", config)
    return all_beta_images, combined_trial_info

# ========== Searchlight分析函数 ==========
def prepare_classification_data(trial_info, beta_images, cond1, cond2, config):
    """
    准备分类分析的数据
    
    参数:
        trial_info (DataFrame): trial信息，包含LSS输出的字段
        beta_images (list): beta图像路径列表（兼容性保留，实际使用trial_info中的路径）
        cond1, cond2 (str): 两个对比条件
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        tuple: (选中的beta图像列表, 标签数组, trial详细信息)
    
    注意:
        - 适配LSS输出格式，使用'original_condition'字段
        - 使用trial_info中的'beta_file_path'字段获取beta图像路径
    """
    # 筛选指定条件的trial，使用LSS的字段名'original_condition'
    cond1_trials = trial_info[trial_info['original_condition'] == cond1]
    cond2_trials = trial_info[trial_info['original_condition'] == cond2]
    
    if len(cond1_trials) == 0 or len(cond2_trials) == 0:
        available_conditions = trial_info['original_condition'].unique()
        log(f"警告: 条件 {cond1} 或 {cond2} 没有有效trial。可用条件: {available_conditions}", config)
        return None, None, None
    
    # 检查样本数量平衡
    min_samples = min(len(cond1_trials), len(cond2_trials))
    if min_samples < 5:
        log(f"警告: 样本数量过少 ({min_samples}), 跳过分析", config)
        return None, None, None
    
    # 平衡样本数量(可选)
    if len(cond1_trials) != len(cond2_trials):
        log(f"样本不平衡: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)}", config)
        # 可以选择随机下采样到相同数量
        # cond1_trials = cond1_trials.sample(n=min_samples, random_state=42)
        # cond2_trials = cond2_trials.sample(n=min_samples, random_state=42)
    
    # 合并trial信息
    selected_trials = pd.concat([cond1_trials, cond2_trials], ignore_index=True)
    
    # 获取对应的beta图像路径，使用LSS输出的beta_file_path字段
    selected_betas = []
    for _, trial in selected_trials.iterrows():
        if 'beta_file_path' in trial and pd.notna(trial['beta_file_path']):
            # 使用LSS输出的完整路径
            selected_betas.append(trial['beta_file_path'])
        else:
            # 备用方案：根据trial_index构建路径
            trial_idx = trial['trial_index'] - 1  # 转换为0-based索引
            if 0 <= trial_idx < len(beta_images):
                selected_betas.append(beta_images[trial_idx])
            else:
                log(f"警告: 无法获取trial的beta图像路径: trial_index={trial.get('trial_index', 'N/A')}", config)
    
    # 验证beta图像数量
    if len(selected_betas) != len(selected_trials):
        log(f"警告: beta图像数量({len(selected_betas)})与trial数量({len(selected_trials)})不匹配", config)
        return None, None, None
    
    # 创建标签
    labels = np.array([0] * len(cond1_trials) + [1] * len(cond2_trials))
    
    log(f"成功准备分类数据: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)}, 总计{len(selected_betas)}个beta图像", config)
    
    return selected_betas, labels, selected_trials

def run_searchlight_classification(functional_imgs, labels, config, mask_img=None):
    """
    运行searchlight分类分析
    
    参数:
        functional_imgs (list): 功能图像路径列表
        labels (array): 分类标签
        config (SearchlightMVPAConfig): 配置对象
        mask_img (str, optional): 脑掩模路径
    
    返回:
        Nifti1Image: 分类准确率图
    """
    try:
        # 如果没有提供mask_img，使用配置中的掩模选择
        if mask_img is None:
            mask_img = config.get_selected_mask()
            
        log(f"使用掩模: {mask_img if mask_img else '全脑分析'}", config)
        
        # 创建分类器pipeline
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', C=1.0, random_state=config.cv_random_state))
        ])
        
        # 创建交叉验证对象
        cv = StratifiedKFold(
            n_splits=config.cv_folds, 
            shuffle=True, 
            random_state=config.cv_random_state
        )
        
        # 创建SearchLight对象
        searchlight = SearchLight(
            mask_img=mask_img,
            radius=config.searchlight_radius,
            estimator=classifier,
            cv=cv,
            scoring='accuracy',
            n_jobs=config.n_jobs,
            verbose=1
        )
        
        log(f"开始searchlight分析: 半径={config.searchlight_radius}mm, CV={config.cv_folds}折", config)
        
        # 运行searchlight分析
        searchlight.fit(functional_imgs, labels)
        
        log(f"Searchlight分析完成", config)
        
        return searchlight.scores_
        
    except Exception as e:
        log(f"Searchlight分析失败: {e}", config)
        return None

def run_permutation_test_searchlight(functional_imgs, labels, config, mask_img=None, n_permutations=100):
    """
    运行searchlight置换检验
    
    参数:
        functional_imgs (list): 功能图像路径列表
        labels (array): 分类标签
        config (SearchlightMVPAConfig): 配置对象
        mask_img (str, optional): 脑掩模路径
        n_permutations (int): 置换次数
    
    返回:
        tuple: (真实准确率图, 置换准确率图列表)
    """
    log(f"开始置换检验: {n_permutations}次置换", config)
    
    # 真实标签分析
    real_scores = run_searchlight_classification(functional_imgs, labels, config, mask_img)
    if real_scores is None:
        return None, None
    
    # 置换检验
    permuted_scores = []
    np.random.seed(config.permutation_random_state)
    
    for i in range(n_permutations):
        if (i + 1) % 20 == 0:
            log(f"  置换进度: {i+1}/{n_permutations}", config)
        
        # 随机打乱标签
        permuted_labels = np.random.permutation(labels)
        
        # 运行分析
        perm_scores = run_searchlight_classification(functional_imgs, permuted_labels, config, mask_img)
        if perm_scores is not None:
            permuted_scores.append(perm_scores)
    
    log(f"置换检验完成: {len(permuted_scores)}次有效置换", config)
    
    return real_scores, permuted_scores

def calculate_statistical_maps(real_scores, permuted_scores, config):
    """
    计算统计显著性图
    
    参数:
        real_scores (Nifti1Image): 真实分类准确率图
        permuted_scores (list): 置换分类准确率图列表
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        tuple: (p值图, 校正后p值图, 阈值图)
    """
    if not permuted_scores:
        log("警告: 没有有效的置换结果", config)
        return None, None, None
    
    try:
        # 获取数据数组
        real_data = real_scores.get_fdata()
        perm_data_stack = np.stack([img.get_fdata() for img in permuted_scores], axis=-1)
        
        # 计算p值
        p_values = np.zeros_like(real_data)
        mask = real_data != 0  # 非零体素掩模
        
        for i, j, k in np.ndindex(real_data.shape):
            if mask[i, j, k]:
                real_acc = real_data[i, j, k]
                perm_accs = perm_data_stack[i, j, k, :]
                # 计算p值 (双侧检验)
                p_values[i, j, k] = np.mean(perm_accs >= real_acc)
        
        # 创建p值图
        p_value_img = image.new_img_like(real_scores, p_values)
        
        # 多重比较校正
        flat_p = p_values[mask]
        if len(flat_p) > 0:
            rejected, corrected_p, _, _ = multipletests(
                flat_p, 
                alpha=config.alpha_level, 
                method=config.correction_method
            )
            
            # 重建校正后的p值图
            corrected_p_values = np.ones_like(p_values)
            corrected_p_values[mask] = corrected_p
            corrected_p_img = image.new_img_like(real_scores, corrected_p_values)
            
            # 创建阈值图
            threshold_map = np.zeros_like(p_values)
            threshold_map[mask] = rejected.astype(float)
            threshold_img = image.new_img_like(real_scores, threshold_map)
            
            log(f"统计校正完成: {np.sum(rejected)}/{len(rejected)} 体素显著", config)
            
            return p_value_img, corrected_p_img, threshold_img
        else:
            log("警告: 没有有效体素进行统计校正", config)
            return p_value_img, None, None
            
    except Exception as e:
        log(f"统计计算失败: {e}", config)
        return None, None, None

# ========== 单被试分析函数 ==========
def run_single_mask_analysis(selected_betas, labels, config, mask_path, contrast_name, subject, mask_name=None):
    """
    为单个mask运行searchlight分析
    
    参数:
        selected_betas: 选择的beta图像
        labels: 分类标签
        config: 配置对象
        mask_path: mask路径(None表示全脑)
        contrast_name: 对比名称
        subject: 被试ID
        mask_name: mask名称(用于文件命名)
    
    返回:
        dict: 分析结果
    """
    # 构建文件名后缀
    name_suffix = f"_{mask_name}" if mask_name else ""
    
    if config.n_permutations > 0:
        # 带置换检验的分析
        real_scores, permuted_scores = run_permutation_test_searchlight(
            selected_betas, labels, config, mask_path, config.n_permutations
        )
        
        if real_scores is not None:
            # 计算统计图
            p_img, corrected_p_img, threshold_img = calculate_statistical_maps(
                real_scores, permuted_scores, config
            )
            
            # 保存结果
            subject_dir = config.results_dir / "subject_results" / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存准确率图
            accuracy_path = subject_dir / f"{contrast_name}{name_suffix}_accuracy.nii.gz"
            real_scores.to_filename(str(accuracy_path))
            
            # 保存统计图
            p_path = None
            corrected_p_path = None
            threshold_path = None
            
            if p_img is not None:
                p_path = subject_dir / f"{contrast_name}{name_suffix}_pvalues.nii.gz"
                p_img.to_filename(str(p_path))
            
            if corrected_p_img is not None:
                corrected_p_path = subject_dir / f"{contrast_name}{name_suffix}_corrected_pvalues.nii.gz"
                corrected_p_img.to_filename(str(corrected_p_path))
            
            if threshold_img is not None:
                threshold_path = subject_dir / f"{contrast_name}{name_suffix}_significant.nii.gz"
                threshold_img.to_filename(str(threshold_path))
            
            # 计算汇总统计
            accuracy_data = real_scores.get_fdata()
            mask = accuracy_data != 0
            
            if np.any(mask):
                mean_accuracy = np.mean(accuracy_data[mask])
                max_accuracy = np.max(accuracy_data[mask])
                
                # 显著体素统计
                n_significant = 0
                if threshold_img is not None:
                    threshold_data = threshold_img.get_fdata()
                    n_significant = np.sum(threshold_data > 0)
                
                result = {
                    'mean_accuracy': mean_accuracy,
                    'max_accuracy': max_accuracy,
                    'n_voxels_analyzed': np.sum(mask),
                    'n_significant_voxels': n_significant,
                    'n_trials_cond1': np.sum(labels == 0),
                    'n_trials_cond2': np.sum(labels == 1),
                    'accuracy_path': str(accuracy_path),
                    'p_value_path': str(p_path) if p_path else None,
                    'corrected_p_path': str(corrected_p_path) if corrected_p_path else None,
                    'significant_path': str(threshold_path) if threshold_path else None,
                    'mask_name': mask_name,
                    'mask_path': mask_path
                }
                
                log(f"    平均准确率: {mean_accuracy:.3f}, 最大准确率: {max_accuracy:.3f}", config)
                log(f"    显著体素数: {n_significant}/{np.sum(mask)}", config)
                
                return result
    else:
        # 仅运行分类分析(无置换检验)
        real_scores = run_searchlight_classification(
            selected_betas, labels, config, mask_path
        )
        
        if real_scores is not None:
            # 保存准确率图
            subject_dir = config.results_dir / "subject_results" / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            accuracy_path = subject_dir / f"{contrast_name}{name_suffix}_accuracy.nii.gz"
            real_scores.to_filename(str(accuracy_path))
            
            # 计算汇总统计
            accuracy_data = real_scores.get_fdata()
            mask = accuracy_data != 0
            
            if np.any(mask):
                mean_accuracy = np.mean(accuracy_data[mask])
                max_accuracy = np.max(accuracy_data[mask])
                
                result = {
                    'mean_accuracy': mean_accuracy,
                    'max_accuracy': max_accuracy,
                    'n_voxels_analyzed': np.sum(mask),
                    'n_trials_cond1': np.sum(labels == 0),
                    'n_trials_cond2': np.sum(labels == 1),
                    'accuracy_path': str(accuracy_path),
                    'mask_name': mask_name,
                    'mask_path': mask_path
                }
                
                log(f"    平均准确率: {mean_accuracy:.3f}, 最大准确率: {max_accuracy:.3f}", config)
                
                return result
    
    return None

def analyze_single_subject_searchlight(subject, runs, contrasts, config):
    """
    单被试searchlight MVPA分析
    
    参数:
        subject (str): 被试ID
        runs (list): run编号列表
        contrasts (list): 对比条件列表
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        dict: 分析结果字典
    """
    log(f"开始分析 {subject}", config)
    
    # 加载多run合并数据
    if isinstance(runs, list) and len(runs) > 1:
        beta_images, trial_info = load_multi_run_lss_data(subject, runs, config)
    else:
        # 兼容单run分析
        single_run = runs[0] if isinstance(runs, list) else runs
        beta_images, trial_info = load_lss_trial_data(subject, single_run, config)
    
    if beta_images is None:
        return None
    
    results = {}
    
    for contrast_name, cond1, cond2 in contrasts:
        log(f"  {subject}: {cond1} vs {cond2}", config)
        
        # 准备分类数据
        selected_betas, labels, trial_details = prepare_classification_data(
            trial_info, beta_images, cond1, cond2, config
        )
        
        if selected_betas is None:
            continue
        
        # 获取当前配置的mask(可能是单个mask或mask列表)
        selected_masks = config.get_selected_mask()
        
        # 如果是mask列表，为每个mask分别分析
        if config.use_mask_list and isinstance(selected_masks, list):
            for i, mask_path in enumerate(selected_masks):
                mask_name = config.mask_list[i] if i < len(config.mask_list) else f"mask_{i}"
                log(f"    使用mask: {mask_name} ({mask_path if mask_path else '全脑'})")
                
                # 为每个mask运行分析
                mask_results = run_single_mask_analysis(
                    selected_betas, labels, config, mask_path, 
                    contrast_name, subject, mask_name
                )
                
                if mask_results:
                    # 将mask名称添加到对比名称中
                    mask_contrast_name = f"{contrast_name}_{mask_name}"
                    results[mask_contrast_name] = mask_results
        else:
            # 单个mask的原有逻辑
            mask_results = run_single_mask_analysis(
                selected_betas, labels, config, selected_masks, 
                contrast_name, subject, None
            )
            
            if mask_results:
                 results[contrast_name] = mask_results
        
        # 保存trial详细信息
        runs_str = "_".join([f"run{r}" for r in (runs if isinstance(runs, list) else [runs])])
        trial_details.to_csv(
            config.results_dir / "subject_results" / subject / f"{contrast_name}_{runs_str}_trial_details.csv",
            index=False
        )
    
    memory_cleanup()  # 清理内存
    return results

# ========== 组水平分析函数 ==========
def perform_group_analysis_searchlight(all_subject_results, config):
    """
    执行组水平searchlight分析
    
    参数:
        all_subject_results (dict): 所有被试的分析结果
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        DataFrame: 组水平统计结果
    """
    log("开始组水平分析", config)
    
    # 收集组水平数据
    group_data = []
    for subject, subject_results in all_subject_results.items():
        for contrast_name, contrast_results in subject_results.items():
            group_data.append({
                'subject': subject,
                'contrast': contrast_name,
                'mean_accuracy': contrast_results['mean_accuracy'],
                'max_accuracy': contrast_results['max_accuracy'],
                'n_voxels_analyzed': contrast_results['n_voxels_analyzed'],
                'n_significant_voxels': contrast_results.get('n_significant_voxels', 0),
                'n_trials_cond1': contrast_results['n_trials_cond1'],
                'n_trials_cond2': contrast_results['n_trials_cond2']
            })
    
    group_df = pd.DataFrame(group_data)
    
    if len(group_df) == 0:
        log("警告: 没有有效的组水平数据", config)
        return None
    
    # 保存组水平原始数据
    group_df.to_csv(config.results_dir / "group_searchlight_results.csv", index=False)
    
    # 组水平统计分析
    stats_results = []
    
    for contrast in group_df['contrast'].unique():
        contrast_data = group_df[group_df['contrast'] == contrast]
        
        # 平均准确率统计
        mean_accs = contrast_data['mean_accuracy'].values
        
        # 单样本t检验 (测试是否显著高于0.5)
        t_stat, p_value = stats.ttest_1samp(mean_accs, 0.5)
        
        # 描述性统计
        stats_results.append({
            'contrast': contrast,
            'metric': 'mean_accuracy',
            'n_subjects': len(mean_accs),
            'mean': np.mean(mean_accs),
            'std': np.std(mean_accs, ddof=1),
            'sem': np.std(mean_accs, ddof=1) / np.sqrt(len(mean_accs)),
            'min': np.min(mean_accs),
            'max': np.max(mean_accs),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < config.alpha_level,
            'effect_size_cohen_d': t_stat / np.sqrt(len(mean_accs))
        })
        
        # 最大准确率统计
        max_accs = contrast_data['max_accuracy'].values
        t_stat_max, p_value_max = stats.ttest_1samp(max_accs, 0.5)
        
        stats_results.append({
            'contrast': contrast,
            'metric': 'max_accuracy',
            'n_subjects': len(max_accs),
            'mean': np.mean(max_accs),
            'std': np.std(max_accs, ddof=1),
            'sem': np.std(max_accs, ddof=1) / np.sqrt(len(max_accs)),
            'min': np.min(max_accs),
            'max': np.max(max_accs),
            't_statistic': t_stat_max,
            'p_value': p_value_max,
            'significant': p_value_max < config.alpha_level,
            'effect_size_cohen_d': t_stat_max / np.sqrt(len(max_accs))
        })
    
    stats_df = pd.DataFrame(stats_results)
    
    # 多重比较校正
    if len(stats_df) > 1:
        rejected, corrected_p, _, _ = multipletests(
            stats_df['p_value'].values,
            alpha=config.alpha_level,
            method=config.correction_method
        )
        stats_df['p_value_corrected'] = corrected_p
        stats_df['significant_corrected'] = rejected
    else:
        stats_df['p_value_corrected'] = stats_df['p_value']
        stats_df['significant_corrected'] = stats_df['significant']
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results.csv", index=False)
    
    log(f"组水平分析完成: {len(stats_df)}个统计测试", config)
    
    return stats_df

def create_group_visualizations(group_df, stats_df, config):
    """
    创建组水平可视化图表
    
    参数:
        group_df (DataFrame): 组水平数据
        stats_df (DataFrame): 统计结果
        config (SearchlightMVPAConfig): 配置对象
    """
    log("创建可视化图表", config)
    
    # 设置绘图样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. 分类准确率分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Searchlight MVPA分类结果', fontsize=16, fontweight='bold')
    
    for i, contrast in enumerate(group_df['contrast'].unique()):
        contrast_data = group_df[group_df['contrast'] == contrast]
        
        # 平均准确率箱线图
        ax1 = axes[0, i] if len(group_df['contrast'].unique()) > 1 else axes[0, 0]
        sns.boxplot(data=contrast_data, y='mean_accuracy', ax=ax1)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        ax1.set_title(f'{contrast}\n平均分类准确率')
        ax1.set_ylabel('分类准确率')
        ax1.legend()
        
        # 最大准确率箱线图
        ax2 = axes[1, i] if len(group_df['contrast'].unique()) > 1 else axes[1, 0]
        sns.boxplot(data=contrast_data, y='max_accuracy', ax=ax2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        ax2.set_title(f'{contrast}\n最大分类准确率')
        ax2.set_ylabel('分类准确率')
        ax2.legend()
    
    # 如果只有一个对比，隐藏多余的子图
    if len(group_df['contrast'].unique()) == 1:
        axes[0, 1].set_visible(False)
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(config.results_dir / 'group_accuracy_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 统计结果汇总表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备表格数据
    table_data = []
    for _, row in stats_df.iterrows():
        table_data.append([
            f"{row['contrast']}\n({row['metric']})",
            f"{row['n_subjects']}",
            f"{row['mean']:.3f} ± {row['sem']:.3f}",
            f"{row['t_statistic']:.3f}",
            f"{row['p_value']:.4f}",
            f"{row['p_value_corrected']:.4f}",
            "是" if row['significant_corrected'] else "否"
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['对比条件\n(指标)', 'N', '均值 ± 标准误', 't统计量', 'p值', '校正p值', '显著性'],
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(7):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 6 and table_data[i-1][6] == "是":  # 显著性列
                    cell.set_facecolor('#E8F5E8')
                else:
                    cell.set_facecolor('#F5F5F5')
    
    ax.set_axis_off()
    ax.set_title('Searchlight MVPA统计结果汇总', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(config.results_dir / 'statistical_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log("可视化图表创建完成", config)

def generate_html_report_searchlight(group_df, stats_df, config):
    """
    生成HTML分析报告
    
    参数:
        group_df (DataFrame): 组水平数据
        stats_df (DataFrame): 统计结果
        config (SearchlightMVPAConfig): 配置对象
    """
    log("生成HTML报告", config)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Searchlight MVPA分析报告</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            .info-box {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
            .success {{
                background-color: #d5f4e6;
                border-left: 4px solid #27ae60;
            }}
            .warning {{
                background-color: #fef9e7;
                border-left: 4px solid #f39c12;
            }}
            .error {{
                background-color: #fadbd8;
                border-left: 4px solid #e74c3c;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .significant {{
                background-color: #d5f4e6 !important;
                font-weight: bold;
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Searchlight MVPA分析报告</h1>
            
            <div class="info-box">
                <h3>📊 分析概览</h3>
                <p><strong>分析时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>分析方法:</strong> Searchlight多体素模式分析 (MVPA)</p>
                <p><strong>被试数量:</strong> {len(group_df['subject'].unique())}</p>
                <p><strong>对比条件:</strong> {', '.join(group_df['contrast'].unique())}</p>
                <p><strong>Searchlight半径:</strong> {config.searchlight_radius} mm</p>
                <p><strong>交叉验证:</strong> {config.cv_folds}折分层交叉验证</p>
                <p><strong>置换检验:</strong> {config.n_permutations}次置换</p>
                <p><strong>多重比较校正:</strong> {config.correction_method.upper()}</p>
            </div>
    """
    
    # 添加统计结果表格
    html_content += """
            <h2>📈 统计结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>对比条件</th>
                        <th>指标</th>
                        <th>被试数</th>
                        <th>均值</th>
                        <th>标准误</th>
                        <th>t统计量</th>
                        <th>p值</th>
                        <th>校正p值</th>
                        <th>显著性</th>
                        <th>效应量(Cohen's d)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in stats_df.iterrows():
        row_class = "significant" if row['significant_corrected'] else ""
        significance = "✅ 显著" if row['significant_corrected'] else "❌ 不显著"
        
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{row['contrast']}</td>
                        <td>{row['metric']}</td>
                        <td>{row['n_subjects']}</td>
                        <td>{row['mean']:.3f}</td>
                        <td>{row['sem']:.3f}</td>
                        <td>{row['t_statistic']:.3f}</td>
                        <td>{row['p_value']:.4f}</td>
                        <td>{row['p_value_corrected']:.4f}</td>
                        <td>{significance}</td>
                        <td>{row['effect_size_cohen_d']:.3f}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
    """
    
    # 添加结果解释
    significant_results = stats_df[stats_df['significant_corrected']]
    if len(significant_results) > 0:
        html_content += f"""
            <div class="info-box success">
                <h3>✅ 显著结果</h3>
                <p>发现 <strong>{len(significant_results)}</strong> 个显著的分类结果:</p>
                <ul>
        """
        for _, row in significant_results.iterrows():
            html_content += f"<li><strong>{row['contrast']} ({row['metric']}):</strong> 平均准确率 = {row['mean']:.3f}, p = {row['p_value_corrected']:.4f}</li>"
        html_content += "</ul></div>"
    else:
        html_content += """
            <div class="info-box warning">
                <h3>⚠️ 无显著结果</h3>
                <p>在当前分析中未发现显著高于偶然水平(0.5)的分类准确率。</p>
            </div>
        """
    
    # 添加可视化图片
    html_content += """
            <h2>📊 可视化结果</h2>
            
            <div class="image-container">
                <h3>分类准确率分布</h3>
                <img src="group_accuracy_distributions.png" alt="分类准确率分布图">
            </div>
            
            <div class="image-container">
                <h3>统计结果汇总</h3>
                <img src="statistical_summary_table.png" alt="统计结果汇总表">
            </div>
    """
    
    # 添加方法说明
    html_content += """
            <h2>🔬 方法说明</h2>
            <div class="info-box">
                <h4>Searchlight MVPA分析流程:</h4>
                <ol>
                    <li><strong>数据准备:</strong> 加载LSS分析得到的trial-wise beta系数</li>
                    <li><strong>Searchlight定义:</strong> 在每个体素周围定义半径为{radius}mm的球形搜索区域</li>
                    <li><strong>特征提取:</strong> 提取每个球形区域内的体素信号作为特征向量</li>
                    <li><strong>分类分析:</strong> 使用SVM分类器进行{cv_folds}折交叉验证</li>
                    <li><strong>置换检验:</strong> 进行{n_perm}次标签置换以评估统计显著性</li>
                    <li><strong>多重比较校正:</strong> 使用{correction}方法校正多重比较问题</li>
                </ol>
                
                <h4>统计分析:</h4>
                <ul>
                    <li><strong>单样本t检验:</strong> 测试分类准确率是否显著高于偶然水平(0.5)</li>
                    <li><strong>效应量:</strong> 使用Cohen's d量化效应大小</li>
                    <li><strong>显著性阈值:</strong> α = {alpha}</li>
                </ul>
            </div>
    """.format(
        radius=config.searchlight_radius,
        cv_folds=config.cv_folds,
        n_perm=config.n_permutations,
        correction=config.correction_method.upper(),
        alpha=config.alpha_level
    )
    
    # 添加潜在问题分析
    html_content += """
            <h2>🔍 潜在问题分析</h2>
            <div class="info-box">
                <h4>如果分类准确率未达到显著水平，可能的原因包括:</h4>
                <ul>
                    <li><strong>信号强度不足:</strong> 实验条件间的神经活动差异可能较小</li>
                    <li><strong>噪声水平过高:</strong> 数据预处理可能需要进一步优化</li>
                    <li><strong>样本量不足:</strong> 每个条件的trial数量可能不够</li>
                    <li><strong>特征选择:</strong> Searchlight半径可能需要调整</li>
                    <li><strong>分类器参数:</strong> SVM参数可能需要进一步优化</li>
                    <li><strong>实验设计:</strong> 条件对比可能需要重新考虑</li>
                </ul>
                
                <h4>建议的改进措施:</h4>
                <ul>
                    <li>检查数据质量和预处理流程</li>
                    <li>尝试不同的searchlight半径(3-6mm)</li>
                    <li>考虑使用特征选择方法</li>
                    <li>增加样本量或trial数量</li>
                    <li>尝试其他分类算法(如随机森林)</li>
                    <li>考虑ROI-based分析作为补充</li>
                </ul>
            </div>
    """
    
    # 添加页脚
    html_content += f"""
            <div class="footer">
                <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>分析工具: Searchlight MVPA Pipeline (Enhanced Version)</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    report_path = config.results_dir / "searchlight_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"HTML报告已保存: {report_path}", config)

# ========== 主分析函数 ==========
def run_group_searchlight_analysis(config):
    """
    运行组水平searchlight MVPA分析
    
    参数:
        config (SearchlightMVPAConfig): 配置对象
    
    返回:
        tuple: (组水平数据, 统计结果) 或 None
    """
    log("开始Searchlight MVPA分析", config)
    
    # 验证mask配置
    is_valid, mask_info = config.validate_mask_availability()
    if not is_valid:
        log(f"错误: {mask_info}", config)
        log(f"可用的mask选项: {config.get_available_masks()}", config)
        return None
    
    log(f"Mask配置验证通过: {mask_info}", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")
    
    # 分析每个被试
    all_subject_results = {}
    
    for subject in config.subjects:
        subject_results = analyze_single_subject_searchlight(
            subject, config.runs, config.contrasts, config
        )
        if subject_results is not None:
            all_subject_results[subject] = subject_results
        else:
            log(f"跳过被试 {subject}: 数据加载失败", config)
    
    if not all_subject_results:
        log("错误: 没有获得任何分析结果", config)
        return None
    
    # 组水平分析
    group_df = None
    stats_df = perform_group_analysis_searchlight(all_subject_results, config)
    
    if stats_df is not None:
        # 重新构建group_df用于可视化
        group_data = []
        for subject, subject_results in all_subject_results.items():
            for contrast_name, contrast_results in subject_results.items():
                group_data.append({
                    'subject': subject,
                    'contrast': contrast_name,
                    'mean_accuracy': contrast_results['mean_accuracy'],
                    'max_accuracy': contrast_results['max_accuracy'],
                    'n_voxels_analyzed': contrast_results['n_voxels_analyzed'],
                    'n_significant_voxels': contrast_results.get('n_significant_voxels', 0),
                    'n_trials_cond1': contrast_results['n_trials_cond1'],
                    'n_trials_cond2': contrast_results['n_trials_cond2']
                })
        
        group_df = pd.DataFrame(group_data)
        
        # 创建可视化
        create_group_visualizations(group_df, stats_df, config)
        
        # 生成HTML报告
        generate_html_report_searchlight(group_df, stats_df, config)
    
    log(f"\nSearchlight MVPA分析完成!", config)
    log(f"结果保存在: {config.results_dir}", config)
    log(f"分析被试数: {len(all_subject_results)}", config)
    log(f"对比数量: {len(config.contrasts)}", config)
    log(f"HTML报告: {config.results_dir / 'searchlight_analysis_report.html'}", config)
    
    return group_df, stats_df

def main():
    """
    主函数
    """
    # 创建配置
    config = SearchlightMVPAConfig()
    
    # 运行分析
    results = run_group_searchlight_analysis(config)
    
    if results is not None:
        group_df, stats_df = results
        logging.info("\n=== Searchlight MVPA分析完成 ===")
        logging.info(f"总测试数: {len(stats_df)}")
        logging.info(f"显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}")
        logging.info(f"详细报告: {config.results_dir / 'searchlight_analysis_report.html'}")
    else:
        logging.error("Searchlight MVPA分析失败")

if __name__ == "__main__":
    main()