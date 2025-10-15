# metaphor_roi_mvpa.py
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 配置参数 ==========
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 5)]
RUNS = [3]
LSS_ROOT = Path(r"H:\PythonAnalysis\learn_LSS")
ROI_DIR = Path(r"H:\PythonAnalysis\learn_mvpa\full_roi_mask")  # ROI掩码目录
RESULTS_DIR = Path(r"H:\PythonAnalysis\learn_mvpa\metaphor_ROI_MVPA")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 关键对比
CONTRASTS = [
    ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间
]


def log(message):
    print(f"[MVPA] {message}", flush=True)


def load_roi_masks(roi_dir):
    """加载预生成的ROI掩模"""
    log("加载ROI掩模...")
    roi_masks = {}

    # 查找所有ROI文件
    roi_files = list(roi_dir.glob("*.nii.gz"))

    for roi_file in roi_files:
        roi_name = roi_file.stem.replace("cluster_", "")  # 从文件名提取ROI名称
        try:
            roi_mask = image.load_img(str(roi_file))
            roi_masks[roi_name] = roi_mask
            log(f"加载ROI: {roi_name}")
        except Exception as e:
            log(f"加载ROI失败 {roi_file}: {e}")

    if not roi_masks:
        log("错误: 没有找到可用的ROI掩模文件")
        log(f"请确保ROI目录包含ROI_*.nii.gz文件: {roi_dir}")
        return None

    log(f"成功加载 {len(roi_masks)} 个ROI掩模")
    return roi_masks


def load_lss_trial_data(subject, run):
    """加载LSS分析的trial数据和beta图像"""
    lss_dir = LSS_ROOT / subject / f"run-{run}_LSS"

    if not lss_dir.exists():
        log(f"LSS目录不存在: {lss_dir}")
        return None, None

    # 加载trial信息
    trial_map_path = lss_dir / "trial_map.csv"
    if not trial_map_path.exists():
        log(f"trial_map.csv不存在: {trial_map_path}")
        return None, None

    trial_info = pd.read_csv(trial_map_path)

    # 加载所有beta图像
    beta_images = []
    valid_trials = []

    for _, trial in trial_info.iterrows():
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        if beta_path.exists():
            beta_images.append(str(beta_path))
            valid_trials.append(trial)
        else:
            log(f"Beta图像缺失: {beta_path}")

    if len(beta_images) == 0:
        log(f"没有找到有效的beta图像: {subject}")
        return None, None

    valid_trials_df = pd.DataFrame(valid_trials)
    log(f"加载 {subject}: {len(beta_images)}个trial")

    return beta_images, valid_trials_df


def extract_roi_timeseries(functional_imgs, roi_mask_path, confounds=None):
    """提取ROI时间序列"""
    try:
        # 使用正确的Masker类
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            detrend=True,
            high_pass=1/128,
            t_r=2.0,
            memory='nilearn_cache',
            memory_level=1
        )

        # 提取时间序列
        timeseries = masker.fit_transform(functional_imgs, confounds=confounds)
        log(f"提取时间序列形状: {timeseries.shape}")

        return timeseries, masker

    except Exception as e:
        log(f"提取ROI时间序列失败: {e}")
        return None, None


def run_roi_classification(X, y, cv_folds=5, n_permutations=100):
    """在单个ROI上运行分类分析"""
    # 创建分类pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])

    # 交叉验证
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    # 置换检验
    null_distribution = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_scores = cross_val_score(pipeline, X, y_perm, cv=cv, scoring='accuracy')
        null_distribution.append(np.mean(perm_scores))

    # 计算p值
    p_value = (np.sum(null_distribution >= np.mean(cv_scores)) + 1) / (n_permutations + 1)

    return {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'p_value': p_value,
        'null_distribution': null_distribution,
        'chance_level': 0.5
    }


def prepare_classification_data(trial_info, beta_images, cond1, cond2):
    """准备分类数据"""
    # 筛选特定条件的trial
    cond1_trials = trial_info[trial_info['trial_condition'].str.contains(cond1, case=False, na=False)]
    cond2_trials = trial_info[trial_info['trial_condition'].str.contains(cond2, case=False, na=False)]

    if len(cond1_trials) < 3 or len(cond2_trials) < 3:
        log(f"条件 {cond1} vs {cond2}: 样本数不足 ({len(cond1_trials)} vs {len(cond2_trials)})")
        return None, None, None

    # 准备数据和标签
    X_indices = []
    y_labels = []
    trial_details = []

    # 条件1的trial
    for _, trial in cond1_trials.iterrows():
        trial_idx = trial['trial_index'] - 1  # 转为0-based索引
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(0)  # 条件1标记为0
            trial_details.append({
                'trial_index': trial['trial_index'],
                'condition': trial['trial_condition'],
                'label': 0
            })

    # 条件2的trial
    for _, trial in cond2_trials.iterrows():
        trial_idx = trial['trial_index'] - 1
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(1)  # 条件2标记为1
            trial_details.append({
                'trial_index': trial['trial_index'],
                'condition': trial['trial_condition'],
                'label': 1
            })

    if len(X_indices) < 6:  # 每个条件至少3个样本
        log(f"有效样本数不足: {len(X_indices)}")
        return None, None, None

    # 选择对应的beta图像
    selected_betas = [beta_images[i] for i in X_indices]

    return selected_betas, np.array(y_labels), pd.DataFrame(trial_details)


def analyze_single_subject_roi(subject, run, contrasts, roi_masks):
    """分析单个被试的ROI MVPA"""
    log(f"开始分析 {subject}")

    # 加载数据
    beta_images, trial_info = load_lss_trial_data(subject, run)
    if beta_images is None:
        return None

    results = {}

    for contrast_name, cond1, cond2 in contrasts:
        log(f"  {subject}: {cond1} vs {cond2}")

        # 准备分类数据
        selected_betas, labels, trial_details = prepare_classification_data(
            trial_info, beta_images, cond1, cond2
        )

        if selected_betas is None:
            continue

        contrast_results = {}

        # 在每个ROI上进行分析
        for roi_name, roi_mask in roi_masks.items():
            log(f"    ROI: {roi_name}")

            # 提取ROI时间序列
            X, masker = extract_roi_timeseries(selected_betas, roi_mask)
            if X is None:
                continue

            # 运行分类
            classification_result = run_roi_classification(X, labels)

            contrast_results[roi_name] = {
                **classification_result,
                'n_trials_cond1': np.sum(labels == 0),
                'n_trials_cond2': np.sum(labels == 1),
                'n_voxels': X.shape[1],
                'masker': masker
            }

        if contrast_results:
            results[contrast_name] = contrast_results
            # 保存trial详细信息
            trial_details.to_csv(
                RESULTS_DIR / f"{subject}_{contrast_name}_trial_details.csv",
                index=False
            )

    return results


def run_group_roi_analysis(subjects, run, contrasts, roi_masks):
    """运行组水平ROI分析"""
    log("开始组水平ROI分析")

    # 分析每个被试
    all_subject_results = {}

    for subject in subjects:
        subject_results = analyze_single_subject_roi(
            subject, run, contrasts, roi_masks
        )
        if subject_results is not None:
            all_subject_results[subject] = subject_results
        else:
            log(f"跳过被试 {subject}: 数据加载失败")

    return all_subject_results


def save_and_analyze_group_results(all_results, contrasts):
    """保存和分析组水平结果"""
    log("保存和分析组水平结果")

    # 收集所有结果
    group_data = []

    for subject, subject_results in all_results.items():
        for contrast_name, contrast_results in subject_results.items():
            for roi_name, roi_result in contrast_results.items():
                group_data.append({
                    'subject': subject,
                    'contrast': contrast_name,
                    'roi': roi_name,
                    'accuracy': roi_result['mean_accuracy'],
                    'std_accuracy': roi_result['std_accuracy'],
                    'p_value': roi_result['p_value'],
                    'n_trials_cond1': roi_result['n_trials_cond1'],
                    'n_trials_cond2': roi_result['n_trials_cond2'],
                    'n_voxels': roi_result['n_voxels']
                })

    group_df = pd.DataFrame(group_data)
    group_df.to_csv(RESULTS_DIR / "group_roi_mvpa_results.csv", index=False)

    # 组水平统计分析
    perform_group_statistics(group_df, contrasts)

    # 可视化结果
    create_group_visualizations(group_df, contrasts)

    return group_df


def perform_group_statistics(group_df, contrasts):
    """执行组水平统计检验"""
    log("执行组水平统计检验")

    stats_results = []

    for contrast_name, cond1, cond2 in contrasts:
        contrast_data = group_df[group_df['contrast'] == contrast_name]

        for roi in group_df['roi'].unique():
            roi_data = contrast_data[contrast_data['roi'] == roi]['accuracy']

            if len(roi_data) > 3:  # 至少4个被试
                # 单样本t检验 vs 随机水平(0.5)
                t_stat, p_value = stats.ttest_1samp(roi_data, 0.5)
                cohens_d = (np.mean(roi_data) - 0.5) / np.std(roi_data)

                stats_results.append({
                    'contrast': contrast_name,
                    'roi': roi,
                    'mean_accuracy': np.mean(roi_data),
                    'std_accuracy': np.std(roi_data),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'n_subjects': len(roi_data)
                })

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(RESULTS_DIR / "group_statistical_results.csv", index=False)

    # 打印显著结果
    significant_results = stats_df[stats_df['p_value'] < 0.05]
    if not significant_results.empty:
        log("\n显著结果:")
        for _, result in significant_results.iterrows():
            log(f"  {result['contrast']} - {result['roi']}: "
                f"accuracy = {result['mean_accuracy']:.3f}, "
                f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                f"p = {result['p_value']:.3f}, d = {result['cohens_d']:.3f}")
    else:
        log("\n无显著结果")

    return stats_df


def create_group_visualizations(group_df, contrasts):
    """创建组水平可视化"""
    log("创建可视化")

    try:
        # 1. 各ROI分类准确率热图
        plt.figure(figsize=(12, 8))

        # 准备热图数据
        heatmap_data = []
        contrast_names = [c[0] for c in contrasts]

        for contrast_name, _, _ in contrasts:
            contrast_row = []
            for roi in sorted(group_df['roi'].unique()):
                roi_data = group_df[(group_df['contrast'] == contrast_name) &
                                    (group_df['roi'] == roi)]['accuracy']
                contrast_row.append(np.mean(roi_data) if len(roi_data) > 0 else 0)
            heatmap_data.append(contrast_row)

        heatmap_data = np.array(heatmap_data)

        # 绘制热图
        sns.heatmap(heatmap_data,
                    xticklabels=sorted(group_df['roi'].unique()),
                    yticklabels=contrast_names,
                    annot=True, fmt='.3f', cmap='RdYlBu_r',
                    vmin=0.4, vmax=0.8, center=0.5)
        plt.title('ROI MVPA Classification Accuracy')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roi_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 各对比的ROI准确率条形图
        for contrast_name, _, _ in contrasts:
            contrast_data = group_df[group_df['contrast'] == contrast_name]

            if not contrast_data.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=contrast_data, x='roi', y='accuracy', capsize=0.1)
                plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='随机水平')
                plt.title(f'Classification Accuracy: {contrast_name}')
                plt.xticks(rotation=45)
                plt.ylabel('Accuracy')
                plt.legend()
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"accuracy_{contrast_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

        log("可视化完成")

    except Exception as e:
        log(f"可视化失败: {e}")


# ========== 主执行函数 ==========
def main():
    log("开始隐喻ROI MVPA分析")

    # 加载ROI掩模
    roi_masks = load_roi_masks(ROI_DIR)
    if roi_masks is None:
        return

    # 运行组分析
    all_results = run_group_roi_analysis(SUBJECTS, RUNS[0], CONTRASTS, roi_masks)

    if all_results is None or len(all_results) == 0:
        log("错误: 没有获得任何分析结果")
        return

    # 保存和分析结果
    group_df = save_and_analyze_group_results(all_results, CONTRASTS)

    log(f"\nMVPA分析完成!")
    log(f"结果保存在: {RESULTS_DIR}")
    log(f"分析被试数: {len(all_results)}")
    log(f"ROI数量: {len(roi_masks)}")
    log(f"对比数量: {len(CONTRASTS)}")


if __name__ == "__main__":
    main()