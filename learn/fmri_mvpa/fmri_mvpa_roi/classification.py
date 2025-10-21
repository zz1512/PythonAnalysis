# classification.py
# 分类分析模块

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import log


def run_roi_classification_enhanced(X, y, config):
    """增强的ROI分类分析 - 修正版"""

    # 外层CV：用于评估泛化性能
    outer_cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.cv_random_state
    )

    # 内层CV：用于超参数选择
    inner_cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.cv_random_state + 1
    )

    # 创建pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=config.svm_params['random_state']))
    ])

    param_grid = {'svm__' + key: value for key, value in config.svm_param_grid.items()}

    # 手动实现嵌套交叉验证
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 在内层训练集上进行网格搜索
        inner_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=1
        )
        inner_search.fit(X_train, y_train)

        # 用最佳模型在测试集上评估
        best_model = inner_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        outer_scores.append(test_score)

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
    
    # 条件1的trial
    for i, (_, trial) in enumerate(cond1_trials.iterrows()):
        # 对于多run合并数据，使用combined_trial_index；对于单run数据，使用trial_index-1
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
            index_type = 'combined_trial_index'
        else:
            trial_idx = trial['trial_index'] - 1
            index_type = 'trial_index-1'
            
        if trial_idx < len(beta_images):
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
                log(f"  {cond1}[{i+1}]: trial_index={trial['trial_index']}, {index_type}={trial_idx}, "
                    f"run={trial.get('run', 'unknown')}", config)
        else:
            log(f"  警告: trial_idx={trial_idx} 超出beta_images范围 (len={len(beta_images)})", config)
    
    # 条件2的trial
    for i, (_, trial) in enumerate(cond2_trials.iterrows()):
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
            index_type = 'combined_trial_index'
        else:
            trial_idx = trial['trial_index'] - 1
            index_type = 'trial_index-1'
            
        if trial_idx < len(beta_images):
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
                log(f"  {cond2}[{i+1}]: trial_index={trial['trial_index']}, {index_type}={trial_idx}, "
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
    log(f"分类数据准备完成: {cond1}={cond1_count}, {cond2}={cond2_count}, 总计{len(selected_beta_images)}个beta文件", config)
    
    # 检查是否有重复的beta文件
    unique_betas = set(selected_beta_images)
    if len(unique_betas) != len(selected_beta_images):
        log(f"警告: 发现重复的beta文件! 唯一文件数: {len(unique_betas)}, 总选择数: {len(selected_beta_images)}", config)
    
    return selected_beta_images, np.array(y_labels), pd.DataFrame(trial_details)