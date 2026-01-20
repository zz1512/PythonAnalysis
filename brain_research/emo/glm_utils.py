#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_utils.py
工具函数库 - 复杂头动处理 & 事件分类
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_complex_confounds(confounds_path):
    """
    实现 Matlab 截图中的回归策略:
    1. Friston-24 (6参数 + 6导数 + 6平方 + 6导数平方)
    2. aCompCor (前6个分量)
    3. Scrubbing (motion_outlier + non_steady_state)
    """
    try:
        df = pd.read_csv(confounds_path, sep='\t')

        # A. Friston-24
        base_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

        # 容错：如果找不到标准列名，尝试查找存在的列
        if not all(col in df.columns for col in base_axes):
            # 简单回退策略：只要是 trans 或 rot 开头的列
            # 实际 BIDS 输出通常都有这6个，名字可能微调，这里假设标准 fmriprep 输出
            pass

        R_motion = df[base_axes].copy()
        R_deriv = R_motion.diff().fillna(0)
        R_power2 = R_motion ** 2
        R_deriv_power2 = R_deriv ** 2

        # 重命名
        R_deriv.columns = [c + '_dt' for c in base_axes]
        R_power2.columns = [c + '_sq' for c in base_axes]
        R_deriv_power2.columns = [c + '_dt_sq' for c in base_axes]

        friston24 = pd.concat([R_motion, R_deriv, R_power2, R_deriv_power2], axis=1)

        # B. aCompCor (前6个)
        acomp_cols = df.filter(regex='^a_comp_cor_0[0-5]$')

        # C. Scrubbing (Outliers)
        outlier_cols = df.filter(regex='motion_outlier|non_steady_state')

        # 合并
        final_confounds = pd.concat([friston24, acomp_cols, outlier_cols], axis=1)
        return final_confounds.fillna(0)

    except Exception as e:
        logger.error(f"读取 Confounds 失败: {confounds_path} | Error: {e}")
        raise


def reclassify_events(df):
    """
    Condition/Emotion 3分类逻辑
    如果不满足特定条件，则保留原始 trial_type
    """
    col_map = {c.lower(): c for c in df.columns}
    cond_col = col_map.get('condition')
    emo_col = col_map.get('emotion')

    # 如果缺少必要列，直接返回原表
    if not cond_col or not emo_col:
        return df

    def _get_new_type(row):
        cond = str(row[cond_col]).strip()
        emo = str(row[emo_col]).strip()

        if cond == 'Reappraisal.png':
            return 'Reappraisal'
        elif cond == 'PassiveLook.png':
            if emo.startswith('Neutral'):
                return 'Passive_Neutral'
            else:
                return 'Passive_Emo'
        else:
            # [关键] 对于 Instruction 或其他事件，保留原 trial_type
            # 这样主程序就能对它们也进行建模
            return row.get('trial_type', 'Other')

    df_new = df.copy()
    df_new['trial_type'] = df_new.apply(_get_new_type, axis=1)
    return df_new


def create_lss_events(target_idx, all_events):
    """LSS 核心：Target vs Others"""
    lss_df = all_events.copy()
    lss_df.loc[target_idx, 'trial_type'] = 'LSS_TARGET'
    return lss_df[['onset', 'duration', 'trial_type']]