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
    # Confounds = Friston-24 + aCompCor(6) + scrubbing regressors (outlier spikes).
    try:
        df = pd.read_csv(confounds_path, sep='\t')

        # A. Friston-24
        base_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

        # 容错：如果找不到标准列名，尝试查找存在的列
        if not all(col in df.columns for col in base_axes):
            # 尝试查找存在的 trans 或 rot 列
            found_cols = []
            for col in df.columns:
                if col.startswith('trans_') or col.startswith('rot_'):
                    found_cols.append(col)
            
            if len(found_cols) >= 6:
                # 使用找到的前6个列
                base_axes = found_cols[:6]
                logger.warning(f"找不到标准运动列名，使用找到的列: {base_axes}")
            else:
                raise ValueError(f"找不到足够的运动参数列，只找到 {len(found_cols)} 列")

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
        if acomp_cols.empty:
            # 尝试其他可能的命名格式
            acomp_cols = df.filter(regex='^a_comp_cor_')
            if not acomp_cols.empty:
                acomp_cols = acomp_cols.iloc[:, :6]  # 取前6个
                logger.warning(f"找不到标准 aCompCor 列名，使用找到的 {len(acomp_cols.columns)} 列")
            else:
                logger.warning("未找到 aCompCor 列，将不包含在回归中")

        # C. Scrubbing (Outliers)
        outlier_cols = df.filter(regex='motion_outlier|non_steady_state')
        if outlier_cols.empty:
            logger.warning("未找到离群值列，将不包含在回归中")

        # 合并
        final_confounds = pd.concat([friston24, acomp_cols, outlier_cols], axis=1)
        return final_confounds.fillna(0)

    except Exception as e:
        logger.error(f"读取 Confounds 失败: {confounds_path} | Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


def reclassify_events(df):
    """
    Condition/Emotion 3分类逻辑
    只保留3类：
    1. Condition = PassiveLook.png 且 Emotion 以 Neutral 开头
    2. Condition = PassiveLook.png 且 Emotion 不以 Neutral 开头
    3. Condition = Reappraisal.png
    """
    try:
        # Map raw condition/emotion into 3 analysis trial types + a catch-all "Other".
        col_map = {c.lower(): c for c in df.columns}
        cond_col = col_map.get('condition')
        emo_col = col_map.get('emotion')

        # 如果缺少必要列，直接返回原表
        if not cond_col or not emo_col:
            logger.warning(f"缺少必要的事件分类列: condition={cond_col}, emotion={emo_col}，将保留原始数据")
            return df

        def _get_new_type(row):
            try:
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
                    # 对于其他事件，返回 'Other'
                    return 'Other'
            except Exception as e:
                logger.warning(f"处理事件分类时出错: {e}，将使用 'Other' 作为类型")
                return 'Other'

        df_new = df.copy()
        df_new['trial_type'] = df_new.apply(_get_new_type, axis=1)
        return df_new
    except Exception as e:
        logger.error(f"事件分类失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # 出错时返回原始数据，确保流程不中断
        return df


def create_lss_events(target_idx, all_events, event_name_col="Choice"):
    """
    LSS 核心：Target vs Others（使用 Observe/Feeling 时间列）

    参数说明：
    - target_idx: 目标试次的索引（list/array，如 [3,5,7]）
    - all_events: E-Prime 行为数据 DataFrame（必须包含 `Observe.OnsetTime` 与 `Feeling.FinishTime`）
    - event_name_col: 保留参数，仅为兼容旧调用（当前不用于时间提取）
    """
    lss_df = all_events.copy()

    # 1. 标记 LSS 试次类型
    # LSS design: one target event vs a pooled "others" event.
    lss_df["trial_type"] = "LSS_OTHERS"
    lss_df.loc[target_idx, "trial_type"] = "LSS_TARGET"

    # --------------------------
    # 2. 提取 Onset（固定使用 Observe.OnsetTime）
    # --------------------------
    def get_onset(row):
        onset_field = "Observe.OnsetTime"
        if onset_field not in lss_df.columns:
            raise ValueError(f"数据缺少列：{onset_field}（请检查 E-Prime 导出的列名）")
        # 毫秒转秒
        return row[onset_field] / 1000

    # --------------------------
    # 3. 提取 Duration（Feeling.FinishTime - Observe.OnsetTime）
    # --------------------------
    def get_duration(row):
        finish_field = "Feeling.FinishTime"
        onset_field = "Observe.OnsetTime"

        for col in (finish_field, onset_field):
            if col not in lss_df.columns:
                raise ValueError(f"数据缺少列：{col}（请检查 E-Prime 导出的列名）")

        # 毫秒差值 + 异常值处理
        duration_ms = row[finish_field] - row[onset_field]
        if pd.isna(duration_ms) or duration_ms <= 0 or duration_ms == -99999:
            return 0  # 处理无效值（如你的截图里的 -99999）
        return duration_ms / 1000

    # --------------------------
    # 4. 应用提取逻辑
    # --------------------------
    lss_df["onset"] = lss_df.apply(get_onset, axis=1)
    lss_df["duration"] = lss_df.apply(get_duration, axis=1)

    # --------------------------
    # 5. 时间零点校准（如果有 Fixation.OnsetTime 列）
    # --------------------------
    if "Fixation.OnsetTime" in lss_df.columns:
        del_time_sec = lss_df["Fixation.OnsetTime"].iloc[0] / 1000
        lss_df["onset"] = lss_df["onset"] - del_time_sec
        lss_df["onset"] = lss_df["onset"].clip(lower=0)  # 避免负时间

    # 返回 fMRI 标准格式
    return lss_df[["onset", "duration", "trial_type"]].dropna()
