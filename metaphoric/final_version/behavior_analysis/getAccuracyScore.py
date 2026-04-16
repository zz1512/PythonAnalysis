#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为数据统计分析脚本：隐喻-空间记忆效果比较

功能描述:
    对run-7的行为数据进行统计分析，比较YY条件和KJ条件下的记忆效果差异。
    使用成对样本t检验分析被试内条件差异，并提供独立样本t检验作为补充分析。

分析方法:
    1. 成对样本t检验（主分析）：比较每个被试在YY和KJ条件下的平均记忆得分
    2. 独立样本Welch t检验（补充）：不按被试配对，直接比较所有YY和KJ试次
    3. 效应量计算：Cohen's dz（配对）和Hedges' g（独立）
    4. 正态性检验：Shapiro-Wilk检验

预期结果示例:
    被试数 n = 27
    YY 平均(被试级) = 0.6799 ± 0.1713
    KJ 平均(被试级) = 0.5878 ± 0.2350
    差值(YY-KJ) = 0.0921 95%CI [0.0473, 0.1369]
    t(26) = 4.2232, p = 0.000261
    Cohen's dz = 0.813

注意事项:
    - 第12个被试没有行为数据，实际分析被试数为27
    - 需要确保数据文件包含'trial_type'和'memory'列

作者: 研究团队
版本: 1.0
日期: 2024
"""

import os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
# === 分析配置参数 ===
# Prefer env-var based root for portability.
_ROOT = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DATA_DIR = os.environ.get("METAPHOR_DATA_EVENTS", str(_ROOT / "data_events"))  # 行为数据文件目录
FILE_PATTERN = "*run-7_events.tsv"     # 文件匹配模式
DV_COL = "memory"                      # 因变量列名（记忆效果指标）

# 实验条件定义
YY_SET = {"yyw", "yyew"}               # YY条件的trial_type值集合
KJ_SET = {"kjw", "kjew"}               # KJ条件的trial_type值集合

# 文件读取说明：
# - 自动识别TSV/CSV格式
# - 优先使用制表符分隔符
# - 支持递归搜索子目录

def smart_read(path: str) -> pd.DataFrame:
    """
    智能读取数据文件，自动识别TSV/CSV格式
    
    Args:
        path: 文件路径
    
    Returns:
        pd.DataFrame: 读取的数据框
    
    Note:
        优先尝试TSV格式（制表符分隔），失败时回退到CSV格式
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    # 兜底策略：先尝试TSV再尝试CSV
    try:
        return pd.read_csv(path, sep="\t")
    except:
        return pd.read_csv(path)

def get_sub_id_from_name(fn: str) -> str:
    """
    从文件名中提取被试ID
    
    Args:
        fn: 文件名或文件路径
    
    Returns:
        str: 标准化的被试ID（格式：sub-XX）
    
    Examples:
        'sub-01_run-7_events.tsv' -> 'sub-01'
        'subject_05_data.csv' -> 'sub-05'
        '12_events.tsv' -> 'sub-12'
    """
    base = os.path.basename(fn)
    m = re.search(r"sub[-_]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        return f"sub-{m.group(1).zfill(2)}"
    # 试试纯数字
    m2 = re.search(r"(\d+)", base)
    if m2:
        return f"sub-{m2.group(1).zfill(2)}"
    return os.path.splitext(base)[0]

# 收集所有文件
paths = sorted(glob.glob(os.path.join(DATA_DIR, "**", FILE_PATTERN), recursive=True))
if len(paths) == 0:
    # 也许你给的是目录但没设通配符；尝试常见两类
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*run-7_events.tsv")))

if len(paths) == 0:
    raise FileNotFoundError("未在目录中找到数据文件，请检查 DATA_DIR / FILE_PATTERN 设置。")

rows = []
all_trials_for_ind = []  # 用于独立样本 Welch t 的补充分析（不做被试内聚合）

for i, p in enumerate(paths, start=1):
    df = smart_read(p)

    # 基本列检查
    if "trial_type" not in df.columns:
        raise ValueError(f"{p} 中缺少 'trial_type' 列。")
    if DV_COL not in df.columns:
        raise ValueError(f"{p} 中缺少指标列 '{DV_COL}'。请修改 DV_COL 或检查数据。")

    # 丢掉DV缺失的行
    df = df.dropna(subset=[DV_COL])

    # 标记条件
    df["cond"] = np.where(df["trial_type"].isin(YY_SET), "YY",
                   np.where(df["trial_type"].isin(KJ_SET), "KJ", "OTHER"))

    # 提取本被试 YY / KJ 的试次
    sub_id = get_sub_id_from_name(p)
    yy_vals = df.loc[df["cond"] == "YY", DV_COL].astype(float)
    kj_vals = df.loc[df["cond"] == "KJ", DV_COL].astype(float)

    # 保存到独立样本（补充分析）堆里
    if len(yy_vals) > 0:
        all_trials_for_ind.append(pd.DataFrame({"sub": sub_id, "cond": "YY", DV_COL: yy_vals.values}))
    if len(kj_vals) > 0:
        all_trials_for_ind.append(pd.DataFrame({"sub": sub_id, "cond": "KJ", DV_COL: kj_vals.values}))

    # 只要两边都有数据，才能纳入“成对 t 检验”的被试级均值
    if len(yy_vals) > 0 and len(kj_vals) > 0:
        rows.append({
            "sub": sub_id,
            "YY_mean": yy_vals.mean(),
            "KJ_mean": kj_vals.mean(),
            "YY_n": len(yy_vals),
            "KJ_n": len(kj_vals)
        })

# 组织被试级数据
sub_df = pd.DataFrame(rows).sort_values("sub")
if sub_df.empty:
    raise RuntimeError("没有找到同时含有 YY 与 KJ 条件的被试，无法进行成对 t 检验。")

sub_df["diff"] = sub_df["YY_mean"] - sub_df["KJ_mean"]
sub_df.to_csv(os.path.join(DATA_DIR, "run7行为结果数据.csv"), index=False, encoding="utf-8-sig")

# ========= 成对样本 t 检验（主分析） =========
yy = sub_df["YY_mean"].values
kj = sub_df["KJ_mean"].values
diff = sub_df["diff"].values
n = len(diff)

t_stat, p_val = stats.ttest_rel(yy, kj, nan_policy="omit")

# 效应量 Cohen's dz（成对）：mean(diff)/sd(diff)
dz = diff.mean() / diff.std(ddof=1)

# 95% CI for mean difference（配对）： mean(diff) ± t_{n-1,0.975} * SE
se = diff.std(ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_low = diff.mean() - t_crit * se
ci_high = diff.mean() + t_crit * se

# 正态性（对差值做 Shapiro）
sh_stat, sh_p = stats.shapiro(diff) if n <= 5000 else (np.nan, np.nan)

# ========= 独立样本 Welch t（补充） =========
ind_df = pd.concat(all_trials_for_ind, ignore_index=True) if len(all_trials_for_ind) > 0 else pd.DataFrame()
welch = None
hedges_g = None
if not ind_df.empty:
    yy_trials = ind_df.loc[ind_df["cond"] == "YY", DV_COL].astype(float).values
    kj_trials = ind_df.loc[ind_df["cond"] == "KJ", DV_COL].astype(float).values
    if len(yy_trials) > 1 and len(kj_trials) > 1:
        t_welch, p_welch = stats.ttest_ind(yy_trials, kj_trials, equal_var=False, nan_policy="omit")
        # Hedges' g（独立样本效应量）
        n1, n2 = len(yy_trials), len(kj_trials)
        s1, s2 = yy_trials.var(ddof=1), kj_trials.var(ddof=1)
        # pooled sd (unbiased)
        sp = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
        d = (yy_trials.mean() - kj_trials.mean()) / sp
        J = 1 - (3 / (4*(n1 + n2) - 9))  # 小样本校正
        hedges_g = d * J
        welch = (t_welch, p_welch, hedges_g, n1, n2)

# ========= 打印结果 =========
print("=== 成对样本 t 检验（主分析，YY_mean vs KJ_mean） ===")
print(f"被试数 n = {n}")
print(f"YY 平均(被试级) = {yy.mean():.4f} ± {yy.std(ddof=1):.4f}")
print(f"KJ 平均(被试级) = {kj.mean():.4f} ± {kj.std(ddof=1):.4f}")
print(f"差值(YY-KJ) = {diff.mean():.4f} 95%CI [{ci_low:.4f}, {ci_high:.4f}]")
print(f"t({n-1}) = {t_stat:.4f}, p = {p_val:.6f}")
print(f"Cohen's dz = {dz:.3f}")
if not np.isnan(sh_stat):
    print(f"Shapiro 正态性（差值）：W = {sh_stat:.4f}, p = {sh_p:.6f}  (p>0.05 则差值近似正态)")

if welch is not None:
    t_welch, p_welch, hedges_g, n1, n2 = welch
    print("\n=== 独立样本 Welch t（补充，不按被试配对聚合） ===")
    print(f"YY 试次数 = {n1}, KJ 试次数 = {n2}")
    print(f"t ≈ {t_welch:.4f}, p = {p_welch:.6f}, Hedges' g = {hedges_g:.3f}")
else:
    print("\n（未执行 Welch 独立样本 t，因为逐试次数据不足或不存在。）")

print(f"\n已导出被试级均值表：{os.path.join(DATA_DIR, 'run7行为结果数据.csv')}")
