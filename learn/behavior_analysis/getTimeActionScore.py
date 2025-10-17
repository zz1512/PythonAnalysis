#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为数据反应时分析脚本：隐喻-空间条件反应时比较

功能描述:
    对run-7的行为数据进行反应时统计分析，比较YY条件和KJ条件下的反应时差异。
    使用成对样本t检验分析被试内条件差异，并提供独立样本t检验作为补充分析。

分析指标:
    - 因变量：action_time（记忆测试的反应时，单位：毫秒）
    - 自变量：条件类型（YY vs KJ）

分析方法:
    1. 成对样本t检验（主分析）：比较每个被试在YY和KJ条件下的平均反应时
    2. 独立样本Welch t检验（补充）：不按被试配对，直接比较所有YY和KJ试次
    3. 效应量计算：Cohen's dz（配对）和Hedges' g（独立）
    4. 正态性检验：Shapiro-Wilk检验

预期结果示例:
    被试数 n = 27
    YY 平均(被试级) = 1028.2862 ± 199.7187
    KJ 平均(被试级) = 1039.8540 ± 200.8450
    差值(YY-KJ) = -11.5677 95%CI [-35.8265, 12.6911]
    t(26) = -0.9802, p = 0.336035
    Cohen's dz = -0.189

注意事项:
    - 反应时数据通常呈右偏分布，建议检查正态性假设
    - 异常值可能对反应时分析产生较大影响
    - 结果保存到../../result/run7_action_time.csv

作者: 研究团队
版本: 1.0
日期: 2024
"""

import os, re, glob
import numpy as np
import pandas as pd
from scipy import stats
# === 分析配置参数 ===
DATA_DIR = r"../../data_events"        # 行为数据文件目录
OUT_DIR = r"../../result"              # 结果输出目录
FILE_PATTERN = "*run-7_events.tsv"     # 文件匹配模式
DV_COL = "action_time"                 # 因变量列名（反应时指标）

# 实验条件定义
YY_SET = {"yyw", "yyew"}               # YY条件的trial_type值集合
KJ_SET = {"kjw", "kjew"}               # KJ条件的trial_type值集合

# 文件读取说明：
# - 自动识别TSV/CSV格式
# - 优先使用制表符分隔符
# - 支持递归搜索子目录

def smart_read(path: str) -> pd.DataFrame:
    """
    智能读取CSV/TSV文件
    
    根据文件扩展名自动选择合适的分隔符读取数据文件。
    
    Args:
        path (str): 文件路径
    
    Returns:
        pd.DataFrame: 读取的数据框
    
    Note:
        - .tsv文件使用制表符分隔
        - 其他格式默认尝试制表符，失败则使用逗号分隔（兜底策略）
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    # 兜底：尝试TSV再CSV
    try:
        return pd.read_csv(path, sep="\t")
    except:
        return pd.read_csv(path)

def get_sub_id_from_name(fn: str) -> str:
    """
    从文件名里提取被试ID（如 'sub-01' 或 '01'）
    
    使用正则表达式从文件名中提取被试编号，并标准化为两位数格式。
    支持多种文件名格式的被试ID提取。
    
    Args:
        fn (str): 文件名或文件路径
    
    Returns:
        str: 标准化的被试ID（如'sub-01'）或文件基名（提取失败时）
    
    Examples:
        >>> get_sub_id_from_name('sub-1_run-7_events.tsv')
        'sub-01'
        >>> get_sub_id_from_name('sub_15_data.csv')
        'sub-15'
        >>> get_sub_id_from_name('participant_03.tsv')
        'sub-03'
    
    Note:
        - 优先匹配'sub-数字'或'sub_数字'格式
        - 其次匹配任意数字序列
        - 提取不到时返回文件基名（不含扩展名）
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

# === 文件收集阶段 ===
# 递归搜索匹配的数据文件
paths = sorted(glob.glob(os.path.join(DATA_DIR, "**", FILE_PATTERN), recursive=True))
if len(paths) == 0:
    # 兜底策略：尝试直接在DATA_DIR下搜索
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*run-7_events.tsv")))

if len(paths) == 0:
    raise FileNotFoundError(f"未在目录 {DATA_DIR} 中找到匹配 {FILE_PATTERN} 的文件")

print(f"找到 {len(paths)} 个数据文件，开始处理...")

# === 数据处理阶段 ===
rows = []                    # 存储被试级汇总数据（用于配对t检验）
all_trials_for_ind = []      # 存储试次级数据（用于独立样本Welch t检验）

for i, p in enumerate(paths, start=1):
    try:
        # 读取数据文件
        df = smart_read(p)
        
        # 数据完整性检查
        if "trial_type" not in df.columns:
            raise ValueError(f"{p} 中缺少 'trial_type' 列")
        if DV_COL not in df.columns:
            raise ValueError(f"{p} 中缺少指标列 '{DV_COL}'")
        
        # 数据清洗：移除缺失值
        original_rows = len(df)
        df = df.dropna(subset=[DV_COL])
        if len(df) < original_rows:
            print(f"  警告：{os.path.basename(p)} 移除了 {original_rows - len(df)} 行缺失数据")
        
        # 条件分类：根据trial_type划分YY/KJ条件
        df["cond"] = np.where(df["trial_type"].isin(YY_SET), "YY",
                       np.where(df["trial_type"].isin(KJ_SET), "KJ", "OTHER"))
        
        # 提取被试ID和条件数据
        sub_id = get_sub_id_from_name(p)
        yy_vals = df.loc[df["cond"] == "YY", DV_COL].astype(float)
        kj_vals = df.loc[df["cond"] == "KJ", DV_COL].astype(float)
        
        # 收集试次级数据（用于独立样本Welch t检验）
        if len(yy_vals) > 0:
            all_trials_for_ind.append(pd.DataFrame({"sub": sub_id, "cond": "YY", DV_COL: yy_vals.values}))
        if len(kj_vals) > 0:
            all_trials_for_ind.append(pd.DataFrame({"sub": sub_id, "cond": "KJ", DV_COL: kj_vals.values}))
        
        # 计算被试级均值（用于配对t检验）
        if len(yy_vals) > 0 and len(kj_vals) > 0:
            rows.append({
                "sub": sub_id,
                "YY_mean": yy_vals.mean(),
                "KJ_mean": kj_vals.mean(),
                "YY_n": len(yy_vals),
                "KJ_n": len(kj_vals)
            })
            print(f"[{i:2d}/{len(paths)}] {sub_id}: YY={len(yy_vals)}试次({yy_vals.mean():.1f}ms), KJ={len(kj_vals)}试次({kj_vals.mean():.1f}ms)")
        else:
            print(f"[{i:2d}/{len(paths)}] {sub_id}: 跳过（YY={len(yy_vals)}, KJ={len(kj_vals)}试次）")
            
    except Exception as e:
        print(f"[{i:2d}/{len(paths)}] 处理 {os.path.basename(p)} 时出错: {e}")
        continue

# === 统计分析阶段 ===
print(f"\n数据处理完成，共收集到 {len(rows)} 个有效被试")

# 构建被试级汇总数据框
sub_df = pd.DataFrame(rows).sort_values("sub")
if sub_df.empty:
    raise RuntimeError("没有找到同时含有 YY 与 KJ 条件的被试，无法进行成对 t 检验")

# 计算条件差值（YY - KJ）
sub_df["diff"] = sub_df["YY_mean"] - sub_df["KJ_mean"]

# 保存被试级结果
output_file = os.path.join(OUT_DIR, "run7_action_time.csv")
os.makedirs(OUT_DIR, exist_ok=True)
sub_df.to_csv(output_file, index=False, encoding="utf-8-sig")

# ========= 成对样本 t 检验（主分析） =========
yy = sub_df["YY_mean"].values
kj = sub_df["KJ_mean"].values
diff = sub_df["diff"].values
n = len(diff)

# 执行成对t检验
t_stat, p_val = stats.ttest_rel(yy, kj, nan_policy="omit")

# 计算效应量：Cohen's dz（配对设计专用）
dz = diff.mean() / diff.std(ddof=1)

# 计算95%置信区间
se = diff.std(ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_low = diff.mean() - t_crit * se
ci_high = diff.mean() + t_crit * se

# 正态性检验：Shapiro-Wilk检验（检验差值分布）
sh_stat, sh_p = stats.shapiro(diff) if n <= 5000 else (np.nan, np.nan)

# === 补充分析：独立样本Welch t检验 ===
# 将所有试次级数据合并（不按被试聚合）
ind_df = pd.concat(all_trials_for_ind, ignore_index=True) if len(all_trials_for_ind) > 0 else pd.DataFrame()
welch = None
hedges_g = None

if not ind_df.empty:
    # 提取YY和KJ条件的所有试次数据
    yy_trials = ind_df.loc[ind_df["cond"] == "YY", DV_COL].astype(float).values
    kj_trials = ind_df.loc[ind_df["cond"] == "KJ", DV_COL].astype(float).values
    
    if len(yy_trials) > 1 and len(kj_trials) > 1:
        # 执行Welch t检验（不假设方差齐性）
        t_welch, p_welch = stats.ttest_ind(yy_trials, kj_trials, equal_var=False, nan_policy="omit")
        
        # 计算Hedges' g效应量（独立样本，含小样本校正）
        n1, n2 = len(yy_trials), len(kj_trials)
        s1, s2 = yy_trials.var(ddof=1), kj_trials.var(ddof=1)
        
        # 计算合并标准差
        sp = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
        d = (yy_trials.mean() - kj_trials.mean()) / sp
        
        # 小样本偏差校正因子
        J = 1 - (3 / (4*(n1 + n2) - 9))
        hedges_g = d * J
        
        welch = (t_welch, p_welch, hedges_g, n1, n2)

# === 结果输出 ===
print("\n" + "="*60)
print("=== 成对样本 t 检验（主分析，YY_mean vs KJ_mean） ===")
print(f"被试数 n = {n}")
print(f"YY 平均(被试级) = {yy.mean():.4f} ± {yy.std(ddof=1):.4f} ms")
print(f"KJ 平均(被试级) = {kj.mean():.4f} ± {kj.std(ddof=1):.4f} ms")
print(f"差值(YY-KJ) = {diff.mean():.4f} ms, 95%CI [{ci_low:.4f}, {ci_high:.4f}]")
print(f"t({n-1}) = {t_stat:.4f}, p = {p_val:.6f}")
print(f"Cohen's dz = {dz:.3f}")

# 正态性检验结果
if not np.isnan(sh_stat):
    normality_status = "正态" if sh_p > 0.05 else "非正态"
    print(f"Shapiro 正态性检验（差值）：W = {sh_stat:.4f}, p = {sh_p:.6f} ({normality_status})")

# 补充分析结果
if welch is not None:
    t_welch, p_welch, hedges_g, n1, n2 = welch
    print("\n=== 独立样本 Welch t检验（补充分析，试次级数据） ===")
    print(f"YY 试次数 = {n1}, KJ 试次数 = {n2}")
    print(f"YY 试次均值 = {yy_trials.mean():.4f} ms, KJ 试次均值 = {kj_trials.mean():.4f} ms")
    print(f"t ≈ {t_welch:.4f}, p = {p_welch:.6f}, Hedges' g = {hedges_g:.3f}")
else:
    print("\n（跳过独立样本Welch t检验：试次级数据不足）")

# 文件保存信息
print(f"\n结果已保存至：{output_file}")
print("="*60)