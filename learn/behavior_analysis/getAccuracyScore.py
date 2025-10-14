import os, re, glob
import numpy as np
import pandas as pd
from scipy import stats

# ========= 行为数据分析：获取隐喻-空间的T检验结果 =========
"""
第12个被试没做行为
被试数 n = 27
YY 平均(被试级) = 0.6799 ± 0.1713
KJ 平均(被试级) = 0.5878 ± 0.2350
差值(YY-KJ) = 0.0921 95%CI [0.0473, 0.1369]
t(26) = 4.2232, p = 0.000261
Cohen's dz = 0.813
Shapiro 正态性（差值）：W = 0.9508, p = 0.224628  (p>0.05 则差值近似正态)
"""
# ========= 配置区（按你的实际情况改这几项） =========
DATA_DIR = r"../../data_events"  # 28个文件所在的目录
FILE_PATTERN = "*run-7_events.tsv"                    # 通配符，例：'*events.tsv' 或 '*events.csv'；默认两者都行
DV_COL = "memory"                           # 你的记忆效果指标列名，比如 'accuracy' 或 'memory_score' 或 'rt'
YY_SET = {"yyw", "yyew"}
KJ_SET = {"kjw", "kjew"}

# 若你的文件是TSV，请设置默认分隔符优先为'\t'
# 代码会根据后缀自动判断，后缀不明时用逗号。
# ===============================================

def smart_read(path: str) -> pd.DataFrame:
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
    从文件名里提取被试ID（如 'sub-01' 或 '01'）。提取不到就返回文件基名。
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