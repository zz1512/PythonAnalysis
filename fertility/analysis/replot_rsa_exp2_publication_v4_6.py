#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replot_rsa_exp2_publication_v4_6.py

仅基于已保存结果表格/plot_data 重新绘图，不重新分析。
依赖目录（由 v4.6 主分析脚本生成）：
- C:/python_fertility/rsa_results_exp2/tables
- C:/python_fertility/rsa_results_exp2/plot_data
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = Path(r"C:/python_fertility")
RSA_DIR = BASE_DIR / "rsa_results_exp2"
TABLE_DIR = RSA_DIR / "tables"
PLOT_DATA_DIR = RSA_DIR / "plot_data"
FIG_DIR = RSA_DIR / "figures_publication_replot"

COND_ORDER = ["fer", "soc", "val", "hed", "exi", "atf"]
COND_LABELS = {
    "fer": "生育威胁",
    "soc": "人际威胁",
    "val": "价值威胁",
    "hed": "享乐威胁",
    "exi": "生存威胁",
    "atf": "不生育威胁",
}
HEATMAP_CMAP = "coolwarm"
HEATMAP_VMIN = -0.2
HEATMAP_VMAX = 0.4
TRIAL_VMIN = -0.2
TRIAL_VMAX = 0.4
FIG_DPI = 200
HEATMAP_SHOW_TEXT = True

def setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

def p_to_star(p):
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def get_condition_block_info(sorted_labels):
    counts = [sum(1 for x in sorted_labels if x == cond) for cond in COND_ORDER]
    boundaries = np.cumsum(counts).tolist()
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    centers = (starts + np.array(counts) / 2.0 - 0.5).tolist()
    tick_labels = [COND_LABELS[c] for c in COND_ORDER]
    return boundaries, centers, tick_labels

def plot_heatmap6x6():
    src_dir = PLOT_DATA_DIR / "heatmaps_6x6"
    out_dir = FIG_DIR / "heatmaps_6x6"
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(src_dir.glob("heatmap6x6_*.csv")):
        roi_name = csv_path.stem.replace("heatmap6x6_", "")
        mat = pd.read_csv(csv_path, index_col=0)
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        im = ax.imshow(mat.values, cmap=HEATMAP_CMAP, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS[x] for x in COND_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([COND_LABELS[x] for x in COND_ORDER])
        ax.set_title(f"{roi_name}：组平均类别相似性矩阵")
        if HEATMAP_SHOW_TEXT:
            for i in range(len(COND_ORDER)):
                for j in range(len(COND_ORDER)):
                    val = mat.values[i, j]
                    color = "white" if val < 0.1 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.2, color=color)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("相似性 (r)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{roi_name}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

def plot_violin():
    out_dir = FIG_DIR / "violin"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PLOT_DATA_DIR / "violin" / "violin_plotdata.csv")
    for roi_name in sorted(df["roi"].unique()):
        roi_df = df[df["roi"] == roi_name].copy()
        tests = list(roi_df["test"].unique())
        if not tests:
            continue
        fig, axes = plt.subplots(1, len(tests), figsize=(6.4 * len(tests), 5.4), sharey=True)
        if len(tests) == 1:
            axes = [axes]
        for ax, test_name in zip(axes, tests):
            sub_df = roi_df[roi_df["test"] == test_name].copy()
            left_vals = sub_df["left_r"].to_numpy(dtype=float)
            right_vals = sub_df["right_r"].to_numpy(dtype=float)
            positions = [0, 1]
            parts = ax.violinplot([left_vals, right_vals], positions=positions, widths=0.7, showmeans=False, showmedians=True)
            for i, body in enumerate(parts["bodies"]):
                body.set_alpha(0.42)
                body.set_facecolor(["#d62728", "#1f77b4"][i])
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
            for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.8)
            for _, row in sub_df.iterrows():
                ax.plot(positions, [row["left_r"], row["right_r"]], color="gray", alpha=0.2, linewidth=0.9)
                ax.scatter(positions, [row["left_r"], row["right_r"]], color=["#d62728", "#1f77b4"], s=13, alpha=0.7)
            mean_left = np.nanmean(left_vals)
            mean_right = np.nanmean(right_vals)
            ax.plot(positions, [mean_left, mean_right], color="black", linewidth=2.0)
            ax.scatter(positions, [mean_left, mean_right], color="black", s=32, zorder=4)
            left_pair = sub_df["left_pair"].iloc[0]
            right_pair = sub_df["right_pair"].iloc[0]
            p = sub_df["p_one_sided"].iloc[0]
            q = sub_df["q_fdr_one_sided"].iloc[0]
            t = sub_df["t"].iloc[0]
            label_p = q if np.isfinite(q) else p
            stars = p_to_star(label_p)
            ymax = np.nanmax([np.nanmax(left_vals), np.nanmax(right_vals)])
            ymin = np.nanmin([np.nanmin(left_vals), np.nanmin(right_vals)])
            yspan = max(0.02, ymax - ymin)
            y = ymax + 0.12 * yspan
            h = 0.04 * yspan
            ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], color="black", linewidth=1.0)
            ax.text(0.5, y+h+0.01*yspan, stars, ha="center", va="bottom", fontsize=13)
            txt = f"t={t:.2f}\np={p:.3f}"
            if np.isfinite(q):
                txt += f"\nq={q:.3f}"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
            ax.set_xticks(positions)
            ax.set_xticklabels([left_pair, right_pair], rotation=20)
            ax.set_title(test_name)
            ax.set_ylabel("相似性 (r)")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        fig.suptitle(f"{roi_name}：核心假设 left/right 分布", y=1.02, fontsize=13)
        plt.tight_layout()
        plt.savefig(out_dir / f"violin_hypothesis_{roi_name}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

def plot_model_fit():
    out_dir = FIG_DIR / "model_fit"
    out_dir.mkdir(parents=True, exist_ok=True)
    subj = pd.read_csv(PLOT_DATA_DIR / "model_fit" / "model_fit_subjectwise.csv")
    grp = pd.read_csv(PLOT_DATA_DIR / "model_fit" / "model_fit_group.csv")
    model_name_map = {
        "M1_fer_exi_only": "M1: 生育≈生存",
        "M2_atf_hed_only": "M2: 不生育≈享乐",
        "M3_combined": "M3: 组合模型",
    }
    for roi_name in sorted(grp["roi"].unique()):
        roi_group = grp[grp["roi"] == roi_name].copy()
        roi_subj = subj[subj["roi"] == roi_name].copy()
        order = roi_group.sort_values("mean_rho", ascending=False)["model"].tolist()
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        for i, model_name in enumerate(order):
            vals = roi_subj[roi_subj["model"] == model_name]["rho"].dropna().to_numpy(dtype=float)
            if len(vals):
                rng = np.random.default_rng(42 + i)
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, alpha=0.35, color="gray")
            row = roi_group[roi_group["model"] == model_name].iloc[0]
            mean_rho = row["mean_rho"]
            ci = np.nan
            if len(vals) >= 2:
                se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
                ci = 1.96 * se
            ax.scatter([i], [mean_rho], color="black", s=55, zorder=3)
            if np.isfinite(ci):
                ax.plot([i, i], [mean_rho - ci, mean_rho + ci], color="black", linewidth=1.4)
            q = row.get("q_fdr_one_sided", np.nan)
            p = row.get("p_one_sided", np.nan)
            stars = p_to_star(q if np.isfinite(q) else p)
            label = f"{stars}\nq={q:.3f}" if np.isfinite(q) else f"{stars}\np={p:.3f}"
            ax.text(i, mean_rho + 0.03, label, ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([model_name_map.get(m, m) for m in order], rotation=15)
        ax.set_ylabel("模型拟合（Spearman rho）")
        best_model = model_name_map.get(order[0], order[0]) if order else "NA"
        ax.set_title(f"{roi_name}：模型拟合比较\n最佳模型 = {best_model}", fontsize=13)
        plt.tight_layout()
        plt.savefig(out_dir / f"model_fit_{roi_name}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

def plot_trial_heatmaps():
    src_dir = PLOT_DATA_DIR / "trial_heatmaps"
    out_dir = FIG_DIR / "trial_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(src_dir.glob("trial_heatmap_group_*.csv")):
        if csv_path.name.endswith("_order.csv"):
            continue
        roi_name = csv_path.stem.replace("trial_heatmap_group_", "")
        mat = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=float)
        order_df = pd.read_csv(src_dir / f"trial_heatmap_group_{roi_name}_order.csv")
        sorted_labels = order_df["label"].tolist()
        vmin = max(np.nanquantile(mat[np.isfinite(mat) & (np.abs(mat) < 0.999)], 0.05), TRIAL_VMIN) if np.isfinite(mat).any() else TRIAL_VMIN
        vmax = min(np.nanquantile(mat[np.isfinite(mat) & (np.abs(mat) < 0.999)], 0.95), TRIAL_VMAX) if np.isfinite(mat).any() else TRIAL_VMAX
        fig, ax = plt.subplots(figsize=(8.8, 7.6))
        im = ax.imshow(mat, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
        boundaries, centers, tick_labels = get_condition_block_info(sorted_labels)
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="white", linewidth=1.3, alpha=0.95)
            ax.axvline(b - 0.5, color="white", linewidth=1.3, alpha=0.95)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_title(f"组平均：{roi_name} trial×trial 相似性\nrange = [{vmin:.2f}, {vmax:.2f}]", fontsize=11.5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("相似性 (r)")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_trials_group_{roi_name}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

def main():
    setup_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap6x6()
    plot_violin()
    plot_model_fit()
    plot_trial_heatmaps()
    print("完成：已基于保存表格重新绘制论文风格图。")

if __name__ == "__main__":
    main()
