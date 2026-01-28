import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _find_latest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-1]


def _auto_find_result_csv() -> Optional[Path]:
    roots = [
        Path.cwd(),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis"),
    ]
    candidates: List[Path] = []
    for r in roots:
        if r.exists():
            candidates.extend(list(r.rglob("joint_analysis_dev_models_perm_fwer.csv")))
    return _find_latest(candidates)


def _parse_star_thresholds(stars: str) -> List[float]:
    parts = [p.strip() for p in str(stars).split(",") if p.strip()]
    vals = [float(x) for x in parts]
    vals = [x for x in vals if x > 0]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("--stars 不能为空，例如 '0.05,0.01,0.001'")
    return vals


def _matrix_sort_key(name: str) -> Tuple[int, int, str]:
    if name == "surface_L":
        return (0, 0, name)
    if name == "surface_R":
        return (0, 1, name)
    m = re.match(r"surface_([LR])_(.*)", name)
    if m:
        hemi = 0 if m.group(1) == "L" else 1
        roi = m.group(2)
        return (1, hemi, roi)
    return (9, 9, name)


def _model_sort_key(name: str) -> int:
    order = {"M_nn": 0, "M_conv": 1, "M_div": 2}
    return order.get(name, 9)


def _stars_for_p(p: float, thresholds: List[float]) -> str:
    if not np.isfinite(p):
        return ""
    k = 0
    for t in thresholds:
        if p <= t:
            k += 1
    return "*" * k


def _format_annot(r: float, p: float, thresholds: List[float]) -> str:
    if not np.isfinite(r):
        return ""
    return f"{r:.2f}{_stars_for_p(p, thresholds)}"


def plot_heatmap(
    result_csv: Path,
    out_dir: Path,
    star_thresholds: List[float],
    dpi: int,
    fmt: str,
) -> List[Path]:
    df = pd.read_csv(result_csv)
    p_col = "p_fwer"
    required = {"matrix", "model", "r_obs", p_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"结果文件缺少列: {sorted(missing)}")

    matrices = sorted(df["matrix"].astype(str).unique().tolist(), key=_matrix_sort_key)
    models = sorted(df["model"].astype(str).unique().tolist(), key=_model_sort_key)

    pivot_r = df.pivot(index="matrix", columns="model", values="r_obs").reindex(index=matrices, columns=models)
    pivot_p = df.pivot(index="matrix", columns="model", values=p_col).reindex(index=matrices, columns=models)

    annot = pivot_r.copy()
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            annot.iat[i, j] = _format_annot(pivot_r.iat[i, j], pivot_p.iat[i, j], star_thresholds)

    n_subjects = int(df["n_subjects"].dropna().iloc[0]) if "n_subjects" in df.columns and df["n_subjects"].notna().any() else None
    n_perm = int(df["n_perm"].dropna().iloc[0]) if "n_perm" in df.columns and df["n_perm"].notna().any() else None

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    v = pivot_r.to_numpy()
    vmax = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
    vmax = max(vmax, 0.05)

    fig_w = 5.2
    fig_h = max(3.8, 0.45 * len(matrices) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot_r,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "r (Similarity vs. Development Model)"},
    )

    title = "Joint Analysis (Permutation + FWER; p_fwer)"
    if n_subjects is not None:
        title += f"\nN={n_subjects}"
    if n_perm is not None:
        title += f", Permutations={n_perm}"
    ax.set_title(title, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    levels = sorted(star_thresholds)
    legend_parts = []
    for i, t in enumerate(levels[::-1], start=1):
        legend_parts.append(f"{'*' * (len(levels) - i + 1)} p≤{t:g}")
    legend_text = ", ".join(legend_parts)
    fig.text(0.99, 0.01, legend_text, ha="right", va="bottom", fontsize=10)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for ext in [fmt]:
        out_path = out_dir / f"joint_analysis_dev_models_heatmap_p_fwer.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="绘制 joint_analysis_dev_models_perm_fwer.csv 论文图")
    p.add_argument("--input-csv", type=Path, default=None, help="joint_analysis_dev_models_perm_fwer.csv 路径（不填则自动搜索最新文件）")
    p.add_argument("--out-dir", type=Path, default=Path("./figures_joint"), help="输出目录")
    p.add_argument("--stars", type=str, default="0.05,0.01,0.001", help="星号阈值（逗号分隔，升序或降序均可）")
    p.add_argument("--dpi", type=int, default=300, help="输出分辨率")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="输出格式")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    star_thresholds = _parse_star_thresholds(args.stars)
    in_csv = Path(args.input_csv) if args.input_csv is not None else None
    if in_csv is None:
        in_csv = _auto_find_result_csv()
    if in_csv is None or not in_csv.exists():
        raise FileNotFoundError("未找到 joint_analysis_dev_models_perm_fwer.csv，请用 --input-csv 指定或在默认目录生成后再运行。")
    outs = plot_heatmap(
        result_csv=in_csv,
        out_dir=args.out_dir,
        star_thresholds=star_thresholds,
        dpi=int(args.dpi),
        fmt=str(args.format),
    )
    for p in outs:
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
