#!/usr/bin/env python3
"""ROI beta matrix validator (single-file config version).

说明：
- 配置已内置在本脚本 `CONFIG` 字典里，不再依赖单独 JSON 配置文件。
- 你只需要修改 CONFIG 中的路径和阈值，然后直接运行本脚本。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ===================== 在这里改配置（无需命令行参数） =====================
CONFIG: dict[str, Any] = {
    "file": "/home/dingrui/fmri_analysis/zz_analysis/roi_results/by_stimulus/Passive_Emo/roi_beta_matrix_200.npz",
    "key": None,
    "expect_size": 200,
    "symmetry_atol": 1e-6,
    "preview_rows": 5,
    "preview_cols": 5,
    "max_slice_brief": 10,
    "save_report": "brain_research/emo_new/data_check/roi_beta_matrix_200_report.md",
    "save_json": "brain_research/emo_new/data_check/roi_beta_matrix_200_report.json",
}
# ======================================================================


@dataclass
class MatrixStats:
    shape: tuple[int, ...]
    dtype: str
    min_val: float
    max_val: float
    mean: float
    std: float
    finite_ratio: float
    nan_count: int
    inf_count: int
    zero_ratio: float
    symmetry_max_abs_diff: float | None
    diagonal_min: float | None
    diagonal_max: float | None
    p01: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


def _validated_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    if not str(cfg.get("file", "")).strip():
        raise ValueError("CONFIG['file'] 不能为空")
    cfg["expect_size"] = int(cfg.get("expect_size", 200))
    cfg["symmetry_atol"] = float(cfg.get("symmetry_atol", 1e-6))
    cfg["preview_rows"] = int(cfg.get("preview_rows", 5))
    cfg["preview_cols"] = int(cfg.get("preview_cols", 5))
    cfg["max_slice_brief"] = int(cfg.get("max_slice_brief", 10))
    cfg.setdefault("key", None)
    cfg.setdefault("save_report", None)
    cfg.setdefault("save_json", None)
    return cfg


def _load_array(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), "(npy)"

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz_data:
            keys = list(npz_data.keys())
            if not keys:
                raise ValueError(f"npz 文件中没有数组: {path}")

            if key is not None:
                if key not in npz_data:
                    raise KeyError(f"指定 key 不存在: {key}; 可用 keys={keys}")
                return npz_data[key], key

            if len(keys) == 1:
                only = keys[0]
                return npz_data[only], only

            candidate_key = max(keys, key=lambda k: (npz_data[k].ndim, npz_data[k].size))
            return npz_data[candidate_key], candidate_key

    raise ValueError(f"仅支持 .npy/.npz，当前文件后缀: {suffix}")


def compute_stats(mat: np.ndarray) -> MatrixStats:
    finite_mask = np.isfinite(mat)
    finite_values = mat[finite_mask]
    if finite_values.size == 0:
        raise ValueError("矩阵不包含任何有限数值，无法计算统计量")

    symmetry_max_abs_diff: float | None = None
    diagonal_min: float | None = None
    diagonal_max: float | None = None

    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        symmetry_max_abs_diff = float(np.max(np.abs(mat - mat.T)))
        diag = np.diag(mat)
        diagonal_min = float(np.min(diag))
        diagonal_max = float(np.max(diag))

    q = np.percentile(finite_values, [1, 5, 25, 50, 75, 95, 99])

    return MatrixStats(
        shape=tuple(int(v) for v in mat.shape),
        dtype=str(mat.dtype),
        min_val=float(np.min(finite_values)),
        max_val=float(np.max(finite_values)),
        mean=float(np.mean(finite_values)),
        std=float(np.std(finite_values)),
        finite_ratio=float(np.mean(finite_mask)),
        nan_count=int(np.isnan(mat).sum()),
        inf_count=int(np.isinf(mat).sum()),
        zero_ratio=float(np.mean(mat == 0)),
        symmetry_max_abs_diff=symmetry_max_abs_diff,
        diagonal_min=diagonal_min,
        diagonal_max=diagonal_max,
        p01=float(q[0]),
        p05=float(q[1]),
        p25=float(q[2]),
        p50=float(q[3]),
        p75=float(q[4]),
        p95=float(q[5]),
        p99=float(q[6]),
    )


def _matrix_preview(mat: np.ndarray, rows: int, cols: int) -> str:
    if mat.ndim < 2:
        return "(矩阵维度 < 2，无法展示视图)"
    return np.array2string(mat[..., :rows, :cols], precision=4, suppress_small=False)


def _report_single(name: str, mat: np.ndarray, cfg: dict[str, Any]) -> str:
    st = compute_stats(mat)

    checks: list[tuple[str, bool, str]] = [
        ("维度是否>=2", mat.ndim >= 2, f"ndim={mat.ndim}"),
        ("是否不含 NaN", st.nan_count == 0, f"nan_count={st.nan_count}"),
        ("是否不含 Inf", st.inf_count == 0, f"inf_count={st.inf_count}"),
    ]

    if mat.ndim >= 2:
        checks.append(("最后两维是否为方阵", mat.shape[-2] == mat.shape[-1], f"shape[-2:]={mat.shape[-2:]}"))
        checks.append(
            (
                f"最后两维是否为 ({cfg['expect_size']}, {cfg['expect_size']})",
                mat.shape[-2:] == (cfg["expect_size"], cfg["expect_size"]),
                f"shape[-2:]={mat.shape[-2:]}",
            )
        )

    if st.symmetry_max_abs_diff is not None:
        checks.append(
            (
                f"是否近似对称(abs_diff <= {cfg['symmetry_atol']})",
                st.symmetry_max_abs_diff <= cfg["symmetry_atol"],
                f"max_abs_diff={st.symmetry_max_abs_diff:.6g}",
            )
        )

    lines = [
        f"\n==================== {name} ====================",
        "📊 矩阵基本信息：",
        f"  - 形状（维度）: {st.shape}",
        f"  - 数据类型: {st.dtype}",
        f"  - 数值范围: [{st.min_val:.6f}, {st.max_val:.6f}]",
        f"  - 均值/标准差: {st.mean:.6f} ± {st.std:.6f}",
        f"  - 有限值占比: {st.finite_ratio:.4%}",
        f"  - 零值占比: {st.zero_ratio:.4%}",
        "\n📈 分位数（有限值）：",
        (
            "  - p01/p05/p25/p50/p75/p95/p99: "
            f"{st.p01:.6f} / {st.p05:.6f} / {st.p25:.6f} / {st.p50:.6f} / "
            f"{st.p75:.6f} / {st.p95:.6f} / {st.p99:.6f}"
        ),
    ]

    if st.symmetry_max_abs_diff is not None:
        lines.extend(
            [
                "\n🧭 方阵性质：",
                f"  - 对称性最大绝对误差 |A-A^T|_max: {st.symmetry_max_abs_diff:.6g}",
                f"  - 对角线范围: [{st.diagonal_min:.6f}, {st.diagonal_max:.6f}]",
            ]
        )

    lines.append(f"\n🔍 矩阵前{cfg['preview_rows']}x{cfg['preview_cols']}内容：")
    lines.append(_matrix_preview(mat, cfg["preview_rows"], cfg["preview_cols"]))

    if mat.ndim > 2:
        first = mat.reshape(-1, mat.shape[-2], mat.shape[-1])[0]
        lines.append("\n🔍 第一张切片（第0个条件）预览：")
        lines.append(np.array2string(first[: cfg["preview_rows"], : cfg["preview_cols"]], precision=4, suppress_small=False))

    lines.append("\n✅ 条件校验结果：")
    for title, ok, detail in checks:
        lines.append(f"  - {'✅' if ok else '❌'} {title} ({detail})")

    if mat.ndim > 2 and mat.shape[-2] == mat.shape[-1]:
        flat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
        sym_errs = np.max(np.abs(flat - np.transpose(flat, (0, 2, 1))), axis=(1, 2))
        lines.extend(
            [
                "\n🧪 多条件切片分析：",
                f"  - 切片数量: {flat.shape[0]}",
                (
                    "  - 每个切片对称误差统计 (min/median/max): "
                    f"{float(np.min(sym_errs)):.6g} / {float(np.median(sym_errs)):.6g} / {float(np.max(sym_errs)):.6g}"
                ),
            ]
        )

    return "\n".join(lines)


def build_report(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    arr, used_key = _load_array(Path(cfg["file"]), cfg["key"])

    report = [
        "# ROI Beta 矩阵校验报告",
        f"- 文件路径: {cfg['file']}",
        f"- 读取 key: {used_key}",
        f"- ndarray 维度: {arr.ndim}",
        "- 当前配置:",
        json.dumps(cfg, ensure_ascii=False, indent=2),
    ]

    summary: dict[str, Any] = {
        "file": cfg["file"],
        "key": used_key,
        "ndim": int(arr.ndim),
        "shape": tuple(int(v) for v in arr.shape),
        "config": cfg,
    }

    if arr.ndim == 2:
        report.append(_report_single("整体矩阵", arr, cfg))
    elif arr.ndim >= 3:
        report.append(_report_single("整体数组（汇总统计）", arr, cfg))
        flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        report.append(f"\n## 条件切片简报（前{cfg['max_slice_brief']}个）")
        topn = min(cfg["max_slice_brief"], flat.shape[0])
        for idx in range(topn):
            st = compute_stats(flat[idx])
            report.append(
                f"- slice[{idx:03d}] mean={st.mean:.6f}, std={st.std:.6f}, "
                f"range=[{st.min_val:.6f}, {st.max_val:.6f}], sym_err={st.symmetry_max_abs_diff:.6g}"
            )
        if flat.shape[0] > topn:
            report.append(f"- ... 共 {flat.shape[0]} 个切片，仅展示前 {topn} 个。")
    else:
        report.append(_report_single("一维数组", arr, cfg))

    return "\n".join(report), summary


def main() -> None:
    cfg = _validated_config(CONFIG)
    report, summary = build_report(cfg)
    print(report)

    if cfg["save_report"]:
        Path(cfg["save_report"]).write_text(report, encoding="utf-8")
        print(f"\n📝 文本报告已保存: {cfg['save_report']}")

    if cfg["save_json"]:
        Path(cfg["save_json"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 JSON 摘要已保存: {cfg['save_json']}")


if __name__ == "__main__":
    main()
