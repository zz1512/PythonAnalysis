#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_similarity_matrices.py

论文级审计（相似性矩阵数值健康检查）：
对 similarity_*.csv（通常为 subject×subject 矩阵）做快速、可批量的“合法性 + 分布”审计。

核心检查：
- 对称性：max|A - A^T|
- 对角线：是否接近 1（均值、最大偏差、合格比例）
- 上三角：NaN/Inf 比例与分布统计（均值/方差/分位数/极值）

输出：
- audit_similarity_matrices.csv：每个矩阵一行汇总。

扩展建议（未来）：
- 增加对“近似半正定”的检查（如特征值最小值）
- 增加对 outlier 被试的定位（行/列均值异常）
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _upper_values(mat: np.ndarray) -> np.ndarray:
    """取上三角（k=1）的向量，用于分布审计。"""
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _summarize_upper(vals: np.ndarray) -> Dict[str, float]:
    """对上三角向量做数值分布汇总（忽略 NaN/Inf）。"""
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    if good.sum() == 0:
        return {
            "n_pairs": float(vals.size),
            "n_finite": 0.0,
            "frac_finite": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    x = vals[good]
    return {
        "n_pairs": float(vals.size),
        "n_finite": float(good.sum()),
        "frac_finite": float(good.mean()),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _check_symmetry(mat: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
    """检查矩阵对称性，返回 (是否通过, max|A-A^T|)。"""
    if mat.shape[0] != mat.shape[1]:
        return False, float("nan")
    diff = mat - mat.T
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return False, float("nan")
    max_abs = float(np.max(np.abs(diff)))
    return bool(max_abs <= tol), max_abs


def _check_diagonal(mat: np.ndarray, tol: float = 1e-6) -> Tuple[float, float, float]:
    """检查对角线是否接近 1，返回 (均值, 最大偏差, 合格比例)。"""
    if mat.shape[0] != mat.shape[1]:
        return float("nan"), float("nan"), float("nan")
    d = np.diag(mat).astype(float)
    good = np.isfinite(d)
    if good.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    d = d[good]
    mean = float(np.mean(d))
    max_dev = float(np.max(np.abs(d - 1.0)))
    frac_ok = float(np.mean(np.abs(d - 1.0) <= tol))
    return mean, max_dev, frac_ok


def audit_one_matrix(path: Path, sym_tol: float, diag_tol: float) -> Dict[str, object]:
    """读取单个矩阵 CSV 并输出一行审计统计。"""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    mat = df.to_numpy(dtype=float)

    n = int(mat.shape[0])
    symmetric_ok, sym_max_abs = _check_symmetry(mat, tol=float(sym_tol))
    diag_mean, diag_max_dev, diag_frac_ok = _check_diagonal(mat, tol=float(diag_tol))

    upper = _upper_values(mat) if n >= 2 else np.array([], dtype=float)
    stats = _summarize_upper(upper)

    return {
        "file": str(path),
        "name": path.name,
        "n_subjects": n,
        "symmetric_ok": symmetric_ok,
        "sym_max_abs": sym_max_abs,
        "diag_mean": diag_mean,
        "diag_max_abs_dev_from_1": diag_max_dev,
        "diag_frac_within_tol": diag_frac_ok,
        **stats,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计相似性矩阵 CSV 的数值健康与合法性")
    p.add_argument("--dir", type=Path, default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"), help="搜索目录")
    p.add_argument("--glob", type=str, default="**/similarity_*AgeSorted.csv", help="矩阵文件 glob")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    p.add_argument("--limit", type=int, default=0, help="仅审计最新 N 个文件（0=全部）")
    p.add_argument("--sym-tol", type=float, default=1e-6, help="对称性容忍阈值：max|A-A^T|≤tol")
    p.add_argument("--diag-tol", type=float, default=1e-6, help="对角线容忍阈值：|diag-1|≤tol")
    p.add_argument("--min-frac-finite", type=float, default=1.0, help="上三角 finite 比例下限（低于则判为未通过）")
    p.add_argument("--log-top", type=int, default=10, help="FAIL 时在日志中打印的示例条目数量")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    files = sorted(Path(args.dir).glob(args.glob), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"未找到矩阵文件: dir={args.dir}, glob={args.glob}")
    if args.limit and args.limit > 0:
        files = files[-int(args.limit):]

    rows: List[Dict[str, object]] = []
    for f in files:
        try:
            rows.append(audit_one_matrix(f, sym_tol=float(args.sym_tol), diag_tol=float(args.diag_tol)))
        except Exception:
            rows.append({"file": str(f), "name": f.name, "error": "read_or_parse_failed"})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_similarity_matrices.csv"
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_path, index=False)

    if len(res_df) == 0:
        print("[QC] audit_similarity_matrices: FAIL")
        print(f"[QC] Saved: {out_path}")
        print("[QC] No files processed.")
        return

    err_mask = res_df["error"].notna() if "error" in res_df.columns else pd.Series([False] * len(res_df))
    bad_sym = (res_df["symmetric_ok"].fillna(False) == False) if "symmetric_ok" in res_df.columns else pd.Series([False] * len(res_df))
    bad_diag = (res_df["diag_frac_within_tol"].fillna(0.0) < 1.0) if "diag_frac_within_tol" in res_df.columns else pd.Series([False] * len(res_df))
    bad_finite = (res_df["frac_finite"].fillna(0.0) < float(args.min_frac_finite)) if "frac_finite" in res_df.columns else pd.Series([False] * len(res_df))

    fail_mask = err_mask | bad_sym | bad_diag | bad_finite
    n_fail = int(fail_mask.sum())
    status = "PASS" if n_fail == 0 else "FAIL"

    print(f"[QC] audit_similarity_matrices: {status}")
    print(f"[QC] Saved: {out_path}")
    print(
        f"[QC] Files={len(res_df)} | Fail={n_fail} | "
        f"BadSym={int(bad_sym.sum())} | "
        f"BadDiag={int(bad_diag.sum())} | "
        f"BadFinite(<{float(args.min_frac_finite):.3f})={int(bad_finite.sum())} | "
        f"ReadError={int(err_mask.sum())}"
    )

    if status == "FAIL":
        top_n = int(args.log_top)
        if top_n <= 0:
            return

        df_fail = res_df.copy()
        df_fail["bad_sym"] = bad_sym
        df_fail["bad_diag"] = bad_diag
        df_fail["bad_finite"] = bad_finite
        df_fail["has_error"] = err_mask
        df_fail["fail"] = fail_mask
        df_fail = df_fail[df_fail["fail"]]

        if not df_fail.empty:
            cols = [
                "name",
                "n_subjects",
                "frac_finite",
                "sym_max_abs",
                "diag_max_abs_dev_from_1",
                "bad_sym",
                "bad_diag",
                "bad_finite",
                "error",
            ]
            cols = [c for c in cols if c in df_fail.columns]
            shown = df_fail.sort_values(["has_error", "bad_finite", "bad_sym", "bad_diag"], ascending=False).head(top_n)
            print(f"[QC] Failed files (showing up to {top_n}):")
            print(shown[cols].to_string(index=False))


if __name__ == "__main__":
    main()
