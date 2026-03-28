from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import stats


DEFAULT_DATA_DIR = Path("/public/home/zhangze1/fenxishuju")

EXCEL_COLUMN_ALIASES = {
    "subject_id": ["被试编号", "被试id", "subject_id", "subject", "sub_id", "subid"],
    "reappraisal_type": ["重评类型", "条件", "type", "condition"],
    "score_diff": ["评分差值", "分值差", "差值", "rating_diff", "score_diff"],
}

LOCATION_COLUMN_ALIASES = {
    "subject_id": ["被试编号", "被试id", "subject_id", "subject", "sub_id", "subid"],
    "region": ["被试地域", "地域", "地区", "location", "region", "group"],
}

TYPE_MAP = {
    "创造": "创造",
    "创意": "创造",
    "creative": "创造",
    "常规": "常规",
    "常规重评": "常规",
    "普通": "常规",
    "conventional": "常规",
    "routine": "常规",
}

REGION_MAP = {
    "城市": "城市",
    "城镇": "城市",
    "城区": "城市",
    "urban": "城市",
    "city": "城市",
    "乡镇": "乡镇",
    "乡村": "乡镇",
    "农村": "乡镇",
    "rural": "乡镇",
    "town": "乡镇",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较城市和乡镇被试在创造/常规重评条件下的评分差值差异。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="包含 fenxishuju.xlsx 和 location.tsv 的目录。",
    )
    parser.add_argument(
        "--excel-name",
        default="fenxishuju.xlsx",
        help="Excel 文件名，默认 fenxishuju.xlsx。",
    )
    parser.add_argument(
        "--location-name",
        default="location.tsv",
        help="地域文件名，默认 location.tsv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录，默认写到脚本所在目录下的 results 文件夹。",
    )
    return parser.parse_args()


def find_column(df: pd.DataFrame, aliases: Iterable[str], table_name: str) -> str:
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        matched = lower_to_original.get(alias.strip().lower())
        if matched is not None:
            return matched
    raise KeyError(f"{table_name} 中未找到列: {list(aliases)}")


def normalize_subject_id(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized


def normalize_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if text in TYPE_MAP:
        return TYPE_MAP[text]
    if lowered in TYPE_MAP:
        return TYPE_MAP[lowered]
    if "创造" in text or "creative" in lowered:
        return "创造"
    if "常规" in text or "普通" in text or "conventional" in lowered or "routine" in lowered:
        return "常规"
    return None


def normalize_region(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if text in REGION_MAP:
        return REGION_MAP[text]
    if lowered in REGION_MAP:
        return REGION_MAP[lowered]
    if "城" in text or "urban" in lowered or "city" in lowered:
        return "城市"
    if "乡" in text or "农" in text or "rural" in lowered or "town" in lowered:
        return "乡镇"
    return None


def hedges_g(group1: pd.Series, group2: pd.Series) -> float:
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")

    std1 = group1.std(ddof=1)
    std2 = group2.std(ddof=1)
    pooled_var = (((n1 - 1) * std1**2) + ((n2 - 1) * std2**2)) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return float("nan")

    pooled_sd = math.sqrt(pooled_var)
    cohen_d = (group1.mean() - group2.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return cohen_d * correction


def build_merged_table(excel_path: Path, location_path: Path) -> pd.DataFrame:
    behavior_df = pd.read_excel(excel_path)
    location_df = pd.read_csv(location_path, sep="\t")

    behavior_subject_col = find_column(
        behavior_df, EXCEL_COLUMN_ALIASES["subject_id"], "fenxishuju.xlsx"
    )
    behavior_type_col = find_column(
        behavior_df, EXCEL_COLUMN_ALIASES["reappraisal_type"], "fenxishuju.xlsx"
    )
    behavior_score_col = find_column(
        behavior_df, EXCEL_COLUMN_ALIASES["score_diff"], "fenxishuju.xlsx"
    )

    location_subject_col = find_column(
        location_df, LOCATION_COLUMN_ALIASES["subject_id"], "location.tsv"
    )
    location_region_col = find_column(
        location_df, LOCATION_COLUMN_ALIASES["region"], "location.tsv"
    )

    behavior_subset = behavior_df[
        [behavior_subject_col, behavior_type_col, behavior_score_col]
    ].copy()
    behavior_subset.columns = ["被试编号", "重评类型", "评分差值"]
    behavior_subset["被试编号"] = normalize_subject_id(behavior_subset["被试编号"])
    behavior_subset["重评类型"] = behavior_subset["重评类型"].map(normalize_type)
    behavior_subset["评分差值"] = pd.to_numeric(behavior_subset["评分差值"], errors="coerce")

    location_subset = location_df[[location_subject_col, location_region_col]].copy()
    location_subset.columns = ["被试编号", "被试地域"]
    location_subset["被试编号"] = normalize_subject_id(location_subset["被试编号"])
    location_subset["被试地域"] = location_subset["被试地域"].map(normalize_region)
    location_subset = location_subset.drop_duplicates(subset=["被试编号"])

    merged = behavior_subset.merge(location_subset, on="被试编号", how="left")
    merged = merged.dropna(subset=["被试编号", "重评类型", "评分差值", "被试地域"]).copy()
    merged = merged[merged["被试地域"].isin(["城市", "乡镇"])].copy()
    return merged


def build_subject_average_table(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby(["被试编号", "重评类型", "被试地域"], as_index=False)["评分差值"]
        .mean()
        .sort_values(["重评类型", "被试地域", "被试编号"])
    )


def compare_regions(subject_avg: pd.DataFrame) -> pd.DataFrame:
    results = []
    for condition in ["创造", "常规"]:
        condition_df = subject_avg[subject_avg["重评类型"] == condition]
        city_scores = condition_df.loc[condition_df["被试地域"] == "城市", "评分差值"].dropna()
        town_scores = condition_df.loc[condition_df["被试地域"] == "乡镇", "评分差值"].dropna()

        if len(city_scores) >= 2 and len(town_scores) >= 2:
            t_stat, p_value = stats.ttest_ind(
                city_scores, town_scores, equal_var=False, nan_policy="omit"
            )
            effect_size = hedges_g(city_scores, town_scores)
        else:
            t_stat = float("nan")
            p_value = float("nan")
            effect_size = float("nan")

        results.append(
            {
                "重评类型": condition,
                "城市样本量": len(city_scores),
                "乡镇样本量": len(town_scores),
                "城市均值": city_scores.mean(),
                "乡镇均值": town_scores.mean(),
                "城市标准差": city_scores.std(ddof=1),
                "乡镇标准差": town_scores.std(ddof=1),
                "t值": t_stat,
                "P值": p_value,
                "效应量(Hedges_g)": effect_size,
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    excel_path = data_dir / args.excel_name
    location_path = data_dir / args.location_name

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = build_merged_table(excel_path, location_path)
    subject_avg = build_subject_average_table(merged)
    stats_df = compare_regions(subject_avg)

    merged_path = output_dir / "merged_behavior_region.csv"
    subject_avg_path = output_dir / "subject_condition_region_mean.csv"
    stats_path = output_dir / "city_vs_town_stats.csv"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    subject_avg.to_csv(subject_avg_path, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

    print(f"已读取行为数据: {excel_path}")
    print(f"已读取地域数据: {location_path}")
    print(f"合并后有效记录数: {len(merged)}")
    print(f"被试水平均值表已保存: {subject_avg_path}")
    print(f"统计结果已保存: {stats_path}")
    print("\n城市 vs 乡镇比较结果:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
