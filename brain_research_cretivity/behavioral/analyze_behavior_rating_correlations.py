from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl import load_workbook
from scipy import stats


DEFAULT_DATA_DIR = Path("/public/home/zhangze1/fenxishuju")
QUESTION_HEADER_ROW_INDEX = 2
DATA_START_ROW_INDEX = 3
PARENT_SUBJECT_COL_INDEX = 3
TEACHER_SUBJECT_COL_INDEX = 2

BEHAVIOR_DATA = {
    "26.2": "家长",
    "26.7": "家长",
    "26.16": "家长",
    "31.1": "家长",
    "31.3": "家长",
    "31.4": "家长",
    "31.5": "家长",
    "31.8": "家长",
    "1.19": "班主任",
    "1.20": "班主任",
    "1.21": "班主任",
    "1.22": "班主任",
    "1.32": "班主任",
    "1.33": "班主任",
    "1.34": "班主任",
    "1.35": "班主任",
    "1.36": "班主任",
}

FENXI_COLUMN_ALIASES = {
    "subject_id": ["被试编号", "被试id", "subject_id", "subject", "sub_id", "subid"],
    "reappraisal_type": ["重评类型", "条件", "type", "condition"],
    "score_diff": ["评分差值", "分值差", "差值", "rating_diff", "score_diff"],
    "gender": ["性别", "gender", "sex"],
}

REAPPRAISAL_TYPE_MAP = {
    "创造": "创造",
    "创意": "创造",
    "creative": "创造",
    "常规": "常规",
    "普通": "常规",
    "conventional": "常规",
    "routine": "常规",
}

GENDER_MAP = {
    "男": "男",
    "男性": "男",
    "male": "男",
    "m": "男",
    "1": "男",
    "女": "女",
    "女性": "女",
    "female": "女",
    "f": "女",
    "2": "女",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "提取家长/班主任题目得分，并与 fenxishuju.xlsx 中创造、常规条件下"
            "的评分差值均值做相关分析。默认使用 Spearman 秩相关，以减弱"
            "1-3、1-4、1-5 不同量尺范围对结果的影响。"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="包含 parents1.xlsx、parents2.xlsx、teacher.xlsx 和 fenxishuju.xlsx 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认写入脚本所在目录下的 results 文件夹。",
    )
    parser.add_argument(
        "--method",
        choices=["spearman", "kendall"],
        default="spearman",
        help="相关系数方法，默认 spearman。",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\n", "").replace("\r", "").replace(" ", "").lower()


def normalize_subject_id(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return normalized


def normalize_teacher_subject_id(series: pd.Series) -> pd.Series:
    normalized = normalize_subject_id(series)
    normalized = normalized.str.replace(r"[（(].*?[）)]", "", regex=True).str.strip()
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return normalized


def normalize_question_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace("．", ".")
    return text or None


def infer_decimal_places(number_format: str | None) -> int | None:
    if not number_format:
        return None

    cleaned_format = re.sub(r'"[^"]*"', "", str(number_format))
    cleaned_format = re.sub(r"\[[^\]]*\]", "", cleaned_format)
    match = re.search(r"\.([0#]+)", cleaned_format)
    if match is None:
        return 0 if "." not in cleaned_format else None
    return len(match.group(1))


def format_question_key_from_cell(value: object, number_format: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_question_key(value)

    decimal_places = infer_decimal_places(number_format)
    if decimal_places is not None and isinstance(value, (int, float)):
        if decimal_places == 0:
            text = str(int(round(float(value))))
        else:
            text = f"{float(value):.{decimal_places}f}"
        return text.replace("．", ".")

    return normalize_question_key(value)


def read_question_row_labels(excel_path: Path) -> dict[int, str]:
    workbook = load_workbook(excel_path, data_only=True)
    try:
        worksheet = workbook.active
        row_number = QUESTION_HEADER_ROW_INDEX + 1
        question_labels: dict[int, str] = {}
        for column_index, cell in enumerate(worksheet[row_number]):
            question_key = format_question_key_from_cell(cell.value, cell.number_format)
            if question_key:
                question_labels[column_index] = question_key
        return question_labels
    finally:
        workbook.close()


def normalize_reappraisal_type(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    lowered = text.lower()
    if text in REAPPRAISAL_TYPE_MAP:
        return REAPPRAISAL_TYPE_MAP[text]
    if lowered in REAPPRAISAL_TYPE_MAP:
        return REAPPRAISAL_TYPE_MAP[lowered]
    if "创造" in text or "创意" in text or "creative" in lowered:
        return "创造"
    if "常规" in text or "普通" in text or "conventional" in lowered or "routine" in lowered:
        return "常规"
    return None


def normalize_gender(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    lowered = text.lower()
    if text in GENDER_MAP:
        return GENDER_MAP[text]
    if lowered in GENDER_MAP:
        return GENDER_MAP[lowered]
    if "男" in text or lowered.startswith("m"):
        return "男"
    if "女" in text or lowered.startswith("f"):
        return "女"
    return None


def find_column(df: pd.DataFrame, aliases: Iterable[str], table_name: str) -> str:
    normalized_columns = {normalize_text(column): column for column in df.columns}
    normalized_aliases = [normalize_text(alias) for alias in aliases]

    for alias in normalized_aliases:
        matched = normalized_columns.get(alias)
        if matched is not None:
            return matched

    for alias in normalized_aliases:
        for normalized_name, original_name in normalized_columns.items():
            if alias and alias in normalized_name:
                return original_name

    raise KeyError(f"{table_name} 中未找到列: {list(aliases)}")


def load_question_records_from_file(
    excel_path: Path,
    target_questions: set[str],
    subject_col_index: int,
    rater_type: str,
) -> tuple[pd.DataFrame, set[str]]:
    raw_df = pd.read_excel(excel_path, header=None)
    if raw_df.shape[0] <= QUESTION_HEADER_ROW_INDEX:
        raise ValueError(f"{excel_path} 至少需要包含第 3 行题目编号。")
    if raw_df.shape[1] <= subject_col_index:
        raise ValueError(
            f"{excel_path} 不包含第 {subject_col_index + 1} 列，无法读取被试编号。"
        )

    question_to_columns: dict[str, list[int]] = {}
    for col_idx, question_key in read_question_row_labels(excel_path).items():
        question_to_columns.setdefault(question_key, []).append(col_idx)

    data_df = raw_df.iloc[DATA_START_ROW_INDEX:].copy()
    subject_ids = normalize_subject_id(data_df.iloc[:, subject_col_index])
    if rater_type == "班主任":
        subject_ids = normalize_teacher_subject_id(subject_ids)

    records: list[dict[str, object]] = []
    found_questions: set[str] = set()

    for question_key in sorted(target_questions):
        column_indices = question_to_columns.get(question_key, [])
        if not column_indices:
            continue

        found_questions.add(question_key)
        for col_idx in column_indices:
            score_series = pd.to_numeric(data_df.iloc[:, col_idx], errors="coerce")
            question_records = pd.DataFrame(
                {
                    "subject_id": subject_ids,
                    "question_key": question_key,
                    "rater_type": rater_type,
                    "rating_score": score_series,
                    "source_file": excel_path.name,
                }
            )
            question_records = question_records.dropna(
                subset=["subject_id", "rating_score"]
            )
            if not question_records.empty:
                records.extend(question_records.to_dict("records"))

    return pd.DataFrame(records), found_questions


def load_behavior_ratings(data_dir: Path) -> pd.DataFrame:
    parent_questions = {key for key, value in BEHAVIOR_DATA.items() if value == "家长"}
    teacher_questions = {key for key, value in BEHAVIOR_DATA.items() if value == "班主任"}

    parent_frames: list[pd.DataFrame] = []
    found_parent_questions: set[str] = set()
    for file_name in ["parents1.xlsx", "parents2.xlsx"]:
        file_path = data_dir / file_name
        frame, found_questions = load_question_records_from_file(
            excel_path=file_path,
            target_questions=parent_questions,
            subject_col_index=PARENT_SUBJECT_COL_INDEX,
            rater_type="家长",
        )
        parent_frames.append(frame)
        found_parent_questions.update(found_questions)

    teacher_path = data_dir / "teacher.xlsx"
    teacher_frame, found_teacher_questions = load_question_records_from_file(
        excel_path=teacher_path,
        target_questions=teacher_questions,
        subject_col_index=TEACHER_SUBJECT_COL_INDEX,
        rater_type="班主任",
    )

    missing_parent_questions = sorted(parent_questions - found_parent_questions)
    missing_teacher_questions = sorted(teacher_questions - found_teacher_questions)
    missing_messages = []
    if missing_parent_questions:
        missing_messages.append(f"家长题目未找到: {missing_parent_questions}")
    if missing_teacher_questions:
        missing_messages.append(f"班主任题目未找到: {missing_teacher_questions}")
    if missing_messages:
        raise KeyError("；".join(missing_messages))

    rating_frames = parent_frames + [teacher_frame]
    ratings = pd.concat(rating_frames, ignore_index=True)
    if ratings.empty:
        raise ValueError("未从 parents1.xlsx、parents2.xlsx、teacher.xlsx 中提取到任何有效评分。")

    ratings = (
        ratings.groupby(["question_key", "rater_type", "subject_id"], as_index=False)
        .agg(
            rating_score=("rating_score", "mean"),
            source_file=("source_file", lambda values: "|".join(sorted(set(values)))),
        )
        .sort_values(["question_key", "rater_type", "subject_id"])
    )
    return ratings


def choose_first_valid(series: pd.Series) -> str | None:
    for value in series:
        if pd.notna(value):
            return str(value)
    return None


def load_subject_summary(data_dir: Path) -> pd.DataFrame:
    behavior_path = data_dir / "fenxishuju.xlsx"
    behavior_df = pd.read_excel(behavior_path)

    subject_col = find_column(behavior_df, FENXI_COLUMN_ALIASES["subject_id"], "fenxishuju.xlsx")
    type_col = find_column(
        behavior_df, FENXI_COLUMN_ALIASES["reappraisal_type"], "fenxishuju.xlsx"
    )
    score_col = find_column(behavior_df, FENXI_COLUMN_ALIASES["score_diff"], "fenxishuju.xlsx")
    gender_col = find_column(behavior_df, FENXI_COLUMN_ALIASES["gender"], "fenxishuju.xlsx")

    summary_df = behavior_df[[subject_col, type_col, score_col, gender_col]].copy()
    summary_df.columns = ["subject_id", "reappraisal_type", "score_diff", "gender"]
    summary_df["subject_id"] = normalize_subject_id(summary_df["subject_id"])
    summary_df["reappraisal_type"] = summary_df["reappraisal_type"].map(
        normalize_reappraisal_type
    )
    summary_df["score_diff"] = pd.to_numeric(summary_df["score_diff"], errors="coerce")
    summary_df["gender"] = summary_df["gender"].map(normalize_gender)

    summary_df = summary_df.dropna(subset=["subject_id", "reappraisal_type", "score_diff"])
    summary_df = summary_df[summary_df["reappraisal_type"].isin(["创造", "常规"])].copy()

    subject_condition_mean = (
        summary_df.groupby(["subject_id", "reappraisal_type"], as_index=False)["score_diff"]
        .mean()
        .pivot(index="subject_id", columns="reappraisal_type", values="score_diff")
        .reset_index()
    )
    subject_condition_mean = subject_condition_mean.rename(
        columns={"创造": "创造评分差值均值", "常规": "常规评分差值均值"}
    )

    subject_gender = (
        summary_df.groupby("subject_id", as_index=False)["gender"]
        .agg(choose_first_valid)
        .rename(columns={"gender": "性别"})
    )

    return (
        subject_condition_mean.merge(subject_gender, on="subject_id", how="left")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )


def calculate_correlation(
    x: pd.Series, y: pd.Series, method: str
) -> tuple[float, float]:
    valid_df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid_df) < 3:
        return float("nan"), float("nan")
    if valid_df["x"].nunique() < 2 or valid_df["y"].nunique() < 2:
        return float("nan"), float("nan")

    if method == "spearman":
        result = stats.spearmanr(valid_df["x"], valid_df["y"])
        return float(result.statistic), float(result.pvalue)

    result = stats.kendalltau(valid_df["x"], valid_df["y"])
    return float(result.statistic), float(result.pvalue)


def build_correlation_results(
    ratings_df: pd.DataFrame,
    subject_summary_df: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_df = ratings_df.merge(subject_summary_df, on="subject_id", how="left")

    results: list[dict[str, object]] = []
    for question_key, rater_type in BEHAVIOR_DATA.items():
        question_df = merged_df[merged_df["question_key"] == question_key].copy()
        result_row: dict[str, object] = {
            "题目": question_key,
            "评分来源": rater_type,
            "相关方法": method,
        }

        for condition_label, value_column in [
            ("创造", "创造评分差值均值"),
            ("常规", "常规评分差值均值"),
        ]:
            for group_label, gender_value in [
                ("总样本", None),
                ("男性", "男"),
                ("女性", "女"),
            ]:
                subset_df = question_df
                if gender_value is not None:
                    subset_df = subset_df[subset_df["性别"] == gender_value]

                valid_df = subset_df[["rating_score", value_column]].dropna()
                sample_size = len(valid_df)
                correlation, p_value = calculate_correlation(
                    valid_df["rating_score"], valid_df[value_column], method
                )

                result_row[f"{condition_label}_{group_label}样本量"] = sample_size
                result_row[f"{condition_label}_{group_label}相关系数"] = correlation
                result_row[f"{condition_label}_{group_label}P值"] = p_value

        results.append(result_row)

    results_df = pd.DataFrame(results)
    return results_df, merged_df


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings_df = load_behavior_ratings(args.data_dir)
    subject_summary_df = load_subject_summary(args.data_dir)
    results_df, merged_df = build_correlation_results(
        ratings_df=ratings_df,
        subject_summary_df=subject_summary_df,
        method=args.method,
    )

    ratings_output_path = output_dir / "behavior_question_scores.csv"
    subject_summary_output_path = output_dir / "subject_reappraisal_summary.csv"
    merged_output_path = output_dir / "behavior_rating_with_reappraisal.csv"
    results_output_path = output_dir / "behavior_question_correlations.csv"

    ratings_df.to_csv(ratings_output_path, index=False, encoding="utf-8-sig")
    subject_summary_df.to_csv(subject_summary_output_path, index=False, encoding="utf-8-sig")
    merged_df.to_csv(merged_output_path, index=False, encoding="utf-8-sig")
    results_df.to_csv(results_output_path, index=False, encoding="utf-8-sig")

    print(f"已读取评分数据目录: {args.data_dir}")
    print(f"相关分析方法: {args.method}")
    print(f"题目评分明细已保存: {ratings_output_path}")
    print(f"被试重评汇总已保存: {subject_summary_output_path}")
    print(f"合并后的被试级数据已保存: {merged_output_path}")
    print(f"每个题目的相关结果已保存: {results_output_path}")
    print("\n相关结果预览:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

