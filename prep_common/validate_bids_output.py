"""
校验 prep_common.dcm2bids_wjq 的 BIDS 产物是否完整且命名合理。

功能：
- 扫描 BIDS 根目录中的 sub-*/ses-*/anat|func|dwi
- 校验关键文件是否成对存在（.nii.gz 与 .json；dwi 还要 .bvec/.bval）
- 校验文件名是否符合常见 BIDS 模式
- 校验 func 的 JSON 中 TaskName 是否与文件名的 task- 实体一致
- 可选生成 participants.txt（供 gifted/run_fmriprep.sh 与 gifted/run_qsiprep.sh 使用）

用法：
  python -m prep_common.validate_bids_output --bids-root /path/to/BIDS
  python -m prep_common.validate_bids_output --bids-root /path/to/BIDS --write-participants-txt /path/to/BIDS/code/participants.txt
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SUBJECT_DIR_RE = re.compile(r"^sub-[A-Za-z0-9]+$")
SESSION_DIR_RE = re.compile(r"^ses-[A-Za-z0-9]+$")

ANAT_T1W_RE = re.compile(r"^sub-[^_]+_ses-[^_]+(_acq-[^_]+)?(_run-\d{2})?_T1w\.nii\.gz$")
DWI_RE = re.compile(r"^sub-[^_]+_ses-[^_]+(_run-\d{2})?_dwi\.nii\.gz$")
FUNC_BOLD_RE = re.compile(
    r"^sub-[^_]+_ses-[^_]+_task-[^_]+(_acq-[^_]+)?(_run-\d{2})?_bold\.nii\.gz$"
)
TASK_ENTITY_RE = re.compile(r"(?:^|_)task-([^_]+)")


@dataclass(frozen=True)
class Issue:
    level: str
    path: Path
    message: str


def _iter_subject_dirs(bids_root: Path) -> List[Path]:
    subs = []
    for p in sorted(bids_root.iterdir()):
        if not p.is_dir():
            continue
        if SUBJECT_DIR_RE.match(p.name):
            subs.append(p)
    return subs


def _iter_session_dirs(sub_dir: Path) -> List[Path]:
    sess = []
    for p in sorted(sub_dir.iterdir()):
        if not p.is_dir():
            continue
        if SESSION_DIR_RE.match(p.name):
            sess.append(p)
    return sess


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _validate_anat(anat_dir: Path) -> List[Issue]:
    issues: List[Issue] = []
    if not anat_dir.exists():
        return issues

    nii_files = sorted(anat_dir.glob("*.nii.gz"))
    for nii in nii_files:
        if not ANAT_T1W_RE.match(nii.name):
            issues.append(Issue("warn", nii, "anat 文件名不符合预期的 T1w BIDS 模式"))
        json_path = nii.with_suffix("").with_suffix(".json")
        if not json_path.exists():
            issues.append(Issue("error", nii, "缺少对应的 .json sidecar"))
    return issues


def _validate_dwi(dwi_dir: Path) -> List[Issue]:
    issues: List[Issue] = []
    if not dwi_dir.exists():
        return issues

    for nii in sorted(dwi_dir.glob("*.nii.gz")):
        if not DWI_RE.match(nii.name):
            issues.append(Issue("warn", nii, "dwi 文件名不符合预期的 BIDS 模式（应为 *_dwi.nii.gz）"))
        json_path = nii.with_suffix("").with_suffix(".json")
        bvec_path = nii.with_suffix("").with_suffix(".bvec")
        bval_path = nii.with_suffix("").with_suffix(".bval")
        if not json_path.exists():
            issues.append(Issue("error", nii, "缺少对应的 .json sidecar"))
        if not bvec_path.exists():
            issues.append(Issue("error", nii, "缺少对应的 .bvec"))
        if not bval_path.exists():
            issues.append(Issue("error", nii, "缺少对应的 .bval"))
    return issues


def _extract_task_from_stem(stem: str) -> Optional[str]:
    m = TASK_ENTITY_RE.search(stem)
    if not m:
        return None
    return m.group(1)


def _validate_func(func_dir: Path) -> List[Issue]:
    issues: List[Issue] = []
    if not func_dir.exists():
        return issues

    for nii in sorted(func_dir.glob("*.nii.gz")):
        if not FUNC_BOLD_RE.match(nii.name):
            issues.append(Issue("warn", nii, "func 文件名不符合预期的 bold BIDS 模式"))
        json_path = nii.with_suffix("").with_suffix(".json")
        if not json_path.exists():
            issues.append(Issue("error", nii, "缺少对应的 .json sidecar"))
            continue

        task = _extract_task_from_stem(nii.stem)
        if not task:
            issues.append(Issue("warn", nii, "未能从文件名解析 task- 实体"))
            continue

        data = _read_json(json_path)
        if data is None:
            issues.append(Issue("warn", json_path, "JSON 无法解析（编码/格式错误）"))
            continue
        taskname = data.get("TaskName")
        if taskname is None:
            issues.append(Issue("warn", json_path, "JSON 缺少 TaskName 字段"))
            continue
        if str(taskname) != task:
            issues.append(Issue("warn", json_path, f"TaskName 不一致: JSON={taskname} filename={task}"))
    return issues


def validate_bids_root(bids_root: Path) -> List[Issue]:
    issues: List[Issue] = []

    if not bids_root.exists():
        return [Issue("error", bids_root, "BIDS 根目录不存在")]

    dataset_desc = bids_root / "dataset_description.json"
    if not dataset_desc.exists():
        issues.append(Issue("warn", dataset_desc, "缺少 dataset_description.json（部分工具/校验器会要求）"))
    else:
        if _read_json(dataset_desc) is None:
            issues.append(Issue("warn", dataset_desc, "dataset_description.json 无法解析"))

    subs = _iter_subject_dirs(bids_root)
    if not subs:
        issues.append(Issue("error", bids_root, "未发现任何 sub-* 目录"))
        return issues

    for sub_dir in subs:
        sess = _iter_session_dirs(sub_dir)
        if not sess:
            issues.append(Issue("warn", sub_dir, "该被试下未发现 ses-* 目录"))
            continue

        for ses_dir in sess:
            anat_dir = ses_dir / "anat"
            func_dir = ses_dir / "func"
            dwi_dir = ses_dir / "dwi"
            issues.extend(_validate_anat(anat_dir))
            issues.extend(_validate_func(func_dir))
            issues.extend(_validate_dwi(dwi_dir))

    return issues


def write_participants_txt(bids_root: Path, out_path: Path) -> int:
    subs = [p.name for p in _iter_subject_dirs(bids_root)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(subs) + ("\n" if subs else ""), encoding="utf-8")
    return len(subs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="validate_bids_output", description="校验 dcm2bids_wjq 的 BIDS 产物。")
    p.add_argument("--bids-root", required=True, help="BIDS 根目录（包含 sub-* 子目录）")
    p.add_argument("--write-participants-txt", default=None, help="写出 participants.txt（每行一个 sub-XXX）")
    p.add_argument("--fail-on-warn", action="store_true", help="将 warn 也视为失败（退出码=1）")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = parse_args(argv)
    bids_root = Path(ns.bids_root).expanduser().resolve()

    issues = validate_bids_root(bids_root)
    errors = [i for i in issues if i.level == "error"]
    warns = [i for i in issues if i.level == "warn"]

    if ns.write_participants_txt:
        out_path = Path(ns.write_participants_txt).expanduser().resolve()
        n = write_participants_txt(bids_root, out_path)
        print(f"[OK] 写入 participants.txt: {out_path} (n={n})")

    print(f"[Summary] errors={len(errors)} warns={len(warns)}")
    for i in issues:
        print(f"[{i.level.upper()}] {i.path}: {i.message}")

    if errors:
        return 1
    if warns and ns.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

