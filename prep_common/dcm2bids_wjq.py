"""
将 dcm2bids_wjq.m 迁移为 Python 版本。

功能概述（对齐 MATLAB 脚本）：
1) 在输出根目录生成/更新：
   - README.txt
   - dataset_description.json
   - participants.json
2) 针对 DICOM 根目录下的每个被试文件夹：
   - 输出 session 默认使用 ses-pre / ses-post（可用 --ses-labels 自定义）
   - anat：优先 dicom_dir/<sub>/session{1,2}/anat；否则 ses-pre 兜底 dicom_dir/<sub>/anat
   - dwi：dicom_dir/<sub>/session1/DTI_mx_137 -> out_dir/sub-<sub>/ses-pre/dwi
          dicom_dir/<sub>/session2/DTI_mx_137 -> out_dir/sub-<sub>/ses-post/dwi
   - func(rest)：dicom_dir/<sub>/session{1,2}/RfMRI* -> out_dir/sub-<sub>/ses-{pre,post}/func
   - func(task)：优先 dicom_dir/<sub>/session{1,2}/TfMRI*；否则 ses-pre 兜底 dicom_dir/<sub>/TfMRI*
3) 对转换得到的文件进行重命名，并写回 func 的 JSON：补充/修正 TaskName 字段

依赖：
- 外部命令 dcm2niix（可通过 --dcm2niix 指定路径，或从 PATH 自动查找）

用法示例：
  python -m prep_common.dcm2bids_wjq --dicom-dir /path/to/Dicom --dcm2niix /path/to/dcm2niix
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_README_TEXT = "Data description: Building on the model of our initial NKI-RS effort"

DEFAULT_DATASET_DESCRIPTION: Dict[str, object] = {
    "Name": "NKI-Rockland Sample - Multiband Imaging Test-RetestPilot Dataset",
    "BIDSVersion": "1.0.2",
    "Authors": ["Dawn Thomsen", "Marissa Jones Issa", "Nancy Duan"],
}

DEFAULT_PARTICIPANTS_JSON: Dict[str, object] = {
    "age": {
        "Description": "age of the participant",
        "Units": "year",
    },
    "sex": {
        "Description": "sex of the participant as reported by the participant",
        "Levels": {
            "M": "male",
            "F": "female",
        },
    },
    "handedness": {
        "Description": "handedness of the participant as reported by the participant",
        "Levels": {
            "left": "left",
            "right": "right",
        },
    },
    "group": {
        "Description": "experimental group the participant belonged to",
        "Levels": {
            "read": "participants who read an inspirational text before the experiment",
            "write": "participants who wrote an inspirational text before the experiment",
        },
    },
}


@dataclass(frozen=True)
class ConvertConfig:
    dicom_dir: Path
    out_dir: Path
    dcm2niix: Path
    # 输出 session 标签（会生成 ses-<label> 目录与文件名中的 ses-<label> 实体）
    ses_labels: Tuple[str, str]
    overwrite: bool
    dry_run: bool
    # 当命名冲突且不覆盖时，自动追加 run-01/run-02…，避免跳过或误覆盖
    auto_run: bool
    subjects: Optional[List[str]]

    write_readme: bool
    write_dataset_description: bool
    write_participants_json: bool

    do_anat: bool
    do_dwi: bool
    do_rest: bool
    do_task: bool
    fix_taskname_in_json: bool


def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path, dry_run: bool) -> None:
    """确保目录存在（支持 dry-run）。"""
    if dry_run:
        _print(f"[DryRun] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def write_text_file(path: Path, content: str, overwrite: bool, dry_run: bool) -> None:
    """写入纯文本文件（支持覆盖与 dry-run）。"""
    if path.exists() and not overwrite:
        _print(f"[Skip] 已存在且未开启覆盖: {path}")
        return
    if dry_run:
        _print(f"[DryRun] write {path} ({len(content)} chars)")
        return
    path.write_text(content, encoding="utf-8")
    _print(f"[OK] 写入: {path}")


def write_json_file(path: Path, obj: object, overwrite: bool, dry_run: bool) -> None:
    """写入 JSON 文件（UTF-8、indent=2；支持覆盖与 dry-run）。"""
    if path.exists() and not overwrite:
        _print(f"[Skip] 已存在且未开启覆盖: {path}")
        return
    if dry_run:
        _print(f"[DryRun] write {path} (json)")
        return
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(f"[OK] 写入: {path}")


def safe_rename(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> bool:
    """安全重命名：当目标存在时按 overwrite 决定是否覆盖；返回是否发生了重命名。"""
    if not src.exists():
        _print(f"[Skip] 源文件不存在: {src}")
        return False

    if dst.exists():
        if not overwrite:
            _print(f"[Skip] 目标已存在且未开启覆盖: {dst.name}")
            return False
        if dry_run:
            _print(f"[DryRun] overwrite {dst} with {src}")
        else:
            dst.unlink()

    if dry_run:
        _print(f"[DryRun] mv {src} -> {dst}")
        return True

    src.rename(dst)
    _print(f"[OK] 重命名: {src.name} -> {dst.name}")
    return True


def find_dcm2niix(dcm2niix_arg: Optional[str]) -> Path:
    """定位 dcm2niix 可执行文件：优先使用 --dcm2niix，否则从 PATH 搜索。"""
    if dcm2niix_arg:
        p = Path(dcm2niix_arg).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"--dcm2niix 指定路径不存在: {p}")

    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    raise FileNotFoundError("未找到 dcm2niix，请使用 --dcm2niix 指定其路径，或将其加入 PATH")


def run_dcm2niix(
    *,
    dcm2niix: Path,
    in_dir: Path,
    out_dir: Path,
    filename_pattern: str,
    extra_args: Sequence[str],
    dry_run: bool,
) -> None:
    """
    调用外部 dcm2niix 将 in_dir 转换到 out_dir。

    注意：
    - 这里不拼接字符串命令，直接使用 argv list，避免路径含空格时出错
    - 文件名模式 filename_pattern 对齐 MATLAB 版本的 "%f_%p"
    """
    if not in_dir.exists():
        _print(f"[Skip] 输入目录不存在: {in_dir}")
        return

    cmd: List[str] = [
        str(dcm2niix),
        "-f",
        filename_pattern,
        "-o",
        str(out_dir),
        *list(extra_args),
        str(in_dir),
    ]

    if dry_run:
        _print("[DryRun] " + " ".join(cmd))
        return

    _print("[Run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def list_subject_dirs(dicom_dir: Path, subjects: Optional[Sequence[str]]) -> List[Path]:
    """列出被试目录：可指定子集；默认扫描 dicom_dir 下一层所有非隐藏目录。"""
    if subjects:
        dirs = []
        for s in subjects:
            p = dicom_dir / s
            if p.is_dir():
                dirs.append(p)
            else:
                _print(f"[Warn] 指定 subject 不存在或不是目录: {p}")
        return sorted(dirs)

    all_dirs = []
    for p in sorted(dicom_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("."):
            continue
        all_dirs.append(p)
    return all_dirs


def _strip_known_suffix(filename: str) -> str:
    """去掉常见后缀，优先处理 .nii.gz（保持与 BIDS/NIfTI 的双后缀一致）。"""
    for suf in (".nii.gz", ".nii", ".json", ".bvec", ".bval"):
        if filename.endswith(suf):
            return filename[: -len(suf)]
    return Path(filename).stem


def _split_tokens(basename_no_ext: str) -> List[str]:
    """按 '_' 分割为 token（去除空 token）。"""
    tokens = [t for t in basename_no_ext.split("_") if t]
    return tokens


def _snapshot_files(folder: Path) -> List[Path]:
    """抓取目录当前文件快照，用于后续识别“本次 dcm2niix 新生成的文件”。"""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file()])


def _collect_new_outputs(out_dir: Path, before: Sequence[Path]) -> List[Path]:
    """用“前后快照差集”识别新输出文件，避免 mtime 精度差异带来的不稳定。"""
    before_set = {p.name for p in before}
    after = _snapshot_files(out_dir)
    return [p for p in after if p.name not in before_set]



def _assert_singleton(candidates: List[Path], context: str) -> None:
    if not candidates:
        raise FileNotFoundError(f"未找到可重命名文件: {context}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates[:10])
        raise RuntimeError(f"检测到多个候选文件({len(candidates)}): {context}. 示例: {names}")

def _split_name_and_ext(filename: str) -> Tuple[str, str]:
    """拆分文件名与扩展名，特殊处理 .nii.gz 双后缀。"""
    if filename.endswith(".nii.gz"):
        return filename[:-7], ".nii.gz"
    p = Path(filename)
    return p.stem, p.suffix


def _with_run_entity(dst: Path, run: int) -> Path:
    """
    为目标文件名追加 run-XX 实体，用于命名冲突时避免覆盖/跳过。

    规则：
    - 尽量插入到 _bold/_dwi/_T1w 之前（更符合 BIDS 实体顺序）
    - 若未匹配到这些后缀，则直接在末尾追加 _run-XX
    """
    stem, ext = _split_name_and_ext(dst.name)
    run_tag = f"_run-{run:02d}"
    for suffix_key in ("_bold", "_dwi", "_T1w"):
        if stem.endswith(suffix_key):
            stem = stem[: -len(suffix_key)] + run_tag + suffix_key
            return dst.with_name(stem + ext)
    return dst.with_name(stem + run_tag + ext)


def safe_rename_with_autorun(
    src: Path,
    dst: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    auto_run: bool,
) -> bool:
    """
    重命名增强版：
    - overwrite=True：允许覆盖目标
    - overwrite=False 且 auto_run=True：自动尝试 run-01/run-02… 生成不冲突的新目标名
    """
    if overwrite or not dst.exists():
        return safe_rename(src, dst, overwrite, dry_run)

    if not auto_run:
        return safe_rename(src, dst, overwrite, dry_run)

    for run in range(1, 100):
        candidate = _with_run_entity(dst, run)
        if not candidate.exists():
            return safe_rename(src, candidate, overwrite=False, dry_run=dry_run)

    raise RuntimeError(f"无法为文件生成唯一 run-* 名称（尝试 1-99 均冲突）: {dst}")


def _rename_anat_outputs(
    *,
    out_dir: Path,
    participant: str,
    ses_label: str,
    overwrite: bool,
    dry_run: bool,
    auto_run: bool,
    recent_outputs: Optional[List[Path]] = None,
) -> None:
    files = recent_outputs if recent_outputs is not None else list(out_dir.iterdir())
    nii_files = [p for p in files if p.name.endswith(".nii.gz")]
    json_files = [p for p in files if p.name.endswith(".json")]

    for p in nii_files:
        base = _strip_known_suffix(p.name)
        parts = _split_tokens(base)
        if len(parts) == 2:
            dst = out_dir / f"sub-{participant}_ses-{ses_label}_T1w.nii.gz"
            safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)
        elif len(parts) == 4 and "Crop" in parts[2]:
            dst = out_dir / f"sub-{participant}_ses-{ses_label}_acq-crop_T1w.nii.gz"
            safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)
        else:
            _print(f"[Skip] 不符合 anat 命名规则: {p.name}")

    for p in json_files:
        base = _strip_known_suffix(p.name)
        parts = _split_tokens(base)
        if len(parts) == 2:
            dst = out_dir / f"sub-{participant}_ses-{ses_label}_T1w.json"
            safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)
        elif len(parts) == 4 and "Crop" in parts[2]:
            dst = out_dir / f"sub-{participant}_ses-{ses_label}_acq-crop_T1w.json"
            safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)
        else:
            _print(f"[Skip] 不符合 anat JSON 命名规则: {p.name}")


def _rename_dwi_outputs(
    *,
    out_dir: Path,
    participant: str,
    ses_label: str,
    overwrite: bool,
    dry_run: bool,
    auto_run: bool,
    recent_outputs: Optional[List[Path]] = None,
) -> None:
    files = recent_outputs if recent_outputs is not None else list(out_dir.iterdir())
    nii_files = [p for p in files if p.name.endswith(".nii.gz")]
    json_files = [p for p in files if p.name.endswith(".json")]
    bvec_files = [p for p in files if p.name.endswith(".bvec")]
    bval_files = [p for p in files if p.name.endswith(".bval")]

    def ok_dwi_name(p: Path) -> bool:
        base = _strip_known_suffix(p.name)
        parts = _split_tokens(base)
        return len(parts) == 7

    nii_candidates = [p for p in nii_files if ok_dwi_name(p)]
    json_candidates = [p for p in json_files if ok_dwi_name(p)]
    bvec_candidates = [p for p in bvec_files if ok_dwi_name(p)]
    bval_candidates = [p for p in bval_files if ok_dwi_name(p)]

    if nii_candidates:
        _assert_singleton(nii_candidates, f"dwi nii.gz sub={participant} ses={ses_label}")
        safe_rename_with_autorun(
            nii_candidates[0],
            out_dir / f"sub-{participant}_ses-{ses_label}_dwi.nii.gz",
            overwrite=overwrite,
            dry_run=dry_run,
            auto_run=auto_run,
        )

    if json_candidates:
        _assert_singleton(json_candidates, f"dwi json sub={participant} ses={ses_label}")
        safe_rename_with_autorun(
            json_candidates[0],
            out_dir / f"sub-{participant}_ses-{ses_label}_dwi.json",
            overwrite=overwrite,
            dry_run=dry_run,
            auto_run=auto_run,
        )

    if bvec_candidates:
        _assert_singleton(bvec_candidates, f"dwi bvec sub={participant} ses={ses_label}")
        safe_rename_with_autorun(
            bvec_candidates[0],
            out_dir / f"sub-{participant}_ses-{ses_label}_dwi.bvec",
            overwrite=overwrite,
            dry_run=dry_run,
            auto_run=auto_run,
        )

    if bval_candidates:
        _assert_singleton(bval_candidates, f"dwi bval sub={participant} ses={ses_label}")
        safe_rename_with_autorun(
            bval_candidates[0],
            out_dir / f"sub-{participant}_ses-{ses_label}_dwi.bval",
            overwrite=overwrite,
            dry_run=dry_run,
            auto_run=auto_run,
        )


def _detect_tr(tokens: Sequence[str]) -> Optional[str]:
    for tr in ("1400", "645", "2500"):
        if tr in tokens:
            return tr
    return None


def _rename_rest_outputs(
    *,
    out_dir: Path,
    participant: str,
    ses_label: str,
    overwrite: bool,
    dry_run: bool,
    auto_run: bool,
    recent_outputs: Optional[List[Path]] = None,
) -> None:
    files = recent_outputs if recent_outputs is not None else list(out_dir.iterdir())
    nii_files = [p for p in files if p.name.endswith(".nii.gz")]
    json_files = [p for p in files if p.name.endswith(".json")]

    for p in nii_files:
        base = _strip_known_suffix(p.name)
        tokens = _split_tokens(base)
        tr = _detect_tr(tokens)
        if not tr:
            _print(f"[Skip] rest 未识别 TR: {p.name}")
            continue
        dst = out_dir / f"sub-{participant}_ses-{ses_label}_task-rest_acq-TR{tr}_bold.nii.gz"
        safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)

    for p in json_files:
        base = _strip_known_suffix(p.name)
        tokens = _split_tokens(base)
        tr = _detect_tr(tokens)
        if not tr:
            _print(f"[Skip] rest JSON 未识别 TR: {p.name}")
            continue
        dst = out_dir / f"sub-{participant}_ses-{ses_label}_task-rest_acq-TR{tr}_bold.json"
        safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)


def _detect_task_and_tr(tokens: Sequence[str]) -> Optional[Tuple[str, str]]:
    token_set = set(tokens)

    def has(*names: str) -> bool:
        return all(n in token_set for n in names)

    if has("1400", "breathHold"):
        return ("breathhold", "1400")
    if has("1400", "eyeMovementCalibration"):
        return ("eyemovement", "1400")
    if has("645", "eyeMovementCalibration"):
        return ("eyemovement", "645")
    if has("1400", "visualCheckerboard"):
        return ("checkerboard", "1400")
    if has("645", "visualCheckerboard"):
        return ("checkerboard", "645")
    return None


def _rename_task_outputs(
    *,
    out_dir: Path,
    participant: str,
    ses_label: str,
    overwrite: bool,
    dry_run: bool,
    auto_run: bool,
    recent_outputs: Optional[List[Path]] = None,
) -> None:
    files = recent_outputs if recent_outputs is not None else list(out_dir.iterdir())
    nii_files = [p for p in files if p.name.endswith(".nii.gz")]
    json_files = [p for p in files if p.name.endswith(".json")]

    for p in nii_files:
        base = _strip_known_suffix(p.name)
        tokens = _split_tokens(base)
        detected = _detect_task_and_tr(tokens)
        if not detected:
            _print(f"[Skip] task 未识别任务/TR: {p.name}")
            continue
        task, tr = detected
        dst = out_dir / f"sub-{participant}_ses-{ses_label}_task-{task}_acq-TR{tr}_bold.nii.gz"
        safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)

    for p in json_files:
        base = _strip_known_suffix(p.name)
        tokens = _split_tokens(base)
        detected = _detect_task_and_tr(tokens)
        if not detected:
            _print(f"[Skip] task JSON 未识别任务/TR: {p.name}")
            continue
        task, tr = detected
        dst = out_dir / f"sub-{participant}_ses-{ses_label}_task-{task}_acq-TR{tr}_bold.json"
        safe_rename_with_autorun(p, dst, overwrite=overwrite, dry_run=dry_run, auto_run=auto_run)


TASK_RE = re.compile(r"(?:^|_)task-([^_]+)")


def _extract_task_from_filename(filename: str) -> Optional[str]:
    m = TASK_RE.search(filename)
    if not m:
        return None
    return m.group(1)


def fix_taskname_in_json_files(folder: Path, dry_run: bool) -> int:
    if not folder.exists():
        return 0

    json_files = sorted(folder.glob("*.json"))
    updated = 0
    for p in json_files:
        task = _extract_task_from_filename(p.stem)
        if not task:
            continue

        if dry_run:
            _print(f"[DryRun] set TaskName={task} in {p.name}")
            updated += 1
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _print(f"[Warn] JSON 解析失败，跳过: {p}")
            continue

        old = data.get("TaskName")
        if old == task:
            continue
        data["TaskName"] = task
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _print(f"[OK] 修正 TaskName: {p.name} ({old} -> {task})")
        updated += 1
    return updated


def prepare_project_files(cfg: ConvertConfig) -> None:
    ensure_dir(cfg.out_dir, cfg.dry_run)

    if cfg.write_readme:
        write_text_file(cfg.out_dir / "README.txt", DEFAULT_README_TEXT, cfg.overwrite, cfg.dry_run)
    if cfg.write_dataset_description:
        write_json_file(
            cfg.out_dir / "dataset_description.json",
            DEFAULT_DATASET_DESCRIPTION,
            cfg.overwrite,
            cfg.dry_run,
        )
    if cfg.write_participants_json:
        write_json_file(
            cfg.out_dir / "participants.json",
            DEFAULT_PARTICIPANTS_JSON,
            cfg.overwrite,
            cfg.dry_run,
        )


def _get_anat_input_dir(subj_dir: Path, session_idx: int) -> Optional[Path]:
    """
    探测 anat 的输入目录。

    兼容逻辑：
    - ses-pre：优先 session1/anat，其次 subj_dir/anat（兼容旧结构）
    - ses-post：使用 session2/anat
    """
    if session_idx == 1:
        cand = subj_dir / "session1" / "anat"
        if cand.exists():
            return cand
        cand = subj_dir / "anat"
        if cand.exists():
            return cand
        return None

    if session_idx == 2:
        cand = subj_dir / "session2" / "anat"
        if cand.exists():
            return cand
        return None

    raise ValueError(f"unsupported session_idx={session_idx}")


def _get_task_series_dirs(subj_dir: Path, session_idx: int) -> List[Path]:
    """
    探测 task-fMRI 的输入序列目录（TfMRI*）。

    兼容逻辑：
    - ses-pre：优先 session1/TfMRI*，否则兜底 subj_dir/TfMRI*
    - ses-post：使用 session2/TfMRI*
    """
    if session_idx == 1:
        cand = subj_dir / "session1"
        if cand.exists():
            dirs = sorted([p for p in cand.glob("TfMRI*") if p.is_dir()])
            if dirs:
                return dirs
        return sorted([p for p in subj_dir.glob("TfMRI*") if p.is_dir()])

    if session_idx == 2:
        cand = subj_dir / "session2"
        if cand.exists():
            return sorted([p for p in cand.glob("TfMRI*") if p.is_dir()])
        return []

    raise ValueError(f"unsupported session_idx={session_idx}")


def process_one_subject(cfg: ConvertConfig, subj_dir: Path) -> None:
    participant = subj_dir.name
    if participant.startswith("sub-"):
        participant = participant[4:]
    _print(f"\n[Subject] {participant}")

    # anat
    if cfg.do_anat:
        # cfg.ses_labels 是 (pre, post)，这里用 enumerate(start=1) 对齐到 session1/session2
        for session_idx, ses_label in enumerate(cfg.ses_labels, start=1):
            anat_in = _get_anat_input_dir(subj_dir, session_idx)
            if not anat_in:
                _print(f"[Info] 未找到 anat 输入目录: sub={participant} ses={ses_label}")
                continue
            anat_out = cfg.out_dir / f"sub-{participant}" / f"ses-{ses_label}" / "anat"
            ensure_dir(anat_out, cfg.dry_run)
            before = _snapshot_files(anat_out)
            run_dcm2niix(
                dcm2niix=cfg.dcm2niix,
                in_dir=anat_in,
                out_dir=anat_out,
                filename_pattern="%f_%p",
                extra_args=["-p", "y", "-x", "y", "-z", "y"],
                dry_run=cfg.dry_run,
            )
            recent = _collect_new_outputs(anat_out, before)
            _rename_anat_outputs(
                out_dir=anat_out,
                participant=participant,
                ses_label=ses_label,
                overwrite=cfg.overwrite,
                dry_run=cfg.dry_run,
                auto_run=cfg.auto_run,
                recent_outputs=recent,
            )

    # dwi (ses-pre / ses-post)
    if cfg.do_dwi:
        # 约定：session1 -> ses-pre，session2 -> ses-post
        for ses_idx, ses_label in enumerate(cfg.ses_labels, start=1):
            dwi_out = cfg.out_dir / f"sub-{participant}" / f"ses-{ses_label}" / "dwi"
            ensure_dir(dwi_out, cfg.dry_run)
            # 约定：DTI_mx_137 是扩散序列目录名（与原 MATLAB 脚本一致）
            dwi_in = subj_dir / f"session{ses_idx}" / "DTI_mx_137"
            before = _snapshot_files(dwi_out)
            run_dcm2niix(
                dcm2niix=cfg.dcm2niix,
                in_dir=dwi_in,
                out_dir=dwi_out,
                filename_pattern="%f_%p",
                extra_args=["-p", "y", "-z", "y"],
                dry_run=cfg.dry_run,
            )
            recent = _collect_new_outputs(dwi_out, before)
            _rename_dwi_outputs(
                out_dir=dwi_out,
                participant=participant,
                ses_label=ses_label,
                overwrite=cfg.overwrite,
                dry_run=cfg.dry_run,
                auto_run=cfg.auto_run,
                recent_outputs=recent,
            )

    # func rest (ses-pre / ses-post)
    if cfg.do_rest:
        for ses_idx, ses_label in enumerate(cfg.ses_labels, start=1):
            func_out = cfg.out_dir / f"sub-{participant}" / f"ses-{ses_label}" / "func"
            ensure_dir(func_out, cfg.dry_run)
            ses_in = subj_dir / f"session{ses_idx}"
            rest_series = sorted([p for p in ses_in.glob("RfMRI*") if p.is_dir()])
            if not rest_series:
                _print(f"[Info] 未找到 rest 序列目录: {ses_in}/RfMRI*")
                continue

            for series_dir in rest_series:
                # 同一 session 可能存在多个 RfMRI* 序列，开启 --auto-run 可自动生成 run-XX 避免冲突
                before = _snapshot_files(func_out)
                run_dcm2niix(
                    dcm2niix=cfg.dcm2niix,
                    in_dir=series_dir,
                    out_dir=func_out,
                    filename_pattern="%f_%p",
                    extra_args=["-p", "y", "-z", "y"],
                    dry_run=cfg.dry_run,
                )
                recent = _collect_new_outputs(func_out, before)
                _rename_rest_outputs(
                    out_dir=func_out,
                    participant=participant,
                    ses_label=ses_label,
                    overwrite=cfg.overwrite,
                    dry_run=cfg.dry_run,
                    auto_run=cfg.auto_run,
                    recent_outputs=recent,
                )

    # func task（默认支持 ses-pre / ses-post；输入目录按数据结构做兼容探测）
    if cfg.do_task:
        for session_idx, ses_label in enumerate(cfg.ses_labels, start=1):
            func_out = cfg.out_dir / f"sub-{participant}" / f"ses-{ses_label}" / "func"
            ensure_dir(func_out, cfg.dry_run)
            # task 数据目录名通常为 TfMRI*；脚本会按 session1/session2 或旧结构自动探测
            task_series = _get_task_series_dirs(subj_dir, session_idx)
            if not task_series:
                _print(f"[Info] 未找到 task 序列目录: sub={participant} ses={ses_label}")
                continue
            for series_dir in task_series:
                before = _snapshot_files(func_out)
                run_dcm2niix(
                    dcm2niix=cfg.dcm2niix,
                    in_dir=series_dir,
                    out_dir=func_out,
                    filename_pattern="%f_%p",
                    extra_args=["-p", "y", "-z", "y"],
                    dry_run=cfg.dry_run,
                )
                recent = _collect_new_outputs(func_out, before)
                _rename_task_outputs(
                    out_dir=func_out,
                    participant=participant,
                    ses_label=ses_label,
                    overwrite=cfg.overwrite,
                    dry_run=cfg.dry_run,
                    auto_run=cfg.auto_run,
                    recent_outputs=recent,
                )

    # 修正 JSON：补充 TaskName（ses-pre / ses-post func）
    if cfg.fix_taskname_in_json:
        for ses_label in cfg.ses_labels:
            func_folder = cfg.out_dir / f"sub-{participant}" / f"ses-{ses_label}" / "func"
            updated = fix_taskname_in_json_files(func_folder, cfg.dry_run)
            if updated:
                _print(f"[OK] TaskName 修正完成: sub={participant} ses={ses_label} (n={updated})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dcm2bids_wjq",
        description="将特定目录结构的 DICOM 数据批量转换为 BIDS 风格目录（对齐 dcm2bids_wjq.m）。",
    )
    p.add_argument("--dicom-dir", required=True, help="DICOM 根目录（包含各被试子目录）")
    p.add_argument("--out-dir", default=None, help="输出根目录（默认 dicom-dir 的同级 Nifti 目录）")
    p.add_argument("--dcm2niix", default=None, help="dcm2niix 可执行文件路径（默认从 PATH 查找）")
    p.add_argument("--subjects", nargs="*", default=None, help="仅处理指定被试目录名（例如 0001 0002）")
    p.add_argument(
        "--ses-labels",
        nargs=2,
        default=("pre", "post"),
        metavar=("PRE_LABEL", "POST_LABEL"),
        help="输出 session 标签（默认 pre post，对应 ses-pre/ses-post）",
    )

    p.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件（重命名/写入）")
    p.add_argument("--dry-run", action="store_true", help="仅打印将执行的动作，不实际运行")
    p.add_argument("--auto-run", action="store_true", help="当命名冲突且不覆盖时，自动追加 run-01/run-02… 以避免跳过")

    p.add_argument("--no-readme", action="store_true", help="不生成 README.txt")
    p.add_argument("--no-dataset-description", action="store_true", help="不生成 dataset_description.json")
    p.add_argument("--no-participants-json", action="store_true", help="不生成 participants.json")

    p.add_argument("--skip-anat", action="store_true", help="跳过 anat 转换")
    p.add_argument("--skip-dwi", action="store_true", help="跳过 dwi 转换")
    p.add_argument("--skip-rest", action="store_true", help="跳过 rest func 转换")
    p.add_argument("--skip-task", action="store_true", help="跳过 task func 转换")
    p.add_argument("--skip-fix-taskname", action="store_true", help="跳过 JSON 中 TaskName 修正")
    return p.parse_args(argv)


def build_config(ns: argparse.Namespace) -> ConvertConfig:
    dicom_dir = Path(ns.dicom_dir).expanduser().resolve()
    if not dicom_dir.exists():
        raise FileNotFoundError(f"--dicom-dir 不存在: {dicom_dir}")

    if ns.out_dir:
        out_dir = Path(ns.out_dir).expanduser().resolve()
    else:
        out_dir = dicom_dir.parent / "Nifti"

    dcm2niix = find_dcm2niix(ns.dcm2niix)
    ses_labels = tuple(ns.ses_labels)
    if len(ses_labels) != 2:
        raise ValueError("--ses-labels 需要两个值，例如: --ses-labels pre post")

    return ConvertConfig(
        dicom_dir=dicom_dir,
        out_dir=out_dir,
        dcm2niix=dcm2niix,
        ses_labels=(str(ses_labels[0]), str(ses_labels[1])),
        overwrite=bool(ns.overwrite),
        dry_run=bool(ns.dry_run),
        auto_run=bool(ns.auto_run),
        subjects=list(ns.subjects) if ns.subjects else None,
        write_readme=not bool(ns.no_readme),
        write_dataset_description=not bool(ns.no_dataset_description),
        write_participants_json=not bool(ns.no_participants_json),
        do_anat=not bool(ns.skip_anat),
        do_dwi=not bool(ns.skip_dwi),
        do_rest=not bool(ns.skip_rest),
        do_task=not bool(ns.skip_task),
        fix_taskname_in_json=not bool(ns.skip_fix_taskname),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        ns = parse_args(argv)
        cfg = build_config(ns)
    except Exception as e:
        _print(f"[Error] 参数解析失败: {e}")
        return 2

    _print("[Config] dicom_dir = " + str(cfg.dicom_dir))
    _print("[Config] out_dir   = " + str(cfg.out_dir))
    _print("[Config] dcm2niix   = " + str(cfg.dcm2niix))
    _print("[Config] ses_labels = " + str(cfg.ses_labels))
    _print("[Config] overwrite  = " + str(cfg.overwrite))
    _print("[Config] dry_run    = " + str(cfg.dry_run))
    _print("[Config] auto_run   = " + str(cfg.auto_run))

    try:
        prepare_project_files(cfg)
        subjects = list_subject_dirs(cfg.dicom_dir, cfg.subjects)
        if not subjects:
            _print(f"[Warn] 未发现任何被试目录: {cfg.dicom_dir}")
            return 0
        for subj_dir in subjects:
            process_one_subject(cfg, subj_dir)
    except subprocess.CalledProcessError as e:
        _print(f"[Error] dcm2niix 执行失败: {e}")
        return 1
    except Exception as e:
        _print(f"[Error] 执行失败: {e}")
        return 1

    _print("\n[Done] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
