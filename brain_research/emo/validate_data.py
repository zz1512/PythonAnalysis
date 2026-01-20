#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_data.py
数据校验 - 适配 midprep(影像) + miniprep(头动)
"""

import os
from pathlib import Path

BASE_DIR = Path("/public/home/dingrui/BIDS_DATA")

# 配置
CHECK_CONFIG = {
    1: {
        "name": "EMO (Run 1)",
        "root": BASE_DIR / "emo_20250623/v1",
        "events_glob": "beh/func_task_emo/run-1/*task-EMO_events.tsv",
        "fmri_dir": "func/emo_midprep",  # 影像目录
        "conf_dir": "func/emo_miniprep"  # 头动目录
    },
    2: {
        "name": "SOC (Run 2)",
        "root": BASE_DIR / "soc_20250623/v1",
        "events_glob": "beh/func_task_soc/run-1/*task-SOC_events.tsv",
        "fmri_dir": "func/soc_midprep",  # 影像目录
        "conf_dir": "func/soc_miniprep"  # 头动目录
    }
}

# 影像文件 Pattern
FMRI_PATTERNS = {
    "Volume": "*space-MNI*.nii.gz",
    "Surf_L": "*hemi-L*space-fsLR*desc-taskHM6_custom-smooth*.shape.gii",
}

# 头动文件 Pattern
CONF_PATTERN = "*desc-confounds_timeseries.tsv"


def check_exists(base_path, relative_pattern):
    files = list(base_path.glob(relative_pattern))
    return len(files) > 0


def main():
    print("=" * 60)
    print("Data Validation (Split Dirs: midprep & miniprep)")
    print("=" * 60)

    for run_id, cfg in CHECK_CONFIG.items():
        if not cfg["root"].exists():
            print(f"Skipping {cfg['name']}: Root dir not found.")
            continue

        subs = sorted([p.name for p in cfg["root"].glob("sub-*") if p.is_dir()])
        print(f"{cfg['name']} Subjects: {len(subs)}")

        error_count = 0
        for sub in subs:
            sub_root = cfg["root"] / sub

            # 1. Check Events
            if not check_exists(sub_root, cfg["events_glob"]):
                print(f"❌ [缺失] {sub} | {cfg['name']} | Events")
                error_count += 1

            # 2. Check fMRI (midprep)
            fmri_path = sub_root / cfg["fmri_dir"]
            if not fmri_path.exists():
                print(f"❌ [缺失] {sub} | {cfg['name']} | fMRI Dir ({cfg['fmri_dir']})")
            else:
                for p_name, p_glob in FMRI_PATTERNS.items():
                    if not check_exists(fmri_path, p_glob):
                        print(f"❌ [缺失] {sub} | {cfg['name']} | {p_name}")
                        error_count += 1

            # 3. Check Confounds (miniprep)
            conf_path = sub_root / cfg["conf_dir"]
            if not conf_path.exists():
                print(f"❌ [缺失] {sub} | {cfg['name']} | Confounds Dir ({cfg['conf_dir']})")
            else:
                if not check_exists(conf_path, CONF_PATTERN):
                    print(f"❌ [缺失] {sub} | {cfg['name']} | Confounds (.tsv)")
                    error_count += 1

        if error_count == 0:
            print(f"✅ {cfg['name']} looks good!")
        else:
            print(f"⚠️ {cfg['name']} has {error_count} missing files.")
        print("-" * 30)


if __name__ == "__main__":
    main()