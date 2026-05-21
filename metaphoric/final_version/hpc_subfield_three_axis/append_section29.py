#!/usr/bin/env python3
"""A: 在 result_new_meta_roi.md 末尾追加 §29 Three-Axis Narrative Reframing。

仅 append；写入前断言文件中不存在 `## 29.` 标题，禁止重复追加；
不修改 §0–§28 任何字符。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_subfield import (
    add_common_args,
    default_config,
    git_status,
    write_outputs,
)

MODULE = "append_section29"
SECTION_TITLE = "## 29. Three-Axis Narrative Reframing (Favila / Schlichting / Bein)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--result-md", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def section_text() -> str:
    return f"""
{SECTION_TITLE}

本节仅追加海马亚区切分 + 三轴叙事的收束说明，不修改 §0–§28 任何既有结论。
此节深化 §28 的 stage-specific reorganization 主线，把分散的 ROI 结果挂到三个独立但互补的轴上。

### 29.1 三轴定义

- **Favila axis (differentiation)**：trained pair edge 在学习后相似度下降 / 几何分化（Favila, Chanales & Kuhl, 2016 *Nat Commun*）。
- **Schlichting axis (anterior–posterior gradient)**：海马前段倾向整合（head 端 drop 趋零或正向），后段倾向分离（tail 端 drop 显著为负）（Schlichting, Mumford & Preston, 2015 *Nat Commun*；Audrain & McAndrews, 2022 *Nat Commun*）。
- **Bein axis (left semantic assimilation)**：左 IFG / 颞极在先验语义脚手架支持下进行同化，与海马侧分化形成互补（Bein, Plotek & Davachi, 2020 *Nat Commun*）。

### 29.2 海马亚区切分（S1–S4）

- 切分来源：FreeSurfer FS60 hipposubfields（优先），缺失时按 MNI y 轴三等分既有 `meta_L/R_hippocampus` mask；provenance 字段在 `subfield_manifest.tsv` 显式标注。
- 6 subROI：`hpc_<L|R>_<head|body|tail>`，体素数 < 30 标 `qc_warn`，< 10 跳过下游。
- S3 subfield Step5C：在每个 subROI 上重做 condition × time RSA，保持 §3 主结果方向；FDR family = `hpc_subfield_three_axis`。
- S4 subfield edge specificity（C1–C4）：YY trained drop 是否（C1）非零、（C2）显著大于 KJ trained、（C3）显著大于 pseudo-edge、（C4）显著大于 non-edge。

### 29.3 前后轴 contrast（S5，Schlichting axis）

- 模型：`drop ~ axis_position * C(condition) + (1|subject) + (1|item)`，axis_position：head=+1 / body=0 / tail=-1。
- 半球分别跑 + 双侧 pooled，三套结果。
- 关键判据：`axis_position × condition[YY]` 项是否显著（FDR family 内）。

### 29.4 R hpc 与 R 颞极的 MDS 几何（V1）

- 5 个 panel：`meta_R_hippocampus` 整体 + `hpc_R_head` / `hpc_R_body` / `hpc_R_tail` + `meta_R_temporal_pole`。
- 距离：1 - Pearson r，跨被试 fisher-z 平均；MDS 参数固定 `n_init=20, random_state=42`。
- 输出 1200 dpi pdf + png + `mds_coords.tsv`，trained-edge drop 在视觉层面应表现为 YY trained 配对在 post 距离扩大。

### 29.5 三轴证据整合（N1）

`three_axis_evidence_table.tsv` 把 §3 / §4 / §9 / §27 与 S3 / S4 / S5 的所有显著项按轴归类，
每条证据携带 source / roi / metric / term / effect / p / q / tier，便于稿件正文与图注同步引用。

### 29.6 与 §28 的关系

§28 已经把主线收束为 stage-specific relation-edge reorganization；
§29 进一步把它分解为三个机制轴，并通过亚区切分 + MDS 提供解剖与几何层面的视觉证据。
若 S3 / S4 / S5 中至少一项通过 family FDR，或 V1 MDS 在视觉上清晰呈现 trained-edge drop，则 §29 升级主线；
否则 §29 保留为 robustness check，§28 仍是主线，不触动 §0–§27。
""".strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    result_md = Path(args.result_md or cfg.final_root / "result_new_meta_roi.md")
    if not result_md.exists():
        raise FileNotFoundError(result_md)
    original = result_md.read_text(encoding="utf-8")
    if "## 29." in original or SECTION_TITLE in original:
        raise FileExistsError(f"{SECTION_TITLE} already exists; refusing to append duplicate section.")
    text = "\n\n" + section_text()
    if not args.dry_run:
        with result_md.open("a", encoding="utf-8") as f:
            f.write(text)
    log = (
        "# Append Section 29\n\n"
        f"- result_md: `{result_md}`\n"
        f"- dry_run: {args.dry_run}\n"
        f"- git_status: `{git_status(result_md)}`\n"
    )
    write_outputs(cfg, MODULE, {
        "section29_preview.md": section_text(),
        "append_section29.log": log,
    })


if __name__ == "__main__":
    main()
