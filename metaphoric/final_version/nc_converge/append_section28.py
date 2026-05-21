#!/usr/bin/env python3
"""A1: append Section 28 storyline reframing to result_new_meta_roi.md."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_nc import add_common_args, default_config, git_status, write_outputs

MODULE = "append_section28"
SECTION_TITLE = "## 28. Storyline Reframing (NC target)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--result-md", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def section_text() -> str:
    return f"""
{SECTION_TITLE}

本节只追加新的投稿收束说明，不修改前文 §0-§27 的任何既有结论。当前 NC 目标下，主线不再写成
"隐喻学习让 source 与 target 更相似"，而是写成 **stage-specific relation-edge reorganization + task-dependent rebinding**。

### 28.1 旧主线与新主线

| 层面 | 不再主张 | NC 目标下主张 |
| --- | --- | --- |
| 核心机制 | 全局 semantic convergence | 阶段性 learned relational edge 重组 |
| 学习阶段 | 已形成强 pair-selective trace | YY/KJ condition-level task geometry 被调动 |
| post 阶段 | 简单相似性升高 | trained-edge differentiation / adaptive separation |
| retrieval 阶段 | 独立的强 mnemonic reinstatement | task-dependent pair-structure rebound，且需要 shared-post component 控制 |
| 行为桥 | retrieval rebinding difference-score 直接预测记忆 | post component subsequent-memory 是更保守的 endpoint |

### 28.2 三个关键词

- **Relation-edge reorganization**：学习后的 pair/edge-level neural geometry 被重新组织，而不是两个词整体更接近。
- **Adaptive separation**：post 阶段 YY trained edge 的相似性下降可解释为降低干扰、提高关系特异性的分化。
- **Task-dependent rebinding**：retrieval 阶段 pair structure 可被任务需求重新绑定，但该效应必须与 post component 区分。

### 28.3 与现有结果的对应

- 行为：YY memory accuracy 稳定高于 KJ，是主线的行为入口。
- Learning：condition-level geometry 强，same-pair learning trace 弱，因此 learning 写成任务/条件几何被调动。
- Post：Step5C 与 edge-specificity 指向 YY trained-edge differentiation，是当前最强机制端点。
- Retrieval：pair-structure rebound 与 YY/KJ decoding 再次出现，但 shared-post audit 后不写成单独因果桥。
- Reviewer-supp：M2/M6/M9/M11 作为防守层，排除 mean activation、novelty/repetition、memory coding 与 KJ-as-null 等简单解释。

### 28.4 后续 Go/No-Go 规则

若 A4/B2/B3 中至少两项支持主方向，则进入机制升级与主文 Results/Discussion 收束；
若 A4/B1/B2/B3 中仅一项稳定，则保持 HOLD：不扩展搜索空间，把不稳链条写成边界条件；
若均不稳，则停止机制扩展，主线退回最稳的 stage-specific reorganization 描述。
""".strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    result_md = Path(args.result_md or cfg.final_root / "result_new_meta_roi.md")
    if not result_md.exists():
        raise FileNotFoundError(result_md)
    original = result_md.read_text(encoding="utf-8")
    if SECTION_TITLE in original:
        raise FileExistsError(f"{SECTION_TITLE} already exists; refusing to append duplicate section.")
    text = "\n\n" + section_text()
    if not args.dry_run:
        with result_md.open("a", encoding="utf-8") as f:
            f.write(text)
    log = f"# Append Section 28\n\n- result_md: `{result_md}`\n- dry_run: {args.dry_run}\n- git_status: `{git_status(result_md)}`\n"
    write_outputs(cfg, MODULE, {"section28_preview.md": section_text(), "append_section28.log": log})


if __name__ == "__main__":
    main()
