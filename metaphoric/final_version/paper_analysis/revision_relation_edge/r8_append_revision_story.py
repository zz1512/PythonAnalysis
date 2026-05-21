#!/usr/bin/env python3
"""Task r8: generate revision story/final drafts without overwriting current files."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_revision import PROJECT_ROOT, out_path, write_text

MODULE = "r8_append_revision_story"


def build_story() -> str:
    src = PROJECT_ROOT / "paper_outputs" / "qc" / "revision_relation_edge" / "r6_update_evidence_tier_and_story" / "storyline_revision.md"
    return src.read_text(encoding="utf-8") if src.exists() else "# Storyline revision\n\nr6 output not found.\n"


def build_final(story: str) -> str:
    return """# Result Final Revision Draft

本稿是 relation_edge_story_revision 的写作草案。它不覆盖当前 result_final.md，而是把新增 revision 分析放到新故事结构中，供下一轮人工 review。

## Revised Core

成功的隐喻学习不是简单增强词对相似性，而是在 post 阶段将 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 在 retrieval 阶段以任务依赖方式重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。

## Revision Analysis Integration

""" + story + """

## Writing Decision

当前主线不需要推翻。新增 r1-r5 分析主要强化边界：材料语义距离和 novelty 不能单独解释主效应；trajectory geometry 当前只能作为 pair-similarity proxy；semantic-hpc coupling 支持 retrieval-stage coordination，但不支持 YY-specific causal communication；MVPA 回到 stage-state evidence 的位置。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-section", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    story = build_story()
    final = build_final(story)
    preview = out_path(MODULE, "revision_story_preview.md")
    write_text(story, preview)

    targets = [
        (PROJECT_ROOT / "result_story_revision.md", story),
        (PROJECT_ROOT / "result_final_revision.md", final),
    ]
    log = [f"dry_run={args.dry_run}", f"replace_section={args.replace_section}", f"skip_if_exists={args.skip_if_exists}", f"preview={preview}"]
    if not args.dry_run:
        for path, text in targets:
            if path.exists() and args.skip_if_exists:
                log.append(f"skip existing: {path}")
                continue
            path.write_text(text, encoding="utf-8")
            log.append(f"wrote: {path}")
    write_text("\n".join(log) + "\n", out_path(MODULE, "revision_append_log.txt"))
    print(f"Wrote r8 preview/log to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
