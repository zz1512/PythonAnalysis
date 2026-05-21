#!/usr/bin/env python3
"""Go/Hold/No-Go decision after A4 and B1-B3."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, default_config, discover_tables, markdown_table, read_table, write_outputs

MODULE = "go_no_go"
INPUT_PATTERNS = ["a4_post_memory_component/*.tsv", "b1_staged_path/*.tsv", "b2_crossnobis_robustness/*.tsv", "b3_repr_coupling/*.tsv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def evidence(path: Path, frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"module": path.parent.name, "status": "empty", "stable": False, "reason": "empty table"}
    ok = frame.get("status", pd.Series(dtype=str)).astype(str).eq("ok") if "status" in frame.columns else pd.Series(True, index=frame.index)
    q = pd.to_numeric(frame.get("q_bh", frame.get("q", pd.Series(dtype=float))), errors="coerce")
    p = pd.to_numeric(frame.get("p", pd.Series(dtype=float)), errors="coerce")
    stable = bool(((ok & (q < 0.05)).any()) or (q.isna().all() and (ok & (p < 0.05)).any()))
    return {"module": path.parent.name, "status": "ok" if stable else "not_stable", "stable": stable, "n_rows": len(frame), "min_p": float(p.min()) if p.notna().any() else None, "min_q": float(q.min()) if q.notna().any() else None}


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.output_root, INPUT_PATTERNS)
    rows = []
    for path in paths:
        try:
            rows.append(evidence(path, read_table(path)))
        except Exception as exc:
            rows.append({"module": path.parent.name, "status": f"failed: {exc}", "stable": False})
    table = pd.DataFrame(rows)
    evidence_modules = {"a4_post_memory_component", "b1_staged_path", "b2_crossnobis_robustness", "b3_repr_coupling"}
    critical_modules = {"a4_post_memory_component", "b2_crossnobis_robustness", "b3_repr_coupling"}
    module_stability = table.groupby("module", dropna=False)["stable"].max() if not table.empty and "stable" in table else pd.Series(dtype=bool)
    stable_count = int(module_stability.reindex(sorted(evidence_modules), fill_value=False).sum())
    critical_stable_count = int(module_stability.reindex(sorted(critical_modules), fill_value=False).sum())
    if critical_stable_count >= 2:
        decision = "GO"
    elif stable_count >= 1:
        decision = "HOLD"
    else:
        decision = "NO-GO"
    md = f"# Go/No-Go Stage 6 Decision\n\nDecision: **{decision}**\n\n"
    md += "Primary GO rule: at least two of A4/B2/B3 support the main direction.\n\n"
    md += "Fallback HOLD rule: at least one of A4/B1/B2/B3 is stable enough to preserve a bounded stagewise reorganization story.\n\n"
    if not table.empty:
        md += markdown_table(table) + "\n\n"
    if decision == "GO":
        md += "Next: proceed to mechanism upgrades and Results/Discussion consolidation.\n"
    elif decision == "HOLD":
        md += "Next: do not broaden the search space; write unstable links as boundary conditions and keep the paper centered on stage-specific reorganization.\n"
    else:
        md += "Next: stop mechanism expansion and return to the most stable descriptive story.\n"
    write_outputs(cfg, MODULE, {"go_no_go_stage6.md": md, "go_no_go_evidence.tsv": table})


if __name__ == "__main__":
    main()
