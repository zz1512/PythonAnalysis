#!/usr/bin/env python3
"""Append Section 27 to result_new_meta_roi.md without modifying earlier text."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared import add_common_args, default_config, read_table, write_text

SECTION_TITLE = '## 27. Reviewer-Requested Supplementary Analyses'


def table_exists(cfg, module: str, filename: str) -> bool:
    return (cfg.output_root / module / filename).exists()


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _short_table(frame: pd.DataFrame, limit: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    return frame.head(limit).to_markdown(index=False)


def _status_line(label: str, path: Path) -> str:
    return f"- {label}: `{path}` {'available' if path.exists() else 'missing'}."


def build_section(cfg) -> str:
    lines = [
        "",
        SECTION_TITLE,
        "",
        "This section is appended by `reviewer_supp/append_section27.py`. Sections 1-26 are not modified.",
        "",
        "### Scope Decisions",
        "",
        "- Stimulus-level neural covariate RSA is not rerun because behavioral-level covariate control already addresses this concern for the current claim strength.",
        "- Additional representational drift control is not rerun because KJ trained-edge and baseline pseudo-edge already serve as time-drift controls in the contrast logic.",
        "- YY-specific whole-brain searchlight is not added because the manuscript story remains ROI-centered.",
        "- Post-test questionnaire alignment is deferred until questionnaire data are available.",
        "",
        "### Generated Reviewer-Supplement Modules",
        "",
    ]
    modules = [
        ("M2", "m2_univariate_sanity", "roi_mean_activation_models.tsv", "Univariate ROI activation sanity check."),
        ("M3", "m3_learning_dm", "learning_dm_univariate.tsv", "Learning-stage subsequent-memory Dm analysis."),
        ("M4", "m4_correct_ers", "ers_same_vs_other.tsv", "Correct learning-to-retrieval ERS analysis."),
        ("M5", "m5_theoretical_reframing", "network_reframing.md", "Network-level separation/reorganization framing."),
        ("M6", "m6_novelty_repetition", "post_minus_pre_amplitude.tsv", "Pre/post univariate novelty/repetition check."),
        ("M9", "m9_logistic_dm", "logistic_dm_strict.tsv", "Strict/lenient binarized-memory GLM sensitivity."),
        ("M11", "m11_kj_fair_characterization", "kj_significant_results.tsv", "Fair characterization of KJ-specific signatures."),
    ]
    for label, module, filename, desc in modules:
        status = "available" if table_exists(cfg, module, filename) else "not generated yet"
        lines.append(f"- {label}: {desc} Output status: `{status}`.")

    lines.extend(
        [
            "",
            "### M2 Univariate Sanity (summary)",
            "",
        ]
    )
    m2_models = _read_if_exists(cfg.output_root / "m2_univariate_sanity" / "roi_mean_activation_models.tsv")
    lines.append(_short_table(m2_models))

    lines.extend(
        [
            "",
            "### M3 Learning Dm (summary)",
            "",
        ]
    )
    m3_univar = _read_if_exists(cfg.output_root / "m3_learning_dm" / "learning_dm_univariate.tsv")
    m3_rsa = _read_if_exists(cfg.output_root / "m3_learning_dm" / "learning_dm_rsa.tsv")
    lines.append("#### Univariate")
    lines.append(_short_table(m3_univar))
    lines.append("")
    lines.append("#### RSA")
    lines.append(_short_table(m3_rsa))

    lines.extend(
        [
            "",
            "### M4 Correct ERS (summary)",
            "",
        ]
    )
    m4_group = _read_if_exists(cfg.output_root / "m4_correct_ers" / "ers_same_vs_other.tsv")
    m4_dm = _read_if_exists(cfg.output_root / "m4_correct_ers" / "ers_dm_interaction.tsv")
    lines.append("#### Same-pair > other-pair")
    lines.append(_short_table(m4_group))
    lines.append("")
    lines.append("#### ERS × memory")
    lines.append(_short_table(m4_dm))

    lines.extend(
        [
            "",
            "### M5 Theoretical Reframing",
            "",
        ]
    )
    m5_text = (cfg.output_root / "m5_theoretical_reframing" / "network_reframing.md")
    if m5_text.exists():
        lines.append(m5_text.read_text(encoding="utf-8"))
    else:
        lines.append("_network_reframing.md not generated yet._")

    lines.extend(
        [
            "",
            "### M6 Novelty/Repetition Univariate (summary)",
            "",
        ]
    )
    m6_group = _read_if_exists(cfg.output_root / "m6_novelty_repetition" / "post_minus_pre_group.tsv")
    m6_model = _read_if_exists(cfg.output_root / "m6_novelty_repetition" / "post_minus_pre_condition_model.tsv")
    lines.append("#### Group one-sample")
    lines.append(_short_table(m6_group))
    lines.append("")
    lines.append("#### Condition model")
    lines.append(_short_table(m6_model))

    lines.extend(
        [
            "",
            "### M9 Logistic Memory Sensitivity (summary)",
            "",
        ]
    )
    m9_strict = _read_if_exists(cfg.output_root / "m9_logistic_dm" / "logistic_dm_strict.tsv")
    m9_lenient = _read_if_exists(cfg.output_root / "m9_logistic_dm" / "logistic_dm_lenient.tsv")
    lines.append("#### Strict")
    lines.append(_short_table(m9_strict))
    lines.append("")
    lines.append("#### Lenient")
    lines.append(_short_table(m9_lenient))

    lines.extend(
        [
            "",
            "### M11 KJ-specific Signature (summary)",
            "",
        ]
    )
    m11 = _read_if_exists(cfg.output_root / "m11_kj_fair_characterization" / "kj_significant_results.tsv")
    lines.append(_short_table(m11))
    lines.extend([
        "",
        "### Evidence Tiering",
        "",
        "- Primary evidence remains the ROI-based trained-edge differentiation result and its mixed-effects replication.",
        "- Supplementary evidence comes from univariate sanity checks, learning-stage Dm, ERS, novelty/repetition amplitude, and logistic-memory sensitivity.",
        "- Boundary evidence includes KJ-specific signatures and any non-convergent module outputs; these should be discussed honestly rather than used as primary claims.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Append Section 27 to result_new_meta_roi.md. Default is preview only.")
    parser.add_argument("--dry-run", action="store_true", help="Deprecated alias for the default preview-only behavior.")
    args = parser.parse_args()
    cfg = default_config(args)
    result_file = args.result_file or Path(__file__).resolve().parents[1] / "result_new_meta_roi.md"
    section = build_section(cfg)
    if args.dry_run or not args.apply:
        preview = cfg.output_root / "section27_preview.md"
        write_text(section, preview)
        return
    original = result_file.read_text(encoding="utf-8")
    if SECTION_TITLE in original:
        raise RuntimeError(f"{SECTION_TITLE} already exists; refusing to append twice.")
    result_file.write_text(original.rstrip() + "\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
