#!/usr/bin/env python3
"""Language / ToM network sensitivity aggregator (A1).

This script is strictly an **aggregation** utility. It does NOT recompute RSA.
It reads the subject-level / group-level products that the primary RSA
scripts have already written under ``$paper/qc/<analysis>_{roi_tag}/`` for
``roi_tag ∈ {language_network, tom_network}``, concatenates them with a
SI-sensitivity flag, and mirrors tidy copies under
``$paper/qc/language_tom_network_sensitivity_{roi_tag}/`` and
``$paper/tables_si/`` so the downstream paper writeup can cite a single
file per ROI set.

Context: switching ROI changes the pattern vectors that go into RSA, so the
primary scripts (``step5c_rsa.py``, ``edge_specificity_rsa.py``,
``learning_condition_rdm.py``, ``retrieval_pair_similarity.py``) MUST be
re-run under ``METAPHOR_ROI_SET ∈ {language_network, tom_network}`` before
invoking this aggregator. See ``new_rerun_list.md`` section P12.A1.

Usage::

    python metaphoric/final_version/rsa_analysis/language_tom_network_sensitivity.py \
        --paper-output-root $paper \
        --roi-sets language_network tom_network

Outputs (per ``roi_tag``):

  $paper/qc/language_tom_network_sensitivity_{roi_tag}/step5c_summary.tsv
  $paper/qc/language_tom_network_sensitivity_{roi_tag}/edge_summary.tsv
  $paper/qc/language_tom_network_sensitivity_{roi_tag}/trajectory_summary.tsv
  $paper/qc/language_tom_network_sensitivity_{roi_tag}/retrieval_summary.tsv
  $paper/qc/language_tom_network_sensitivity_{roi_tag}/missing_sources.tsv
  $paper/qc/language_tom_network_sensitivity_{roi_tag}/sensitivity_manifest.json
  $paper/tables_si/table_language_tom_network_sensitivity_{roi_tag}.tsv

All output rows carry an explicit ``family = "SI_sensitivity"`` column so
downstream writeups never confuse these with primary BH-FDR families.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


# Per-analysis candidate source files, written by existing primary scripts.
# The keys are the "source" labels; values are a list of candidate relative
# paths (first existing one wins). ``{roi_tag}`` is substituted at runtime.
CANDIDATE_SOURCES: dict[str, list[str]] = {
    "step5c": [
        "qc/step5c_rsa_{roi_tag}/step5c_group_summary.tsv",
        "qc/step5c_rsa_{roi_tag}/step5c_group_fdr.tsv",
        "qc/step5c_rsa_{roi_tag}/step5c_pair_similarity_group.tsv",
    ],
    "edge": [
        "qc/edge_specificity_{roi_tag}/edge_specificity_group.tsv",
        "qc/edge_specificity_{roi_tag}/edge_specificity_group_fdr.tsv",
    ],
    "trajectory": [
        "qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_one_sample.tsv",
        "qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_pairwise.tsv",
    ],
    "retrieval": [
        "qc/retrieval_geometry_{roi_tag}/retrieval_geometry_group_fdr.tsv",
        "qc/retrieval_geometry/retrieval_geometry_group_fdr_{roi_tag}.tsv",
        "qc/retrieval_pair_similarity_{roi_tag}/retrieval_geometry_group_fdr.tsv",
    ],
}


def _resolve_first_existing(
    paper_root: Path, candidates: list[str], roi_tag: str
) -> Path | None:
    for rel in candidates:
        candidate = paper_root / rel.format(roi_tag=roi_tag)
        if candidate.exists():
            return candidate
    return None


def _load_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        frame = read_table(path)
    except Exception as exc:  # pragma: no cover - defensive
        return pd.DataFrame({"_load_error": [str(exc)], "_source_path": [str(path)]})
    frame = frame.copy()
    frame["_source_path"] = str(path)
    return frame


def _annotate(frame: pd.DataFrame, *, roi_set: str, roi_tag: str, analysis: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["roi_set_requested"] = roi_set
    frame["roi_tag"] = roi_tag
    frame["analysis"] = analysis
    frame["family"] = "SI_sensitivity"
    frame["in_main_fdr_family"] = False
    return frame


def aggregate_roi_set(paper_root: Path, roi_set: str) -> dict[str, pd.DataFrame]:
    roi_tag = sanitize_roi_tag(roi_set)
    per_analysis: dict[str, pd.DataFrame] = {}
    for analysis, candidates in CANDIDATE_SOURCES.items():
        resolved = _resolve_first_existing(paper_root, candidates, roi_tag)
        frame = _load_if_exists(resolved)
        frame = _annotate(frame, roi_set=roi_set, roi_tag=roi_tag, analysis=analysis)
        per_analysis[analysis] = frame
    return per_analysis


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Aggregate Language/ToM network RSA outputs into SI sensitivity tables.",
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=base_dir / "paper_outputs",
        help="Root used by primary RSA scripts (default: $PYTHON_METAPHOR_ROOT/paper_outputs).",
    )
    parser.add_argument(
        "--roi-sets",
        nargs="+",
        default=["language_network", "tom_network"],
        help="ROI sets to aggregate. Must match roi_tag of previously re-run primary RSA.",
    )
    args = parser.parse_args()

    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    for roi_set in args.roi_sets:
        roi_tag = sanitize_roi_tag(roi_set)
        out_dir = ensure_dir(
            args.paper_output_root / "qc" / f"language_tom_network_sensitivity_{roi_tag}"
        )
        per_analysis = aggregate_roi_set(args.paper_output_root, roi_set)

        missing_rows: list[dict[str, object]] = []
        tidy_rows: list[pd.DataFrame] = []
        for analysis, frame in per_analysis.items():
            target = out_dir / f"{analysis}_summary.tsv"
            write_table(frame, target)
            if frame.empty:
                missing_rows.append(
                    {
                        "roi_set": roi_set,
                        "roi_tag": roi_tag,
                        "analysis": analysis,
                        "status": "missing",
                        "candidates_searched": "; ".join(
                            str(args.paper_output_root / rel.format(roi_tag=roi_tag))
                            for rel in CANDIDATE_SOURCES[analysis]
                        ),
                    }
                )
            else:
                tidy_rows.append(frame.assign(analysis_label=analysis))

        write_table(
            pd.DataFrame(missing_rows),
            out_dir / "missing_sources.tsv",
        )

        tidy = (
            pd.concat(tidy_rows, ignore_index=True, sort=False)
            if tidy_rows
            else pd.DataFrame()
        )
        write_table(
            tidy,
            tables_si / f"table_language_tom_network_sensitivity_{roi_tag}.tsv",
        )

        save_json(
            {
                "roi_set": roi_set,
                "roi_tag": roi_tag,
                "paper_output_root": str(args.paper_output_root),
                "analyses": list(CANDIDATE_SOURCES.keys()),
                "n_missing_analyses": int(len(missing_rows)),
                "family": "SI_sensitivity",
                "in_main_fdr_family": False,
                "note": (
                    "Aggregation only. Primary RSA must already have been re-run under "
                    "METAPHOR_ROI_SET={roi_set} by step5c_rsa.py / edge_specificity_rsa.py / "
                    "learning_condition_rdm.py / retrieval_pair_similarity.py."
                ).format(roi_set=roi_set),
            },
            out_dir / "sensitivity_manifest.json",
        )


if __name__ == "__main__":
    main()
