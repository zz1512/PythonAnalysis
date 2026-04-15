from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, paired_t_summary, save_json, write_table
from common.pattern_metrics import load_masked_samples, neural_rdm_vector


def model_from_condition(metadata: pd.DataFrame) -> np.ndarray:
    labels = metadata["condition"].astype(str).to_numpy()
    matrix = (labels[:, None] != labels[None, :]).astype(float)
    return matrix[np.triu_indices_from(matrix, k=1)]


def model_from_numeric(metadata: pd.DataFrame, column: str) -> np.ndarray:
    values = metadata[column].to_numpy(dtype=float)
    matrix = np.abs(values[:, None] - values[None, :])
    return matrix[np.triu_indices_from(matrix, k=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare neural RDMs with candidate model RDMs.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--metadata-template", default="{time}_{condition}_metadata.tsv")
    parser.add_argument("--numeric-model-cols", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows = []
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                image_paths = []
                meta_frames = []
                for condition in ["yy", "kj"]:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    meta_path = subject_dir / args.metadata_template.format(time=time, condition=condition)
                    if image_path.exists() and meta_path.exists():
                        image_paths.append((condition, image_path))
                        meta = pd.read_csv(meta_path, sep="\t")
                        meta["condition"] = condition
                        meta_frames.append(meta)
                if len(image_paths) != 2:
                    continue
                samples = np.vstack([load_masked_samples(path, roi_path) for _, path in image_paths])
                metadata = pd.concat(meta_frames, ignore_index=True)
                neural = neural_rdm_vector(samples, metric="correlation")

                models = {"condition": model_from_condition(metadata)}
                for column in args.numeric_model_cols:
                    if column in metadata.columns:
                        models[column] = model_from_numeric(metadata, column)

                for model_name, model_vector in models.items():
                    rho, pvalue = stats.spearmanr(neural, model_vector)
                    rows.append({
                        "subject": subject_dir.name,
                        "roi": roi_name,
                        "time": time,
                        "model": model_name,
                        "rho": rho,
                        "p_value": pvalue,
                    })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "model_rdm_subject_metrics.tsv")

    summaries = []
    for (roi_name, model_name), sub_frame in frame.groupby(["roi", "model"]):
        pivot = sub_frame.pivot(index="subject", columns="time", values="rho").dropna()
        summary = {"roi": roi_name, "model": model_name, **paired_t_summary(pivot["post"], pivot["pre"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_{model_name}_rdm_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "model_rdm_group_summary.tsv")


if __name__ == "__main__":
    main()
