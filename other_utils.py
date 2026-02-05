import nibabel as nib
from pathlib import Path
import pandas as pd


def export_brain_research_py_to_txt(
    source_root: Path,
    output_root: Path,
    flatten: bool = True,
) -> tuple[int, int, list[str], list[dict]]:
    py_files = sorted(source_root.rglob("*.py"))
    success = 0
    failed = []
    manifest = []
    for py_file in py_files:
        rel_path = py_file.relative_to(source_root)
        if flatten:
            safe_name = rel_path.as_posix().replace("/", "__")
            out_path = output_root / f"{Path(safe_name).with_suffix('.txt')}"
        else:
            out_path = output_root / rel_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            out_path.write_text(text, encoding="utf-8")
            success += 1
            manifest.append(
                {
                    "source_path": str(py_file),
                    "relative_path": rel_path.as_posix(),
                    "output_path": str(out_path),
                }
            )
        except Exception as exc:
            failed.append(f"{py_file} -> {exc}")
    return len(py_files), success, failed, manifest


def summarize_nii_gz_masks(mask_dir: Path) -> pd.DataFrame:
    mask_files = sorted(mask_dir.glob("*.nii.gz"))
    records = []
    for f in mask_files:
        img = nib.load(str(f))
        shape = img.shape
        voxel_size = nib.affines.voxel_sizes(img.affine)
        records.append(
            {
                "文件名": f.name,
                "维度": shape,
                "体素大小(mm)": tuple(round(v, 2) for v in voxel_size),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "brain_research"
    output_root = Path("/Users/bytedance/Documents/brain_research_txt_export")
    total, success, failed, manifest = export_brain_research_py_to_txt(
        source_root,
        output_root,
        flatten=True,
    )
    pd.DataFrame(manifest).to_csv(output_root / "export_manifest.csv", index=False)
    print(f"导出完成: 总计 {total}，成功 {success}，失败 {len(failed)}")
    if failed:
        print("失败列表:")
        for item in failed:
            print(item)


if __name__ == "__main__":
    main()
