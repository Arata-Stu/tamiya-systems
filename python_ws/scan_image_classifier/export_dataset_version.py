from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.constants import LABELS, VERSION_MANIFEST_COLUMNS


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_rows(
    rows: Sequence[Dict[str, str]],
    splits: Iterable[str],
    import_ids: Sequence[str],
    require_reviewed: bool,
) -> List[Dict[str, str]]:
    valid_splits = {value.strip() for value in splits if value.strip()}
    selected_import_ids = {value.strip() for value in import_ids if value.strip()}
    valid_labels = set(LABELS)

    selected_rows: List[Dict[str, str]] = []
    for row in rows:
        split = row.get("split", "").strip()
        label = row.get("label", "").strip()
        import_id = row.get("import_id", "").strip()

        if valid_splits and split not in valid_splits:
            continue
        if label not in valid_labels:
            continue
        if selected_import_ids and import_id not in selected_import_ids:
            continue
        if require_reviewed and not _as_bool(row.get("reviewed", "")):
            continue

        selected_rows.append(row)

    return selected_rows


def prepare_output_dir(version_dir: Path, overwrite: bool) -> None:
    if version_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Version directory already exists: {version_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)


def export_image(src_path: Path, dst_path: Path, export_mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if export_mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if export_mode == "symlink":
        os.symlink(src_path, dst_path)
        return
    if export_mode == "hardlink":
        os.link(src_path, dst_path)
        return

    raise ValueError(f"Unsupported export_mode: {export_mode}")


def build_manifest_row(
    row: Dict[str, str],
    version_name: str,
    export_relpath: Path,
) -> Dict[str, str]:
    return {
        "version_name": version_name,
        "sample_id": row.get("sample_id", ""),
        "import_id": row.get("import_id", ""),
        "import_name": row.get("import_name", ""),
        "sequence_id": row.get("sequence_id", ""),
        "bag_path": row.get("bag_path", ""),
        "stamp_ns": row.get("stamp_ns", ""),
        "width": row.get("width", ""),
        "height": row.get("height", ""),
        "source_topic": row.get("source_topic", ""),
        "label": row.get("label", ""),
        "label_id": row.get("label_id", ""),
        "split": row.get("split", ""),
        "reviewed": row.get("reviewed", ""),
        "annotated_at": row.get("annotated_at", ""),
        "src_image_path": row.get("image_path", ""),
        "export_image_path": export_relpath.as_posix(),
        "notes": row.get("notes", ""),
    }


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=VERSION_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in VERSION_MANIFEST_COLUMNS})


def write_metadata(
    path: Path,
    version_name: str,
    export_mode: str,
    import_ids: Sequence[str],
    splits: Sequence[str],
    require_reviewed: bool,
    manifest_rows: Sequence[Dict[str, str]],
) -> None:
    label_counts = Counter(row["label"] for row in manifest_rows)
    split_counts = Counter(row["split"] for row in manifest_rows)

    metadata = {
        "version_name": version_name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "export_mode": export_mode,
        "require_reviewed": str(require_reviewed).lower(),
        "import_ids": ",".join(import_ids),
        "splits": ",".join(splits),
        "num_samples": str(len(manifest_rows)),
        "num_imports": str(len({row["import_id"] for row in manifest_rows if row.get("import_id", "").strip()})),
        "num_sequences": str(len({row["sequence_id"] for row in manifest_rows if row.get("sequence_id", "").strip()})),
        "num_train": str(split_counts.get("train", 0)),
        "num_val": str(split_counts.get("val", 0)),
        "num_test": str(split_counts.get("test", 0)),
        "num_rc_car": str(label_counts.get("rc_car", 0)),
        "num_duct_tube": str(label_counts.get("duct_tube", 0)),
        "num_background": str(label_counts.get("background", 0)),
    }

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata.keys()))
        writer.writeheader()
        writer.writerow(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export reviewed annotations into a versioned ImageFolder-style dataset."
    )
    parser.add_argument("--dataset_root", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--annotations", default=None, help="Path to annotations.csv")
    parser.add_argument("--version_name", required=True, help="Dataset version name, e.g. ver1")
    parser.add_argument("--versions_dir", default=None, help="Root directory for exported versions.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to export.")
    parser.add_argument("--import_ids", nargs="*", default=[], help="Limit export to specific import IDs.")
    parser.add_argument(
        "--export_mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to place images into the version directory.",
    )
    parser.add_argument(
        "--include_unreviewed",
        action="store_true",
        help="Include rows even if reviewed=false.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the version directory if it already exists.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations
        else dataset_root / "annotations.csv"
    )
    versions_root = (
        Path(args.versions_dir).expanduser().resolve()
        if args.versions_dir
        else dataset_root / "versions"
    )
    version_dir = versions_root / args.version_name

    rows = load_rows(annotations_path)
    selected_rows = select_rows(
        rows,
        splits=args.splits,
        import_ids=args.import_ids,
        require_reviewed=not args.include_unreviewed,
    )
    if not selected_rows:
        raise RuntimeError("No rows matched the requested filters for dataset export.")

    prepare_output_dir(version_dir, overwrite=args.overwrite)

    manifest_rows: List[Dict[str, str]] = []
    for row in selected_rows:
        src_path = dataset_root / row["image_path"]
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        label = row["label"].strip()
        split = row["split"].strip()
        suffix = src_path.suffix or ".png"
        file_name = f"{row['sample_id']}{suffix}"
        export_relpath = Path(args.version_name) / split / label / file_name
        export_abspath = versions_root / export_relpath

        export_image(src_path, export_abspath, args.export_mode)
        manifest_rows.append(build_manifest_row(row, args.version_name, export_relpath))

    write_manifest(version_dir / "manifest.csv", manifest_rows)
    write_metadata(
        version_dir / "metadata.csv",
        version_name=args.version_name,
        export_mode=args.export_mode,
        import_ids=args.import_ids,
        splits=args.splits,
        require_reviewed=not args.include_unreviewed,
        manifest_rows=manifest_rows,
    )

    label_counts = Counter(row["label"] for row in manifest_rows)
    split_counts = Counter(row["split"] for row in manifest_rows)
    print(f"[DONE] Exported version: {version_dir}")
    print(
        "[DONE] samples="
        f"{len(manifest_rows)} "
        f"train={split_counts.get('train', 0)} "
        f"val={split_counts.get('val', 0)} "
        f"test={split_counts.get('test', 0)} "
        f"rc_car={label_counts.get('rc_car', 0)} "
        f"duct_tube={label_counts.get('duct_tube', 0)} "
        f"background={label_counts.get('background', 0)}"
    )


if __name__ == "__main__":
    main()
