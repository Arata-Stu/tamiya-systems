from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from src.constants import (
    ANNOTATION_COLUMNS,
    IMPORT_COLUMNS,
    LABELS,
    LABEL_ALIASES,
    LABEL_TO_ID,
)

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "import"


def _make_import_id(import_name: Optional[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if import_name:
        return f"{timestamp}_{_slugify(import_name)}"
    return timestamp


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_label(value: str) -> str:
    key = value.strip().lower().replace("/", "_")
    if key not in LABEL_ALIASES:
        raise ValueError(
            f"Unsupported label '{value}'. Supported labels or aliases: {sorted(LABEL_ALIASES.keys())}"
        )
    return LABEL_ALIASES[key]


def _load_existing_annotations(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            if sample_id:
                rows[sample_id] = row
    return rows


def _merge_annotations(
    existing_rows: Dict[str, Dict[str, str]],
    imported_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged = dict(existing_rows)
    preserve_keys = {
        "label",
        "label_id",
        "split",
        "reviewed",
        "annotated_at",
        "auto_label",
        "auto_confidence",
        "notes",
    }

    for row in imported_rows:
        sample_id = row["sample_id"]
        if sample_id in merged:
            preserved = merged[sample_id]
            combined = dict(row)
            for key in preserve_keys:
                combined[key] = preserved.get(key, combined.get(key, ""))
            merged[sample_id] = combined
        else:
            merged[sample_id] = row

    return [merged[key] for key in sorted(merged.keys())]


def _write_annotations(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ANNOTATION_COLUMNS})


def _load_existing_imports(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            import_id = row.get("import_id", "").strip()
            if import_id:
                rows[import_id] = row
    return rows


def _write_imports(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=IMPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in IMPORT_COLUMNS})


def _build_import_summaries(
    rows: Sequence[Dict[str, str]],
    existing_imports: Dict[str, Dict[str, str]],
    default_created_at: str,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        import_id = row.get("import_id", "").strip()
        if not import_id:
            continue
        grouped.setdefault(import_id, []).append(row)

    summaries: List[Dict[str, str]] = []
    for import_id in sorted(grouped.keys()):
        import_rows = grouped[import_id]
        base = existing_imports.get(import_id, {})
        import_names = [
            row.get("import_name", "").strip()
            for row in import_rows
            if row.get("import_name", "").strip()
        ]
        source_topics = sorted(
            {
                row.get("source_topic", "").strip()
                for row in import_rows
                if row.get("source_topic", "").strip()
            }
        )

        summaries.append(
            {
                "import_id": import_id,
                "import_name": import_names[-1] if import_names else base.get("import_name", import_id),
                "created_at": base.get("created_at", default_created_at),
                "source_topic": ",".join(source_topics),
                "num_sequences": str(len({row.get("sequence_id", "").strip() for row in import_rows})),
                "num_samples": str(len(import_rows)),
                "notes": base.get("notes", ""),
            }
        )

    return summaries


def _image_size(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _copy_image(src_path: Path, dst_path: Path, mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if mode == "symlink":
        os.symlink(src_path, dst_path)
        return
    if mode == "hardlink":
        os.link(src_path, dst_path)
        return
    raise ValueError(f"Unsupported copy mode: {mode}")


def _sequence_id(label: str, group_key: str) -> str:
    digest = hashlib.sha1(f"{label}:{group_key}".encode("utf-8")).hexdigest()[:8]
    slug = _slugify(group_key)
    return f"external_{label}_{slug}_{digest}"


def _iter_class_dir_images(source_dir: Path) -> List[Tuple[Path, str, Path]]:
    tasks: List[Tuple[Path, str, Path]] = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue

        try:
            label = _normalize_label(child.name)
        except ValueError:
            continue

        for image_path in sorted(child.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                continue
            tasks.append((image_path, label, child))
    return tasks


def _iter_single_label_images(source_dir: Path, label: str) -> List[Tuple[Path, str, Path]]:
    tasks: List[Tuple[Path, str, Path]] = []
    for image_path in sorted(source_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        tasks.append((image_path, label, source_dir))
    return tasks


def _build_row(
    dataset_root: Path,
    import_id: str,
    import_name: str,
    source_topic: str,
    copy_mode: str,
    reviewed: bool,
    image_path: Path,
    label: str,
    label_root: Path,
) -> Dict[str, str]:
    source_path = image_path.expanduser().resolve()
    label_root = label_root.expanduser().resolve()
    rel_parent = source_path.parent.relative_to(label_root)
    group_key = rel_parent.as_posix() if rel_parent.as_posix() != "." else "_root"
    sequence_id = _sequence_id(label, group_key)

    rel_source = source_path.relative_to(label_root)
    digest = hashlib.sha1(rel_source.as_posix().encode("utf-8")).hexdigest()[:10]
    suffix = source_path.suffix.lower() or ".png"
    file_name = f"{source_path.stem}_{digest}{suffix}"
    image_relpath = Path("imports") / import_id / "raw" / sequence_id / "images" / file_name
    image_abspath = dataset_root / image_relpath

    width, height = _image_size(source_path)
    _copy_image(source_path, image_abspath, mode=copy_mode)

    sample_id = f"{import_id}__{sequence_id}_{digest}"
    annotated_at = datetime.now().isoformat(timespec="seconds") if reviewed else ""
    return {
        "sample_id": sample_id,
        "import_id": import_id,
        "import_name": import_name,
        "sequence_id": sequence_id,
        "bag_path": str(source_path.parent),
        "image_path": image_relpath.as_posix(),
        "stamp_ns": "",
        "width": str(width),
        "height": str(height),
        "source_topic": source_topic,
        "label": label,
        "label_id": str(LABEL_TO_ID[label]),
        "split": "",
        "reviewed": str(reviewed).lower(),
        "annotated_at": annotated_at,
        "auto_label": "",
        "auto_confidence": "",
        "notes": "",
    }


def _write_sequence_manifests(dataset_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["import_id"], row["sequence_id"]), []).append(row)

    for (import_id, sequence_id), group_rows in grouped.items():
        manifest_path = dataset_root / "imports" / import_id / "raw" / sequence_id / "samples.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
            writer.writeheader()
            for row in group_rows:
                writer.writerow({column: row.get(column, "") for column in ANNOTATION_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import external labeled images into the scan image classifier dataset."
    )
    parser.add_argument("--dataset_root", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--source_dir", required=True, help="Directory containing images to import.")
    parser.add_argument("--annotations", default=None, help="Path to annotations.csv")
    parser.add_argument("--imports", default=None, help="Path to imports.csv")
    parser.add_argument("--import_id", default=None, help="Existing or explicit import ID.")
    parser.add_argument("--import_name", default=None, help="Human readable name for this import batch.")
    parser.add_argument(
        "--label",
        default=None,
        help="If set, import every image under source_dir as this label. "
             "If omitted, source_dir/<class_dir> layout is expected.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to place imported images into dataset_root/imports.",
    )
    parser.add_argument(
        "--reviewed",
        default="true",
        help="Whether imported labels should start as reviewed=true.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")

    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations
        else dataset_root / "annotations.csv"
    )
    imports_path = (
        Path(args.imports).expanduser().resolve()
        if args.imports
        else dataset_root / "imports.csv"
    )

    dataset_root.mkdir(parents=True, exist_ok=True)

    import_id = args.import_id.strip() if args.import_id else _make_import_id(args.import_name)
    import_name = args.import_name.strip() if args.import_name else import_id
    reviewed = _as_bool(args.reviewed)
    import_created_at = datetime.now().isoformat(timespec="seconds")
    source_topic = "external_images"

    if args.label:
        tasks = _iter_single_label_images(source_dir, _normalize_label(args.label))
    else:
        tasks = _iter_class_dir_images(source_dir)

    if not tasks:
        raise RuntimeError(
            "No importable images found. Expected image files under source_dir "
            "or source_dir/<car|duct_tube|other>/..."
        )

    imported_rows = [
        _build_row(
            dataset_root=dataset_root,
            import_id=import_id,
            import_name=import_name,
            source_topic=source_topic,
            copy_mode=args.copy_mode,
            reviewed=reviewed,
            image_path=image_path,
            label=label,
            label_root=label_root,
        )
        for image_path, label, label_root in tasks
    ]
    _write_sequence_manifests(dataset_root, imported_rows)

    merged_rows = _merge_annotations(_load_existing_annotations(annotations_path), imported_rows)
    _write_annotations(annotations_path, merged_rows)

    existing_imports = _load_existing_imports(imports_path)
    existing_imports.setdefault(
        import_id,
        {
            "import_id": import_id,
            "import_name": import_name,
            "created_at": import_created_at,
            "source_topic": source_topic,
            "num_sequences": "0",
            "num_samples": "0",
            "notes": "",
        },
    )
    _write_imports(
        imports_path,
        _build_import_summaries(
            merged_rows,
            existing_imports=existing_imports,
            default_created_at=import_created_at,
        ),
    )

    label_counts: Dict[str, int] = {label: 0 for label in LABELS}
    for row in imported_rows:
        label_counts[row["label"]] += 1

    print(f"[DONE] import_id={import_id} import_name={import_name}")
    print(
        "[DONE] imported="
        f"{len(imported_rows)} "
        f"rc_car={label_counts['rc_car']} "
        f"duct_tube={label_counts['duct_tube']} "
        f"background={label_counts['background']}"
    )


if __name__ == "__main__":
    main()
