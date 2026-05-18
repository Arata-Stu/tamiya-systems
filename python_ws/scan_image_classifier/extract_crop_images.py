from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from src.constants import ANNOTATION_COLUMNS, DEFAULT_IMAGE_TOPIC, IMPORT_COLUMNS


def _decode_raw_image_to_rgb(msg) -> Optional[np.ndarray]:
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = msg.encoding.lower()
    step = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if step <= 0 or raw.size < height * step:
        return None

    raw = raw[: height * step].reshape(height, step)

    if encoding == "rgb8":
        return raw[:, : width * 3].reshape(height, width, 3).copy()
    if encoding in ("bgr8", "8uc3"):
        bgr = raw[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if encoding in ("mono8", "8uc1"):
        mono = raw[:, :width].reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    if encoding in ("rgba8",):
        rgba = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    if encoding in ("bgra8",):
        bgra = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

    return None


def _decode_compressed_image_to_rgb(msg) -> Optional[np.ndarray]:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stamp_to_ns(msg, fallback_timestamp: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_timestamp)
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _make_sequence_id(bag_path: Path) -> str:
    digest = hashlib.sha1(str(bag_path).encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", bag_path.name).strip("_")
    slug = slug or "rosbag"
    return f"{slug}_{digest}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "import"


def _make_import_id(import_name: Optional[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if import_name:
        return f"{timestamp}_{_slugify(import_name)}"
    return timestamp


def _write_sequence_manifest(rows: Sequence[Dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def extract_single_bag(
    bag_path_str: str,
    output_root_str: str,
    image_topic: str,
    import_id: str,
    import_name: str,
    import_created_at: str,
) -> List[Dict[str, str]]:
    bag_path = Path(bag_path_str).expanduser().resolve()
    output_root = Path(output_root_str).expanduser().resolve()
    sequence_id = _make_sequence_id(bag_path)
    sequence_dir = output_root / "imports" / import_id / "raw" / sequence_id
    images_dir = sequence_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_rows: List[Dict[str, str]] = []

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == image_topic]
        for index, (conn, timestamp, raw) in enumerate(reader.messages(connections=connections)):
            msg = reader.deserialize(raw, conn.msgtype)
            image_rgb: Optional[np.ndarray] = None
            if conn.msgtype == "sensor_msgs/msg/Image":
                image_rgb = _decode_raw_image_to_rgb(msg)
            elif conn.msgtype == "sensor_msgs/msg/CompressedImage":
                image_rgb = _decode_compressed_image_to_rgb(msg)

            if image_rgb is None:
                continue

            file_name = f"{index:06d}.png"
            image_relpath = Path("imports") / import_id / "raw" / sequence_id / "images" / file_name
            image_abspath = output_root / image_relpath
            cv2.imwrite(str(image_abspath), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            stamp_ns = _stamp_to_ns(msg, timestamp)
            sample_id = f"{import_id}__{sequence_id}_{index:06d}"
            saved_rows.append(
                {
                    "sample_id": sample_id,
                    "import_id": import_id,
                    "import_name": import_name,
                    "sequence_id": sequence_id,
                    "bag_path": str(bag_path),
                    "image_path": image_relpath.as_posix(),
                    "stamp_ns": str(stamp_ns),
                    "width": str(image_rgb.shape[1]),
                    "height": str(image_rgb.shape[0]),
                    "source_topic": image_topic,
                    "label": "",
                    "label_id": "",
                    "split": "",
                    "reviewed": "false",
                    "annotated_at": "",
                    "auto_label": "",
                    "auto_confidence": "",
                    "notes": "",
                }
            )

    _write_sequence_manifest(saved_rows, sequence_dir / "samples.csv")
    print(f"[SAVE] {bag_path.name}: {len(saved_rows)} images -> {sequence_dir}")
    return saved_rows


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
    extracted_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged = dict(existing_rows)
    preserve_keys = {"label", "label_id", "split", "reviewed", "annotated_at", "auto_label", "auto_confidence", "notes"}

    for row in extracted_rows:
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


def _parse_bag_dirs(args) -> List[Path]:
    bag_dirs: List[Path] = []

    if args.bags_dir:
        bags_dir_path = Path(args.bags_dir).expanduser().resolve()
        for metadata_path in bags_dir_path.rglob("metadata.yaml"):
            bag_dirs.append(metadata_path.parent)
    else:
        for seq_dir_str in args.seq_dirs:
            seq_dir = Path(seq_dir_str).expanduser().resolve()
            if (seq_dir / "metadata.yaml").is_file():
                bag_dirs.append(seq_dir)

    deduped = []
    seen = set()
    for bag_dir in sorted(bag_dirs):
        if bag_dir in seen:
            continue
        seen.add(bag_dir)
        deduped.append(bag_dir)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract crop images from rosbag files and initialize annotations.csv."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bags_dir", help="Directory to search recursively for rosbag sequences.")
    group.add_argument("--seq_dirs", nargs="+", help="Explicit rosbag sequence directories.")
    parser.add_argument("--outdir", required=True, help="Dataset root output directory.")
    parser.add_argument("--image_topic", default=DEFAULT_IMAGE_TOPIC, help="Image topic to extract.")
    parser.add_argument("--import_id", default=None, help="Existing or explicit import ID.")
    parser.add_argument("--import_name", default=None, help="Human readable name for this extraction batch.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count.")
    args = parser.parse_args()

    bag_dirs = _parse_bag_dirs(args)
    if not bag_dirs:
        raise RuntimeError("No valid rosbag directories found.")

    output_root = Path(args.outdir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    annotations_path = output_root / "annotations.csv"
    imports_path = output_root / "imports.csv"

    import_id = args.import_id.strip() if args.import_id else _make_import_id(args.import_name)
    import_name = args.import_name.strip() if args.import_name else import_id
    import_created_at = datetime.now().isoformat(timespec="seconds")

    tasks: List[Tuple[str, str, str, str, str, str]] = [
        (
            str(path),
            str(output_root),
            args.image_topic,
            import_id,
            import_name,
            import_created_at,
        )
        for path in bag_dirs
    ]

    num_workers = args.workers if args.workers else min(max((os.cpu_count() or 1) - 1, 1), 8)
    if num_workers <= 1:
        extracted_batches = [extract_single_bag(*task) for task in tasks]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            extracted_batches = pool.starmap(extract_single_bag, tasks)

    extracted_rows = [row for batch in extracted_batches for row in batch]
    if not extracted_rows:
        raise RuntimeError("No images were extracted from the selected rosbag sequences.")

    merged_rows = _merge_annotations(_load_existing_annotations(annotations_path), extracted_rows)
    _write_annotations(annotations_path, merged_rows)
    existing_imports = _load_existing_imports(imports_path)
    existing_imports.setdefault(
        import_id,
        {
            "import_id": import_id,
            "import_name": import_name,
            "created_at": import_created_at,
            "source_topic": args.image_topic,
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
    print(f"[DONE] import_id={import_id} import_name={import_name}")
    print(f"[DONE] annotations.csv updated: {annotations_path}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
