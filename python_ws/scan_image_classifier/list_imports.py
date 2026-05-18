from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def filter_import_rows(
    rows: Sequence[Dict[str, str]],
    import_ids: Sequence[str],
) -> List[Dict[str, str]]:
    selected_import_ids = {value.strip() for value in import_ids if value.strip()}
    if not selected_import_ids:
        return list(rows)
    return [
        row for row in rows
        if row.get("import_id", "").strip() in selected_import_ids
    ]


def summarize_annotations(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for row in rows:
        import_id = row.get("import_id", "").strip()
        if not import_id:
            continue

        stats = summary.setdefault(
            import_id,
            {
                "samples": 0,
                "labeled": 0,
                "reviewed": 0,
                "train": 0,
                "val": 0,
                "test": 0,
            },
        )
        stats["samples"] += 1
        if row.get("label", "").strip():
            stats["labeled"] += 1
        if _as_bool(row.get("reviewed", "")):
            stats["reviewed"] += 1

        split = row.get("split", "").strip()
        if split in {"train", "val", "test"}:
            stats[split] += 1

    return summary


def synthesize_import_rows(annotation_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for row in annotation_rows:
        import_id = row.get("import_id", "").strip()
        if not import_id or import_id in grouped:
            continue
        grouped[import_id] = {
            "import_id": import_id,
            "import_name": row.get("import_name", "").strip() or import_id,
            "created_at": "",
            "source_topic": row.get("source_topic", "").strip(),
            "num_sequences": "",
            "num_samples": "",
            "notes": "",
        }
    return [grouped[key] for key in sorted(grouped.keys())]


def print_import_summary(
    import_rows: Sequence[Dict[str, str]],
    annotation_summary: Dict[str, Dict[str, int]],
) -> None:
    if not import_rows:
        print("No matching imports found.")
        return

    header = (
        "import_id | import_name | created_at | samples | labeled | reviewed | "
        "train/val/test | source_topic"
    )
    print(header)
    print("-" * len(header))

    for row in import_rows:
        import_id = row.get("import_id", "").strip()
        stats = annotation_summary.get(import_id, {})
        samples = stats.get("samples", int(row.get("num_samples", "0") or "0"))
        labeled = stats.get("labeled", 0)
        reviewed = stats.get("reviewed", 0)
        split_summary = f"{stats.get('train', 0)}/{stats.get('val', 0)}/{stats.get('test', 0)}"

        print(
            f"{import_id} | "
            f"{row.get('import_name', '').strip()} | "
            f"{row.get('created_at', '').strip()} | "
            f"{samples} | "
            f"{labeled} | "
            f"{reviewed} | "
            f"{split_summary} | "
            f"{row.get('source_topic', '').strip()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List scan image classifier dataset imports.")
    parser.add_argument("--dataset_root", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--annotations", default=None, help="Path to annotations.csv")
    parser.add_argument("--imports", default=None, help="Path to imports.csv")
    parser.add_argument("--import_ids", nargs="*", default=[], help="Limit output to specific import IDs.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
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

    annotation_rows = filter_import_rows(load_csv_rows(annotations_path), args.import_ids)
    import_rows = filter_import_rows(load_csv_rows(imports_path), args.import_ids)
    if not import_rows and annotation_rows:
        import_rows = synthesize_import_rows(annotation_rows)
    annotation_summary = summarize_annotations(annotation_rows)
    print_import_summary(import_rows, annotation_summary)


if __name__ == "__main__":
    main()
