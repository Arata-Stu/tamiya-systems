from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.constants import ANNOTATION_COLUMNS


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ANNOTATION_COLUMNS})


def group_value(row: Dict[str, str], group_by: str) -> str:
    if group_by == "import_sequence_id":
        return f"{row.get('import_id', '').strip()}::{row.get('sequence_id', '').strip()}"
    return row[group_by]


def assign_group_splits(
    rows: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_by: str,
    reassign_all: bool,
) -> List[Dict[str, str]]:
    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        group_key = group_value(row, group_by)
        groups.setdefault(group_key, []).append(index)

    grouped_items: List[Tuple[str, List[int]]] = list(groups.items())
    random.Random(seed).shuffle(grouped_items)

    total_samples = len(rows)
    targets = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * test_ratio,
    }
    counts = {"train": 0, "val": 0, "test": 0}

    if not reassign_all:
        for _, indices in grouped_items:
            assigned_splits = {
                rows[index].get("split", "").strip()
                for index in indices
                if rows[index].get("split", "").strip()
            }
            if len(assigned_splits) == 1:
                assigned_split = next(iter(assigned_splits))
                counts[assigned_split] += len(indices)

    for group_key, indices in grouped_items:
        if not reassign_all:
            assigned_splits = {
                rows[index].get("split", "").strip()
                for index in indices
                if rows[index].get("split", "").strip()
            }
            if len(assigned_splits) == 1:
                assigned_split = next(iter(assigned_splits))
                for index in indices:
                    rows[index]["split"] = assigned_split
                continue

        size = len(indices)
        deficits = {
            split: targets[split] - counts[split]
            for split in ("train", "val", "test")
            if targets[split] > 0.0
        }

        if deficits:
            split = max(deficits, key=deficits.get)
        else:
            split = "train"

        for index in indices:
            rows[index]["split"] = split
        counts[split] += size

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign train/val/test splits grouped by sequence.")
    parser.add_argument("--dataset_root", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--annotations", default=None, help="Path to annotations.csv")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", default="bag_path", choices=["sequence_id", "bag_path", "import_id", "import_sequence_id"])
    parser.add_argument("--reassign_all", action="store_true", help="Reassign existing split rows instead of preserving them.")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations
        else dataset_root / "annotations.csv"
    )

    rows = load_rows(annotations_path)
    if not rows:
        raise RuntimeError(f"No rows found in {annotations_path}")

    rows = assign_group_splits(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_by=args.group_by,
        reassign_all=args.reassign_all,
    )
    save_rows(annotations_path, rows)
    print(f"[DONE] Updated split assignments in {annotations_path}")


if __name__ == "__main__":
    main()
