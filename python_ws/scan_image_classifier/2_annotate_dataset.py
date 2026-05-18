from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2

from src.constants import ANNOTATION_COLUMNS, LABELS, LABEL_TO_ID


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ANNOTATION_COLUMNS})


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def render_overlay(image_bgr, row: Dict[str, str], index: int, total: int, scale: int) -> "cv2.typing.MatLike":
    scaled = cv2.resize(
        image_bgr,
        (image_bgr.shape[1] * scale, image_bgr.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = cv2.copyMakeBorder(scaled, 0, 220, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    label = row.get("label", "").strip() or "<unlabeled>"
    reviewed = row.get("reviewed", "").strip() or "false"
    lines = [
        f"[{index + 1}/{total}] {row['sample_id']}",
        f"import: {row.get('import_id', '')}",
        f"sequence: {row['sequence_id']}",
        f"label: {label} (reviewed={reviewed})",
        "",
        "1=rc_car  2=duct_tube  3=background",
        "a=prev  d=next  u=clear  j=next_unlabeled",
        "s=save  q=save_and_quit",
    ]

    y = scaled.shape[0] + 30
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 28

    return canvas


def find_first_unlabeled(rows: List[Dict[str, str]], start_index: int = 0) -> int:
    for idx in range(start_index, len(rows)):
        if not rows[idx].get("label", "").strip():
            return idx
    for idx in range(0, start_index):
        if not rows[idx].get("label", "").strip():
            return idx
    return start_index


def filter_row_indices(rows: List[Dict[str, str]], import_ids: List[str]) -> List[int]:
    selected_imports = {value.strip() for value in import_ids if value.strip()}
    if not selected_imports:
        return list(range(len(rows)))
    return [
        index for index, row in enumerate(rows)
        if row.get("import_id", "").strip() in selected_imports
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual annotation tool for scan image classifier dataset.")
    parser.add_argument("--dataset_root", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--annotations", default=None, help="Path to annotations.csv")
    parser.add_argument("--import_ids", nargs="*", default=[], help="Limit annotation to specific import IDs.")
    parser.add_argument("--window_name", default="scan_image_classifier_annotator")
    parser.add_argument("--display_scale", type=int, default=8, help="Display upscaling factor for 64x64 images.")
    parser.add_argument("--start_from_beginning", action="store_true", help="Start from the first sample instead of the first unlabeled sample.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations
        else dataset_root / "annotations.csv"
    )

    all_rows = load_rows(annotations_path)
    if not all_rows:
        raise RuntimeError(f"No rows found in {annotations_path}")

    row_indices = filter_row_indices(all_rows, args.import_ids)
    if not row_indices:
        raise RuntimeError("No rows matched the requested import_ids filter.")

    filtered_rows = [all_rows[index] for index in row_indices]
    current_index = 0 if args.start_from_beginning else find_first_unlabeled(filtered_rows, 0)
    cv2.namedWindow(args.window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        absolute_index = row_indices[current_index]
        row = all_rows[absolute_index]
        image_path = dataset_root / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        frame = render_overlay(image, row, current_index, len(row_indices), max(1, args.display_scale))
        cv2.imshow(args.window_name, frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("1"), ord("2"), ord("3")):
            label = LABELS[key - ord("1")]
            row["label"] = label
            row["label_id"] = str(LABEL_TO_ID[label])
            row["reviewed"] = "true"
            row["annotated_at"] = datetime.now().isoformat(timespec="seconds")
            current_index = min(current_index + 1, len(row_indices) - 1)
            continue

        if key == ord("u"):
            row["label"] = ""
            row["label_id"] = ""
            row["reviewed"] = "false"
            row["annotated_at"] = ""
            continue

        if key == ord("a"):
            current_index = max(0, current_index - 1)
            continue

        if key == ord("d"):
            current_index = min(len(row_indices) - 1, current_index + 1)
            continue

        if key == ord("j"):
            filtered_rows = [all_rows[index] for index in row_indices]
            current_index = find_first_unlabeled(filtered_rows, current_index + 1)
            continue

        if key == ord("s"):
            save_rows(annotations_path, all_rows)
            print(f"[SAVE] {annotations_path}")
            continue

        if key == ord("q"):
            save_rows(annotations_path, all_rows)
            print(f"[SAVE] {annotations_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
