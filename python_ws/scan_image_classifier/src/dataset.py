from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import torch
from torch.utils.data import Dataset

from src.constants import LABEL_TO_ID


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_annotation_rows(
    annotations_path: Path,
    split: str,
    labels: Sequence[str],
    require_reviewed: bool = True,
    import_ids: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    valid_labels = set(labels)
    selected_import_ids = {value.strip() for value in import_ids or [] if value.strip()}

    with annotations_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("split", "").strip() != split:
                continue
            if selected_import_ids and row.get("import_id", "").strip() not in selected_import_ids:
                continue
            label = row.get("label", "").strip()
            if label not in valid_labels:
                continue
            if require_reviewed and not _as_bool(row.get("reviewed", "")):
                continue
            rows.append(row)

    return rows


class ScanImageClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        annotations_path: str,
        split: str,
        labels: Sequence[str],
        transform,
        require_reviewed: bool = True,
        import_ids: Sequence[str] | None = None,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.labels = list(labels)
        self.transform = transform
        self.rows = load_annotation_rows(
            self.annotations_path,
            split=split,
            labels=self.labels,
            require_reviewed=require_reviewed,
            import_ids=import_ids,
        )

        if not self.rows:
            raise RuntimeError(
                f"No labeled samples found for split='{split}' in {self.annotations_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.dataset_root / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = self.transform(image)
        label_name = row["label"].strip()
        label_id = LABEL_TO_ID[label_name]

        return {
            "image": image_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
            "sample_id": row["sample_id"],
            "import_id": row.get("import_id", ""),
            "sequence_id": row["sequence_id"],
            "image_path": row["image_path"],
        }

    def class_counts(self) -> Dict[str, int]:
        counts = {label: 0 for label in self.labels}
        for row in self.rows:
            counts[row["label"]] += 1
        return counts

    def import_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.rows:
            import_id = row.get("import_id", "").strip() or "<unknown>"
            counts[import_id] = counts.get(import_id, 0) + 1
        return counts
