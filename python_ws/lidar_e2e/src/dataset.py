from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class ScanDataset(Dataset):
    def __init__(self, seq_dir: str, transform=None, use_virtual_scan: bool = False):
        self.seq_dir = Path(seq_dir)
        self.transform = transform

        scan_name = "virtual_scans.npy" if use_virtual_scan else "scans.npy"
        self.scan_file = self.seq_dir / scan_name
        self.steer_file = self.seq_dir / "steers.npy"
        self.speed_file = self.seq_dir / "speeds.npy"

        for path in [self.scan_file, self.steer_file, self.speed_file]:
            if not path.exists():
                raise FileNotFoundError(f"{path.name} not found in {self.seq_dir}")

        self.scans = np.load(self.scan_file).astype(np.float32)
        self.steers = np.load(self.steer_file).astype(np.float32)
        self.speeds = np.load(self.speed_file).astype(np.float32)

        if not (
            len(self.scans) == len(self.steers) == len(self.speeds)
        ):
            raise RuntimeError(
                f"Data length mismatch in {seq_dir}: "
                f"Scans({len(self.scans)}), Steers({len(self.steers)}), "
                f"Speeds({len(self.speeds)})"
            )

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "scan": torch.from_numpy(self.scans[idx]),
            "steer": torch.tensor(self.steers[idx], dtype=torch.float32),
            "speed": torch.tensor(self.speeds[idx], dtype=torch.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MultiScanDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform=None,
        select_sequences: Optional[Union[List[int], range]] = None,
        use_virtual_scan: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.transform = transform

        self.seq_dirs = self._find_sequence_dirs(self.base_dir, use_virtual_scan=use_virtual_scan)
        if not self.seq_dirs:
            raise RuntimeError(f"No valid sequence directories found under {base_dir}")

        if select_sequences is not None:
            selected_indices = list(select_sequences)
            self.seq_dirs = [
                self.seq_dirs[i] for i in selected_indices if i < len(self.seq_dirs)
            ]

        self.datasets: List[ScanDataset] = []
        for seq_dir in self.seq_dirs:
            try:
                self.datasets.append(ScanDataset(seq_dir, transform=self.transform, use_virtual_scan=use_virtual_scan))
            except Exception as exc:
                print(f"[MultiScanDataset WARN] Skipping {seq_dir}: {exc}")

        if not self.datasets:
            raise RuntimeError(f"No readable sequence directories found under {base_dir}")

        self.concat_dataset = ConcatDataset(self.datasets)

    def _find_sequence_dirs(self, base_dir: Path, use_virtual_scan: bool = False) -> List[Path]:
        seq_dirs = []
        scan_name = "virtual_scans.npy" if use_virtual_scan else "scans.npy"
        for speed_file in base_dir.rglob("speeds.npy"):
            path = speed_file.parent
            if (path / scan_name).exists() and (path / "steers.npy").exists():
                seq_dirs.append(path)

        seq_dirs.sort()
        return seq_dirs

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.concat_dataset[idx]


# Backward-compatible aliases for old imports.
SequenceDataset = ScanDataset
MultiSequenceDataset = MultiScanDataset
