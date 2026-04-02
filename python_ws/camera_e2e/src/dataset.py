from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class ImageSequenceDataset(Dataset):
    def __init__(self, seq_dir: str, transform=None, seq_len: int = 1):
        self.seq_dir = Path(seq_dir)
        self.transform = transform
        self.seq_len = seq_len

        self.steer_file = self.seq_dir / "steers.npy"
        self.speed_file = self.seq_dir / "speeds.npy"
        self.images_npy = self.seq_dir / "images.npy"
        self.images_dir = self.seq_dir / "images"

        for f in (self.steer_file, self.speed_file):
            if not f.exists():
                raise FileNotFoundError(f"{f.name} not found in {self.seq_dir}")

        self.steers = np.load(self.steer_file).astype(np.float32)
        self.speeds = np.load(self.speed_file).astype(np.float32)

        self.image_mode = None
        self.images = None
        self.image_paths: List[Path] = []

        if self.images_npy.exists():
            self.image_mode = "npy"
            self.images = np.load(self.images_npy, mmap_mode="r")
            if self.images.ndim != 4 or self.images.shape[-1] != 3:
                raise ValueError(
                    f"Expected images.npy shape (N,H,W,3), got {self.images.shape} in {self.seq_dir}"
                )
            self.num_frames = int(self.images.shape[0])
        elif self.images_dir.exists():
            self.image_mode = "png"
            self.image_paths = sorted(self.images_dir.glob("*.png"))
            self.num_frames = len(self.image_paths)
            if self.num_frames == 0:
                raise RuntimeError(f"No png files found in {self.images_dir}")
        else:
            raise FileNotFoundError(f"Neither images.npy nor images/*.png found in {self.seq_dir}")

        if not (self.num_frames == len(self.steers) == len(self.speeds)):
            raise ValueError(
                f"Data length mismatch in {self.seq_dir}: "
                f"images={self.num_frames}, steers={len(self.steers)}, speeds={len(self.speeds)}"
            )

    def __len__(self) -> int:
        if self.num_frames < self.seq_len:
            return 0
        return self.num_frames - self.seq_len + 1

    def _load_frame(self, idx: int) -> np.ndarray:
        if self.image_mode == "npy":
            return np.array(self.images[idx], dtype=np.uint8)

        image_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb.astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_seq = [self._load_frame(i) for i in range(idx, idx + self.seq_len)]
        image_seq = np.stack(image_seq, axis=0)  # (S, H, W, C)

        steer_seq = self.steers[idx : idx + self.seq_len]
        speed_seq = self.speeds[idx : idx + self.seq_len]

        sample = {
            "image": torch.from_numpy(image_seq),  # uint8
            "steer": torch.from_numpy(steer_seq),
            "speed": torch.from_numpy(speed_seq),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MultiImageSequenceDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform=None,
        seq_len: int = 1,
        select_sequences: Optional[Union[List[int], range]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.seq_len = seq_len

        self.seq_dirs = self._find_sequence_dirs(self.base_dir)
        if not self.seq_dirs:
            raise RuntimeError(f"No valid sequence directories found under {base_dir}")

        if select_sequences is not None:
            selected_indices = list(select_sequences)
            self.seq_dirs = [self.seq_dirs[i] for i in selected_indices if i < len(self.seq_dirs)]

        self.datasets: List[ImageSequenceDataset] = []
        for d in self.seq_dirs:
            try:
                self.datasets.append(ImageSequenceDataset(d, transform=self.transform, seq_len=self.seq_len))
            except Exception as e:
                print(f"[MultiImageSequenceDataset WARN] Skipping {d}: {e}")

        if not self.datasets:
            raise RuntimeError(f"All sequence directories were skipped under {base_dir}")

        self.concat_dataset = ConcatDataset(self.datasets)

    def _find_sequence_dirs(self, base_dir: Path) -> List[Path]:
        seq_dirs = []
        for steer_file in base_dir.rglob("steers.npy"):
            path = steer_file.parent
            has_npy = (path / "images.npy").exists()
            has_png = (path / "images").is_dir()
            if has_npy or has_png:
                if (path / "speeds.npy").exists():
                    seq_dirs.append(path)
        seq_dirs.sort()
        return seq_dirs

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

