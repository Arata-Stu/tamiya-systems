import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np

# =========================================================
# 1. SequenceDataset: 2D Scan + Steer + Accel (加速度)
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, seq_dir: str, transform=None, seq_len: int = 10):
        self.seq_dir = Path(seq_dir)
        self.transform = transform
        self.seq_len = seq_len

        # --- 各データのパスを確認 ---
        self.scan_file = self.seq_dir / "scans.npy"
        self.steer_file = self.seq_dir / "steers.npy"
        self.accel_file = self.seq_dir / "accelerations.npy" 

        # --- 必須ファイルの存在チェック ---
        for f in [self.scan_file, self.steer_file, self.accel_file]:
            if not f.exists():
                raise FileNotFoundError(f"{f.name} not found in {self.seq_dir}")

        # --- データを一括読み込み ---
        self.scans = np.load(self.scan_file).astype(np.float32)
        self.steers = np.load(self.steer_file).astype(np.float32)
        self.accels = np.load(self.accel_file).astype(np.float32) # 加速度

        # --- アサーション ---
        assert len(self.scans) == len(self.steers) == len(self.accels), \
            f"Data length mismatch in {seq_dir}"

    def __len__(self):
        num_frames = len(self.scans)
        if num_frames < self.seq_len:
            return 0
        return num_frames - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # スライスによるシーケンス抽出
        scan_seq = self.scans[idx : idx + self.seq_len]
        steer_seq = self.steers[idx : idx + self.seq_len]
        accel_seq = self.accels[idx : idx + self.seq_len]

        sample = {
            'scan': torch.from_numpy(scan_seq),
            'steer': torch.from_numpy(steer_seq),
            'accel': torch.from_numpy(accel_seq) # キー名をaccelに変更
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# =========================================================
# 2. MultiSequenceDataset (再帰探索版)
# =========================================================
class MultiSequenceDataset(Dataset):
    def __init__(self, base_dir: str, transform=None, seq_len: int = 10, select_sequences=None):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.seq_len = seq_len

        self.seq_dirs = self._find_sequence_dirs(self.base_dir)
        if select_sequences is not None:
            self.seq_dirs = [self.seq_dirs[i] for i in select_sequences if i < len(self.seq_dirs)]

        self.datasets = [SequenceDataset(d, transform=self.transform, seq_len=self.seq_len) for d in self.seq_dirs]
        self.concat_dataset = ConcatDataset(self.datasets)

    def _find_sequence_dirs(self, base_dir: Path) -> List[Path]:
        seq_dirs = [p.parent for p in base_dir.rglob('accelerations.npy') 
                    if (p.parent / 'scans.npy').exists()]
        seq_dirs.sort()
        return seq_dirs

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]