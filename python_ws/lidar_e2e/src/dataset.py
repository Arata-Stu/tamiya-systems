import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np


# =========================================================
# 1. SequenceDataset: LiDAR Scan + Steer + Acceleration
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, seq_dir: str, transform=None, seq_len: int = 10):
        """
        単一のシーケンスディレクトリを読み込む
        """
        self.seq_dir = Path(seq_dir)
        self.transform = transform
        self.seq_len = seq_len

        # --- 各データのパスを確認 ---
        self.scan_file = self.seq_dir / "scans.npy"
        self.steer_file = self.seq_dir / "steers.npy"
        self.speed_file = self.seq_dir / "speeds.npy" 

        # --- 必須ファイルの存在チェック ---
        for f in [self.scan_file, self.steer_file, self.speed_file]:
            if not f.exists():
                raise FileNotFoundError(f"{f.name} not found in {self.seq_dir}")

        # --- データを一括読み込み (メモリ上に保持) ---
        self.scans = np.load(self.scan_file).astype(np.float32)
        self.steers = np.load(self.steer_file).astype(np.float32)
        self.speeds = np.load(self.speed_file).astype(np.float32) 

        # --- アサーション ---
        assert len(self.scans) == len(self.steers) == len(self.speeds), \
            f"Data length mismatch in {seq_dir}: " \
            f"Scans({len(self.scans)}), Steers({len(self.steers)}), Speed({len(self.speeds)})"

    def __len__(self):
        num_frames = len(self.scans)
        if num_frames < self.seq_len:
            return 0
        return num_frames - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # --- 指定した長さ(seq_len)のデータをスライス ---
        scan_seq = self.scans[idx : idx + self.seq_len]
        steer_seq = self.steers[idx : idx + self.seq_len]
        speed_seq = self.speeds[idx : idx + self.seq_len] 

        # PyTorch Tensorに変換
        sample = {
            'scan': torch.from_numpy(scan_seq),
            'steer': torch.from_numpy(steer_seq),
            'speed': torch.from_numpy(speed_seq) 
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# =========================================================
# 2. MultiSequenceDataset: 複数シーケンスの統合
# =========================================================
class MultiSequenceDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform=None,
        seq_len: int = 10,
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

        self.datasets: List[SequenceDataset] = []
        for d in self.seq_dirs:
            try:
                self.datasets.append(SequenceDataset(d, transform=self.transform, seq_len=self.seq_len))
            except Exception as e:
                print(f"[MultiSequenceDataset WARN] Skipping {d}: {e}")

        self.concat_dataset = ConcatDataset(self.datasets)

    def _find_sequence_dirs(self, base_dir: Path) -> List[Path]:
        """
        再帰的に探索し、'scans.npy' と 'speeds.npy' があるディレクトリを sequence と認定
        """
        seq_dirs = []
        # 速度ファイルを基準に探索
        for speed_file in base_dir.rglob('speeds.npy'):
            path = speed_file.parent
            if (path / 'scans.npy').exists() and (path / 'steers.npy').exists():
                seq_dirs.append(path)
        
        seq_dirs.sort()
        return seq_dirs

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]