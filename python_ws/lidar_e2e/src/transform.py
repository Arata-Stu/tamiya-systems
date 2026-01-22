import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class NormalizeScan:
    """LiDARスキャンの正規化 (seq_len, pts) 全体に適用"""
    def __init__(self, max_range: float):
        self.max_range = max_range

    def __call__(self, sample):
        # Dataset側ですでにTensor化されている前提
        scan = sample['scan']
        # 無効値処理とクリッピング (Tensor版)
        scan = torch.nan_to_num(scan, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        scan = torch.clamp(scan, 0.0, self.max_range)
        
        sample['scan'] = scan / self.max_range
        return sample

class AddTemporalNoise:
    """時系列の各フレームに独立したノイズを付与"""
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        # scanの形状 (seq_len, pts) に合わせたノイズを作成
        noise = torch.randn_like(sample['scan']) * self.std
        sample['scan'] = torch.clamp(sample['scan'] + noise, 0.0, 1.0)
        return sample

