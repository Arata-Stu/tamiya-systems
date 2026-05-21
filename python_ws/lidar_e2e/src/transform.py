import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class NormalizeScan:
    """Normalize a single LiDAR scan into the [0, 1] range."""

    def __init__(self, max_range: float):
        self.max_range = max_range

    def __call__(self, sample):
        scan = sample["scan"]
        scan = torch.nan_to_num(
            scan,
            nan=self.max_range,
            posinf=self.max_range,
            neginf=0.0,
        )
        scan = torch.clamp(scan, 0.0, self.max_range)
        sample["scan"] = scan / self.max_range
        return sample


class AddScanNoise:
    """Add independent noise to each LiDAR beam after normalization."""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        noise = torch.randn_like(sample["scan"]) * self.std
        sample["scan"] = torch.clamp(sample["scan"] + noise, 0.0, 1.0)
        return sample


class AddChannelDim:
    """Convert a single scan from [points] to [1, points] for Conv1d."""

    def __call__(self, sample):
        sample["scan"] = sample["scan"].unsqueeze(0)
        return sample


# Backward-compatible alias for older training configs.
AddTemporalNoise = AddScanNoise
