from typing import List, Sequence

import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class CropImage:
    """画像の上下を比率指定でクロップする。"""

    def __init__(self, top_ratio: float = 0.35, bottom_ratio: float = 1.0):
        self.top_ratio = max(0.0, min(1.0, top_ratio))
        self.bottom_ratio = max(0.0, min(1.0, bottom_ratio))

    def __call__(self, sample):
        image = sample["image"]  # (S,H,W,C) or (S,C,H,W)

        if image.dim() != 4:
            return sample

        if image.shape[-1] in (1, 3, 4):  # NHWC
            h = image.shape[1]
            top = int(h * self.top_ratio)
            bottom = int(h * self.bottom_ratio)
            if bottom <= top:
                bottom = h
                top = 0
            sample["image"] = image[:, top:bottom, :, :]
            return sample

        # NCHW
        h = image.shape[2]
        top = int(h * self.top_ratio)
        bottom = int(h * self.bottom_ratio)
        if bottom <= top:
            bottom = h
            top = 0
        sample["image"] = image[:, :, top:bottom, :]
        return sample


class ResizeImage:
    """画像を (height, width) にリサイズする。"""

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def __call__(self, sample):
        image = sample["image"]  # (S,H,W,C) or (S,C,H,W)
        if image.dim() != 4:
            return sample

        is_nhwc = image.shape[-1] in (1, 3, 4)
        current_h = image.shape[1] if is_nhwc else image.shape[2]
        current_w = image.shape[2] if is_nhwc else image.shape[3]
        if current_h == self.height and current_w == self.width:
            return sample

        if is_nhwc:
            image = image.permute(0, 3, 1, 2)

        image = image.to(torch.float32)
        image = F.interpolate(image, size=(self.height, self.width), mode="bilinear", align_corners=False)

        if is_nhwc:
            image = image.permute(0, 2, 3, 1)

        sample["image"] = image
        return sample


class ConvertToGray3Channel:
    """RGB/mono入力を 3ch grayscale に正規化する。"""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def __call__(self, sample):
        if not self.enabled:
            return sample

        image = sample["image"]  # (S,H,W,C) or (S,C,H,W)
        if image.dim() != 4:
            return sample

        if image.shape[-1] in (1, 3, 4):  # NHWC
            if image.shape[-1] == 1:
                image = image.repeat(1, 1, 1, 3)
            else:
                rgb = image[..., :3].to(torch.float32)
                gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
                image = gray.unsqueeze(-1).repeat(1, 1, 1, 3)
            sample["image"] = image.to(sample["image"].dtype)
            return sample

        # NCHW
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        else:
            rgb = image[:, :3, :, :].to(torch.float32)
            gray = 0.2989 * rgb[:, 0, :, :] + 0.5870 * rgb[:, 1, :, :] + 0.1140 * rgb[:, 2, :, :]
            image = gray.unsqueeze(1).repeat(1, 3, 1, 1)
        sample["image"] = image.to(sample["image"].dtype)
        return sample


class NormalizeImage:
    """uint8画像を [0,1] に正規化し、NCHW に変換して標準化する。"""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, len(mean), 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, len(std), 1, 1)

    def __call__(self, sample):
        image = sample["image"]  # (S,H,W,C) or (S,C,H,W)
        if image.dim() != 4:
            return sample

        if image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)

        if image.shape[1] == 4:
            image = image[:, :3, :, :]

        image = image.to(torch.float32)
        if image.max() > 1.0:
            image = image / 255.0

        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        sample["image"] = (image - mean) / std
        return sample


class AddImageNoise:
    """正規化済み画像に小さなガウシアンノイズを付与する。"""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        if image.dim() != 4:
            return sample
        noise = torch.randn_like(image) * self.std
        sample["image"] = image + noise
        return sample
