from __future__ import annotations

import random
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        output = image
        for transform in self.transforms:
            output = transform(output)
        return output


class ResizeImage:
    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] == self.height and image.shape[1] == self.width:
            return image
        interpolation = cv2.INTER_AREA if image.shape[0] > self.height else cv2.INTER_LINEAR
        return cv2.resize(image, (self.width, self.height), interpolation=interpolation)


class ConvertToGray3Channel:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class RandomHorizontalFlip:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        return cv2.flip(image, 1)


class RandomRotate:
    def __init__(self, probability: float, max_degrees: float, border_value: int = 127):
        self.probability = float(probability)
        self.max_degrees = max(0.0, float(max_degrees))
        self.border_value = int(np.clip(border_value, 0, 255))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.max_degrees <= 0.0 or random.random() >= self.probability:
            return image
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        center = ((image.shape[1] - 1) * 0.5, (image.shape[0] - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_value = (self.border_value,) * 3 if image.ndim == 3 else self.border_value
        return cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )


class RandomTranslateScale:
    def __init__(
        self,
        probability: float,
        max_translate_ratio: float,
        min_scale: float,
        max_scale: float,
        border_value: int = 127,
    ):
        self.probability = float(probability)
        self.max_translate_ratio = max(0.0, float(max_translate_ratio))
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.border_value = int(np.clip(border_value, 0, 255))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image

        height, width = image.shape[:2]
        tx = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        ty = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        scale = random.uniform(self.min_scale, self.max_scale)

        center = ((width - 1) * 0.5, (height - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        border_value = (self.border_value,) * 3 if image.ndim == 3 else self.border_value
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )


class RandomBrightnessContrast:
    def __init__(self, brightness: float, contrast: float):
        self.brightness = max(0.0, float(brightness))
        self.contrast = max(0.0, float(contrast))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * random.uniform(-self.brightness, self.brightness)
        adjusted = image.astype(np.float32) * alpha + beta
        return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


class RandomGaussianBlur:
    def __init__(self, probability: float, kernel_size: int = 3):
        self.probability = float(probability)
        kernel_size = max(1, int(kernel_size))
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0.0)


class RandomGaussianNoise:
    def __init__(self, probability: float, sigma: float):
        self.probability = float(probability)
        self.sigma = max(0.0, float(sigma))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.sigma <= 0.0 or random.random() >= self.probability:
            return image
        noise = np.random.normal(0.0, self.sigma, size=image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


class RandomCutout:
    def __init__(self, probability: float, min_ratio: float, max_ratio: float, fill_value: int = 127):
        self.probability = float(probability)
        self.min_ratio = max(0.0, float(min_ratio))
        self.max_ratio = max(self.min_ratio, float(max_ratio))
        self.fill_value = int(np.clip(fill_value, 0, 255))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.max_ratio <= 0.0 or random.random() >= self.probability:
            return image

        height, width = image.shape[:2]
        cut_h = max(1, int(round(height * random.uniform(self.min_ratio, self.max_ratio))))
        cut_w = max(1, int(round(width * random.uniform(self.min_ratio, self.max_ratio))))
        y0 = random.randint(0, max(height - cut_h, 0))
        x0 = random.randint(0, max(width - cut_w, 0))

        output = image.copy()
        if output.ndim == 2:
            output[y0:y0 + cut_h, x0:x0 + cut_w] = self.fill_value
        else:
            output[y0:y0 + cut_h, x0:x0 + cut_w, :] = self.fill_value
        return output


class ToTensorNormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.float32)
        tensor = torch.clamp(tensor, 0.0, 255.0) / 255.0
        return (tensor - self.mean) / self.std
