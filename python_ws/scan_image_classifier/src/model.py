from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X0_5_Weights,
    mobilenet_v3_small,
    shufflenet_v2_x0_5,
)


class TinyScanClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_classifier_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
):
    architecture = architecture.strip().lower()

    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        if hasattr(model.classifier[0], "p"):
            model.classifier[0].p = float(dropout)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        model = shufflenet_v2_x0_5(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if architecture == "tiny_cnn":
        return TinyScanClassifier(num_classes=num_classes)

    raise ValueError(f"Unsupported architecture: {architecture}")


def set_backbone_trainable(model, architecture: str, trainable: bool) -> None:
    architecture = architecture.strip().lower()

    if architecture == "mobilenet_v3_small":
        for parameter in model.features.parameters():
            parameter.requires_grad = trainable
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if architecture == "shufflenet_v2_x0_5":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable if not name.startswith("fc.") else True
        return

    for parameter in model.parameters():
        parameter.requires_grad = True


def infer_architecture_from_checkpoint(checkpoint: dict) -> str:
    architecture = checkpoint.get("model_architecture")
    if architecture:
        return str(architecture)
    raise KeyError("Checkpoint does not contain 'model_architecture'.")
