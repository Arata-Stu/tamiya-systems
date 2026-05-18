from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.model import create_classifier_model, infer_architecture_from_checkpoint


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean, dtype=torch.float32).view(1, len(mean), 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, len(std), 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.clamp(x, 0.0, 255.0) / 255.0
        x = (x - self.mean) / self.std
        return self.model(x)


def main(args) -> None:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    )

    print("--- Configuration ---")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Output ONNX Path: {output_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint must contain 'model_state_dict'.")

    architecture = infer_architecture_from_checkpoint(checkpoint)
    labels = checkpoint.get("labels", ["rc_car", "duct_tube", "background"])
    num_classes = len(labels)
    channels = int(checkpoint.get("input_channels", args.channels))
    height = int(checkpoint.get("image_height", args.height))
    width = int(checkpoint.get("image_width", args.width))
    mean = checkpoint.get("pixel_mean", args.mean)
    std = checkpoint.get("pixel_std", args.std)

    print(f"Architecture: {architecture}")
    print(f"Labels: {labels}")
    print(f"Input Shape: (1, {channels}, {height}, {width})")
    print(f"Input Normalization: {args.input_normalization}")
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = create_classifier_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
        dropout=args.dropout,
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    if args.input_normalization == "internal":
        model = NormalizedModel(base_model, mean=mean, std=std)
        dummy_input = torch.randint(0, 255, (1, channels, height, width), dtype=torch.float32)
    else:
        model = base_model
        dummy_input = torch.randn(1, channels, height, width, dtype=torch.float32)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_tensor"],
        output_names=["output_logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            "input_tensor": {0: "batch_size"},
            "output_logits": {0: "batch_size"},
        },
    )
    print(f"ONNX export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export scan image classifier to ONNX format.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mean", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--std", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument(
        "--input_normalization",
        type=str,
        choices=["external", "internal"],
        default="external",
        help="external: input already normalized by isaac_ros_dnn_image_encoder.",
    )
    main(parser.parse_args())
