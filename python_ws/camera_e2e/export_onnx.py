import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.model import PilotNetControl


class NormalizedModel(nn.Module):
    """入力画像を正規化してからモデルに入力するラッパー。"""

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean, dtype=torch.float32).view(1, len(mean), 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, len(std), 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # x: [B, C, H, W], float想定 (0-255)
        x = x.to(torch.float32)
        x = torch.clamp(x, 0.0, 255.0) / 255.0
        x = (x - self.mean) / self.std
        return self.model(x)


def main(args):
    checkpoint_path = Path(args.checkpoint).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"

    print("--- Configuration ---")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Output ONNX Path: {output_path}")
    print(f"Input Shape: (1, {args.channels}, {args.height}, {args.width})")
    print(f"Input Normalization: {args.input_normalization}")
    print(f"Mean: {args.mean}")
    print(f"Std: {args.std}")
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = PilotNetControl(
        num_outputs=args.num_outputs,
        input_channels=args.channels,
        input_height=args.height,
        input_width=args.width,
    )

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        base_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        base_model.load_state_dict(checkpoint)

    base_model.eval()
    if args.input_normalization == "internal":
        model = NormalizedModel(base_model, mean=args.mean, std=args.std)
        dummy_input = torch.randint(
            low=0,
            high=255,
            size=(1, args.channels, args.height, args.width),
            dtype=torch.float32,
        )
    else:
        model = base_model
        dummy_input = torch.randn(1, args.channels, args.height, args.width, dtype=torch.float32)
    model.eval()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image_input"],
            output_names=["control_output"],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={
                "image_input": {0: "batch_size"},
                "control_output": {0: "batch_size"},
            },
        )
        print(f"ONNX export complete: {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PilotNetControl to ONNX format.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_outputs", type=int, default=2)
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Per-channel mean used in training",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Per-channel std used in training",
    )
    parser.add_argument(
        "--input_normalization",
        type=str,
        choices=["external", "internal"],
        default="external",
        help=(
            "external: input is already normalized (recommended with isaac_ros_dnn_image_encoder). "
            "internal: model includes [0,255]->normalize step."
        ),
    )
    args = parser.parse_args()
    main(args)
