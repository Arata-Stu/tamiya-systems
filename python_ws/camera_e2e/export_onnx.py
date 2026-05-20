import argparse
from pathlib import Path

import torch

from src.model import PilotNetControl


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
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PilotNetControl(
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
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Isaac ROSのエンコーダから送られてくる「正規化済み」のテンソルを想定
    dummy_input = torch.randn(1, args.channels, args.height, args.width, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image_input"],
            output_names=["control_output"],
            opset_version=12,
            do_constant_folding=True,
            # バッチサイズ(0次元目)の動的化は維持。
            # Tritonで将来的に複数枚のバッチ処理を行う可能性も考慮して残しています。
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
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=212)
    parser.add_argument("--num_outputs", type=int, default=2)
    parser.add_argument(
        "--input_normalization",
        type=str,
        choices=["external"],
        default="external",
        help=(
            "Compatibility flag for existing README/deploy commands. "
            "Normalization is handled by isaac_ros_dnn_image_encoder, so only 'external' is supported."
        ),
    )
    args = parser.parse_args()
    main(args)
