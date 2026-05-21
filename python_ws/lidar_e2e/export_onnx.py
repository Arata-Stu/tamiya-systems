import argparse
from pathlib import Path

import torch

from src.model import TinyLidarNet


def main(args):
    checkpoint_path = Path(args.checkpoint).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"

    print("--- Configuration ---")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Output ONNX Path: {output_path}")
    print(f"Scan Points: {args.scan_points}")
    print("Input Domain: normalized scan tensor in [0, 1]")
    print(f"Input Shape: (1, 1, {args.scan_points})")
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = TinyLidarNet(input_dim=args.scan_points, output_dim=2)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.rand(1, 1, args.scan_points, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["scan_input"],
            output_names=["control_output"],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={
                "scan_input": {0: "batch_size"},
                "control_output": {0: "batch_size"},
            },
        )
        print(f"ONNX export complete: {output_path}")
    except Exception as exc:
        print(f"Error during ONNX export: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TinyLidarNet to ONNX format.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-p", "--scan_points", type=int, default=320)
    parser.add_argument(
        "--max_range",
        type=float,
        default=12.0,
        help="Compatibility flag only. Inputs are expected to be normalized before the model.",
    )
    main(parser.parse_args())
