import argparse
from pathlib import Path

import torch

from src.model import create_trajectory_model, infer_trajectory_architecture


def main(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve() if args.output else checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"

    print("--- Configuration ---")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Output ONNX Path: {output_path}")
    print(f"Input Shape: (1, {args.channels}, {args.height}, {args.width})")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # チェックポイントの読み込みとメタデータの抽出
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        architecture = checkpoint.get("model_architecture") or infer_trajectory_architecture(state_dict, args.num_points)
        num_points = int(checkpoint.get("num_points", args.num_points))
        output_scale = float(checkpoint.get("output_scale", args.output_scale))
    else:
        state_dict = checkpoint
        architecture = infer_trajectory_architecture(state_dict, args.num_points)
        num_points = args.num_points
        output_scale = args.output_scale

    print(f"Model Architecture: {architecture}")
    print(f"Output Shape: (1, {num_points}, 2)")
    print(f"Output Scale: {output_scale}")
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # モデルの生成と重みのロード
    model = create_trajectory_model(
        architecture=architecture,
        num_points=num_points,
        input_channels=args.channels,
        input_height=args.height,
        input_width=args.width,
        output_scale=output_scale,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Isaac ROSのエンコーダから送られてくる「正規化済み」のテンソルを想定したダミー入力
    dummy_input = torch.randn(1, args.channels, args.height, args.width, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image_input"],
            output_names=["trajectory_output"],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={
                "image_input": {0: "batch_size"},
                "trajectory_output": {0: "batch_size"},
            },
        )
        print(f"ONNX export complete: {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PilotNetTrajectory to ONNX format.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_points", type=int, default=20)
    parser.add_argument("--output_scale", type=float, default=10.0)
    
    # 正規化用の引数 (--mean, --std, --input_normalization) を削除しました
    
    main(parser.parse_args())