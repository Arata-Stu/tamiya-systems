import torch
import argparse
import os
from pathlib import Path

from src.model import TinyLidarNet


def main(args):
    """PyTorchのTinyLidarNetモデルをONNX形式に変換します。"""

    checkpoint_path = Path(args.checkpoint).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"

    print("--- Configuration ---")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Output ONNX Path: {output_path}")
    print(f"Scan Points: {args.scan_points}")
    print(f"Input Shape: (1, 1, {args.scan_points})") 
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. モデルの構築
    model = TinyLidarNet(input_dim=args.scan_points, output_dim=2)

    # 2. 重みのロード
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. ONNXエクスポート用のダミー入力データを作成
    # 修正点: ScanEncoderNode の仕様に合わせて [1, 1, Points] の3次元にする
    # (Batch=1, History/Channel=1, Points=1080)
    dummy_input = torch.randn(1, 1, args.scan_points)

    # 4. ONNXとしてエクスポート
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),         
            input_names=['scan_input'],  
            output_names=['control_output'],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={
                'scan_input': {
                    0: 'batch_size',
                    1: 'history_size'  # 将来的に history_size を増やしても対応可能にする
                }, 
                'control_output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX export complete: {output_path}") 
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export TinyLidarNet to ONNX format.")
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-p', '--scan_points', type=int, default=1081)
    args = parser.parse_args()
    main(args)