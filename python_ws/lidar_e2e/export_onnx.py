import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from src.model import TinyLidarNet


class NormalizedModel(nn.Module):
    """
    入力スキャンを正規化してからモデルに入力するラッパー
    """
    def __init__(self, model, max_range=30.0):
        super().__init__()
        self.max_range = max_range
        self.model = model

    def forward(self, x):
        # x: [Batch, 1, Points]
        
        # 1. 無効値 (NaN, Inf) の処理
        # ONNX export対応のため、単純なclampやwhereを使用
        # 注: TensorRT等でのNaN挙動は実装依存だが、ここでは学習時と同じく
        # nan -> max_range として扱うのが安全。ただしONNX opsetでのサポート考慮し
        # ここでは単純に clamp(0, max_range) してから割る簡易実装とする。
        # 厳密な nan_to_num はONNXで重くなることがあるため、実環境Lidarドライバが
        # 0.0 や inf を出す場合の対策として clampを入れる。
        
        x_clamped = torch.clamp(x, 0.0, self.max_range)
        x_norm = x_clamped / self.max_range
        
        return self.model(x_norm)


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
    print(f"Max Range: {args.max_range} m (Normalization)")
    print(f"Input Shape: (1, 1, {args.scan_points})") 
    print("---------------------")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. モデルの構築
    base_model = TinyLidarNet(input_dim=args.scan_points, output_dim=2)

    # 2. 重みのロード
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)

    base_model.eval()
    
    # 3. 正規化ラッパーの適用
    model = NormalizedModel(base_model, max_range=args.max_range)
    model.eval()

    # 4. ONNXエクスポート用のダミー入力データを作成
    # ScanEncoderNode の仕様に合わせて [1, 1, Points] の3次元にする
    dummy_input = torch.randn(1, 1, args.scan_points)

    # 5. ONNXとしてエクスポート
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
                    1: 'history_size' 
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
    parser.add_argument('--max_range', type=float, default=30.0, help="Max lidar range for normalization")
    args = parser.parse_args()
    main(args)