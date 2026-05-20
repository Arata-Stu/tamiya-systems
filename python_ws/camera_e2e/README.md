# camera_e2e (PilotNet baseline)

`lidar_e2e` をベースにしたカメラE2E学習パイプラインです。
入力は `sensor_msgs/Image`（または `sensor_msgs/CompressedImage`）を想定し、出力は `[steer, speed]` です。

## 1. Dataset 作成

```bash
cd python_ws/camera_e2e
chmod +x 1_create_dataset.sh

./1_create_dataset.sh \
  --base_dir /path/to/rosbag_sequences \
  --outdir ./datasets \
  --image_topic /camera/left/image_raw \
  --cmd_topic /jetracer/cmd_drive \
  --image_storage npy
```

`--image_storage` は `npy` または `png`。

## 2. 学習

```bash
cd python_ws/camera_e2e
python3 2_train.py
```

設定は `config/train.yaml` で変更できます。

## 3. ONNX Export

```bash
python3 export_onnx.py \
  --checkpoint ./ckpts/train/<date>/<time>/best_model.pth \
  --channels 3 \
  --height 120 \
  --width 212 \
  --input_normalization external
```

## 4. Deploy (optional)

```bash
chmod +x 3_deploy_model.sh
./3_deploy_model.sh
```

デフォルトでは `/workspaces/isaac_ros_assets/models/pilotnet/` 配下の既存 numeric version directory
（`1`, `2`, `3` など）を削除してから deploy します。Triton/Isaac ROS で古い TensorRT engine が同居すると、
export は成功しても推論時に古い version や互換性のない plan を掴んで失敗することがあるためです。
過去 version を残したい場合だけ `--keep-versions` を付けてください。

## 画像保存形式の推奨

- `npy` (推奨):
  - 学習I/Oが速い
  - 1ファイル読み出しで済み、DataLoaderが安定しやすい
  - 容量は増えやすい
- `png`:
  - 可視化・目視確認しやすい
  - 可逆圧縮で画質劣化なし
  - 学習時のデコードコストと小ファイル管理コストが高い

まずは `npy` で学習速度を優先し、必要なら別途 `png` をサンプル保存して確認する運用が実用的です。

## 解像度方針 (Left Gray)

デフォルトは `left gray 424x240` を入力にし、その半分スケールの `212x120`（`W x H`）で学習する前提です。

- `config/train.yaml`:
  - `image_width: 212`
  - `image_height: 120`
  - `force_grayscale_3ch: true`
  - `crop_top_ratio: 0.0`（画角スケール維持）

## ROS2 推論時の正規化

`isaac_ros_dnn_image_encoder` 側で `image_mean=[0.5,0.5,0.5]` / `image_stddev=[0.5,0.5,0.5]` を設定する場合、  
`export_onnx.py` は `--input_normalization external`（デフォルト）でエクスポートしてください。

また、学習デフォルトでは `force_grayscale_3ch: true` のため、ROS2 側も `force_grayscale_3ch:=true`
で推論する前提です。color カメラ画像をそのまま 3ch で学習したときだけ `false` にしてください。

## PyTorch / Triton Artifact Compare

同じ 1 枚の画像に対して、以下を並べて比較できます。

- 学習前処理 + PyTorch checkpoint
- ROS2 推論前処理 + PyTorch checkpoint
- ROS2 推論前処理 + Triton 配備用 `model.onnx`（ONNX Runtime）

これにより、ズレが

- dataset / 学習前処理由来なのか
- ROS2 側の前処理由来なのか
- ONNX export / deploy artifact 由来なのか

を切り分けやすくなります。

```bash
python3 compare_pytorch_triton.py \
  --checkpoint ./ckpts/train/<date>/<time>/best_model.pth \
  --dataset-root ./datasets/test \
  --sample-index 0 \
  --output-dir ./outputs/compare/sample_000
```

raw bag の 1 フレームから直接比較したい場合:

```bash
python3 compare_pytorch_triton.py \
  --checkpoint ./ckpts/train/<date>/<time>/best_model.pth \
  --bag-dir /path/to/rosbag_sequence \
  --image-topic /camera/left/image_raw \
  --frame-index 0 \
  --output-dir ./outputs/compare/bag_frame_000
```

注意:

- `ONNX(ros2)` は live Triton server への問い合わせではなく、Triton に載せる `model.onnx` を ONNX Runtime で実行した結果です。
- ここで `PyTorch(ros2)` と `ONNX(ros2)` が一致するのに実機だけおかしい場合は、古い `model.plan` / version directory を Triton が掴んでいる可能性が高いです。
