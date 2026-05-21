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

interactive terminal で `--model-name` を省略した場合は、その場で Triton model 名を入力できます。

デフォルトでは `/workspaces/isaac_ros_assets/models/pilotnet/` 配下の既存 numeric version directory
（`1`, `2`, `3` など）を削除してから deploy します。Triton/Isaac ROS で古い TensorRT engine が同居すると、
export は成功しても推論時に古い version や互換性のない plan を掴んで失敗することがあるためです。
過去 version を残したい場合だけ `--keep-versions` を付けてください。

複数の camera E2E model を共存させたい場合は、version directory を増やすのではなく
別の Triton model 名で deploy するのが安全です。たとえば:

```bash
./3_deploy_model.sh --model-name pilotnet_normal
./3_deploy_model.sh --model-name pilotnet_avoid
```

このとき `config.pbtxt` の `name:` も deploy 時に自動で合わせて書き換えられます。

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

## ROS2 Node Pipeline Validation With Rosbag

launch 済みの `isaac_ros_camera_e2e_control` と、`--pause` で止めた `ros2 bag play` を使って、
実際の ROS2 node pipeline の出力 `/autonomous/cmd_drive_raw` を offline 推論結果と比較できます。

想定ワークフロー:

1. camera e2e launch を起動
2. `ros2 bag play /path/to/bag --clock --pause`
3. 別ターミナルで下記スクリプトを実行

```bash
python3 validate_ros2_pipeline_with_bag.py \
  --checkpoint ./ckpts/train/<date>/<time>/best_model.pth \
  --image-topic /camera/left/image_raw \
  --cmd-topic /autonomous/cmd_drive \
  --output-csv ./outputs/ros2_pipeline_eval.csv \
  --max-samples 50
```

このスクリプトは `rosbag2_player/play_next` を使って 1 メッセージずつ再生を進め、
画像を受けるたびに

- live ROS2 pipeline の出力
- `PyTorch(ros2 前処理)`
- `ONNX(ros2 前処理)`

を同じ行に CSV 出力します。

解釈の目安:

- `live` と `PyTorch(ros2)` がズレる:
  launch 中の node pipeline、Triton 配備物、または runtime 設定差を疑う
- `PyTorch(ros2)` と `ONNX(ros2)` がズレる:
  export artifact を疑う
- 3者が一致するのに実機走行だけおかしい:
  車体側の steering/throttle 校正、遅延、古い `model.plan` 掴みなどを疑う

補足:

- `isaac_ros_camera_e2e.launch.xml` のデフォルト (`control_filter:=false`) では decoder 出力は `/autonomous/cmd_drive` です。
- `/autonomous/cmd_drive_raw` を使うのは `control_filter:=true` で launch した場合です。
- `system.launch` / `launch_system.sh` から使うときは `e2e_camera_model_name:=pilotnet_normal` のように model 名を指定できます。
