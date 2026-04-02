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
  --height 66 \
  --width 200
```

## 4. Deploy (optional)

```bash
chmod +x 3_deploy_model.sh
./3_deploy_model.sh
```

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

