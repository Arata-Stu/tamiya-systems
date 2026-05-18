# Camera Trajectory E2E

Camera imageから将来のローカル経路点列を直接推定する学習環境です。速度は教師信号に含めず、出力は `base_link` 基準の `(x, y)` 点列です。

## Dataset

```bash
cd python_ws/camera_trajectory
./1_create_dataset.sh \
  -b /path/to/rosbags \
  -o ./datasets \
  --image_topic /camera/left/image_raw \
  --pose_topic /visual_slam/tracking/odometry
```

システム既定の構成では、RealSense の `left gray 424x240` を入力に使う想定です。

`--pose_topic` は以下を想定しています。

- `/visual_slam/tracking/slam_path` (`nav_msgs/msg/Path`)
- `/visual_slam/tracking/odometry` (`nav_msgs/msg/Odometry`)
- `/visual_slam/tracking/vo_pose` (`geometry_msgs/msg/PoseStamped`)

各画像時刻に最も近い pose を始点にし、将来の走行軌跡から等距離サンプリングした点列を `trajectories.npy` に保存します。

## Train

```bash
python3 2_train.py data_path=./datasets model.num_points=20 model.output_scale=8.0
```

デフォルトの `model.architecture` は `bezier` です。モデルは cubic Bezier の制御点を予測し、
学習・可視化・export では従来と同じ `(num_points, 2)` の軌跡点列へサンプルして扱います。
従来の20点直接回帰に戻す場合は `model.architecture=direct` を指定してください。

## Export

```bash
python3 export_onnx.py \
  -c ./ckpts/train/YYYY-MM-DD/HH-MM-SS/best_model.pth \
  -o ./best_model.onnx \
  --num_points 20 \
  --output_scale 8.0
```

## SCP Checkpoints

```bash
chmod +x scp_ckpts.sh
./scp_ckpts.sh
```

`./ckpts/train/YYYY-MM-DD/HH-MM-SS` のような2階層目の checkpoint directory を複数選択して、
デフォルトでは `/home/tamiya/workspace/tamiya-systems/python_ws/ckpts/pilotnet_trajectory/` へ転送します。

## Visualize Inference

```bash
python3 visualize_inference.py \
  --checkpoint ./ckpts/train/YYYY-MM-DD/HH-MM-SS/best_model.pth \
  --data-dir ./datasets/test \
  --num-samples 12 \
  --stride 20 \
  --camera-height 0.18 \
  --camera-pitch-down-deg 15.0
```

出力は `./outputs/trajectory_vis/trajectory_*.png` です。左にカメラ画像と画像上への近似投影、右に `base_link` 基準の
top-down 軌跡を表示し、推論結果を緑、教師軌跡を黄で描画します。RealSense の `camera_info` から得た内部パラメータ
（`fx=615.9686`, `fy=616.2639`, `cx=320.4421`, `cy=246.1154`）をデフォルトで使います。
画像上の投影位置がずれる場合は、実機の取り付けに合わせて `--camera-height` と `--camera-pitch-down-deg` を調整してください。
画像投影を無効化する場合は `--no-image-projection` を付けます。

## Deploy

```bash
chmod +x 3_deploy_model.sh
./3_deploy_model.sh
```

デフォルトでは `/workspaces/isaac_ros_assets/models/pilotnet_trajectory/` 配下の既存 numeric version directory
（`1`, `2`, `3` など）を削除してから deploy します。Triton/Isaac ROS で古い TensorRT engine が同居すると、
export は成功しても推論時に古い version や互換性のない plan を掴んで失敗することがあるためです。
過去 version を残したい場合だけ `--keep-versions` を付けてください。
Triton の `config.pbtxt` は `config/config.pbtxt` を source としてコピーします。

```bash
./3_deploy_model.sh --num-points 20 --output-scale 8.0 --precision fp16
```
