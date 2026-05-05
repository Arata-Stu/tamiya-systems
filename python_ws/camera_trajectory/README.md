# Camera Trajectory E2E

Camera imageから将来のローカル経路点列を直接推定する学習環境です。速度は教師信号に含めず、出力は `base_link` 基準の `(x, y)` 点列です。

## Dataset

```bash
cd python_ws/camera_trajectory
./1_create_dataset.sh \
  -b /path/to/rosbags \
  -o ./datasets \
  --image_topic /realsense2_camera/color/image_raw \
  --pose_topic /visual_slam/tracking/odometry
```

`--pose_topic` は以下を想定しています。

- `/visual_slam/tracking/slam_path` (`nav_msgs/msg/Path`)
- `/visual_slam/tracking/odometry` (`nav_msgs/msg/Odometry`)
- `/visual_slam/tracking/vo_pose` (`geometry_msgs/msg/PoseStamped`)

各画像時刻に最も近い pose を始点にし、将来の走行軌跡から等距離サンプリングした点列を `trajectories.npy` に保存します。

## Train

```bash
python3 2_train.py data_path=./datasets model.num_points=20 model.output_scale=10.0
```

## Export

```bash
python3 export_onnx.py \
  -c ./ckpts/train/YYYY-MM-DD/HH-MM-SS/best_model.pth \
  -o ./best_model.onnx \
  --num_points 20 \
  --output_scale 10.0
```
