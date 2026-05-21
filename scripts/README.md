# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets. `sensor_data_recording` / `mapping` / `vslam_map` は canonical な sensor-bag recording preset で、mono/stereo を `424x240x90` で起動します。interactive 起動時は `jetracer` の速度プロファイルを `slow (throttle_gain=0.1)` と `normal (throttle_gain=1.0)` から選べます。非 interactive では従来どおり mapping 系 preset の既定値として `slow` を使います。MAP 用の LUT 収集には `identification` を使います。評価用には `localization_eval`、`perception_eval`、`vslam_eval` の lean preset もあります。Perception は `--set use_perception=true` や interactive toggle で有効化できます。crop 後の分類器も使う場合は `--set use_perception_classifier=true` を追加します。D435 の mono/stereo を使うときは `--set perception_camera_source=left` または `right` を使えます。
- `tmux.sh`: create tmux layouts for robot, map creation, identification / MAP lookup recording, localization evaluation, perception evaluation, VSLAM evaluation, Python work, and simulator work.
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: VSLAM 専用。offline visual-map generation plus lightweight `scan + odom + tf` bag creation. `--mode vslam` は `launch_system.sh vslam_map` の `424x240x90` 録画に合わせた preset。
- `create_2d_map_from_bag.sh`: 2D map 作成用。mode は `no_odom_offline_vslam` / `no_odom_online_vslam` / `with_odom_offline_vslam` / `with_odom_online_vslam` の 4 通りです。前半は Cartographer が odom を使うか、後半は Cartographer と同時に VSLAM を走らせるかを表します。`default` は `no_odom_offline_vslam`、`2d_slam` は `no_odom_online_vslam` の互換 alias です。`--mode` を省略すると実行時に 4択で選べます。`with_odom_online_vslam` は live の `/visual_slam/tracking/odometry` を使い、`with_odom_offline_vslam` は先に VSLAM map を作ってから、その map を読み込んだ VSLAM で odom bag を生成し、その bag を Cartographer に渡します。この mode では source bag を 2 回 replay します。`with_odom_offline_vslam` では、odom bag の録画前に `ros2 topic hz -w 10` 相当の平均レート確認を行い、既定では `--image-fps` の 90% 以上に達してから録画を始めます。必要なら `--odom-ready-window` / `--odom-ready-min-rate` / `--odom-ready-timeout` / `--no-odom-ready-wait` で調整できます。centerline 前に GUI で map を手修正したい場合は `--edit-map`、都度確認したい場合は既定の `--map-edit-mode auto` を使います。完了後の転送前メニューから `section_editor.py` を開いて `sections_pixels.csv` も作れます。
- `scp_data.sh`: data transfer helper.

## Terminal viewers

The terminal image/ROS viewers live in `terminal_viewers/`:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_map_viewer.py
python3 /scripts/terminal_viewers/ros2_terminal_image_viewer.py
python3 /scripts/terminal_viewers/ros2_terminal_scan_viewer.py
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py
```

`ros2_terminal_dashboard.py` is the custom rviz-like entrypoint for SSH
debugging. It keeps topic names as command-line options, then lets you toggle
map, localization, scan, image, crop image, sections, gates, particles,
`slam_path`, `vo_path`, and planning paths from the keyboard.
`--camera-info-topic` を与えると image panel にも LaserScan を投影します。
crop preview は既定で `/perception/crop/image` を購読し、BEST_EFFORT QoS で
受け取ります。評価用途では topic 時刻の近い message だけを組み合わせ、
exact-time TF が引けないときは overlay を隠す既定動作です。旧来どおり
latest TF へ緩く fallback したい場合だけ `--allow-latest-tf-fallback` を
明示してください。

## Shared helpers

- `common/tui.sh`: shared shell UI helpers.

## Evaluation presets

`launch_system.sh` の lean preset は以下を想定しています。

- `identification`: `stereo camera + VSLAM + vehicle + rosbag manager`。MAP lookup 用の `odom + cmd_drive` 収録向けです。global localization や LiDAR は起動しません。
- `localization_eval`: `LiDAR + stereo camera + VSLAM + global localization + localization_manager`。車両 driver や perception は起動しません。
- `perception_eval`: `LiDAR + stereo left/right + perception cropper + classifier`。localization や vehicle 制御は起動しません。
- `vslam_eval`: `stereo camera + VSLAM` のみ。global localization や perception は起動しません。

例:

```bash
./scripts/launch_system.sh identification
./scripts/launch_system.sh production --set use_map_controller=true
./scripts/launch_system.sh localization_eval --set map_dir=/map/course_a
./scripts/launch_system.sh perception_eval
./scripts/launch_system.sh vslam_eval --set map_dir=/map/course_a
```
