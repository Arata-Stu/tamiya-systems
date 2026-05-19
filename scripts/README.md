# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets. `sensor_data_recording` / `mapping` / `vslam_map` は canonical な sensor-bag recording preset で、mono/stereo を `424x240x90` で起動します。MAP 用の LUT 収集には `identification` を使います。評価用には `localization_eval`、`perception_eval`、`vslam_eval` の lean preset もあります。Perception は `--set use_perception=true` や interactive toggle で有効化できます。crop 後の分類器も使う場合は `--set use_perception_classifier=true` を追加します。D435 の mono/stereo を使うときは `--set perception_camera_source=left` または `right` を使えます。
- `tmux.sh`: create tmux layouts for robot, map creation, identification / MAP lookup recording, localization evaluation, perception evaluation, VSLAM evaluation, Python work, and simulator work.
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: VSLAM 専用。offline visual-map generation plus lightweight `scan + odom + tf` bag creation. `--mode vslam` は `launch_system.sh vslam_map` の `424x240x90` 録画に合わせた preset。
- `create_2d_map_from_bag.sh`: 2D map 作成用。既定では Cartographer 単体で 2D map を作成します。`--mode 2d_slam` または `--run-vslam` を付けると、source bag の replay 開始時から VSLAM と Cartographer を同時実行し、Cartographer は scan-only のまま 2D map を作りつつ、並行して VSLAM map も保存します。`/visual_slam/tracking/odometry` を Cartographer に渡す旧経路は廃止しました。centerline 前に GUI で map を手修正したい場合は `--edit-map`、都度確認したい場合は既定の `--map-edit-mode auto` を使います。完了後の転送前メニューから `section_editor.py` を開いて `sections_pixels.csv` も作れます。
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
map, localization, scan, image, sections, gates, particles, and path from the
keyboard. `--camera-info-topic` を与えると image panel にも LaserScan を投影します。

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
