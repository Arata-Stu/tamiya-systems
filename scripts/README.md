# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets. `sensor_data_recording` / `mapping` / `vslam_map` は canonical な sensor-bag recording preset で、mono/stereo を `424x240x90` で起動します。interactive 起動時は `jetracer` の速度プロファイルを `slow (throttle_gain=0.1)` と `normal (throttle_gain=1.0)` から選べます。非 interactive では従来どおり mapping 系 preset の既定値として `slow` を使います。MAP 用の LUT 収集には `identification` を使います。評価用には `localization_eval`、`perception_eval`、`vslam_eval` の lean preset もあります。Perception は `--set use_perception=true` や interactive toggle で有効化できます。crop 後の分類器も使う場合は `--set use_perception_classifier=true` を追加します。D435 の mono/stereo を使うときは `--set perception_camera_source=left` または `right` を使えます。
- `edit_control_filter_config.sh`: `sections_pixels.csv` を読み、section ごとの class 割り当てと class 別 filter/scale パラメータを対話編集して `control_filter.param.yaml` を生成します。
- `tmux.sh`: create tmux layouts for robot, map creation, identification / MAP lookup recording, localization evaluation, perception evaluation, VSLAM evaluation, Python work, and simulator work. `map` mode は `create_2d_map_from_bag.sh --prepare-vslam-map-alignment --trace-vslam-landmarks` を prefill し、alignment 用 RViz pane と manual TF node 用の空 pane も並べます。
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: VSLAM 専用。offline visual-map generation plus lightweight `scan + odom + tf` bag creation. `--mode vslam` は `launch_system.sh vslam_map` の `424x240x90` 録画に合わせた preset。
- `create_2d_map_from_bag.sh`: 2D map 作成用。mode は `no_odom_offline_vslam` / `no_odom_online_vslam` / `with_odom_offline_vslam` / `with_odom_online_vslam` の 4 通りです。前半は Cartographer が odom を使うか、後半は Cartographer と同時に VSLAM を走らせるかを表します。`default` は `no_odom_offline_vslam`、`2d_slam` は `no_odom_online_vslam` の互換 alias です。`--mode` を省略すると実行時に 4択で選べます。`with_odom_online_vslam` は live の `/visual_slam/tracking/odometry` を使い、`with_odom_offline_vslam` は先に VSLAM map を作ってから、その map を読み込んだ VSLAM で odom bag を生成し、その bag を Cartographer に渡します。この mode では source bag を 2 回 replay します。`with_odom_offline_vslam` では、odom bag の録画前に `ros2 topic hz -w 10` 相当の平均レート確認を行い、既定では `--image-fps` の 90% 以上に達してから録画を始めます。必要なら `--odom-ready-window` / `--odom-ready-min-rate` / `--odom-ready-timeout` / `--no-odom-ready-wait` で調整できます。`--prepare-vslam-map-alignment` を使うと provisional 2D map 生成後に `2D map publish + saved VSLAM path/odom republish` を立ち上げ、別 pane の `manual_tf_alignment_node.py` で落ち着いて `map -> vslam_map` を合わせられます。online mode の live alignment が必要なら `--live-vslam-map-align` も残っています。centerline 前に GUI で map を手修正したい場合は `--edit-map`、VSLAM landmarks を replay して tracing 用の blank-canvas editor へ流したい場合は `--trace-vslam-landmarks` を使います。saved `map -> vslam_map` 補正を再利用したいときは `--vslam-map-alignment-config /path/to/vslam_map_alignment.yaml` を併用できます。完了後の転送前メニューから `section_editor.py` を開いて `sections_pixels.csv` も作れます。
- `create_hd_map_from_vslam_bag.sh`: VSLAM landmarks/path の snapshot を下絵 PNG/YAML にして、lane の `left_bound` / `right_bound` / `centerline` を描く editable HD map YAML、primary centerline CSV、raceline CSV まで作る実験用 flow。bag 選択と offline VSLAM replay は `create_2d_map_from_bag.sh` の helper を再利用します。
- `scp_data.sh`: data transfer helper.

## VSLAM map alignment helpers

`ros2_ws/src/tools/vslam_map_tools/` に、VSLAM map と 2D map の橋渡し用ツールを追加しています。

- `manual_tf_alignment_node.py`: `map -> vslam_map` を publish しながらキー操作で微調整
- `export_landmarks_png.py`: `/visual_slam/vis/landmarks_cloud` を 2D PNG に rasterize

`system_launch` 経由で alignment node を有効にする例:

```bash
./scripts/launch_system.sh vslam_eval --set map_dir=/map/course_a \
  --set use_vslam_map_alignment_node=true \
  --set vslam_map_alignment_enable_keyboard=true \
  --set enable_slam_visualization=true \
  --set enable_landmarks_view=true
```

landmarks から下絵 PNG を保存する例:

```bash
ros2 run vslam_map_tools export_landmarks_png.py -- \
  --target-frame map \
  --reference-yaml /map/course_a/course_a.yaml \
  --output-image /map/course_a/course_a_vslam_landmarks.png
```

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
明示してください。localization overlay が有効なときは、その pose 時刻を
基準に scan / image を巻き戻して合わせる既定動作です。

## Shared helpers

- `common/tui.sh`: shared shell UI helpers.

## Control Filter Config

section localizer の `sections_pixels.csv` から `control_filter` 用の
section-class mapping YAML を作るには:

```bash
./scripts/edit_control_filter_config.sh --map-dir /map/course_a
```

既定では `sections_pixels.csv` を読み、`/map/course_a/control_filter.param.yaml`
へ保存します。保存後は例えば次のように使えます:

```bash
./scripts/launch_system.sh production \
  --set map_dir=/map/course_a \
  --set use_control_filter=true
```

`use_control_filter=true` のとき controller 群は `..._raw` topic へ publish し、
`control_filter` が filtered な `autonomous_control_cmd` へ戻します。
`/map/course_a/control_filter.param.yaml` が存在すれば自動で読みます。明示的に
別ファイルを使いたい場合だけ `--set control_filter_param=/path/to/file.yaml`
を追加してください。

## Unified E2E

`launch_system.sh` から E2E 系をまとめて扱う場合は `--e2e ...` を使います。
現状の variant は
`camera` / `lidar` / `camera_trajectory` / `lidar_trajectory` です。

```bash
./scripts/launch_system.sh production \
  --e2e camera
```

```bash
./scripts/launch_system.sh production \
  --e2e lidar_trajectory
```

`lidar_trajectory` は既存の MAGP RL trajectory launch をこの unified interface
から起動する alias です。必要なら `--set e2e_run_pure_pursuit=false` で
trajectory のみ出す構成にもできます。
従来どおり `--set use_e2e=true --set e2e_variant=...` でも指定できますが、
片方だけでは E2E pipeline が選べないため `launch_system.sh` は起動前に
エラーにします。

camera E2E と VSLAM を同時に起動するときも、`system.launch` は camera と
VSLAM の NITROS graph を同じ `camera_container` に置く既定です。executor
分離の効果を計測したい場合だけ、VSLAM の rectify / format conversion /
VisualSlamNode を別 component process に移す
`--set isolate_vslam_container=true` を試してください。

RealSense は stereo infrared を rectified な `image_rect_raw` として出すため、
`system.launch` の `vslam_use_image_preprocessors=auto` は RealSense path で
VSLAM 専用の rectify / mono conversion を省きます。raw image を出す camera
path では preprocessors を残します。必要なら
`--set vslam_use_image_preprocessors=true` か `false` で固定できます。

camera E2E は grayscale 画像から学習した PilotNet を前提に、encoder の
`ResizeNode` へ既定で `mono8` input を渡し、encoder 内の format conversion
で推論用の 3ch RGB tensor にします。カラー画像から同じ grayscale 前処理へ
落とす場合だけ `--set e2e_camera_input_is_grayscale=false` を指定します。
encoder の input image QoS は既定で `SENSOR_DATA` です。比較用に
`--set e2e_camera_encoder_input_qos=DEFAULT` などで差し替えられます。

component container を分けても Jetson 全体の CPU 使用量を制限するわけではなく、
高レート image topic が process 境界をまたぐ tradeoff もあります。
E2E が odometry だけを必要とする走行では、SLAM map 更新を止める構成と
camera FPS を落とす構成も比較してください。

```bash
./scripts/launch_system.sh production \
  --e2e camera \
  --set enable_localization_and_mapping=false \
  --set image_fps=30.0
```

VSLAM param YAML を差し替えて `slam_throttling_time_ms` などを調整する場合は
`--set vslam_param=/path/to/vslam.param.yaml` を使えます。

## Drive Mode Manager

`drive_mode_manager` は `section` と `path_obstacle_filter` の
`avoidance/following` 状態から `/control/drive_mode` を出します。

```bash
./scripts/launch_system.sh production \
  --set use_drive_mode_manager=true
```

section ごとの policy は `drive_mode_manager` の param YAML で
`allow_avoid` / `follow_only` / `normal_only` を設定できます。

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
./scripts/launch_system.sh vslam_eval --set map_dir=/map/course_a --set use_camera=false --set use_hd_map=true --set use_planning=true
```

最後の `vslam_eval` 例は source bag を `ros2 bag play <bag_path> --clock --start-paused`
で replay し、saved cuVSLAM map 上の VSLAM pose と HD lane/raceline を
offline で重ねて見る用途です。landmarks が必要な回だけ
`--set enable_slam_visualization=true --set enable_landmarks_view=true` を足します。
