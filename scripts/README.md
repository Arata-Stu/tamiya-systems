# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets. `sensor_data_recording` is the canonical sensor-bag recording preset, and `vslam_map` records offline VSLAM-map bags at `424x240x30`.
- `tmux.sh`: create tmux layouts for robot, mapping, Python, and simulator work.
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: VSLAM 専用。offline visual-map generation plus lightweight `scan + odom + tf` bag creation. `--mode vslam` は `launch_system.sh vslam_map` の `424x240x30` 録画に合わせた preset。
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
keyboard.

## Shared helpers

- `common/tui.sh`: shared shell UI helpers.
