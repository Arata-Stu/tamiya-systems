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
- `create_2d_map_from_bag.sh`: 2D map 作成用。`--mode 2d_slam` では、まず source bag から VSLAM map を作り、次にその保存済み VSLAM map を使った map-localization をもう一度 source bag へ流して lightweight `scan + odom + tf` bag を記録します。その lightweight bag で provisional 2D map を作り、さらに固定した 2D map の scan global localization 結果を `visual_slam/set_slam_pose` に入れて、source bag の先頭から VSLAM map だけを作り直す 3 パス目も試みます。既存の lightweight bag から 2D map 作成だけをやり直したい場合は fast path も使えます。`--pipeline-mode online` を使うと VSLAM と Cartographer を同じ source bag replay 上で同時実行し、VSLAM の live odom をそのまま 2D SLAM に使います。centerline 前に GUI で map を手修正したい場合は `--edit-map`、都度確認したい場合は既定の `--map-edit-mode auto` を使います。完了後の転送前メニューから `section_editor.py` を開いて `sections_pixels.csv` も作れます。
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
