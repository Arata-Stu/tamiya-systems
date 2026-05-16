# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets. `sensor_data_recording` is the canonical sensor-bag recording preset, and `vslam_map` records offline VSLAM-map bags at `1280x720x30`.
- `tmux.sh`: create tmux layouts for robot, mapping, Python, and simulator work.
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: VSLAM 専用。offline visual-map generation plus lightweight `scan + odom + tf` bag creation. `--mode vslam` は `launch_system.sh vslam_map` の `1280x720x30` 録画に合わせた preset。
- `create_2d_map_from_bag.sh`: 2D map 作成用。`--mode 2d_slam` で VSLAM map 保存と lightweight 2D input bag 作成もまとめて実行でき、既定では provisional 2D map 作成後にその固定 2D map の scan global localization 結果を `visual_slam/set_slam_pose` に入れて、VSLAM map だけを source bag から作り直す 3 パス目も試みます。centerline 前に GUI で map を手修正したい場合は `--edit-map`、都度確認したい場合は既定の `--map-edit-mode auto` を使います。完了後の転送前メニューから `section_editor.py` を開いて `sections_pixels.csv` も作れます。
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
