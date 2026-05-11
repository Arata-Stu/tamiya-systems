# scripts

Operational scripts used inside the development container are kept at this
directory root because several launch/tmux/docs commands call `/scripts/*.sh`
directly.

## Main entrypoints

- `run_docker.sh`: start the Isaac ROS development container.
- `launch_system.sh`: launch system presets.
- `tmux.sh`: create tmux layouts for robot, mapping, Python, and simulator work.
- `monitor.sh`: terminal monitoring dashboard.
- `create_vslam_map_from_bag.sh`: offline visual-map generation plus lightweight `scan + odom + tf` bag creation.
- `create_2d_map_from_bag.sh`: offline 2D map generation. `--run-vslam` / `--use-vslam-odom` で VSLAM map 保存と lightweight 2D input bag 作成もまとめて実行できます。
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
