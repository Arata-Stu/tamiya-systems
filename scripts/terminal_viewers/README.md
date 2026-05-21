# terminal_viewers

Kitty-compatible terminal visualization tools for SSH-first ROS debugging.

- `terminal_image_smoke_test.py`: dependency-free terminal image protocol test.
- `ros2_terminal_image_viewer.py`: `sensor_msgs/Image` and `CompressedImage` viewer.
- `ros2_terminal_scan_viewer.py`: simple `sensor_msgs/LaserScan` viewer.
- `ros2_terminal_map_viewer.py`: 2D map, localization result, scan, path, particles, and section markers.
- `ros2_terminal_dashboard.py`: custom rviz-like dashboard with keyboard or mouse toggles for map, localization, scan, image, crop image, sections, gates, particles, `slam_path`, `vo_path`, and planning paths. `camera_info` があれば image panel にも scan 投影を重ね、`odom` があればヘッダに速度を表示します。

## Dashboard

Run the dashboard on the SSH destination where ROS 2 topics are visible:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --amcl-pose-topic /amcl_pose \
  --initial-pose-topic /initialpose \
  --scan-topic /scan \
  --odom-topic /visual_slam/tracking/odometry \
  --image-topic /camera/left/image_raw \
  --camera-info-topic /camera/left/camera_info \
  --best-effort
```

By default the dashboard also subscribes to `/perception/crop/image` for the
crop preview panel and uses BEST_EFFORT QoS for that topic.
Section-localizer debugging is also wired in by default through
`/localization/current_section`, `/localization/section_markers`, and
`/localization/current_section_marker`.

By default the dashboard shows the latest data from each topic so it stays
responsive during normal SSH debugging.

If you need delayed-log inspection, pass `--historical-sync`. In that mode the
dashboard matches recent messages by timestamp. It uses exact-time TF by
default and hides overlays when a synced scan / image / pose cannot be found
within `--sync-tolerance-ms` instead of silently mixing latest data. Path and
odom panels also use timestamped history, but they keep the latest message from
the past rather than jumping ahead to a future update when delayed logs arrive.
If you intentionally want the old forgiving behavior, pass
`--allow-latest-tf-fallback`.

If your bag playback or relay introduces longer delays for path / odom topics,
increase `--state-sync-buffer-size` when `--historical-sync` is enabled.
If terminal mouse handling gets in the way, pass `--no-mouse`.

For compressed images:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --scan-topic /scan \
  --odom-topic /visual_slam/tracking/odometry \
  --image-topic /camera/left/image_raw/compressed \
  --camera-info-topic /camera/left/camera_info \
  --compressed-image \
  --best-effort
```

Keyboard controls:

- `m`: map
- `l`: localization result
- `a`: AMCL pose
- `u`: AMCL initial pose
- `s`: scan。map overlay と image projection の両方を切り替え
- `i`: image
- `r`: crop image
- `c`: sections
- `g`: section gates
- `p`: particles
- `t`: slam_path
- `v`: vo_path
- `y`: global path
- `h`: local path
- `space`: pause
- `q`: quit

Mouse controls:

- click the header buttons to toggle overlays on/off
