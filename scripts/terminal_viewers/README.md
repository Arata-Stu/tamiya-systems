# terminal_viewers

Kitty-compatible terminal visualization tools for SSH-first ROS debugging.

- `terminal_image_smoke_test.py`: dependency-free terminal image protocol test.
- `ros2_terminal_image_viewer.py`: `sensor_msgs/Image` and `CompressedImage` viewer.
- `ros2_terminal_scan_viewer.py`: simple `sensor_msgs/LaserScan` viewer.
- `ros2_terminal_map_viewer.py`: 2D map, localization result, scan, path, particles, and section markers.
- `ros2_terminal_dashboard.py`: custom rviz-like dashboard with keyboard toggles for map, localization, scan, image, sections, gates, particles, and path. `camera_info` があれば image panel にも scan 投影を重ねます。

## Dashboard

Run the dashboard on the SSH destination where ROS 2 topics are visible:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --amcl-pose-topic /amcl_pose \
  --initial-pose-topic /initialpose \
  --scan-topic /scan \
  --image-topic /camera/left/image_raw \
  --camera-info-topic /camera/left/camera_info \
  --best-effort
```

For compressed images:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --scan-topic /scan \
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
- `c`: sections
- `g`: section gates
- `p`: particles
- `t`: path
- `space`: pause
- `q`: quit
