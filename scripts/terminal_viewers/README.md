# terminal_viewers

Kitty-compatible terminal visualization tools for SSH-first ROS debugging.

- `terminal_image_smoke_test.py`: dependency-free terminal image protocol test.
- `ros2_terminal_image_viewer.py`: `sensor_msgs/Image` and `CompressedImage` viewer.
- `ros2_terminal_scan_viewer.py`: simple `sensor_msgs/LaserScan` viewer.
- `ros2_terminal_map_viewer.py`: 2D map, localization result, scan, path, particles, and section markers.
- `ros2_terminal_dashboard.py`: custom rviz-like dashboard with keyboard toggles for map, localization, scan, image, sections, gates, particles, and path.

## Dashboard

Run the dashboard on the SSH destination where ROS 2 topics are visible:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --scan-topic /scan \
  --image-topic /camera/image_raw \
  --best-effort
```

For compressed images:

```bash
python3 /scripts/terminal_viewers/ros2_terminal_dashboard.py \
  --map-topic /map \
  --localization-topic /localization_result \
  --scan-topic /scan \
  --image-topic /camera/image_raw/compressed \
  --compressed-image \
  --best-effort
```

Keyboard controls:

- `m`: map
- `l`: localization result
- `s`: scan
- `i`: image
- `c`: sections
- `g`: section gates
- `p`: particles
- `t`: path
- `space`: pause
- `q`: quit
