# hd_map_virtual_scan

HD map YAML の lane boundary を仮想的な壁として扱い、自己位置から ray を飛ばして
`sensor_msgs/msg/LaserScan` を生成する composable node です。

主な用途は、VSLAM + HD map だけで LiDAR E2E 用の 1D scan を作ることです。
既定では `/visual_slam/tracking/odometry` を購読し、320 点の scan を
`/virtual_scan` に publish します。

```bash
ros2 launch hd_map_virtual_scan hd_map_virtual_scan.launch.xml \
  hd_map_yaml_path:=/map/course_a/course_a_hd_map.yaml \
  scan_topic:=/virtual_scan
```

LiDAR E2E の入力にそのまま使う場合は、E2E 側の scan topic を `/virtual_scan`
に向けるか、この node の `scan_topic` を `/scan` に remap してください。

```bash
./scripts/launch_system.sh hd_map_eval \
  --set map_dir=/map/course_a \
  --set use_virtual_scan=true \
  --set virtual_scan_topic=/virtual_scan
```

学習データを作る場合は、この node を起動した状態で source bag を replay し、
`/virtual_scan` と教師コマンド `/jetracer/cmd_drive` を bag に記録するか、
`python_ws/lidar_e2e/extract_topics.py --scan_topic /virtual_scan` で抽出します。
`launch_system.sh` の bag manager preset は `/virtual_scan` も記録対象に含みます。
