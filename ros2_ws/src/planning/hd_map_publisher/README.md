# hd_map_publisher

Editable local HD map YAML から runtime 用の lane 表示と primary centerline path を publish します。
landmarks raster は HD map 作成時の下絵であり、この node は通常走行時に VSLAM
landmarks topic を必要としません。

## Topics

- `/hd_map/lane_markers` (`visualization_msgs/msg/MarkerArray`)
  - lane ごとの `left_bound` / `right_bound` / `centerline`
- `/hd_map/primary_centerline_path` (`nav_msgs/msg/Path`)
  - `primary_lane_id` の centerline

marker と path は YAML の `frame_id` を使います。必要なら
`frame_id_override` parameter で上書きできます。

## Run

```bash
ros2 launch hd_map_publisher hd_map_publisher.launch.xml \
  hd_map_yaml_path:=/map/course_a/course_a_hd_map.yaml
```

`system_launch` から map directory 規約を使う場合:

```bash
./scripts/launch_system.sh production \
  --set map_dir=/map/course_a \
  --set use_hd_map=true
```

`map_dir` を使うと既定で `<map_dir>/<map_dir_name>_hd_map.yaml` を読みます。
別ファイルなら `--set hd_map_yaml_path=/absolute/path/to/file.yaml` を追加してください。

raceline の global/local path と local trajectory は既存
`raceline_path_publisher` が担当します。まず primary centerline だけを RViz で
確認したいときはこの package 単体で十分です。
