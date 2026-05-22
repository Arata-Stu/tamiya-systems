# hd_map_publisher

Editable local HD map YAML から runtime 用の lane 表示と primary centerline path を publish します。
landmarks raster は HD map 作成時の下絵であり、この node は通常走行時に VSLAM
landmarks topic を必要としません。現在の VSLAM/HD map 実験フローでは VSLAM も
HD map も `map` frame を使います。

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

## Offline bag debug

saved cuVSLAM map と HD map/raceline 成果物を置いた `map_dir` に対して、
`vslam_eval` を bag time で起動できます。

```bash
./scripts/launch_system.sh vslam_eval \
  --set map_dir=/map/course_a \
  --set use_camera=false \
  --set use_hd_map=true \
  --set use_planning=true \
  --set localize_on_startup=true \
  --set planning_publish_local_path=false \
  --set planning_publish_local_reference=false
```

別 terminal で source bag を clock 付き再生します。

```bash
ros2 bag play /record/session/take --clock --start-paused
```

RViz は既存の VSLAM debug config を使えます。

```bash
rviz2 -d "$(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_debug.rviz" \
  --ros-args -p use_sim_time:=true
```

この RViz config には `/hd_map/lane_markers`、
`/hd_map/primary_centerline_path`、`/planning/global_raceline` を追加しています。
`use_camera=false` は bag の stereo image/camera info と `/tf_static` を使う
offline replay 向けです。sensor node も同時に起動したい場合だけ外してください。
`map_dir` から `<map_dir>/cuvslam_map` を cuVSLAM の
`load_map_folder_path` に渡します。`localize_on_startup=true` は replay が
saved map の原点付近から始まる実験向けです。途中位置から始める bag では
identity pose hint が外れるため、RViz の initial pose など別の hint を使ってください。
上の `planning_publish_local_*` は static overlay 用です。local trajectory は
車両姿勢を切り出すため `map -> base_link` TF が必要なので、Pure Pursuit 側まで
確認するときだけ有効に戻してください。VSLAM TF を使う場合は追跡開始後に
`--set publish_map_to_odom_tf=true` も必要です。

landmarks も一時的に見たい場合は launch に次を追加します。

```bash
--set enable_slam_visualization=true \
--set enable_landmarks_view=true
```

load path の確認には composed VSLAM node の parameter を見ます。

```bash
ros2 param get /visual_slam_node load_map_folder_path
ros2 param get /visual_slam_node enable_localization_n_mapping
ros2 param get /visual_slam_node localize_on_startup
```
