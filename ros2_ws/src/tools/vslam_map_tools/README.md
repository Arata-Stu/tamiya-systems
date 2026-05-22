# vslam_map_tools

VSLAM map と 2D map を人手で合わせたり、landmarks 可視化から 2D の下絵 PNG を作るための補助ツールです。

## manual_tf_alignment_node.py

`map -> vslam_map` を動的に publish しながら、端末のキー操作で平行移動と回転を微調整します。
RViz2 で `/visual_slam/tracking/slam_path` や `/visual_slam/vis/landmarks_cloud` を見ながら合わせる用途です。

起動例:

```bash
ros2 launch vslam_map_tools manual_tf_alignment.launch.xml \
  parent_frame:=map \
  child_frame:=vslam_map \
  enable_keyboard:=true \
  config_path:=/tmp/vslam_map_alignment.yaml
```

主な操作:

- `w/s`: x 移動
- `a/d`: y 移動
- `r/f`: z 移動
- `q/e`: yaw 回転
- `t/g`: roll 回転
- `y/h`: pitch 回転
- `[` / `]`: 並進 step を半分 / 2 倍
- `-` / `=`: 回転 step を半分 / 2 倍
- `p`: 現在値を YAML 保存
- `c`: 保存せず session 終了
- `0`: 起動時の値へ戻す

大きく動かしたいときは大文字キーで 10 倍 step を使えます。

`system_launch` から使う場合は、たとえば次のようにします。

```bash
ros2 launch system_launch system.launch.xml \
  vslam:=true \
  use_vslam_map_alignment_node:=true \
  vslam_map_alignment_enable_keyboard:=true \
  enable_slam_visualization:=true \
  enable_landmarks_view:=true
```

`create_2d_map_from_bag.sh` から live alignment session を使う場合は、
`--live-vslam-map-align` か `--trace-vslam-landmarks` を付けた online mode で実行すると、
bag replay 中に RViz2 を開き、この node を前景で受け付けます。

## export_landmarks_png.py

`/visual_slam/vis/landmarks_cloud` を 2D に投影して PNG を作ります。
`--reference-yaml` を指定すると、既存 2D map と同じ `resolution/origin/image size` を使って rasterize できます。

```bash
ros2 run vslam_map_tools export_landmarks_png.py -- \
  --target-frame map \
  --reference-yaml /map/course_a/course_a.yaml \
  --output-image /map/course_a/course_a_vslam_landmarks.png \
  --output-yaml /map/course_a/course_a_vslam_landmarks.yaml
```

`--path-topic /visual_slam/tracking/slam_path` は既定で有効です。landmarks に加えて SLAM path も下絵に含めたいときに使えます。

## record_vslam_reference_snapshot.py / publish_saved_vslam_reference.py

`/visual_slam/tracking/slam_path` と `/visual_slam/tracking/odometry` の最後の状態を JSON に保存し、
後で current time 付きで再 publish する補助です。`create_2d_map_from_bag.sh --prepare-vslam-map-alignment`
ではこの組み合わせを使って、2D map と saved path を見ながら `map -> vslam_map` を後追い調整できます。

保存:

```bash
ros2 run vslam_map_tools record_vslam_reference_snapshot.py -- \
  --path-topic /visual_slam/tracking/slam_path \
  --odom-topic /visual_slam/tracking/odometry \
  --output /tmp/vslam_reference.json
```

再 publish:

```bash
ros2 run vslam_map_tools publish_saved_vslam_reference.py -- \
  --input /tmp/vslam_reference.json \
  --path-topic /visual_slam/tracking/slam_path \
  --odom-topic /visual_slam/tracking/odometry
```

## export_aligned_landmarks_offline.py

reference snapshot に保存した landmarks と path を offline PNG/YAML にします。
既存 2D map に重ねるときは従来どおり `--alignment` と `--reference-yaml`
を渡せます。HD map editor の下絵だけを作る場合は両方を省略でき、
snapshot points の bounds から `resolution` / `origin` を持つ raster を作ります。

```bash
python3 ros2_ws/src/tools/vslam_map_tools/vslam_map_tools/export_aligned_landmarks_offline.py \
  --snapshot /map/course_a/course_a_vslam_reference.json \
  --output-image /map/course_a/course_a_vslam_landmarks.png \
  --output-yaml /map/course_a/course_a_vslam_landmarks.yaml \
  --resolution 0.02 \
  --padding-m 0.5
```
