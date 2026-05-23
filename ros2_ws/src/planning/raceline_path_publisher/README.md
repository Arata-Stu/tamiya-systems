# raceline_path_publisher

Raceline CSV から ROS 2 trajectory message を publish する package です。

- `global_path` (`nav_msgs/msg/Path`, frame=`map`)
- `trajectory` (`nav_msgs/msg/Path`, frame=`base_link`)
- `trajectory_reference` (`race_planning_msgs/msg/Trajectory`, frame=`base_link`)

`trajectory` は `pure_pursuit_controller` がそのまま受け取りやすいように、
グローバル raceline から車両前方の一部を切り出して `base_link` 座標系に変換して publish します。
`trajectory_reference` は同じ局所区間に対して、各点の pose に加えて
`track_s_m / path_s_m / speed_mps / curvature_radpm / acceleration_mps2`
を載せて publish します。将来の MPC など、速度付き参照軌道が必要な controller 向けです。
CSV に速度系の列がない場合、それらの値は 0 のまま publish されます。
HD map section の `speed_override_mps` を使う場合は、先に
`python_ws/data_analysis/apply_hd_map_section_speeds.py` で `vx_mps` を上書きした
raceline CSV を作り、その CSV を `raceline_csv_path` に渡します。

## 対応 CSV

- `generate_raceline.py` の出力
  - `s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`
- ヘッダ付きの `x,y` 系 CSV

区切り文字は `;` と `,` を自動判定します。

## 使い方

```bash
ros2 launch raceline_path_publisher raceline_path_publisher.launch.xml \
  raceline_csv_path:=/absolute/path/to/map_raceline.csv
```

逆周回で使う場合:

```bash
ros2 launch raceline_path_publisher raceline_path_publisher.launch.xml \
  raceline_csv_path:=/absolute/path/to/map_raceline.csv \
  direction:=reverse
```

`global_path` だけを地図に重ねて確認し、`map -> base_link` TF がまだない
offline debug では local 出力を止められます。

```bash
ros2 launch raceline_path_publisher raceline_path_publisher.launch.xml \
  raceline_csv_path:=/absolute/path/to/map_raceline.csv \
  publish_local_path:=false \
  publish_local_reference:=false
```
