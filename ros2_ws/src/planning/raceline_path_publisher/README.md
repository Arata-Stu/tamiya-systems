# raceline_path_publisher

Raceline CSV から標準 ROS 2 メッセージを publish する package です。

- `global_path` (`nav_msgs/msg/Path`, frame=`map`)
- `trajectory` (`nav_msgs/msg/Path`, frame=`base_link`)

`trajectory` は `pure_pursuit_controller` がそのまま受け取りやすいように、
グローバル raceline から車両前方の一部を切り出して `base_link` 座標系に変換して publish します。

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
