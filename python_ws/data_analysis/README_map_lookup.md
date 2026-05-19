# MAP Lookup Bag Recording

`build_map_steering_lookup.py` は、現在の system が出している `vSLAM odom` と `cmd_drive` から、MAP controller 向けの初期 steering lookup table を作るためのスクリプトです。

この LUT は物理ベースの tire model ではなく、まずは

- `speed_mps = hypot(vx, vy)`
- `yaw_rate_radps = odom.twist.twist.angular.z`
- `lateral_accel_mps2 ~= speed_mps * yaw_rate_radps`

という経験モデルで `speed x steer -> lateral_accel` の関係を集めるものです。

## 何を記録するか

最低限必要な topic:

- `/visual_slam/tracking/odometry`
- `/jetracer/cmd_drive`

今回の実装では `scripts/launch_system.sh identification` を追加してあり、この preset を使うと次の方針で起動します。

- `vslam=true`
- `record=true`
- `use_camera=true`
- `use_vehicle=true`
- `localization=false`
- `use_lidar=false`

IMU は必須ではありません。今回の lookup 生成は `odom` と `cmd_drive` だけで動きます。

`sensor_data_recording` / `mapping` preset は初期マッピング向けで、vSLAM odom や cmd_drive を記録しない設定です。MAP lookup 用の bag では使わず、`identification` を使ってください。

また、default bag manager には将来の controller 側検証用として次も含めています。

- `/autonomous/trajectory_reference`

## 起動手順

1. システムを起動する

```bash
cd /Users/at/project/competition/tamiya-systems
bash scripts/launch_system.sh identification
```

2. 別ターミナルで録画開始

```bash
ros2 service call /bag_manager_node/start_recording std_srvs/srv/Trigger "{}"
```

3. 収録が終わったら録画停止

```bash
ros2 service call /bag_manager_node/stop_recording std_srvs/srv/Trigger "{}"
```

bag は通常 `/record/<session_timestamp>/<take_timestamp>/` 以下に保存されます。

## offline VSLAM について

はい、offline でも進められます。最低限

- stereo camera image
- camera_info
- tf / tf_static
- cmd_drive

が bag に入っていれば、あとから rosbag 再生で VSLAM を動かして odometry を生成し、その odom を使って lookup を作る流れは成立します。

今回の `build_map_steering_lookup.py` 自体は `odom + cmd_drive` を入力にするので、offline でやる場合は「先に VSLAM を再実行して odom を得る」段階が 1 回増える、というイメージです。

## 収録のしかた

LUT 品質を上げるコツは「速度」と「操舵」を一度に大きく振らず、なるべく定常状態のサンプルを増やすことです。

おすすめの流れ:

1. 5〜10秒ほど直進してウォームアップする
2. 安全な範囲で 2〜4 個くらいの目標速度帯を決める
3. 各速度帯で、小舵角、中舵角、大舵角をそれぞれ 2〜3 秒ずつ保つ
4. 左旋回と右旋回を両方集める
5. 急な加減速をしながら同時に舵を当てる区間は、なるべく解析用の主データにしない

実際には次のような感覚で十分です。

- 低速で左に一定舵を 2〜3 秒
- 低速で右に一定舵を 2〜3 秒
- 中速で左に一定舵を 2〜3 秒
- 中速で右に一定舵を 2〜3 秒
- 可能なら少し大きめの舵角でも同じことを繰り返す

ポイント:

- 1 本の bag で全部を取ってもよい
- 左右どちらかに偏るより、両方向を揃えた方が lookup の確認がしやすい
- LUT の最初の版は「十分に密な網羅」より「破綻しない定常サンプル」を優先すると進めやすい

## 解析手順

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws
python data_analysis/build_map_steering_lookup.py \
  --bag /record/<session_timestamp>/<take_timestamp>/metadata.yaml
```

主なオプション:

- `--cmd-topic /jetracer/cmd_drive`
  - 既定値。最終的に車体へ入った指令から LUT を作る
- `--cmd-topic /autonomous/cmd_drive`
  - emergency や teleop を通す前の controller 出力で見たいときに使う
- `--command-delay-sec 0.05`
  - 指令から車体応答までの遅れを少し見込みたいときに使う
- `--min-speed 1.0`
  - 低速ノイズを減らしたいときの下限
- `--min-samples-per-bin 5`
  - lookup の 1 セルを直接観測扱いにするための最低サンプル数

出力:

- `/tmp/<bag_name>_map_lookup_table.csv`
  - race_stacks 互換の lookup table
- `/tmp/<bag_name>_map_lookup_table_counts.csv`
  - 各セルに何サンプル入ったか
- `/tmp/<bag_name>_map_lookup_table_raw.csv`
  - odom と cmd を付き合わせた生データ

## 調整の目安

`matched_samples` が少ない場合:

- `--min-speed` を下げる
- `--min-abs-steer` を下げる
- `--command-delay-sec` を 0.0, 0.03, 0.05, 0.08 などで試す

`observed_cells` が少ない場合:

- 速度帯を少し減らす
- 舵角 bin を粗くする
- `--speed-bin-size` や `--steer-bin-size` を大きくする
- 1 つの定常旋回をもう少し長く取る

横加速度が不自然な場合:

- 急加速や急減速を含む区間が多すぎないか確認する
- `raw.csv` を見て speed / yaw_rate / steer の関係を点検する
- 必要なら今後 `/camera/imu` を使った改善版へ広げる
