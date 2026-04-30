# data_analysis

LiDAR / Camera など、複数センサの rosbag データ解析・調査用スクリプトを置くディレクトリです。

## 現在のスクリプト

- `visualize_scan_gradient.py`
  - rosbag から `sensor_msgs/msg/LaserScan`（既定: `/scan`）を取得
  - 指定フレームを極座標から XY に変換
  - ビームインデックス順グラデーションで散布図表示
  - 任意で PNG 保存
  - 任意で時系列動画（MP4 / GIF）保存

- `evaluate_global_localization_sweep.py`
  - `rosbag2_player` の `play_next` / `pause` / `resume` サービスを使って再生を自動ステップ実行
  - 一定 scan 間隔ごとに global localization を trigger
  - `localization_result` と参照自己位置（例: vSLAM pose）を比較して誤差CSVを出力

- `plot_localization_quality_map.py`
  - 既存の評価CSVと `map.yaml` から、良否ポイント図と成功率ヒートマップを生成

## 使い方（LiDAR）

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws

# 例1: rosbag2 ディレクトリの最後の /scan を表示
python data_analysis/visualize_scan_gradient.py \
  --bag /path/to/rosbag2_dir \
  --topic /scan

# 例2: 100フレーム目を保存のみ（表示なし）
python data_analysis/visualize_scan_gradient.py \
  --bag /path/to/rosbag2_dir \
  --topic /scan \
  --frame 100 \
  --output /tmp/scan_frame100.png \
  --no_show

# 例3: 0〜300フレームを10fpsでMP4出力（2フレームおき）
python data_analysis/visualize_scan_gradient.py \
  --bag /path/to/rosbag2_dir \
  --topic /scan \
  --video_output /tmp/scan_timeseries.mp4 \
  --video_fps 10 \
  --video_start 0 \
  --video_end 300 \
  --video_step 2 \
  --no_show

# 例4: 全フレームをGIF出力
python data_analysis/visualize_scan_gradient.py \
  --bag /path/to/rosbag2_dir \
  --topic /scan \
  --video_output /tmp/scan_timeseries.gif \
  --no_show
```

`--bag` は以下を受け付けます。
- rosbag2 ディレクトリ
- rosbag2 の `metadata.yaml`
- rosbag1 の `.bag`

動画モード時に使う主な引数:
- `--video_output`: 出力先（拡張子 `.mp4` または `.gif`）
- `--video_fps`: フレームレート
- `--video_start`: 開始フレーム
- `--video_end`: 終了フレーム（含む）
- `--video_step`: 何フレームおきに描画するか

## 備考

- 描画には `matplotlib` が必要です。未導入なら `pip install matplotlib` を実行してください。
- MP4出力には `ffmpeg` が必要です（未導入時はGIF出力を使ってください）。

## 使い方（global localization 自動評価）

1. 別ターミナルで rosbag 再生（開始時 paused 推奨）

```bash
ros2 bag play /path/to/bag_dir --clock --pause
```

2. システム側（localizationノード群）を起動

3. 評価スクリプトを実行

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws
python data_analysis/evaluate_global_localization_sweep.py \
  --use-sim-time \
  --scan-topic /scan \
  --scan-stride 50 \
  --localization-trigger-service /trigger_grid_search_localization \
  --localization-topic /localization_result \
  --reference-topic /visual_slam/tracking/vo_pose \
  --reference-type pose_stamped \
  --player-prefix /rosbag2_player \
  --localization-timeout-sec 8.0 \
  --map-yaml /path/to/levine.yaml \
  --good-pos-error-threshold-m 0.5 \
  --quality-grid-size-m 1.0 \
  --quality-min-samples-per-cell 1 \
  --output-csv /tmp/localization_sweep_eval.csv
```

主な出力列:
- `position_error_m`, `yaw_error_rad`: 誤差
- `localization_latency_sec`: trigger から結果到着までの時間
- `status`: timeout などの判定
- `reference_x`, `reference_y`: trigger時の参照自己位置（失敗行でも可能な限り保存）

`--map-yaml` を指定した場合は、CSVに加えて次の画像も保存されます。
- `<output-csvのstem>_points.png`: 良否ポイント可視化（青=good、赤=bad、黒x=timeout/trigger失敗）
- `<output-csvのstem>_success_rate.png`: 成功率ヒートマップ（緑=良い、赤=悪い）

補足:
- `reference-topic` は擬似GTとして使う自己位置（vSLAM等）に合わせて変更してください。
- `reference-type` は `pose_stamped` / `pose_cov` / `odom` から選べます。
- 典型的な組み合わせ:
  - `--reference-type pose_stamped --reference-topic /visual_slam/tracking/vo_pose`
  - `--reference-type pose_cov --reference-topic /visual_slam/tracking/vo_pose_covariance`
  - `--reference-type odom --reference-topic /visual_slam/tracking/odometry`
- 出力画像パスを固定したい場合は `--quality-points-output` / `--quality-rate-output` で指定できます。

既存CSVから再描画だけしたい場合:

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws
python data_analysis/plot_localization_quality_map.py \
  --eval-csv /tmp/localization_sweep_eval.csv \
  --map-yaml /path/to/levine.yaml \
  --good-pos-error-threshold-m 0.5 \
  --quality-grid-size-m 1.0 \
  --quality-min-samples-per-cell 1
```
