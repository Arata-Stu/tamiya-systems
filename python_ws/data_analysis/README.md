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

- `generate_centerline.py`
  - PNG / PGM occupancy map から centerline CSV を生成
  - 出力形式: `x_m,y_m,w_tr_right_m,w_tr_left_m`

- `generate_raceline.py`
  - centerline CSV から軽量な近似 raceline CSV を生成
  - race_stacks の重いROS/optimizer依存を直接持ち込まず、後で optimizer backend に差し替えやすいCLI境界にしている
  - 出力形式: `s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`

- `visualize_race_lines.py`
  - map画像に centerline / raceline CSV を重ねた確認用PNGを生成
  - `map.yaml` を指定すると、resolution / origin を使って world座標を画像座標へ変換する

- `check_global_opt_env.py`
  - optionalなglobal optimizer依存が現在のPython環境でimportできるか確認

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

## 使い方（centerline / raceline 生成）

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws

python data_analysis/generate_centerline.py \
  --map /path/to/map.png \
  --yaml /path/to/map.yaml \
  --output /path/to/map_centerline.csv \
  --debug-dir /path/to/map_centerline_debug

python data_analysis/generate_raceline.py \
  --centerline /path/to/map_centerline.csv \
  --output /path/to/map_raceline.csv

# Dockerにglobal optimizer依存が入っている場合:
python data_analysis/generate_raceline.py \
  --backend global-opt \
  --optimizer-root /workspaces/tamiya-systems/python_ws/global_racetrajectory_optimization \
  --opt-type mincurv_iqp \
  --centerline /path/to/map_centerline.csv \
  --output /path/to/map_raceline.csv

# 依存がなければglobal-optを試して軽量版へfallback:
python data_analysis/generate_raceline.py \
  --backend auto \
  --centerline /path/to/map_centerline.csv \
  --output /path/to/map_raceline.csv

python data_analysis/visualize_race_lines.py \
  --yaml /path/to/map.yaml \
  --centerline /path/to/map_centerline.csv \
  --raceline /path/to/map_raceline.csv \
  --output /path/to/map_lines.png
```

`scripts/create_2d_map_from_bag.sh` では、centerline / raceline 生成後に preview画像も生成します。スキップしたい場合は `--no-raceline` または `--no-line-preview` を指定してください。

global optimizerを使う場合は `python_ws/requirements_global_opt.txt` の依存をDockerへ入れてください。現在のglobal-opt backendは `shortest_path` / `mincurv` / `mincurv_iqp` を対象にしており、古い `casadi` 依存を避けるため `mintime` はまだ扱いません。

依存確認:

```bash
python data_analysis/check_global_opt_env.py \
  --optimizer-root /workspaces/tamiya-systems/python_ws/global_racetrajectory_optimization
```

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
  --scan-topic /scan \
  --scan-stride 50 \
  --max-play-next-calls-per-trigger 0 \
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

最短実行（主要デフォルトを利用）:

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws
python data_analysis/evaluate_global_localization_sweep.py \
  --map-yaml /path/to/levine.yaml
```

主な出力列:
- `position_error_m`, `yaw_error_rad`: 誤差
- `localization_latency_sec`: trigger から結果到着までの時間
- `status`: timeout などの判定
- `reference_x`, `reference_y`: trigger時の参照自己位置（失敗行でも可能な限り保存）

`--map-yaml` を指定した場合は、CSVに加えて次の画像も保存されます。
- `<output-csvのstem>_points.png`: ポイント可視化
  `reference` ありでは青=good、赤=bad、黒x=timeout/trigger失敗。
  `reference` なしでは橙=localized (no reference) を重ねます。
- `<output-csvのstem>_success_rate.png`: ヒートマップ
  `reference` ありでは成功率ヒートマップ（緑=良い、赤=悪い）。
  `reference` なしでは localization が返った地点の相対密度ヒートマップ。

補足:
- `use_sim_time` はデフォルトで有効です。無効化したい場合だけ `--no-use-sim-time` を指定してください。
- bag 内に `/joy` や画像などの高頻度トピックが多い場合、`play_next` 1回で `/scan` が1件進むとは限りません。`--scan-stride 50` でも数千回の `play_next` が必要になることがあるため、`--max-play-next-calls-per-trigger 0` を推奨します。
- `reference-topic` は擬似GTとして使う自己位置（vSLAM等）に合わせて変更してください。
- `reference-topic` が取れない場合でも評価は継続されます。その場合の `status` は `ok_no_reference` になり、誤差列は空欄のまま、位置プロットと密度ヒートマップだけ生成されます。
- `reference-type` は `pose_stamped` / `pose_cov` から選べます。
- 典型的な組み合わせ:
  - `--reference-type pose_stamped --reference-topic /visual_slam/tracking/vo_pose`
  - `--reference-type pose_cov --reference-topic /visual_slam/tracking/vo_pose_covariance`
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
