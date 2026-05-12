# data_analysis

LiDAR / Camera など、複数センサの rosbag データ解析・調査用スクリプトを置くディレクトリです。

## 現在のスクリプト

- `visualize_scan_gradient.py`
  - rosbag から `sensor_msgs/msg/LaserScan`（既定: `/scan`）を取得
  - 指定フレームを極座標から XY に変換
  - ビームインデックス順グラデーションで散布図表示
  - 任意で PNG 保存
  - 任意で時系列動画（MP4 / GIF）保存

- `visualize_camera_crop.py`
  - rosbag から `sensor_msgs/msg/Image` / `sensor_msgs/msg/CompressedImage` を取得
  - 指定した crop 比率を画像へ重ねて時系列動画（MP4 / GIF）化
  - 切り落とす領域を半透明オーバーレイで表示
  - 目視で `camera_crop` の妥当性を確認したいとき向け

- `analyze_camera_crop.py`
  - rosbag から `sensor_msgs/msg/Image` / `sensor_msgs/msg/CompressedImage` を取得
  - 各フレームで特徴点を抽出し、画像内の分布を集計
  - 上下左右それぞれについて crop 量に対する「特徴点保持率」を CSV 出力
  - 推奨 crop 比率を算出し、プレビュー画像とヒートマップ付き PNG を生成
  - `camera_crop` や vSLAM の `img_mask_top/bottom/left/right` を決める判断材料に使える

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

## 使い方（camera crop 解析）

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws

# 例1: 左右カメラの特徴点分布から推奨 crop を確認
python data_analysis/analyze_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topics /camera/camera/infra1/image_rect_raw /camera/camera/infra2/image_rect_raw

# 例2: フレーム数を増やして、95%特徴保持を満たす crop を見る
python data_analysis/analyze_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topics /camera/camera/infra1/image_rect_raw /camera/camera/infra2/image_rect_raw \
  --max_frames_per_topic 1000 \
  --frame_stride 2 \
  --retained_feature_ratio 0.95 \
  --frame_quantile 0.10 \
  --output_dir /tmp/vslam_crop_check

# 例3: PNG不要でCSVだけ欲しい場合
python data_analysis/analyze_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topics /camera/left/image_raw \
  --no_plots
```

主な出力:
- `recommended_crop_summary.csv`
  - 各トピックごとの推奨 `top/bottom/left/right` crop 比率と pixel 値
  - その4辺を同時適用したときの特徴点保持率も併記
- `<topic>_top_retained.csv` など
  - crop 比率ごとの aggregate 特徴保持率と per-frame 分位点
- `<topic>_summary.png`
  - 特徴点ヒートマップ
  - 推奨 crop 線
  - crop 比率に対する保持率カーブ

推奨値の意味:
- 既定では「各フレーム特徴点保持率の 10 パーセンタイルが 95% 以上となる最大 crop 比率」を推奨値にします
- つまり、厳しめに見ても大半のフレームで特徴点を残せる範囲を返します
- もっと保守的にしたい場合は `--retained_feature_ratio` を上げるか、`--frame_quantile` を下げてください

注意:
- このスクリプトは「特徴点が画像のどこに多いか」を見ています。最終的な vSLAM 成功率は視差、露出、モーションブラー、同期精度にも依存します
- まずこの解析で候補値を決め、その後に短い bag で実際に vSLAM を再生して確認する運用を想定しています

## 使い方（camera crop 可視化）

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws

# 例1: 左カメラに crop 枠を重ねた MP4 を出力
python data_analysis/visualize_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topic /camera/camera/infra1/image_rect_raw \
  --output /tmp/camera_crop_preview.mp4 \
  --top_ratio 0.10 \
  --bottom_ratio 0.15 \
  --left_ratio 0.00 \
  --right_ratio 0.00

# 例2: GIF で軽く確認
python data_analysis/visualize_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topic /camera/camera/infra1/image_rect_raw \
  --output /tmp/camera_crop_preview.gif \
  --video_start 0 \
  --video_end 300 \
  --video_step 3 \
  --top_ratio 0.08 \
  --bottom_ratio 0.20 \
  --shade_alpha 0.40

# 例3: pixel 指定で確認
python data_analysis/visualize_camera_crop.py \
  --bag /path/to/rosbag2_dir \
  --topic /camera/camera/infra1/image_rect_raw \
  --output /tmp/camera_crop_preview_px.mp4 \
  --top_px 40 \
  --bottom_px 72 \
  --left_px 0 \
  --right_px 0
```

主な引数:
- `--top_ratio --bottom_ratio --left_ratio --right_ratio`
  - 各辺から何割切るかを指定
- `--top_px --bottom_px --left_px --right_px`
  - 各辺から何 pixel 切るかを指定
  - 指定した辺では ratio より pixel が優先される
- `--shade_alpha`
  - 切り落とす領域の色の濃さ
- `--video_start --video_end --video_step`
  - どのフレーム範囲を出力するか
- `--resize_width`
  - 出力サイズを軽くしたいときの横幅指定

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

`quadprog` は環境によっては install 後も import 時に `undefined symbol` で失敗することがあります。これは依存解決エラーではなく、ネイティブ拡張の ABI 不整合です。その場合は一度 `quadprog` と `trajectory-planning-helpers` を削除し、`trajectory-planning-helpers` の自動依存解決を使わずに `quadprog==0.1.6` を先に入れてから `trajectory-planning-helpers --no-deps` を入れる回避策を試してください。

また、`trajectory-planning-helpers 0.79` は新しめの `scipy` で `spline_approximation` が落ちることがあります。このリポジトリの `generate_raceline.py` では互換パッチを当てて回避しています。

centerline の形状によっては spline 近似後の法線が交差し、global-opt backend が失敗することがあります。このリポジトリの `generate_raceline.py` では `--global-opt-spline-smoothing` を段階的に引き上げて自動リトライします。まだ失敗する場合は、手動で `--global-opt-spline-smoothing 40` や `80` を指定してください。

`trajectory-planning-helpers` の版によっては `iqp_handler` の引数が異なります。このリポジトリの `generate_raceline.py` では版差分を吸収し、必要なら `mincurv_iqp` から `mincurv` 相当へフォールバックします。

```bash
pip uninstall -y trajectory-planning-helpers quadprog
pip install --no-cache-dir "quadprog==0.1.6"
pip install --no-cache-dir "trajectory-planning-helpers==0.79" --no-deps
python data_analysis/check_global_opt_env.py \
  --optimizer-root /workspaces/tamiya-systems/python_ws/global_racetrajectory_optimization
```

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
