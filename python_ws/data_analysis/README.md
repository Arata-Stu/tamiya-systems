# data_analysis

LiDAR / Camera など、複数センサの rosbag データ解析・調査用スクリプトを置くディレクトリです。

## 現在のスクリプト

- `visualize_scan_gradient.py`
  - rosbag から `sensor_msgs/msg/LaserScan`（既定: `/scan`）を取得
  - 指定フレームを極座標から XY に変換
  - ビームインデックス順グラデーションで散布図表示
  - 任意で PNG 保存
  - 任意で時系列動画（MP4 / GIF）保存

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
