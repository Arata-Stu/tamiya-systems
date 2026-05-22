# map_section_editor

2D map 向けの GUI ツール置き場です。

- `section_editor.py`: `map.yaml` 上でセクションをポリゴン分割し、ROS2ノード用のセクション定義CSVを作ります。
- `control_filter_config_editor.py`: `sections_pixels.csv` を読み、section ごとの class 割り当てと class 別 `control_filter` パラメータを terminal で編集して YAML を生成します。
- `map_cleanup_editor.py`: centerline 前処理用。起動時に map を黒白へ二値化したうえで、黒塗り/白戻しして cleaned PNG を保存します。
- `hd_map_editor.py`: landmarks raster や occupancy map を下絵にして lane の `left_bound` / `right_bound` / `centerline` を描き、編集可能な HD map YAML と primary lane の centerline CSV を保存します。

## hd_map_editor.py

VSLAM landmarks と SLAM path を rasterize した map YAML、または既存の
occupancy map YAML を下絵にして、ローカル `map` frame の lane geometry を作ります。
保存先 YAML は world 座標の点列なので、GUI で再読込して直すことも、YAML を直接編集することもできます。

bag 選択から offline VSLAM、landmarks raster、editor、raceline 生成まで進める
実験用 entrypoint:

```bash
cd /Users/at/project/competition/tamiya-systems
bash scripts/create_hd_map_from_vslam_bag.sh
```

前回の VSLAM snapshot を使って HD map editor からやり直す場合:

```bash
bash scripts/create_hd_map_from_vslam_bag.sh \
  --skip-vslam \
  --map-dir /map/course_a \
  --map-name course_a \
  --snapshot /map/course_a/course_a_vslam_reference.json
```

landmarks 下絵を使う最小例:

```bash
ros2 run vslam_map_tools export_landmarks_png.py -- \
  --target-frame map \
  --output-image /map/course_a/course_a_vslam_landmarks.png \
  --output-yaml /map/course_a/course_a_vslam_landmarks.yaml

cd python_ws
python3 map_section_editor/hd_map_editor.py \
  --map-yaml /map/course_a/course_a_vslam_landmarks.yaml \
  --output /map/course_a/course_a_hd_map.yaml \
  --centerline-output /map/course_a/course_a_centerline.csv
```

`export_landmarks_png.py` は既定で `/visual_slam/tracking/slam_path` も下絵へ描きます。
下絵の灰色 path を centerline の初期目安にしつつ、landmarks のコース形状を見て
境界と中心線を補正してください。

操作:

- `1` / `2` / `3`: `centerline` / `left_bound` / `right_bound` を選択
- 左クリック: 選択中の線へ点を追加。既存点の近くではドラッグ移動
- `d` または `Backspace`: カーソル近傍の点を削除。近傍に無ければ末尾を削除
- `u`: 選択中の線の末尾点を削除
- `n`: lane を追加
- `[` / `]`: active lane を切替
- `p`: active lane を primary lane に設定。centerline CSV は primary lane から出力
- `o`: active lane の open / closed loop を切替
- `s`: HD map YAML と primary lane centerline CSV を保存
- ホイール / `+` / `-`: ズーム
- 右ドラッグ or `H/J/K/L`: パン
- `0`: raster 全体をフィット表示
- `i`: ヘルプ表示切替
- `q` or `Esc`: 終了

editor は既定で raster の native zoom から開きます。まず全体像を見たい場合は
`0`、script 起動時から全体表示にしたい場合は `create_hd_map_from_vslam_bag.sh`
へ `--editor-scale 0` を渡してください。

primary lane の centerline CSV は
`x_m,y_m,w_tr_right_m,w_tr_left_m` 形式です。左右幅は描いた境界から
centerline 点ごとに計算します。既存 raceline generator へ渡せます。
`create_hd_map_from_vslam_bag.sh` は editor 終了後に `<map>_lines.png` も作り、
landmarks raster の `resolution` / `origin` で HD map YAML の lane bounds /
lane centerline、exported centerline CSV、raceline を下絵へ投影します。
line preview を飛ばす場合は `--no-line-preview` です。

```bash
cd python_ws
python3 data_analysis/generate_raceline.py \
  --preset race-stacks \
  --backend global-opt \
  --opt-type mincurv \
  --centerline /map/course_a/course_a_centerline.csv \
  --output /map/course_a/course_a_raceline.csv

python3 data_analysis/visualize_race_lines.py \
  --yaml /map/course_a/course_a_vslam_landmarks.yaml \
  --hd-map /map/course_a/course_a_hd_map.yaml \
  --centerline /map/course_a/course_a_centerline.csv \
  --raceline /map/course_a/course_a_raceline.csv \
  --output /map/course_a/course_a_lines.png
```

この HD map wrapper の raceline 既定値も `race-stacks` / `global-opt` /
`mincurv` です。global optimizer 依存が未導入なら raceline 生成だけ warning にして、
HD map YAML と centerline CSV は残します。下絵 raster は landmarks を黒寄り、
saved VSLAM path を濃い青で描きます。

YAML を手で編集したあと CSV だけ作り直す場合:

```bash
cd python_ws
python3 map_section_editor/hd_map_editor.py \
  --map-yaml /map/course_a/course_a_vslam_landmarks.yaml \
  --output /map/course_a/course_a_hd_map.yaml \
  --centerline-output /map/course_a/course_a_centerline.csv \
  --export-only
```

最初の HD map YAML には source raster の `resolution` / `origin` / image size と
lane 点列を残します。分岐や追い越し lane を描き始める場合も lane を複数保存できますが、
現時点で CSV export 対象は `primary_lane_id` の 1 本です。

## map_cleanup_editor.py

分岐やノイズで centerline がうまく引けない地図を、人手で少し整えてから
`generate_centerline.py` / `generate_raceline.py` に渡したいときのツールです。
既定では、`250` 以上の画素を白、未満を黒に寄せてから編集します。

```bash
cd python_ws
python3 map_section_editor/map_cleanup_editor.py \
  --input ../ros2_ws/src/launch/system_launch/config/simulator/levine.png \
  --output ./map_section_editor/levine_centerline_input.png
```

landmarks PNG を下絵にして新しく map を描きたい場合:

```bash
cd python_ws
python3 map_section_editor/map_cleanup_editor.py \
  --input /map/course_a/course_a_vslam_landmarks.png \
  --output /map/course_a/course_a_vslam_traced.png \
  --initialize-mode blank_black
```

この mode では、入力 PNG 自体は編集せず reference overlay として表示し、
黒い blank canvas の上に白で走行可能領域を描いていけます。

操作:

- 左ドラッグ: 現在のブラシ色で描画
- `1` / `2` / `3`: brush / line / smooth curve tool
- line tool: クリックした 2 点の間に直線を引く
- curve tool: 制御点を複数クリックして `Enter` で滑らかな曲線を確定
- `x`: pending line / curve をクリア
- `v`: reference overlay の表示/非表示
- `b` または上部 `Black` ボタン: 黒で塗る
- `e` または上部 `White` ボタン: 白で戻す
- `i` または右上の `Help` ボタン: 説明パネルの表示/非表示
- `u`: undo
- `r`: 今回開いた状態に戻す
- `R`: 今回の二値化後入力画像に戻す
- `s`: 保存
- `[` / `]` または `,` / `.` または右上の `-` / `+` ボタン: ブラシサイズ変更
- ホイール / `+` / `-`: ズーム
- 右ドラッグ or `H/J/K/L` or 矢印キー: パン
- `q` or `Esc`: 終了

`create_2d_map_from_bag.sh` からは `--edit-map` または `--map-edit-mode auto`
で呼び出せます。section editor は同スクリプトの転送前メニューからも開けます。

主な追加 option:

- `--initialize-mode binarized|blank_white|blank_black`
- `--reference-image PATH`
- `--reference-alpha 0.45`
- `--binarize-white-threshold N`

## section_editor.py

## 使い方

```bash
cd python_ws
python3 map_section_editor/section_editor.py \
  --map-yaml ../ros2_ws/src/launch/system_launch/config/simulator/levine.yaml \
  --output ./map_section_editor/levine_sections.csv \
  --window-width 1600 \
  --window-height 1000 \
  --overlap-mode overwrite
```

小さい地図で描きづらい場合は、`--scale 3.0` のように初期倍率を上げるか、
起動後にホイール/`+`キーで拡大してください（`--scale 0` は自動フィット）。

画面上部に大きく `Sections: N` が表示され、現在までに確定したセクション数を常に確認できます。
編集中ポリゴンは赤、確定済みポリゴンは青/赤系で太線表示されます。

重複の整理:
- `overwrite`（既定）: 新しいセクションが既存セクションの重複部分を上書き
- `keep_old`: 既存セクションを優先して新規重複部分を捨てる

`overwrite` を使うと、重複しても領域が1つに整理されるため、境界がきれいに揃いやすいです。

## 操作

- 左クリック: 現在編集中ポリゴンへ頂点追加
- マウスホイール or `+`/`-`: ズーム
- 右ドラッグ or `H/J/K/L` or 矢印キー: パン（表示位置移動）
- `0`: 表示倍率/位置をリセット
- `u`: 現在編集中ポリゴンの最後の頂点をUndo
- `c`: 現在編集中ポリゴンをクリア
- `n`: 現在編集中ポリゴンを確定（3点以上必要）
  - 確定時のセクション名は `section_01`, `section_02`, ... が自動採番されます
- `d`: 最後に確定したセクションを削除
- `o`: 重複ポリシー切替（`overwrite` / `keep_old`）
- `i` または右上の `Help` ボタン: 説明パネルの表示/非表示
- `s`: セクションCSV保存 + ゲート候補CSV保存
- `g`: ゲート候補CSVのみ保存
- `q` or `Esc`: 終了

## 出力CSV形式

```text
# map_section_definition_v1
map_yaml,/absolute/or/relative/path/to/map.yaml
image_width,2048
image_height,2048
section,section_01,10,20,120,20,120,80,10,80
section,section_02,130,20,220,20,220,80,130,80
```

- `section` 行は `section,<name>,u1,v1,u2,v2,...` 形式
- 1セクションあたり3点以上が必要

## ゲートCSV（ハイブリッド用）

`section_localizer` のハイブリッド判定で使うゲート定義は、
`s` または `g` で自動生成されます（`<output>_gates.csv`）。
テンプレート例: `python_ws/map_section_editor/levine_gates.example.csv`

```text
gate,gate_01,section_01,section_02,u0,v0,u1,v1
```

- `u0,v0 -> u1,v1` の向きが遷移方向の基準です
- 右側から左側へ横切ると `section_01 -> section_02`
- ゲート線は、隣接セクションの共通境界点から「直線フィット」で作成されます

## control_filter_config_editor.py

`section_editor.py` で作った `sections_pixels.csv` から、section ごとの
class 切り替えつき `control_filter.param.yaml` を作る terminal editor です。

```bash
cd python_ws
python3 map_section_editor/control_filter_config_editor.py \
  --sections-csv /map/course_a/sections_pixels.csv \
  --output /map/course_a/control_filter.param.yaml \
  --base-config ../ros2_ws/src/control/control_filter/config/control_filter.param.yaml
```

操作の流れ:

- `a`: section へ class を割り当て
- `c`: class ごとの filter / scale パラメータを編集
- `d`: default パラメータを編集
- `p`: YAML preview
- `s`: 保存

`control_filter_node` は `section_classes` に現れる class だけを読み込むため、
未割り当ての class は出力 YAML へ書かれません。
