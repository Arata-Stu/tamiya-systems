# map_section_editor

2D map 向けの GUI ツール置き場です。

- `section_editor.py`: `map.yaml` 上でセクションをポリゴン分割し、ROS2ノード用のセクション定義CSVを作ります。
- `map_cleanup_editor.py`: centerline 前処理用。PNG/PGM を黒塗り/白戻しして、cleaned PNG を保存します。

## map_cleanup_editor.py

分岐やノイズで centerline がうまく引けない地図を、人手で少し整えてから
`generate_centerline.py` / `generate_raceline.py` に渡したいときのツールです。

```bash
cd python_ws
python3 map_section_editor/map_cleanup_editor.py \
  --input ../ros2_ws/src/launch/system_launch/config/simulator/levine.png \
  --output ./map_section_editor/levine_centerline_input.png
```

操作:

- 左ドラッグ: 現在のブラシ色で描画
- `b`: 黒で塗る
- `e`: 白で戻す
- `i` または右上の `Help` ボタン: 説明パネルの表示/非表示
- `u`: undo
- `r`: 今回開いた状態に戻す
- `R`: 元の入力画像に戻す
- `s`: 保存
- `[` / `]` または `,` / `.` または右上の `-` / `+` ボタン: ブラシサイズ変更
- ホイール / `+` / `-`: ズーム
- 右ドラッグ or `H/J/K/L` or 矢印キー: パン
- `q` or `Esc`: 終了

`create_2d_map_from_bag.sh` からは `--edit-map` または `--map-edit-mode auto`
で呼び出せます。section editor は同スクリプトの転送前メニューからも開けます。

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
