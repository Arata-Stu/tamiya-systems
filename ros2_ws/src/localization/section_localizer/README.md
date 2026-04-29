# section_localizer

`map.yaml` + セクションCSV（任意でゲートCSV）から、現在の自己位置がどのセクションにいるかを判定するROS2ノードです。
ハイブリッド方式で、以下を組み合わせます。

- フォールバック: セクションポリゴン内判定（常時復帰可能）
- 補助遷移: ゲート交差で意図したセクション遷移

## 入力

- TF: `map_frame` -> `base_frame`（既定: `map` -> `base_link`）
- map metadata: `map_yaml_path`
- section definition CSV: `section_definition_path`
- gate definition CSV（任意）: `gate_definition_path`

## 起動例

```bash
ros2 launch section_localizer section_localizer.launch.xml \
  map_yaml_path:=/path/to/levine.yaml \
  section_definition_path:=/path/to/levine_sections.csv \
  gate_definition_path:=/path/to/levine_gates.csv \
  debug_mode:=true
```

## 出力

- `std_msgs/String` (`current_section_topic`): 現在セクション名（未所属は `unknown`）
- debug mode時:
  - `visualization_msgs/MarkerArray` (`marker_topic`): 全セクション
    - セクション境界
    - ゲート線と `from -> to` ラベル
  - `visualization_msgs/Marker` (`current_marker_topic`): 現在セクション強調

## セクションCSV形式

```text
# map_section_definition_v1
image_width,2048
image_height,2048
section,section_01,u1,v1,u2,v2,u3,v3,...
section,section_02,u1,v1,u2,v2,u3,v3,...
```

## ゲートCSV形式

```text
# map_section_gate_definition_v1
gate,gate_01,section_01,section_02,u0,v0,u1,v1
gate,gate_02,section_02,section_03,u0,v0,u1,v1
```

- 1行ごとに1つのゲート線分
- 向き `u0,v0 -> u1,v1` に対して
  - 線分の右側から左側へ横切ると `from_section -> to_section`
  - `enable_reverse_gate_transition=true` なら逆向き遷移も許可

## ハイブリッド挙動

1. ゲート交差を検出したらセクション遷移
2. ただし取り逃しやノイズ時は、ポリゴン判定を `fallback_confirm_count` 回連続一致で復帰
3. これにより「ゲート取り逃しで復帰不能」を防止

## 座標変換

ピクセル座標 `(u, v)` から map座標 `(x, y)` への変換式:

```text
gx = u + 0.5
gy = H - v - 0.5
x = ox + res * (gx*cos(theta) - gy*sin(theta))
y = oy + res * (gx*sin(theta) + gy*cos(theta))
```

`H` は map画像高さ、`ox/oy/theta/res` は `map.yaml` の `origin/resolution` です。
