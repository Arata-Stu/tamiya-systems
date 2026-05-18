# scan_image_projection_cropper

`LaserScan` のクラスタを画像平面へ投影し、横幅は scan から、縦幅は固定値で決めた crop を publish する composable node です。

## 入出力

- Subscribe
  - `scan` (`sensor_msgs/msg/LaserScan`)
  - `image` (`sensor_msgs/msg/Image`)
  - `camera_info` (`sensor_msgs/msg/CameraInfo`)
- Publish
  - `crop/image`
  - `crop/camera_info`
  - `crop/roi`
  - `debug/image` (`debug=true` のとき)

## 前提

- `scan` フレームから `camera_info.header.frame_id` への TF が必要です。
- 歪みを避けるため、`image` と `camera_info` には rectified topic を使う前提がおすすめです。
- debug は `debug/image_input` と `debug/camera_info` を別入力にできるので、crop 本体は rectified、overlay は raw に分けられます。
- 縦方向は `fixed_crop_height_px` で固定、横方向は scan クラスタの投影幅から決めます。

## 使い方

この node 自体で、scan から決めた ROI を固定サイズへ `resize + pad` して publish できます。

```xml
<include file="$(find-pkg-share scan_image_projection_cropper)/launch/scan_image_projection_cropper.launch.xml">
  <arg name="scan_topic" value="/scan"/>
  <arg name="image_topic" value="/camera/left/image_rect"/>
  <arg name="camera_info_topic" value="/camera/left/camera_info_rect"/>
  <arg name="debug_input_image_topic" value="/camera/left/image_raw"/>
  <arg name="debug_input_camera_info_topic" value="/camera/left/camera_info"/>
</include>
```

主なパラメータ:

- `fixed_crop_height_px`: 縦方向の固定 crop 高さ
- `cluster_separation_threshold_m`: scan クラスタを分ける距離しきい値
- `min_cluster_width_m`, `max_cluster_width_m`: 対象クラスタの物理幅レンジ
- `horizontal_padding_px`: 横方向の余白
- `bottom_padding_px`: scan で見えた高さから下側へ足す余白
- `output_image_width_px`, `output_image_height_px`: 出力 crop の固定サイズ
- `output_padding_value`: pad 部分の画素値
- `candidate_selection_mode`: `closest` または `widest`
- `debug`: debug overlay publish の on/off
