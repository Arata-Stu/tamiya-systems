# isaac_ros_scan_image_classifier

Triton の出力テンソルを 3 クラス分類結果へデコードする composable node です。

既定クラス:

- `rc_car`
- `duct_tube`
- `background`

主な出力トピック:

- `classification/label`
- `classification/class_id`
- `classification/confidence`
- `classification/scores`
- `classification/target_detected`
