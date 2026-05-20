# isaac_ros_camera_e2e_control

Camera E2E inference pipeline for JetRacer:

1. `isaac_ros_dnn_image_encoder` (Image -> Tensor)
2. `isaac_ros_triton` (Tensor inference)
3. `CameraNetDecoderNode` (Tensor -> `ackermann_msgs/AckermannDriveStamped`)

## Launch

```bash
ros2 launch isaac_ros_camera_e2e_control isaac_ros_camera_e2e.launch.xml \
  container_name:=camera_container \
  model_name:=pilotnet \
  input_image_topic:=/camera/left/image_raw \
  input_camera_info_topic:=/camera/left/camera_info \
  original_image_width:=424 \
  original_image_height:=240 \
  network_image_width:=212 \
  network_image_height:=120
```

学習側の `python_ws/camera_e2e/config/train.yaml` はデフォルトで `force_grayscale_3ch: true` です。
そのため launch 側もデフォルトで `force_grayscale_3ch:=true` とし、入力が color でも mono でも
`gray -> 3ch` にそろえてから encoder に渡します。学習を color 3ch のままで行った場合だけ
`force_grayscale_3ch:=false` を指定してください。

Precision-specific presets:

```bash
# FP32 transport (default behavior)
ros2 launch isaac_ros_camera_e2e_control isaac_ros_camera_e2e_fp32.launch.xml

# FP16 TensorRT engine deployment preset (transport I/O remains FP32)
ros2 launch isaac_ros_camera_e2e_control isaac_ros_camera_e2e_fp16.launch.xml
```

## Tensor naming defaults

- Encoder output tensor name: `input_tensor`
- Triton input tensor name: `input_tensor`
- Triton input binding name: `image_input`
- Triton output tensor name: `output_control`
- Triton output binding name: `control_output`

If you change model or encoder settings, keep these names consistent.
