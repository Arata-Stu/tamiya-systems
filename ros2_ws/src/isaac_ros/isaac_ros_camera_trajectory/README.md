# Isaac ROS Camera Trajectory

Camera-to-trajectory inference package. It reuses `isaac_ros_dnn_image_encoder` and `isaac_ros_triton`, then decodes the trajectory tensor into `nav_msgs/msg/Path`.

Default output:

- Topic: `/autonomous/trajectory`
- Type: `nav_msgs/msg/Path`
- Frame: `base_link`
- Tensor binding: `trajectory_output`
- Tensor shape: `[20, 2]`

Launch:

```bash
ros2 launch isaac_ros_camera_trajectory isaac_ros_camera_trajectory.launch.xml \
  model_repository_path:=/workspaces/isaac_ros_assets/models/ \
  model_name:=pilotnet_trajectory
```
