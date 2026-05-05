#pragma once

#include <memory>
#include <string>

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros_camera_trajectory
{

class CameraTrajectoryDecoderNode : public rclcpp::Node
{
public:
  explicit CameraTrajectoryDecoderNode(const rclcpp::NodeOptions & options);
  ~CameraTrajectoryDecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;

  std::string output_tensor_name_;
  std::string output_frame_id_;
  int num_points_;
  bool use_clip_;
  double max_abs_x_;
  double max_abs_y_;
};

}  // namespace isaac_ros_camera_trajectory
