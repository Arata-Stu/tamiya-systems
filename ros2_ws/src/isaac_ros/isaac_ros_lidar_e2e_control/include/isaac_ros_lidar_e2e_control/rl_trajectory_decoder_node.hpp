// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros_lidar_e2e_control
{

class RlTrajectoryDecoderNode : public rclcpp::Node
{
public:
  explicit RlTrajectoryDecoderNode(const rclcpp::NodeOptions & options);
  ~RlTrajectoryDecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);
  std::array<double, 2> SampleBezier(
    const std::array<std::array<double, 2>, 3> & control_points,
    double t) const;
  std::array<std::array<double, 2>, 3> DecodeControlPoints(const float action[6]) const;

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;

  std::string output_tensor_name_;
  std::string frame_id_;
  int num_points_;
  std::vector<double> x_anchors_;
  double x_offset_scale_;
  double y_scale_;
  double min_control_dx_;
  double min_forward_distance_;
  bool clip_action_;
};

}  // namespace isaac_ros_lidar_e2e_control
