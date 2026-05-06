// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_lidar_e2e_control/rl_trajectory_decoder_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace isaac_ros_lidar_e2e_control
{

RlTrajectoryDecoderNode::RlTrajectoryDecoderNode(const rclcpp::NodeOptions & options)
: Node("rl_trajectory_decoder_node", options)
{
  output_tensor_name_ =
    declare_parameter<std::string>("output_tensor_name", "output_trajectory_action");
  frame_id_ = declare_parameter<std::string>("frame_id", "base_link");
  num_points_ = declare_parameter<int>("num_points", 20);
  x_anchors_ = declare_parameter<std::vector<double>>(
    "x_anchors", std::vector<double>{0.4, 1.0, 1.8});
  x_offset_scale_ = declare_parameter<double>("x_offset_scale", 0.25);
  y_scale_ = declare_parameter<double>("y_scale", 0.8);
  min_control_dx_ = declare_parameter<double>("min_control_dx", 0.05);
  min_forward_distance_ = declare_parameter<double>("min_forward_distance", 0.05);
  clip_action_ = declare_parameter<bool>("clip_action", true);

  if (num_points_ < 2) {
    RCLCPP_WARN(get_logger(), "num_points must be >= 2. Clamping to 2.");
    num_points_ = 2;
  }
  if (x_anchors_.size() != 3) {
    RCLCPP_WARN(
      get_logger(),
      "x_anchors must contain exactly 3 values. Falling back to [0.4, 1.0, 1.8].");
    x_anchors_ = {0.4, 1.0, 1.8};
  }

  trajectory_pub_ = create_publisher<nav_msgs::msg::Path>("autonomous/trajectory", 1);

  using MySubscriber = nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
    nvidia::isaac_ros::nitros::NitrosTensorListView>;

  nitros_sub_ = std::make_shared<MySubscriber>(
    this, "inference_output",
    "nitros_tensor_list_nhwc_rgb_f32",
    std::bind(&RlTrajectoryDecoderNode::InputCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "RlTrajectoryDecoderNode initialized (tensor='%s' -> topic='autonomous/trajectory')",
    output_tensor_name_.c_str());
}

RlTrajectoryDecoderNode::~RlTrajectoryDecoderNode() = default;

std::array<std::array<double, 2>, 3> RlTrajectoryDecoderNode::DecodeControlPoints(
  const float action[6]) const
{
  std::array<double, 6> a{};
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>(action[i]);
    if (clip_action_) {
      a[i] = std::clamp(a[i], -1.0, 1.0);
    }
  }

  const double x1 = std::max(x_anchors_[0] + a[0] * x_offset_scale_, min_forward_distance_);
  const double x2 = std::max(x_anchors_[1] + a[2] * x_offset_scale_, x1 + min_control_dx_);
  const double x3 = std::max(x_anchors_[2] + a[4] * x_offset_scale_, x2 + min_control_dx_);

  return {{
    {{x1, a[1] * y_scale_}},
    {{x2, a[3] * y_scale_}},
    {{x3, a[5] * y_scale_}},
  }};
}

std::array<double, 2> RlTrajectoryDecoderNode::SampleBezier(
  const std::array<std::array<double, 2>, 3> & control_points,
  double t) const
{
  const double omt = 1.0 - t;
  const double b1 = 3.0 * omt * omt * t;
  const double b2 = 3.0 * omt * t * t;
  const double b3 = t * t * t;

  return {{
    b1 * control_points[0][0] + b2 * control_points[1][0] + b3 * control_points[2][0],
    b1 * control_points[0][1] + b2 * control_points[1][1] + b3 * control_points[2][1],
  }};
}

void RlTrajectoryDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(output_tensor_name_);

  if (tensor.GetBuffer() == nullptr) {
    RCLCPP_WARN(
      get_logger(), "Tensor '%s' not found or buffer is null.", output_tensor_name_.c_str());
    return;
  }

  float host_data[6];
  const cudaError_t cuda_status = cudaMemcpy(
    host_data, tensor.GetBuffer(), 6 * sizeof(float), cudaMemcpyDeviceToHost);

  if (cuda_status != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaMemcpy failed: %s", cudaGetErrorString(cuda_status));
    return;
  }

  const auto control_points = DecodeControlPoints(host_data);

  nav_msgs::msg::Path path;
  path.header.stamp.sec = msg.GetTimestampSeconds();
  path.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  path.header.frame_id = frame_id_;
  path.poses.reserve(static_cast<size_t>(num_points_));

  for (int i = 0; i < num_points_; ++i) {
    const double t = static_cast<double>(i + 1) / static_cast<double>(num_points_);
    const auto point = SampleBezier(control_points, t);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = point[0];
    pose.pose.position.y = point[1];
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }

  trajectory_pub_->publish(path);

  RCLCPP_DEBUG(
    get_logger(),
    "Published RL trajectory: p1=(%.3f, %.3f), p2=(%.3f, %.3f), p3=(%.3f, %.3f)",
    control_points[0][0], control_points[0][1],
    control_points[1][0], control_points[1][1],
    control_points[2][0], control_points[2][1]);
}

}  // namespace isaac_ros_lidar_e2e_control

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_lidar_e2e_control::RlTrajectoryDecoderNode)
