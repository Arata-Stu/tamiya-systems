#include "isaac_ros_camera_trajectory/camera_trajectory_decoder_node.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace isaac_ros_camera_trajectory {

CameraTrajectoryDecoderNode::CameraTrajectoryDecoderNode(const rclcpp::NodeOptions & options)
: Node("camera_trajectory_decoder_node", options)
{
  output_tensor_name_ = declare_parameter<std::string>("output_tensor_name", "trajectory_output");
  output_frame_id_ = declare_parameter<std::string>("output_frame_id", "base_link");
  num_points_ = declare_parameter<int>("num_points", 20);
  use_clip_ = declare_parameter<bool>("use_clip", true);
  max_abs_x_ = declare_parameter<double>("max_abs_x", 20.0);
  max_abs_y_ = declare_parameter<double>("max_abs_y", 10.0);

  if (num_points_ <= 0) {
    throw std::runtime_error("num_points must be positive.");
  }

  pub_path_ = create_publisher<nav_msgs::msg::Path>("autonomous/trajectory", 1);

  using MySubscriber = nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
    nvidia::isaac_ros::nitros::NitrosTensorListView>;

  nitros_sub_ = std::make_shared<MySubscriber>(
    this, "inference_output",
    "nitros_tensor_list_nhwc_rgb_f32",
    std::bind(&CameraTrajectoryDecoderNode::InputCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "CameraTrajectoryDecoderNode initialized (tensor='%s', points=%d, topic='autonomous/trajectory')",
    output_tensor_name_.c_str(), num_points_);
}

CameraTrajectoryDecoderNode::~CameraTrajectoryDecoderNode() = default;

void CameraTrajectoryDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(output_tensor_name_);

  if (tensor.GetBuffer() == nullptr) {
    RCLCPP_WARN(
      get_logger(), "Tensor '%s' not found or buffer is null.", output_tensor_name_.c_str());
    return;
  }

  const size_t element_count = static_cast<size_t>(num_points_) * 2;
  std::vector<float> host_data(element_count, 0.0F);
  cudaError_t cuda_status = cudaMemcpy(
    host_data.data(), tensor.GetBuffer(), element_count * sizeof(float), cudaMemcpyDeviceToHost);

  if (cuda_status != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaMemcpy failed: %s", cudaGetErrorString(cuda_status));
    return;
  }

  nav_msgs::msg::Path path;
  path.header.stamp.sec = msg.GetTimestampSeconds();
  path.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  path.header.frame_id = output_frame_id_;
  path.poses.reserve(static_cast<size_t>(num_points_));

  for (int i = 0; i < num_points_; ++i) {
    float x = host_data[static_cast<size_t>(i) * 2];
    float y = host_data[static_cast<size_t>(i) * 2 + 1];
    if (use_clip_) {
      x = std::clamp(x, static_cast<float>(-max_abs_x_), static_cast<float>(max_abs_x_));
      y = std::clamp(y, static_cast<float>(-max_abs_y_), static_cast<float>(max_abs_y_));
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }

  pub_path_->publish(path);
  RCLCPP_DEBUG(get_logger(), "Published trajectory path with %zu poses", path.poses.size());
}

}  // namespace isaac_ros_camera_trajectory

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_camera_trajectory::CameraTrajectoryDecoderNode)
