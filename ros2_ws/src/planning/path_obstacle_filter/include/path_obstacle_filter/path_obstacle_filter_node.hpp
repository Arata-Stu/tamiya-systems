#ifndef PATH_OBSTACLE_FILTER__PATH_OBSTACLE_FILTER_NODE_HPP_
#define PATH_OBSTACLE_FILTER__PATH_OBSTACLE_FILTER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace path_obstacle_filter {

class PathObstacleFilterNode : public rclcpp::Node {
public:
  explicit PathObstacleFilterNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void LoadParameters();
  void TrackedObjectCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void TrajectoryCallback(const nav_msgs::msg::Path::SharedPtr msg);
  void TimerCallback();

  void PublishDebugMarkers(bool on_path, double obs_x, double obs_y, double distance_m, double lateral_m);

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr tracked_object_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr trajectory_sub_;

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr obstacle_on_path_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr deceleration_requested_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr avoidance_requested_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr following_requested_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr target_speed_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obstacle_distance_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obstacle_lateral_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_markers_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  mutable std::mutex data_mutex_;
  nav_msgs::msg::Odometry::SharedPtr latest_tracked_object_;
  nav_msgs::msg::Path::SharedPtr latest_trajectory_;

  // Parameters
  double forward_distance_m_ = 3.0;
  double lateral_half_width_m_ = 0.25;
  double stopped_speed_threshold_m_s_ = 0.3; // これ未満の絶対速度なら「停止」とみなす
  bool deceleration_on_obstacle_ = true;
  double obstacle_timeout_sec_ = 0.5;
  std::string map_frame_ = "map";
  std::string base_frame_ = "base_link";
  double tf_timeout_sec_ = 0.05;
  bool publish_debug_markers_ = true;
};

}  // namespace path_obstacle_filter

#endif  // PATH_OBSTACLE_FILTER__PATH_OBSTACLE_FILTER_NODE_HPP_
