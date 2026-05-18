#ifndef PERCEPTION_OBJECT_TRACKER__PERCEPTION_OBJECT_TRACKER_NODE_HPP_
#define PERCEPTION_OBJECT_TRACKER__PERCEPTION_OBJECT_TRACKER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace perception_object_tracker {

class PerceptionObjectTrackerNode : public rclcpp::Node {
public:
  explicit PerceptionObjectTrackerNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void LoadParameters();
  void TargetDetectedCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void ObstaclePositionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
  void TimerCallback();

  void PublishDebugMarkers(bool active, double x, double y, double vx, double vy);

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr target_detected_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr obstacle_position_sub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr tracked_object_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_markers_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  mutable std::mutex data_mutex_;
  bool latest_target_detected_ = false;
  rclcpp::Time last_target_detected_time_;
  
  geometry_msgs::msg::PointStamped::SharedPtr latest_obstacle_position_;
  bool new_measurement_ = false;

  // Tracker state (in odom frame)
  bool is_tracking_ = false;
  rclcpp::Time last_update_time_;
  double x_est_ = 0.0;
  double y_est_ = 0.0;
  double vx_est_ = 0.0;
  double vy_est_ = 0.0;

  // Parameters
  bool require_classification_ = true;
  double timeout_sec_ = 1.0;
  double alpha_ = 0.6;
  double beta_ = 0.1;
  std::string odom_frame_ = "odom";
  std::string base_frame_ = "base_link";
  double tf_timeout_sec_ = 0.05;
  double update_rate_hz_ = 20.0;
  bool publish_debug_markers_ = true;
};

}  // namespace perception_object_tracker

#endif  // PERCEPTION_OBJECT_TRACKER__PERCEPTION_OBJECT_TRACKER_NODE_HPP_
