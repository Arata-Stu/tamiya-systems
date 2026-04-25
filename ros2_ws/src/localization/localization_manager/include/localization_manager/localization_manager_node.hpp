#ifndef LOCALIZATION_MANAGER_NODE_HPP_
#define LOCALIZATION_MANAGER_NODE_HPP_

#include <string>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

class LocalizationManagerNode : public rclcpp::Node {
public:
  LocalizationManagerNode();

private:
  void trigger_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void localization_result_callback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void timer_callback();
  void request_localization();
  void update_localization_tf(
      const geometry_msgs::msg::PoseWithCovarianceStamped &msg);
  void publish_localization_tf();

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr trigger_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      localization_result_sub_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr localization_trigger_client_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::string localization_trigger_topic_ = "/localization/trigger";
  std::string localization_trigger_service_ = "/trigger_grid_search_localization";
  std::string localization_result_topic_ = "/localization_result";

  double localization_feedback_timeout_sec_ = 0.0;
  bool waiting_localization_result_ = false;
  bool localization_result_timed_out_ = false;
  rclcpp::Time last_localization_trigger_time_;

  bool publish_localization_tf_ = true;
  std::string localization_tf_mode_ = "map_to_odom";
  std::string localization_tf_map_frame_ = "map";
  std::string localization_tf_odom_frame_ = "odom";
  std::string localization_tf_base_frame_ = "base_link";
  double localization_tf_publish_rate_hz_ = 20.0;
  bool has_localization_tf_ = false;
  geometry_msgs::msg::TransformStamped last_localization_tf_;
  rclcpp::Time last_localization_tf_publish_time_;
};

#endif
