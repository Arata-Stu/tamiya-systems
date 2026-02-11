#ifndef CONTROL_FILTER_NODE_HPP_
#define CONTROL_FILTER_NODE_HPP_

#include <deque>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"

class ControlFilterNode : public rclcpp::Node {
public:
  ControlFilterNode();

private:
  void TopicCallback(
      const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  rcl_interfaces::msg::SetParametersResult
  ParametersCallback(const std::vector<rclcpp::Parameter> &parameters);

  void ApplySlewRateFilter(ackermann_msgs::msg::AckermannDrive &msg, double dt);
  void ApplyAverageFilter(ackermann_msgs::msg::AckermannDrive &msg);
  void ApplyAdvancedScaleFilter(ackermann_msgs::msg::AckermannDrive &msg);

  void PrintParameters() const;

  // ROS 2 interfaces
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_;
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;

  // Node Parameters
  std::string filter_type_;
  int window_size_;
  double max_speed_slew_rate_; // [unit/s]
  double max_steer_slew_rate_; // [unit/s]

  // Scale Filter Parameters
  bool use_scale_filter_;
  double straight_steer_threshold_;
  double straight_speed_scale_ratio_;
  double cornering_speed_scale_ratio_;
  double steer_scale_ratio_;

  // Filter States
  rclcpp::Time last_callback_time_;
  double prev_speed_ = 0.0;
  double prev_steer_ = 0.0;
  bool is_first_msg_ = true;

  // Buffers for MA
  std::deque<double> speed_buffer_;
  std::deque<double> steering_angle_buffer_;
};

#endif // CONTROL_FILTER_NODE_HPP_