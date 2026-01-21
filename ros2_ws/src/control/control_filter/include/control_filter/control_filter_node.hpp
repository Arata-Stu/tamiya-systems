#ifndef ACKERMANN_FILTER_NODE_HPP_
#define ACKERMANN_FILTER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <deque>
#include <string>
#include <vector>
#include "rcl_interfaces/msg/set_parameters_result.hpp"

class AckermannFilterNode : public rclcpp::Node
{
public:
  AckermannFilterNode();

private:
  void topic_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> &parameters);

  void apply_average_filter(ackermann_msgs::msg::AckermannDrive &msg);
  void apply_median_filter(ackermann_msgs::msg::AckermannDrive &msg);
  double calculate_median(const std::deque<double>& data);
  void apply_advanced_scale_filter(ackermann_msgs::msg::AckermannDrive &msg);

  void print_parameters();

  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;

  // Parameters
  std::string filter_type_;  
  int window_size_;
  bool use_scale_filter_;

  double straight_steer_threshold_;
  double straight_accel_scale_ratio_;
  double cornering_accel_scale_ratio_;
  double steer_scale_ratio_;

  // Buffers
  std::deque<double> accel_buffer_;
  std::deque<double> steering_angle_buffer_;
};

#endif  // ACKERMANN_FILTER_NODE_HPP_
