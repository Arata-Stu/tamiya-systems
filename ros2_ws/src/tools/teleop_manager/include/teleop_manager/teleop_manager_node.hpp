#ifndef TELEOP_MANAGER_NODE_HPP_
#define TELEOP_MANAGER_NODE_HPP_

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

class TeleopManagerNode : public rclcpp::Node
{
public:
  TeleopManagerNode();

private:
  bool check_button_press(bool curr, bool &prev_flag);
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
  void ack_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  void timer_callback();

  // --- Subscribers / Publishers ---
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr ack_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trigger_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr memo_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr steer_offset_inc_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr steer_offset_dec_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr speed_offset_inc_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr speed_offset_dec_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // --- Parameters ---
  double speed_scale_, steer_scale_;
  int joy_button_index_, ack_button_index_;
  int start_button_index_, stop_button_index_;
  int good_button_index_, bad_button_index_;
  int dpad_lr_axis_index_, dpad_ud_axis_index_;
  int axis_speed_index_, axis_steer_index_;
  double timer_hz_, joy_timeout_sec_;

  // --- Runtime State ---
  bool joy_active_, ack_active_;
  double joy_speed_, joy_steer_;
  ackermann_msgs::msg::AckermannDriveStamped last_autonomy_msg_;
  bool ack_received_{false};
  rclcpp::Time last_joy_msg_time_;

  // --- Debounce flags ---
  bool prev_start_pressed_, prev_stop_pressed_;
  bool prev_steer_offset_inc_pressed_;
  bool prev_steer_offset_dec_pressed_;
  bool prev_speed_offset_inc_pressed_;
  bool prev_speed_offset_dec_pressed_;
  bool prev_circle_pressed_, prev_cross_pressed_;
};

#endif  // TELEOP_MANAGER_NODE_HPP_
