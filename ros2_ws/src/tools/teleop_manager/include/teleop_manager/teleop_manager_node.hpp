#ifndef TELEOP_MANAGER_NODE_HPP_
#define TELEOP_MANAGER_NODE_HPP_

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/empty.hpp"
#include "teleop_manager_core.hpp"

class TeleopManagerNode : public rclcpp::Node {
public:
  TeleopManagerNode();

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
  void
  ack_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  void emergency_ack_callback(
      const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  void emergency_signal_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void timer_callback();
  void publish_events();
  void call_localization_trigger();
  bool IsEmergencySignalActive() const;
  bool HasFreshEmergencyCommand() const;

  std::unique_ptr<TeleopManagerCore> core_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      ack_sub_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      emergency_ack_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_signal_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trigger_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr memo_pub_;

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr steer_offset_inc_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr steer_offset_dec_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr speed_offset_inc_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr speed_offset_dec_pub_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr localization_trigger_client_;

  rclcpp::TimerBase::SharedPtr timer_;

  double joy_timeout_sec_;
  rclcpp::Time last_joy_msg_time_;
  ackermann_msgs::msg::AckermannDriveStamped last_autonomy_msg_;
  ackermann_msgs::msg::AckermannDriveStamped last_emergency_msg_;
  rclcpp::Time last_emergency_signal_time_;
  rclcpp::Time last_emergency_msg_time_;
  bool emergency_signal_active_ = false;
  bool has_emergency_msg_ = false;
  bool enable_emergency_override_ = true;
  double emergency_signal_timeout_sec_ = 0.3;
  double emergency_cmd_timeout_sec_ = 0.3;
};

#endif
