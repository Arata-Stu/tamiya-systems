#ifndef EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_COMPONENT_HPP_
#define EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_COMPONENT_HPP_

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/bool.hpp"

#include "emergency_controller/emergency_controller_core.hpp"

namespace emergency_controller {

class EmergencyControllerComponent : public rclcpp::Node {
public:
  explicit EmergencyControllerComponent(const rclcpp::NodeOptions &options);

private:
  void LoadParameters();
  void ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void InputDriveCallback(
      const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  void PublishCommand();
  SafetyDecision GetDecisionWithTimeout() const;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      input_drive_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      emergency_drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr emergency_signal_pub_;

  EmergencyControllerCore core_;

  SafetyDecision latest_scan_decision_;
  rclcpp::Time last_scan_received_time_;
  bool has_scan_ = false;

  ackermann_msgs::msg::AckermannDriveStamped latest_input_drive_;
  bool has_input_drive_ = false;

  double scan_timeout_sec_ = 0.25;
  bool stop_on_scan_timeout_ = true;
  bool publish_override_only_ = false;
};

} // namespace emergency_controller

#endif // EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_COMPONENT_HPP_
