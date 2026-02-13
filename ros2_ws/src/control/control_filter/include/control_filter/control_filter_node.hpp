#ifndef CONTROL_FILTER_NODE_HPP_
#define CONTROL_FILTER_NODE_HPP_

#include <deque>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"

struct ControlFilterParams {
  std::string filter_type = "slew_rate";
  int window_size = 5;
  double max_speed_slew_rate = 2.0; // [unit/s]
  double max_steer_slew_rate = 1.5; // [unit/s]

  // Scale Filter Parameters
  bool use_scale_filter = true;
  double straight_steer_threshold = 0.1;
  double straight_speed_scale_ratio = 1.0;
  double cornering_speed_scale_ratio = 0.5;
  double steer_scale_ratio = 1.0;
};

class ControlFilterCore {
public:
  ControlFilterCore();
  ~ControlFilterCore() = default;

  void SetParams(const ControlFilterParams &params);
  const ControlFilterParams &GetParams() const;

  ackermann_msgs::msg::AckermannDrive
  Filter(const ackermann_msgs::msg::AckermannDrive &raw_drive, double dt);

private:
  void ApplySlewRateFilter(ackermann_msgs::msg::AckermannDrive &drive,
                           double dt);
  void ApplyAverageFilter(ackermann_msgs::msg::AckermannDrive &drive);
  void ApplyAdvancedScaleFilter(ackermann_msgs::msg::AckermannDrive &drive);

  ControlFilterParams params_;

  // Filter States
  double prev_speed_ = 0.0;
  double prev_steer_ = 0.0;
  bool is_first_msg_ = true;

  // Buffers for MA
  std::deque<double> speed_buffer_;
  std::deque<double> steering_angle_buffer_;
};

class ControlFilterNode : public rclcpp::Node {
public:
  ControlFilterNode();

private:
  void TopicCallback(
      const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  rcl_interfaces::msg::SetParametersResult
  ParametersCallback(const std::vector<rclcpp::Parameter> &parameters);

  void PrintParameters() const;

  // ROS 2 interfaces
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_;
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;

  ControlFilterCore core_;
  rclcpp::Time last_callback_time_;
};

#endif // CONTROL_FILTER_NODE_HPP_