#include "emergency_controller/emergency_controller_component.hpp"

#include <algorithm>
#include <limits>

#include "rclcpp_components/register_node_macro.hpp"

namespace emergency_controller {

namespace {

double ClampPositive(double value, double fallback) {
  if (value <= 0.0) {
    return fallback;
  }
  return value;
}

double ClampNonNegative(double value) {
  return std::max(0.0, value);
}

} // namespace

EmergencyControllerComponent::EmergencyControllerComponent(
    const rclcpp::NodeOptions &options)
    : Node("emergency_controller_node", options),
      last_scan_received_time_(this->now()) {
  LoadParameters();

  emergency_drive_pub_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "emergency/cmd_drive", rclcpp::QoS(10));
  emergency_signal_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("emergency/signal",
                                                  rclcpp::QoS(10));

  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      std::bind(&EmergencyControllerComponent::ScanCallback, this,
                std::placeholders::_1));

  input_drive_sub_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "input/cmd_drive", rclcpp::QoS(10),
          std::bind(&EmergencyControllerComponent::InputDriveCallback, this,
                    std::placeholders::_1));

  const auto &params = core_.GetParams();
  RCLCPP_INFO(this->get_logger(),
              "Emergency controller initialized "
              "(front=%.2f deg, stop=%.2f m, slow=%.2f m, timeout=%.2f s)",
              params.front_angle_deg, params.stop_distance_m,
              params.slow_distance_m, scan_timeout_sec_);
}

void EmergencyControllerComponent::LoadParameters() {
  EmergencyParams params;

  params.front_angle_deg =
      ClampPositive(this->declare_parameter<double>("front_angle_deg", 55.0),
                    55.0);
  params.side_angle_deg =
      ClampPositive(this->declare_parameter<double>("side_angle_deg", 75.0),
                    75.0);

  params.stop_distance_m =
      ClampPositive(this->declare_parameter<double>("stop_distance_m", 0.35),
                    0.35);
  params.slow_distance_m =
      ClampPositive(this->declare_parameter<double>("slow_distance_m", 0.80),
                    0.80);

  params.slow_speed_mps =
      ClampNonNegative(this->declare_parameter<double>("slow_speed_mps", 0.40));
  params.stop_speed_mps =
      ClampNonNegative(this->declare_parameter<double>("stop_speed_mps", 0.0));

  params.max_override_steer = ClampPositive(
      this->declare_parameter<double>("max_override_steer", 0.40), 0.40);
  params.min_turn_steer = ClampPositive(
      this->declare_parameter<double>("min_turn_steer", 0.15), 0.15);
  params.steer_bias_gain = ClampPositive(
      this->declare_parameter<double>("steer_bias_gain", 1.20), 1.20);
  params.max_considered_range_m = ClampPositive(
      this->declare_parameter<double>("max_considered_range_m", 8.0), 8.0);

  scan_timeout_sec_ =
      ClampPositive(this->declare_parameter<double>("scan_timeout_sec", 0.25),
                    0.25);
  stop_on_scan_timeout_ =
      this->declare_parameter<bool>("stop_on_scan_timeout", true);
  publish_override_only_ =
      this->declare_parameter<bool>("publish_override_only", false);

  core_.SetParams(params);
}

void EmergencyControllerComponent::ScanCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  LidarScan scan;
  scan.ranges = msg->ranges;
  scan.angle_min = msg->angle_min;
  scan.angle_increment = msg->angle_increment;
  scan.range_min = msg->range_min;
  scan.range_max = msg->range_max;

  latest_scan_decision_ = core_.EvaluateScan(scan);
  has_scan_ = true;
  last_scan_received_time_ = this->now();

  // Publish immediately when scan enters override zone to reduce reaction delay.
  if (has_input_drive_ && latest_scan_decision_.override_active) {
    PublishCommand();
  }
}

void EmergencyControllerComponent::InputDriveCallback(
    const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  latest_input_drive_ = *msg;
  has_input_drive_ = true;
  PublishCommand();
}

void EmergencyControllerComponent::PublishCommand() {
  if (!has_input_drive_) {
    return;
  }

  const SafetyDecision decision = GetDecisionWithTimeout();
  bool was_overridden = false;

  ackermann_msgs::msg::AckermannDriveStamped out = latest_input_drive_;
  out.header.stamp = this->now();
  out.drive = core_.ApplyOverride(latest_input_drive_.drive, decision,
                                  &was_overridden);

  if (publish_override_only_ && !was_overridden) {
    std_msgs::msg::Bool signal_msg;
    signal_msg.data = false;
    emergency_signal_pub_->publish(signal_msg);
    return;
  }

  emergency_drive_pub_->publish(out);
  std_msgs::msg::Bool signal_msg;
  signal_msg.data = was_overridden;
  emergency_signal_pub_->publish(signal_msg);

  if (!was_overridden) {
    return;
  }

  if (decision.stop_zone) {
    if (decision.scan_valid) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "Emergency STOP override active (front distance=%.2f m)",
          decision.min_front_distance);
    } else {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "Emergency STOP override active (scan timeout)");
    }
    return;
  }

  RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Emergency SLOW override active (front distance=%.2f m, steer bias=%.2f)",
      decision.min_front_distance, decision.steer_bias);
}

SafetyDecision EmergencyControllerComponent::GetDecisionWithTimeout() const {
  if (!has_scan_) {
    SafetyDecision decision;
    if (stop_on_scan_timeout_) {
      decision.override_active = true;
      decision.stop_zone = true;
    }
    return decision;
  }

  const double dt = (this->now() - last_scan_received_time_).seconds();
  if (dt <= scan_timeout_sec_) {
    return latest_scan_decision_;
  }

  if (!stop_on_scan_timeout_) {
    return latest_scan_decision_;
  }

  SafetyDecision timeout_decision;
  timeout_decision.override_active = true;
  timeout_decision.stop_zone = true;
  timeout_decision.min_front_distance = std::numeric_limits<double>::infinity();
  return timeout_decision;
}

} // namespace emergency_controller

RCLCPP_COMPONENTS_REGISTER_NODE(emergency_controller::EmergencyControllerComponent)
