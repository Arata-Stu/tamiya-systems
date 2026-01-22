#include "control_filter/control_filter_node.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

ControlFilterNode::ControlFilterNode() : Node("control_filter_node") {
  // --- 制御パラメータの宣言 ---
  this->declare_parameter<std::string>("filter_type", "slew_rate");
  this->declare_parameter<int>("window_size", 5);
  this->declare_parameter<double>("max_accel_slew_rate", 2.0);
  this->declare_parameter<double>("max_steer_slew_rate", 1.5);

  // 初期のパラメータ取得
  this->get_parameter("filter_type", filter_type_);
  this->get_parameter("window_size", window_size_);
  this->get_parameter("max_accel_slew_rate", max_accel_slew_rate_);
  this->get_parameter("max_steer_slew_rate", max_steer_slew_rate_);

  // --- Pub/Sub の設定 ---
  publisher_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_filtered", rclcpp::QoS(10));

  subscription_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_raw", rclcpp::QoS(1),
          std::bind(&ControlFilterNode::TopicCallback, this,
                    std::placeholders::_1));

  // 動的パラメータのコールバック登録
  parameters_callback_handle_ = this->add_on_set_parameters_callback(std::bind(
      &ControlFilterNode::ParametersCallback, this, std::placeholders::_1));

  last_callback_time_ = this->now();
  PrintParameters();
}

void ControlFilterNode::TopicCallback(
    const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  const rclcpp::Time now = this->now();
  const double dt = (now - last_callback_time_).seconds();
  last_callback_time_ = now;

  if (is_first_msg_) {
    prev_accel_ = msg->drive.acceleration;
    prev_steer_ = msg->drive.steering_angle;
    is_first_msg_ = false;
    publisher_->publish(*msg);
    return;
  }

  ackermann_msgs::msg::AckermannDrive filtered_drive = msg->drive;

  if (filter_type_ == "slew_rate") {
    ApplySlewRateFilter(filtered_drive, dt);
  } else if (filter_type_ == "average") {
    accel_buffer_.push_back(msg->drive.acceleration);
    steering_angle_buffer_.push_back(msg->drive.steering_angle);
    ApplyAverageFilter(filtered_drive);
  } else {
    // フィルタなし
  }

  ackermann_msgs::msg::AckermannDriveStamped filtered_msg = *msg;
  filtered_msg.drive = filtered_drive;
  publisher_->publish(filtered_msg);
}

void ControlFilterNode::ApplySlewRateFilter(
    ackermann_msgs::msg::AckermannDrive &drive, double dt) {
  double max_accel_step = max_accel_slew_rate_ * dt;
  double max_steer_step = max_steer_slew_rate_ * dt;

  double accel_diff = drive.acceleration - prev_accel_;
  double steer_diff = drive.steering_angle - prev_steer_;

  if (std::abs(accel_diff) > max_accel_step) {
    drive.acceleration =
        prev_accel_ + std::copysign(max_accel_step, accel_diff);
  }
  if (std::abs(steer_diff) > max_steer_step) {
    drive.steering_angle =
        prev_steer_ + std::copysign(max_steer_step, steer_diff);
  }

  prev_accel_ = drive.acceleration;
  prev_steer_ = drive.steering_angle;
}

void ControlFilterNode::ApplyAverageFilter(
    ackermann_msgs::msg::AckermannDrive &drive) {
  if (static_cast<int>(accel_buffer_.size()) > window_size_) {
    accel_buffer_.pop_front();
    steering_angle_buffer_.pop_front();
  }

  if (accel_buffer_.empty())
    return;

  double accel_sum =
      std::accumulate(accel_buffer_.begin(), accel_buffer_.end(), 0.0);
  double steer_sum = std::accumulate(steering_angle_buffer_.begin(),
                                     steering_angle_buffer_.end(), 0.0);

  drive.acceleration = accel_sum / accel_buffer_.size();
  drive.steering_angle = steer_sum / steering_angle_buffer_.size();
}

rcl_interfaces::msg::SetParametersResult ControlFilterNode::ParametersCallback(
    const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  for (const auto &param : parameters) {
    if (param.get_name() == "filter_type") {
      filter_type_ = param.as_string();
    } else if (param.get_name() == "window_size") {
      window_size_ = param.as_int();
    } else if (param.get_name() == "max_accel_slew_rate") {
      max_accel_slew_rate_ = param.as_double();
    } else if (param.get_name() == "max_steer_slew_rate") {
      max_steer_slew_rate_ = param.as_double();
    }
  }

  PrintParameters();
  return result;
}

void ControlFilterNode::PrintParameters() {
  RCLCPP_INFO(this->get_logger(), "--- Control Filter Parameters ---");
  RCLCPP_INFO(this->get_logger(), "filter_type: %s", filter_type_.c_str());
  RCLCPP_INFO(this->get_logger(), "window_size: %d", window_size_);
  RCLCPP_INFO(this->get_logger(), "max_accel_slew_rate: %.2f",
              max_accel_slew_rate_);
  RCLCPP_INFO(this->get_logger(), "max_steer_slew_rate: %.2f",
              max_steer_slew_rate_);
}