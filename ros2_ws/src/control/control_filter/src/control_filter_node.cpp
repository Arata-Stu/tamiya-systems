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
          "/control_cmd_filtered", rclcpp::QoS(10));

  subscription_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "/control_cmd_raw", rclcpp::QoS(1),
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
  } else if (filter_type_ == "median") {
    accel_buffer_.push_back(msg->drive.acceleration);
    steering_angle_buffer_.push_back(msg->drive.steering_angle);
    ApplyMedianFilter(filtered_drive);
  }

  // キューのサイズ維持
  while (accel_buffer_.size() > static_cast<size_t>(window_size_)) {
    accel_buffer_.pop_front();
    steering_angle_buffer_.pop_front();
  }

  auto out_msg = std::make_unique<ackermann_msgs::msg::AckermannDriveStamped>();
  out_msg->header = msg->header;
  out_msg->drive = filtered_drive;
  publisher_->publish(std::move(out_msg));
}

void ControlFilterNode::ApplySlewRateFilter(
    ackermann_msgs::msg::AckermannDrive &msg, double dt) {
  if (dt <= 0.0)
    return;

  const double max_accel_change = max_accel_slew_rate_ * dt;
  const double max_steer_change = max_steer_slew_rate_ * dt;

  // 加速度リミッタ
  const double accel_diff = msg.acceleration - prev_accel_;
  msg.acceleration =
      prev_accel_ + std::clamp(accel_diff, -max_accel_change, max_accel_change);

  // ステアリングリミッタ
  const double steer_diff = msg.steering_angle - prev_steer_;
  msg.steering_angle =
      prev_steer_ + std::clamp(steer_diff, -max_steer_change, max_steer_change);

  prev_accel_ = msg.acceleration;
  prev_steer_ = msg.steering_angle;
}

void ControlFilterNode::ApplyAverageFilter(
    ackermann_msgs::msg::AckermannDrive &msg) {
  if (accel_buffer_.empty())
    return;
  msg.acceleration =
      std::accumulate(accel_buffer_.begin(), accel_buffer_.end(), 0.0) /
      accel_buffer_.size();
  msg.steering_angle = std::accumulate(steering_angle_buffer_.begin(),
                                       steering_angle_buffer_.end(), 0.0) /
                       steering_angle_buffer_.size();
}

void ControlFilterNode::ApplyMedianFilter(
    ackermann_msgs::msg::AckermannDrive &msg) {
  msg.acceleration = CalculateMedian(accel_buffer_);
  msg.steering_angle = CalculateMedian(steering_angle_buffer_);
}

double ControlFilterNode::CalculateMedian(const std::deque<double> &data) {
  if (data.empty())
    return 0.0;
  std::vector<double> sorted(data.begin(), data.end());
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  return (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                      : sorted[n / 2];
}

rcl_interfaces::msg::SetParametersResult ControlFilterNode::ParametersCallback(
    const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  for (const auto &param : parameters) {
    const std::string name = param.get_name();
    if (name == "filter_type") {
      filter_type_ = param.as_string();
    } else if (name == "window_size") {
      window_size_ = param.as_int();
    } else if (name == "max_accel_slew_rate") {
      max_accel_slew_rate_ = param.as_double();
    } else if (name == "max_steer_slew_rate") {
      max_steer_slew_rate_ = param.as_double();
    }
  }
  return result;
}

void ControlFilterNode::PrintParameters() const {
  RCLCPP_INFO(this->get_logger(), "--- Control Filter Node Loaded ---");
  RCLCPP_INFO(this->get_logger(), "Default Input: /cmd_drive");
  RCLCPP_INFO(this->get_logger(), "Default Output: /cmd_drive_filtered");
  RCLCPP_INFO(this->get_logger(), "Active Filter: %s", filter_type_.c_str());
  RCLCPP_INFO(this->get_logger(), "----------------------------------");
}