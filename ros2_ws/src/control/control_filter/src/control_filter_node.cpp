#include "control_filter/control_filter_node.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

ControlFilterNode::ControlFilterNode() : Node("control_filter_node") {
  // --- 制御パラメータの宣言 ---
  this->declare_parameter<std::string>("filter_type", "slew_rate");
  this->declare_parameter<int>("window_size", 5);
  this->declare_parameter<double>("max_speed_slew_rate", 2.0);
  this->declare_parameter<double>("max_steer_slew_rate", 1.5);

  // Scale Filter パラメータ
  this->declare_parameter<bool>("use_scale_filter", true);
  this->declare_parameter<double>("straight_steer_threshold", 0.1);
  this->declare_parameter<double>("straight_speed_scale_ratio", 1.0);
  this->declare_parameter<double>("cornering_speed_scale_ratio", 0.5);
  this->declare_parameter<double>("steer_scale_ratio", 1.0);

  // 初期のパラメータ取得
  this->get_parameter("filter_type", filter_type_);
  this->get_parameter("window_size", window_size_);
  this->get_parameter("max_speed_slew_rate", max_speed_slew_rate_);
  this->get_parameter("max_steer_slew_rate", max_steer_slew_rate_);

  this->get_parameter("use_scale_filter", use_scale_filter_);
  this->get_parameter("straight_steer_threshold", straight_steer_threshold_);
  this->get_parameter("straight_speed_scale_ratio",
                      straight_speed_scale_ratio_);
  this->get_parameter("cornering_speed_scale_ratio",
                      cornering_speed_scale_ratio_);
  this->get_parameter("steer_scale_ratio", steer_scale_ratio_);

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
    prev_speed_ = msg->drive.speed;
    prev_steer_ = msg->drive.steering_angle;
    is_first_msg_ = false;

    ackermann_msgs::msg::AckermannDrive filtered_drive = msg->drive;
    if (use_scale_filter_) {
      ApplyAdvancedScaleFilter(filtered_drive);
    }

    ackermann_msgs::msg::AckermannDriveStamped out_msg = *msg;
    out_msg.drive = filtered_drive;
    publisher_->publish(out_msg);
    return;
  }

  ackermann_msgs::msg::AckermannDrive filtered_drive = msg->drive;

  // 1. Smoothing / Slew Rate Logic
  if (filter_type_ == "slew_rate") {
    ApplySlewRateFilter(filtered_drive, dt);
  } else if (filter_type_ == "average") {
    speed_buffer_.push_back(msg->drive.speed);
    steering_angle_buffer_.push_back(msg->drive.steering_angle);
    ApplyAverageFilter(filtered_drive);
  } else {
    // フィルタなし (raw)
  }

  // 2. Scale Filter Logic (Optional)
  if (use_scale_filter_) {
    ApplyAdvancedScaleFilter(filtered_drive);
  }

  ackermann_msgs::msg::AckermannDriveStamped filtered_msg = *msg;
  filtered_msg.drive = filtered_drive;
  publisher_->publish(filtered_msg);
}

void ControlFilterNode::ApplySlewRateFilter(
    ackermann_msgs::msg::AckermannDrive &drive, double dt) {
  double max_speed_step = max_speed_slew_rate_ * dt;
  double max_steer_step = max_steer_slew_rate_ * dt;

  double speed_diff = drive.speed - prev_speed_;
  double steer_diff = drive.steering_angle - prev_steer_;

  if (std::abs(speed_diff) > max_speed_step) {
    drive.speed = prev_speed_ + std::copysign(max_speed_step, speed_diff);
  }
  if (std::abs(steer_diff) > max_steer_step) {
    drive.steering_angle =
        prev_steer_ + std::copysign(max_steer_step, steer_diff);
  }

  prev_speed_ = drive.speed;
  prev_steer_ = drive.steering_angle;
}

void ControlFilterNode::ApplyAverageFilter(
    ackermann_msgs::msg::AckermannDrive &drive) {
  if (static_cast<int>(speed_buffer_.size()) > window_size_) {
    speed_buffer_.pop_front();
    steering_angle_buffer_.pop_front();
  }

  if (speed_buffer_.empty())
    return;

  double speed_sum =
      std::accumulate(speed_buffer_.begin(), speed_buffer_.end(), 0.0);
  double steer_sum = std::accumulate(steering_angle_buffer_.begin(),
                                     steering_angle_buffer_.end(), 0.0);

  drive.speed = speed_sum / speed_buffer_.size();
  drive.steering_angle = steer_sum / steering_angle_buffer_.size();
}

void ControlFilterNode::ApplyAdvancedScaleFilter(
    ackermann_msgs::msg::AckermannDrive &drive) {
  if (std::fabs(drive.steering_angle) < straight_steer_threshold_) {
    drive.speed *= straight_speed_scale_ratio_;
  } else {
    drive.speed *= cornering_speed_scale_ratio_;
  }
  drive.steering_angle *= steer_scale_ratio_;

  // Note: Removed hard clamping to -1.0/1.0 based on user preference to respect
  // physical values. If clamping is needed, it should be a separate parameter.
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
    } else if (param.get_name() == "max_speed_slew_rate") {
      max_speed_slew_rate_ = param.as_double();
    } else if (param.get_name() == "max_steer_slew_rate") {
      max_steer_slew_rate_ = param.as_double();
    } else if (param.get_name() == "use_scale_filter") {
      use_scale_filter_ = param.as_bool();
    } else if (param.get_name() == "straight_steer_threshold") {
      straight_steer_threshold_ = param.as_double();
    } else if (param.get_name() == "straight_speed_scale_ratio") {
      straight_speed_scale_ratio_ = param.as_double();
    } else if (param.get_name() == "cornering_speed_scale_ratio") {
      cornering_speed_scale_ratio_ = param.as_double();
    } else if (param.get_name() == "steer_scale_ratio") {
      steer_scale_ratio_ = param.as_double();
    }
  }

  PrintParameters();
  return result;
}

void ControlFilterNode::PrintParameters() const {
  RCLCPP_INFO(this->get_logger(), "--- Control Filter Parameters ---");
  RCLCPP_INFO(this->get_logger(), "filter_type: %s", filter_type_.c_str());
  RCLCPP_INFO(this->get_logger(), "window_size: %d", window_size_);
  RCLCPP_INFO(this->get_logger(), "max_speed_slew_rate: %.2f",
              max_speed_slew_rate_);
  RCLCPP_INFO(this->get_logger(), "max_steer_slew_rate: %.2f",
              max_steer_slew_rate_);

  RCLCPP_INFO(this->get_logger(), "use_scale_filter: %s",
              use_scale_filter_ ? "true" : "false");
  if (use_scale_filter_) {
    RCLCPP_INFO(this->get_logger(), "  straight_steer_threshold: %.2f",
                straight_steer_threshold_);
    RCLCPP_INFO(this->get_logger(), "  straight_speed_scale: %.2f",
                straight_speed_scale_ratio_);
    RCLCPP_INFO(this->get_logger(), "  cornering_speed_scale: %.2f",
                cornering_speed_scale_ratio_);
    RCLCPP_INFO(this->get_logger(), "  steer_scale: %.2f", steer_scale_ratio_);
  }
}