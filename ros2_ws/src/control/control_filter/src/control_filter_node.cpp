#include "control_filter/ackermann_filter_node.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

const char* INPUT_TOPIC = "/cmd_drive";
const char* OUTPUT_TOPIC = "/cmd_drive_filtered";

AckermannFilterNode::AckermannFilterNode()
: Node("ackermann_filter_node")
{
  // --- Parameter declarations ---
  this->declare_parameter<std::string>("filter_type", "none");
  this->declare_parameter<int>("window_size", 5);
  this->declare_parameter<bool>("use_scale_filter", true);

  this->declare_parameter<double>("straight_steer_threshold", 0.1);
  this->declare_parameter<double>("straight_accel_scale_ratio", 1.0);
  this->declare_parameter<double>("cornering_accel_scale_ratio", 0.5);
  this->declare_parameter<double>("steer_scale_ratio", 1.0);

  // --- Load parameters ---
  this->get_parameter("filter_type", filter_type_);
  this->get_parameter("window_size", window_size_);
  this->get_parameter("use_scale_filter", use_scale_filter_);
  this->get_parameter("straight_steer_threshold", straight_steer_threshold_);
  this->get_parameter("straight_accel_scale_ratio", straight_accel_scale_ratio_);
  this->get_parameter("cornering_accel_scale_ratio", cornering_accel_scale_ratio_);
  this->get_parameter("steer_scale_ratio", steer_scale_ratio_);

  print_parameters();

  // --- Dynamic parameter callback ---
  parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&AckermannFilterNode::parameters_callback, this, std::placeholders::_1));

  // --- ROS2 interfaces ---
  publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(OUTPUT_TOPIC, 10);
  subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
    INPUT_TOPIC, 1, std::bind(&AckermannFilterNode::topic_callback, this, std::placeholders::_1));
}

// ======================================================================
// メインコールバック
// ======================================================================
void AckermannFilterNode::topic_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg)
{
  if (window_size_ <= 1 || filter_type_ == "none") {
    ackermann_msgs::msg::AckermannDriveStamped out_msg = *msg;
    if (use_scale_filter_) {
      apply_advanced_scale_filter(out_msg.drive);
    }
    publisher_->publish(out_msg);
    return;
  }

  // --- バッファ更新 ---
  accel_buffer_.push_back(msg->drive.acceleration);
  steering_angle_buffer_.push_back(msg->drive.steering_angle);
  while (accel_buffer_.size() > static_cast<size_t>(window_size_)) {
    accel_buffer_.pop_front();
    steering_angle_buffer_.pop_front();
  }

  // --- フィルタ処理 ---
  ackermann_msgs::msg::AckermannDrive filtered_drive;
  if (filter_type_ == "average") {
    apply_average_filter(filtered_drive);
  } else if (filter_type_ == "median") {
    apply_median_filter(filtered_drive);
  } else {
    filtered_drive = msg->drive;
  }

  if (use_scale_filter_) {
    apply_advanced_scale_filter(filtered_drive);
  }

  // --- 出力 ---
  ackermann_msgs::msg::AckermannDriveStamped filtered_msg;
  filtered_msg.header = msg->header;
  filtered_msg.drive = filtered_drive;
  publisher_->publish(filtered_msg);
}

// ======================================================================
// 動的パラメータ更新
// ======================================================================
rcl_interfaces::msg::SetParametersResult AckermannFilterNode::parameters_callback(
  const std::vector<rclcpp::Parameter> &parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  for (const auto &param : parameters) {
    const std::string &name = param.get_name();
    if (name == "filter_type") filter_type_ = param.as_string();
    else if (name == "window_size") window_size_ = param.as_int();
    else if (name == "use_scale_filter") use_scale_filter_ = param.as_bool();
    else if (name == "straight_steer_threshold") straight_steer_threshold_ = param.as_double();
    else if (name == "straight_accel_scale_ratio") straight_accel_scale_ratio_ = param.as_double();
    else if (name == "cornering_accel_scale_ratio") cornering_accel_scale_ratio_ = param.as_double();
    else if (name == "steer_scale_ratio") steer_scale_ratio_ = param.as_double();
  }

  print_parameters();
  return result;
}

// ======================================================================
// 平均フィルタ
// ======================================================================
void AckermannFilterNode::apply_average_filter(ackermann_msgs::msg::AckermannDrive &msg)
{
  double accel_sum = std::accumulate(accel_buffer_.begin(), accel_buffer_.end(), 0.0);
  double steer_sum = std::accumulate(steering_angle_buffer_.begin(), steering_angle_buffer_.end(), 0.0);
  msg.acceleration = accel_sum / accel_buffer_.size();
  msg.steering_angle = steer_sum / steering_angle_buffer_.size();
}

// ======================================================================
// 中央値フィルタ
// ======================================================================
void AckermannFilterNode::apply_median_filter(ackermann_msgs::msg::AckermannDrive &msg)
{
  msg.acceleration = calculate_median(accel_buffer_);
  msg.steering_angle = calculate_median(steering_angle_buffer_);
}

double AckermannFilterNode::calculate_median(const std::deque<double>& data)
{
  if (data.empty()) return 0.0;
  std::vector<double> sorted(data.begin(), data.end());
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  return (n % 2 == 0)
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    : sorted[n / 2];
}

// ======================================================================
// 高度スケールフィルタ
// ======================================================================
void AckermannFilterNode::apply_advanced_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
{
  if (std::fabs(msg.steering_angle) < straight_steer_threshold_) {
    msg.acceleration *= straight_accel_scale_ratio_;
  } else {
    msg.acceleration *= cornering_accel_scale_ratio_;
  }
  msg.steering_angle *= steer_scale_ratio_;

  msg.acceleration = std::clamp(msg.acceleration, -1.0f, 1.0f);
  msg.steering_angle = std::clamp(msg.steering_angle, -1.0f, 1.0f);
}

// ======================================================================
// パラメータ出力
// ======================================================================
void AckermannFilterNode::print_parameters()
{
  RCLCPP_INFO(this->get_logger(), "--- Ackermann Filter Node ---");
  RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
  RCLCPP_INFO(this->get_logger(), "Window size: %d", window_size_);
  RCLCPP_INFO(this->get_logger(), "Use scale filter: %s", use_scale_filter_ ? "true" : "false");
  if (use_scale_filter_) {
    RCLCPP_INFO(this->get_logger(), "  Straight steer threshold: %.2f", straight_steer_threshold_);
    RCLCPP_INFO(this->get_logger(), "  Straight accel scale ratio: %.2f", straight_accel_scale_ratio_);
    RCLCPP_INFO(this->get_logger(), "  Cornering accel scale ratio: %.2f", cornering_accel_scale_ratio_);
    RCLCPP_INFO(this->get_logger(), "  Steer scale ratio: %.2f", steer_scale_ratio_);
  }
  RCLCPP_INFO(this->get_logger(), "------------------------------------");
}
