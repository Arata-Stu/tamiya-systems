#include "teleop_manager/teleop_manager_node.hpp"

#include <algorithm>

using std::placeholders::_1;

namespace {
constexpr const char *kDefaultLocalizationTriggerTopic = "/localization/trigger";
}

TeleopManagerNode::TeleopManagerNode() : Node("teleop_manager_node") {
  // Parameters
  TeleopManagerCore::Config config;
  config.speed_scale = this->declare_parameter("speed_scale", 1.0);
  config.steer_scale = this->declare_parameter("steer_scale", 1.0);
  config.joy_button_idx = this->declare_parameter("joy_button_idx", 4);
  config.ack_button_idx = this->declare_parameter("ack_button_idx", 5);
  config.localization_trigger_button_idx =
      this->declare_parameter("localization_trigger_button_idx", 8);
  config.start_button_idx = this->declare_parameter("start_button_idx", 0);
  config.stop_button_idx = this->declare_parameter("stop_button_idx", 1);
  config.good_button_idx = this->declare_parameter("good_button_idx", 2);
  config.bad_button_idx = this->declare_parameter("bad_button_idx", 3);
  config.dpad_lr_axis_idx = this->declare_parameter("dpad_lr_axis_idx", 6);
  config.dpad_ud_axis_idx = this->declare_parameter("dpad_ud_axis_idx", 7);
  config.axis_speed_idx = this->declare_parameter("axis_speed_idx", 1);
  config.axis_steer_idx = this->declare_parameter("axis_steer_idx", 3);
  joy_timeout_sec_ = this->declare_parameter("joy_timeout_sec", 0.5);
  double update_rate_hz = this->declare_parameter("timer_hz", 50.0);

  enable_emergency_override_ =
      this->declare_parameter("enable_emergency_override", true);
  emergency_signal_timeout_sec_ =
      std::max(0.0, this->declare_parameter("emergency_signal_timeout_sec", 0.3));
  emergency_cmd_timeout_sec_ =
      std::max(0.0, this->declare_parameter("emergency_cmd_timeout_sec", 0.3));
  localization_trigger_topic_ = this->declare_parameter<std::string>(
      "localization_trigger_topic", kDefaultLocalizationTriggerTopic);

  core_ = std::make_unique<TeleopManagerCore>(config);

  // Subscribers
  joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "joy", 10, std::bind(&TeleopManagerNode::joy_callback, this, _1));
  ack_sub_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "autonomous/cmd_drive", 10,
          std::bind(&TeleopManagerNode::ack_callback, this, _1));
  emergency_ack_sub_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "emergency/cmd_drive", 10,
          std::bind(&TeleopManagerNode::emergency_ack_callback, this, _1));
  emergency_signal_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "emergency/signal", 10,
      std::bind(&TeleopManagerNode::emergency_signal_callback, this, _1));

  // Publishers
  drive_pub_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "cmd_drive", 10);
  trigger_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/rosbag2_recorder/trigger", 10);
  memo_pub_ = this->create_publisher<std_msgs::msg::String>(
      "/rosbag2_recorder/memo", 10);

  steer_offset_inc_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("steer_offset_inc", 10);
  steer_offset_dec_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("steer_offset_dec", 10);
  speed_offset_inc_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("speed_offset_inc", 10);
  speed_offset_dec_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("speed_offset_dec", 10);
  localization_trigger_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      localization_trigger_topic_, 10);

  // Timer Initialization
  auto timer_period = std::chrono::duration<double>(1.0 / update_rate_hz);
  timer_ =
      this->create_wall_timer(timer_period,
                              std::bind(&TeleopManagerNode::timer_callback, this));

  last_joy_msg_time_ = this->now();
  last_emergency_signal_time_ = this->now();
  last_emergency_msg_time_ = this->now();

  RCLCPP_INFO(this->get_logger(),
              "Emergency override in teleop_manager: %s "
              "(signal timeout=%.2f s, cmd timeout=%.2f s)",
              enable_emergency_override_ ? "enabled" : "disabled",
              emergency_signal_timeout_sec_, emergency_cmd_timeout_sec_);
  RCLCPP_INFO(this->get_logger(), "Localization trigger topic: %s",
              localization_trigger_topic_.c_str());

  // 動的パラメータ変更のコールバックを登録
  param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto &param : parameters) {
          if (param.get_name() == "update_rate_hz" ||
              param.get_name() == "timer_hz") {
            double new_hz = param.as_double();
            if (new_hz > 0.0) {
              if (this->timer_) {
                this->timer_->cancel();
              }
              auto new_period = std::chrono::duration<double>(1.0 / new_hz);
              this->timer_ = this->create_wall_timer(
                  new_period,
                  std::bind(&TeleopManagerNode::timer_callback, this));
              RCLCPP_INFO(this->get_logger(), "Update rate changed to %.2f Hz",
                          new_hz);
            } else {
              result.successful = false;
              result.reason = "timer_hz must be greater than 0.0";
            }
          } else if (param.get_name() == "enable_emergency_override") {
            this->enable_emergency_override_ = param.as_bool();
          } else if (param.get_name() == "emergency_signal_timeout_sec") {
            this->emergency_signal_timeout_sec_ = std::max(0.0, param.as_double());
          } else if (param.get_name() == "emergency_cmd_timeout_sec") {
            this->emergency_cmd_timeout_sec_ = std::max(0.0, param.as_double());
          }
        }
        return result;
      });
}

void TeleopManagerNode::joy_callback(
    const sensor_msgs::msg::Joy::SharedPtr msg) {
  last_joy_msg_time_ = this->now();
  std::vector<int> buttons;
  buttons.reserve(msg->buttons.size());
  for (auto b : msg->buttons) {
    buttons.push_back(b);
  }
  core_->update_joy_input(msg->axes, buttons);
}

void TeleopManagerNode::ack_callback(
    const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  last_autonomy_msg_ = *msg;
}

void TeleopManagerNode::emergency_ack_callback(
    const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  last_emergency_msg_ = *msg;
  last_emergency_msg_time_ = this->now();
  has_emergency_msg_ = true;
}

void TeleopManagerNode::emergency_signal_callback(
    const std_msgs::msg::Bool::SharedPtr msg) {
  emergency_signal_active_ = msg->data;
  last_emergency_signal_time_ = this->now();
}

bool TeleopManagerNode::IsEmergencySignalActive() const {
  if (!enable_emergency_override_ || !emergency_signal_active_) {
    return false;
  }

  if (emergency_signal_timeout_sec_ <= 0.0) {
    return true;
  }

  return (this->now() - last_emergency_signal_time_).seconds() <=
         emergency_signal_timeout_sec_;
}

bool TeleopManagerNode::HasFreshEmergencyCommand() const {
  if (!has_emergency_msg_) {
    return false;
  }

  if (emergency_cmd_timeout_sec_ <= 0.0) {
    return true;
  }

  return (this->now() - last_emergency_msg_time_).seconds() <=
         emergency_cmd_timeout_sec_;
}

void TeleopManagerNode::timer_callback() {
  bool is_timeout =
      (this->now() - last_joy_msg_time_).seconds() > joy_timeout_sec_;

  auto cmd =
      core_->calculate_drive_command(is_timeout, last_autonomy_msg_.drive.speed,
                                     last_autonomy_msg_.drive.steering_angle);

  auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
  drive_msg.header.stamp = this->now();
  drive_msg.header.frame_id = "base_link";
  drive_msg.drive.speed = cmd.speed;
  drive_msg.drive.acceleration = 0.0;
  drive_msg.drive.steering_angle = cmd.steering_angle;
  drive_msg.drive.steering_angle_velocity = cmd.steering_velocity;

  if (IsEmergencySignalActive()) {
    if (HasFreshEmergencyCommand()) {
      drive_msg.drive = last_emergency_msg_.drive;
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Emergency override applied in teleop_manager.");
    } else {
      drive_msg.drive.speed = 0.0;
      drive_msg.drive.acceleration = 0.0;
      drive_msg.drive.steering_angle = 0.0;
      drive_msg.drive.steering_angle_velocity = 0.0;
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "Emergency signal active, but emergency command is stale. "
          "Publishing forced stop.");
    }
  }

  drive_pub_->publish(drive_msg);

  publish_events();
}

void TeleopManagerNode::publish_events() {
  if (core_->pop_start_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    trigger_pub_->publish(msg);
  }
  if (core_->pop_stop_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = false;
    trigger_pub_->publish(msg);
  }

  if (core_->pop_localization_trigger_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    localization_trigger_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Published localization trigger on %s",
                localization_trigger_topic_.c_str());
  }

  auto memo = core_->pop_memo_requested();
  if (memo) {
    std_msgs::msg::String msg;
    msg.data = *memo;
    memo_pub_->publish(msg);
  }

  if (core_->pop_steer_inc_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    steer_offset_inc_pub_->publish(msg);
  }
  if (core_->pop_steer_dec_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    steer_offset_dec_pub_->publish(msg);
  }
  if (core_->pop_speed_inc_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    speed_offset_inc_pub_->publish(msg);
  }
  if (core_->pop_speed_dec_requested()) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    speed_offset_dec_pub_->publish(msg);
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopManagerNode>());
  rclcpp::shutdown();
  return 0;
}
