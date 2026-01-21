#include "teleop_manager/teleop_manager_node.hpp"

using std::placeholders::_1;

TeleopManagerNode::TeleopManagerNode() : Node("teleop_manager_node") {
  // Parameters
  TeleopManagerCore::Config config;
  config.speed_scale = this->declare_parameter("speed_scale", 1.0);
  config.steer_scale = this->declare_parameter("steer_scale", 1.0);
  config.joy_button_idx = this->declare_parameter("joy_button_idx", 4);
  config.ack_button_idx = this->declare_parameter("ack_button_idx", 5);
  config.start_button_idx = this->declare_parameter("start_button_idx", 0);
  config.stop_button_idx = this->declare_parameter("stop_button_idx", 1);
  config.good_button_idx = this->declare_parameter("good_button_idx", 2);
  config.bad_button_idx = this->declare_parameter("bad_button_idx", 3);
  config.dpad_lr_axis_idx = this->declare_parameter("dpad_lr_axis_idx", 6);
  config.dpad_ud_axis_idx = this->declare_parameter("dpad_ud_axis_idx", 7);
  config.axis_speed_idx = this->declare_parameter("axis_speed_idx", 1);
  config.axis_steer_idx = this->declare_parameter("axis_steer_idx", 3);

  joy_timeout_sec_ = this->declare_parameter("joy_timeout", 0.5);

  core_ = std::make_unique<TeleopManagerCore>(config);

  // Subscribers
  joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "joy", 10, std::bind(&TeleopManagerNode::joy_callback, this, _1));
  ack_sub_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "ackermann_cmd", 10,
          std::bind(&TeleopManagerNode::ack_callback, this, _1));

  // Publishers
  drive_pub_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "drive", 10);
  trigger_pub_ = this->create_publisher<std_msgs::msg::Bool>("trigger", 10);
  memo_pub_ = this->create_publisher<std_msgs::msg::String>("memo", 10);

  steer_offset_inc_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("steer_offset_inc", 10);
  steer_offset_dec_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("steer_offset_dec", 10);
  speed_offset_inc_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("speed_offset_inc", 10);
  speed_offset_dec_pub_ =
      this->create_publisher<std_msgs::msg::Bool>("speed_offset_dec", 10);

  // Timer
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&TeleopManagerNode::timer_callback, this));

  last_joy_msg_time_ = this->now();
}

void TeleopManagerNode::joy_callback(
    const sensor_msgs::msg::Joy::SharedPtr msg) {
  last_joy_msg_time_ = this->now();
  // sensor_msgs::msg::Joy buttons are int32, explicitly cast or construct
  // vector if needed. std::vector<int> expected by core.
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

void TeleopManagerNode::timer_callback() {
  bool is_timeout =
      (this->now() - last_joy_msg_time_).seconds() > joy_timeout_sec_;

  auto cmd = core_->calculate_drive_command(
      is_timeout, last_autonomy_msg_.drive.acceleration,
      last_autonomy_msg_.drive.steering_angle);

  auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
  drive_msg.header.stamp = this->now();
  drive_msg.header.frame_id = "base_link";
  drive_msg.drive.acceleration = cmd.acceleration;
  drive_msg.drive.steering_angle = cmd.steering_angle;
  drive_msg.drive.steering_angle_velocity = cmd.steering_velocity;

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