#include "teleop_manager/teleop_manager_node.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <utility>

using namespace std::chrono_literals;
using std::placeholders::_1;

TeleopManagerNode::TeleopManagerNode()
: Node("teleop_manager_node"),
  joy_active_(false),
  ack_active_(false),
  joy_speed_(0.0),
  joy_steer_(0.0),
  prev_start_pressed_(false),
  prev_stop_pressed_(false),
  prev_steer_offset_inc_pressed_(false),
  prev_steer_offset_dec_pressed_(false),
  prev_speed_offset_inc_pressed_(false),
  prev_speed_offset_dec_pressed_(false),
  prev_good_pressed_(false),
  prev_bad_pressed_(false)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));

  declare_parameter<double>("speed_scale", 1.0);
  declare_parameter<double>("steer_scale", 1.0);
  declare_parameter<int>("joy_button_index",   2);
  declare_parameter<int>("ack_button_index",   3);
  declare_parameter<int>("start_button_index", 9);
  declare_parameter<int>("stop_button_index",  8);
  declare_parameter<int>("good_button_index", 1);
  declare_parameter<int>("bad_button_index", 0);
  declare_parameter<double>("timer_hz", 40.0);
  declare_parameter<double>("joy_timeout_sec", 0.5);
  declare_parameter<int>("dpad_lr_axis_index", 6);
  declare_parameter<int>("dpad_ud_axis_index", 7);
  declare_parameter<int>("axis_speed_index", 1);
  declare_parameter<int>("axis_steer_index", 3);

  get_parameter("speed_scale", speed_scale_);
  get_parameter("steer_scale", steer_scale_);
  get_parameter("joy_button_index", joy_button_index_);
  get_parameter("ack_button_index", ack_button_index_);
  get_parameter("start_button_index", start_button_index_);
  get_parameter("stop_button_index", stop_button_index_);
  get_parameter("good_button_index", good_button_index_);
  get_parameter("bad_button_index", bad_button_index_);
  get_parameter("timer_hz", timer_hz_);
  get_parameter("joy_timeout_sec", joy_timeout_sec_);
  get_parameter("dpad_lr_axis_index", dpad_lr_axis_index_);
  get_parameter("dpad_ud_axis_index", dpad_ud_axis_index_);
  get_parameter("axis_speed_index", axis_speed_index_);
  get_parameter("axis_steer_index", axis_steer_index_);

  last_autonomy_msg_.drive.steering_angle = 0.0;
  last_autonomy_msg_.drive.acceleration = 0.0;
  last_joy_msg_time_ = this->get_clock()->now();

  joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
    "/joy", 10, std::bind(&TeleopManagerNode::joy_callback, this, _1));

  ack_sub_ = create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
    "/ackermann_cmd", 10, std::bind(&TeleopManagerNode::ack_callback, this, _1));

  drive_pub_   = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/cmd_drive", 1);
  trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/rosbag2_recorder/trigger", 1);
  memo_pub_    = create_publisher<std_msgs::msg::String>("/rosbag2_recorder/memo", 1);

  steer_offset_inc_pub_ = create_publisher<std_msgs::msg::Bool>("/steer_offset_inc", 1);
  steer_offset_dec_pub_ = create_publisher<std_msgs::msg::Bool>("/steer_offset_dec", 1);
  speed_offset_inc_pub_ = create_publisher<std_msgs::msg::Bool>("/speed_offset_inc", 1);
  speed_offset_dec_pub_ = create_publisher<std_msgs::msg::Bool>("/speed_offset_dec", 1);

  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / timer_hz_),
    std::bind(&TeleopManagerNode::timer_callback, this));
}

bool TeleopManagerNode::check_button_press(bool curr, bool &prev_flag)
{
  if (curr && !prev_flag) {
    prev_flag = true;
    return true;
  } else if (!curr) {
    prev_flag = false;
  }
  return false;
}

void TeleopManagerNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  last_joy_msg_time_ = this->get_clock()->now();

  bool curr_start = (msg->buttons.size() > start_button_index_ && msg->buttons[start_button_index_] == 1);
  bool curr_stop  = (msg->buttons.size() > stop_button_index_  && msg->buttons[stop_button_index_]  == 1);

  if (check_button_press(curr_start, prev_start_pressed_)) {
    std_msgs::msg::Bool b; b.data = true;
    trigger_pub_->publish(b);
    RCLCPP_INFO(get_logger(), "Start trigger published");
  }
  if (check_button_press(curr_stop, prev_stop_pressed_)) {
    std_msgs::msg::Bool b; b.data = false;
    trigger_pub_->publish(b);
    RCLCPP_INFO(get_logger(), "Stop trigger published");
  }

  bool curr_good = (msg->buttons.size() > good_button_index_ && msg->buttons[good_button_index_] == 1);
  bool curr_bad  = (msg->buttons.size() > bad_button_index_  && msg->buttons[bad_button_index_]  == 1);

  if (check_button_press(curr_good, prev_good_pressed_)) {
    std_msgs::msg::String s; s.data = "good";
    memo_pub_->publish(s);
    RCLCPP_INFO(get_logger(), "Published memo: good");
  }
  if (check_button_press(curr_bad, prev_bad_pressed_)) {
    std_msgs::msg::String s; s.data = "bad";
    memo_pub_->publish(s);
    RCLCPP_INFO(get_logger(), "Published memo: bad");
  }

  bool joy_pressed = (msg->buttons.size() > joy_button_index_ && msg->buttons[joy_button_index_] == 1);
  bool ack_pressed = (msg->buttons.size() > ack_button_index_ && msg->buttons[ack_button_index_] == 1);

  if (ack_pressed) {
    ack_active_ = true; joy_active_ = false;
  } else if (joy_pressed) {
    joy_active_ = true; ack_active_ = false;
  } else {
    joy_active_ = false; ack_active_ = false;
  }

  if (joy_active_) {
    double raw_speed = (msg->axes.size() > axis_speed_index_ ? msg->axes[axis_speed_index_] : 0.0);
    double raw_steer = (msg->axes.size() > axis_steer_index_ ? msg->axes[axis_steer_index_] : 0.0);
    joy_speed_ = raw_speed * speed_scale_;
    joy_steer_ = raw_steer * steer_scale_;
  }

  double a_lr = (msg->axes.size() > dpad_lr_axis_index_ ? msg->axes[dpad_lr_axis_index_] : 0.0);
  double a_ud = (msg->axes.size() > dpad_ud_axis_index_ ? msg->axes[dpad_ud_axis_index_] : 0.0);

  bool steer_offset_inc = std::abs(a_lr + 1.0) < 1e-3;
  bool steer_offset_dec = std::abs(a_lr - 1.0) < 1e-3;
  bool speed_offset_inc = std::abs(a_ud - 1.0) < 1e-3;
  bool speed_offset_dec = std::abs(a_ud + 1.0) < 1e-3;

  if (check_button_press(steer_offset_inc, prev_steer_offset_inc_pressed_)) {
    std_msgs::msg::Bool msg_out; msg_out.data = true;
    steer_offset_inc_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published /steer_offset_inc");
  }
  if (check_button_press(steer_offset_dec, prev_steer_offset_dec_pressed_)) {
    std_msgs::msg::Bool msg_out; msg_out.data = true;
    steer_offset_dec_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published /steer_offset_dec");
  }
  if (check_button_press(speed_offset_inc, prev_speed_offset_inc_pressed_)) {
    std_msgs::msg::Bool msg_out; msg_out.data = true;
    speed_offset_inc_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published /speed_offset_inc");
  }
  if (check_button_press(speed_offset_dec, prev_speed_offset_dec_pressed_)) {
    std_msgs::msg::Bool msg_out; msg_out.data = true;
    speed_offset_dec_pub_->publish(msg_out);
    RCLCPP_INFO(get_logger(), "Published /speed_offset_dec");
  }
}

void TeleopManagerNode::ack_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg)
{
  last_autonomy_msg_ = *msg;
  ack_received_ = true;
}

void TeleopManagerNode::timer_callback()
{
  ackermann_msgs::msg::AckermannDriveStamped out;
  rclcpp::Time current_time = this->get_clock()->now();

  if ((current_time - last_joy_msg_time_).seconds() > joy_timeout_sec_) {
    if (joy_active_ || ack_active_) {
      RCLCPP_WARN(get_logger(), "Joy message timed out! Stopping vehicle.");
    }
    joy_active_ = false;
    ack_active_ = false;
  }

  out.header.stamp = current_time;
  out.header.frame_id = "base_link";

  if (joy_active_) {
    out.drive.steering_angle = joy_steer_;
    out.drive.acceleration = joy_speed_;
    out.drive.steering_angle_velocity = 1.0;
  } else if (ack_active_) {
    out = last_autonomy_msg_;
    out.drive.steering_angle_velocity = 0.0;
  } else {
    out.drive.steering_angle = 0.0;
    out.drive.acceleration = 0.0;
    out.drive.steering_angle_velocity = 0.0;
  }

  out.drive.speed = 0.0;
  out.drive.jerk = 0.0;

  drive_pub_->publish(out);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopManagerNode>());
  rclcpp::shutdown();
  return 0;
}
