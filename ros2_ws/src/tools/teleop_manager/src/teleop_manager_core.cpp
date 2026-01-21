#include "teleop_manager/teleop_manager_core.hpp"
#include <cmath>

TeleopManagerCore::TeleopManagerCore(const Config &config) : config_(config) {}

void TeleopManagerCore::update_joy_input(const std::vector<float> &axes,
                                         const std::vector<int> &buttons) {
  // Update joystick state
  if (config_.axis_speed_idx >= 0 &&
      config_.axis_speed_idx < static_cast<int>(axes.size())) {
    joy_speed_ = axes[config_.axis_speed_idx];
  }
  if (config_.axis_steer_idx >= 0 &&
      config_.axis_steer_idx < static_cast<int>(axes.size())) {
    joy_steer_ = axes[config_.axis_steer_idx];
  }

  // Handle buttons
  auto get_btn = [&](int idx) -> bool {
    if (idx >= 0 && idx < static_cast<int>(buttons.size())) {
      return buttons[idx] != 0;
    }
    return false;
  };

  auto get_axis_as_btn = [&](int idx, int direction) -> bool {
    if (idx >= 0 && idx < static_cast<int>(axes.size())) {
      if (direction > 0)
        return axes[idx] > 0.5;
      if (direction < 0)
        return axes[idx] < -0.5;
    }
    return false;
  };

  joy_active_ = get_btn(config_.joy_button_idx);
  ack_active_ = get_btn(config_.ack_button_idx);

  if (check_button_press(get_btn(config_.start_button_idx), prev_start_)) {
    start_req_ = true;
  }
  if (check_button_press(get_btn(config_.stop_button_idx), prev_stop_)) {
    stop_req_ = true;
  }

  if (check_button_press(get_btn(config_.good_button_idx), prev_good_)) {
    memo_req_ = "good";
  }
  if (check_button_press(get_btn(config_.bad_button_idx), prev_bad_)) {
    memo_req_ = "bad";
  }

  bool dpad_left = get_axis_as_btn(config_.dpad_lr_axis_idx, 1);
  bool dpad_right = get_axis_as_btn(config_.dpad_lr_axis_idx, -1);
  bool dpad_up = get_axis_as_btn(config_.dpad_ud_axis_idx, 1);
  bool dpad_down = get_axis_as_btn(config_.dpad_ud_axis_idx, -1);

  if (check_button_press(dpad_left, prev_st_inc_))
    st_inc_req_ = true;
  if (check_button_press(dpad_right, prev_st_dec_))
    st_dec_req_ = true;
  if (check_button_press(dpad_up, prev_sp_inc_))
    sp_inc_req_ = true;
  if (check_button_press(dpad_down, prev_sp_dec_))
    sp_dec_req_ = true;
}

TeleopManagerCore::DriveCommand TeleopManagerCore::calculate_drive_command(
    bool is_timeout, double autonomy_speed, double autonomy_steer) {
  DriveCommand cmd{0.0, 0.0, 0.0};

  if (joy_active_ && !is_timeout) {
    cmd.acceleration = joy_speed_ * config_.speed_scale;
    cmd.steering_angle = joy_steer_ * config_.steer_scale;
    cmd.steering_velocity = 1.0;
  } else if (ack_active_) {
    cmd.acceleration = autonomy_speed;
    cmd.steering_angle = autonomy_steer;
  }
  return cmd;
}

bool TeleopManagerCore::pop_start_requested() {
  bool ret = start_req_;
  start_req_ = false;
  return ret;
}

bool TeleopManagerCore::pop_stop_requested() {
  bool ret = stop_req_;
  stop_req_ = false;
  return ret;
}

std::optional<std::string> TeleopManagerCore::pop_memo_requested() {
  auto ret = memo_req_;
  memo_req_ = std::nullopt;
  return ret;
}

bool TeleopManagerCore::pop_steer_inc_requested() {
  bool ret = st_inc_req_;
  st_inc_req_ = false;
  return ret;
}

bool TeleopManagerCore::pop_steer_dec_requested() {
  bool ret = st_dec_req_;
  st_dec_req_ = false;
  return ret;
}

bool TeleopManagerCore::pop_speed_inc_requested() {
  bool ret = sp_inc_req_;
  sp_inc_req_ = false;
  return ret;
}

bool TeleopManagerCore::pop_speed_dec_requested() {
  bool ret = sp_dec_req_;
  sp_dec_req_ = false;
  return ret;
}

bool TeleopManagerCore::check_button_press(bool curr, bool &prev_flag) {
  bool pressed = curr && !prev_flag;
  prev_flag = curr;
  return pressed;
}
