#include "control_filter/control_filter_node.hpp"
#include <algorithm> // for std::clamp if needed, std::fabs
#include <cmath>
#include <numeric>

ControlFilterCore::ControlFilterCore() {}

void ControlFilterCore::SetParams(const ControlFilterParams &params) {
  params_ = params;
}

const ControlFilterParams &ControlFilterCore::GetParams() const {
  return params_;
}

ackermann_msgs::msg::AckermannDrive
ControlFilterCore::Filter(const ackermann_msgs::msg::AckermannDrive &raw_drive,
                          double dt) {
  if (is_first_msg_) {
    prev_speed_ = raw_drive.speed;
    prev_steer_ = raw_drive.steering_angle;
    is_first_msg_ = false;

    ackermann_msgs::msg::AckermannDrive filtered_drive = raw_drive;
    if (params_.use_scale_filter) {
      ApplyAdvancedScaleFilter(filtered_drive);
    }
    return filtered_drive;
  }

  ackermann_msgs::msg::AckermannDrive filtered_drive = raw_drive;

  // 1. Smoothing / Slew Rate Logic
  if (params_.filter_type == "slew_rate") {
    ApplySlewRateFilter(filtered_drive, dt);
  } else if (params_.filter_type == "average") {
    speed_buffer_.push_back(raw_drive.speed);
    steering_angle_buffer_.push_back(raw_drive.steering_angle);
    ApplyAverageFilter(filtered_drive);
  } else {
    // No filter
  }

  // 2. Scale Filter Logic
  if (params_.use_scale_filter) {
    ApplyAdvancedScaleFilter(filtered_drive);
  }

  return filtered_drive;
}

void ControlFilterCore::ApplySlewRateFilter(
    ackermann_msgs::msg::AckermannDrive &drive, double dt) {
  double max_speed_step = params_.max_speed_slew_rate * dt;
  double max_steer_step = params_.max_steer_slew_rate * dt;

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

void ControlFilterCore::ApplyAverageFilter(
    ackermann_msgs::msg::AckermannDrive &drive) {
  if (static_cast<int>(speed_buffer_.size()) > params_.window_size) {
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

  // Note: updating prev_speed_/_steer_ isn't strictly necessary for average
  // filter unless we switch filters dynamically, but good to keep in sync.
  prev_speed_ = drive.speed;
  prev_steer_ = drive.steering_angle;
}

void ControlFilterCore::ApplyAdvancedScaleFilter(
    ackermann_msgs::msg::AckermannDrive &drive) {
  if (std::fabs(drive.steering_angle) < params_.straight_steer_threshold) {
    drive.speed *= params_.straight_speed_scale_ratio;
  } else {
    drive.speed *= params_.cornering_speed_scale_ratio;
  }
  drive.steering_angle *= params_.steer_scale_ratio;
}
