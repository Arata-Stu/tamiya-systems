#include "ftg_controller/ftg_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ftg_controller {

void FtgCore::SetParams(const FtgParams &params) { params_ = params; }

const FtgParams &FtgCore::GetParams() const { return params_; }

void FtgCore::SetVelocity(double velocity) {
  velocity_ = std::max(0.0, velocity);
}

FtgResult FtgCore::Process(const LidarScan &scan) const {
  FtgResult result;

  PreprocessResult preprocessed;
  if (!PreprocessLidar(scan, preprocessed)) {
    return result;
  }

  ApplySafetyBorder(preprocessed.ranges);

  const double radius = GetRadius();
  std::size_t gap_left = 0;
  std::size_t gap_right = 0;
  if (!FindLargestGap(preprocessed.ranges, radius, gap_left, gap_right)) {
    result.proc_ranges = std::move(preprocessed.ranges);
    result.proc_angles = std::move(preprocessed.angles);
    return result;
  }

  const std::size_t best_index = (gap_left + gap_right) / 2;
  const double best_angle = preprocessed.angles.at(best_index);

  // Forward=x, Left=y
  const double best_x = std::cos(best_angle) * radius;
  const double best_y = std::sin(best_angle) * radius;

  const double steering_angle = GetSteerAngle(best_x, best_y);
  const double speed = GetSpeedFromSteer(steering_angle);

  result.valid = true;
  result.speed = speed;
  result.steering_angle = steering_angle;
  result.radius = radius;
  result.best_x = best_x;
  result.best_y = best_y;
  result.gap_left = gap_left;
  result.gap_right = gap_right;
  result.best_index = best_index;
  result.proc_ranges = std::move(preprocessed.ranges);
  result.proc_angles = std::move(preprocessed.angles);

  return result;
}

bool FtgCore::PreprocessLidar(const LidarScan &scan, PreprocessResult &out) const {
  out.ranges.clear();
  out.angles.clear();

  const std::size_t n = scan.ranges.size();
  const int conv = std::max(1, params_.preprocess_conv_size);
  const int offset = std::max(0, params_.range_offset);

  if (n == 0 || static_cast<std::size_t>(2 * offset + conv) > n) {
    return false;
  }

  const std::size_t start = static_cast<std::size_t>(offset);
  const std::size_t end = n - static_cast<std::size_t>(offset);
  if (start >= end) {
    return false;
  }

  std::vector<float> cropped;
  cropped.reserve(end - start);

  for (std::size_t i = start; i < end; ++i) {
    float value = scan.ranges[i];
    if (!std::isfinite(value)) {
      value = std::isinf(value) ? static_cast<float>(params_.max_lidar_dist)
                                : 0.0F;
    }
    if (value < 0.0F) {
      value = 0.0F;
    }
    cropped.push_back(std::min(value, static_cast<float>(params_.max_lidar_dist)));
  }

  if (cropped.size() < static_cast<std::size_t>(conv)) {
    return false;
  }

  const std::size_t valid_size = cropped.size() - static_cast<std::size_t>(conv) + 1U;
  out.ranges.reserve(valid_size);
  out.angles.reserve(valid_size);

  const std::size_t conv_center = static_cast<std::size_t>(conv / 2);

  for (std::size_t i = 0; i < valid_size; ++i) {
    double sum = 0.0;
    for (int k = 0; k < conv; ++k) {
      sum += static_cast<double>(cropped[i + static_cast<std::size_t>(k)]);
    }

    const float averaged = static_cast<float>(sum / static_cast<double>(conv));
    const float clipped = std::clamp(averaged, 0.0F, static_cast<float>(params_.max_lidar_dist));
    out.ranges.push_back(clipped);

    const std::size_t raw_idx = start + i + conv_center;
    const float angle = scan.angle_min + static_cast<float>(raw_idx) * scan.angle_increment;
    out.angles.push_back(angle);
  }

  std::reverse(out.ranges.begin(), out.ranges.end());
  std::reverse(out.angles.begin(), out.angles.end());

  return !out.ranges.empty();
}

void FtgCore::ApplySafetyBorder(std::vector<float> &ranges) const {
  const std::size_t n = ranges.size();
  if (n < 2 || params_.safety_radius <= 0) {
    return;
  }

  const std::size_t safety = static_cast<std::size_t>(params_.safety_radius);

  for (std::size_t i = 0; i + 1 < n; ++i) {
    const float jump = ranges[i + 1] - ranges[i];
    if (jump > static_cast<float>(params_.jump_threshold)) {
      const float near_value = ranges[i];
      for (std::size_t k = 0; k < safety; ++k) {
        const std::size_t idx = i + 1 + k;
        if (idx >= n) {
          break;
        }
        ranges[idx] = std::min(ranges[idx], near_value);
      }
    }
  }

  for (std::size_t i = n - 1; i > 0; --i) {
    const float jump = ranges[i - 1] - ranges[i];
    if (jump > static_cast<float>(params_.jump_threshold)) {
      const float near_value = ranges[i];
      for (std::size_t k = 0; k < safety; ++k) {
        if (i < 1 + k) {
          break;
        }
        const std::size_t idx = i - 1 - k;
        ranges[idx] = std::min(ranges[idx], near_value);
      }
    }
  }
}

double FtgCore::GetRadius() const {
  if (!params_.use_dynamic_radius) {
    return std::max(0.05, params_.fixed_radius);
  }

  if (params_.max_speed <= std::numeric_limits<double>::epsilon()) {
    return std::max(0.05, std::min(params_.max_gap_radius, params_.fixed_radius));
  }

  const double dynamic_radius =
      (params_.track_width * 0.5) + (2.0 * (velocity_ / params_.max_speed));
  return std::max(0.05, std::min(params_.max_gap_radius, dynamic_radius));
}

bool FtgCore::FindLargestGap(const std::vector<float> &ranges, double radius,
                             std::size_t &gap_left,
                             std::size_t &gap_right) const {
  if (ranges.empty()) {
    return false;
  }

  bool in_gap = false;
  std::size_t current_left = 0;
  std::size_t best_left = 0;
  std::size_t best_right = 0;
  std::size_t best_len = 0;

  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const bool is_free = static_cast<double>(ranges[i]) >= radius;
    if (is_free && !in_gap) {
      in_gap = true;
      current_left = i;
    }

    const bool closes_gap = in_gap && (!is_free || i + 1 == ranges.size());
    if (!closes_gap) {
      continue;
    }

    const std::size_t current_right = is_free ? i : (i - 1);
    const std::size_t current_len = current_right - current_left + 1;
    if (current_len > best_len) {
      best_len = current_len;
      best_left = current_left;
      best_right = current_right;
    }
    in_gap = false;
  }

  if (best_len == 0) {
    return false;
  }

  gap_left = best_left;
  gap_right = best_right;
  return true;
}

double FtgCore::GetSteerAngle(double best_x, double best_y) const {
  const double raw = std::atan2(best_y, best_x);
  return std::clamp(raw, -params_.steering_limit, params_.steering_limit);
}

double FtgCore::GetSpeedFromSteer(double steering_angle) const {
  if (params_.mapping) {
    return std::clamp(params_.mapping_speed, 0.0, params_.max_speed);
  }

  const double abs_steer = std::abs(steering_angle);

  double speed = params_.ultra_straights_speed;
  if (abs_steer > params_.mild_curve_angle) {
    speed = params_.corners_speed;
  } else if (abs_steer > params_.straights_steering_angle) {
    speed = params_.mild_corners_speed;
  } else if (abs_steer > params_.ultra_straights_angle) {
    speed = params_.straights_speed;
  }

  return std::clamp(speed, 0.0, params_.max_speed);
}

} // namespace ftg_controller
