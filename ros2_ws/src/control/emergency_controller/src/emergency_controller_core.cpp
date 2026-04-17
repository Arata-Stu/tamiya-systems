#include "emergency_controller/emergency_controller_core.hpp"

#include <algorithm>
#include <cmath>

namespace emergency_controller {

namespace {

double ClampPositive(double value, double fallback) {
  if (value <= 0.0) {
    return fallback;
  }
  return value;
}

} // namespace

void EmergencyControllerCore::SetParams(const EmergencyParams &params) {
  params_ = params;
}

const EmergencyParams &EmergencyControllerCore::GetParams() const {
  return params_;
}

SafetyDecision EmergencyControllerCore::EvaluateScan(const LidarScan &scan) const {
  SafetyDecision decision;

  if (scan.ranges.empty() || !std::isfinite(scan.angle_increment) ||
      std::abs(scan.angle_increment) < 1e-6F) {
    return decision;
  }

  const double front_half_angle = std::abs(DegToRad(params_.front_angle_deg));
  const double side_half_angle = std::max(front_half_angle,
                                          std::abs(DegToRad(params_.side_angle_deg)));
  const double max_considered = ClampPositive(params_.max_considered_range_m, 8.0);

  bool has_front_point = false;

  for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
    float raw_range = scan.ranges[i];
    if (!IsValidRange(raw_range)) {
      continue;
    }

    const float scan_min = std::max(0.0F, scan.range_min);
    if (raw_range < scan_min) {
      continue;
    }

    const float range_upper =
        std::isfinite(scan.range_max) && scan.range_max > 0.0F
            ? std::min(scan.range_max, static_cast<float>(max_considered))
            : static_cast<float>(max_considered);
    if (raw_range > range_upper) {
      raw_range = range_upper;
    }

    const double angle =
        static_cast<double>(scan.angle_min) +
        static_cast<double>(i) * static_cast<double>(scan.angle_increment);
    const double abs_angle = std::abs(angle);
    const double range = static_cast<double>(raw_range);

    if (abs_angle <= front_half_angle) {
      decision.min_front_distance = std::min(decision.min_front_distance, range);
      has_front_point = true;
    }

    if (angle >= 0.0 && abs_angle <= side_half_angle) {
      decision.left_min_distance = std::min(decision.left_min_distance, range);
    } else if (angle < 0.0 && abs_angle <= side_half_angle) {
      decision.right_min_distance = std::min(decision.right_min_distance, range);
    }
  }

  decision.scan_valid = has_front_point &&
                        std::isfinite(decision.min_front_distance);
  if (!decision.scan_valid) {
    return decision;
  }

  const double stop_distance = ClampPositive(params_.stop_distance_m, 0.35);
  const double slow_distance =
      std::max(stop_distance, ClampPositive(params_.slow_distance_m, 0.80));

  decision.stop_zone = decision.min_front_distance <= stop_distance;
  decision.slow_zone =
      !decision.stop_zone && decision.min_front_distance <= slow_distance;
  decision.override_active = decision.stop_zone || decision.slow_zone;

  if (!decision.override_active) {
    return decision;
  }

  const double left_clearance = std::isfinite(decision.left_min_distance)
                                    ? decision.left_min_distance
                                    : decision.min_front_distance;
  const double right_clearance = std::isfinite(decision.right_min_distance)
                                     ? decision.right_min_distance
                                     : decision.min_front_distance;

  const double denom = std::max(0.05, slow_distance);
  const double clearance_diff = left_clearance - right_clearance;
  const double normalized = std::clamp(clearance_diff / denom, -1.0, 1.0);

  const double max_steer =
      ClampPositive(params_.max_override_steer, 0.40);
  double steer =
      std::clamp(params_.steer_bias_gain * normalized, -1.0, 1.0) * max_steer;

  if (decision.stop_zone &&
      std::abs(steer) < std::abs(params_.min_turn_steer)) {
    steer = (clearance_diff >= 0.0 ? 1.0 : -1.0) *
            std::abs(params_.min_turn_steer);
  }

  decision.steer_bias = std::clamp(steer, -max_steer, max_steer);
  return decision;
}

ackermann_msgs::msg::AckermannDrive EmergencyControllerCore::ApplyOverride(
    const ackermann_msgs::msg::AckermannDrive &input,
    const SafetyDecision &decision, bool *was_overridden) const {
  ackermann_msgs::msg::AckermannDrive output = input;
  bool overridden = false;

  const double max_steer =
      ClampPositive(params_.max_override_steer, 0.40);

  if (decision.stop_zone) {
    output.speed = static_cast<float>(params_.stop_speed_mps);
    output.steering_angle =
        static_cast<float>(std::clamp(decision.steer_bias, -max_steer, max_steer));
    overridden = true;
  } else if (decision.slow_zone) {
    output.speed =
        std::min(output.speed, static_cast<float>(params_.slow_speed_mps));
    const double steered =
        static_cast<double>(input.steering_angle) + decision.steer_bias;
    output.steering_angle =
        static_cast<float>(std::clamp(steered, -max_steer, max_steer));
    overridden = true;
  }

  if (was_overridden != nullptr) {
    *was_overridden = overridden;
  }

  return output;
}

bool EmergencyControllerCore::IsValidRange(float value) {
  return std::isfinite(value) && value > 0.0F;
}

double EmergencyControllerCore::DegToRad(double degree) {
  constexpr double kPi = 3.14159265358979323846;
  return degree * kPi / 180.0;
}

} // namespace emergency_controller
