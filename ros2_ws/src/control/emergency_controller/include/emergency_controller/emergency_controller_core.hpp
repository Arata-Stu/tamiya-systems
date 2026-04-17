#ifndef EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_CORE_HPP_
#define EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_CORE_HPP_

#include <limits>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive.hpp"

namespace emergency_controller {

struct EmergencyParams {
  double front_angle_deg = 55.0;
  double side_angle_deg = 75.0;

  double stop_distance_m = 0.35;
  double slow_distance_m = 0.80;

  double slow_speed_mps = 0.40;
  double stop_speed_mps = 0.0;

  double max_override_steer = 0.40;
  double min_turn_steer = 0.15;
  double steer_bias_gain = 1.20;

  double max_considered_range_m = 8.0;
};

struct LidarScan {
  std::vector<float> ranges;
  float angle_min = 0.0F;
  float angle_increment = 0.0F;
  float range_min = 0.0F;
  float range_max = 0.0F;
};

struct SafetyDecision {
  bool scan_valid = false;
  bool slow_zone = false;
  bool stop_zone = false;
  bool override_active = false;

  double min_front_distance = std::numeric_limits<double>::infinity();
  double left_min_distance = std::numeric_limits<double>::infinity();
  double right_min_distance = std::numeric_limits<double>::infinity();

  double steer_bias = 0.0;
};

class EmergencyControllerCore {
public:
  EmergencyControllerCore() = default;

  void SetParams(const EmergencyParams &params);
  const EmergencyParams &GetParams() const;

  SafetyDecision EvaluateScan(const LidarScan &scan) const;

  ackermann_msgs::msg::AckermannDrive ApplyOverride(
      const ackermann_msgs::msg::AckermannDrive &input,
      const SafetyDecision &decision, bool *was_overridden = nullptr) const;

private:
  static bool IsValidRange(float value);
  static double DegToRad(double degree);

  EmergencyParams params_;
};

} // namespace emergency_controller

#endif // EMERGENCY_CONTROLLER__EMERGENCY_CONTROLLER_CORE_HPP_
