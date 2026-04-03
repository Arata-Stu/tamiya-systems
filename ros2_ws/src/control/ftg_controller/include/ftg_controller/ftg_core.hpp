#ifndef FTG_CONTROLLER__FTG_CORE_HPP_
#define FTG_CONTROLLER__FTG_CORE_HPP_

#include <cstddef>
#include <vector>

namespace ftg_controller {

struct FtgParams {
  bool mapping = false;
  bool debug = false;

  int range_offset = 180;
  int preprocess_conv_size = 3;
  int safety_radius = 3;

  double max_lidar_dist = 8.0;
  double max_speed = 1.2;
  double track_width = 0.9;

  bool use_dynamic_radius = false;
  double fixed_radius = 1.0;
  double max_gap_radius = 5.0;

  double jump_threshold = 0.5;
  double steering_limit = 0.4;

  double straights_steering_angle = 0.1745329252;   // pi / 18
  double mild_curve_angle = 0.5235987756;           // pi / 6
  double ultra_straights_angle = 0.0523598776;      // pi / 60

  double corners_speed = 0.6;
  double mild_corners_speed = 0.8;
  double straights_speed = 1.0;
  double ultra_straights_speed = 1.2;
  double mapping_speed = 1.0;
};

struct LidarScan {
  std::vector<float> ranges;
  float angle_min = 0.0F;
  float angle_increment = 0.0F;
};

struct FtgResult {
  bool valid = false;
  double speed = 0.0;
  double steering_angle = 0.0;

  double radius = 0.0;
  double best_x = 0.0;
  double best_y = 0.0;

  std::vector<float> proc_ranges;
  std::vector<float> proc_angles;
  std::size_t gap_left = 0;
  std::size_t gap_right = 0;
  std::size_t best_index = 0;
};

class FtgCore {
public:
  FtgCore() = default;

  void SetParams(const FtgParams &params);
  const FtgParams &GetParams() const;

  void SetVelocity(double velocity);
  FtgResult Process(const LidarScan &scan) const;

private:
  struct PreprocessResult {
    std::vector<float> ranges;
    std::vector<float> angles;
  };

  bool PreprocessLidar(const LidarScan &scan, PreprocessResult &out) const;
  void ApplySafetyBorder(std::vector<float> &ranges) const;
  double GetRadius() const;
  bool FindLargestGap(const std::vector<float> &ranges, double radius,
                      std::size_t &gap_left, std::size_t &gap_right) const;
  double GetSteerAngle(double best_x, double best_y) const;
  double GetSpeedFromSteer(double steering_angle) const;

  FtgParams params_;
  double velocity_ = 0.0;
};

} // namespace ftg_controller

#endif // FTG_CONTROLLER__FTG_CORE_HPP_
