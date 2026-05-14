#ifndef RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_CORE_HPP_
#define RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_CORE_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "nav_msgs/msg/path.hpp"

namespace raceline_path_publisher {

struct RacelineSample {
  double s = 0.0;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double speed = 0.0;
  double curvature = 0.0;
  double acceleration = 0.0;
};

struct RacelineData {
  std::vector<RacelineSample> samples;
  double total_length = 0.0;
  double nominal_spacing = 0.0;
};

class RacelinePathCore {
public:
  bool LoadCsv(const std::string &csv_path, const std::string &direction,
               std::string &error_message);

  const RacelineData &GetData() const;

  nav_msgs::msg::Path BuildPath(const std::string &frame_id,
                                const builtin_interfaces::msg::Time &stamp) const;
  nav_msgs::msg::Path BuildPathFromIndices(
      const std::vector<std::size_t> &indices, const std::string &frame_id,
      const builtin_interfaces::msg::Time &stamp) const;

  std::size_t FindNearestIndex(double x, double y) const;
  std::vector<std::size_t> SelectForwardIndices(double vehicle_x, double vehicle_y,
                                                double forward_distance_m,
                                                std::size_t max_points) const;

private:
  RacelineData data_;
};

} // namespace raceline_path_publisher

#endif // RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_CORE_HPP_
