#include "map_controller/map_controller_component.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace map_controller {

namespace {

double ClampPositive(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

double ClampNonNegative(double value) { return std::max(0.0, value); }

double ClampUnitInterval(double value) { return std::clamp(value, 0.0, 1.0); }

double Distance2D(const geometry_msgs::msg::Point &point) {
  return std::hypot(point.x, point.y);
}

bool IsFinitePoint(const geometry_msgs::msg::Point &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

double SpeedFromVector(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

double SanitizeFiniteOr(double value, double fallback) {
  return std::isfinite(value) ? value : fallback;
}

std::vector<double> ParseCsvRow(const std::string &line) {
  std::vector<double> values;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    try {
      values.push_back(std::stod(cell));
    } catch (const std::exception &) {
      values.push_back(std::numeric_limits<double>::quiet_NaN());
    }
  }
  return values;
}

} // namespace

MapControllerComponent::MapControllerComponent(
    const rclcpp::NodeOptions &options)
    : Node("map_controller_node", options) {
  LoadParameters();

  drive_pub_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "autonomous/cmd_drive", rclcpp::QoS(10));

  trajectory_sub_ =
      this->create_subscription<race_planning_msgs::msg::Trajectory>(
          "trajectory_reference", rclcpp::QoS(10),
          std::bind(&MapControllerComponent::TrajectoryCallback, this,
                    std::placeholders::_1));

  odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", rclcpp::QoS(10),
      std::bind(&MapControllerComponent::OdometryCallback, this,
                std::placeholders::_1));

  if (debug_) {
    nearest_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "map_controller/nearest_marker", rclcpp::QoS(1));
    lookahead_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "map_controller/lookahead_marker", rclcpp::QoS(1));
    lookahead_line_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "map_controller/lookahead_line_marker", rclcpp::QoS(1));
  }

  RCLCPP_INFO(
      this->get_logger(),
      "MAP Controller initialized (lookup=%s, wheelbase=%.3f, L1=[%.3f, %.3f], "
      "speed=[%.3f, %.3f], frame=%s)",
      lookup_table_.valid() ? "table" : "kinematic-fallback", wheelbase_,
      t_clip_min_, t_clip_max_, min_speed_, max_speed_,
      expected_frame_id_.c_str());
}

void MapControllerComponent::LoadParameters() {
  expected_frame_id_ =
      this->declare_parameter<std::string>("expected_frame_id", "base_link");
  lookup_table_csv_path_ =
      this->declare_parameter<std::string>("lookup_table_csv_path", "");
  use_lookup_table_ =
      this->declare_parameter<bool>("use_lookup_table", true);
  stop_on_invalid_reference_ =
      this->declare_parameter<bool>("stop_on_invalid_reference", true);
  debug_ = this->declare_parameter<bool>("debug", false);

  wheelbase_ =
      ClampPositive(this->declare_parameter<double>("wheelbase", 0.26), 0.26);
  min_kinematic_speed_for_steer_ = ClampPositive(
      this->declare_parameter<double>("min_kinematic_speed_for_steer", 0.20),
      0.20);
  min_forward_distance_ = ClampNonNegative(
      this->declare_parameter<double>("min_forward_distance", 0.02));

  m_l1_ = ClampNonNegative(this->declare_parameter<double>("m_l1", 0.60));
  q_l1_ = this->declare_parameter<double>("q_l1", 0.20);
  t_clip_min_ = ClampPositive(
      this->declare_parameter<double>("t_clip_min", 0.35), 0.35);
  t_clip_max_ = ClampPositive(
      this->declare_parameter<double>("t_clip_max", 1.20), 1.20);
  if (t_clip_max_ < t_clip_min_) {
    t_clip_max_ = t_clip_min_;
  }

  speed_lookahead_sec_ = ClampNonNegative(
      this->declare_parameter<double>("speed_lookahead", 0.25));
  speed_lookahead_for_steer_sec_ = ClampNonNegative(
      this->declare_parameter<double>("speed_lookahead_for_steer", 0.0));

  min_speed_ =
      ClampNonNegative(this->declare_parameter<double>("min_speed", 0.10));
  max_speed_ =
      ClampNonNegative(this->declare_parameter<double>("max_speed", 0.30));
  if (max_speed_ < min_speed_) {
    max_speed_ = min_speed_;
  }
  fallback_speed_ = ClampNonNegative(
      this->declare_parameter<double>("fallback_speed", 0.25));
  lat_err_coeff_ = ClampUnitInterval(
      this->declare_parameter<double>("lat_err_coeff", 1.0));
  lateral_error_clip_m_ = ClampPositive(
      this->declare_parameter<double>("lateral_error_clip_m", 0.50), 0.50);
  curvature_normalization_ = ClampPositive(
      this->declare_parameter<double>("curvature_normalization", 0.80), 0.80);
  const int curvature_window_points = static_cast<int>(
      this->declare_parameter<int>("curvature_window_points", 20));
  curvature_window_points_ = std::max(1, curvature_window_points);

  steering_limit_ = ClampPositive(
      this->declare_parameter<double>("steering_limit", 0.45), 0.45);
  max_steering_delta_ = ClampPositive(
      this->declare_parameter<double>("max_steering_delta", 0.12), 0.12);

  acc_scaler_for_steer_ = ClampPositive(
      this->declare_parameter<double>("acc_scaler_for_steer", 1.0), 1.0);
  dec_scaler_for_steer_ = ClampPositive(
      this->declare_parameter<double>("dec_scaler_for_steer", 1.0), 1.0);
  accel_scale_threshold_ =
      this->declare_parameter<double>("accel_scale_threshold", 1.0);
  decel_scale_threshold_ =
      this->declare_parameter<double>("decel_scale_threshold", -1.0);
  start_scale_speed_ = ClampNonNegative(
      this->declare_parameter<double>("start_scale_speed", 10.0));
  end_scale_speed_ = ClampNonNegative(
      this->declare_parameter<double>("end_scale_speed", 11.0));
  if (end_scale_speed_ < start_scale_speed_) {
    end_scale_speed_ = start_scale_speed_;
  }
  downscale_factor_ = ClampUnitInterval(
      this->declare_parameter<double>("downscale_factor", 0.0));

  LoadLookupTable();
}

bool MapControllerComponent::LoadLookupTable() {
  lookup_table_ = LookupTable{};
  warned_lookup_fallback_ = false;

  if (!use_lookup_table_) {
    return false;
  }
  if (lookup_table_csv_path_.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "lookup_table_csv_path is empty. Falling back to kinematic steering.");
    return false;
  }

  std::ifstream input(lookup_table_csv_path_);
  if (!input.is_open()) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to open lookup table CSV: %s. Falling back to kinematic steering.",
                lookup_table_csv_path_.c_str());
    return false;
  }

  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    rows.push_back(ParseCsvRow(line));
  }

  if (rows.size() < 2U || rows.front().size() < 2U) {
    RCLCPP_WARN(this->get_logger(),
                "Lookup table CSV is too small: %s. Falling back to kinematic steering.",
                lookup_table_csv_path_.c_str());
    return false;
  }

  const std::size_t column_count = rows.front().size();
  for (auto &row : rows) {
    if (row.size() < column_count) {
      row.resize(column_count, std::numeric_limits<double>::quiet_NaN());
    }
  }

  lookup_table_.speed_bins_mps.reserve(column_count - 1U);
  for (std::size_t column = 1U; column < column_count; ++column) {
    lookup_table_.speed_bins_mps.push_back(std::abs(SanitizeFiniteOr(
        rows.front()[column], lookup_table_.speed_bins_mps.empty()
                                 ? 0.0
                                 : lookup_table_.speed_bins_mps.back())));
  }

  lookup_table_.steer_bins_rad.reserve(rows.size() - 1U);
  lookup_table_.lateral_accel_mps2.reserve(rows.size() - 1U);
  for (std::size_t row_index = 1U; row_index < rows.size(); ++row_index) {
    lookup_table_.steer_bins_rad.push_back(std::abs(SanitizeFiniteOr(
        rows[row_index][0], lookup_table_.steer_bins_rad.empty()
                                 ? 0.0
                                 : lookup_table_.steer_bins_rad.back())));
    std::vector<double> accel_row;
    accel_row.reserve(column_count - 1U);
    for (std::size_t column = 1U; column < column_count; ++column) {
      const double fallback =
          accel_row.empty() ? 0.0 : accel_row.back();
      accel_row.push_back(std::max(
          0.0, std::abs(SanitizeFiniteOr(rows[row_index][column], fallback))));
    }
    lookup_table_.lateral_accel_mps2.push_back(std::move(accel_row));
  }

  if (!lookup_table_.valid()) {
    RCLCPP_WARN(this->get_logger(),
                "Lookup table CSV is invalid: %s. Falling back to kinematic steering.",
                lookup_table_csv_path_.c_str());
    lookup_table_ = LookupTable{};
    return false;
  }

  RCLCPP_INFO(this->get_logger(),
              "Loaded steering lookup table: %s (steer_bins=%zu, speed_bins=%zu)",
              lookup_table_csv_path_.c_str(),
              lookup_table_.steer_bins_rad.size(),
              lookup_table_.speed_bins_mps.size());
  return true;
}

void MapControllerComponent::TrajectoryCallback(
    const race_planning_msgs::msg::Trajectory::SharedPtr msg) {
  if (msg->points.empty()) {
    PublishStop(msg->header, "empty trajectory_reference");
    return;
  }

  if (!expected_frame_id_.empty() && !msg->header.frame_id.empty() &&
      msg->header.frame_id != expected_frame_id_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "trajectory_reference frame_id is '%s', but this controller expects '%s'. "
        "No TF transform is applied.",
        msg->header.frame_id.c_str(), expected_frame_id_.c_str());
  }

  const auto nearest_index_opt = SelectNearestReferenceIndex(*msg);
  if (!nearest_index_opt.has_value()) {
    PublishStop(msg->header, "no valid trajectory_reference point");
    return;
  }
  const std::size_t nearest_index = *nearest_index_opt;
  const auto &nearest_point = msg->points[nearest_index].pose.position;

  const double lateral_error_m = std::abs(nearest_point.y);
  const double lat_err_norm = ComputeLateralErrorNorm(lateral_error_m);
  const double curvature_mean = ComputeMeanAbsCurvature(*msg, nearest_index);

  const double speed_lookahead_distance_m =
      current_speed_mps_ * speed_lookahead_sec_;
  const std::size_t speed_index =
      SelectLookaheadIndex(*msg, nearest_index, speed_lookahead_distance_m);
  const double reference_speed_mps =
      ResolveReferenceSpeed(msg->points[speed_index].speed_mps);
  const double command_speed_mps =
      AdjustSpeedForLateralError(reference_speed_mps, lat_err_norm,
                                 curvature_mean);

  const double lookahead_distance_m =
      ComputeLookaheadDistance(lateral_error_m);
  const std::size_t lookahead_index =
      SelectLookaheadIndex(*msg, nearest_index, lookahead_distance_m);
  const auto &lookahead_point = msg->points[lookahead_index].pose.position;
  const double lookahead_norm = Distance2D(lookahead_point);
  if (lookahead_norm < 1.0e-6) {
    PublishStop(msg->header, "lookahead target too close to base_link");
    return;
  }

  const double steer_speed_lookahead_distance_m =
      current_speed_mps_ * speed_lookahead_for_steer_sec_;
  const std::size_t steer_speed_index = SelectLookaheadIndex(
      *msg, nearest_index, steer_speed_lookahead_distance_m);
  double speed_for_lookup_mps = AdjustSpeedForLateralError(
      ResolveReferenceSpeed(msg->points[steer_speed_index].speed_mps),
      lat_err_norm, curvature_mean);
  if (speed_for_lookup_mps <= 0.0) {
    speed_for_lookup_mps =
        std::max(command_speed_mps, current_speed_mps_);
  }

  const double sin_eta =
      std::clamp(lookahead_point.y / lookahead_norm, -1.0, 1.0);
  const double lateral_accel_mps2 =
      2.0 * speed_for_lookup_mps * speed_for_lookup_mps /
      std::max(lookahead_distance_m, 1.0e-3) * sin_eta;

  double steering_angle_rad =
      ComputeSteeringFromLateralAccel(lateral_accel_mps2, speed_for_lookup_mps);
  steering_angle_rad = ApplyAccelerationScaling(steering_angle_rad);
  steering_angle_rad =
      ApplySpeedSteeringScaling(steering_angle_rad, speed_for_lookup_mps);
  steering_angle_rad =
      std::clamp(steering_angle_rad, -steering_limit_, steering_limit_);
  steering_angle_rad = ApplySteeringRateLimit(steering_angle_rad);

  ackermann_msgs::msg::AckermannDriveStamped drive_msg;
  drive_msg.header = msg->header;
  drive_msg.drive.speed = static_cast<float>(command_speed_mps);
  drive_msg.drive.steering_angle = static_cast<float>(steering_angle_rad);
  drive_pub_->publish(drive_msg);

  last_command_speed_mps_ = command_speed_mps;

  if (debug_) {
    PublishDebugMarkers(msg->header, *msg, nearest_index, lookahead_index,
                        lookahead_distance_m);
  }
}

void MapControllerComponent::OdometryCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  const double speed_mps = ClampNonNegative(SpeedFromVector(
      msg->twist.twist.linear.x, msg->twist.twist.linear.y,
      msg->twist.twist.linear.z));
  const rclcpp::Time stamp = msg->header.stamp.sec == 0 &&
                                     msg->header.stamp.nanosec == 0
                                 ? this->now()
                                 : rclcpp::Time(msg->header.stamp);

  if (have_odom_) {
    const double dt = (stamp - previous_odom_stamp_).seconds();
    if (dt > 1.0e-3 && dt < 1.0) {
      current_accel_mps2_ = (speed_mps - previous_odom_speed_mps_) / dt;
    }
  }

  previous_odom_stamp_ = stamp;
  previous_odom_speed_mps_ = speed_mps;
  current_speed_mps_ = speed_mps;
  have_odom_ = true;
}

std::optional<std::size_t> MapControllerComponent::SelectNearestReferenceIndex(
    const race_planning_msgs::msg::Trajectory &trajectory) const {
  std::optional<std::size_t> best_forward_index;
  std::optional<std::size_t> fallback_index;
  double best_forward_distance = std::numeric_limits<double>::infinity();
  double fallback_distance = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0U; i < trajectory.points.size(); ++i) {
    const auto &point = trajectory.points[i].pose.position;
    if (!IsFinitePoint(point)) {
      continue;
    }

    const double distance = Distance2D(point);
    if (distance < fallback_distance) {
      fallback_distance = distance;
      fallback_index = i;
    }
    if (point.x >= min_forward_distance_ && distance < best_forward_distance) {
      best_forward_distance = distance;
      best_forward_index = i;
    }
  }

  if (best_forward_index.has_value()) {
    return best_forward_index;
  }
  return fallback_index;
}

std::size_t MapControllerComponent::SelectLookaheadIndex(
    const race_planning_msgs::msg::Trajectory &trajectory,
    std::size_t start_index, double distance_ahead_m) const {
  if (trajectory.points.empty()) {
    return 0U;
  }
  if (start_index >= trajectory.points.size()) {
    return trajectory.points.size() - 1U;
  }
  if (distance_ahead_m <= 0.0) {
    return start_index;
  }

  const auto &start_point = trajectory.points[start_index].pose.position;
  geometry_msgs::msg::Point previous_point = start_point;
  std::size_t last_valid_index = start_index;
  double accumulated_distance = 0.0;

  for (std::size_t i = start_index + 1U; i < trajectory.points.size(); ++i) {
    const auto &point = trajectory.points[i].pose.position;
    if (!IsFinitePoint(point)) {
      continue;
    }

    accumulated_distance +=
        std::hypot(point.x - previous_point.x, point.y - previous_point.y);
    previous_point = point;
    last_valid_index = i;

    if (accumulated_distance >= distance_ahead_m) {
      return i;
    }
  }

  return last_valid_index;
}

double MapControllerComponent::ComputeLateralErrorNorm(
    double lateral_error_m) const {
  const double clipped =
      std::clamp(lateral_error_m, 0.0, lateral_error_clip_m_);
  return 0.5 * (clipped / lateral_error_clip_m_);
}

double MapControllerComponent::ComputeMeanAbsCurvature(
    const race_planning_msgs::msg::Trajectory &trajectory,
    std::size_t start_index) const {
  double sum = 0.0;
  int count = 0;
  for (std::size_t i = start_index;
       i < trajectory.points.size() && count < curvature_window_points_; ++i) {
    const double curvature = trajectory.points[i].curvature_radpm;
    if (!std::isfinite(curvature)) {
      continue;
    }
    sum += std::abs(curvature);
    ++count;
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double MapControllerComponent::ResolveReferenceSpeed(double speed_mps) const {
  if (std::isfinite(speed_mps) && speed_mps > 0.0) {
    return speed_mps;
  }
  return fallback_speed_;
}

double MapControllerComponent::AdjustSpeedForLateralError(
    double reference_speed_mps, double lat_err_norm,
    double curvature_mean) const {
  double command_speed_mps = ClampNonNegative(reference_speed_mps);
  const double curvature_term = std::clamp(
      2.0 * (curvature_mean / curvature_normalization_) - 2.0, 0.0, 1.0);
  const double lateral_term = lat_err_norm * 2.0;
  const double scale =
      1.0 - lat_err_coeff_ +
      lat_err_coeff_ * std::exp(-lateral_term * curvature_term);
  command_speed_mps *= scale;

  if (command_speed_mps <= 0.0) {
    return 0.0;
  }
  return std::clamp(command_speed_mps, min_speed_, max_speed_);
}

double MapControllerComponent::ComputeLookaheadDistance(
    double lateral_error_m) const {
  const double raw_distance = q_l1_ + m_l1_ * current_speed_mps_;
  const double lower_bound =
      std::max(t_clip_min_, std::sqrt(2.0) * lateral_error_m);
  return std::clamp(raw_distance, lower_bound, std::max(lower_bound, t_clip_max_));
}

double MapControllerComponent::ComputeSteeringFromLateralAccel(
    double lateral_accel_mps2, double speed_mps) {
  if (use_lookup_table_ && lookup_table_.valid()) {
    return LookupSteeringAngle(lateral_accel_mps2, speed_mps);
  }
  if (use_lookup_table_ && !lookup_table_.valid() && !warned_lookup_fallback_) {
    warned_lookup_fallback_ = true;
    RCLCPP_WARN(this->get_logger(),
                "No valid steering lookup table loaded. Using kinematic fallback.");
  }
  return KinematicSteeringFromLateralAccel(lateral_accel_mps2, speed_mps);
}

double MapControllerComponent::LookupSteeringAngle(double lateral_accel_mps2,
                                                   double speed_mps) const {
  if (!lookup_table_.valid()) {
    return KinematicSteeringFromLateralAccel(lateral_accel_mps2, speed_mps);
  }

  const double accel_magnitude = std::abs(lateral_accel_mps2);
  const double speed_magnitude = std::abs(speed_mps);
  const double sign = lateral_accel_mps2 >= 0.0 ? 1.0 : -1.0;
  const auto &speed_bins = lookup_table_.speed_bins_mps;

  double steer_magnitude = 0.0;
  if (speed_bins.size() == 1U || speed_magnitude <= speed_bins.front()) {
    steer_magnitude = InterpolateSteerForSpeedColumn(accel_magnitude, 0U);
  } else if (speed_magnitude >= speed_bins.back()) {
    steer_magnitude = InterpolateSteerForSpeedColumn(
        accel_magnitude, speed_bins.size() - 1U);
  } else {
    auto upper_it =
        std::lower_bound(speed_bins.begin(), speed_bins.end(), speed_magnitude);
    const std::size_t upper_index =
        static_cast<std::size_t>(std::distance(speed_bins.begin(), upper_it));
    const std::size_t lower_index = upper_index - 1U;
    const double lower_speed = speed_bins[lower_index];
    const double upper_speed = speed_bins[upper_index];
    const double lower_steer =
        InterpolateSteerForSpeedColumn(accel_magnitude, lower_index);
    const double upper_steer =
        InterpolateSteerForSpeedColumn(accel_magnitude, upper_index);
    const double blend = (speed_magnitude - lower_speed) /
                         std::max(upper_speed - lower_speed, 1.0e-6);
    steer_magnitude =
        lower_steer + blend * (upper_steer - lower_steer);
  }

  return sign * steer_magnitude;
}

double MapControllerComponent::InterpolateSteerForSpeedColumn(
    double accel_mps2, std::size_t speed_index) const {
  const auto &steer_bins = lookup_table_.steer_bins_rad;
  if (steer_bins.empty()) {
    return 0.0;
  }

  double previous_accel = lookup_table_.lateral_accel_mps2.front()[speed_index];
  double previous_steer = steer_bins.front();
  if (accel_mps2 <= previous_accel) {
    return previous_steer;
  }

  for (std::size_t row = 1U; row < steer_bins.size(); ++row) {
    const double current_accel =
        lookup_table_.lateral_accel_mps2[row][speed_index];
    const double current_steer = steer_bins[row];
    if (current_accel <= previous_accel + 1.0e-9) {
      previous_accel = current_accel;
      previous_steer = current_steer;
      continue;
    }
    if (accel_mps2 <= current_accel) {
      const double blend = (accel_mps2 - previous_accel) /
                           (current_accel - previous_accel);
      return previous_steer + blend * (current_steer - previous_steer);
    }
    previous_accel = current_accel;
    previous_steer = current_steer;
  }

  return steer_bins.back();
}

double MapControllerComponent::KinematicSteeringFromLateralAccel(
    double lateral_accel_mps2, double speed_mps) const {
  const double speed_for_geometry =
      std::max(std::abs(speed_mps), min_kinematic_speed_for_steer_);
  const double curvature =
      lateral_accel_mps2 / (speed_for_geometry * speed_for_geometry);
  return std::atan(wheelbase_ * curvature);
}

double MapControllerComponent::ApplyAccelerationScaling(
    double steering_angle_rad) const {
  if (current_accel_mps2_ >= accel_scale_threshold_) {
    return steering_angle_rad * acc_scaler_for_steer_;
  }
  if (current_accel_mps2_ <= decel_scale_threshold_) {
    return steering_angle_rad * dec_scaler_for_steer_;
  }
  return steering_angle_rad;
}

double MapControllerComponent::ApplySpeedSteeringScaling(
    double steering_angle_rad, double speed_mps) const {
  const double speed_range =
      std::max(0.1, end_scale_speed_ - start_scale_speed_);
  const double scale = 1.0 - std::clamp(
                                 (speed_mps - start_scale_speed_) / speed_range,
                                 0.0, 1.0) *
                                 downscale_factor_;
  return steering_angle_rad * scale;
}

double MapControllerComponent::ApplySteeringRateLimit(
    double steering_angle_rad) {
  const double limited = std::clamp(
      steering_angle_rad, last_steering_angle_rad_ - max_steering_delta_,
      last_steering_angle_rad_ + max_steering_delta_);
  last_steering_angle_rad_ = limited;
  return limited;
}

void MapControllerComponent::PublishStop(const std_msgs::msg::Header &header,
                                         const std::string &reason) {
  if (!stop_on_invalid_reference_) {
    return;
  }

  ackermann_msgs::msg::AckermannDriveStamped drive_msg;
  drive_msg.header = header;
  drive_msg.drive.speed = 0.0F;
  drive_msg.drive.steering_angle = 0.0F;
  drive_pub_->publish(drive_msg);

  RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                       "MAP controller published stop: %s", reason.c_str());
}

void MapControllerComponent::PublishDebugMarkers(
    const std_msgs::msg::Header &header,
    const race_planning_msgs::msg::Trajectory &trajectory,
    std::size_t nearest_index, std::size_t lookahead_index,
    double lookahead_distance_m) {
  if (!nearest_marker_pub_ || !lookahead_marker_pub_ ||
      !lookahead_line_marker_pub_ || nearest_index >= trajectory.points.size() ||
      lookahead_index >= trajectory.points.size()) {
    return;
  }

  const auto &nearest_point = trajectory.points[nearest_index].pose.position;
  const auto &lookahead_point =
      trajectory.points[lookahead_index].pose.position;

  visualization_msgs::msg::Marker nearest_marker;
  nearest_marker.header = header;
  nearest_marker.ns = "map_controller";
  nearest_marker.id = 0;
  nearest_marker.type = visualization_msgs::msg::Marker::SPHERE;
  nearest_marker.action = visualization_msgs::msg::Marker::ADD;
  nearest_marker.pose.position = nearest_point;
  nearest_marker.pose.orientation.w = 1.0;
  nearest_marker.scale.x = 0.10;
  nearest_marker.scale.y = 0.10;
  nearest_marker.scale.z = 0.10;
  nearest_marker.color.r = 1.0F;
  nearest_marker.color.g = 0.2F;
  nearest_marker.color.b = 0.2F;
  nearest_marker.color.a = 0.9F;
  nearest_marker_pub_->publish(nearest_marker);

  visualization_msgs::msg::Marker lookahead_marker = nearest_marker;
  lookahead_marker.id = 1;
  lookahead_marker.pose.position = lookahead_point;
  lookahead_marker.scale.x = 0.12;
  lookahead_marker.scale.y = 0.12;
  lookahead_marker.scale.z = 0.12;
  lookahead_marker.color.r = 0.2F;
  lookahead_marker.color.g = 1.0F;
  lookahead_marker.color.b = 0.2F;
  lookahead_marker_pub_->publish(lookahead_marker);

  visualization_msgs::msg::Marker line_marker;
  line_marker.header = header;
  line_marker.ns = "map_controller";
  line_marker.id = 2;
  line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  line_marker.action = visualization_msgs::msg::Marker::ADD;
  line_marker.pose.orientation.w = 1.0;
  line_marker.scale.x = 0.03;
  line_marker.color.r = 0.2F;
  line_marker.color.g = 0.8F;
  line_marker.color.b = 1.0F;
  line_marker.color.a = 0.8F;
  geometry_msgs::msg::Point origin;
  origin.x = 0.0;
  origin.y = 0.0;
  origin.z = 0.0;
  line_marker.points.push_back(origin);
  line_marker.points.push_back(lookahead_point);
  line_marker.text = std::to_string(lookahead_distance_m);
  lookahead_line_marker_pub_->publish(line_marker);
}

} // namespace map_controller

RCLCPP_COMPONENTS_REGISTER_NODE(map_controller::MapControllerComponent)
