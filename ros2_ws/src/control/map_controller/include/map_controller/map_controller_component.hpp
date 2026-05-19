#ifndef MAP_CONTROLLER__MAP_CONTROLLER_COMPONENT_HPP_
#define MAP_CONTROLLER__MAP_CONTROLLER_COMPONENT_HPP_

#include <optional>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "race_planning_msgs/msg/trajectory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace map_controller {

class MapControllerComponent : public rclcpp::Node {
public:
  explicit MapControllerComponent(const rclcpp::NodeOptions &options);

private:
  struct LookupTable {
    std::vector<double> speed_bins_mps;
    std::vector<double> steer_bins_rad;
    std::vector<std::vector<double>> lateral_accel_mps2;

    bool valid() const {
      return !speed_bins_mps.empty() && !steer_bins_rad.empty() &&
             lateral_accel_mps2.size() == steer_bins_rad.size();
    }
  };

  void LoadParameters();
  bool LoadLookupTable();
  void TrajectoryCallback(
      const race_planning_msgs::msg::Trajectory::SharedPtr msg);
  void OdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  std::optional<std::size_t> SelectNearestReferenceIndex(
      const race_planning_msgs::msg::Trajectory &trajectory) const;
  std::size_t SelectLookaheadIndex(
      const race_planning_msgs::msg::Trajectory &trajectory,
      std::size_t start_index, double distance_ahead_m) const;
  double ComputeLateralErrorNorm(double lateral_error_m) const;
  double ComputeMeanAbsCurvature(
      const race_planning_msgs::msg::Trajectory &trajectory,
      std::size_t start_index) const;
  double ResolveReferenceSpeed(double speed_mps) const;
  double AdjustSpeedForLateralError(double reference_speed_mps,
                                    double lat_err_norm,
                                    double curvature_mean) const;
  double ComputeLookaheadDistance(double lateral_error_m) const;
  double ComputeSteeringFromLateralAccel(double lateral_accel_mps2,
                                         double speed_mps);
  double LookupSteeringAngle(double lateral_accel_mps2,
                             double speed_mps) const;
  double InterpolateSteerForSpeedColumn(double accel_mps2,
                                        std::size_t speed_index) const;
  double KinematicSteeringFromLateralAccel(double lateral_accel_mps2,
                                           double speed_mps) const;
  double ApplyAccelerationScaling(double steering_angle_rad) const;
  double ApplySpeedSteeringScaling(double steering_angle_rad,
                                   double speed_mps) const;
  double ApplySteeringRateLimit(double steering_angle_rad);
  void PublishStop(const std_msgs::msg::Header &header,
                   const std::string &reason);
  void PublishDebugMarkers(const std_msgs::msg::Header &header,
                           const race_planning_msgs::msg::Trajectory &trajectory,
                           std::size_t nearest_index,
                           std::size_t lookahead_index,
                           double lookahead_distance_m);

  rclcpp::Subscription<race_planning_msgs::msg::Trajectory>::SharedPtr
      trajectory_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      drive_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      nearest_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      lookahead_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      lookahead_line_marker_pub_;

  LookupTable lookup_table_;

  std::string expected_frame_id_ = "base_link";
  std::string lookup_table_csv_path_;
  bool use_lookup_table_ = true;
  bool stop_on_invalid_reference_ = true;
  bool debug_ = false;

  double wheelbase_ = 0.26;
  double min_kinematic_speed_for_steer_ = 0.20;
  double min_forward_distance_ = 0.02;

  double m_l1_ = 0.60;
  double q_l1_ = 0.20;
  double t_clip_min_ = 0.35;
  double t_clip_max_ = 1.20;

  double speed_lookahead_sec_ = 0.25;
  double speed_lookahead_for_steer_sec_ = 0.0;

  double min_speed_ = 0.10;
  double max_speed_ = 0.30;
  double fallback_speed_ = 0.25;
  double lat_err_coeff_ = 1.0;
  double lateral_error_clip_m_ = 0.50;
  double curvature_normalization_ = 0.80;
  int curvature_window_points_ = 20;

  double steering_limit_ = 0.45;
  double max_steering_delta_ = 0.12;

  double acc_scaler_for_steer_ = 1.0;
  double dec_scaler_for_steer_ = 1.0;
  double accel_scale_threshold_ = 1.0;
  double decel_scale_threshold_ = -1.0;
  double start_scale_speed_ = 10.0;
  double end_scale_speed_ = 11.0;
  double downscale_factor_ = 0.0;

  bool have_odom_ = false;
  bool warned_lookup_fallback_ = false;
  double current_speed_mps_ = 0.0;
  double current_accel_mps2_ = 0.0;
  double previous_odom_speed_mps_ = 0.0;
  rclcpp::Time previous_odom_stamp_{0, 0, RCL_ROS_TIME};

  double last_steering_angle_rad_ = 0.0;
  double last_command_speed_mps_ = 0.0;
};

} // namespace map_controller

#endif // MAP_CONTROLLER__MAP_CONTROLLER_COMPONENT_HPP_
