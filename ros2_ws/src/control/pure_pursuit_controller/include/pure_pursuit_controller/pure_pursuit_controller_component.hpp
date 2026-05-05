#ifndef PURE_PURSUIT_CONTROLLER__PURE_PURSUIT_CONTROLLER_COMPONENT_HPP_
#define PURE_PURSUIT_CONTROLLER__PURE_PURSUIT_CONTROLLER_COMPONENT_HPP_

#include <optional>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace pure_pursuit_controller {

class PurePursuitControllerComponent : public rclcpp::Node {
public:
  explicit PurePursuitControllerComponent(const rclcpp::NodeOptions &options);

private:
  struct TargetPoint {
    geometry_msgs::msg::Point point;
    double distance = 0.0;
    std::size_t index = 0U;
  };

  void LoadParameters();
  void TrajectoryCallback(const nav_msgs::msg::Path::SharedPtr msg);
  void VelocityCallback(const std_msgs::msg::Float32::SharedPtr msg);
  void OdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  std::optional<TargetPoint> SelectTargetPoint(const nav_msgs::msg::Path &path,
                                               double lookahead_distance) const;
  double ComputeLookaheadDistance() const;
  double ComputeSteeringAngle(const TargetPoint &target) const;
  double ComputeSpeed(double steering_angle, const TargetPoint &target,
                      std::size_t path_size) const;
  double ApplySteeringRateLimit(double steering_angle);
  void PublishStop(const std_msgs::msg::Header &header,
                   const std::string &reason);
  void PublishDebugMarkers(const nav_msgs::msg::Path &path,
                           const TargetPoint &target,
                           double lookahead_distance,
                           double steering_angle);

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr trajectory_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr velocity_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      drive_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      target_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      lookahead_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      target_line_marker_pub_;

  double wheelbase_ = 0.26;
  double lookahead_min_ = 0.35;
  double lookahead_max_ = 1.2;
  double lookahead_gain_ = 0.35;
  double lookahead_base_ = 0.35;
  double min_forward_distance_ = 0.05;
  double steering_limit_ = 0.45;
  double max_steering_delta_ = 0.12;

  double min_speed_ = 0.25;
  double max_speed_ = 1.0;
  double curvature_speed_gain_ = 1.5;
  double steering_speed_gain_ = 1.0;
  double short_path_speed_scale_ = 0.6;
  bool stop_on_invalid_path_ = true;
  bool debug_ = false;
  std::string expected_frame_id_ = "base_link";
  std::string velocity_source_ = "odometry";

  double last_steering_angle_ = 0.0;
  double last_command_speed_ = 0.0;
  double current_speed_ = 0.0;
};

} // namespace pure_pursuit_controller

#endif // PURE_PURSUIT_CONTROLLER__PURE_PURSUIT_CONTROLLER_COMPONENT_HPP_
