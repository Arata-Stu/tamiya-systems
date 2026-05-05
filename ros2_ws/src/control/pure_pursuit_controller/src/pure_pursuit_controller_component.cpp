#include "pure_pursuit_controller/pure_pursuit_controller_component.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"

namespace pure_pursuit_controller {

namespace {

double ClampPositive(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

double ClampNonNegative(double value) { return std::max(0.0, value); }

double Distance2D(const geometry_msgs::msg::Point &point) {
  return std::hypot(point.x, point.y);
}

bool IsFinitePoint(const geometry_msgs::msg::Point &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

double SpeedFromVector(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

} // namespace

PurePursuitControllerComponent::PurePursuitControllerComponent(
    const rclcpp::NodeOptions &options)
    : Node("pure_pursuit_controller_node", options) {
  LoadParameters();

  drive_pub_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "autonomous/cmd_drive", rclcpp::QoS(10));

  trajectory_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "trajectory", rclcpp::QoS(10),
      std::bind(&PurePursuitControllerComponent::TrajectoryCallback, this,
                std::placeholders::_1));

  if (velocity_source_ == "float32") {
    velocity_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "current_velocity", rclcpp::QoS(10),
        std::bind(&PurePursuitControllerComponent::VelocityCallback, this,
                  std::placeholders::_1));
  } else if (velocity_source_ == "odometry") {
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odometry", rclcpp::QoS(10),
        std::bind(&PurePursuitControllerComponent::OdometryCallback, this,
                  std::placeholders::_1));
  }

  if (debug_) {
    target_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "pure_pursuit/target_marker", rclcpp::QoS(1));
    lookahead_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "pure_pursuit/lookahead_marker", rclcpp::QoS(1));
    target_line_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "pure_pursuit/target_line_marker", rclcpp::QoS(1));
  }

  RCLCPP_INFO(
      this->get_logger(),
      "Pure Pursuit Controller initialized "
      "(wheelbase=%.3f, lookahead=[%.3f, %.3f], speed=[%.3f, %.3f], velocity_source=%s)",
      wheelbase_, lookahead_min_, lookahead_max_, min_speed_, max_speed_,
      velocity_source_.c_str());
}

void PurePursuitControllerComponent::LoadParameters() {
  wheelbase_ =
      ClampPositive(this->declare_parameter<double>("wheelbase", 0.26), 0.26);
  lookahead_min_ = ClampPositive(
      this->declare_parameter<double>("lookahead_min", 0.35), 0.35);
  lookahead_max_ = ClampPositive(
      this->declare_parameter<double>("lookahead_max", 1.2), 1.2);
  lookahead_gain_ =
      ClampNonNegative(this->declare_parameter<double>("lookahead_gain", 0.35));
  lookahead_base_ =
      ClampNonNegative(this->declare_parameter<double>("lookahead_base", 0.35));
  min_forward_distance_ = ClampNonNegative(
      this->declare_parameter<double>("min_forward_distance", 0.05));
  steering_limit_ = ClampPositive(
      this->declare_parameter<double>("steering_limit", 0.45), 0.45);
  max_steering_delta_ = ClampPositive(
      this->declare_parameter<double>("max_steering_delta", 0.12), 0.12);

  min_speed_ =
      ClampNonNegative(this->declare_parameter<double>("min_speed", 0.25));
  max_speed_ =
      ClampNonNegative(this->declare_parameter<double>("max_speed", 1.0));
  if (max_speed_ < min_speed_) {
    max_speed_ = min_speed_;
  }
  curvature_speed_gain_ = ClampNonNegative(
      this->declare_parameter<double>("curvature_speed_gain", 1.5));
  steering_speed_gain_ = ClampNonNegative(
      this->declare_parameter<double>("steering_speed_gain", 1.0));
  short_path_speed_scale_ = std::clamp(
      this->declare_parameter<double>("short_path_speed_scale", 0.6), 0.0, 1.0);
  stop_on_invalid_path_ =
      this->declare_parameter<bool>("stop_on_invalid_path", true);
  debug_ = this->declare_parameter<bool>("debug", false);
  expected_frame_id_ =
      this->declare_parameter<std::string>("expected_frame_id", "base_link");
  velocity_source_ =
      this->declare_parameter<std::string>("velocity_source", "odometry");

  if (velocity_source_ != "command" && velocity_source_ != "float32" &&
      velocity_source_ != "odometry") {
    RCLCPP_WARN(
        this->get_logger(),
        "Unsupported velocity_source '%s'. Falling back to command speed.",
        velocity_source_.c_str());
    velocity_source_ = "command";
  }
}

void PurePursuitControllerComponent::TrajectoryCallback(
    const nav_msgs::msg::Path::SharedPtr msg) {
  if (msg->poses.empty()) {
    PublishStop(msg->header, "empty trajectory");
    return;
  }

  if (!expected_frame_id_.empty() && !msg->header.frame_id.empty() &&
      msg->header.frame_id != expected_frame_id_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Trajectory frame_id is '%s', but this controller expects '%s'. "
        "No TF transform is applied.",
        msg->header.frame_id.c_str(), expected_frame_id_.c_str());
  }

  const double lookahead_distance = ComputeLookaheadDistance();
  const auto target = SelectTargetPoint(*msg, lookahead_distance);
  if (!target.has_value()) {
    PublishStop(msg->header, "no valid forward target");
    return;
  }

  double steering_angle = ComputeSteeringAngle(*target);
  steering_angle = ApplySteeringRateLimit(steering_angle);
  const double speed = ComputeSpeed(steering_angle, *target, msg->poses.size());

  ackermann_msgs::msg::AckermannDriveStamped drive_msg;
  drive_msg.header = msg->header;
  drive_msg.drive.speed = static_cast<float>(speed);
  drive_msg.drive.steering_angle = static_cast<float>(steering_angle);
  drive_pub_->publish(drive_msg);

  last_command_speed_ = speed;
  if (velocity_source_ == "command") {
    current_speed_ = speed;
  }

  if (debug_) {
    PublishDebugMarkers(*msg, *target, lookahead_distance, steering_angle);
  }
}

void PurePursuitControllerComponent::VelocityCallback(
    const std_msgs::msg::Float32::SharedPtr msg) {
  current_speed_ = ClampNonNegative(static_cast<double>(msg->data));
}

void PurePursuitControllerComponent::OdometryCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  current_speed_ = ClampNonNegative(SpeedFromVector(
      msg->twist.twist.linear.x, msg->twist.twist.linear.y,
      msg->twist.twist.linear.z));
}

std::optional<PurePursuitControllerComponent::TargetPoint>
PurePursuitControllerComponent::SelectTargetPoint(
    const nav_msgs::msg::Path &path, double lookahead_distance) const {
  std::optional<TargetPoint> fallback;
  double cumulative_distance = 0.0;
  geometry_msgs::msg::Point previous_point;
  bool has_previous = false;

  for (std::size_t i = 0; i < path.poses.size(); ++i) {
    const auto &point = path.poses[i].pose.position;
    if (!IsFinitePoint(point) || point.x < min_forward_distance_) {
      continue;
    }

    if (has_previous) {
      cumulative_distance +=
          std::hypot(point.x - previous_point.x, point.y - previous_point.y);
    } else {
      cumulative_distance = Distance2D(point);
    }
    previous_point = point;
    has_previous = true;

    TargetPoint candidate;
    candidate.point = point;
    candidate.distance = Distance2D(point);
    candidate.index = i;
    fallback = candidate;

    if (cumulative_distance >= lookahead_distance ||
        candidate.distance >= lookahead_distance) {
      return candidate;
    }
  }

  return fallback;
}

double PurePursuitControllerComponent::ComputeLookaheadDistance() const {
  const double raw_lookahead = lookahead_base_ + lookahead_gain_ * current_speed_;
  return std::clamp(raw_lookahead, lookahead_min_, lookahead_max_);
}

double PurePursuitControllerComponent::ComputeSteeringAngle(
    const TargetPoint &target) const {
  const double x = target.point.x;
  const double y = target.point.y;
  const double distance_sq = std::max(1.0e-6, x * x + y * y);
  const double curvature = 2.0 * y / distance_sq;
  const double steering_angle = std::atan(wheelbase_ * curvature);
  return std::clamp(steering_angle, -steering_limit_, steering_limit_);
}

double PurePursuitControllerComponent::ComputeSpeed(
    double steering_angle, const TargetPoint &target, std::size_t path_size) const {
  const double x = target.point.x;
  const double y = target.point.y;
  const double distance_sq = std::max(1.0e-6, x * x + y * y);
  const double curvature = std::abs(2.0 * y / distance_sq);

  double speed = max_speed_ / (1.0 + curvature_speed_gain_ * curvature);
  const double steer_ratio =
      std::min(1.0, std::abs(steering_angle) / std::max(1.0e-6, steering_limit_));
  speed *= 1.0 / (1.0 + steering_speed_gain_ * steer_ratio);

  if (path_size < 3U || target.index + 1U >= path_size) {
    speed *= short_path_speed_scale_;
  }

  return std::clamp(speed, min_speed_, max_speed_);
}

double PurePursuitControllerComponent::ApplySteeringRateLimit(
    double steering_angle) {
  const double limited = std::clamp(steering_angle,
                                   last_steering_angle_ - max_steering_delta_,
                                   last_steering_angle_ + max_steering_delta_);
  if (std::abs(limited - steering_angle) > 1.0e-6) {
    RCLCPP_DEBUG(this->get_logger(), "Steering rate limited: %.3f -> %.3f",
                 steering_angle, limited);
  }
  last_steering_angle_ = limited;
  return limited;
}

void PurePursuitControllerComponent::PublishStop(
    const std_msgs::msg::Header &header, const std::string &reason) {
  ackermann_msgs::msg::AckermannDriveStamped drive_msg;
  drive_msg.header = header;
  if (stop_on_invalid_path_) {
    drive_msg.drive.speed = 0.0F;
    drive_msg.drive.steering_angle = 0.0F;
    last_command_speed_ = 0.0;
    current_speed_ = 0.0;
  } else {
    drive_msg.drive.speed = static_cast<float>(last_command_speed_);
    drive_msg.drive.steering_angle = static_cast<float>(last_steering_angle_);
  }
  drive_pub_->publish(drive_msg);
  RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                       "Pure Pursuit stop/fallback: %s", reason.c_str());
}

void PurePursuitControllerComponent::PublishDebugMarkers(
    const nav_msgs::msg::Path &path, const TargetPoint &target,
    double lookahead_distance, double steering_angle) {
  if (!target_marker_pub_ || !lookahead_marker_pub_ || !target_line_marker_pub_) {
    return;
  }

  visualization_msgs::msg::Marker target_marker;
  target_marker.header = path.header;
  target_marker.ns = "pure_pursuit_controller_target";
  target_marker.id = 0;
  target_marker.type = visualization_msgs::msg::Marker::SPHERE;
  target_marker.action = visualization_msgs::msg::Marker::ADD;
  target_marker.pose.position = target.point;
  target_marker.pose.orientation.w = 1.0;
  target_marker.scale.x = 0.08;
  target_marker.scale.y = 0.08;
  target_marker.scale.z = 0.08;
  target_marker.color.r = 0.1F;
  target_marker.color.g = 1.0F;
  target_marker.color.b = 0.2F;
  target_marker.color.a = 0.95F;
  target_marker_pub_->publish(target_marker);

  visualization_msgs::msg::Marker target_line_marker;
  target_line_marker.header = path.header;
  target_line_marker.ns = "pure_pursuit_controller_target_line";
  target_line_marker.id = 0;
  target_line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  target_line_marker.action = visualization_msgs::msg::Marker::ADD;
  target_line_marker.pose.orientation.w = 1.0;
  target_line_marker.scale.x = 0.025;
  target_line_marker.color.r = 0.1F;
  target_line_marker.color.g = 1.0F;
  target_line_marker.color.b = 0.2F;
  target_line_marker.color.a = 0.9F;
  geometry_msgs::msg::Point origin;
  origin.x = 0.0;
  origin.y = 0.0;
  origin.z = 0.0;
  target_line_marker.points.push_back(origin);
  target_line_marker.points.push_back(target.point);
  target_line_marker_pub_->publish(target_line_marker);

  visualization_msgs::msg::Marker lookahead_marker;
  lookahead_marker.header = path.header;
  lookahead_marker.ns = "pure_pursuit_controller_lookahead";
  lookahead_marker.id = 0;
  lookahead_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  lookahead_marker.action = visualization_msgs::msg::Marker::ADD;
  lookahead_marker.pose.orientation.w = 1.0;
  lookahead_marker.scale.x = lookahead_distance * 2.0;
  lookahead_marker.scale.y = lookahead_distance * 2.0;
  lookahead_marker.scale.z = 0.01;
  lookahead_marker.color.r = 0.2F;
  lookahead_marker.color.g = 0.6F;
  lookahead_marker.color.b = 1.0F;
  lookahead_marker.color.a = 0.18F;
  lookahead_marker_pub_->publish(lookahead_marker);

  RCLCPP_DEBUG(this->get_logger(),
               "target idx=%zu point=(%.3f, %.3f), Ld=%.3f, steer=%.3f",
               target.index, target.point.x, target.point.y, lookahead_distance,
               steering_angle);
}

} // namespace pure_pursuit_controller

RCLCPP_COMPONENTS_REGISTER_NODE(
    pure_pursuit_controller::PurePursuitControllerComponent)
