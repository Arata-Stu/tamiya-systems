#include "ftg_controller/ftg_controller_component.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace ftg_controller {

namespace {

double ClampNonNegative(double value) { return std::max(0.0, value); }

double ClampPositive(double value, double fallback) {
  if (value <= 0.0) {
    return fallback;
  }
  return value;
}

} // namespace

FtgControllerComponent::FtgControllerComponent(const rclcpp::NodeOptions &options)
    : Node("ftg_controller_node", options) {
  LoadParameters();

  drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "autonomous/cmd_drive", rclcpp::QoS(10));

  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      std::bind(&FtgControllerComponent::ScanCallback, this,
                std::placeholders::_1));

  if (use_velocity_topic_) {
    velocity_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "current_velocity", rclcpp::QoS(10),
        std::bind(&FtgControllerComponent::VelocityCallback, this,
                  std::placeholders::_1));
  }

  if (core_.GetParams().debug) {
    scan_proc_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/scan_proc/markers", rclcpp::QoS(1));
    best_gap_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/best_gap/markers", rclcpp::QoS(1));
    best_point_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "/best_points/marker", rclcpp::QoS(1));
  }

  const auto &params = core_.GetParams();
  RCLCPP_INFO(this->get_logger(),
              "FTG Controller initialized (mapping=%s, debug=%s, dynamic_radius=%s)",
              params.mapping ? "true" : "false",
              params.debug ? "true" : "false",
              params.use_dynamic_radius ? "true" : "false");
}

void FtgControllerComponent::LoadParameters() {
  FtgParams params;

  params.mapping = this->declare_parameter<bool>("mapping", false);
  params.debug = this->declare_parameter<bool>("debug", false);

  params.range_offset = this->declare_parameter<int>("range_offset", 180);
  params.preprocess_conv_size =
      this->declare_parameter<int>("preprocess_conv_size", 3);
  params.safety_radius = this->declare_parameter<int>("safety_radius", 3);

  params.max_lidar_dist =
      ClampPositive(this->declare_parameter<double>("max_lidar_dist", 8.0),
                    8.0);
  params.max_speed =
      ClampNonNegative(this->declare_parameter<double>("max_speed", 1.2));
  params.track_width =
      ClampPositive(this->declare_parameter<double>("track_width", 0.9), 0.9);

  params.use_dynamic_radius =
      this->declare_parameter<bool>("use_dynamic_radius", false);
  params.fixed_radius =
      ClampPositive(this->declare_parameter<double>("fixed_radius", 1.0), 1.0);
  params.max_gap_radius =
      ClampPositive(this->declare_parameter<double>("max_gap_radius", 5.0),
                    5.0);

  params.jump_threshold =
      ClampPositive(this->declare_parameter<double>("jump_threshold", 0.5),
                    0.5);
  params.steering_limit =
      ClampPositive(this->declare_parameter<double>("steering_limit", 0.4),
                    0.4);

  params.straights_steering_angle = ClampPositive(
      this->declare_parameter<double>("straights_steering_angle", 0.1745329252),
      0.1745329252);
  params.mild_curve_angle =
      ClampPositive(this->declare_parameter<double>("mild_curve_angle", 0.5235987756),
                    0.5235987756);
  params.ultra_straights_angle = ClampPositive(
      this->declare_parameter<double>("ultra_straights_angle", 0.0523598776),
      0.0523598776);

  const double default_corners_speed = params.max_speed * 0.5;
  const double default_mild_corners_speed = params.max_speed * 0.7;
  const double default_straights_speed = params.max_speed * 0.85;

  params.corners_speed = ClampNonNegative(
      this->declare_parameter<double>("corners_speed", default_corners_speed));
  params.mild_corners_speed =
      ClampNonNegative(this->declare_parameter<double>("mild_corners_speed",
                                                       default_mild_corners_speed));
  params.straights_speed = ClampNonNegative(
      this->declare_parameter<double>("straights_speed", default_straights_speed));
  params.ultra_straights_speed = ClampNonNegative(
      this->declare_parameter<double>("ultra_straights_speed", params.max_speed));
  params.mapping_speed = ClampNonNegative(
      this->declare_parameter<double>("mapping_speed", default_corners_speed));

  use_velocity_topic_ = this->declare_parameter<bool>("use_velocity_topic", false);

  core_.SetParams(params);
}

void FtgControllerComponent::ScanCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  if (!use_velocity_topic_) {
    core_.SetVelocity(last_command_speed_);
  }

  LidarScan scan;
  scan.ranges = msg->ranges;
  scan.angle_min = msg->angle_min;
  scan.angle_increment = msg->angle_increment;

  const FtgResult result = core_.Process(scan);

  ackermann_msgs::msg::AckermannDriveStamped drive_msg;
  drive_msg.header = msg->header;

  if (result.valid) {
    drive_msg.drive.speed = static_cast<float>(result.speed);
    drive_msg.drive.steering_angle = static_cast<float>(result.steering_angle);
    last_command_speed_ = result.speed;
  } else {
    drive_msg.drive.speed = 0.0F;
    drive_msg.drive.steering_angle = 0.0F;
    last_command_speed_ = 0.0;

    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "FTG failed to find a valid gap. Publishing stop command.");
  }

  drive_pub_->publish(drive_msg);

  if (core_.GetParams().debug) {
    PublishDebugMarkers(result, msg->header);
  }
}

void FtgControllerComponent::VelocityCallback(
    const std_msgs::msg::Float32::SharedPtr msg) {
  core_.SetVelocity(static_cast<double>(msg->data));
}

void FtgControllerComponent::PublishDebugMarkers(
    const FtgResult &result, const std_msgs::msg::Header &header) {
  if (!scan_proc_marker_pub_ || !best_gap_marker_pub_ || !best_point_marker_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray proc_array;
  visualization_msgs::msg::Marker proc_marker;
  proc_marker.header = header;
  proc_marker.ns = "scan_proc";
  proc_marker.id = 0;
  proc_marker.type = visualization_msgs::msg::Marker::POINTS;
  proc_marker.action = visualization_msgs::msg::Marker::ADD;
  proc_marker.pose.orientation.w = 1.0;
  proc_marker.scale.x = 0.03;
  proc_marker.scale.y = 0.03;
  proc_marker.color.r = 0.2F;
  proc_marker.color.g = 0.6F;
  proc_marker.color.b = 1.0F;
  proc_marker.color.a = 0.7F;

  proc_marker.points.reserve(result.proc_ranges.size());
  for (std::size_t i = 0; i < result.proc_ranges.size(); ++i) {
    geometry_msgs::msg::Point point;
    point.x = std::cos(result.proc_angles[i]) * result.proc_ranges[i];
    point.y = std::sin(result.proc_angles[i]) * result.proc_ranges[i];
    point.z = 0.0;
    proc_marker.points.push_back(std::move(point));
  }
  proc_array.markers.push_back(std::move(proc_marker));
  scan_proc_marker_pub_->publish(proc_array);

  visualization_msgs::msg::MarkerArray gap_array;
  visualization_msgs::msg::Marker gap_marker;
  gap_marker.header = header;
  gap_marker.ns = "best_gap";
  gap_marker.id = 0;
  gap_marker.type = visualization_msgs::msg::Marker::POINTS;
  gap_marker.pose.orientation.w = 1.0;
  gap_marker.scale.x = 0.05;
  gap_marker.scale.y = 0.05;
  gap_marker.color.r = 0.2F;
  gap_marker.color.g = 1.0F;
  gap_marker.color.b = 0.2F;
  gap_marker.color.a = 0.9F;

  if (result.valid && result.gap_left <= result.gap_right &&
      result.gap_right < result.proc_ranges.size()) {
    gap_marker.action = visualization_msgs::msg::Marker::ADD;
    gap_marker.points.reserve(result.gap_right - result.gap_left + 1U);

    for (std::size_t i = result.gap_left; i <= result.gap_right; ++i) {
      geometry_msgs::msg::Point point;
      point.x = std::cos(result.proc_angles[i]) * result.proc_ranges[i];
      point.y = std::sin(result.proc_angles[i]) * result.proc_ranges[i];
      point.z = 0.0;
      gap_marker.points.push_back(std::move(point));
    }
  } else {
    gap_marker.action = visualization_msgs::msg::Marker::DELETE;
  }

  gap_array.markers.push_back(std::move(gap_marker));
  best_gap_marker_pub_->publish(gap_array);

  visualization_msgs::msg::Marker best_point_marker;
  best_point_marker.header = header;
  best_point_marker.ns = "best_point";
  best_point_marker.id = 0;
  best_point_marker.type = visualization_msgs::msg::Marker::SPHERE;
  best_point_marker.pose.orientation.w = 1.0;
  best_point_marker.scale.x = 0.12;
  best_point_marker.scale.y = 0.12;
  best_point_marker.scale.z = 0.12;
  best_point_marker.color.r = 1.0F;
  best_point_marker.color.g = 0.3F;
  best_point_marker.color.b = 0.1F;
  best_point_marker.color.a = 1.0F;

  if (result.valid) {
    best_point_marker.action = visualization_msgs::msg::Marker::ADD;
    best_point_marker.pose.position.x = result.best_x;
    best_point_marker.pose.position.y = result.best_y;
    best_point_marker.pose.position.z = 0.0;
  } else {
    best_point_marker.action = visualization_msgs::msg::Marker::DELETE;
  }

  best_point_marker_pub_->publish(best_point_marker);
}

} // namespace ftg_controller

RCLCPP_COMPONENTS_REGISTER_NODE(ftg_controller::FtgControllerComponent)
