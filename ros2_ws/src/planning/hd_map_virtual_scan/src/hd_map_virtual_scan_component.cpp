#include "hd_map_virtual_scan/hd_map_virtual_scan_component.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "yaml-cpp/yaml.h"

namespace hd_map_virtual_scan {

namespace {

constexpr double kEpsilon = 1.0e-9;

double PositiveOrFallback(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

std::string ReadString(const YAML::Node &node, const std::string &fallback) {
  if (!node) {
    return fallback;
  }
  try {
    return node.as<std::string>();
  } catch (const YAML::Exception &) {
    return fallback;
  }
}

bool ReadBool(const YAML::Node &node, bool fallback) {
  if (!node) {
    return fallback;
  }
  try {
    return node.as<bool>();
  } catch (const YAML::Exception &) {
    return fallback;
  }
}

} // namespace

HdMapVirtualScanComponent::HdMapVirtualScanComponent(
    const rclcpp::NodeOptions &options)
    : Node("hd_map_virtual_scan", options) {
  LoadParameters();

  if (!LoadHdMap()) {
    RCLCPP_ERROR(this->get_logger(),
                 "HD map virtual scan node started without usable lane bounds.");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  scan_pub_ =
      this->create_publisher<sensor_msgs::msg::LaserScan>("scan",
                                                          rclcpp::SensorDataQoS());
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", rclcpp::SensorDataQoS(),
      std::bind(&HdMapVirtualScanComponent::OdometryCallback, this,
                std::placeholders::_1));

  if (publish_debug_markers_) {
    debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "debug_markers", rclcpp::QoS(1));
  }

  RCLCPP_INFO(this->get_logger(),
              "HD map virtual scan initialized (segments=%zu, points=%zu, "
              "angle=[%.3f, %.3f], range=[%.3f, %.3f], frame=%s)",
              segments_.size(), scan_points_, angle_min_, angle_max_,
              range_min_, range_max_, scan_frame_id_.c_str());
}

void HdMapVirtualScanComponent::LoadParameters() {
  hd_map_yaml_path_ = this->declare_parameter<std::string>("hd_map_yaml_path", "");
  frame_id_override_ =
      this->declare_parameter<std::string>("frame_id_override", "");
  scan_frame_id_ = this->declare_parameter<std::string>("scan_frame_id", "laser");
  lane_id_filter_ = this->declare_parameter<std::string>("lane_id", "");
  use_primary_lane_only_ =
      this->declare_parameter<bool>("use_primary_lane_only", false);
  include_left_bound_ =
      this->declare_parameter<bool>("include_left_bound", true);
  include_right_bound_ =
      this->declare_parameter<bool>("include_right_bound", true);
  include_centerline_ =
      this->declare_parameter<bool>("include_centerline", false);
  publish_inf_for_miss_ =
      this->declare_parameter<bool>("publish_inf_for_miss", false);
  publish_debug_markers_ =
      this->declare_parameter<bool>("publish_debug_markers", false);
  warn_on_frame_mismatch_ =
      this->declare_parameter<bool>("warn_on_frame_mismatch", true);

  angle_min_ = this->declare_parameter<double>("angle_min", angle_min_);
  angle_max_ = this->declare_parameter<double>("angle_max", angle_max_);
  range_min_ = PositiveOrFallback(
      this->declare_parameter<double>("range_min", range_min_), range_min_);
  range_max_ = PositiveOrFallback(
      this->declare_parameter<double>("range_max", range_max_), range_max_);
  scan_time_ = std::max(0.0, this->declare_parameter<double>("scan_time", 0.0));
  lidar_x_ = this->declare_parameter<double>("lidar_x", lidar_x_);
  lidar_y_ = this->declare_parameter<double>("lidar_y", lidar_y_);
  lidar_yaw_ = this->declare_parameter<double>("lidar_yaw", lidar_yaw_);

  const int scan_points_param = this->declare_parameter<int>("scan_points", 320);
  scan_points_ = scan_points_param > 1 ? static_cast<std::size_t>(scan_points_param)
                                       : 320U;
  if (range_max_ <= range_min_) {
    RCLCPP_WARN(this->get_logger(),
                "range_max must be greater than range_min. Falling back to 12.0.");
    range_max_ = 12.0;
  }
  if (angle_max_ <= angle_min_) {
    RCLCPP_WARN(this->get_logger(),
                "angle_max must be greater than angle_min. Falling back to [-pi, pi].");
    angle_min_ = -3.14159265358979323846;
    angle_max_ = 3.14159265358979323846;
  }
}

bool HdMapVirtualScanComponent::LoadHdMap() {
  segments_.clear();

  if (hd_map_yaml_path_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Parameter hd_map_yaml_path is empty.");
    return false;
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(hd_map_yaml_path_);
  } catch (const YAML::Exception &exc) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load HD map YAML '%s': %s",
                 hd_map_yaml_path_.c_str(), exc.what());
    return false;
  }

  map_frame_id_ = frame_id_override_.empty()
                      ? ReadString(root["frame_id"], "map")
                      : frame_id_override_;
  const std::string primary_lane_id =
      ReadString(root["primary_lane_id"], std::string());

  const YAML::Node lanes = root["lanes"];
  if (!lanes || !lanes.IsSequence()) {
    RCLCPP_ERROR(this->get_logger(), "HD map YAML has no lanes sequence: %s",
                 hd_map_yaml_path_.c_str());
    return false;
  }

  for (const auto &lane : lanes) {
    const std::string lane_id = ReadString(lane["id"], std::string());
    if (!lane_id_filter_.empty() && lane_id != lane_id_filter_) {
      continue;
    }
    if (lane_id_filter_.empty() && use_primary_lane_only_ &&
        !primary_lane_id.empty() && lane_id != primary_lane_id) {
      continue;
    }

    const bool closed_loop = ReadBool(lane["closed_loop"], true);
    if (include_left_bound_) {
      AppendLaneBoundSegments(lane, "left_bound", closed_loop);
    }
    if (include_right_bound_) {
      AppendLaneBoundSegments(lane, "right_bound", closed_loop);
    }
    if (include_centerline_) {
      AppendLaneBoundSegments(lane, "centerline", closed_loop);
    }
  }

  if (segments_.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "No virtual-scan segments were loaded from HD map YAML: %s",
                 hd_map_yaml_path_.c_str());
    return false;
  }

  RCLCPP_INFO(this->get_logger(),
              "Loaded HD map virtual scan geometry: %s (frame=%s, segments=%zu)",
              hd_map_yaml_path_.c_str(), map_frame_id_.c_str(), segments_.size());
  return true;
}

void HdMapVirtualScanComponent::AppendLaneBoundSegments(
    const YAML::Node &lane, const std::string &field_name, bool closed_loop) {
  AppendPolylineSegments(ReadPoints(lane[field_name]), closed_loop);
}

void HdMapVirtualScanComponent::AppendPolylineSegments(
    const std::vector<Point2D> &points, bool closed_loop) {
  if (points.size() < 2U) {
    return;
  }
  for (std::size_t index = 1; index < points.size(); ++index) {
    segments_.push_back(Segment2D{points[index - 1U], points[index]});
  }
  if (closed_loop && points.size() >= 3U) {
    segments_.push_back(Segment2D{points.back(), points.front()});
  }
}

std::vector<Point2D> HdMapVirtualScanComponent::ReadPoints(
    const YAML::Node &node) const {
  std::vector<Point2D> points;
  if (!node || !node.IsSequence()) {
    return points;
  }
  for (const auto &row : node) {
    if (!row.IsSequence() || row.size() < 2U) {
      continue;
    }
    try {
      points.push_back(Point2D{row[0].as<double>(), row[1].as<double>()});
    } catch (const YAML::Exception &) {
      continue;
    }
  }
  return points;
}

void HdMapVirtualScanComponent::OdometryCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (segments_.empty()) {
    return;
  }

  nav_msgs::msg::Odometry odom_in_map = *msg;

  if (!map_frame_id_.empty() && !msg->header.frame_id.empty() &&
      msg->header.frame_id != map_frame_id_) {
    if (warn_on_frame_mismatch_) {
      RCLCPP_INFO_ONCE(
          this->get_logger(),
          "Odometry frame '%s' differs from HD map frame '%s'. Attempting to transform via tf2.",
          msg->header.frame_id.c_str(), map_frame_id_.c_str());
    }

    geometry_msgs::msg::PoseStamped pose_in;
    pose_in.header = msg->header;
    pose_in.pose = msg->pose.pose;

    try {
      geometry_msgs::msg::PoseStamped pose_out = tf_buffer_->transform(
          pose_in, map_frame_id_, tf2::durationFromSec(0.1));

      odom_in_map.header.frame_id = map_frame_id_;
      odom_in_map.pose.pose = pose_out.pose;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Could not transform odometry from '%s' to '%s': %s",
          msg->header.frame_id.c_str(), map_frame_id_.c_str(), ex.what());
      // Fallback: Proceed with untransformed odometry
    }
  }

  sensor_msgs::msg::LaserScan scan = BuildScan(odom_in_map);
  scan_pub_->publish(scan);
  if (publish_debug_markers_) {
    PublishDebugMarkers(scan);
  }
}

sensor_msgs::msg::LaserScan HdMapVirtualScanComponent::BuildScan(
    const nav_msgs::msg::Odometry &odom) {
  sensor_msgs::msg::LaserScan scan;
  scan.header.stamp = odom.header.stamp;
  scan.header.frame_id = scan_frame_id_;
  scan.angle_min = static_cast<float>(angle_min_);
  scan.angle_max = static_cast<float>(angle_max_);
  scan.angle_increment = static_cast<float>((angle_max_ - angle_min_) /
                                            static_cast<double>(scan_points_ - 1U));
  scan.time_increment =
      scan_points_ > 1U ? static_cast<float>(scan_time_ / static_cast<double>(scan_points_ - 1U))
                        : 0.0F;
  scan.scan_time = static_cast<float>(scan_time_);
  scan.range_min = static_cast<float>(range_min_);
  scan.range_max = static_cast<float>(range_max_);
  scan.ranges.assign(scan_points_,
                     publish_inf_for_miss_
                         ? std::numeric_limits<float>::infinity()
                         : static_cast<float>(range_max_));

  const double base_yaw = YawFromQuaternion(odom);
  const double cos_base = std::cos(base_yaw);
  const double sin_base = std::sin(base_yaw);
  const Point2D origin{
      odom.pose.pose.position.x + cos_base * lidar_x_ - sin_base * lidar_y_,
      odom.pose.pose.position.y + sin_base * lidar_x_ + cos_base * lidar_y_,
  };
  const double scan_yaw = base_yaw + lidar_yaw_;

  for (std::size_t beam = 0; beam < scan_points_; ++beam) {
    const double beam_angle =
        angle_min_ + static_cast<double>(beam) * static_cast<double>(scan.angle_increment);
    const Point2D direction{std::cos(scan_yaw + beam_angle),
                            std::sin(scan_yaw + beam_angle)};

    double best_distance = range_max_ + 1.0;
    for (const auto &segment : segments_) {
      const double distance = RaySegmentDistance(origin, direction, segment);
      if (distance >= range_min_ && distance <= range_max_ &&
          distance < best_distance) {
        best_distance = distance;
      }
    }

    if (best_distance <= range_max_) {
      scan.ranges[beam] = static_cast<float>(best_distance);
    }
  }

  return scan;
}

void HdMapVirtualScanComponent::PublishDebugMarkers(
    const sensor_msgs::msg::LaserScan &scan) {
  if (!debug_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker hits;
  hits.header = scan.header;
  hits.ns = "hd_map_virtual_scan/hits";
  hits.id = 0;
  hits.type = visualization_msgs::msg::Marker::POINTS;
  hits.action = visualization_msgs::msg::Marker::ADD;
  hits.pose.orientation.w = 1.0;
  hits.scale.x = 0.035;
  hits.scale.y = 0.035;
  hits.color.r = 0.1F;
  hits.color.g = 0.8F;
  hits.color.b = 1.0F;
  hits.color.a = 0.8F;

  for (std::size_t beam = 0; beam < scan.ranges.size(); ++beam) {
    const float range = scan.ranges[beam];
    if (!std::isfinite(range) || range >= scan.range_max) {
      continue;
    }
    const double angle =
        static_cast<double>(scan.angle_min) +
        static_cast<double>(beam) * static_cast<double>(scan.angle_increment);
    geometry_msgs::msg::Point point;
    point.x = std::cos(angle) * static_cast<double>(range);
    point.y = std::sin(angle) * static_cast<double>(range);
    point.z = 0.0;
    hits.points.push_back(point);
  }

  markers.markers.push_back(std::move(hits));
  debug_pub_->publish(markers);
}

double HdMapVirtualScanComponent::RaySegmentDistance(
    const Point2D &origin, const Point2D &direction,
    const Segment2D &segment) const {
  const Point2D segment_vec{segment.b.x - segment.a.x,
                            segment.b.y - segment.a.y};
  const double denom = Cross(direction, segment_vec);
  if (std::abs(denom) < kEpsilon) {
    return std::numeric_limits<double>::infinity();
  }

  const Point2D diff{segment.a.x - origin.x, segment.a.y - origin.y};
  const double ray_distance = Cross(diff, segment_vec) / denom;
  const double segment_ratio = Cross(diff, direction) / denom;
  if (ray_distance < 0.0 || segment_ratio < -kEpsilon ||
      segment_ratio > 1.0 + kEpsilon) {
    return std::numeric_limits<double>::infinity();
  }
  return ray_distance;
}

double HdMapVirtualScanComponent::Cross(const Point2D &a, const Point2D &b) {
  return a.x * b.y - a.y * b.x;
}

double HdMapVirtualScanComponent::YawFromQuaternion(
    const nav_msgs::msg::Odometry &odom) {
  const auto &q = odom.pose.pose.orientation;
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

} // namespace hd_map_virtual_scan

RCLCPP_COMPONENTS_REGISTER_NODE(hd_map_virtual_scan::HdMapVirtualScanComponent)
