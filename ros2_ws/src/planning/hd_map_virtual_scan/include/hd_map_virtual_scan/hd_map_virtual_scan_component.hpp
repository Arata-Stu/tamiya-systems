#ifndef HD_MAP_VIRTUAL_SCAN__HD_MAP_VIRTUAL_SCAN_COMPONENT_HPP_
#define HD_MAP_VIRTUAL_SCAN__HD_MAP_VIRTUAL_SCAN_COMPONENT_HPP_

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yaml-cpp/yaml.h"

namespace hd_map_virtual_scan {

struct Point2D {
  double x{0.0};
  double y{0.0};
};

struct Segment2D {
  Point2D a;
  Point2D b;
};

class HdMapVirtualScanComponent : public rclcpp::Node {
public:
  explicit HdMapVirtualScanComponent(const rclcpp::NodeOptions &options);

private:
  void LoadParameters();
  bool LoadHdMap();
  void AppendLaneBoundSegments(const YAML::Node &lane, const std::string &field_name,
                               bool closed_loop);
  void AppendPolylineSegments(const std::vector<Point2D> &points,
                              bool closed_loop);
  std::vector<Point2D> ReadPoints(const YAML::Node &node) const;

  void OdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  sensor_msgs::msg::LaserScan BuildScan(const nav_msgs::msg::Odometry &odom);
  void PublishDebugMarkers(const sensor_msgs::msg::LaserScan &scan);

  double RaySegmentDistance(const Point2D &origin, const Point2D &direction,
                            const Segment2D &segment) const;
  static double Cross(const Point2D &a, const Point2D &b);
  static double YawFromQuaternion(const nav_msgs::msg::Odometry &odom);

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::vector<Segment2D> segments_;

  std::string hd_map_yaml_path_;
  std::string map_frame_id_;
  std::string frame_id_override_;
  std::string scan_frame_id_;
  std::string lane_id_filter_;
  bool use_primary_lane_only_{false};
  bool include_left_bound_{true};
  bool include_right_bound_{true};
  bool include_centerline_{false};
  bool publish_inf_for_miss_{false};
  bool publish_debug_markers_{false};
  bool warn_on_frame_mismatch_{true};

  double angle_min_{-3.14159265358979323846};
  double angle_max_{3.14159265358979323846};
  double range_min_{0.02};
  double range_max_{12.0};
  double scan_time_{0.0};
  double lidar_x_{0.2725};
  double lidar_y_{0.0};
  double lidar_yaw_{0.0};
  std::size_t scan_points_{320U};

  // Processing time statistics (reset on node restart only)
  std::size_t perf_count_{0};
  double perf_total_ms_{0.0};
  double perf_max_ms_{0.0};
};

} // namespace hd_map_virtual_scan

#endif // HD_MAP_VIRTUAL_SCAN__HD_MAP_VIRTUAL_SCAN_COMPONENT_HPP_
