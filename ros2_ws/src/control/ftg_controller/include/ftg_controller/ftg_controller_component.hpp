#ifndef FTG_CONTROLLER__FTG_CONTROLLER_COMPONENT_HPP_
#define FTG_CONTROLLER__FTG_CONTROLLER_COMPONENT_HPP_

#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "ftg_controller/ftg_core.hpp"

namespace ftg_controller {

class FtgControllerComponent : public rclcpp::Node {
public:
  explicit FtgControllerComponent(const rclcpp::NodeOptions &options);

private:
  void LoadParameters();
  void ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void VelocityCallback(const std_msgs::msg::Float32::SharedPtr msg);

  void PublishDebugMarkers(const FtgResult &result,
                           const std_msgs::msg::Header &header);

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr velocity_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      drive_pub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      scan_proc_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      best_gap_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      best_point_marker_pub_;

  FtgCore core_;
  double last_command_speed_ = 0.0;
  bool use_velocity_topic_ = false;
};

} // namespace ftg_controller

#endif // FTG_CONTROLLER__FTG_CONTROLLER_COMPONENT_HPP_
