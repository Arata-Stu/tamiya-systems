#ifndef RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_NODE_HPP_
#define RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_NODE_HPP_

#include <string>

#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "raceline_path_publisher/raceline_path_publisher_core.hpp"

namespace raceline_path_publisher {

class RacelinePathPublisherNode : public rclcpp::Node {
public:
  RacelinePathPublisherNode();

private:
  void LoadParameters();
  bool LoadRaceline();
  void TimerCallback();
  void PublishGlobalPath(const rclcpp::Time &stamp);
  void PublishLocalPath(const rclcpp::Time &stamp);

  RacelinePathCore core_;

  std::string raceline_csv_path_;
  std::string direction_;
  std::string map_frame_;
  std::string base_frame_;
  double publish_rate_hz_ = 10.0;
  double local_path_length_m_ = 6.0;
  int max_local_points_ = 80;
  double tf_timeout_sec_ = 0.05;
  bool publish_global_path_ = true;
  bool publish_local_path_ = true;

  bool raceline_loaded_ = false;
  rclcpp::Time last_load_attempt_time_;
  rclcpp::Duration reload_retry_interval_{2, 0};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

} // namespace raceline_path_publisher

#endif // RACELINE_PATH_PUBLISHER__RACELINE_PATH_PUBLISHER_NODE_HPP_
