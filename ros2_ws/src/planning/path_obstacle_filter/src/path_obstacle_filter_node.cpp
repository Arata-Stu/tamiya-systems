#include "path_obstacle_filter/path_obstacle_filter_node.hpp"

#include <cmath>
#include <limits>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace path_obstacle_filter {

PathObstacleFilterNode::PathObstacleFilterNode(const rclcpp::NodeOptions &options)
    : Node("path_obstacle_filter_node", options) {
  LoadParameters();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  tracked_object_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/perception/tracked_object", rclcpp::QoS(10),
      std::bind(&PathObstacleFilterNode::TrackedObjectCallback, this, std::placeholders::_1));

  trajectory_sub_ = create_subscription<nav_msgs::msg::Path>(
      "/trajectory", rclcpp::QoS(10),
      std::bind(&PathObstacleFilterNode::TrajectoryCallback, this, std::placeholders::_1));

  obstacle_on_path_pub_ = create_publisher<std_msgs::msg::Bool>("~/obstacle_on_path", rclcpp::QoS(10));
  deceleration_requested_pub_ = create_publisher<std_msgs::msg::Bool>("~/deceleration_requested", rclcpp::QoS(10));
  avoidance_requested_pub_ = create_publisher<std_msgs::msg::Bool>("~/avoidance_requested", rclcpp::QoS(10));
  following_requested_pub_ = create_publisher<std_msgs::msg::Bool>("~/following_requested", rclcpp::QoS(10));
  target_speed_pub_ = create_publisher<std_msgs::msg::Float32>("~/target_speed_m_s", rclcpp::QoS(10));
  obstacle_distance_pub_ = create_publisher<std_msgs::msg::Float32>("~/obstacle_distance_m", rclcpp::QoS(10));
  obstacle_lateral_pub_ = create_publisher<std_msgs::msg::Float32>("~/obstacle_lateral_m", rclcpp::QoS(10));
  
  if (publish_debug_markers_) {
    debug_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", rclcpp::QoS(10));
  }

  last_target_detected_time_ = this->now();

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100), // 10 Hz
      std::bind(&PathObstacleFilterNode::TimerCallback, this));

  RCLCPP_INFO(this->get_logger(), "PathObstacleFilterNode initialized.");
}

void PathObstacleFilterNode::LoadParameters() {
  forward_distance_m_ = declare_parameter<double>("forward_distance_m", 3.0);
  lateral_half_width_m_ = declare_parameter<double>("lateral_half_width_m", 0.25);
  stopped_speed_threshold_m_s_ = declare_parameter<double>("stopped_speed_threshold_m_s", 0.3);
  deceleration_on_obstacle_ = declare_parameter<bool>("deceleration_on_obstacle", true);
  obstacle_timeout_sec_ = declare_parameter<double>("obstacle_timeout_sec", 0.5);
  map_frame_ = declare_parameter<std::string>("map_frame", "map");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  tf_timeout_sec_ = declare_parameter<double>("tf_timeout_sec", 0.05);
  publish_debug_markers_ = declare_parameter<bool>("publish_debug_markers", true);
}

void PathObstacleFilterNode::TrackedObjectCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_tracked_object_ = msg;
}

void PathObstacleFilterNode::TrajectoryCallback(const nav_msgs::msg::Path::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_trajectory_ = msg;
}

void PathObstacleFilterNode::TimerCallback() {
  bool obstacle_on_path = false;
  bool is_stopped = false;
  double obstacle_speed = 0.0;
  double min_distance_m = -1.0;
  double min_lateral_m = -1.0;
  double obs_x = 0.0;
  double obs_y = 0.0;

  {
    std::scoped_lock lock(data_mutex_);

    if (latest_tracked_object_ && latest_trajectory_) {
      const auto now = this->now();
      const double object_age = (now - latest_tracked_object_->header.stamp).seconds();

      if (object_age <= obstacle_timeout_sec_) {
        try {
          // Transform obstacle pose from its frame (e.g. odom) to base_link
          geometry_msgs::msg::PoseStamped pose_in;
          pose_in.header = latest_tracked_object_->header;
          pose_in.pose = latest_tracked_object_->pose.pose;
          
          const auto tf_to_base = tf_buffer_->lookupTransform(
              base_frame_, pose_in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_sec_));
              
          geometry_msgs::msg::PoseStamped pose_base;
          tf2::doTransform(pose_in, pose_base, tf_to_base);

          obs_x = pose_base.pose.position.x;
          obs_y = pose_base.pose.position.y;
          
          // Calculate absolute speed from twist (which is in odom frame usually)
          const double vx = latest_tracked_object_->twist.twist.linear.x;
          const double vy = latest_tracked_object_->twist.twist.linear.y;
          obstacle_speed = std::hypot(vx, vy);
          is_stopped = (obstacle_speed < stopped_speed_threshold_m_s_);
          
          if (std::hypot(obs_x, obs_y) <= forward_distance_m_) {
            // Find closest waypoint on trajectory
            double closest_sq = std::numeric_limits<double>::max();
            double closest_lateral = 0.0;
            double closest_longitudinal = 0.0;
            
            for (const auto & pose : latest_trajectory_->poses) {
              const double px = pose.pose.position.x;
              const double py = pose.pose.position.y;
              const double dx = px - obs_x;
              const double dy = py - obs_y;
              const double dist_sq = dx * dx + dy * dy;
              
              if (dist_sq < closest_sq) {
                closest_sq = dist_sq;
                closest_lateral = std::sqrt(dist_sq);
                closest_longitudinal = std::hypot(px, py);
              }
            }

            if (closest_lateral <= lateral_half_width_m_) {
              obstacle_on_path = true;
              min_distance_m = closest_longitudinal;
              min_lateral_m = closest_lateral;
            }
          }
        } catch (const tf2::TransformException &ex) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "Tracked object TF lookup failed: %s", ex.what());
        }
      }
    }
  }

  std_msgs::msg::Bool on_path_msg;
  on_path_msg.data = obstacle_on_path;
  obstacle_on_path_pub_->publish(on_path_msg);

  std_msgs::msg::Bool decel_msg;
  decel_msg.data = obstacle_on_path && deceleration_on_obstacle_;
  deceleration_requested_pub_->publish(decel_msg);
  
  std_msgs::msg::Bool avoid_msg;
  avoid_msg.data = obstacle_on_path && is_stopped;
  avoidance_requested_pub_->publish(avoid_msg);
  
  std_msgs::msg::Bool follow_msg;
  follow_msg.data = obstacle_on_path && !is_stopped;
  following_requested_pub_->publish(follow_msg);

  if (obstacle_on_path) {
    std_msgs::msg::Float32 target_spd_msg;
    target_spd_msg.data = is_stopped ? 0.0 : obstacle_speed;
    target_speed_pub_->publish(target_spd_msg);

    std_msgs::msg::Float32 dist_msg;
    dist_msg.data = min_distance_m;
    obstacle_distance_pub_->publish(dist_msg);

    std_msgs::msg::Float32 lat_msg;
    lat_msg.data = min_lateral_m;
    obstacle_lateral_pub_->publish(lat_msg);
  }

  if (publish_debug_markers_) {
    PublishDebugMarkers(obstacle_on_path, obs_x, obs_y, min_distance_m, min_lateral_m);
  }
}

void PathObstacleFilterNode::PublishDebugMarkers(bool on_path, double obs_x, double obs_y, double distance_m, double lateral_m) {
  if (!debug_markers_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  rclcpp::Time now = this->now();

  geometry_msgs::msg::TransformStamped tf_to_map;
  bool can_transform = false;
  try {
    tf_to_map = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_sec_));
    can_transform = true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Debug markers TF lookup failed: %s", ex.what());
    // If we can't transform to map, we can publish in base_link, but map is usually better for static viewing.
    // For simplicity, we just publish in base_frame if map transform fails.
    can_transform = false;
  }

  std::string target_frame = can_transform ? map_frame_ : base_frame_;

  auto create_base_marker = [&](int id, int type) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = target_frame;
    marker.header.stamp = now;
    marker.ns = "path_obstacle_filter";
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    return marker;
  };

  auto transform_point = [&](double x, double y) {
    geometry_msgs::msg::Point pt;
    if (can_transform) {
      geometry_msgs::msg::PointStamped pt_base;
      pt_base.header.frame_id = base_frame_;
      pt_base.point.x = x;
      pt_base.point.y = y;
      pt_base.point.z = 0.0;
      
      geometry_msgs::msg::PointStamped pt_map;
      tf2::doTransform(pt_base, pt_map, tf_to_map);
      pt.x = pt_map.point.x;
      pt.y = pt_map.point.y;
      pt.z = pt_map.point.z;
    } else {
      pt.x = x;
      pt.y = y;
      pt.z = 0.0;
    }
    return pt;
  };

  // 1. Detection zone
  visualization_msgs::msg::Marker zone_marker = create_base_marker(0, visualization_msgs::msg::Marker::LINE_STRIP);
  zone_marker.scale.x = 0.05;
  zone_marker.color.r = 0.0f;
  zone_marker.color.g = 1.0f;
  zone_marker.color.b = 1.0f;
  zone_marker.color.a = 0.5f;

  zone_marker.points.push_back(transform_point(0.0, lateral_half_width_m_));
  zone_marker.points.push_back(transform_point(forward_distance_m_, lateral_half_width_m_));
  zone_marker.points.push_back(transform_point(forward_distance_m_, -lateral_half_width_m_));
  zone_marker.points.push_back(transform_point(0.0, -lateral_half_width_m_));
  zone_marker.points.push_back(transform_point(0.0, lateral_half_width_m_));
  marker_array.markers.push_back(zone_marker);

  // 2. Obstacle position
  visualization_msgs::msg::Marker obs_marker = create_base_marker(1, visualization_msgs::msg::Marker::SPHERE);
  obs_marker.scale.x = 0.3;
  obs_marker.scale.y = 0.3;
  obs_marker.scale.z = 0.3;
  
  if (obs_x > 0.0) { // Active obstacle
    obs_marker.pose.position = transform_point(obs_x, obs_y);
    obs_marker.color.a = 0.8f;
    if (on_path) {
      obs_marker.color.r = 1.0f;
      obs_marker.color.g = 0.0f;
      obs_marker.color.b = 0.0f;
    } else {
      obs_marker.color.r = 0.0f;
      obs_marker.color.g = 1.0f;
      obs_marker.color.b = 0.0f;
    }
  } else {
    // Hide
    obs_marker.color.a = 0.0f;
  }
  marker_array.markers.push_back(obs_marker);

  // 3. Text info
  visualization_msgs::msg::Marker text_marker = create_base_marker(2, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
  if (on_path && obs_x > 0.0) {
    text_marker.pose.position = transform_point(obs_x, obs_y);
    text_marker.pose.position.z += 0.5; // Hover above
    text_marker.scale.z = 0.2;
    text_marker.color.r = 1.0f;
    text_marker.color.g = 1.0f;
    text_marker.color.b = 1.0f;
    text_marker.color.a = 1.0f;
    text_marker.text = "Dist: " + std::to_string(distance_m).substr(0, 4) + "m\nLat: " + std::to_string(lateral_m).substr(0, 4) + "m";
  } else {
    text_marker.color.a = 0.0f;
  }
  marker_array.markers.push_back(text_marker);

  debug_markers_pub_->publish(marker_array);
}

}  // namespace path_obstacle_filter

RCLCPP_COMPONENTS_REGISTER_NODE(path_obstacle_filter::PathObstacleFilterNode)
