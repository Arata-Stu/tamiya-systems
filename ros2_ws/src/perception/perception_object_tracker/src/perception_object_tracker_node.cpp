#include "perception_object_tracker/perception_object_tracker_node.hpp"

#include <cmath>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace perception_object_tracker {

PerceptionObjectTrackerNode::PerceptionObjectTrackerNode(const rclcpp::NodeOptions &options)
    : Node("perception_object_tracker_node", options) {
  LoadParameters();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  target_detected_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/perception/classification/target_detected", rclcpp::QoS(10),
      std::bind(&PerceptionObjectTrackerNode::TargetDetectedCallback, this, std::placeholders::_1));

  obstacle_position_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "crop/obstacle_position", rclcpp::QoS(10),
      std::bind(&PerceptionObjectTrackerNode::ObstaclePositionCallback, this, std::placeholders::_1));

  tracked_object_pub_ = create_publisher<nav_msgs::msg::Odometry>("~/tracked_object", rclcpp::QoS(10));
  
  if (publish_debug_markers_) {
    debug_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/tracker_markers", rclcpp::QoS(10));
  }

  last_target_detected_time_ = this->now();

  const auto timer_period = std::chrono::duration<double>(1.0 / std::max(1.0, update_rate_hz_));
  timer_ = this->create_wall_timer(
      timer_period,
      std::bind(&PerceptionObjectTrackerNode::TimerCallback, this));

  RCLCPP_INFO(this->get_logger(), "PerceptionObjectTrackerNode initialized (alpha=%.2f, beta=%.2f).", alpha_, beta_);
}

void PerceptionObjectTrackerNode::LoadParameters() {
  require_classification_ = declare_parameter<bool>("require_classification", true);
  timeout_sec_ = declare_parameter<double>("timeout_sec", 1.0);
  alpha_ = std::clamp(declare_parameter<double>("alpha", 0.6), 0.0, 1.0);
  beta_ = std::clamp(declare_parameter<double>("beta", 0.1), 0.0, 1.0);
  odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  tf_timeout_sec_ = declare_parameter<double>("tf_timeout_sec", 0.05);
  update_rate_hz_ = declare_parameter<double>("update_rate_hz", 20.0);
  publish_debug_markers_ = declare_parameter<bool>("publish_debug_markers", true);
}

void PerceptionObjectTrackerNode::TargetDetectedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_target_detected_ = msg->data;
  last_target_detected_time_ = this->now();
}

void PerceptionObjectTrackerNode::ObstaclePositionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_obstacle_position_ = msg;
  new_measurement_ = true;
}

void PerceptionObjectTrackerNode::TimerCallback() {
  rclcpp::Time now = this->now();
  geometry_msgs::msg::PointStamped::SharedPtr measurement;
  bool has_measurement = false;

  {
    std::scoped_lock lock(data_mutex_);
    if (new_measurement_ && latest_obstacle_position_) {
      const double target_age = (now - last_target_detected_time_).seconds();
      if (!require_classification_ || (latest_target_detected_ && target_age <= timeout_sec_)) {
        measurement = latest_obstacle_position_;
        has_measurement = true;
      }
      new_measurement_ = false; // Consume measurement
    }
  }

  // 1. Prediction step
  double dt = 0.0;
  if (is_tracking_) {
    dt = (now - last_update_time_).seconds();
    if (dt > timeout_sec_) {
      RCLCPP_INFO(this->get_logger(), "Track lost due to timeout.");
      is_tracking_ = false;
    } else {
      // Predict state (Constant Velocity Model)
      x_est_ += vx_est_ * dt;
      y_est_ += vy_est_ * dt;
    }
  }

  // 2. Update step
  if (has_measurement) {
    try {
      // Transform measurement to fixed frame (odom)
      const auto tf_to_odom = tf_buffer_->lookupTransform(
          odom_frame_, measurement->header.frame_id,
          tf2::TimePointZero, tf2::durationFromSec(tf_timeout_sec_));

      geometry_msgs::msg::PointStamped pt_odom;
      tf2::doTransform(*measurement, pt_odom, tf_to_odom);

      const double z_x = pt_odom.point.x;
      const double z_y = pt_odom.point.y;

      if (!is_tracking_) {
        // Initialize
        x_est_ = z_x;
        y_est_ = z_y;
        vx_est_ = 0.0;
        vy_est_ = 0.0;
        is_tracking_ = true;
        RCLCPP_INFO(this->get_logger(), "New track initialized.");
      } else {
        // Alpha-Beta update
        if (dt > 0.0) {
          const double residual_x = z_x - x_est_;
          const double residual_y = z_y - y_est_;
          
          x_est_ += alpha_ * residual_x;
          y_est_ += alpha_ * residual_y;
          vx_est_ += (beta_ / dt) * residual_x;
          vy_est_ += (beta_ / dt) * residual_y;
        }
      }
      last_update_time_ = now;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Tracker TF lookup failed: %s", ex.what());
    }
  }

  // 3. Output
  if (is_tracking_) {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = now;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_; // Twist will be in base_frame
    
    // Pose in odom
    odom_msg.pose.pose.position.x = x_est_;
    odom_msg.pose.pose.position.y = y_est_;
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.pose.pose.orientation.w = 1.0;

    // Convert absolute velocity (vx_est, vy_est) to relative velocity in base_link
    try {
      geometry_msgs::msg::Vector3Stamped v_odom;
      v_odom.header.stamp = now;
      v_odom.header.frame_id = odom_frame_;
      v_odom.vector.x = vx_est_;
      v_odom.vector.y = vy_est_;
      v_odom.vector.z = 0.0;

      // Transform absolute velocity vector to base_link orientation to get V_abs_in_base
      const auto tf_to_base = tf_buffer_->lookupTransform(
          base_frame_, odom_frame_, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_sec_));
      
      geometry_msgs::msg::Vector3Stamped v_abs_in_base;
      tf2::doTransform(v_odom, v_abs_in_base, tf_to_base);

      // We need ego velocity to compute relative velocity.
      // But wait! If we just output absolute velocity transformed to base_link orientation,
      // it's still absolute velocity, just expressed in base_link axes.
      // If the controller needs relative velocity, we can compute it if we know ego velocity.
      // For now, let's output absolute velocity in base_link frame (which is standard for Odometry child_frame).
      // Twist is usually the velocity of the child frame relative to the header frame.
      // So here, it's the velocity of the object (child) relative to odom (header).
      // Expressed in base_frame axes? Standard Odometry msg expresses twist in child_frame axes.
      // Wait, child_frame is usually the object itself (e.g. "obstacle_link").
      // If child_frame_id = "obstacle_link", twist is in "obstacle_link".
      // Let's just output twist in odom frame for simplicity, and set child_frame_id = odom_frame_.
      
      odom_msg.child_frame_id = odom_frame_;
      odom_msg.twist.twist.linear.x = vx_est_;
      odom_msg.twist.twist.linear.y = vy_est_;

      tracked_object_pub_->publish(odom_msg);

    } catch (const tf2::TransformException &ex) {
      // fallback
    }
  }

  if (publish_debug_markers_) {
    PublishDebugMarkers(is_tracking_, x_est_, y_est_, vx_est_, vy_est_);
  }
}

void PerceptionObjectTrackerNode::PublishDebugMarkers(bool active, double x, double y, double vx, double vy) {
  if (!debug_markers_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  rclcpp::Time now = this->now();

  // Position Marker
  visualization_msgs::msg::Marker pos_marker;
  pos_marker.header.frame_id = odom_frame_;
  pos_marker.header.stamp = now;
  pos_marker.ns = "perception_object_tracker_pos";
  pos_marker.id = 0;
  pos_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  pos_marker.action = active ? visualization_msgs::msg::Marker::ADD : visualization_msgs::msg::Marker::DELETE;
  pos_marker.pose.position.x = x;
  pos_marker.pose.position.y = y;
  pos_marker.pose.orientation.w = 1.0;
  pos_marker.scale.x = 0.5;
  pos_marker.scale.y = 0.5;
  pos_marker.scale.z = 0.1;
  pos_marker.color.r = 1.0f;
  pos_marker.color.g = 0.5f;
  pos_marker.color.b = 0.0f;
  pos_marker.color.a = 0.8f;
  marker_array.markers.push_back(pos_marker);

  // Velocity Vector Marker
  visualization_msgs::msg::Marker vel_marker;
  vel_marker.header.frame_id = odom_frame_;
  vel_marker.header.stamp = now;
  vel_marker.ns = "perception_object_tracker_vel";
  vel_marker.id = 1;
  vel_marker.type = visualization_msgs::msg::Marker::ARROW;
  vel_marker.action = active ? visualization_msgs::msg::Marker::ADD : visualization_msgs::msg::Marker::DELETE;
  vel_marker.scale.x = 0.05; // shaft diameter
  vel_marker.scale.y = 0.1;  // head diameter
  vel_marker.scale.z = 0.1;  // head length
  vel_marker.color.r = 1.0f;
  vel_marker.color.g = 1.0f;
  vel_marker.color.b = 0.0f;
  vel_marker.color.a = 0.8f;
  
  if (active) {
    geometry_msgs::msg::Point p_start, p_end;
    p_start.x = x; p_start.y = y; p_start.z = 0.1;
    // Show 1-second prediction arrow
    p_end.x = x + vx; p_end.y = y + vy; p_end.z = 0.1;
    vel_marker.points.push_back(p_start);
    vel_marker.points.push_back(p_end);
  }
  marker_array.markers.push_back(vel_marker);

  debug_markers_pub_->publish(marker_array);
}

}  // namespace perception_object_tracker

RCLCPP_COMPONENTS_REGISTER_NODE(perception_object_tracker::PerceptionObjectTrackerNode)
