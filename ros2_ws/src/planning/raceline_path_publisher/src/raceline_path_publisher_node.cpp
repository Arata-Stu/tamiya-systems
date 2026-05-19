#include "raceline_path_publisher/raceline_path_publisher_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <cmath>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "race_planning_msgs/msg/trajectory.hpp"
#include "race_planning_msgs/msg/trajectory_point.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace raceline_path_publisher {

namespace {

double ClampPositive(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

int ClampPositiveInt(int value, int fallback) {
  return value > 0 ? value : fallback;
}

builtin_interfaces::msg::Time ToBuiltinTime(const rclcpp::Time &stamp) {
  builtin_interfaces::msg::Time msg;
  const auto nanoseconds = stamp.nanoseconds();
  msg.sec = static_cast<std::int32_t>(nanoseconds / 1000000000LL);
  msg.nanosec = static_cast<std::uint32_t>(nanoseconds % 1000000000LL);
  return msg;
}

} // namespace

RacelinePathPublisherNode::RacelinePathPublisherNode()
    : Node("raceline_path_publisher_node"), tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_) {
  LoadParameters();
  last_load_attempt_time_ = this->now() - reload_retry_interval_;

  if (publish_global_path_) {
    global_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "global_path", rclcpp::QoS(1).transient_local().reliable());
  }
  if (publish_local_path_) {
    local_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("trajectory", rclcpp::QoS(10));
  }
  if (publish_local_reference_) {
    local_reference_pub_ =
        this->create_publisher<race_planning_msgs::msg::Trajectory>(
            "trajectory_reference", rclcpp::QoS(10));
  }

  if (!raceline_csv_path_.empty()) {
    LoadRaceline();
  } else {
    RCLCPP_WARN(this->get_logger(),
                "Parameter raceline_csv_path is empty. Waiting for a valid CSV path.");
  }

  const auto timer_period =
      std::chrono::duration<double>(1.0 / ClampPositive(publish_rate_hz_, 10.0));
  timer_ = this->create_wall_timer(
      timer_period,
      std::bind(&RacelinePathPublisherNode::TimerCallback, this));
}

void RacelinePathPublisherNode::LoadParameters() {
  raceline_csv_path_ =
      this->declare_parameter<std::string>("raceline_csv_path", "");
  direction_ = this->declare_parameter<std::string>("direction", "forward");
  map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
  publish_rate_hz_ =
      ClampPositive(this->declare_parameter<double>("publish_rate_hz", 10.0), 10.0);
  local_path_length_m_ = ClampPositive(
      this->declare_parameter<double>("local_path_length_m", 6.0), 6.0);
  max_local_points_ = ClampPositiveInt(
      this->declare_parameter<int>("max_local_points", 80), 80);
  tf_timeout_sec_ = ClampPositive(
      this->declare_parameter<double>("tf_timeout_sec", 0.05), 0.05);
  publish_global_path_ =
      this->declare_parameter<bool>("publish_global_path", true);
  publish_local_path_ =
      this->declare_parameter<bool>("publish_local_path", true);
  publish_local_reference_ =
      this->declare_parameter<bool>("publish_local_reference", true);
}

bool RacelinePathPublisherNode::LoadRaceline() {
  if (raceline_csv_path_.empty()) {
    raceline_loaded_ = false;
    return false;
  }

  std::string error_message;
  if (!core_.LoadCsv(raceline_csv_path_, direction_, error_message)) {
    raceline_loaded_ = false;
    RCLCPP_ERROR(this->get_logger(), "%s", error_message.c_str());
    return false;
  }

  const auto &data = core_.GetData();
  raceline_loaded_ = !data.samples.empty();
  if (raceline_loaded_) {
    RCLCPP_INFO(this->get_logger(),
                "Loaded raceline CSV: %s (direction=%s, points=%zu, length=%.3f m, spacing=%.3f m)",
                raceline_csv_path_.c_str(), direction_.c_str(),
                data.samples.size(), data.total_length, data.nominal_spacing);
  }
  return raceline_loaded_;
}

void RacelinePathPublisherNode::TimerCallback() {
  if (!raceline_loaded_) {
    if (!raceline_csv_path_.empty() &&
        (this->now() - last_load_attempt_time_).seconds() >=
            reload_retry_interval_.seconds()) {
      last_load_attempt_time_ = this->now();
      LoadRaceline();
    }
    return;
  }

  const rclcpp::Time stamp = this->now();
  if (publish_global_path_ && global_path_pub_) {
    PublishGlobalPath(stamp);
  }
  if ((publish_local_path_ && local_path_pub_) ||
      (publish_local_reference_ && local_reference_pub_)) {
    PublishLocalOutputs(stamp);
  }
}

void RacelinePathPublisherNode::PublishGlobalPath(const rclcpp::Time &stamp) {
  global_path_pub_->publish(core_.BuildPath(map_frame_, ToBuiltinTime(stamp)));
}

void RacelinePathPublisherNode::PublishLocalOutputs(const rclcpp::Time &stamp) {
  geometry_msgs::msg::TransformStamped map_from_base;
  geometry_msgs::msg::TransformStamped base_from_map;

  try {
    map_from_base = tf_buffer_.lookupTransform(
        map_frame_, base_frame_, tf2::TimePointZero,
        tf2::durationFromSec(tf_timeout_sec_));
    base_from_map = tf_buffer_.lookupTransform(
        base_frame_, map_frame_, tf2::TimePointZero,
        tf2::durationFromSec(tf_timeout_sec_));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "Failed to look up transform between %s and %s: %s",
                         map_frame_.c_str(), base_frame_.c_str(), ex.what());
    return;
  }

  const double vehicle_x = map_from_base.transform.translation.x;
  const double vehicle_y = map_from_base.transform.translation.y;
  const auto indices = core_.SelectForwardIndices(
      vehicle_x, vehicle_y, local_path_length_m_,
      static_cast<std::size_t>(std::max(max_local_points_, 1)));
  if (indices.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "Raceline contains no usable points for local path output.");
    return;
  }

  const auto map_path =
      core_.BuildPathFromIndices(indices, map_frame_, ToBuiltinTime(stamp));

  nav_msgs::msg::Path local_path;
  race_planning_msgs::msg::Trajectory local_reference;
  if (publish_local_path_ && local_path_pub_) {
    local_path.header.frame_id = base_frame_;
    local_path.header.stamp = stamp;
    local_path.poses.reserve(map_path.poses.size());
  }
  if (publish_local_reference_ && local_reference_pub_) {
    local_reference.header.frame_id = base_frame_;
    local_reference.header.stamp = stamp;
    local_reference.points.reserve(map_path.poses.size());
  }

  const auto &samples = core_.GetData().samples;
  double path_s = 0.0;
  bool has_previous_point = false;
  geometry_msgs::msg::Point previous_position;

  for (std::size_t i = 0U; i < map_path.poses.size(); ++i) {
    geometry_msgs::msg::PoseStamped transformed_pose;
    tf2::doTransform(map_path.poses[i], transformed_pose, base_from_map);
    transformed_pose.header.frame_id = base_frame_;
    transformed_pose.header.stamp = stamp;

    if (publish_local_path_ && local_path_pub_) {
      local_path.poses.push_back(transformed_pose);
    }

    if (publish_local_reference_ && local_reference_pub_) {
      if (has_previous_point) {
        path_s += std::hypot(
            transformed_pose.pose.position.x - previous_position.x,
            transformed_pose.pose.position.y - previous_position.y);
      }

      race_planning_msgs::msg::TrajectoryPoint point;
      point.pose = transformed_pose.pose;
      if (i < indices.size() && indices[i] < samples.size()) {
        const auto &sample = samples[indices[i]];
        point.track_s_m = sample.s;
        point.speed_mps = sample.speed;
        point.curvature_radpm = sample.curvature;
        point.acceleration_mps2 = sample.acceleration;
      }
      point.path_s_m = path_s;
      local_reference.points.push_back(point);
      previous_position = transformed_pose.pose.position;
      has_previous_point = true;
    }
  }

  if (publish_local_path_ && local_path_pub_) {
    local_path_pub_->publish(local_path);
  }
  if (publish_local_reference_ && local_reference_pub_) {
    local_reference_pub_->publish(local_reference);
  }
}

} // namespace raceline_path_publisher
