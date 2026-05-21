#include "localization_manager/localization_manager_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using std::placeholders::_1;

LocalizationManagerNode::LocalizationManagerNode()
    : Node("localization_manager_node") {
  localization_trigger_topic_ = this->declare_parameter<std::string>(
      "localization_trigger_topic", localization_trigger_topic_);
  localization_trigger_service_ = this->declare_parameter<std::string>(
      "localization_trigger_service", localization_trigger_service_);
  localization_result_topic_ = this->declare_parameter<std::string>(
      "localization_result_topic", localization_result_topic_);
  use_amcl_pose_ =
      this->declare_parameter("use_amcl_pose", use_amcl_pose_);
  amcl_pose_topic_ =
      this->declare_parameter<std::string>("amcl_pose_topic", amcl_pose_topic_);
  amcl_pose_update_mode_ =
      this->declare_parameter<std::string>("amcl_pose_update_mode",
                                           amcl_pose_update_mode_);
  amcl_pose_max_xy_variance_ =
      this->declare_parameter("amcl_pose_max_xy_variance",
                              amcl_pose_max_xy_variance_);
  amcl_pose_max_yaw_variance_ =
      this->declare_parameter("amcl_pose_max_yaw_variance",
                              amcl_pose_max_yaw_variance_);
  publish_initialpose_to_amcl_ =
      this->declare_parameter("publish_initialpose_to_amcl",
                              publish_initialpose_to_amcl_);
  initial_pose_topic_ =
      this->declare_parameter<std::string>("initial_pose_topic",
                                           initial_pose_topic_);
  localization_result_offset_x_ =
      this->declare_parameter("localization_result_offset_x",
                              localization_result_offset_x_);
  localization_result_offset_y_ =
      this->declare_parameter("localization_result_offset_y",
                              localization_result_offset_y_);
  localization_result_offset_z_ =
      this->declare_parameter("localization_result_offset_z",
                              localization_result_offset_z_);
  localization_result_offset_roll_rad_ =
      this->declare_parameter("localization_result_offset_roll_rad",
                              localization_result_offset_roll_rad_);
  localization_result_offset_pitch_rad_ =
      this->declare_parameter("localization_result_offset_pitch_rad",
                              localization_result_offset_pitch_rad_);
  localization_result_offset_yaw_rad_ =
      this->declare_parameter("localization_result_offset_yaw_rad",
                              localization_result_offset_yaw_rad_);
  localization_feedback_timeout_sec_ = std::max(
      0.0, this->declare_parameter("localization_feedback_timeout_sec", 0.0));

  publish_localization_tf_ =
      this->declare_parameter("publish_localization_tf", publish_localization_tf_);
  localization_tf_mode_ =
      this->declare_parameter<std::string>("localization_tf_mode",
                                           localization_tf_mode_);
  localization_tf_map_frame_ =
      this->declare_parameter<std::string>("localization_tf_map_frame",
                                           localization_tf_map_frame_);
  localization_tf_odom_frame_ =
      this->declare_parameter<std::string>("localization_tf_odom_frame",
                                           localization_tf_odom_frame_);
  localization_tf_base_frame_ =
      this->declare_parameter<std::string>("localization_tf_base_frame",
                                           localization_tf_base_frame_);
  localization_tf_publish_rate_hz_ = std::max(
      0.0, this->declare_parameter("localization_tf_publish_rate_hz",
                                   localization_tf_publish_rate_hz_));

  const bool is_map_to_odom_mode = (localization_tf_mode_ == "map_to_odom");
  const bool is_map_to_base_mode =
      (localization_tf_mode_ == "map_to_base_link" ||
       localization_tf_mode_ == "map_to_base");
  const bool is_valid_amcl_update_mode =
      (amcl_pose_update_mode_ == "continuous" ||
       amcl_pose_update_mode_ == "once" ||
       amcl_pose_update_mode_ == "never");
  if (!is_map_to_odom_mode && !is_map_to_base_mode) {
    RCLCPP_WARN(this->get_logger(),
                "Invalid localization_tf_mode '%s'. "
                "Expected map_to_odom or map_to_base_link. "
                "Fallback to map_to_odom.",
                localization_tf_mode_.c_str());
    localization_tf_mode_ = "map_to_odom";
  }
  if (!is_valid_amcl_update_mode) {
    RCLCPP_WARN(this->get_logger(),
                "Invalid amcl_pose_update_mode '%s'. Fallback to once.",
                amcl_pose_update_mode_.c_str());
    amcl_pose_update_mode_ = "once";
  }
  allow_amcl_pose_tf_update_ = (amcl_pose_update_mode_ != "never");

  trigger_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      localization_trigger_topic_, rclcpp::QoS(10),
      std::bind(&LocalizationManagerNode::trigger_callback, this, _1));
  localization_result_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          localization_result_topic_, rclcpp::QoS(10),
          std::bind(&LocalizationManagerNode::localization_result_callback, this,
                    _1));
  localization_trigger_client_ =
      this->create_client<std_srvs::srv::Empty>(localization_trigger_service_);

  if (use_amcl_pose_) {
    amcl_pose_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            amcl_pose_topic_, rclcpp::QoS(10),
            std::bind(&LocalizationManagerNode::amcl_pose_callback, this, _1));
  }

  const auto strip_leading_slashes = [](std::string topic) {
    topic.erase(topic.begin(),
                std::find_if(topic.begin(), topic.end(),
                             [](const char c) { return c != '/'; }));
    return topic;
  };
  if (publish_initialpose_to_amcl_ &&
      strip_leading_slashes(localization_result_topic_) ==
          strip_leading_slashes(initial_pose_topic_)) {
    RCLCPP_WARN(this->get_logger(),
                "AMCL initialpose forwarding disabled because "
                "localization_result_topic and initial_pose_topic are both %s.",
                initial_pose_topic_.c_str());
    publish_initialpose_to_amcl_ = false;
  }

  if (publish_initialpose_to_amcl_) {
    initial_pose_pub_ =
        this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            initial_pose_topic_, rclcpp::QoS(1));
  }

  if (publish_localization_tf_) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&LocalizationManagerNode::timer_callback, this));

  last_localization_trigger_time_ = this->now();
  last_localization_tf_publish_time_ =
      rclcpp::Time(0, 0, this->get_clock()->get_clock_type());

  RCLCPP_INFO(this->get_logger(),
              "Localization manager started: trigger_topic=%s, service=%s, "
              "result_topic=%s",
              localization_trigger_topic_.c_str(),
              localization_trigger_service_.c_str(),
              localization_result_topic_.c_str());
  RCLCPP_INFO(this->get_logger(),
              "AMCL pose input: %s (topic=%s, update_mode=%s, max_xy_var=%.3f, "
              "max_yaw_var=%.3f, initialpose_forward=%s -> %s, "
              "result_offset_xyz=(%.3f, %.3f, %.3f), rpy=(%.3f, %.3f, %.3f))",
              use_amcl_pose_ ? "enabled" : "disabled", amcl_pose_topic_.c_str(),
              amcl_pose_update_mode_.c_str(),
              amcl_pose_max_xy_variance_, amcl_pose_max_yaw_variance_,
              publish_initialpose_to_amcl_ ? "enabled" : "disabled",
              initial_pose_topic_.c_str(), localization_result_offset_x_,
              localization_result_offset_y_, localization_result_offset_z_,
              localization_result_offset_roll_rad_,
              localization_result_offset_pitch_rad_,
              localization_result_offset_yaw_rad_);
  RCLCPP_INFO(this->get_logger(),
              "Localization TF bridge: %s (mode=%s, map=%s, odom=%s, base=%s, "
              "publish_rate=%.2f Hz)",
              publish_localization_tf_ ? "enabled" : "disabled",
              localization_tf_mode_.c_str(), localization_tf_map_frame_.c_str(),
              localization_tf_odom_frame_.c_str(),
              localization_tf_base_frame_.c_str(),
              localization_tf_publish_rate_hz_);
}

void LocalizationManagerNode::trigger_callback(
    const std_msgs::msg::Bool::SharedPtr msg) {
  if (!msg->data) {
    return;
  }
  request_localization();
}

void LocalizationManagerNode::request_localization() {
  if (!localization_trigger_client_) {
    RCLCPP_ERROR(this->get_logger(),
                 "Localization trigger client is not initialized.");
    return;
  }

  if (!localization_trigger_client_->service_is_ready()) {
    RCLCPP_WARN(this->get_logger(),
                "Localization trigger service is not ready: %s",
                localization_trigger_service_.c_str());
    return;
  }

  auto request = std::make_shared<std_srvs::srv::Empty::Request>();
  waiting_localization_result_ = true;
  localization_result_timed_out_ = false;
  last_localization_trigger_time_ = this->now();
  (void)localization_trigger_client_->async_send_request(request);
  RCLCPP_INFO(this->get_logger(), "Requested localization trigger: %s",
              localization_trigger_service_.c_str());
}

void LocalizationManagerNode::localization_result_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  const double elapsed_sec =
      (this->now() - last_localization_trigger_time_).seconds();
  update_localization_tf(*msg, "global localization");
  allow_amcl_pose_tf_update_ = (amcl_pose_update_mode_ != "never");
  publish_initial_pose(*msg);

  if (waiting_localization_result_) {
    waiting_localization_result_ = false;
    localization_result_timed_out_ = false;
    RCLCPP_INFO(
        this->get_logger(),
        "Localization success (%.3f s): frame=%s pos=(%.3f, %.3f) qz=%.3f qw=%.3f",
        elapsed_sec, msg->header.frame_id.c_str(), msg->pose.pose.position.x,
        msg->pose.pose.position.y, msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    return;
  }

  if (localization_result_timed_out_) {
    localization_result_timed_out_ = false;
    RCLCPP_WARN(this->get_logger(),
                "Localization result arrived after timeout (%.3f s): frame=%s "
                "pos=(%.3f, %.3f)",
                elapsed_sec, msg->header.frame_id.c_str(),
                msg->pose.pose.position.x, msg->pose.pose.position.y);
    return;
  }

  RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Localization result received on %s: frame=%s pos=(%.3f, %.3f)",
      localization_result_topic_.c_str(), msg->header.frame_id.c_str(),
      msg->pose.pose.position.x, msg->pose.pose.position.y);
}

void LocalizationManagerNode::amcl_pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  if (!is_amcl_pose_accepted(*msg)) {
    return;
  }
  if (!should_apply_amcl_pose_update()) {
    return;
  }

  update_localization_tf(*msg, "AMCL");
  if (amcl_pose_update_mode_ == "once") {
    allow_amcl_pose_tf_update_ = false;
  }
  RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "AMCL pose accepted on %s: frame=%s pos=(%.3f, %.3f) cov=(%.4f, %.4f, %.4f)",
      amcl_pose_topic_.c_str(), msg->header.frame_id.c_str(),
      msg->pose.pose.position.x, msg->pose.pose.position.y,
      msg->pose.covariance[0], msg->pose.covariance[7],
      msg->pose.covariance[35]);
}

bool LocalizationManagerNode::should_apply_amcl_pose_update() {
  if (amcl_pose_update_mode_ == "never") {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "AMCL pose accepted but TF update is disabled by amcl_pose_update_mode=never");
    return false;
  }

  if (amcl_pose_update_mode_ == "once" && !allow_amcl_pose_tf_update_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "AMCL pose accepted but one-shot TF update has already been consumed");
    return false;
  }

  return true;
}

geometry_msgs::msg::Pose LocalizationManagerNode::apply_localization_result_offset(
    const geometry_msgs::msg::Pose &pose) const {
  tf2::Quaternion map_to_result_q;
  tf2::fromMsg(pose.orientation, map_to_result_q);
  if (map_to_result_q.length2() < 1e-12) {
    map_to_result_q.setRPY(0.0, 0.0, 0.0);
  } else {
    map_to_result_q.normalize();
  }

  const tf2::Transform map_to_result_tf(
      map_to_result_q, tf2::Vector3(pose.position.x, pose.position.y,
                                    pose.position.z));

  tf2::Quaternion result_to_base_q;
  result_to_base_q.setRPY(localization_result_offset_roll_rad_,
                          localization_result_offset_pitch_rad_,
                          localization_result_offset_yaw_rad_);
  result_to_base_q.normalize();
  const tf2::Transform result_to_base_tf(
      result_to_base_q,
      tf2::Vector3(localization_result_offset_x_, localization_result_offset_y_,
                   localization_result_offset_z_));

  const tf2::Transform map_to_base_tf = map_to_result_tf * result_to_base_tf;

  geometry_msgs::msg::Pose corrected_pose;
  corrected_pose.position.x = map_to_base_tf.getOrigin().x();
  corrected_pose.position.y = map_to_base_tf.getOrigin().y();
  corrected_pose.position.z = map_to_base_tf.getOrigin().z();
  corrected_pose.orientation = tf2::toMsg(map_to_base_tf.getRotation());
  return corrected_pose;
}

void LocalizationManagerNode::update_localization_tf(
    const geometry_msgs::msg::PoseWithCovarianceStamped &msg,
    const std::string &source_name) {
  if (!publish_localization_tf_ || !tf_broadcaster_) {
    return;
  }

  const std::string map_frame =
      msg.header.frame_id.empty() ? localization_tf_map_frame_ : msg.header.frame_id;
  const geometry_msgs::msg::Pose corrected_pose =
      apply_localization_result_offset(msg.pose.pose);
  tf2::Quaternion map_to_base_q;
  tf2::fromMsg(corrected_pose.orientation, map_to_base_q);
  if (map_to_base_q.length2() < 1e-12) {
    map_to_base_q.setRPY(0.0, 0.0, 0.0);
  } else {
    map_to_base_q.normalize();
  }
  const tf2::Transform map_to_base_tf(
      map_to_base_q,
      tf2::Vector3(corrected_pose.position.x, corrected_pose.position.y,
                   corrected_pose.position.z));

  geometry_msgs::msg::TransformStamped output_tf;
  output_tf.header.frame_id = map_frame;
  output_tf.header.stamp = this->now();

  if (localization_tf_mode_ == "map_to_odom") {
    if (!tf_buffer_) {
      return;
    }

    geometry_msgs::msg::TransformStamped odom_to_base_tf_msg;
    try {
      // lookupTransform(target, source) returns the transform that converts
      // source-frame data into the target frame.  Requesting target=odom and
      // source=base_link gives T_odom_base, which is the term needed for:
      //   T_map_odom = T_map_base * inverse(T_odom_base)
      odom_to_base_tf_msg = tf_buffer_->lookupTransform(
          localization_tf_odom_frame_, localization_tf_base_frame_,
          tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "%s TF skipped: lookup %s <- %s (odom<-base) failed: %s",
          source_name.c_str(), localization_tf_odom_frame_.c_str(),
          localization_tf_base_frame_.c_str(), ex.what());
      return;
    }

    tf2::Transform odom_to_base_tf;
    tf2::fromMsg(odom_to_base_tf_msg.transform, odom_to_base_tf);
    const tf2::Transform map_to_odom_tf = map_to_base_tf * odom_to_base_tf.inverse();

    output_tf.child_frame_id = localization_tf_odom_frame_;
    output_tf.transform = tf2::toMsg(map_to_odom_tf);
  } else {
    output_tf.child_frame_id = localization_tf_base_frame_;
    output_tf.transform.translation.x = corrected_pose.position.x;
    output_tf.transform.translation.y = corrected_pose.position.y;
    output_tf.transform.translation.z = corrected_pose.position.z;
    output_tf.transform.rotation = tf2::toMsg(map_to_base_q);
  }

  last_localization_tf_ = output_tf;
  has_localization_tf_ = true;
  publish_localization_tf();
  RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Updated localization TF from %s: %s -> %s", source_name.c_str(),
      output_tf.header.frame_id.c_str(), output_tf.child_frame_id.c_str());
}

bool LocalizationManagerNode::is_amcl_pose_accepted(
    const geometry_msgs::msg::PoseWithCovarianceStamped &msg) {
  const double x_variance = msg.pose.covariance[0];
  const double y_variance = msg.pose.covariance[7];
  const double yaw_variance = msg.pose.covariance[35];

  if (amcl_pose_max_xy_variance_ >= 0.0) {
    if (!std::isfinite(x_variance) || !std::isfinite(y_variance) ||
        x_variance > amcl_pose_max_xy_variance_ ||
        y_variance > amcl_pose_max_xy_variance_) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "AMCL pose rejected by xy covariance: x=%.4f y=%.4f max=%.4f",
          x_variance, y_variance, amcl_pose_max_xy_variance_);
      return false;
    }
  }

  if (amcl_pose_max_yaw_variance_ >= 0.0) {
    if (!std::isfinite(yaw_variance) ||
        yaw_variance > amcl_pose_max_yaw_variance_) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "AMCL pose rejected by yaw covariance: yaw=%.4f max=%.4f",
          yaw_variance, amcl_pose_max_yaw_variance_);
      return false;
    }
  }

  return true;
}

void LocalizationManagerNode::publish_initial_pose(
    const geometry_msgs::msg::PoseWithCovarianceStamped &msg) {
  if (!initial_pose_pub_) {
    return;
  }

  auto initial_pose = msg;
  if (initial_pose.header.frame_id.empty()) {
    initial_pose.header.frame_id = localization_tf_map_frame_;
  }
  initial_pose.pose.pose = apply_localization_result_offset(initial_pose.pose.pose);
  initial_pose.header.stamp = this->now();
  initial_pose_pub_->publish(initial_pose);
  RCLCPP_INFO(this->get_logger(),
              "Forwarded localization result to AMCL initial pose: %s",
              initial_pose_topic_.c_str());
}

void LocalizationManagerNode::publish_localization_tf() {
  if (!publish_localization_tf_ || !tf_broadcaster_ || !has_localization_tf_) {
    return;
  }

  const auto now = this->now();
  if (localization_tf_publish_rate_hz_ > 0.0 &&
      last_localization_tf_publish_time_.nanoseconds() > 0) {
    const double elapsed_sec =
        (now - last_localization_tf_publish_time_).seconds();
    const double min_period_sec = 1.0 / localization_tf_publish_rate_hz_;
    if (elapsed_sec < min_period_sec) {
      return;
    }
  }

  auto tf_msg = last_localization_tf_;
  tf_msg.header.stamp = now;
  tf_broadcaster_->sendTransform(tf_msg);
  last_localization_tf_publish_time_ = now;
}

void LocalizationManagerNode::timer_callback() {
  if (waiting_localization_result_ && localization_feedback_timeout_sec_ > 0.0 &&
      (this->now() - last_localization_trigger_time_).seconds() >
          localization_feedback_timeout_sec_) {
    waiting_localization_result_ = false;
    localization_result_timed_out_ = true;
    RCLCPP_WARN(this->get_logger(),
                "Localization result timeout (%.2f s): no message on %s",
                localization_feedback_timeout_sec_,
                localization_result_topic_.c_str());
  }

  publish_localization_tf();
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationManagerNode>());
  rclcpp::shutdown();
  return 0;
}
