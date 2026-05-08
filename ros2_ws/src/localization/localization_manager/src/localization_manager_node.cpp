#include "localization_manager/localization_manager_node.hpp"

#include <algorithm>
#include <chrono>

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
  if (!is_map_to_odom_mode && !is_map_to_base_mode) {
    RCLCPP_WARN(this->get_logger(),
                "Invalid localization_tf_mode '%s'. Fallback to map_to_odom.",
                localization_tf_mode_.c_str());
    localization_tf_mode_ = "map_to_odom";
  }

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
  update_localization_tf(*msg);

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

void LocalizationManagerNode::update_localization_tf(
    const geometry_msgs::msg::PoseWithCovarianceStamped &msg) {
  if (!publish_localization_tf_ || !tf_broadcaster_) {
    return;
  }

  const std::string map_frame =
      msg.header.frame_id.empty() ? localization_tf_map_frame_ : msg.header.frame_id;
  tf2::Quaternion map_to_base_q;
  tf2::fromMsg(msg.pose.pose.orientation, map_to_base_q);
  if (map_to_base_q.length2() < 1e-12) {
    map_to_base_q.setRPY(0.0, 0.0, 0.0);
  } else {
    map_to_base_q.normalize();
  }
  const tf2::Transform map_to_base_tf(
      map_to_base_q,
      tf2::Vector3(msg.pose.pose.position.x, msg.pose.pose.position.y,
                   msg.pose.pose.position.z));

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
          "Localization TF skipped: lookup %s <- %s (odom<-base) failed: %s",
          localization_tf_odom_frame_.c_str(), localization_tf_base_frame_.c_str(),
          ex.what());
      return;
    }

    tf2::Transform odom_to_base_tf;
    tf2::fromMsg(odom_to_base_tf_msg.transform, odom_to_base_tf);
    const tf2::Transform map_to_odom_tf = map_to_base_tf * odom_to_base_tf.inverse();

    output_tf.child_frame_id = localization_tf_odom_frame_;
    output_tf.transform = tf2::toMsg(map_to_odom_tf);
  } else {
    output_tf.child_frame_id = localization_tf_base_frame_;
    output_tf.transform.translation.x = msg.pose.pose.position.x;
    output_tf.transform.translation.y = msg.pose.pose.position.y;
    output_tf.transform.translation.z = msg.pose.pose.position.z;
    output_tf.transform.rotation = tf2::toMsg(map_to_base_q);
  }

  last_localization_tf_ = output_tf;
  has_localization_tf_ = true;
  publish_localization_tf();
  RCLCPP_INFO(this->get_logger(), "Updated localization TF %s -> %s",
              output_tf.header.frame_id.c_str(),
              output_tf.child_frame_id.c_str());
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
