#pragma once

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"

namespace isaac_ros_scan_image_classifier {

class ScanImageClassifierDecoderNode : public rclcpp::Node {
public:
  explicit ScanImageClassifierDecoderNode(const rclcpp::NodeOptions &options);
  ~ScanImageClassifierDecoderNode();

private:
  void InputCallback(
      const nvidia::isaac_ros::nitros::NitrosTensorListView &msg);
  std::vector<float> ComputeScores(const std::vector<float> &logits) const;
  std::string ResolveLabel(std::size_t class_id) const;

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>>
      nitros_sub_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr label_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr class_id_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr confidence_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr scores_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr target_detected_pub_;

  std::string output_tensor_name_;
  std::vector<std::string> labels_;
  bool apply_softmax_;
  int target_class_id_;
  double target_confidence_threshold_;
};

}  // namespace isaac_ros_scan_image_classifier
