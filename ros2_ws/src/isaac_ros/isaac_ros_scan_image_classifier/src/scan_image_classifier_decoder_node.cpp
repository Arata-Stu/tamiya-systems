#include "isaac_ros_scan_image_classifier/scan_image_classifier_decoder_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "rclcpp_components/register_node_macro.hpp"

namespace isaac_ros_scan_image_classifier {

namespace {

std::string JoinLabels(const std::vector<std::string> &labels) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (index > 0U) {
      stream << ", ";
    }
    stream << labels[index];
  }
  return stream.str();
}

}  // namespace

ScanImageClassifierDecoderNode::ScanImageClassifierDecoderNode(
    const rclcpp::NodeOptions &options)
    : Node("scan_image_classifier_decoder_node", options) {
  output_tensor_name_ =
      declare_parameter<std::string>("output_tensor_name", "output_logits");
  labels_ = declare_parameter<std::vector<std::string>>(
      "labels", std::vector<std::string>{"rc_car", "duct_tube", "background"});
  apply_softmax_ = declare_parameter<bool>("apply_softmax", true);
  target_class_id_ = declare_parameter<int>("target_class_id", 0);
  target_confidence_threshold_ =
      declare_parameter<double>("target_confidence_threshold", 0.50);

  if (labels_.empty()) {
    throw std::runtime_error(
        "scan_image_classifier_decoder_node requires at least one label.");
  }

  if (target_class_id_ < 0) {
    RCLCPP_WARN(get_logger(),
                "target_class_id must be non-negative. Falling back to 0.");
    target_class_id_ = 0;
  }

  label_pub_ =
      create_publisher<std_msgs::msg::String>("classification/label", 10);
  class_id_pub_ =
      create_publisher<std_msgs::msg::Int32>("classification/class_id", 10);
  confidence_pub_ = create_publisher<std_msgs::msg::Float32>(
      "classification/confidence", 10);
  scores_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "classification/scores", 10);
  target_detected_pub_ = create_publisher<std_msgs::msg::Bool>(
      "classification/target_detected", 10);

  using MySubscriber = nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>;

  nitros_sub_ = std::make_shared<MySubscriber>(
      this, "inference_output", "nitros_tensor_list_nhwc_rgb_f32",
      std::bind(&ScanImageClassifierDecoderNode::InputCallback, this,
                std::placeholders::_1));

  RCLCPP_INFO(
      get_logger(),
      "ScanImageClassifierDecoderNode initialized (tensor='%s', labels=[%s])",
      output_tensor_name_.c_str(), JoinLabels(labels_).c_str());
}

ScanImageClassifierDecoderNode::~ScanImageClassifierDecoderNode() = default;

std::vector<float> ScanImageClassifierDecoderNode::ComputeScores(
    const std::vector<float> &logits) const {
  if (!apply_softmax_) {
    return logits;
  }

  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> scores(logits.size(), 0.0F);
  double sum = 0.0;

  for (std::size_t index = 0; index < logits.size(); ++index) {
    scores[index] = std::exp(logits[index] - max_logit);
    sum += static_cast<double>(scores[index]);
  }

  if (sum <= std::numeric_limits<double>::epsilon()) {
    return scores;
  }

  for (auto &score : scores) {
    score = static_cast<float>(static_cast<double>(score) / sum);
  }

  return scores;
}

std::string ScanImageClassifierDecoderNode::ResolveLabel(
    std::size_t class_id) const {
  if (class_id < labels_.size()) {
    return labels_[class_id];
  }
  return "unknown";
}

void ScanImageClassifierDecoderNode::InputCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorListView &msg) {
  auto tensor = msg.GetNamedTensor(output_tensor_name_);

  if (tensor.GetBuffer() == nullptr) {
    RCLCPP_WARN(get_logger(),
                "Tensor '%s' not found or buffer is null.",
                output_tensor_name_.c_str());
    return;
  }

  std::vector<float> logits(labels_.size(), 0.0F);
  const cudaError_t cuda_status =
      cudaMemcpy(logits.data(), tensor.GetBuffer(),
                 logits.size() * sizeof(float), cudaMemcpyDeviceToHost);

  if (cuda_status != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaMemcpy failed: %s",
                 cudaGetErrorString(cuda_status));
    return;
  }

  const std::vector<float> scores = ComputeScores(logits);
  const auto best_it = std::max_element(scores.begin(), scores.end());
  const std::size_t best_index =
      static_cast<std::size_t>(std::distance(scores.begin(), best_it));
  const float best_score = *best_it;

  std_msgs::msg::String label_msg;
  label_msg.data = ResolveLabel(best_index);
  label_pub_->publish(label_msg);

  std_msgs::msg::Int32 class_id_msg;
  class_id_msg.data = static_cast<int32_t>(best_index);
  class_id_pub_->publish(class_id_msg);

  std_msgs::msg::Float32 confidence_msg;
  confidence_msg.data = best_score;
  confidence_pub_->publish(confidence_msg);

  std_msgs::msg::Float32MultiArray scores_msg;
  scores_msg.layout.dim.resize(1);
  scores_msg.layout.dim[0].label = "classes";
  scores_msg.layout.dim[0].size = static_cast<uint32_t>(scores.size());
  scores_msg.layout.dim[0].stride = static_cast<uint32_t>(scores.size());
  scores_msg.data = scores;
  scores_pub_->publish(scores_msg);

  std_msgs::msg::Bool target_detected_msg;
  target_detected_msg.data =
      best_index == static_cast<std::size_t>(target_class_id_) &&
      static_cast<double>(best_score) >= target_confidence_threshold_;
  target_detected_pub_->publish(target_detected_msg);

  RCLCPP_DEBUG(get_logger(),
               "Published classification: class_id=%zu label=%s score=%.3f",
               best_index, label_msg.data.c_str(), best_score);
}

}  // namespace isaac_ros_scan_image_classifier

RCLCPP_COMPONENTS_REGISTER_NODE(
    isaac_ros_scan_image_classifier::ScanImageClassifierDecoderNode)
