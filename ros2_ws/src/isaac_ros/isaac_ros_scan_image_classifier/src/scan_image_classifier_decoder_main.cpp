#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_scan_image_classifier/scan_image_classifier_decoder_node.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<
      isaac_ros_scan_image_classifier::ScanImageClassifierDecoderNode>(
      rclcpp::NodeOptions{});
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
