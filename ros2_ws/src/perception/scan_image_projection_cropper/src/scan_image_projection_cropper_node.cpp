#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "scan_image_projection_cropper/scan_image_projection_cropper_component.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<
      scan_image_projection_cropper::ScanImageProjectionCropperComponent>(
      rclcpp::NodeOptions{});
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
