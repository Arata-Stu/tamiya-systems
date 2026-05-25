#include "hd_map_virtual_scan/hd_map_virtual_scan_component.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hd_map_virtual_scan::HdMapVirtualScanComponent>(
      rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
