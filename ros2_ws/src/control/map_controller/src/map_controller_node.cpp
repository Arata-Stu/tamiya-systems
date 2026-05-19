#include "map_controller/map_controller_component.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<map_controller::MapControllerComponent>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
