#include "ftg_controller/ftg_controller_component.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<ftg_controller::FtgControllerComponent>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
