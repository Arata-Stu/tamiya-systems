#include "pure_pursuit_controller/pure_pursuit_controller_component.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node =
      std::make_shared<pure_pursuit_controller::PurePursuitControllerComponent>(
          options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
