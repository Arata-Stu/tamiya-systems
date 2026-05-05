#include "trajectory_pure_pursuit/trajectory_pure_pursuit_component.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node =
      std::make_shared<trajectory_pure_pursuit::TrajectoryPurePursuitComponent>(
          options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
