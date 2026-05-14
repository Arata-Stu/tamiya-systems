#include "rclcpp/rclcpp.hpp"

#include "raceline_path_publisher/raceline_path_publisher_node.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<raceline_path_publisher::RacelinePathPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
