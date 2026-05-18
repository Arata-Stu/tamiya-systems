cmake_minimum_required(VERSION 3.8)
project(perception_object_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(perception_object_tracker_node SHARED
  src/perception_object_tracker_node.cpp
)

rclcpp_components_register_nodes(perception_object_tracker_node
  "perception_object_tracker::PerceptionObjectTrackerNode"
)

ament_auto_package(
  INSTALL_TO_SHARE
    config
    launch
)
