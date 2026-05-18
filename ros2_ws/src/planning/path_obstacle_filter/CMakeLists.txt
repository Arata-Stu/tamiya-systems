cmake_minimum_required(VERSION 3.8)
project(path_obstacle_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# executableではなくshared library
ament_auto_add_library(path_obstacle_filter_node SHARED
  src/path_obstacle_filter_node.cpp
)

rclcpp_components_register_nodes(path_obstacle_filter_node
  "path_obstacle_filter::PathObstacleFilterNode"
)

ament_auto_package(
  INSTALL_TO_SHARE
    config
    launch
)