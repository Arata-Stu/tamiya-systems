cmake_minimum_required(VERSION 3.8)
project(trajectory_pure_pursuit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(trajectory_pure_pursuit_component SHARED
  src/trajectory_pure_pursuit_component.cpp
)

rclcpp_components_register_nodes(trajectory_pure_pursuit_component
  "trajectory_pure_pursuit::TrajectoryPurePursuitComponent"
)

ament_auto_add_executable(trajectory_pure_pursuit_node
  src/trajectory_pure_pursuit_node.cpp
)

target_link_libraries(trajectory_pure_pursuit_node trajectory_pure_pursuit_component)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
