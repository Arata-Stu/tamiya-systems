cmake_minimum_required(VERSION 3.8)
project(map_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(map_controller_component SHARED
  src/map_controller_component.cpp
)

rclcpp_components_register_nodes(map_controller_component
  "map_controller::MapControllerComponent"
)

ament_auto_add_executable(map_controller_node
  src/map_controller_node.cpp
)

target_link_libraries(map_controller_node map_controller_component)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
