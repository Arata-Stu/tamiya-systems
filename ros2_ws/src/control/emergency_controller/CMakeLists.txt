cmake_minimum_required(VERSION 3.8)
project(emergency_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(emergency_controller_component SHARED
  src/emergency_controller_component.cpp
  src/emergency_controller_core.cpp
)

rclcpp_components_register_nodes(emergency_controller_component
  "emergency_controller::EmergencyControllerComponent"
)

ament_auto_add_executable(emergency_controller_node
  src/emergency_controller_node.cpp
)

target_link_libraries(emergency_controller_node emergency_controller_component)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
