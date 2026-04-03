cmake_minimum_required(VERSION 3.8)
project(ftg_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(ftg_controller_component SHARED
  src/ftg_controller_component.cpp
  src/ftg_core.cpp
)

rclcpp_components_register_nodes(ftg_controller_component
  "ftg_controller::FtgControllerComponent"
)

ament_auto_add_executable(ftg_controller_node
  src/ftg_controller_node.cpp
)

target_link_libraries(ftg_controller_node ftg_controller_component)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
