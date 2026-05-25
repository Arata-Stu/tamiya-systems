cmake_minimum_required(VERSION 3.8)
project(hd_map_virtual_scan)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(yaml-cpp REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(hd_map_virtual_scan_component SHARED
  src/hd_map_virtual_scan_component.cpp
)
if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(hd_map_virtual_scan_component yaml-cpp::yaml-cpp)
else()
  target_link_libraries(hd_map_virtual_scan_component yaml-cpp)
endif()

rclcpp_components_register_nodes(hd_map_virtual_scan_component
  "hd_map_virtual_scan::HdMapVirtualScanComponent"
)

ament_auto_add_executable(hd_map_virtual_scan_node
  src/hd_map_virtual_scan_node.cpp
)
target_link_libraries(hd_map_virtual_scan_node hd_map_virtual_scan_component)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
