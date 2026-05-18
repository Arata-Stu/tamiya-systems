cmake_minimum_required(VERSION 3.8)
project(isaac_ros_scan_image_classifier)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(isaac_ros_scan_image_classifier_decoder_component SHARED
  src/scan_image_classifier_decoder_node.cpp
)

target_include_directories(isaac_ros_scan_image_classifier_decoder_component PUBLIC
  include
)

rclcpp_components_register_nodes(isaac_ros_scan_image_classifier_decoder_component
  "isaac_ros_scan_image_classifier::ScanImageClassifierDecoderNode"
)

ament_auto_add_executable(isaac_ros_scan_image_classifier_decoder_node
  src/scan_image_classifier_decoder_main.cpp
)

target_link_libraries(isaac_ros_scan_image_classifier_decoder_node
  isaac_ros_scan_image_classifier_decoder_component
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)
