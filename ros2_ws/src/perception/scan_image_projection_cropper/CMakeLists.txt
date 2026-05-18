cmake_minimum_required(VERSION 3.8)
project(scan_image_projection_cropper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core imgproc)
find_package(rclcpp_components REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(scan_image_projection_cropper_component SHARED
  src/scan_image_projection_cropper_component.cpp
)

target_include_directories(scan_image_projection_cropper_component PUBLIC
  include
)

target_link_libraries(scan_image_projection_cropper_component
  ${OpenCV_LIBS}
)

rclcpp_components_register_nodes(scan_image_projection_cropper_component
  "scan_image_projection_cropper::ScanImageProjectionCropperComponent"
)

ament_auto_add_executable(scan_image_projection_cropper_node
  src/scan_image_projection_cropper_node.cpp
)

target_link_libraries(scan_image_projection_cropper_node
  scan_image_projection_cropper_component
)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
