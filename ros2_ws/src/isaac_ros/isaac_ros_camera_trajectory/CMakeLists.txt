cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_camera_trajectory LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(camera_trajectory_decoder_node SHARED src/camera_trajectory_decoder_node.cpp)
rclcpp_components_register_nodes(camera_trajectory_decoder_node "isaac_ros_camera_trajectory::CameraTrajectoryDecoderNode")
set(node_plugins "${node_plugins}isaac_ros_camera_trajectory::CameraTrajectoryDecoderNode;$<TARGET_FILE:camera_trajectory_decoder_node>\n")
set_target_properties(camera_trajectory_decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE config launch)
