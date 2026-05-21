cmake_minimum_required(VERSION 3.8)
project(drive_mode_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(drive_mode_manager_node
  src/drive_mode_manager_node.cpp
)

target_compile_features(drive_mode_manager_node PUBLIC cxx_std_17)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
