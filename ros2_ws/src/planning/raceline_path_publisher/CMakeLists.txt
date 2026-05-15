cmake_minimum_required(VERSION 3.8)
project(raceline_path_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(raceline_path_publisher_node
  src/raceline_path_publisher_main.cpp
  src/raceline_path_publisher_node.cpp
  src/raceline_path_publisher_core.cpp
)

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR INSTALL_TO_SHARE
  config
  launch
)
