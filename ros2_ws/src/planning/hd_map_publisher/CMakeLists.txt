cmake_minimum_required(VERSION 3.8)
project(hd_map_publisher)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  hd_map_publisher/hd_map_publisher_node.py
  hd_map_publisher/hd_map_section_localizer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
