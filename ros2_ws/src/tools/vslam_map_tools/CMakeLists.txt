cmake_minimum_required(VERSION 3.8)
project(vslam_map_tools)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  vslam_map_tools/manual_tf_alignment_node.py
  vslam_map_tools/export_landmarks_png.py
  vslam_map_tools/record_vslam_reference_snapshot.py
  vslam_map_tools/publish_saved_vslam_reference.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
