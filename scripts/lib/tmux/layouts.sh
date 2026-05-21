#!/bin/bash

create_tamiya_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_PRODUCTION"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  add_pane "$WINDOW_DATA" "$RECORD_DIR" "" ""
  add_pane "$WINDOW_DATA" "$SCRIPTS_DIR" "" ""
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_python_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$PYTHON_DIR" "$ROS_SETUP" ""
  add_pane "$WINDOW_MAIN" "$RECORD_DIR" "$ROS_SETUP" ""
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_map_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_CREATE_MAP"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$RVIZ_VSLAM_ALIGNMENT"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" ""
  add_pane "$WINDOW_EVAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_PLAY_BAG"
  add_pane "$WINDOW_EVAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_LOCALIZATION_EVAL"
  add_pane "$WINDOW_EVAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_LOCALIZATION_TRIGGER"
  add_pane "$WINDOW_EVAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  add_pane "$WINDOW_EVAL" "$WORK_DIR" "$ROS_SETUP" "$RVIZ_LOCALIZATION_EVAL"
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_identification_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_IDENTIFICATION"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_DASHBOARD_IDENTIFICATION"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_LEFT_IMAGE_VIEWER"
  add_pane "$WINDOW_DATA" "$WORK_DIR" "$ROS_SETUP" "$CMD_RECORD_START"
  add_pane "$WINDOW_DATA" "$WORK_DIR" "$ROS_SETUP" "$CMD_RECORD_STOP"
  add_pane "$WINDOW_DATA" "$PYTHON_DIR" "$ROS_SETUP" "$CMD_BUILD_MAP_LOOKUP"
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_localization_eval_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_PLAY_BAG"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_LOCALIZATION_EVAL"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_LOCALIZATION_TRIGGER"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_DASHBOARD_LOCALIZATION"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$RVIZ_LOCALIZATION_EVAL"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  create_layout_from_panes "$WINDOW_MAIN" 1
}

create_perception_eval_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_PLAY_BAG"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_PERCEPTION_EVAL"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_DEBUG_IMAGE_VIEWER"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_PERCEPTION_LABEL"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_PERCEPTION_CONFIDENCE"
  create_layout_from_panes "$WINDOW_MAIN" 1
}

create_vslam_eval_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_PLAY_BAG"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_VSLAM_EVAL"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_MONITOR"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$RVIZ_VSLAM_DEBUG"
  add_pane "$WINDOW_VISUAL" "$WORK_DIR" "$ROS_SETUP" "$CMD_LEFT_IMAGE_VIEWER"
  create_layout_from_panes "$WINDOW_MAIN" 1
}

create_simulator_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_SIMULATOR"
  add_pane "$WINDOW_MAIN" "$WORK_DIR" "$ROS_SETUP" "$CMD_LOCALIZATION_TRIGGER"
  create_layout_from_panes "$WINDOW_MAIN" 0
}

