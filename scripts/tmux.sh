#!/bin/bash


# --- Source Library Modules ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for lib in "${SCRIPT_DIR}/lib/tmux/"*.sh; do
    source "${lib}"
done
# ------------------------------

set -euo pipefail

INITIAL_PWD="${PWD}"


SCRIPT_PATH="$(resolve_real_path "${BASH_SOURCE[0]}")"




MODE_RECORD="record"
MODE_RECORD_MAPPING="record_mapping"
MODE_RACE="race"
MODE_RACE_PP="race_pp"
MODE_RACE_E2E="race_e2e"
MODE_MAP="map"
MODE_IDENTIFICATION="identification"
MODE_E2E="e2e"
MODE_E2E_TRAIN="e2e_train"
MODE_LIDAR_E2E="lidar_e2e"
MODE_LIDAR_E2E_TRAIN="lidar_e2e_train"
MODE_RECORD_VIRTUAL_SCAN="record_virtual_scan"
MODE_OFFLINE_EVAL="offline_eval"
MODE_HD_MAP_EVAL="hd_map_eval"
MODE_PERCEPTION_EVAL="perception_eval"

# Legacy aliases used in ui_utils.sh
MODE_MAPPING="mapping"
MODE_MAP_BUILD="map_build"
MODE_HD_MAP="hd_map"
MODE_TAMIYA="tamiya"
MODE_PRODUCTION="production"
MODE_E2E_BACKUP="e2e_backup"

WINDOW_RECORD="record"
WINDOW_MAP="map"
WINDOW_DATA="data"
WINDOW_EVAL="eval"
WINDOW_VISUAL="visual"
WINDOW_TOOLS="tools"
WINDOW_RACE="race"
WINDOW_E2E="e2e"
WINDOW_TRAIN="train"

SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(resolve_repo_root)"



SETUP_SCRIPT="$(resolve_setup_command)"
ROS_SETUP="export ROS_LOCALHOST_ONLY=0"
if [[ -n "$SETUP_SCRIPT" ]]; then
  ROS_SETUP="${ROS_SETUP} && ${SETUP_SCRIPT}"
fi

WORK_DIR="$(resolve_existing_dir "$REPO_ROOT" /workspaces)"
PYTHON_DIR="$(resolve_existing_dir "${REPO_ROOT}/python_ws" /python_ws "$WORK_DIR")"
CAMERA_E2E_DIR="$(resolve_existing_dir "${PYTHON_DIR}/camera_e2e" /python_ws/camera_e2e "$PYTHON_DIR")"
LIDAR_E2E_DIR="$(resolve_existing_dir "${PYTHON_DIR}/lidar_e2e" /python_ws/lidar_e2e "$PYTHON_DIR")"

LAUNCH_SYSTEM_SH="${SCRIPT_DIR}/launch_system.sh"
MONITOR_SH="${SCRIPT_DIR}/monitor.sh"
CREATE_MAP_AND_HD_MAP_SH="${SCRIPT_DIR}/create_map_and_hd_map_from_bag.sh"
CREATE_VIRTUAL_SCAN_SH="${SCRIPT_DIR}/create_virtual_scan_from_bag.sh"
EDIT_HD_MAP_SECTIONS_SH="${SCRIPT_DIR}/edit_hd_map_sections.sh"
BUILD_SPEED_FEEDFORWARD_SH="${SCRIPT_DIR}/build_speed_feedforward_from_bag.sh"
TERMINAL_DASHBOARD_PY="${SCRIPT_DIR}/terminal_viewers/ros2_terminal_dashboard.py"
TERMINAL_IMAGE_VIEWER_PY="${SCRIPT_DIR}/terminal_viewers/ros2_terminal_image_viewer.py"
TERMINAL_VSLAM_DASHBOARD_PY="${SCRIPT_DIR}/terminal_viewers/ros2_terminal_vslam_dashboard.py"

CMD_RECORD_MAPPING="bash ${LAUNCH_SYSTEM_SH} record_mapping"
CMD_RECORD_MAPPING_DEBUG="bash ${LAUNCH_SYSTEM_SH} record_mapping_debug"
CMD_RACE_MAP="bash ${LAUNCH_SYSTEM_SH} race --set map_dir=<map_dir>"
CMD_RACE_PP="bash ${LAUNCH_SYSTEM_SH} race_pp --set map_dir=<map_dir>"
CMD_RACE_E2E="bash ${LAUNCH_SYSTEM_SH} race_e2e --set map_dir=<map_dir>"
CMD_HD_MAP_EVAL="bash ${LAUNCH_SYSTEM_SH} hd_map_eval --set map_dir=<map_dir> --set use_virtual_scan=true"
CMD_LOCALIZATION_TRIGGER='ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'
CMD_CREATE_HD_MAP="bash ${CREATE_MAP_AND_HD_MAP_SH} --rate 1.0 --editor-scale 0"
CMD_EDIT_HD_MAP_SECTIONS="bash ${EDIT_HD_MAP_SECTIONS_SH} --map-dir <map_dir> --scale 0"
CMD_IDENTIFICATION="bash ${LAUNCH_SYSTEM_SH} identification"
CMD_PLAY_BAG="ros2 bag play <bag_path> --clock --start-paused"
CMD_RECORD_VIRTUAL_SCAN_RUN="bash ${LAUNCH_SYSTEM_SH} record_virtual_scan --set map_dir=<map_dir>"
CMD_LIDAR_E2E_RUN="bash ${LAUNCH_SYSTEM_SH} lidar_e2e --set map_dir=<map_dir>"
CMD_E2E_CAMERA_BACKUP="bash ${LAUNCH_SYSTEM_SH} e2e_backup"
CMD_RECORD_START='ros2 service call /bag_manager_node/start_recording std_srvs/srv/Trigger "{}"'
CMD_RECORD_STOP='ros2 service call /bag_manager_node/stop_recording std_srvs/srv/Trigger "{}"'
CMD_E2E_PREPROCESS="bash ${PYTHON_DIR}/camera_e2e/1_create_dataset.sh"
CMD_E2E_TRAIN="python3 ${PYTHON_DIR}/camera_e2e/2_train.py"
CMD_E2E_SCP="bash ${PYTHON_DIR}/camera_e2e/scp_ckpts.sh"
CMD_E2E_DEPLOY="bash ${PYTHON_DIR}/camera_e2e/3_deploy_model.sh"
CMD_LIDAR_E2E_GENERATE_VIRTUAL_SCAN="bash ${CREATE_VIRTUAL_SCAN_SH} --bag-path <source_bag_path> --map-dir <map_dir> --output-root /record/virtual_scan --virtual-scan-topic /virtual_scan --cmd-topic /jetracer/cmd_drive"
CMD_LIDAR_E2E_CHECK_TOPICS="ros2 bag info /record/virtual_scan/<generated_run>/virtual_scan_bag"
CMD_LIDAR_E2E_PREPROCESS="bash ./1_create_dataset.sh -b /record/virtual_scan -o ./datasets --scan_topic /virtual_scan --cmd_topic /jetracer/cmd_drive --use_virtual_scan"
CMD_LIDAR_E2E_TRAIN="python3 ./2_train.py data_path=./datasets dataset.use_virtual_scan=true"
CMD_LIDAR_E2E_TENSORBOARD="tensorboard --logdir ./logs/train --bind_all --port 6006"
CMD_LIDAR_E2E_SCP="bash ./scp_ckpts.sh"
CMD_LIDAR_E2E_DEPLOY="bash ./3_deploy_model.sh ./ckpts --precision fp16 --scan-points 320"
CMD_PUSH_MAP="bash ${SCRIPT_DIR}/push_map.sh"
CMD_BUILD_MAP_LOOKUP='python data_analysis/build_map_steering_lookup.py --bag /record/<session_timestamp>/<take_timestamp>/metadata.yaml'
CMD_BUILD_SPEED_FEEDFORWARD="bash ${BUILD_SPEED_FEEDFORWARD_SH} --bag /record/<session_timestamp>/<take_timestamp>/metadata.yaml --map-dir <map_dir> -- --max-abs-steer 0.10"
CMD_APPLY_SECTION_SPEEDS='python data_analysis/apply_hd_map_section_speeds.py --raceline /map/<course>/<course>_raceline.csv --hd-map /map/<course>/<course>_hd_map.yaml --output /map/<course>/<course>_raceline_section_speeds.csv'
CMD_DASHBOARD_IDENTIFICATION="python3 ${TERMINAL_DASHBOARD_PY} --map-topic '' --localization-topic '' --scan-topic '' --odom-topic /visual_slam/tracking/odometry --image-topic /camera/left/image_raw --camera-info-topic /camera/left/camera_info --best-effort"
CMD_DASHBOARD_RECORD_MAPPING="python3 ${TERMINAL_DASHBOARD_PY} --map-topic '' --localization-topic '' --amcl-pose-topic '' --initial-pose-topic '' --scan-topic /scan --odom-topic '' --image-topic /camera/left/image_raw --camera-info-topic /camera/left/camera_info --crop-image-topic '' --particles-topic '' --path-topic '' --vo-path-topic '' --global-path-topic '' --local-path-topic '' --section-markers-topic '' --hd-lane-markers-topic '' --hd-section-markers-topic '' --current-section-marker-topic '' --current-section-topic '' --best-effort"
CMD_DASHBOARD_OFFLINE_EVAL="python3 ${TERMINAL_DASHBOARD_PY} --map-topic '' --localization-topic '' --scan-topic /virtual_scan --odom-topic /visual_slam/tracking/odometry --image-topic /camera/left/image_raw --camera-info-topic /camera/left/camera_info --global-path-topic /planning/global_raceline --local-path-topic /autonomous/trajectory --section-markers-topic '' --hd-lane-markers-topic /hd_map/lane_markers --hd-section-markers-topic /hd_map/section_markers --current-section-marker-topic /localization/current_section_marker --current-section-topic /localization/current_section --best-effort"
CMD_DASHBOARD_VSLAM_HD="MAP_DIR=<map_dir>; MAP_NAME=\"\$(basename \"\${MAP_DIR%/}\")\"; python3 ${TERMINAL_VSLAM_DASHBOARD_PY} --hd-map-yaml \"\${MAP_DIR%/}/\${MAP_NAME}_hd_map.yaml\""
CMD_LEFT_IMAGE_VIEWER="python3 ${TERMINAL_IMAGE_VIEWER_PY} --topic /camera/left/image_raw --best-effort --max-fps 10"
CMD_SPEED_DEBUG='ros2 topic echo /speed_controller/throttle_cmd'
RVIZ_LOCALIZATION_EVAL='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/localization_eval.rviz --ros-args -p use_sim_time:=true'
CMD_PERCEPTION_RUN="ros2 launch system_launch perception.launch.xml use_sim_time:=true use_classifier:=true"
CMD_CROP_IMAGE_VIEWER="python3 ${TERMINAL_IMAGE_VIEWER_PY} --topic /perception/crop/image --best-effort --max-fps 10"
CMD_DEBUG_IMAGE_VIEWER="python3 ${TERMINAL_IMAGE_VIEWER_PY} --topic /perception/debug/image --best-effort --max-fps 10"
CMD_ECHO_CLASSIFIER="ros2 topic echo /perception/classification/target_detected"
CMD_TF_ECHO="ros2 run tf2_ros tf2_echo base_link camera_left_optical_frame"

PANE_WINDOWS=()
PANE_DIRS=()
PANE_SETUPS=()
PANE_PREPARES=()

is_mode_token() {
  case "$1" in
    "$MODE_RECORD"|"$MODE_RECORD_MAPPING"|"$MODE_MAP"|"$MODE_RACE"|"$MODE_RACE_PP"|"$MODE_RACE_E2E"|"$MODE_E2E"|"$MODE_LIDAR_E2E"|"$MODE_LIDAR_E2E_TRAIN"|"$MODE_IDENTIFICATION"|"$MODE_E2E_TRAIN"|"$MODE_OFFLINE_EVAL"|"$MODE_HD_MAP_EVAL"|"$MODE_RECORD_VIRTUAL_SCAN"|"$MODE_PERCEPTION_EVAL")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_mode() {
  case "$1" in
    "$MODE_RECORD"|"$MODE_RECORD_MAPPING")
      echo "$MODE_RECORD"
      ;;
    "$MODE_MAP")
      echo "$MODE_MAP"
      ;;
    "$MODE_RACE")
      echo "$MODE_RACE"
      ;;
    "$MODE_RACE_PP")
      echo "$MODE_RACE_PP"
      ;;
    "$MODE_RACE_E2E")
      echo "$MODE_RACE_E2E"
      ;;
    "$MODE_E2E")
      echo "$MODE_E2E"
      ;;
    "$MODE_LIDAR_E2E")
      echo "$MODE_LIDAR_E2E"
      ;;
    "$MODE_LIDAR_E2E_TRAIN")
      echo "$MODE_LIDAR_E2E_TRAIN"
      ;;
    "$MODE_E2E_TRAIN")
      echo "$MODE_E2E_TRAIN"
      ;;
    "$MODE_IDENTIFICATION")
      echo "$MODE_IDENTIFICATION"
      ;;
    "$MODE_OFFLINE_EVAL"|"$MODE_HD_MAP_EVAL")
      echo "$MODE_OFFLINE_EVAL"
      ;;
    "$MODE_RECORD_VIRTUAL_SCAN")
      echo "$MODE_RECORD_VIRTUAL_SCAN"
      ;;
    "$MODE_PERCEPTION_EVAL")
      echo "$MODE_PERCEPTION_EVAL"
      ;;
    *)
      return 1
      ;;
  esac
}





















MODE=""
SESSION_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --mode|-m)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    --session|-s)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      SESSION_NAME="$2"
      shift 2
      ;;
    *)
      if is_mode_token "$1"; then
        if [[ -z "$MODE" ]]; then
          MODE="$(normalize_mode "$1")"
        elif [[ -z "$SESSION_NAME" ]]; then
          SESSION_NAME="$1"
        else
          echo "Unexpected argument: $1" >&2
          usage
          exit 1
        fi
      elif [[ -z "$SESSION_NAME" ]]; then
        SESSION_NAME="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  MODE="$(choose_mode_interactive)"
fi

if ! MODE="$(normalize_mode "$MODE")"; then
  echo "Invalid mode: $MODE" >&2
  echo "Use --mode record, map, race, e2e, lidar_e2e_train, or identification" >&2
  exit 1
fi

if [[ -z "$SESSION_NAME" ]]; then
  SESSION_NAME="$MODE"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  case "$MODE" in
    "$MODE_RECORD")
      create_record_layout
      ;;
    "$MODE_RACE")
      create_race_layout
      ;;
    "$MODE_RACE_PP")
      create_race_pp_layout
      ;;
    "$MODE_RACE_E2E")
      create_race_e2e_layout
      ;;
    "$MODE_MAP")
      create_map_layout
      ;;
    "$MODE_IDENTIFICATION")
      create_identification_layout
      ;;
    "$MODE_OFFLINE_EVAL")
      create_eval_layout
      ;;
    "$MODE_RECORD_VIRTUAL_SCAN")
      create_record_virtual_scan_layout
      ;;
    "$MODE_PERCEPTION_EVAL")
      create_perception_eval_layout
      ;;
    "$MODE_E2E")
      create_e2e_layout
      ;;
    "$MODE_LIDAR_E2E")
      create_lidar_e2e_layout
      ;;
    "$MODE_LIDAR_E2E_TRAIN")
      create_lidar_e2e_train_layout
      ;;
    "$MODE_E2E_TRAIN")
      create_e2e_train_layout
      ;;
  esac
fi

if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION_NAME"
else
  tmux attach-session -t "$SESSION_NAME"
fi
