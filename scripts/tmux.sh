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




MODE_TAMIYA="tamiya"
MODE_PYTHON="python"
MODE_MAP="map"
MODE_IDENTIFICATION="identification"
MODE_LOCALIZATION_EVAL="localization_eval"
MODE_PERCEPTION_EVAL="perception_eval"
MODE_VSLAM_EVAL="vslam_eval"
MODE_SIMULATOR="simulator"

DEFAULT_SESSION_TAMIYA="tamiya"
DEFAULT_SESSION_PYTHON="python"
DEFAULT_SESSION_MAP="map"
DEFAULT_SESSION_IDENTIFICATION="identification"
DEFAULT_SESSION_LOCALIZATION_EVAL="localization_eval"
DEFAULT_SESSION_PERCEPTION_EVAL="perception_eval"
DEFAULT_SESSION_VSLAM_EVAL="vslam_eval"
DEFAULT_SESSION_SIMULATOR="simulator"

WINDOW_MAIN="main"
WINDOW_DATA="data"
WINDOW_EVAL="eval"
WINDOW_VISUAL="visual"

SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(resolve_repo_root)"



SETUP_SCRIPT="$(resolve_setup_command)"
ROS_SETUP="export ROS_LOCALHOST_ONLY=0"
if [[ -n "$SETUP_SCRIPT" ]]; then
  ROS_SETUP="${ROS_SETUP} && ${SETUP_SCRIPT}"
fi

WORK_DIR="$(resolve_existing_dir "$REPO_ROOT" /workspaces)"
RECORD_DIR="$(resolve_existing_dir /record "${REPO_ROOT}/record" "$WORK_DIR")"
PYTHON_DIR="$(resolve_existing_dir "${REPO_ROOT}/python_ws" /python_ws "$WORK_DIR")"
SCRIPTS_DIR="$SCRIPT_DIR"

LAUNCH_SYSTEM_SH="${SCRIPT_DIR}/launch_system.sh"
MONITOR_SH="${SCRIPT_DIR}/monitor.sh"
CREATE_VSLAM_MAP_SH="${SCRIPT_DIR}/create_vslam_map_from_bag.sh"
CREATE_MAP_SH="${SCRIPT_DIR}/create_2d_map_from_bag.sh"
TERMINAL_DASHBOARD_PY="${SCRIPT_DIR}/terminal_viewers/ros2_terminal_dashboard.py"
TERMINAL_IMAGE_VIEWER_PY="${SCRIPT_DIR}/terminal_viewers/ros2_terminal_image_viewer.py"

CMD_PRODUCTION="bash ${LAUNCH_SYSTEM_SH} production --set map_dir=<map_dir>"
CMD_MONITOR="bash ${MONITOR_SH}"
CMD_LOCALIZATION_TRIGGER='ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'
CMD_CREATE_VSLAM_MAP="bash ${CREATE_VSLAM_MAP_SH} --mode vslam --rate 1.0"
CMD_CREATE_MAP="bash ${CREATE_MAP_SH} --mode no_odom_online_vslam --rate 1.0 --prepare-vslam-map-alignment --trace-vslam-landmarks"
CMD_IDENTIFICATION="bash ${LAUNCH_SYSTEM_SH} identification"
CMD_PLAY_BAG="ros2 bag play <bag_path> --clock --start-paused"
CMD_LOCALIZATION_EVAL="bash ${LAUNCH_SYSTEM_SH} localization_eval --set map_dir=<map_dir>"
CMD_PERCEPTION_EVAL="bash ${LAUNCH_SYSTEM_SH} perception_eval"
CMD_VSLAM_EVAL="bash ${LAUNCH_SYSTEM_SH} vslam_eval --set map_dir=<map_dir>"
CMD_SIMULATOR="ros2 launch system_launch simulator.launch.xml use_ftg:=false record:=false rviz:=false localization:=false"
CMD_RECORD_START='ros2 service call /bag_manager_node/start_recording std_srvs/srv/Trigger "{}"'
CMD_RECORD_STOP='ros2 service call /bag_manager_node/stop_recording std_srvs/srv/Trigger "{}"'
CMD_BUILD_MAP_LOOKUP='python data_analysis/build_map_steering_lookup.py --bag /record/<session_timestamp>/<take_timestamp>/metadata.yaml'
CMD_DASHBOARD_IDENTIFICATION="python3 ${TERMINAL_DASHBOARD_PY} --map-topic '' --localization-topic '' --scan-topic '' --odom-topic /visual_slam/tracking/odometry --image-topic /camera/left/image_raw --camera-info-topic /camera/left/camera_info --best-effort"
CMD_DASHBOARD_LOCALIZATION="python3 ${TERMINAL_DASHBOARD_PY} --map-topic /map --localization-topic /localization_result --scan-topic /scan --odom-topic /visual_slam/tracking/odometry --image-topic /camera/left/image_raw --camera-info-topic /camera/left/camera_info --best-effort"
CMD_LEFT_IMAGE_VIEWER="python3 ${TERMINAL_IMAGE_VIEWER_PY} --topic /camera/left/image_raw --best-effort --max-fps 10"
CMD_DEBUG_IMAGE_VIEWER="python3 ${TERMINAL_IMAGE_VIEWER_PY} --topic /perception/debug/image --best-effort --max-fps 10"
CMD_PERCEPTION_LABEL="ros2 topic echo /perception/classification/label"
CMD_PERCEPTION_CONFIDENCE="ros2 topic echo /perception/classification/confidence"
RVIZ_LOCALIZATION_EVAL='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/localization_eval.rviz --ros-args -p use_sim_time:=true'
RVIZ_VSLAM_ALIGNMENT='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_map_alignment.rviz'
RVIZ_VSLAM_DEBUG='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_debug.rviz --ros-args -p use_sim_time:=true'

PANE_WINDOWS=()
PANE_DIRS=()
PANE_SETUPS=()
PANE_PREPARES=()





















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
    "$MODE_TAMIYA"|"$MODE_PYTHON"|"$MODE_MAP"|"$MODE_IDENTIFICATION"|"$MODE_LOCALIZATION_EVAL"|"$MODE_PERCEPTION_EVAL"|"$MODE_VSLAM_EVAL"|"$MODE_SIMULATOR")
      if [[ -z "$MODE" ]]; then
        MODE="$1"
      elif [[ -z "$SESSION_NAME" ]]; then
        SESSION_NAME="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
    *)
      if [[ -z "$SESSION_NAME" ]]; then
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

case "$MODE" in
  "$MODE_TAMIYA"|"$MODE_PYTHON"|"$MODE_MAP"|"$MODE_IDENTIFICATION"|"$MODE_LOCALIZATION_EVAL"|"$MODE_PERCEPTION_EVAL"|"$MODE_VSLAM_EVAL"|"$MODE_SIMULATOR")
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    echo "Use --mode tamiya, python, map, identification, localization_eval, perception_eval, vslam_eval, or simulator" >&2
    exit 1
    ;;
esac

if [[ -z "$SESSION_NAME" ]]; then
  case "$MODE" in
    "$MODE_TAMIYA")
      SESSION_NAME="$DEFAULT_SESSION_TAMIYA"
      ;;
    "$MODE_PYTHON")
      SESSION_NAME="$DEFAULT_SESSION_PYTHON"
      ;;
    "$MODE_MAP")
      SESSION_NAME="$DEFAULT_SESSION_MAP"
      ;;
    "$MODE_IDENTIFICATION")
      SESSION_NAME="$DEFAULT_SESSION_IDENTIFICATION"
      ;;
    "$MODE_LOCALIZATION_EVAL")
      SESSION_NAME="$DEFAULT_SESSION_LOCALIZATION_EVAL"
      ;;
    "$MODE_PERCEPTION_EVAL")
      SESSION_NAME="$DEFAULT_SESSION_PERCEPTION_EVAL"
      ;;
    "$MODE_VSLAM_EVAL")
      SESSION_NAME="$DEFAULT_SESSION_VSLAM_EVAL"
      ;;
    *)
      SESSION_NAME="$DEFAULT_SESSION_SIMULATOR"
      ;;
  esac
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  case "$MODE" in
    "$MODE_TAMIYA")
      create_tamiya_layout
      ;;
    "$MODE_PYTHON")
      create_python_layout
      ;;
    "$MODE_MAP")
      create_map_layout
      ;;
    "$MODE_IDENTIFICATION")
      create_identification_layout
      ;;
    "$MODE_LOCALIZATION_EVAL")
      create_localization_eval_layout
      ;;
    "$MODE_PERCEPTION_EVAL")
      create_perception_eval_layout
      ;;
    "$MODE_VSLAM_EVAL")
      create_vslam_eval_layout
      ;;
    "$MODE_SIMULATOR")
      create_simulator_layout
      ;;
  esac
fi

tmux attach-session -t "$SESSION_NAME"
