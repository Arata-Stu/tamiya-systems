#!/bin/bash

set -euo pipefail

INITIAL_PWD="${PWD}"

resolve_real_path() {
  local input_path="$1"

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$input_path" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
    return 0
  fi

  if command -v perl >/dev/null 2>&1; then
    perl -MCwd=realpath -e 'print realpath($ARGV[0])' "$input_path"
    return 0
  fi

  printf '%s\n' "$input_path"
}

SCRIPT_PATH="$(resolve_real_path "${BASH_SOURCE[0]}")"

is_repo_root() {
  local candidate="$1"

  [[ -d "$candidate" ]] || return 1
  [[ -d "$candidate/scripts" ]] || return 1
  [[ -d "$candidate/ros2_ws" ]] || return 1
  [[ -f "$candidate/scripts/tmux.sh" ]] || return 1
}

find_repo_root_from() {
  local current="$1"

  [[ -n "$current" ]] || return 1

  if [[ -f "$current" ]]; then
    current="$(dirname "$current")"
  fi

  [[ -d "$current" ]] || return 1

  while true; do
    if is_repo_root "$current"; then
      (cd "$current" && pwd)
      return 0
    fi

    [[ "$current" == "/" ]] && break
    current="$(dirname "$current")"
  done

  return 1
}

resolve_repo_root() {
  local candidate

  for candidate in \
    "${INITIAL_PWD}" \
    "${SCRIPT_PATH}" \
    "$(dirname "${SCRIPT_PATH}")" \
    "$(dirname "$(dirname "${SCRIPT_PATH}")")" \
    /workspaces \
    /workspace; do
    [[ -n "$candidate" ]] || continue
    if find_repo_root_from "$candidate"; then
      return 0
    fi
  done

  (cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)
}

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

resolve_existing_dir() {
  local candidate

  for candidate in "$@"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

resolve_setup_command() {
  local candidate

  for candidate in \
    "${TMUX_WORKSPACE_SETUP:-}" \
    "${REPO_ROOT}/install/setup.bash" \
    "/workspaces/install/setup.bash" \
    "install/setup.bash"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf 'source %s\n' "$candidate"
      return 0
    fi
  done

  printf '\n'
}

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
CMD_CREATE_MAP="bash ${CREATE_MAP_SH} --mode no_odom_online_vslam --rate 1.0 --live-vslam-map-align --trace-vslam-landmarks"
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
RVIZ_VSLAM_ALIGNMENT='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_map_alignment.rviz --ros-args -p use_sim_time:=true'
RVIZ_VSLAM_DEBUG='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_debug.rviz --ros-args -p use_sim_time:=true'

PANE_WINDOWS=()
PANE_DIRS=()
PANE_SETUPS=()
PANE_PREPARES=()

reset_panes() {
  PANE_WINDOWS=()
  PANE_DIRS=()
  PANE_SETUPS=()
  PANE_PREPARES=()
}

add_pane() {
  local window="$1"
  local dir="${2:-}"
  local setup="${3:-}"
  local prepare="${4:-}"

  PANE_WINDOWS+=("$window")
  PANE_DIRS+=("$dir")
  PANE_SETUPS+=("$setup")
  PANE_PREPARES+=("$prepare")
}

pane_count_for_window() {
  local window="$1"
  local count=0
  local idx

  for idx in "${!PANE_WINDOWS[@]}"; do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

window_exists_in_specs() {
  local window="$1"
  local idx

  for idx in "${!PANE_WINDOWS[@]}"; do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      return 0
    fi
  done

  return 1
}

create_window_panes() {
  local window="$1"
  local pane_count
  local idx

  pane_count="$(pane_count_for_window "$window")"
  for ((idx = 1; idx < pane_count; idx++)); do
    tmux split-window -v -t "$SESSION_NAME":"$window".0
  done
  tmux select-layout -t "$SESSION_NAME":"$window" tiled >/dev/null
}

pane_index_for_spec() {
  local spec_idx="$1"
  local window="${PANE_WINDOWS[$spec_idx]}"
  local pane_index=0
  local idx

  for ((idx = 0; idx < spec_idx; idx++)); do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      pane_index=$((pane_index + 1))
    fi
  done

  echo "$pane_index"
}

build_init_cmd() {
  local dir="$1"
  local setup="$2"
  local cmd=""

  if [[ -n "$dir" ]]; then
    cmd="cd $dir"
  fi

  if [[ -n "$setup" ]]; then
    if [[ -n "$cmd" ]]; then
      cmd="$cmd && $setup"
    else
      cmd="$setup"
    fi
  fi

  echo "$cmd"
}

init_pane() {
  local target="$1"
  local cmd="$2"
  [[ -z "$cmd" ]] && return
  tmux send-keys -t "$target" "$cmd" C-m
}

prepare_cmd() {
  local target="$1"
  local cmd="$2"

  tmux send-keys -t "$target" C-l
  sleep 0.2

  if [[ -n "$cmd" ]]; then
    tmux send-keys -t "$target" "$cmd"
  fi
}

create_layout_from_panes() {
  local select_window="$1"
  local select_pane="${2:-0}"
  local idx
  local window
  local pane_index
  local init_cmd
  local created_windows=" "

  if [[ "${#PANE_WINDOWS[@]}" -eq 0 ]]; then
    echo "No pane specs configured" >&2
    exit 1
  fi

  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "${PANE_WINDOWS[0]}"

  for window in "${PANE_WINDOWS[@]}"; do
    if [[ "$created_windows" == *" $window "* ]]; then
      continue
    fi

    if [[ "$window" != "${PANE_WINDOWS[0]}" ]]; then
      tmux new-window -t "$SESSION_NAME" -n "$window"
    fi

    created_windows="$created_windows$window "
  done

  created_windows=" "
  for window in "${PANE_WINDOWS[@]}"; do
    if [[ "$created_windows" == *" $window "* ]]; then
      continue
    fi

    create_window_panes "$window"
    created_windows="$created_windows$window "
  done

  for idx in "${!PANE_WINDOWS[@]}"; do
    window="${PANE_WINDOWS[$idx]}"
    pane_index="$(pane_index_for_spec "$idx")"

    init_cmd="$(build_init_cmd "${PANE_DIRS[$idx]}" "${PANE_SETUPS[$idx]}")"
    init_pane "$SESSION_NAME":"$window"."$pane_index" "$init_cmd"
  done

  sleep 2.0

  for idx in "${!PANE_WINDOWS[@]}"; do
    window="${PANE_WINDOWS[$idx]}"
    pane_index="$(pane_index_for_spec "$idx")"

    prepare_cmd "$SESSION_NAME":"$window"."$pane_index" "${PANE_PREPARES[$idx]}"
  done

  if window_exists_in_specs "$select_window"; then
    tmux select-window -t "$SESSION_NAME":"$select_window"
    tmux select-pane -t "$SESSION_NAME":"$select_window"."$select_pane"
  fi
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [SESSION_NAME] [--mode tamiya|python|map|identification|localization_eval|perception_eval|vslam_eval|simulator]
  $(basename "$0") [--session SESSION_NAME] [--mode tamiya|python|map|identification|localization_eval|perception_eval|vslam_eval|simulator]

Notes:
  - Replace <map_dir> and <bag_path> placeholders in the prepared commands.
  - map mode prepares the live VSLAM-alignment + landmark-tracing flow and also keeps a lean localization-eval window.
EOF
}

choose_mode_interactive() {
  local answer

  if [[ ! -t 0 ]]; then
    echo "$MODE_TAMIYA"
    return
  fi

  while true; do
    echo "Select mode:" >&2
    echo "  1) $MODE_TAMIYA (production + monitor)" >&2
    echo "  2) $MODE_PYTHON (python workspace)" >&2
    echo "  3) $MODE_MAP (live alignment map creation + localization eval)" >&2
    echo "  4) $MODE_IDENTIFICATION (live VSLAM + bag recording for MAP lookup)" >&2
    echo "  5) $MODE_LOCALIZATION_EVAL (bag replay + localization eval)" >&2
    echo "  6) $MODE_PERCEPTION_EVAL (bag replay + perception eval)" >&2
    echo "  7) $MODE_VSLAM_EVAL (bag replay + VSLAM eval)" >&2
    echo "  8) $MODE_SIMULATOR (simulator setup)" >&2
    read -r -p "Enter 1-8: " answer

    case "$answer" in
      1|"$MODE_TAMIYA")
        echo "$MODE_TAMIYA"
        return
        ;;
      2|"$MODE_PYTHON")
        echo "$MODE_PYTHON"
        return
        ;;
      3|"$MODE_MAP")
        echo "$MODE_MAP"
        return
        ;;
      4|"$MODE_IDENTIFICATION")
        echo "$MODE_IDENTIFICATION"
        return
        ;;
      5|"$MODE_LOCALIZATION_EVAL")
        echo "$MODE_LOCALIZATION_EVAL"
        return
        ;;
      6|"$MODE_PERCEPTION_EVAL")
        echo "$MODE_PERCEPTION_EVAL"
        return
        ;;
      7|"$MODE_VSLAM_EVAL")
        echo "$MODE_VSLAM_EVAL"
        return
        ;;
      8|"$MODE_SIMULATOR")
        echo "$MODE_SIMULATOR"
        return
        ;;
      *)
        echo "Invalid choice: $answer" >&2
        ;;
    esac
  done
}

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
