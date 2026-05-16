#!/bin/bash

set -euo pipefail

# --- 設定項目 ---
MODE_TAMIYA="tamiya"
MODE_PYTHON="python"
MODE_MAP="map"
MODE_SIMULATOR="simulator"

DEFAULT_SESSION_TAMIYA="tamiya"
DEFAULT_SESSION_PYTHON="python"
DEFAULT_SESSION_MAP="map"
DEFAULT_SESSION_SIMULATOR="simulator"

WINDOW_MAIN="main"
WINDOW_DATA="data"
WINDOW_LOCALIZATION_EVAL="localization_eval"

SETUP_SCRIPT="source /workspaces/install/setup.bash"
CMD_BASE="bash /scripts/launch_system.sh production -- map_dir:=<map_dir>"
CMD_MONITOR="bash /scripts/monitor.sh"
CMD_LOCALIZATION_TRIGGER='ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'
CMD_CREATE_VSLAM_MAP="bash /scripts/create_vslam_map_from_bag.sh --mode vslam --rate 1.0"
CMD_CREATE_MAP="bash /scripts/create_2d_map_from_bag.sh --mode 2d_slam --rate 1.0"
CMD_PLAY_BAG="ros2 bag play <bag_path> --clock --start-paused"
CMD_LOCALIZATION_EVAL="bash /scripts/launch_system.sh production -- map_dir:=<map_dir>"
CMD_LIDAR_CONTAINER="ros2 run rclcpp_components component_container --ros-args -r __node:=lidar_container"
CMD_SIMULATOR="ros2 launch system_launch simulator.launch.xml use_ftg:=false record:=false rviz:=false localization:=false"
RVIZ_LOCALIZATION_EVAL='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/localization_eval.rviz'
RVIZ_VSLAM_DEBUG='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/vslam_debug.rviz'

PYTHON_PANE1_DIR="/python_ws"
PYTHON_PANE2_DIR="/record/"

PANE_WINDOWS=()
PANE_DIRS=()
PANE_SETUPS=()
PANE_PREPARES=()

# --- ヘルパー関数 ---
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

# ペインの初期化（cd/source などを実行してEnter）
init_pane() {
  local target="$1"
  local cmd="$2"
  [[ -z "$cmd" ]] && return
  tmux send-keys -t "$target" "$cmd" C-m
}

# ペインに準備コマンドを流し込む（表示崩れ対策済み）
prepare_cmd() {
  local target="$1"
  local cmd="$2"
  
  # bashのショートカット(Ctrl+L)で画面をクリア＆プロンプトを綺麗に再描画
  tmux send-keys -t "$target" C-l
  sleep 0.2
  
  # コマンドを文字入力（Enterは押さない）
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

  # ターミナル幅が狭いと判断されないよう、初期仮想サイズを大きく設定 (-x 250 -y 80)
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
  $(basename "$0") [SESSION_NAME] [--mode tamiya|python|map|simulator]
  $(basename "$0") [--session SESSION_NAME] [--mode tamiya|python|map|simulator]

If mode is omitted, you can choose interactively.
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
    echo "  1) $MODE_TAMIYA (current setup)" >&2
    echo "  2) $MODE_PYTHON (study setup)" >&2
    echo "  3) $MODE_MAP (map creation setup)" >&2
    echo "  4) $MODE_SIMULATOR (simulator setup)" >&2
    read -r -p "Enter 1, 2, 3, or 4: " answer

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
      4|"$MODE_SIMULATOR")
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
  # add_pane <window> <cd directory> <setup command> <prepared command>
  # 同じ window に add_pane を足すと、pane 分割は自動で増える。
  add_pane "$WINDOW_MAIN" "" "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT" "$CMD_BASE"
  add_pane "$WINDOW_MAIN" "" "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT" "$CMD_MONITOR"
  add_pane "$WINDOW_DATA" "/record" "" ""
  add_pane "$WINDOW_DATA" "/scripts/" "" ""
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_python_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "$PYTHON_PANE1_DIR" "$SETUP_SCRIPT" ""
  add_pane "$WINDOW_MAIN" "$PYTHON_PANE2_DIR" "$SETUP_SCRIPT" ""
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_map_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "/workspaces" "$SETUP_SCRIPT" "$CMD_CREATE_MAP"
  add_pane "$WINDOW_MAIN" "" "$SETUP_SCRIPT" "$RVIZ_VSLAM_DEBUG"
  add_pane "$WINDOW_LOCALIZATION_EVAL" "/workspaces" "$SETUP_SCRIPT" "$CMD_PLAY_BAG"
  add_pane "$WINDOW_LOCALIZATION_EVAL" "/workspaces" "$SETUP_SCRIPT" "$CMD_LOCALIZATION_EVAL"
  add_pane "$WINDOW_LOCALIZATION_EVAL" "/workspaces" "$SETUP_SCRIPT" "$CMD_LOCALIZATION_TRIGGER"
  add_pane "$WINDOW_LOCALIZATION_EVAL" "/workspaces" "$SETUP_SCRIPT" "$CMD_LIDAR_CONTAINER"
  add_pane "$WINDOW_LOCALIZATION_EVAL" "/workspaces" "$SETUP_SCRIPT" "$RVIZ_LOCALIZATION_EVAL"
  create_layout_from_panes "$WINDOW_MAIN" 0
}

create_simulator_layout() {
  reset_panes
  add_pane "$WINDOW_MAIN" "/workspaces" "$SETUP_SCRIPT" "$CMD_SIMULATOR"
  add_pane "$WINDOW_MAIN" "/workspaces" "$SETUP_SCRIPT" "$CMD_LOCALIZATION_TRIGGER"
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
    "$MODE_TAMIYA"|"$MODE_PYTHON"|"$MODE_MAP"|"$MODE_SIMULATOR")
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
  "$MODE_TAMIYA"|"$MODE_PYTHON"|"$MODE_MAP"|"$MODE_SIMULATOR")
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    echo "Use --mode tamiya, --mode python, --mode map, or --mode simulator" >&2
    exit 1
    ;;
esac

if [[ -z "$SESSION_NAME" ]]; then
  if [[ "$MODE" == "$MODE_TAMIYA" ]]; then
    SESSION_NAME="$DEFAULT_SESSION_TAMIYA"
  elif [[ "$MODE" == "$MODE_PYTHON" ]]; then
    SESSION_NAME="$DEFAULT_SESSION_PYTHON"
  elif [[ "$MODE" == "$MODE_MAP" ]]; then
    SESSION_NAME="$DEFAULT_SESSION_MAP"
  else
    SESSION_NAME="$DEFAULT_SESSION_SIMULATOR"
  fi
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [[ "$MODE" == "$MODE_TAMIYA" ]]; then
    create_tamiya_layout
  elif [[ "$MODE" == "$MODE_PYTHON" ]]; then
    create_python_layout
  elif [[ "$MODE" == "$MODE_MAP" ]]; then
    create_map_layout
  elif [[ "$MODE" == "$MODE_SIMULATOR" ]]; then
    create_simulator_layout
  fi
fi

tmux attach-session -t "$SESSION_NAME"
