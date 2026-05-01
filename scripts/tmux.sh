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
CMD_BASE="ros2 launch system_launch system.launch.xml use_section_localizer:=false localization:=false vslam:=true use_camera:=true use_lidar:=true use_ftg:=false use_emergency:=false record:=false"
CMD_MONITOR="bash /scripts/monitor.sh"

PYTHON_PANE1_DIR="/python_ws"
PYTHON_PANE2_DIR="/record/"

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
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_MAIN"

  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  for pane in 0 1; do
    tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".$pane \
      "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT && clear" C-m
  done

  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "$CMD_BASE"
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 "$CMD_MONITOR"

  tmux new-window -t "$SESSION_NAME" -n "$WINDOW_DATA"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_DATA".0

  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".0 "cd /record && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".1 "cd /scripts/ && clear" C-m

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_python_layout() {
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "cd $PYTHON_PANE1_DIR && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 "cd $PYTHON_PANE2_DIR && $SETUP_SCRIPT && clear" C-m

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_map_layout() {
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_MAIN"

  # 2ペインに分割（map作成をシンプルに）
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  # --- 1ペイン目: create_2d_map コマンド ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "bash /scripts/create_2d_map_from_bag.sh --rate 1.0 --use-vslam-odom /record/  <map_name>"

  # --- 2ペイン目: /map に移動 ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 "cd /map && clear" C-m

  # 最初は1ペイン目にフォーカス
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0

  tmux new-window -t "$SESSION_NAME" -n "$WINDOW_LOCALIZATION_EVAL"

  # 2x2 の均等グリッドを作る
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1
  tmux select-layout -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL" tiled

  # --- 1ペイン目 (上): rosbag play ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0 "ros2 bag play <bag_path> --clock --start-paused"

  # --- 2ペイン目 (右上): localization launch ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1 "ros2 launch system_launch localization.launch.xml lidar_container_name:=lidar_container map_yaml_path:=<yaml> scan_topic:=/scan flatscan_topic:=/flatscan use_localization_manager:=false publish_localization_tf:=false"

  # --- 3ペイン目 (左下): evaluate script ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".2 "cd /python_ws/data_analysis/ && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".2 "python3 evaluate_global_localization_sweep.py --map-yaml <yaml>"

  # --- 4ペイン目 (右下): component container ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".3 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".3 "ros2 run rclcpp_components component_container --ros-args -r __node:=lidar_container"

  # map作成導線を優先して、最初は main ウィンドウを開く
  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_simulator_layout() {
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  
  # 上下2つにペインを分割
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  # --- 1ペイン目 (上) ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  # コマンドを準備（末尾に C-m を付けない）
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "ros2 launch system_launch simulator.launch.xml use_ftg:=false record:=false rviz:=false localization:=false"

  # --- 2ペイン目 (下) ---
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 "cd /workspaces && $SETUP_SCRIPT && clear" C-m
  # コマンドを準備（末尾に C-m を付けない）※ダブルクォーテーションを保持するために全体をシングルクォーテーションで囲む
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 'ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'

  # 最初は1ペイン目にフォーカスを合わせておく
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
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
