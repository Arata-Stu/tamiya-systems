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
CMD_BASE="bash /scripts/launch_system.sh base"
CMD_MONITOR="bash /scripts/monitor.sh"
CMD_LOCALIZATION_TRIGGER='ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'
RVIZ_LOCALIZATION_EVAL='rviz2 -d $(ros2 pkg prefix system_launch)/share/system_launch/rviz/localization_eval.rviz'

PYTHON_PANE1_DIR="/python_ws"
PYTHON_PANE2_DIR="/record/"

# --- ヘルパー関数 ---
# 1. ペインの初期化（コマンドを実行してEnter）
init_pane() {
  local target="$1"
  local cmd="$2"
  tmux send-keys -t "$target" "$cmd" C-m
}

# 2. ペインに準備コマンドを流し込む（表示崩れ対策済み）
prepare_cmd() {
  local target="$1"
  local cmd="$2"
  
  # bashのショートカット(Ctrl+L)で画面をクリア＆プロンプトを綺麗に再描画
  tmux send-keys -t "$target" C-l
  sleep 0.2
  
  # コマンドを文字入力（Enterは押さない）
  tmux send-keys -t "$target" "$cmd"
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
  # ターミナル幅が狭いと判断されないよう、初期仮想サイズを大きく設定 (-x 250 -y 80)
  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  init_pane "$SESSION_NAME":"$WINDOW_MAIN".0 "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_MAIN".1 "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT"

  tmux new-window -t "$SESSION_NAME" -n "$WINDOW_DATA"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_DATA".0

  init_pane "$SESSION_NAME":"$WINDOW_DATA".0 "cd /record"
  init_pane "$SESSION_NAME":"$WINDOW_DATA".1 "cd /scripts/"

  sleep 2.0 # sourceコマンドの完了を待機

  prepare_cmd "$SESSION_NAME":"$WINDOW_MAIN".0 "$CMD_BASE"
  prepare_cmd "$SESSION_NAME":"$WINDOW_MAIN".1 "$CMD_MONITOR"
  
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".0 C-l
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".1 C-l

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_python_layout() {
  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  init_pane "$SESSION_NAME":"$WINDOW_MAIN".0 "cd $PYTHON_PANE1_DIR && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_MAIN".1 "cd $PYTHON_PANE2_DIR && $SETUP_SCRIPT"

  sleep 2.0 # sourceコマンドの完了を待機
  
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 C-l
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 C-l

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_map_layout() {
  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  tmux new-window -t "$SESSION_NAME" -n "$WINDOW_LOCALIZATION_EVAL"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".2
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1

  # 全てのペインで初期化（source等）を先に流す
  init_pane "$SESSION_NAME":"$WINDOW_MAIN".0 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_MAIN".1 "cd /map"
  
  init_pane "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".2 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".3 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".4 "cd /workspaces && $SETUP_SCRIPT"

  # 全ペインの初期化が終わるのを一括で待機
  sleep 2.0

  # main ウィンドウのコマンド準備
  prepare_cmd "$SESSION_NAME":"$WINDOW_MAIN".0 "bash /scripts/create_2d_map_from_bag.sh --rate 1.0 --use-vslam-odom"
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 C-l

  # localization_eval ウィンドウのコマンド準備
  prepare_cmd "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".0 "ros2 bag play <bag_path> --clock --start-paused"
  prepare_cmd "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".1 "ros2 launch system_launch localization.launch.xml lidar_container_name:=lidar_container map_yaml_path:=<yaml> scan_topic:=/scan flatscan_topic:=/flatscan use_localization_manager:=true publish_localization_tf:=true"
  prepare_cmd "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".2 "$CMD_LOCALIZATION_TRIGGER"
  prepare_cmd "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".3 "ros2 run rclcpp_components component_container --ros-args -r __node:=lidar_container"
  prepare_cmd "$SESSION_NAME":"$WINDOW_LOCALIZATION_EVAL".4 "$RVIZ_LOCALIZATION_EVAL"

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_simulator_layout() {
  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  init_pane "$SESSION_NAME":"$WINDOW_MAIN".0 "cd /workspaces && $SETUP_SCRIPT"
  init_pane "$SESSION_NAME":"$WINDOW_MAIN".1 "cd /workspaces && $SETUP_SCRIPT"

  sleep 2.0

  prepare_cmd "$SESSION_NAME":"$WINDOW_MAIN".0 "ros2 launch system_launch simulator.launch.xml use_ftg:=false record:=false rviz:=false localization:=false"
  prepare_cmd "$SESSION_NAME":"$WINDOW_MAIN".1 'ros2 topic pub --once /localization/trigger std_msgs/msg/Bool "{data: true}"'

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
