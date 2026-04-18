#!/bin/bash

set -euo pipefail

# --- 設定項目 ---
MODE_TAMIYA="tamiya"
MODE_PYTHON="python"

DEFAULT_SESSION_TAMIYA="tamiya"
DEFAULT_SESSION_PYTHON="python"

WINDOW_MAIN="main"
WINDOW_DATA="data"

SETUP_SCRIPT="source /workspaces/install/setup.bash"
CMD_BASE="ros2 launch system_launch system.launch.xml localization:=false vslam:=false use_camera:=true use_lidar:=true use_ftg:=false use_emergency:=false record:=false"
CMD_MONITOR="bash /scripts/monitor.sh"

PYTHON_PANE1_DIR="/python_ws"
PYTHON_PANE2_DIR="/workspaces/record/"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [SESSION_NAME] [--mode tamiya|python]
  $(basename "$0") [--session SESSION_NAME] [--mode tamiya|python]

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
    echo "Select mode:"
    echo "  1) $MODE_TAMIYA (current setup)"
    echo "  2) $MODE_PYTHON (study setup)"
    read -r -p "Enter 1 or 2: " answer

    case "$answer" in
      1|"$MODE_TAMIYA")
        echo "$MODE_TAMIYA"
        return
        ;;
      2|"$MODE_PYTHON")
        echo "$MODE_PYTHON"
        return
        ;;
      *)
        echo "Invalid choice: $answer"
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

  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".0 "cd /workspaces/record && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_DATA".1 "cd /scripts/ && clear" C-m

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_MAIN".0
}

create_python_layout() {
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_MAIN"
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_MAIN".0

  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".0 "cd $PYTHON_PANE1_DIR && clear" C-m
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_MAIN".1 "cd $PYTHON_PANE2_DIR && clear" C-m

  tmux select-window -t "$SESSION_NAME":"$WINDOW_MAIN"
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
    "$MODE_TAMIYA"|"$MODE_PYTHON")
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
  "$MODE_TAMIYA"|"$MODE_PYTHON")
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    echo "Use --mode tamiya or --mode python" >&2
    exit 1
    ;;
esac

if [[ -z "$SESSION_NAME" ]]; then
  if [[ "$MODE" == "$MODE_TAMIYA" ]]; then
    SESSION_NAME="$DEFAULT_SESSION_TAMIYA"
  else
    SESSION_NAME="$DEFAULT_SESSION_PYTHON"
  fi
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [[ "$MODE" == "$MODE_TAMIYA" ]]; then
    create_tamiya_layout
  else
    create_python_layout
  fi
fi

tmux attach-session -t "$SESSION_NAME"
