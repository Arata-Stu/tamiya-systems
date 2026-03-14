#!/bin/bash

# --- 設定項目 ---
DEFAULT_SESSION_NAME="tamiya"                       # デフォルトのセッション名
WINDOW_NAME="main"                                  # ウィンドウ名
ROS_WS_PATH="${ISAAC_ROS_WS}"                       # ROS 2ワークスペースのパス（必要なら利用）
SETUP_SCRIPT="source /workspaces/install/setup.bash" # setup.bashへのフルパスを指定

# --- 実行するコマンド群 ---
CMD_BASE="ros2 launch system_launch system.launch.xml record:=false vslam:=false use_camera:=false use_lidar:=false"
CMD_MONITOR="ros2 launch system_launch monitor.launch.xml"
CMD_JTOP="jtop"
CMD_BAG="ros2 launch bag_manager_py bag_manager_node.launch.xml"

# --- セッション名の決定 ---
if [ -n "$1" ]; then
  SESSION_NAME="$1"
else
  SESSION_NAME="$DEFAULT_SESSION_NAME"
fi

# --- tmuxセッションの準備 ---
# セッションが存在するかチェック（exit code 0 -> 存在する）
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -ne 0 ]; then
  # セッションが無ければ新規作成して最初のウィンドウを作成
  tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME"

  # ==========================================
  # ウィンドウ1: "main" (ROSコマンド用・4分割)
  # ==========================================
  # 1. 縦に分割（上: pane 0, 下: pane 1）
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_NAME".0

  # 2. 上ペインを横に分割（左上: pane 0, 右上: pane 2）
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_NAME".0

  # 3. 下ペインを横に分割（左下: pane 1, 右下: pane 3）
  tmux split-window -h -t "$SESSION_NAME":"$WINDOW_NAME".1

  # 各ペインで初期化コマンドを実行（環境変数設定・setup読み込み・クリア）
  for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".$pane \
      "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT && clear" C-m
  done

  # 各ペインへ個別コマンドを送信 (Enter待ち状態)
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".0 "$CMD_BASE"         # 左上
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".2 "$CMD_BAG"          # 右上
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".1 "$CMD_MONITOR"      # 左下
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".3 "$CMD_JTOP"         # 右下

  # ==========================================
  # ウィンドウ2: "data" (データ確認用・2分割)
  # ==========================================
  # 新しいウィンドウを作成
  tmux new-window -t "$SESSION_NAME" -n "data"

  # 画面を縦に2分割（垂直線で左右に分割。上下にしたい場合は -v に変更）
  tmux split-window -h -t "$SESSION_NAME":"data"

  # 画面1 (pane 0) で cd /record/ を実行 (即時実行のため C-m を付与)
  tmux send-keys -t "$SESSION_NAME":"data".0 "cd /workspaces/record && clear" C-m

  # 画面2 (pane 1) で cd /scripts/ を実行 (即時実行のため C-m を付与)
  tmux send-keys -t "$SESSION_NAME":"data".1 "cd /scripts/ && clear" C-m

  # ==========================================
  # 仕上げ: 初期表示ウィンドウの設定
  # ==========================================
  # 最終的に main ウィンドウの左上ペインをアクティブにしておく
  tmux select-window -t "$SESSION_NAME":"$WINDOW_NAME"
  tmux select-pane -t "$SESSION_NAME":"$WINDOW_NAME".0
fi

# --- セッションへアタッチ（既に存在していても接続） ---
tmux attach-session -t "$SESSION_NAME"