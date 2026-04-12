#!/bin/bash

# --- 設定項目 ---
DEFAULT_SESSION_NAME="tamiya"                       # デフォルトのセッション名
WINDOW_NAME="main"                                  # ウィンドウ名
ROS_WS_PATH="${ISAAC_ROS_WS}"                       # ROS 2ワークスペースのパス（必要なら利用）
SETUP_SCRIPT="source /workspaces/install/setup.bash" # setup.bashへのフルパスを指定

# --- 実行するコマンド群 ---
CMD_BASE="ros2 launch system_launch system.launch.xml vslam:=true use_camera:=true use_lidar:=true use_ftg:=true localization:=false record:=false"  # 左ペインで実行するROS 2起動コマンド
CMD_MONITOR="bash /scripts/monitor.sh"

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
  # ウィンドウ1: "main" (ROSコマンド用・2分割)
  # ==========================================
  # 画面を左右に2分割
  tmux split-window -v -t "$SESSION_NAME":"$WINDOW_NAME".0

  # 各ペインで初期化コマンドを実行（環境変数設定・setup読み込み・クリア）
  for pane in 0 1; do
    tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".$pane \
      "export ROS_LOCALHOST_ONLY=0 && $SETUP_SCRIPT && clear" C-m
  done

  # 各ペインへ個別コマンドを送信 (Enter待ち状態)
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".0 "$CMD_BASE"         # 左
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME".1 "$CMD_MONITOR"      # 右

  # ==========================================
  # ウィンドウ2: "data" (データ確認用・2分割)
  # ==========================================
  # 新しいウィンドウを作成
  tmux new-window -t "$SESSION_NAME" -n "data"

  # 画面を縦に2分割（垂直線で左右に分割。上下にしたい場合は -v に変更）
  tmux split-window -v -t "$SESSION_NAME":"data"

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
