#!/bin/bash

# セッション名（引数があれば採用、なければ 'default'）
SESSION_NAME="${1:-default}"

if [ -n "$STY" ]; then
    echo "現在 screen セッション ($STY) 内にいます。デタッチします..."
    screen -d
else
    # セッションが存在するかチェック（状態問わず）
    if screen -ls | grep -q "\.${SESSION_NAME}[[:space:]]"; then
        echo "セッション '${SESSION_NAME}' に再接続します（必要に応じて他をデタッチ）。"
        screen -rd "${SESSION_NAME}"
    else
        echo "セッション '${SESSION_NAME}' を新規作成します。"
        screen -S "${SESSION_NAME}"
    fi
fi