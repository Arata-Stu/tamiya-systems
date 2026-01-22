#!/bin/bash

# セッション名を指定
SESSION_NAME="screen"

if [ -n "$STY" ]; then
    # 1a. セッション内にいる場合
    echo "既にscreenセッション（$STY）内にいます。デタッチします。"
    screen -d

else
    if screen -ls | grep -q "\.${SESSION_NAME}\b"; then
        # 存在する場合
        echo "セッション '${SESSION_NAME}' にアタッチします。"
        screen -r ${SESSION_NAME}
    else
        # 存在しない場合
        echo "セッション '${SESSION_NAME}' を新規作成します。"
        screen -S ${SESSION_NAME}
    fi
fi