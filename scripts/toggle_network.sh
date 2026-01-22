#!/bin/bash

SCRIPT_NAME=$(basename "$0")

show_help() {
  echo "使用法: sudo $SCRIPT_NAME [インターフェース名]"
  echo "ネットワークインターフェースの状態を UP/DOWN で切り替えます。"
  echo
  echo "引数:"
  echo "  インターフェース名   対象のインターフェース名 (例: enp3s0, wlan0)"
  echo "                     省略した場合はデフォルト値 'wlP1p1s0' が使用されます。"
  echo
  echo "利用可能なインターフェース:"
  ip -br link
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  exit 0
fi

INTERFACE=${1:-"wlP1p1s0"}
CURRENT_STATE=$(ip link show "$INTERFACE" 2>/dev/null | grep -o "<[^>]*>" | grep -q "UP" && echo "UP" || echo "DOWN")

if [ -z "$CURRENT_STATE" ]; then
  echo "エラー: インターフェース '$INTERFACE' が見つかりません。"
  ip -br link
  exit 1
fi

echo "✅ 対象インターフェース: $INTERFACE (現在の状態: $CURRENT_STATE)"

if [ "$CURRENT_STATE" == "UP" ]; then
  echo "🔌 ---> '$INTERFACE' を DOWN にします..."
  sudo ip link set "$INTERFACE" down
else
  echo "⚡️ ---> '$INTERFACE' を UP にします..."
  sudo ip link set "$INTERFACE" up
fi

NEW_STATE=$(ip link show "$INTERFACE" | grep -o "<[^>]*>" | grep -q "UP" && echo "UP" || echo "DOWN")
echo "👍 完了しました。新しい状態: $NEW_STATE"