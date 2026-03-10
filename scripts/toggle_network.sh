#!/bin/bash

SCRIPT_NAME=$(basename "$0")

# 利用可能なインターフェースを配列で取得（lo以外）
INTERFACES=($(ip -br link | awk '$1 != "lo" {print $1}'))

show_help() {
  echo "使用法: sudo $SCRIPT_NAME [インターフェース名]"
  echo "引数がない場合は、対話メニューが表示されます。"
  echo
  echo "引数:"
  echo "  インターフェース名   対象のインターフェース名 (例: enp3s0, wlan0)"
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  exit 0
fi

# --- インタラクティブ・ロジック ---
if [ -z "$1" ]; then
  echo "🌐 インターフェースを選択してください:"
  
  # インデックス付きで一覧表示
  for i in "${!INTERFACES[@]}"; do
    STATE=$(ip -br link show "${INTERFACES[$i]}" | awk '{print $2}')
    printf "  %d) %-10s [%s]\n" "$((i+1))" "${INTERFACES[$i]}" "$STATE"
  done

  read -p "番号を入力 (1-${#INTERFACES[@]}): " CHOICE
  
  # 入力値のバリデーション
  if [[ ! "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "${#INTERFACES[@]}" ]; then
    echo "❌ 無効な選択です。"
    exit 1
  fi
  
  INTERFACE="${INTERFACES[$((CHOICE-1))]}"
else
  INTERFACE=$1
fi
# --------------------------------

# 現在の状態確認
CURRENT_STATE=$(ip -br link show "$INTERFACE" 2>/dev/null | awk '{print $2}')

if [ -z "$CURRENT_STATE" ]; then
  echo "エラー: インターフェース '$INTERFACE' が見つかりません。"
  exit 1
fi

echo "✅ 対象: $INTERFACE (現在の状態: $CURRENT_STATE)"

# 切り替え実行
if [[ "$CURRENT_STATE" == "UP" || "$CURRENT_STATE" == "UNKNOWN" ]]; then
  echo "🔌 ---> '$INTERFACE' を DOWN にします..."
  sudo ip link set "$INTERFACE" down
else
  echo "⚡️ ---> '$INTERFACE' を UP にします..."
  sudo ip link set "$INTERFACE" up
fi

# 結果表示
NEW_STATE=$(ip -br link show "$INTERFACE" | awk '{print $2}')
echo "👍 完了しました。新しい状態: $NEW_STATE"