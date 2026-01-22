#!/bin/bash

# ======== 登録済みMACアドレス ========
DEFAULT_CONTROLLER_MAC="A0:AB:51:5F:62:86"   # PS4 DualShock 4
SECOND_CONTROLLER_MAC="4C:B9:9B:E0:EF:24"    # PS5 DualSense

echo "🎮 DualShock/DualSense 自動ペアリングスクリプト"
echo "========================================"
echo
echo "接続したいコントローラーを選択してください:"
echo "  [1] PS4 DualShock 4 ($DEFAULT_CONTROLLER_MAC)"
echo "  [2] PS5 DualSense  ($SECOND_CONTROLLER_MAC)"
echo "  [3] 手動入力"
echo

read -p "番号を入力してください (1-3): " choice
echo

case "$choice" in
    1)
        CONTROLLER_MAC="$DEFAULT_CONTROLLER_MAC"
        CONTROLLER_NAME="PS4 DualShock 4"
        ;;
    2)
        CONTROLLER_MAC="$SECOND_CONTROLLER_MAC"
        CONTROLLER_NAME="PS5 DualSense"
        ;;
    3)
        read -p "接続したいコントローラーのMACアドレスを入力してください: " CONTROLLER_MAC
        CONTROLLER_NAME="カスタムコントローラー"
        ;;
    *)
        echo "⚠️ 無効な選択です。スクリプトを終了します。"
        exit 1
        ;;
esac

echo "🎯 選択: $CONTROLLER_NAME ($CONTROLLER_MAC)"
echo

# --- スクリプト本体 ---
echo "========================================"
echo "コントローラーのPSボタンとSHAREボタンを長押しして、"
echo "ライトバーが【白く点滅】するペアリングモードにしてください。"
echo
read -p "準備ができたら Enterキー を押してください..."

echo
echo "🔗 ペアリングを開始します..."
echo "========================================"
echo

{
    echo -e "remove $CONTROLLER_MAC\n"
    sleep 2
    echo -e "scan on\n"
    sleep 5
    echo -e "scan off\n"
    sleep 1
    echo -e "pair $CONTROLLER_MAC\n"
    sleep 3
    echo -e "trust $CONTROLLER_MAC\n"
    sleep 2
    echo -e "connect $CONTROLLER_MAC\n"
    sleep 4
} | sudo bluetoothctl

echo
echo "========================================"
echo "✅ 処理が完了しました。"
echo "コントローラーのライトバーが【青色に点灯】していれば成功です。"
echo
echo "失敗した場合は、再度ペアリングモードにしてからもう一度試してください。"
echo "========================================"
