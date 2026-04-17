#!/bin/bash

echo "=== インタラクティブ SCP 転送スクリプト (複数選択・2階層厳密対応) ==="

# ==========================================
# デフォルト設定
# ==========================================
DEFAULT_BASE_DIR="./ckpts/train/"
DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IP="192.168.55.1"
DEFAULT_REMOTE_DIR="/home/tamiya/workspace/tamiya-systems/python_ws/ckpts/pilotnet/"
# ==========================================

# 1. ベースディレクトリの指定
read -p "ベースディレクトリを入力 (Enterで '${DEFAULT_BASE_DIR}'): " BASE_DIR
BASE_DIR=${BASE_DIR:-$DEFAULT_BASE_DIR}

BASE_DIR="${BASE_DIR%/}/"

if [ ! -d "$BASE_DIR" ]; then
    echo "エラー: ディレクトリ '$BASE_DIR' が見つかりません。"
    exit 1
fi

# 2. ベースディレクトリ内のディレクトリを配列に取得
IFS=$'\n' read -r -d '' -a dirs < <(find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "エラー: '$BASE_DIR' の中に2階層目のディレクトリが見つかりません。"
    exit 1
fi

# 3. 送信ディレクトリの選択
echo ""
echo "送信するディレクトリを以下の番号から選択してください:"
i=1
for d in "${dirs[@]}"; do
    rel_path="${d#$BASE_DIR}"
    echo "  $i) $rel_path"
    ((i++))
done
echo ""

read -p "番号をスペース区切りで入力してください (例: 1 3 4): " DIR_CHOICES

if [ -z "$DIR_CHOICES" ]; then
    echo "エラー: ディレクトリが選択されませんでした。"
    exit 1
fi

SELECTED_DIRS=()
for choice in $DIR_CHOICES; do
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#dirs[@]}" ]; then
        SELECTED_DIRS+=("${dirs[$((choice-1))]}")
    else
        echo "警告: 無効な入力 '$choice' はスキップされます。"
    fi
done

if [ ${#SELECTED_DIRS[@]} -eq 0 ]; then
    echo "エラー: 有効なディレクトリが1つも選択されませんでした。"
    exit 1
fi

# 4. リモート情報の入力
read -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

read -p "相手のIPアドレス (Enterで '${DEFAULT_REMOTE_IP}'): " REMOTE_IP
REMOTE_IP=${REMOTE_IP:-$DEFAULT_REMOTE_IP}

read -p "送信先のディレクトリパス (Enterで '${DEFAULT_REMOTE_DIR}'): " REMOTE_DIR
REMOTE_DIR=${REMOTE_DIR:-$DEFAULT_REMOTE_DIR}

# 5. 最終確認
echo ""
echo "================ 転送内容の確認 ================"
echo "送信元ディレクトリ (${#SELECTED_DIRS[@]}件):"
for d in "${SELECTED_DIRS[@]}"; do
    echo "  - ${d#$BASE_DIR}"
done
echo "送信先宛先        : ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"
echo "================================================"

read -p "この内容で転送を開始しますか？ (Y/n, Enterで実行): " CONFIRM
CONFIRM=${CONFIRM:-y} 

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "転送を開始します..."
    
    # ★ 追加：リモート側にディレクトリが存在しない場合の対策
    # 事前に mkdir -p を実行し、ディレクトリを確実に存在させる
    echo "リモートの送信先ディレクトリを確認・作成しています..."
    ssh "${REMOTE_USER}@${REMOTE_IP}" "mkdir -p \"${REMOTE_DIR}\""
    
    # scpコマンドの実行
    scp -r "${SELECTED_DIRS[@]}" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "✅ 転送が正常に完了しました！"
    else
        echo "❌ 転送中にエラーが発生しました。"
    fi
else
    echo "転送をキャンセルしました。"
fi