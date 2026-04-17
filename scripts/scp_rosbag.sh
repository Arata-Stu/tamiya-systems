#!/bin/bash

echo "=== インタラクティブ SCP 受信(ダウンロード)スクリプト ==="

# ==========================================
# デフォルト設定
# ==========================================
DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IP="192.168.55.1"
DEFAULT_REMOTE_BASE_DIR="/home/tamiya/workspace/tamiya-systems/ros2_ws/record/"
DEFAULT_LOCAL_DEST_DIR="/home/arata-22/workspace/ros2_ws/record/"
# ==========================================

# 1. リモート情報の入力 (Enterでデフォルト値適用)
read -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

read -p "相手のIPアドレス (Enterで '${DEFAULT_REMOTE_IP}'): " REMOTE_IP
REMOTE_IP=${REMOTE_IP:-$DEFAULT_REMOTE_IP}

read -p "取得元のベースディレクトリ (Enterで '${DEFAULT_REMOTE_BASE_DIR}'): " REMOTE_BASE_DIR
REMOTE_BASE_DIR=${REMOTE_BASE_DIR:-$DEFAULT_REMOTE_BASE_DIR}

# 2. リモートサーバーからディレクトリ一覧を取得
echo ""
echo "リモートサーバー (${REMOTE_IP}) からディレクトリ一覧を取得中..."

# ★修正ポイント: whileループを使って、複数件の結果を最後まで確実に配列へ格納する
# ssh に -n を付けて、標準入力を奪わないように安全対策
dirs=()
while IFS= read -r -d '' d; do
    dirs+=("$d")
done < <(ssh -n "${REMOTE_USER}@${REMOTE_IP}" "find \"$REMOTE_BASE_DIR\" -maxdepth 1 -mindepth 1 -type d -print0" 2>/dev/null)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "エラー: リモートの '$REMOTE_BASE_DIR' にディレクトリが見つからないか、SSH接続に失敗しました。"
    exit 1
fi

# 3. ダウンロードするディレクトリの選択 (複数選択対応)
echo ""
echo "ダウンロードするディレクトリを以下の番号から選択してください:"
i=1
for d in "${dirs[@]}"; do
    echo "  $i) $(basename "$d")"
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

# 4. ローカルの保存先の指定
echo ""
read -p "ローカルの保存先ディレクトリを入力 (Enterで '${DEFAULT_LOCAL_DEST_DIR}'): " LOCAL_DEST_DIR
LOCAL_DEST_DIR=${LOCAL_DEST_DIR:-$DEFAULT_LOCAL_DEST_DIR}

# 保存先が存在しない場合は自動で作成
if [ ! -d "$LOCAL_DEST_DIR" ]; then
    mkdir -p "$LOCAL_DEST_DIR"
    echo "-> 保存先ディレクトリ '$LOCAL_DEST_DIR' を作成しました。"
fi

# 5. 最終確認
echo ""
echo "================ 転送内容の確認 ================"
echo "取得元サーバー: ${REMOTE_USER}@${REMOTE_IP}"
echo "ダウンロード対象 (${#SELECTED_DIRS[@]}件):"
for d in "${SELECTED_DIRS[@]}"; do
    echo "  - $(basename "$d")"
done
echo "ローカル保存先: $LOCAL_DEST_DIR"
echo "================================================"

read -p "この内容でダウンロードを開始しますか？ (Y/n, Enterで実行): " CONFIRM
CONFIRM=${CONFIRM:-y} 

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "ダウンロードを開始します..."
    
    # scpの引数用配列を作成
    SCP_TARGETS=()
    for d in "${SELECTED_DIRS[@]}"; do
        escaped_dir=$(echo "$d" | sed 's/ /\\ /g')
        SCP_TARGETS+=("${REMOTE_USER}@${REMOTE_IP}:$escaped_dir")
    done
    
    # scpコマンドの実行
    scp -r "${SCP_TARGETS[@]}" "$LOCAL_DEST_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ ダウンロードが正常に完了しました！"
    else
        echo "❌ ダウンロード中にエラーが発生しました。"
    fi
else
    echo "キャンセルしました。"
fi