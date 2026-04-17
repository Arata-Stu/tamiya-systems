#!/bin/bash

echo "=== インタラクティブ SCP 転送スクリプト (複数選択・2階層厳密・複数IP対応) ==="

# ==========================================
# デフォルト設定
# ==========================================
DEFAULT_BASE_DIR="./ckpts/train/"
DEFAULT_REMOTE_USER="tamiya"
# ★ 複数IPのリストを定義
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspace/tamiya-systems/python_ws/ckpts/tinylidarnet/"
# ==========================================

# 1. ベースディレクトリの指定
read -p "ベースディレクトリを入力 (Enterで '${DEFAULT_BASE_DIR}'): " BASE_DIR
BASE_DIR=${BASE_DIR:-$DEFAULT_BASE_DIR}

# 相対パス表示を綺麗にするため、末尾の / を確実につける
BASE_DIR="${BASE_DIR%/}/"

if [ ! -d "$BASE_DIR" ]; then
    echo "エラー: ディレクトリ '$BASE_DIR' が見つかりません。"
    exit 1
fi

# 2. ベースディレクトリ内のディレクトリを配列に取得
# -mindepth 2 -maxdepth 2 で「必ず2階層目」のみを取得し、名前順にソート
IFS=$'\n' read -r -d '' -a dirs < <(find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "エラー: '$BASE_DIR' の中に2階層目のディレクトリが見つかりません。"
    exit 1
fi

# 3. 送信ディレクトリの選択 (複数選択対応)
echo ""
echo "送信するディレクトリを以下の番号から選択してください:"
i=1
for d in "${dirs[@]}"; do
    # BASE_DIR部分を削り取り、「親/子 (例: date/time)」の相対パスにする
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

# 入力された番号を解析して送信対象リストを作成
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
echo ""
read -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

# ★ IPアドレスの選択ロジック
echo ""
echo "相手のIPアドレスを選択、または直接入力してください:"
i=1
for ip in "${DEFAULT_REMOTE_IPS[@]}"; do
    if [ $i -eq 1 ]; then
        echo "  $i) $ip (Enterのデフォルト)"
    else
        echo "  $i) $ip"
    fi
    ((i++))
done
echo ""

read -p "番号、またはIPを直接入力 (Enterで '${DEFAULT_REMOTE_IPS[0]}'): " IP_CHOICE

if [ -z "$IP_CHOICE" ]; then
    # Enterのみの場合は1つ目のIPを使用
    REMOTE_IP="${DEFAULT_REMOTE_IPS[0]}"
elif [[ "$IP_CHOICE" =~ ^[0-9]+$ ]] && [ "$IP_CHOICE" -ge 1 ] && [ "$IP_CHOICE" -le "${#DEFAULT_REMOTE_IPS[@]}" ]; then
    # 番号が入力された場合は配列から取得
    REMOTE_IP="${DEFAULT_REMOTE_IPS[$((IP_CHOICE-1))]}"
else
    # 番号以外の文字列が入力された場合はそれを直接IPとして扱う
    REMOTE_IP="$IP_CHOICE"
fi

echo ""
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

# 実行確認
read -p "この内容で転送を開始しますか？ (Y/n, Enterで実行): " CONFIRM
CONFIRM=${CONFIRM:-y} 

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "転送を開始します..."
    
    # scpコマンドの実行 (余計な事前確認なしで直接送信)
    scp -r "${SELECTED_DIRS[@]}" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "✅ 転送が正常に完了しました！"
    else
        echo "❌ 転送中にエラーが発生しました。"
    fi
else
    echo "転送をキャンセルしました。"
fi