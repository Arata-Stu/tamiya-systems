#!/bin/bash

echo "=== インタラクティブ Map 転送スクリプト (複数選択・複数IP対応) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
TUI_SCRIPT_PATH="${PROJECT_ROOT}/scripts/common/tui.sh"

if [ -f "$TUI_SCRIPT_PATH" ]; then
    # shellcheck source=scripts/common/tui.sh
    source "$TUI_SCRIPT_PATH"
fi

LEGACY_SELECT="false"
if [[ "${1:-}" == "--legacy-select" ]]; then
    LEGACY_SELECT="true"
    shift
fi

# ==========================================
# デフォルト設定
# ==========================================
DEFAULT_BASE_DIR="${PROJECT_ROOT}/map/"
DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspaces/tamiya-systems/map/"
LOCAL_LIST_MAX_DEPTH=${LOCAL_LIST_MAX_DEPTH:-2}
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
dirs=()
while IFS= read -r -d '' d; do
    dirs+=("$d")
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth "$LOCAL_LIST_MAX_DEPTH" -type d -print0 | sort -z)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "エラー: '$BASE_DIR' の中にディレクトリが見つかりません。"
    exit 1
fi

SELECTED_DIRS=()

if [[ "$LEGACY_SELECT" != "true" ]] && declare -F tui_select_paths >/dev/null 2>&1; then
    tui_select_paths "送信する Map ディレクトリを選択してください。" dirs SELECTED_DIRS "$BASE_DIR" || true
fi

if [ ${#SELECTED_DIRS[@]} -eq 0 ]; then
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

    for choice in $DIR_CHOICES; do
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#dirs[@]}" ]; then
            SELECTED_DIRS+=("${dirs[$((choice-1))]}")
        else
            echo "警告: 無効な入力 '$choice' はスキップされます。"
        fi
    done
fi

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

    if ! ssh "${REMOTE_USER}@${REMOTE_IP}" "mkdir -p '${REMOTE_DIR}'"; then
        echo "❌ 送信先ディレクトリの作成に失敗しました。"
        exit 1
    fi
    transfer_status=0
    for src_dir in "${SELECTED_DIRS[@]}"; do
        rel_path="${src_dir#$BASE_DIR}"
        echo ">>> Transferring: $rel_path"
        rsync -avzP -R "$BASE_DIR./$rel_path" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}" || transfer_status=1
    done
    if [ "$transfer_status" -ne 0 ]; then
        echo "❌ 転送中にエラーが発生しました。"
        exit 1
    fi
    echo "✅ 転送が正常に完了しました！"
else
    echo "転送をキャンセルしました。"
fi
