#!/bin/bash

echo "=== インタラクティブ rsync 受信スクリプト ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TUI_SCRIPT_PATH="${SCRIPT_DIR}/common/tui.sh"

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
DEFAULT_REMOTE_USER="tamiya"
IP_CANDIDATES=("10.42.0.1" "192.168.55.1" "192.168.11.98")
REMOTE_LIST_MAX_DEPTH=${REMOTE_LIST_MAX_DEPTH:-4}

# 種別ごとのパス設定
ROSBAG_REMOTE_DIR="/home/tamiya/workspaces/tamiya-systems/record/"
ROSBAG_LOCAL_DIR="/home/arata-22/workspaces/tamiya-systems/record/"

MAP_REMOTE_DIR="/home/tamiya/workspaces/tamiya-systems/map/"
MAP_LOCAL_DIR="/home/arata-22/workspaces/tamiya-systems/map/"
# ==========================================

# 0. 転送対象選択
echo ""
echo "何を取得しますか？"
echo "  1) rosbag"
echo "  2) map"
read -p "選択 (1 or 2): " TARGET_CHOICE

case "$TARGET_CHOICE" in
    2)
        DEFAULT_REMOTE_BASE_DIR="$MAP_REMOTE_DIR"
        DEFAULT_LOCAL_DEST_DIR="$MAP_LOCAL_DIR"
        TARGET_NAME="map"
        ;;
    *)
        DEFAULT_REMOTE_BASE_DIR="$ROSBAG_REMOTE_DIR"
        DEFAULT_LOCAL_DEST_DIR="$ROSBAG_LOCAL_DIR"
        TARGET_NAME="rosbag"
        ;;
esac

# 1. 接続情報の入力
read -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

echo ""
echo "接続先IPアドレスを選択してください:"
for i in "${!IP_CANDIDATES[@]}"; do
    echo "  $((i+1))) ${IP_CANDIDATES[$i]}"
done
echo "  n) 新しいIPを手動入力する"
read -p "番号を選択 (1-${#IP_CANDIDATES[@]} / n): " IP_CHOICE

case "$IP_CHOICE" in
    [1-9]) REMOTE_IP=${IP_CANDIDATES[$((IP_CHOICE-1))]} ;;
    n|N)   read -p "IPアドレスを入力: " REMOTE_IP ;;
    *)     REMOTE_IP=${IP_CANDIDATES[0]}; echo "-> ${REMOTE_IP} を使用します。" ;;
esac

read -p "リモートのベースディレクトリ (Enterで '${DEFAULT_REMOTE_BASE_DIR}'): " REMOTE_BASE_DIR
REMOTE_BASE_DIR=${REMOTE_BASE_DIR:-$DEFAULT_REMOTE_BASE_DIR}

# 2. 取得モードの選択
echo ""
echo "ディレクトリの指定方法を選んでください:"
echo "  1) リモートから一覧を取得して選択"
echo "  2) ディレクトリ名を直接入力する"
read -p "選択 (1 or 2): " MODE_CHOICE

SELECTED_DIRS=()

remote_relative_path() {
    local path="$1"
    local base="${REMOTE_BASE_DIR%/}"

    if [[ "$path" == "$base"/* ]]; then
        printf '%s\n' "${path#"$base"/}"
    else
        printf '%s\n' "$path"
    fi
}

is_under_remote_base() {
    local path="$1"
    local base="${REMOTE_BASE_DIR%/}"

    [[ "$path" == "$base"/* ]]
}

remote_rsync_source() {
    local path="$1"
    local base="${REMOTE_BASE_DIR%/}"
    local rel

    if [[ "$path" == "$base"/* ]]; then
        rel="${path#"$base"/}"
        printf '%s@%s:%s/./%s\n' "$REMOTE_USER" "$REMOTE_IP" "$base" "$rel"
    else
        printf '%s@%s:%s\n' "$REMOTE_USER" "$REMOTE_IP" "$path"
    fi
}

if [ "$MODE_CHOICE" = "1" ]; then
    echo "リモートサーバー (${REMOTE_IP}) からディレクトリ一覧を取得中..."

    dirs=()
    while IFS= read -r -d '' d; do
        dirs+=("$d")
    done < <(
        ssh -n -o ConnectTimeout=5 "${REMOTE_USER}@${REMOTE_IP}" \
        "find \"$REMOTE_BASE_DIR\" -mindepth 1 -maxdepth \"$REMOTE_LIST_MAX_DEPTH\" -type d -print0 | sort -z" \
        2>/dev/null
    )

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "エラー: ディレクトリが見つからないか、接続に失敗しました。"
        exit 1
    fi

    if [[ "$LEGACY_SELECT" != "true" ]] && declare -F tui_select_paths >/dev/null 2>&1; then
        tui_select_paths "取得対象を選択してください。" dirs SELECTED_DIRS "$REMOTE_BASE_DIR" || true
    fi

    if [ ${#SELECTED_DIRS[@]} -eq 0 ]; then
        echo ""
        echo "取得対象を選択 (例: 1 3)"
        for i in "${!dirs[@]}"; do
            printf "  %2d) %s\n" "$((i+1))" "$(remote_relative_path "${dirs[$i]}")"
        done

        read -p "番号を入力: " DIR_CHOICES

        for choice in $DIR_CHOICES; do
            if [[ "$choice" =~ ^[0-9]+$ ]] && \
               [ "$choice" -ge 1 ] && \
               [ "$choice" -le "${#dirs[@]}" ]; then
                SELECTED_DIRS+=("${dirs[$((choice-1))]}")
            fi
        done
    fi

else
    echo ""
    echo "ベースディレクトリ: $REMOTE_BASE_DIR"
    read -p "転送したいディレクトリ名を入力してください (スペース区切り可): " MANUAL_INPUT

    for name in $MANUAL_INPUT; do
        if [[ "$name" = /* ]]; then
            SELECTED_DIRS+=("$name")
        else
            SELECTED_DIRS+=("${REMOTE_BASE_DIR%/}/$name")
        fi
    done
fi

if [ ${#SELECTED_DIRS[@]} -eq 0 ]; then
    echo "エラー: 対象が選択されていません。"
    exit 1
fi

# 3. 保存先と実行
echo ""
read -p "ローカル保存先 (Enterで '${DEFAULT_LOCAL_DEST_DIR}'): " LOCAL_DEST_DIR
LOCAL_DEST_DIR=${LOCAL_DEST_DIR:-$DEFAULT_LOCAL_DEST_DIR}

mkdir -p "$LOCAL_DEST_DIR"

echo ""
echo "================ 転送内容 ================"
echo "対象    : $TARGET_NAME"
echo "リモート: ${REMOTE_USER}@${REMOTE_IP}"
for d in "${SELECTED_DIRS[@]}"; do
    echo "  - $d"
done
echo "保存先  : $LOCAL_DEST_DIR"
echo "=========================================="

read -p "rsyncを開始しますか？ (Y/n): " CONFIRM

if [[ "${CONFIRM:-y}" =~ ^[Yy]$ ]]; then
    for target_dir in "${SELECTED_DIRS[@]}"; do
        echo ">>> Transferring: $(remote_relative_path "$target_dir")"
        if is_under_remote_base "$target_dir"; then
            rsync -avP -R "$(remote_rsync_source "$target_dir")" "$LOCAL_DEST_DIR/"
        else
            rsync -avP "$(remote_rsync_source "$target_dir")" "$LOCAL_DEST_DIR/"
        fi
    done
    echo "✅ 完了しました。"
else
    echo "キャンセルしました。"
fi
