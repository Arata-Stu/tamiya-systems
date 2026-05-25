#!/bin/bash

# --- スクリプト設定 ---
PREPROCESS_SCRIPT_NAME="extract_topics.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
PREPROCESS_SCRIPT_PATH="${SCRIPT_DIR}/${PREPROCESS_SCRIPT_NAME}"
TUI_SCRIPT_PATH="${PROJECT_ROOT}/scripts/common/tui.sh"

if [ -f "$TUI_SCRIPT_PATH" ]; then
    # shellcheck source=scripts/common/tui.sh
    source "$TUI_SCRIPT_PATH"
fi

# --- 色付け (オプション) ---
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- ヘルプ関数 ---
show_help() {
    echo "Usage: $0 -b <path> -o <path> [--scan_topic <topic>] [--cmd_topic <topic>]"
    echo ""
    echo "Interactively select sequences and preprocess them to create train/test datasets."
    echo ""
    echo "Options:"
    echo "  -b, --base_dir   Base directory to search for sequences (recursively)"
    echo "  -o, --outdir     Output root directory for datasets (e.g., ./datasets)"
    echo "  --scan_topic     LaserScan topic to extract (default: /virtual_scan)"
    echo "  --cmd_topic      AckermannDriveStamped topic to extract (default: /jetracer/cmd_drive)"
    echo "  --use_virtual_scan Force save extracted scans as virtual_scans.npy"
    echo "  --no_use_virtual_scan Force save extracted scans as scans.npy"
    echo "  --legacy-select  Use the old number-input selector instead of checkbox TUI"
    echo "  -h, --help       Show this help message"
}

# --- 引数解析 ---
BASE_DIR=""
OUTDIR=""
SCAN_TOPIC="/virtual_scan"
CMD_TOPIC="/jetracer/cmd_drive"
LEGACY_SELECT="false"
USE_VIRTUAL_SCAN="auto"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -b|--base_dir)
        BASE_DIR="$2"
        shift 2
        ;;
        -o|--outdir)
        OUTDIR="$2"
        shift 2
        ;;
        --scan_topic|--scan-topic)
        SCAN_TOPIC="$2"
        shift 2
        ;;
        --cmd_topic|--cmd-topic)
        CMD_TOPIC="$2"
        shift 2
        ;;
        --use_virtual_scan|--use-virtual-scan)
        USE_VIRTUAL_SCAN="true"
        shift
        ;;
        --no_use_virtual_scan|--no-use-virtual-scan)
        USE_VIRTUAL_SCAN="false"
        shift
        ;;
        --legacy-select)
        LEGACY_SELECT="true"
        shift
        ;;
        -h|--help)
        show_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
done

# 必須引数のチェック
if [ -z "$BASE_DIR" ] || [ -z "$OUTDIR" ]; then
    echo -e "${RED}ERROR: Both -b (--base_dir) and -o (--outdir) are required.${NC}"
    show_help
    exit 1
fi

# Pythonスクリプトの存在チェック
if [ ! -f "$PREPROCESS_SCRIPT_PATH" ]; then
    echo -e "${RED}CRITICAL ERROR: Preprocessing script not found at: $PREPROCESS_SCRIPT_PATH${NC}"
    exit 1
fi

# --- 1. シーケンスを探索 ---
echo -e "🔄 Searching for sequences under: ${CYAN}$BASE_DIR${NC}"
sequences=()
while IFS= read -r metadata_path; do
    sequences+=("$(dirname "$metadata_path")")
done < <(find "$BASE_DIR" -name "metadata.yaml" -print | sort)

if [ ${#sequences[@]} -eq 0 ]; then
    echo -e "${YELLOW}❌ No sequences found.${NC}"
    exit 0
fi

# --- 2. シーケンスを表示 ---
echo -e "\n--- 📂 Found Sequences ---"
for i in "${!sequences[@]}"; do
    relative_path=$(echo "${sequences[$i]}" | sed "s|^${BASE_DIR}/||")
    printf "  [${GREEN}%02d${NC}] %s\n" "$((i+1))" "$relative_path"
done
echo -e "----------------------------\n"

# --- 3. 選択関数 ---
select_sequences_legacy() {
    local prompt_message="$1"
    local output_name="$2"
    local selected=()
    local idx
    local path
    
    echo -e "👉 ${CYAN}$prompt_message${NC}"
    echo "   (Enter numbers separated by space, e.g., 1 3 5)"
    read -p "   Select: " -a indices

    eval "$output_name=()"
    for idx in "${indices[@]}"; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#sequences[@]}" ]; then
            path="${sequences[$((idx-1))]}"
            eval "$output_name+=(\"\$path\")"
        else
            echo -e "   ${YELLOW}⚠️ Invalid number skipped: $idx${NC}"
        fi
    done
    
    eval "selected=(\"\${${output_name}[@]}\")"
    echo "   Selected:"
    for p in "${selected[@]}"; do
        echo -e "     ${GREEN}✅ $(basename "$p")${NC}"
    done
    echo ""
}

select_sequences() {
    local prompt_message="$1"
    local output_name="$2"

    if [[ "$LEGACY_SELECT" != "true" ]] && declare -F tui_select_paths >/dev/null 2>&1; then
        if tui_select_paths "$prompt_message" sequences "$output_name" "$BASE_DIR"; then
            return
        fi
    fi

    select_sequences_legacy "$prompt_message" "$output_name"
}

# --- 4. 実行関数 ---
run_extraction() {
    local output_dir="$1"
    local dataset_name="$2"
    shift 2
    local seq_paths=("$@")
    
    if [ ${#seq_paths[@]} -eq 0 ]; then
        echo -e "${YELLOW}ℹ️ No sequences selected for ${dataset_name}. Skipping.${NC}"
        return 0
    fi

    mkdir -p "$output_dir"
    echo -e "\n🚀 Starting preprocessing for ${GREEN}${dataset_name}${NC} dataset..."
    echo -e "   Outputting to: ${CYAN}$output_dir${NC}"
    echo -e "   scan_topic: ${CYAN}$SCAN_TOPIC${NC}"
    echo -e "   cmd_topic : ${CYAN}$CMD_TOPIC${NC}"

    local extra_args=()
    local resolved_use_virtual_scan="$USE_VIRTUAL_SCAN"
    if [ "$resolved_use_virtual_scan" = "auto" ]; then
        if [[ "$SCAN_TOPIC" == *"virtual"* ]]; then
            resolved_use_virtual_scan="true"
        else
            resolved_use_virtual_scan="false"
        fi
    fi
    if [ "$resolved_use_virtual_scan" = "true" ]; then
        extra_args+=("--use_virtual_scan")
    fi

    python3 "$PREPROCESS_SCRIPT_PATH" --seq_dirs "${seq_paths[@]}" --outdir "$output_dir" --scan_topic "$SCAN_TOPIC" --cmd_topic "$CMD_TOPIC" "${extra_args[@]}"
    
    if [ $? -eq 0 ]; then
        echo -e "✅ Finished preprocessing for ${GREEN}${dataset_name}${NC}."
        return 0
    else
        echo -e "${RED}❌ ERROR: Preprocessing failed for ${dataset_name}.${NC}"
        return 1
    fi
}

# --- 5. メイン処理 ---
declare -a train_paths
declare -a test_paths

select_sequences "Select TRAIN sequences." train_paths
select_sequences "Select TEST sequences." test_paths

run_extraction "$OUTDIR/train" "TRAIN" "${train_paths[@]}"
if [ $? -ne 0 ]; then exit 1; fi

run_extraction "$OUTDIR/test" "TEST" "${test_paths[@]}"
if [ $? -ne 0 ]; then exit 1; fi

echo -e "\n🎉 Dataset created successfully at ${CYAN}$OUTDIR${NC}"
