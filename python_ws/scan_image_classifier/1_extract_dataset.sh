#!/bin/bash

PREPROCESS_SCRIPT_NAME="extract_crop_images.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
PREPROCESS_SCRIPT_PATH="${SCRIPT_DIR}/${PREPROCESS_SCRIPT_NAME}"
TUI_SCRIPT_PATH="${PROJECT_ROOT}/scripts/common/tui.sh"

if [ -f "$TUI_SCRIPT_PATH" ]; then
    # shellcheck source=scripts/common/tui.sh
    source "$TUI_SCRIPT_PATH"
fi

show_help() {
    echo "Usage: $0 -b <path> -o <path> [options]"
    echo ""
    echo "Interactively select rosbag sequences and extract crop images into a classification dataset."
    echo ""
    echo "Options:"
    echo "  -b, --base_dir        Base directory to search for rosbag2 sequences"
    echo "  -o, --outdir          Output dataset root (e.g., ./datasets)"
    echo "  --image_topic         Crop image topic (default: /perception/crop/image)"
    echo "  --import_name         Human readable name for this extraction batch"
    echo "  --import_id           Explicit import ID to append/update"
    echo "  --workers             Number of parallel workers"
    echo "  --legacy-select       Use number-input selector instead of checkbox TUI"
    echo "  -h, --help            Show this help message"
}

BASE_DIR=""
OUTDIR=""
IMAGE_TOPIC="/perception/crop/image"
IMPORT_NAME=""
IMPORT_ID=""
WORKERS=""
LEGACY_SELECT="false"

while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        -b|--base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -o|--outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --image_topic)
            IMAGE_TOPIC="$2"
            shift 2
            ;;
        --import_name)
            IMPORT_NAME="$2"
            shift 2
            ;;
        --import_id)
            IMPORT_ID="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
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

if [ -z "$BASE_DIR" ] || [ -z "$OUTDIR" ]; then
    echo "ERROR: Both -b (--base_dir) and -o (--outdir) are required."
    show_help
    exit 1
fi

if [ ! -f "$PREPROCESS_SCRIPT_PATH" ]; then
    echo "ERROR: Preprocessing script not found at: $PREPROCESS_SCRIPT_PATH"
    exit 1
fi

sequences=()
while IFS= read -r metadata_path; do
    sequences+=("$(dirname "$metadata_path")")
done < <(find "$BASE_DIR" -name "metadata.yaml" -print | sort)

if [ ${#sequences[@]} -eq 0 ]; then
    echo "No rosbag sequences found under: $BASE_DIR"
    exit 0
fi

select_sequences_legacy() {
    local output_name="$1"
    eval "$output_name=()"

    echo "Select sequences to extract:"
    for i in "${!sequences[@]}"; do
        relative_path=$(echo "${sequences[$i]}" | sed "s|^${BASE_DIR}/||")
        printf "  [%02d] %s\n" "$((i+1))" "$relative_path"
    done

    read -p "Enter numbers separated by spaces: " -a indices
    for idx in "${indices[@]}"; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#sequences[@]}" ]; then
            path="${sequences[$((idx-1))]}"
            eval "$output_name+=(\"\$path\")"
        fi
    done
}

select_sequences() {
    local output_name="$1"
    if [[ "$LEGACY_SELECT" != "true" ]] && declare -F tui_select_paths >/dev/null 2>&1; then
        if tui_select_paths "Select rosbag sequences to extract." sequences "$output_name" "$BASE_DIR"; then
            return
        fi
    fi
    select_sequences_legacy "$output_name"
}

declare -a selected_sequences
select_sequences selected_sequences

if [ ${#selected_sequences[@]} -eq 0 ]; then
    echo "No sequences selected."
    exit 1
fi

cmd=(python3 "$PREPROCESS_SCRIPT_PATH" --seq_dirs "${selected_sequences[@]}" --outdir "$OUTDIR" --image_topic "$IMAGE_TOPIC")
if [ -n "$IMPORT_NAME" ]; then
    cmd+=(--import_name "$IMPORT_NAME")
fi
if [ -n "$IMPORT_ID" ]; then
    cmd+=(--import_id "$IMPORT_ID")
fi
if [ -n "$WORKERS" ]; then
    cmd+=(--workers "$WORKERS")
fi

"${cmd[@]}"
