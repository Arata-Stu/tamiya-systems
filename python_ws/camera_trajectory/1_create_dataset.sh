#!/bin/bash

PREPROCESS_SCRIPT_NAME="extract_topics.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
PREPROCESS_SCRIPT_PATH="${SCRIPT_DIR}/${PREPROCESS_SCRIPT_NAME}"
TUI_SCRIPT_PATH="${PROJECT_ROOT}/scripts/common/tui.sh"

if [ -f "$TUI_SCRIPT_PATH" ]; then
    # shellcheck source=scripts/common/tui.sh
    source "$TUI_SCRIPT_PATH"
fi

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 -b <path> -o <path> [options]"
    echo ""
    echo "Interactively select sequences and preprocess them to create camera trajectory datasets."
    echo ""
    echo "Options:"
    echo "  -b, --base_dir              Base directory to search for rosbag2 sequences"
    echo "  -o, --outdir                Output root directory for datasets (e.g., ./datasets)"
    echo "  --image_topic               Image topic name (default: /camera/left/image_raw)"
    echo "  --pose_topic                Pose/path topic (default: /visual_slam/tracking/odometry)"
    echo "                              Useful candidates:"
    echo "                                /visual_slam/tracking/slam_path"
    echo "                                /visual_slam/tracking/odometry"
    echo "                                /visual_slam/tracking/vo_pose"
    echo "  --image_storage             Image storage format: npy or png (default: npy)"
    echo "  --num_points                Number of future path points (default: 20)"
    echo "  --min_distance              First target distance in meters (default: 0.5)"
    echo "  --max_distance              Last target distance in meters (default: 8.0)"
    echo "  --target_distances          Explicit distances in meters, e.g. 0.5 1.0 1.5 2.0"
    echo "  --max_pose_time_diff        Max image/pose sync offset in seconds (default: 0.2)"
    echo "  --workers                   Number of parallel workers"
    echo "  --legacy-select             Use the old number-input selector instead of checkbox TUI"
    echo "  -h, --help                  Show this help message"
}

BASE_DIR=""
OUTDIR=""
IMAGE_TOPIC="/camera/left/image_raw"
POSE_TOPIC="/visual_slam/tracking/odometry"
IMAGE_STORAGE="npy"
NUM_POINTS="20"
MIN_DISTANCE="0.5"
MAX_DISTANCE="8.0"
MAX_POSE_TIME_DIFF="0.2"
WORKERS=""
TARGET_DISTANCES=()
LEGACY_SELECT="false"

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
        --image_topic)
        IMAGE_TOPIC="$2"
        shift 2
        ;;
        --pose_topic)
        POSE_TOPIC="$2"
        shift 2
        ;;
        --image_storage)
        IMAGE_STORAGE="$2"
        shift 2
        ;;
        --num_points)
        NUM_POINTS="$2"
        shift 2
        ;;
        --min_distance)
        MIN_DISTANCE="$2"
        shift 2
        ;;
        --max_distance)
        MAX_DISTANCE="$2"
        shift 2
        ;;
        --target_distances)
        shift
        TARGET_DISTANCES=()
        while [[ $# -gt 0 && "$1" != --* ]]; do
            TARGET_DISTANCES+=("$1")
            shift
        done
        ;;
        --max_pose_time_diff)
        MAX_POSE_TIME_DIFF="$2"
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
    echo -e "${RED}ERROR: Both -b (--base_dir) and -o (--outdir) are required.${NC}"
    show_help
    exit 1
fi

if [[ "$IMAGE_STORAGE" != "npy" && "$IMAGE_STORAGE" != "png" ]]; then
    echo -e "${RED}ERROR: --image_storage must be 'npy' or 'png'.${NC}"
    exit 1
fi

if [ ! -f "$PREPROCESS_SCRIPT_PATH" ]; then
    echo -e "${RED}CRITICAL ERROR: Preprocessing script not found at: $PREPROCESS_SCRIPT_PATH${NC}"
    exit 1
fi

echo -e "Searching sequences under: ${CYAN}$BASE_DIR${NC}"
sequences=()
while IFS= read -r metadata_path; do
    sequences+=("$(dirname "$metadata_path")")
done < <(find "$BASE_DIR" -name "metadata.yaml" -print | sort)

if [ ${#sequences[@]} -eq 0 ]; then
    echo -e "${YELLOW}No sequences found.${NC}"
    exit 0
fi

echo -e "\n--- Found Sequences ---"
for i in "${!sequences[@]}"; do
    relative_path=$(echo "${sequences[$i]}" | sed "s|^${BASE_DIR}/||")
    printf "  [${GREEN}%02d${NC}] %s\n" "$((i+1))" "$relative_path"
done
echo -e "-----------------------\n"

select_sequences_legacy() {
    local prompt_message="$1"
    local output_name="$2"
    local selected=()
    local idx
    local path

    echo -e "${CYAN}$prompt_message${NC}"
    echo "  Enter numbers separated by spaces (e.g. 1 3 5)"
    read -p "  Select: " -a indices

    eval "$output_name=()"
    for idx in "${indices[@]}"; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#sequences[@]}" ]; then
            path="${sequences[$((idx-1))]}"
            eval "$output_name+=(\"\$path\")"
        else
            echo -e "  ${YELLOW}Skip invalid number: $idx${NC}"
        fi
    done

    eval "selected=(\"\${${output_name}[@]}\")"
    echo "  Selected:"
    for p in "${selected[@]}"; do
        echo -e "    ${GREEN}$(basename "$p")${NC}"
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

run_extraction() {
    local output_dir="$1"
    local dataset_name="$2"
    shift 2
    local seq_paths=("$@")

    if [ ${#seq_paths[@]} -eq 0 ]; then
        echo -e "${YELLOW}No sequences selected for ${dataset_name}. Skipping.${NC}"
        return 0
    fi

    mkdir -p "$output_dir"
    echo -e "\nStart preprocessing for ${GREEN}${dataset_name}${NC}"
    echo -e "Output: ${CYAN}$output_dir${NC}"

    local cmd=(python3 "$PREPROCESS_SCRIPT_PATH" --seq_dirs "${seq_paths[@]}" --outdir "$output_dir" --image_topic "$IMAGE_TOPIC" --pose_topic "$POSE_TOPIC" --image_storage "$IMAGE_STORAGE" --num_points "$NUM_POINTS" --min_distance "$MIN_DISTANCE" --max_distance "$MAX_DISTANCE" --max_pose_time_diff "$MAX_POSE_TIME_DIFF")
    if [ ${#TARGET_DISTANCES[@]} -gt 0 ]; then
        cmd+=(--target_distances "${TARGET_DISTANCES[@]}")
    fi
    if [ -n "$WORKERS" ]; then
        cmd+=(--workers "$WORKERS")
    fi

    "${cmd[@]}"
    if [ $? -eq 0 ]; then
        echo -e "Finished ${GREEN}${dataset_name}${NC}"
        return 0
    fi

    echo -e "${RED}ERROR: Preprocessing failed for ${dataset_name}.${NC}"
    return 1
}

declare -a train_paths
declare -a test_paths

select_sequences "Select TRAIN sequences." train_paths
select_sequences "Select TEST sequences." test_paths

run_extraction "$OUTDIR/train" "TRAIN" "${train_paths[@]}"
if [ $? -ne 0 ]; then exit 1; fi

run_extraction "$OUTDIR/test" "TEST" "${test_paths[@]}"
if [ $? -ne 0 ]; then exit 1; fi

echo -e "\nDataset created successfully at ${CYAN}$OUTDIR${NC}"
