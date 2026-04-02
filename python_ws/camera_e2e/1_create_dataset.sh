#!/bin/bash

# --- Script Settings ---
PREPROCESS_SCRIPT_NAME="extract_topics.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PREPROCESS_SCRIPT_PATH="${SCRIPT_DIR}/${PREPROCESS_SCRIPT_NAME}"

# --- Colors (optional) ---
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 -b <path> -o <path> [options]"
    echo ""
    echo "Interactively select sequences and preprocess them to create camera train/test datasets."
    echo ""
    echo "Options:"
    echo "  -b, --base_dir        Base directory to search for sequences (recursively)"
    echo "  -o, --outdir          Output root directory for datasets (e.g., ./datasets)"
    echo "  --image_topic         Image topic name (default: /camera/left/image_raw)"
    echo "  --cmd_topic           Ackermann topic name (default: /jetracer/cmd_drive)"
    echo "  --image_storage       Image storage format: npy or png (default: npy)"
    echo "  --workers             Number of parallel workers"
    echo "  -h, --help            Show this help message"
}

BASE_DIR=""
OUTDIR=""
IMAGE_TOPIC="/camera/left/image_raw"
CMD_TOPIC="/jetracer/cmd_drive"
IMAGE_STORAGE="npy"
WORKERS=""

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
        --cmd_topic)
        CMD_TOPIC="$2"
        shift 2
        ;;
        --image_storage)
        IMAGE_STORAGE="$2"
        shift 2
        ;;
        --workers)
        WORKERS="$2"
        shift 2
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
mapfile -t sequences < <(find "$BASE_DIR" -name "metadata.yaml" -print0 | xargs -0 -I {} dirname {} | sort)

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

select_sequences() {
    local prompt_message="$1"
    local -n output_array=$2

    echo -e "${CYAN}$prompt_message${NC}"
    echo "  Enter numbers separated by spaces (e.g. 1 3 5)"
    read -p "  Select: " -a indices

    output_array=()
    for idx in "${indices[@]}"; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#sequences[@]}" ]; then
            output_array+=("${sequences[$((idx-1))]}")
        else
            echo -e "  ${YELLOW}Skip invalid number: $idx${NC}"
        fi
    done

    echo "  Selected:"
    for p in "${output_array[@]}"; do
        echo -e "    ${GREEN}$(basename "$p")${NC}"
    done
    echo ""
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

    local cmd=(python3 "$PREPROCESS_SCRIPT_PATH" --seq_dirs "${seq_paths[@]}" --outdir "$output_dir" --image_topic "$IMAGE_TOPIC" --cmd_topic "$CMD_TOPIC" --image_storage "$IMAGE_STORAGE")
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

