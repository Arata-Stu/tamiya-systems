#!/bin/bash

# One-shot local map build flow:
#   sensor bag -> online VSLAM + Cartographer 2D map -> HD map editor

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
INITIAL_PWD="${PWD}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/core_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/rosbag_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/map_processing.sh"

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_map_and_hd_map_from_bag.sh [OPTIONS]

Options:
  --bag-path DIR          input rosbag2 directory (skip interactive selection)
  --map-name NAME         output map name (skip interactive prompt)
  --record-root DIR       rosbag search root for interactive selection (default: /record)
  --mode NAME             2D map mode (default: with_odom_online_vslam)
  --rate RATE             ros2 bag play rate for 2D/VSLAM mapping (default: 1.0)
  --image-width PX        offline VSLAM image width (default: 424)
  --image-height PX       offline VSLAM image height (default: 240)
  --image-fps FPS         offline VSLAM image fps (default: 90.0)
  --with-imu              replay /camera/imu during mapping (default)
  --no-imu                do not replay /camera/imu
  --play-all-topics       replay every source-bag topic instead of filtered topics
  --use-image-preprocessors run rectify/mono preprocessing before VSLAM
  --no-image-preprocessors make VSLAM subscribe to recorded camera topics directly (default)
  --launch-offline-tf     publish fallback base_link TFs instead of using only bag TFs
  --skip-2d-map           reuse existing <map_dir>/<map_name>.yaml and VSLAM snapshot
  --editor-scale SCALE    initial HD editor zoom; 0 fits the whole raster (default: 0)
  --no-editor             only create 2D/VSLAM/HD raster outputs; do not open HD editor
  --no-raceline           skip raceline generation after the HD editor exits
  --no-line-preview       skip centerline/raceline overlay PNG generation
  -h, --help              show this help

Outputs:
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<source_bag>/<MAP_NAME>/cuvslam_map/
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_reference.json
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_hd_map.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_centerline.csv
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_raceline.csv
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

CREATE_2D_MAP_SH="${SCRIPT_DIR}/create_2d_map_from_bag.sh"
CREATE_HD_MAP_SH="${SCRIPT_DIR}/create_hd_map_from_vslam_bag.sh"

BAG_PATH=""
MAP_NAME=""
RECORD_ROOT="/record"
MODE="with_odom_online_vslam"
PLAY_RATE="1.0"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
IMAGE_FPS="90.0"
USE_IMU=true
PLAY_ALL_TOPICS=false
USE_IMAGE_PREPROCESSORS=false
LAUNCH_OFFLINE_TF=false
SKIP_2D_MAP=false
OPEN_EDITOR=true
GENERATE_RACELINE=true
GENERATE_LINE_PREVIEW=true
EDITOR_SCALE="0"
ROSBAG_CANDIDATES=()
SCAN_TOPIC="/scan"
SOURCE_PLAY_TOPICS=()

while (($#)); do
    case "$1" in
        --bag-path)
            BAG_PATH="$2"
            shift 2
            ;;
        --map-name)
            MAP_NAME="$2"
            shift 2
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --rate)
            PLAY_RATE="$2"
            shift 2
            ;;
        --image-width)
            IMAGE_WIDTH="$2"
            shift 2
            ;;
        --image-height)
            IMAGE_HEIGHT="$2"
            shift 2
            ;;
        --image-fps)
            IMAGE_FPS="$2"
            shift 2
            ;;
        --with-imu)
            USE_IMU=true
            shift
            ;;
        --no-imu)
            USE_IMU=false
            shift
            ;;
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --use-image-preprocessors)
            USE_IMAGE_PREPROCESSORS=true
            shift
            ;;
        --no-image-preprocessors)
            USE_IMAGE_PREPROCESSORS=false
            shift
            ;;
        --launch-offline-tf)
            LAUNCH_OFFLINE_TF=true
            shift
            ;;
        --skip-2d-map)
            SKIP_2D_MAP=true
            shift
            ;;
        --editor-scale)
            EDITOR_SCALE="$2"
            shift 2
            ;;
        --no-editor)
            OPEN_EDITOR=false
            shift
            ;;
        --no-raceline)
            GENERATE_RACELINE=false
            shift
            ;;
        --no-line-preview)
            GENERATE_LINE_PREVIEW=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "Unknown option: $1"
            ;;
        *)
            die "Positional arguments are not supported: $1"
            ;;
    esac
done

if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi
BAG_PATH="${BAG_PATH%/}"
if [ ! -d "${BAG_PATH}" ] || [ ! -f "${BAG_PATH}/metadata.yaml" ]; then
    die "Invalid rosbag2 directory: ${BAG_PATH}"
fi

if [ -z "${MAP_NAME}" ]; then
    prompt_map_name_interactive
fi
if [[ "${MAP_NAME}" == *"/"* ]]; then
    die "map name must not contain '/'"
fi

BAG_DIR_NAME="$(basename "${BAG_PATH}")"
MAP_DIR="/map/${BAG_DIR_NAME}/${MAP_NAME}"
MAP_STEM="${MAP_DIR}/${MAP_NAME}"
MAP_YAML_PATH="${MAP_STEM}.yaml"
SNAPSHOT_PATH="${MAP_STEM}_vslam_reference.json"

echo ""
echo "================ One-shot map build ================"
echo "source bag : ${BAG_PATH}"
echo "map name   : ${MAP_NAME}"
echo "map dir    : ${MAP_DIR}"
echo "2D mode    : ${MODE}"
echo "IMU replay : ${USE_IMU}"
echo "image prep : ${USE_IMAGE_PREPROCESSORS}"
echo "===================================================="

if [ "${SKIP_2D_MAP}" != true ]; then
    map_cmd=(
        bash "${CREATE_2D_MAP_SH}"
        --mode "${MODE}"
        --bag-path "${BAG_PATH}"
        --map-name "${MAP_NAME}"
        --rate "${PLAY_RATE}"
        --image-width "${IMAGE_WIDTH}"
        --image-height "${IMAGE_HEIGHT}"
        --image-fps "${IMAGE_FPS}"
        --record-root "${RECORD_ROOT}"
        --save-vslam-reference
        --vslam-vis
        --no-live-vslam-map-align
        --no-centerline
        --no-raceline
        --no-line-preview
        --no-scp
    )
    if [ "${USE_IMU}" = true ]; then
        map_cmd+=(--with-imu)
    else
        map_cmd+=(--no-imu)
    fi
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        map_cmd+=(--play-all-topics)
    fi
    if [ "${USE_IMAGE_PREPROCESSORS}" = true ]; then
        map_cmd+=(--use-image-preprocessors)
    else
        map_cmd+=(--no-image-preprocessors)
    fi
    if [ "${LAUNCH_OFFLINE_TF}" = true ]; then
        map_cmd+=(--launch-offline-tf)
    fi

    echo ""
    echo "[1/2] Build 2D map and VSLAM reference"
    printf '  %q' "${map_cmd[@]}"
    echo ""
    "${map_cmd[@]}"
else
    echo ""
    echo "[1/2] Reuse existing 2D map and VSLAM reference"
fi

if [ ! -f "${MAP_YAML_PATH}" ]; then
    die "2D map YAML was not created: ${MAP_YAML_PATH}"
fi
if [ ! -f "${SNAPSHOT_PATH}" ]; then
    die "VSLAM reference snapshot was not created: ${SNAPSHOT_PATH}"
fi

hd_cmd=(
    bash "${CREATE_HD_MAP_SH}"
    --skip-vslam
    --bag-path "${BAG_PATH}"
    --map-name "${MAP_NAME}"
    --map-dir "${MAP_DIR}"
    --snapshot "${SNAPSHOT_PATH}"
    --reference-yaml "${MAP_YAML_PATH}"
    --editor-scale "${EDITOR_SCALE}"
)
if [ "${OPEN_EDITOR}" != true ]; then
    hd_cmd+=(--no-editor)
fi
if [ "${GENERATE_RACELINE}" != true ]; then
    hd_cmd+=(--no-raceline)
fi
if [ "${GENERATE_LINE_PREVIEW}" != true ]; then
    hd_cmd+=(--no-line-preview)
fi

echo ""
echo "[2/2] Open HD map editor from VSLAM reference"
printf '  %q' "${hd_cmd[@]}"
echo ""
"${hd_cmd[@]}"

echo ""
echo "✅ Map bundle ready:"
echo "  - map dir    : ${MAP_DIR}"
echo "  - 2D map     : ${MAP_YAML_PATH}"
echo "  - snapshot   : ${SNAPSHOT_PATH}"
echo "  - HD map     : ${MAP_STEM}_hd_map.yaml"
