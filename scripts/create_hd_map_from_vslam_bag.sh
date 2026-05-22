#!/bin/bash

# Local HD map experiment flow:
#   sensor bag -> offline VSLAM reference snapshot -> landmark raster -> HD map editor

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
INITIAL_PWD="${PWD}"

# Reuse the bag-selection, path-resolution, and process helpers used by the
# existing create_2d_map flow without coupling this script to Cartographer.
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
  create_hd_map_from_vslam_bag.sh [OPTIONS]

Options:
  --bag-path DIR          input rosbag2 directory (skip interactive selection)
  --map-name NAME         output HD map experiment name (skip interactive prompt)
  --map-dir DIR           explicit output directory
  --record-root DIR       rosbag search root for interactive selection (default: /record)
  --map-root DIR          output root used without --map-dir (default: /map)
  --rate RATE             ros2 bag play rate (default: 1.0)
  --image-width PX        offline VSLAM image width (default: 424)
  --image-height PX       offline VSLAM image height (default: 240)
  --with-imu              replay /camera/imu in addition to stereo topics
  --play-all-topics       replay every source-bag topic instead of filtered topics
  --snapshot PATH         VSLAM reference snapshot path
  --skip-vslam            reuse --snapshot instead of replaying the bag
  --no-save-vslam-map     do not save cuVSLAM native map after replay
  --reference-yaml PATH   optional raster geometry from an existing 2D map YAML
  --alignment PATH        optional vslam_map_alignment.yaml for snapshot export
  --landmark-resolution M auto-raster resolution without --reference-yaml (default: 0.02)
  --landmark-padding M    auto-raster padding without --reference-yaml (default: 0.5)
  --landmark-downsample M XY downsample cell size for landmarks (default: 0.05)
  --landmark-min-z M      keep exported landmark points at or above z
  --landmark-max-z M      keep exported landmark points at or below z
  --no-editor             only create snapshot/raster outputs; do not open HD editor
  --no-raceline           skip raceline generation after the editor exits
  --no-line-preview       skip centerline/raceline overlay PNG generation
  --hd-map-yaml PATH      editable HD map YAML path
  --centerline-csv PATH   primary lane centerline CSV path
  --raceline-csv PATH     generated raceline CSV path
  --line-preview-png PATH centerline/raceline debug overlay PNG path
  --raceline-preset NAME  generate_raceline.py preset (default: race-stacks)
  --raceline-backend NAME generate_raceline.py backend (default: global-opt)
  --raceline-opt-type NAME global-opt objective (default: mincurv)
  --optimizer-root DIR    optional global raceline optimizer checkout
  --editor-scale SCALE    initial HD editor zoom; 0 fits the whole raster (default: 1.0)
  --hd-map-editor PATH    explicit hd_map_editor.py path
  --raceline-script PATH  explicit generate_raceline.py path
  --line-preview-script PATH explicit visualize_race_lines.py path
  -h, --help              show this help

Default outputs:
  /map/<source_bag>/<MAP_NAME>/cuvslam_map/
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_reference.json
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.png
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_hd_map.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_centerline.csv
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_raceline.csv
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_lines.png

Typical flow:
  1) bag and map name are selected interactively unless options provide them
  2) offline VSLAM runs in the local map frame and records final landmarks/path
  3) snapshot data is rasterized into a PNG+YAML HD editor background
  4) hd_map_editor.py opens for left/right/centerline editing
  5) a centerline CSV, raceline CSV, and debug overlay PNG are produced when saved
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

resolve_hd_map_editor_script() {
    if [ -n "${HD_MAP_EDITOR_SCRIPT_PATH}" ]; then
        [ -f "${HD_MAP_EDITOR_SCRIPT_PATH}" ] && echo "${HD_MAP_EDITOR_SCRIPT_PATH}" && return 0
        return 1
    fi

    resolve_python_ws_file "map_section_editor/hd_map_editor.py"
}

resolve_snapshot_exporter_script() {
    resolve_repo_file "ros2_ws/src/tools/vslam_map_tools/vslam_map_tools/export_aligned_landmarks_offline.py"
}

stop_vslam_stack() {
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

stop_snapshot_recorder() {
    stop_background_process "VSLAM_REFERENCE_PID" "VSLAM_REFERENCE_USES_SETSID"
}

cleanup_all() {
    stop_snapshot_recorder
    stop_vslam_stack
}

launch_hdmap_vslam_stack() {
    local -a launch_args=(
        "use_sim_time:=true"
        "image_width:=${IMAGE_WIDTH}"
        "image_height:=${IMAGE_HEIGHT}"
        "camera_container_name:=${CAMERA_CONTAINER_NAME}"
        "vslam_map_frame:=map"
        "vslam_map_parent_frame:=map"
        "publish_vslam_map_identity_tf:=false"
        "enable_localization_and_mapping:=true"
        "enable_slam_visualization:=true"
        "enable_landmarks_view:=true"
    )

    if [ "${SAVE_VSLAM_MAP}" = true ]; then
        launch_args+=("save_map_path:=${VSLAM_MAP_DIR}")
    fi

    build_system_launch_cmd "offline_sensor_tf.launch.xml"
    launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        > "${TF_LOG_PATH}" 2>&1

    sleep 2

    launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
        ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

    sleep 2

    build_system_launch_cmd "vslam.launch.xml"
    launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        "${launch_args[@]}" \
        > "${VSLAM_LOG_PATH}" 2>&1
}

start_vslam_snapshot_recorder() {
    local -a recorder_cmd=(
        env PYTHONUNBUFFERED=1 ros2 run vslam_map_tools record_vslam_reference_snapshot.py
        --path-topic "/visual_slam/tracking/slam_path"
        --odom-topic "/visual_slam/tracking/odometry"
        --landmarks-topic "/visual_slam/vis/landmarks_cloud"
        --output "${SNAPSHOT_PATH}"
    )

    launch_background_process "VSLAM_REFERENCE_PID" "VSLAM_REFERENCE_USES_SETSID" \
        "${recorder_cmd[@]}" \
        > "${SNAPSHOT_LOG_PATH}" 2>&1
}

run_vslam_snapshot_capture() {
    echo "[1/4] Launch offline VSLAM for HD map reference"
    echo "  - TF log    : ${TF_LOG_PATH}"
    echo "  - VSLAM log : ${VSLAM_LOG_PATH}"
    launch_hdmap_vslam_stack

    echo "[2/4] Wait for visual_slam/save_map"
    if ! wait_for_service "/visual_slam/save_map" 60; then
        die "Visual SLAM service did not become ready. Check ${VSLAM_LOG_PATH}"
    fi

    echo "[3/4] Record VSLAM landmarks/path snapshot while replaying bag"
    start_vslam_snapshot_recorder
    sleep 2

    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - replay all topics"
        play_rosbag "${BAG_PATH}" "${PLAYER_LOG_PATH}"
    else
        build_online_source_play_topics
        echo "  - replay topics: ${SOURCE_PLAY_TOPICS[*]}"
        play_rosbag "${BAG_PATH}" "${PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"
    fi

    sleep 2
    stop_snapshot_recorder

    if [ ! -f "${SNAPSHOT_PATH}" ]; then
        die "VSLAM snapshot was not written. Check ${SNAPSHOT_LOG_PATH}"
    fi

    echo "[4/4] Save native cuVSLAM map and stop stack"
    if [ "${SAVE_VSLAM_MAP}" = true ]; then
        mkdir -p "${VSLAM_MAP_DIR}"
        if ! ros2 service call /visual_slam/save_map \
            isaac_ros_visual_slam_interfaces/srv/FilePath \
            "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null; then
            echo "Warning: visual_slam/save_map failed. Snapshot export can still continue." >&2
        fi
    else
        echo "  - native map save disabled"
    fi
    stop_vslam_stack
}

export_landmark_raster() {
    local exporter_script_path
    local -a export_cmd

    if ! exporter_script_path="$(resolve_snapshot_exporter_script)"; then
        die "export_aligned_landmarks_offline.py was not found"
    fi

    export_cmd=(
        python3 "${exporter_script_path}"
        --snapshot "${SNAPSHOT_PATH}"
        --output-image "${LANDMARK_IMAGE_PATH}"
        --output-yaml "${LANDMARK_YAML_PATH}"
        --resolution "${LANDMARK_RESOLUTION}"
        --padding-m "${LANDMARK_PADDING_M}"
        --landmark-downsample-m "${LANDMARK_DOWNSAMPLE_M}"
    )

    if [ -n "${REFERENCE_YAML_PATH}" ]; then
        export_cmd+=(--reference-yaml "${REFERENCE_YAML_PATH}")
    fi
    if [ -n "${ALIGNMENT_PATH}" ]; then
        export_cmd+=(--alignment "${ALIGNMENT_PATH}")
    fi
    if [ -n "${LANDMARK_MIN_Z}" ]; then
        export_cmd+=(--min-z "${LANDMARK_MIN_Z}")
    fi
    if [ -n "${LANDMARK_MAX_Z}" ]; then
        export_cmd+=(--max-z "${LANDMARK_MAX_Z}")
    fi

    echo "[post] Export landmark raster for HD map editing"
    if ! "${export_cmd[@]}" > "${EXPORT_LOG_PATH}" 2>&1; then
        die "Landmark raster export failed. Check ${EXPORT_LOG_PATH}"
    fi
    if [ ! -f "${LANDMARK_IMAGE_PATH}" ] || [ ! -f "${LANDMARK_YAML_PATH}" ]; then
        die "Landmark raster outputs were not written. Check ${EXPORT_LOG_PATH}"
    fi
    echo "  - raster image: ${LANDMARK_IMAGE_PATH}"
    echo "  - raster yaml : ${LANDMARK_YAML_PATH}"
}

run_hd_map_editor() {
    local editor_script_path
    local -a editor_cmd

    if [ "${OPEN_EDITOR}" != true ]; then
        echo "[post] Skip HD map editor"
        return 0
    fi

    if ! editor_script_path="$(resolve_hd_map_editor_script)"; then
        die "hd_map_editor.py was not found"
    fi

    editor_cmd=(
        python3 "${editor_script_path}"
        --map-yaml "${LANDMARK_YAML_PATH}"
        --output "${HD_MAP_YAML_PATH}"
        --centerline-output "${CENTERLINE_CSV_PATH}"
        --scale "${EDITOR_SCALE}"
    )

    echo "[post] Launch HD map editor"
    "${editor_cmd[@]}"
}

run_raceline_export() {
    local raceline_script_path

    if [ "${GENERATE_RACELINE}" != true ]; then
        echo "[post] Skip raceline generation"
        return 0
    fi
    if [ ! -f "${CENTERLINE_CSV_PATH}" ]; then
        echo "[post] Skip raceline generation because centerline CSV was not saved."
        return 0
    fi
    if ! raceline_script_path="$(resolve_raceline_script)"; then
        echo "Warning: generate_raceline.py was not found. Centerline is still available." >&2
        return 0
    fi

    echo "[post] Generate raceline from primary lane centerline"
    local -a raceline_cmd=(
        python3 "${raceline_script_path}"
        --preset "${RACELINE_PRESET}"
        --backend "${RACELINE_BACKEND}"
        --opt-type "${RACELINE_OPT_TYPE}"
        --centerline "${CENTERLINE_CSV_PATH}"
        --output "${RACELINE_CSV_PATH}"
    )
    if [ -n "${GLOBAL_OPTIMIZER_ROOT}" ]; then
        raceline_cmd+=(--optimizer-root "${GLOBAL_OPTIMIZER_ROOT}")
    fi

    if ! "${raceline_cmd[@]}" > "${RACELINE_LOG_PATH}" 2>&1; then
        echo "Warning: raceline generation failed. Check ${RACELINE_LOG_PATH}" >&2
        return 0
    fi
    RACELINE_CREATED=true
    echo "  - raceline CSV: ${RACELINE_CSV_PATH}"
}

run_line_preview() {
    local preview_script_path
    local -a preview_cmd

    if [ "${GENERATE_LINE_PREVIEW}" != true ]; then
        echo "[post] Skip line preview generation"
        return 0
    fi
    if [ ! -f "${HD_MAP_YAML_PATH}" ] && [ ! -f "${CENTERLINE_CSV_PATH}" ] && [ "${RACELINE_CREATED}" != true ]; then
        echo "[post] Skip line preview generation because no HD map/centerline/raceline was saved."
        return 0
    fi
    if ! preview_script_path="$(resolve_line_preview_script)"; then
        echo "Warning: visualize_race_lines.py was not found. Skip line preview." >&2
        return 0
    fi

    preview_cmd=(
        python3 "${preview_script_path}"
        --yaml "${LANDMARK_YAML_PATH}"
        --output "${LINE_PREVIEW_PNG_PATH}"
        --centerline-thickness 2
        --raceline-thickness 2
    )
    if [ -f "${HD_MAP_YAML_PATH}" ]; then
        preview_cmd+=(--hd-map "${HD_MAP_YAML_PATH}")
    fi
    if [ -f "${CENTERLINE_CSV_PATH}" ]; then
        preview_cmd+=(--centerline "${CENTERLINE_CSV_PATH}")
    fi
    if [ "${RACELINE_CREATED}" = true ] && [ -f "${RACELINE_CSV_PATH}" ]; then
        preview_cmd+=(--raceline "${RACELINE_CSV_PATH}")
    fi

    echo "[post] Project centerline/raceline onto landmark raster"
    if ! "${preview_cmd[@]}" > "${LINE_PREVIEW_LOG_PATH}" 2>&1; then
        echo "Warning: line preview generation failed. Check ${LINE_PREVIEW_LOG_PATH}" >&2
        return 0
    fi
    echo "  - line preview PNG: ${LINE_PREVIEW_PNG_PATH}"
}

print_summary() {
    echo ""
    echo "HD map experiment outputs:"
    echo "  - directory       : ${MAP_DIR}"
    echo "  - VSLAM snapshot  : ${SNAPSHOT_PATH}"
    echo "  - landmark raster : ${LANDMARK_YAML_PATH}"
    echo "  - HD map YAML     : ${HD_MAP_YAML_PATH}"
    echo "  - centerline CSV  : ${CENTERLINE_CSV_PATH}"
    echo "  - raceline CSV    : ${RACELINE_CSV_PATH}"
    echo "  - line preview    : ${LINE_PREVIEW_PNG_PATH}"
    if [ "${SAVE_VSLAM_MAP}" = true ] && [ "${SKIP_VSLAM}" != true ]; then
        echo "  - cuVSLAM map     : ${VSLAM_MAP_DIR}"
    fi
    echo ""
    echo "Pure Pursuit input can be the centerline/raceline CSV through raceline_path_publisher."
}

PLAY_RATE="1.0"
RECORD_ROOT="/record"
MAP_ROOT="/map"
MAP_DIR_OVERRIDE=""
BAG_PATH=""
MAP_NAME=""
SCAN_TOPIC="/scan"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
USE_IMU=false
PLAY_ALL_TOPICS=false
SAVE_VSLAM_MAP=true
SKIP_VSLAM=false
OPEN_EDITOR=true
GENERATE_RACELINE=true
GENERATE_LINE_PREVIEW=true
RACELINE_CREATED=false
SNAPSHOT_OVERRIDE_PATH=""
REFERENCE_YAML_PATH=""
ALIGNMENT_PATH=""
LANDMARK_RESOLUTION="0.02"
LANDMARK_PADDING_M="0.5"
LANDMARK_DOWNSAMPLE_M="0.05"
LANDMARK_MIN_Z=""
LANDMARK_MAX_Z=""
HD_MAP_YAML_OVERRIDE_PATH=""
CENTERLINE_CSV_OVERRIDE_PATH=""
RACELINE_CSV_OVERRIDE_PATH=""
LINE_PREVIEW_PNG_OVERRIDE_PATH=""
RACELINE_PRESET="race-stacks"
RACELINE_BACKEND="global-opt"
RACELINE_OPT_TYPE="mincurv"
GLOBAL_OPTIMIZER_ROOT=""
EDITOR_SCALE="1.0"
HD_MAP_EDITOR_SCRIPT_PATH=""
RACELINE_SCRIPT_PATH=""
LINE_PREVIEW_SCRIPT_PATH=""
ROSBAG_CANDIDATES=()
SOURCE_PLAY_TOPICS=()
SYSTEM_LAUNCH_CMD=()

OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false
CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
VSLAM_REFERENCE_PID=""
VSLAM_REFERENCE_USES_SETSID=false
CAMERA_CONTAINER_NAME="offline_hdmap_camera_container_$$"

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
        --map-dir)
            MAP_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --map-root)
            MAP_ROOT="$2"
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
        --with-imu)
            USE_IMU=true
            shift
            ;;
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --snapshot)
            SNAPSHOT_OVERRIDE_PATH="$2"
            shift 2
            ;;
        --skip-vslam)
            SKIP_VSLAM=true
            shift
            ;;
        --no-save-vslam-map)
            SAVE_VSLAM_MAP=false
            shift
            ;;
        --reference-yaml)
            REFERENCE_YAML_PATH="$2"
            shift 2
            ;;
        --alignment)
            ALIGNMENT_PATH="$2"
            shift 2
            ;;
        --landmark-resolution)
            LANDMARK_RESOLUTION="$2"
            shift 2
            ;;
        --landmark-padding)
            LANDMARK_PADDING_M="$2"
            shift 2
            ;;
        --landmark-downsample)
            LANDMARK_DOWNSAMPLE_M="$2"
            shift 2
            ;;
        --landmark-min-z)
            LANDMARK_MIN_Z="$2"
            shift 2
            ;;
        --landmark-max-z)
            LANDMARK_MAX_Z="$2"
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
        --hd-map-yaml)
            HD_MAP_YAML_OVERRIDE_PATH="$2"
            shift 2
            ;;
        --centerline-csv)
            CENTERLINE_CSV_OVERRIDE_PATH="$2"
            shift 2
            ;;
        --raceline-csv)
            RACELINE_CSV_OVERRIDE_PATH="$2"
            shift 2
            ;;
        --line-preview-png)
            LINE_PREVIEW_PNG_OVERRIDE_PATH="$2"
            shift 2
            ;;
        --raceline-preset)
            RACELINE_PRESET="$2"
            shift 2
            ;;
        --raceline-backend)
            RACELINE_BACKEND="$2"
            shift 2
            ;;
        --raceline-opt-type)
            RACELINE_OPT_TYPE="$2"
            shift 2
            ;;
        --optimizer-root)
            GLOBAL_OPTIMIZER_ROOT="$2"
            shift 2
            ;;
        --editor-scale)
            EDITOR_SCALE="$2"
            shift 2
            ;;
        --hd-map-editor)
            HD_MAP_EDITOR_SCRIPT_PATH="$2"
            shift 2
            ;;
        --raceline-script)
            RACELINE_SCRIPT_PATH="$2"
            shift 2
            ;;
        --line-preview-script)
            LINE_PREVIEW_SCRIPT_PATH="$2"
            shift 2
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

REPO_ROOT="$(resolve_repo_root)"
SYSTEM_LAUNCH_SOURCE_SHARE="$(resolve_system_launch_source_share || true)"
source_setup_if_available
require_command python3

if [ "${SKIP_VSLAM}" != true ]; then
    require_command ros2
    if [ -z "${BAG_PATH}" ]; then
        select_rosbag_path_interactive
    fi
    BAG_PATH="${BAG_PATH%/}"
    if [ ! -d "${BAG_PATH}" ] || [ ! -f "${BAG_PATH}/metadata.yaml" ]; then
        die "Invalid rosbag2 directory: ${BAG_PATH}"
    fi
fi

if [ -n "${MAP_DIR_OVERRIDE}" ]; then
    MAP_DIR="${MAP_DIR_OVERRIDE%/}"
    if [ -z "${MAP_NAME}" ]; then
        MAP_NAME="$(basename "${MAP_DIR}")"
    fi
else
    if [ -z "${BAG_PATH}" ]; then
        die "--map-dir is required with --skip-vslam when --bag-path is omitted"
    fi
    if [ -z "${MAP_NAME}" ]; then
        prompt_map_name_interactive
    fi
    BAG_DIR_NAME="$(basename "${BAG_PATH%/}")"
    MAP_DIR="${MAP_ROOT%/}/${BAG_DIR_NAME}/${MAP_NAME}"
fi

if [ -z "${MAP_NAME}" ]; then
    die "map name could not be resolved"
fi
if [[ "${MAP_NAME}" == *"/"* ]]; then
    die "map name must not contain '/'"
fi
if [ -n "${REFERENCE_YAML_PATH}" ] && [ ! -f "${REFERENCE_YAML_PATH}" ]; then
    die "reference map YAML not found: ${REFERENCE_YAML_PATH}"
fi
if [ -n "${ALIGNMENT_PATH}" ] && [ ! -f "${ALIGNMENT_PATH}" ]; then
    die "VSLAM alignment YAML not found: ${ALIGNMENT_PATH}"
fi

mkdir -p "${MAP_DIR}"
MAP_STEM="${MAP_DIR}/${MAP_NAME}"
VSLAM_MAP_DIR="${MAP_DIR}/cuvslam_map"
SNAPSHOT_PATH="${SNAPSHOT_OVERRIDE_PATH:-${MAP_STEM}_vslam_reference.json}"
LANDMARK_IMAGE_PATH="${MAP_STEM}_vslam_landmarks.png"
LANDMARK_YAML_PATH="${MAP_STEM}_vslam_landmarks.yaml"
HD_MAP_YAML_PATH="${HD_MAP_YAML_OVERRIDE_PATH:-${MAP_STEM}_hd_map.yaml}"
CENTERLINE_CSV_PATH="${CENTERLINE_CSV_OVERRIDE_PATH:-${MAP_STEM}_centerline.csv}"
RACELINE_CSV_PATH="${RACELINE_CSV_OVERRIDE_PATH:-${MAP_STEM}_raceline.csv}"
LINE_PREVIEW_PNG_PATH="${LINE_PREVIEW_PNG_OVERRIDE_PATH:-${MAP_STEM}_lines.png}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
TF_LOG_PATH="/tmp/create_hd_map_vslam_tf_${RUN_STAMP}.log"
VSLAM_LOG_PATH="/tmp/create_hd_map_vslam_${RUN_STAMP}.log"
PLAYER_LOG_PATH="/tmp/create_hd_map_vslam_player_${RUN_STAMP}.log"
SNAPSHOT_LOG_PATH="${MAP_DIR}/hd_map_vslam_snapshot_${RUN_STAMP}.log"
EXPORT_LOG_PATH="${MAP_DIR}/hd_map_landmark_export_${RUN_STAMP}.log"
RACELINE_LOG_PATH="${MAP_DIR}/hd_map_raceline_${RUN_STAMP}.log"
LINE_PREVIEW_LOG_PATH="${MAP_DIR}/hd_map_line_preview_${RUN_STAMP}.log"

trap cleanup_all EXIT INT TERM

if [ "${SKIP_VSLAM}" = true ]; then
    if [ ! -f "${SNAPSHOT_PATH}" ]; then
        die "--skip-vslam needs an existing snapshot: ${SNAPSHOT_PATH}"
    fi
    echo "[prep] Reuse VSLAM snapshot: ${SNAPSHOT_PATH}"
else
    run_vslam_snapshot_capture
fi

export_landmark_raster
run_hd_map_editor
run_raceline_export
run_line_preview
print_summary
