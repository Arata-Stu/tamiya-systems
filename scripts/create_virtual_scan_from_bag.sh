#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_vslam_map/process_utils.sh"

BAG_PATH=""
RECORD_ROOT="/record"
MAP_DIR=""
HD_MAP_YAML=""
VSLAM_MAP_DIR=""
ALLOW_NO_VSLAM_MAP=false
OUTPUT_DIR=""
OUTPUT_ROOT="/record/virtual_scan"
PLAY_RATE="1.0"
PLAY_ALL_TOPICS=false
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
USE_IMAGE_PREPROCESSORS="true"
USE_IMU=false
OFFLINE_TF_MODE="auto"
INCLUDE_SOURCE_SCAN=true
ENABLE_DEBUG_MARKERS=false
SOURCE_SCAN_TOPIC="/scan"
INCLUDE_CMD=true
CMD_TOPIC="/jetracer/cmd_drive"
VIRTUAL_SCAN_TOPIC="/virtual_scan"
VIRTUAL_SCAN_DEBUG_TOPIC="/virtual_scan/debug_markers"
VIRTUAL_SCAN_PARAM=""
ODOM_TOPIC="/visual_slam/tracking/odometry"
LOCALIZE_ON_STARTUP="auto"
PUBLISH_MAP_TO_ODOM_TF="auto"
CAMERA_CONTAINER_NAME="offline_virtual_scan_camera_$$"
VIRTUAL_SCAN_CONTAINER_NAME="offline_virtual_scan_container_$$"

OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false
CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
VIRTUAL_SCAN_LAUNCH_PID=""
VIRTUAL_SCAN_LAUNCH_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_CANDIDATES=()
BAG_TOPICS=()

usage() {
    cat <<'EOF'
Usage:
  create_virtual_scan_from_bag.sh [OPTIONS]

Purpose:
  Replay a camera/scan rosbag offline, run VSLAM, generate /virtual_scan from
  an HD map, and record the generated topics into a new rosbag.

Required:
  --bag-path DIR              input rosbag2 directory
  --map-dir DIR               map bundle directory. Defaults:
                              <map-dir>/<basename>_hd_map.yaml
                              <map-dir>/cuvslam_map

Options:
  --record-root DIR           rosbag search root for interactive selection (default: /record)
  --hd-map-yaml PATH          explicit HD map YAML path
  --vslam-map-dir DIR         explicit saved cuVSLAM map directory
  --allow-no-vslam-map        run VSLAM without loading a saved map
  --output-dir DIR            output directory. Default: /record/virtual_scan/<bag>_<map>_<timestamp>
  --output-root DIR           root used by default --output-dir (default: /record/virtual_scan)
  --rate RATE                 ros2 bag play rate (default: 1.0)
  --image-width PX            offline VSLAM image width (default: 424)
  --image-height PX           offline VSLAM image height (default: 240)
  --use-image-preprocessors true|false
                              pass through to vslam.launch.xml (default: true)
  --with-imu                  replay /camera/imu when present
  --play-all-topics           replay the entire source bag instead of filtered camera/scan/TF topics
  --launch-offline-tf         publish fallback base_link->laser/camera TFs
  --no-offline-tf             never publish fallback TFs
  --source-scan-topic TOPIC   source scan topic to replay/record (default: /scan)
  --no-source-scan            do not record/replay the source scan topic
  --cmd-topic TOPIC           command topic used as training label (default: /jetracer/cmd_drive)
  --no-cmd                    do not record/replay the command topic
  --virtual-scan-topic TOPIC  generated LaserScan topic (default: /virtual_scan)
  --virtual-scan-param PATH   explicit hd_map_virtual_scan param YAML
  --odom-topic TOPIC          VSLAM odometry topic (default: /visual_slam/tracking/odometry)
  --debug-markers             record virtual scan debug markers too when the param YAML publishes them
  --localize-on-startup true|false
                              default: true when a saved VSLAM map is loaded
  --publish-map-to-odom-tf true|false
                              default: true when a saved VSLAM map is loaded
  -h, --help                  show this help

Outputs:
  <output-dir>/virtual_scan_bag/
  <output-dir>/*.log
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

source_setup_script() {
    local setup_path="$1"
    local nounset_was_enabled=0

    if [[ $- == *u* ]]; then
        nounset_was_enabled=1
        set +u
    fi

    # shellcheck source=/dev/null
    source "${setup_path}"

    if [ "${nounset_was_enabled}" -eq 1 ]; then
        set -u
    fi
}

source_setup_if_available() {
    if [ -n "${CREATE_VIRTUAL_SCAN_SETUP:-}" ]; then
        source_setup_script "${CREATE_VIRTUAL_SCAN_SETUP}"
    elif [ -f "/workspaces/install/setup.bash" ]; then
        source_setup_script "/workspaces/install/setup.bash"
    elif [ -f "${SCRIPT_DIR}/../install/setup.bash" ]; then
        source_setup_script "${SCRIPT_DIR}/../install/setup.bash"
    elif [ -f "install/setup.bash" ]; then
        source_setup_script "install/setup.bash"
    fi
}

discover_rosbag_candidates() {
    local search_root="$1"
    local metadata_path
    local -a discovered_dirs=()

    ROSBAG_CANDIDATES=()
    [ -d "${search_root}" ] || return 1

    while IFS= read -r -d '' metadata_path; do
        discovered_dirs+=("$(dirname "${metadata_path}")")
    done < <(find "${search_root}" -type f -name metadata.yaml -print0 2>/dev/null)

    [ "${#discovered_dirs[@]}" -gt 0 ] || return 1

    while IFS= read -r dir; do
        [ -n "${dir}" ] && ROSBAG_CANDIDATES+=("${dir}")
    done < <(printf '%s\n' "${discovered_dirs[@]}" | sort -u)

    [ "${#ROSBAG_CANDIDATES[@]}" -gt 0 ]
}

select_rosbag_path_interactive() {
    local choice
    local i

    if discover_rosbag_candidates "${RECORD_ROOT}"; then
        echo ""
        echo "metadata.yaml を検出した rosbag2 ディレクトリ:"
        for i in "${!ROSBAG_CANDIDATES[@]}"; do
            printf "  %2d) %s\n" "$((i + 1))" "${ROSBAG_CANDIDATES[$i]}"
        done
        echo ""
        while :; do
            read -r -p "virtual scan を生成する source bag を番号で選択 (1-${#ROSBAG_CANDIDATES[@]}): " choice
            if [[ "${choice}" =~ ^[0-9]+$ ]] && \
               [ "${choice}" -ge 1 ] && \
               [ "${choice}" -le "${#ROSBAG_CANDIDATES[@]}" ]; then
                BAG_PATH="${ROSBAG_CANDIDATES[$((choice - 1))]}"
                return 0
            fi
            echo "無効な入力です。番号で選択してください。"
        done
    fi

    echo "Warning: ${RECORD_ROOT} 配下で metadata.yaml を持つ rosbag2 を見つけられませんでした。" >&2
    while :; do
        read -r -p "rosbag2 ディレクトリを直接入力してください: " BAG_PATH
        if [ -d "${BAG_PATH}" ] && [ -f "${BAG_PATH%/}/metadata.yaml" ]; then
            BAG_PATH="${BAG_PATH%/}"
            return 0
        fi
        echo "metadata.yaml が見つからないため再入力してください。"
    done
}

normalize_bool() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES|y|Y|on|ON)
            echo "true"
            ;;
        false|False|FALSE|0|no|No|NO|n|N|off|OFF)
            echo "false"
            ;;
        *)
            die "Invalid boolean value: $1"
            ;;
    esac
}

array_contains() {
    local needle="$1"
    local item
    shift

    for item in "$@"; do
        [ "${item}" = "${needle}" ] && return 0
    done
    return 1
}

add_unique_bag_topic() {
    local topic="$1"

    if ! array_contains "${topic}" "${BAG_TOPICS[@]}"; then
        BAG_TOPICS+=("${topic}")
    fi
}

add_unique_play_topic() {
    local topic="$1"

    if ! array_contains "${topic}" "${PLAY_TOPICS[@]}"; then
        PLAY_TOPICS+=("${topic}")
    fi
}

discover_bag_topics() {
    local metadata_path="$1"
    local topic

    BAG_TOPICS=()
    while IFS= read -r topic; do
        [ -n "${topic}" ] || continue
        add_unique_bag_topic "${topic}"
    done < <(
        awk '/^[[:space:]]+name:[[:space:]]/ { name=$2; gsub(/["\047]/, "", name); print name }' "${metadata_path}"
    )
}

bag_has_topic() {
    local topic="$1"
    array_contains "${topic}" "${BAG_TOPICS[@]}"
}

add_play_topic_if_present() {
    local topic="$1"

    if bag_has_topic "${topic}"; then
        add_unique_play_topic "${topic}"
    fi
}

wait_for_topic() {
    local topic_name="$1"
    local timeout_sec="$2"
    local count=0

    while [ "${count}" -lt "${timeout_sec}" ]; do
        if ros2 topic list | grep -Fxq "${topic_name}"; then
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    return 1
}

cleanup_all() {
    stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
    stop_background_process "VIRTUAL_SCAN_LAUNCH_PID" "VIRTUAL_SCAN_LAUNCH_USES_SETSID"
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

derive_map_paths() {
    local map_base

    if [ -n "${MAP_DIR}" ]; then
        MAP_DIR="${MAP_DIR%/}"
        map_base="$(basename "${MAP_DIR}")"

        if [ -z "${HD_MAP_YAML}" ]; then
            HD_MAP_YAML="${MAP_DIR}/${map_base}_hd_map.yaml"
        fi

        if [ -z "${VSLAM_MAP_DIR}" ] && [ -d "${MAP_DIR}/cuvslam_map" ]; then
            VSLAM_MAP_DIR="${MAP_DIR}/cuvslam_map"
        fi
    fi
}

resolve_output_dir() {
    local bag_name="$1"
    local map_label="$2"
    local timestamp

    timestamp="$(date +%Y%m%d_%H%M%S)"

    if [ -z "${map_label}" ]; then
        map_label="no_map"
    fi

    if [ -z "${OUTPUT_DIR}" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT%/}/${bag_name}_${map_label}_${timestamp}"
    fi
}

build_filtered_play_topics() {
    PLAY_TOPICS=()

    add_play_topic_if_present "/clock"
    add_play_topic_if_present "/tf"
    add_play_topic_if_present "/tf_static"

    add_play_topic_if_present "/camera/left/image_raw"
    add_play_topic_if_present "/camera/left/image_rect"
    add_play_topic_if_present "/camera/left/image_rect_raw"
    add_play_topic_if_present "/camera/left/image_rect_mono"
    add_play_topic_if_present "/camera/left/camera_info"
    add_play_topic_if_present "/camera/left/camera_info_rect"
    add_play_topic_if_present "/camera/right/image_raw"
    add_play_topic_if_present "/camera/right/image_rect"
    add_play_topic_if_present "/camera/right/image_rect_raw"
    add_play_topic_if_present "/camera/right/image_rect_mono"
    add_play_topic_if_present "/camera/right/camera_info"
    add_play_topic_if_present "/camera/right/camera_info_rect"

    if [ "${USE_IMU}" = true ]; then
        add_play_topic_if_present "/camera/imu"
    fi

    if [ "${INCLUDE_SOURCE_SCAN}" = true ]; then
        add_play_topic_if_present "${SOURCE_SCAN_TOPIC}"
    fi

    if [ "${INCLUDE_CMD}" = true ]; then
        add_play_topic_if_present "${CMD_TOPIC}"
    fi

    [ "${#PLAY_TOPICS[@]}" -gt 0 ] || die "No playable camera/scan/TF topics were found in ${BAG_PATH}/metadata.yaml"
}

while (($#)); do
    case "$1" in
        --bag-path)
            BAG_PATH="$2"
            shift 2
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --map-dir)
            MAP_DIR="$2"
            shift 2
            ;;
        --hd-map-yaml)
            HD_MAP_YAML="$2"
            shift 2
            ;;
        --vslam-map-dir)
            VSLAM_MAP_DIR="$2"
            shift 2
            ;;
        --allow-no-vslam-map)
            ALLOW_NO_VSLAM_MAP=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
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
        --use-image-preprocessors)
            USE_IMAGE_PREPROCESSORS="$(normalize_bool "$2")"
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
        --launch-offline-tf)
            OFFLINE_TF_MODE="true"
            shift
            ;;
        --no-offline-tf)
            OFFLINE_TF_MODE="false"
            shift
            ;;
        --source-scan-topic)
            SOURCE_SCAN_TOPIC="$2"
            shift 2
            ;;
        --no-source-scan)
            INCLUDE_SOURCE_SCAN=false
            shift
            ;;
        --cmd-topic)
            CMD_TOPIC="$2"
            shift 2
            ;;
        --no-cmd)
            INCLUDE_CMD=false
            shift
            ;;
        --virtual-scan-topic)
            VIRTUAL_SCAN_TOPIC="$2"
            shift 2
            ;;
        --virtual-scan-param)
            VIRTUAL_SCAN_PARAM="$2"
            shift 2
            ;;
        --odom-topic)
            ODOM_TOPIC="$2"
            shift 2
            ;;
        --debug-markers)
            ENABLE_DEBUG_MARKERS=true
            shift
            ;;
        --localize-on-startup)
            LOCALIZE_ON_STARTUP="$(normalize_bool "$2")"
            shift 2
            ;;
        --publish-map-to-odom-tf)
            PUBLISH_MAP_TO_ODOM_TF="$(normalize_bool "$2")"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            echo "Positional arguments are not supported: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

source_setup_if_available

if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi

BAG_PATH="${BAG_PATH%/}"
[ -d "${BAG_PATH}" ] || die "Bag directory not found: ${BAG_PATH}"
[ -f "${BAG_PATH}/metadata.yaml" ] || die "metadata.yaml not found: ${BAG_PATH}"

derive_map_paths

[ -n "${HD_MAP_YAML}" ] || die "--map-dir or --hd-map-yaml is required"
[ -f "${HD_MAP_YAML}" ] || die "HD map YAML not found: ${HD_MAP_YAML}"

if [ -n "${VIRTUAL_SCAN_PARAM}" ]; then
    [ -f "${VIRTUAL_SCAN_PARAM}" ] || die "Virtual scan param YAML not found: ${VIRTUAL_SCAN_PARAM}"
fi

if [ -n "${VSLAM_MAP_DIR}" ]; then
    VSLAM_MAP_DIR="${VSLAM_MAP_DIR%/}"
    [ -d "${VSLAM_MAP_DIR}" ] || die "VSLAM map directory not found: ${VSLAM_MAP_DIR}"
elif [ "${ALLOW_NO_VSLAM_MAP}" != true ]; then
    die "Saved cuVSLAM map was not found. Pass --vslam-map-dir or --allow-no-vslam-map."
fi

if [ "${LOCALIZE_ON_STARTUP}" = "auto" ]; then
    if [ -n "${VSLAM_MAP_DIR}" ]; then
        LOCALIZE_ON_STARTUP="true"
    else
        LOCALIZE_ON_STARTUP="false"
    fi
fi

if [ "${PUBLISH_MAP_TO_ODOM_TF}" = "auto" ]; then
    if [ -n "${VSLAM_MAP_DIR}" ]; then
        PUBLISH_MAP_TO_ODOM_TF="true"
    else
        PUBLISH_MAP_TO_ODOM_TF="false"
    fi
fi

discover_bag_topics "${BAG_PATH}/metadata.yaml"

if [ "${OFFLINE_TF_MODE}" = "auto" ]; then
    if bag_has_topic "/tf" || bag_has_topic "/tf_static"; then
        LAUNCH_OFFLINE_TF=false
    else
        LAUNCH_OFFLINE_TF=true
    fi
elif [ "${OFFLINE_TF_MODE}" = "true" ]; then
    LAUNCH_OFFLINE_TF=true
else
    LAUNCH_OFFLINE_TF=false
fi

BAG_NAME="$(basename "${BAG_PATH}")"
MAP_LABEL="$(basename "$(dirname "${HD_MAP_YAML}")")"
resolve_output_dir "${BAG_NAME}" "${MAP_LABEL}"

OUTPUT_DIR="${OUTPUT_DIR%/}"
OUTPUT_BAG_DIR="${OUTPUT_DIR}/virtual_scan_bag"
TF_LOG_PATH="${OUTPUT_DIR}/offline_tf.log"
VSLAM_LOG_PATH="${OUTPUT_DIR}/vslam.log"
VIRTUAL_SCAN_LOG_PATH="${OUTPUT_DIR}/virtual_scan.log"
RECORDER_LOG_PATH="${OUTPUT_DIR}/virtual_scan_record.log"
PLAYER_LOG_PATH="${OUTPUT_DIR}/bag_play.log"

[ ! -e "${OUTPUT_BAG_DIR}" ] || die "Output bag already exists: ${OUTPUT_BAG_DIR}"
mkdir -p "${OUTPUT_DIR}"

trap cleanup_all EXIT INT TERM

echo "[1/5] Launch offline VSLAM and virtual scan"
echo "  - source bag : ${BAG_PATH}"
echo "  - HD map     : ${HD_MAP_YAML}"
if [ "${INCLUDE_CMD}" = true ]; then
    echo "  - cmd topic  : ${CMD_TOPIC}"
else
    echo "  - cmd topic  : disabled"
fi
if [ -n "${VIRTUAL_SCAN_PARAM}" ]; then
    echo "  - scan param : ${VIRTUAL_SCAN_PARAM}"
fi
if [ -n "${VSLAM_MAP_DIR}" ]; then
    echo "  - VSLAM map  : ${VSLAM_MAP_DIR}"
else
    echo "  - VSLAM map  : none (--allow-no-vslam-map)"
fi
echo "  - output     : ${OUTPUT_DIR}"
echo "  - fallback TF: ${LAUNCH_OFFLINE_TF}"

if [ "${LAUNCH_OFFLINE_TF}" = true ]; then
    launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
        ros2 launch system_launch offline_sensor_tf.launch.xml \
        > "${TF_LOG_PATH}" 2>&1

    sleep 2
fi

launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
    ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

sleep 2

VSLAM_ARGS=(
    "use_sim_time:=true"
    "image_width:=${IMAGE_WIDTH}"
    "image_height:=${IMAGE_HEIGHT}"
    "camera_container_name:=${CAMERA_CONTAINER_NAME}"
    "vslam_map_frame:=map"
    "vslam_map_parent_frame:=map"
    "publish_vslam_map_identity_tf:=false"
    "use_image_preprocessors:=${USE_IMAGE_PREPROCESSORS}"
    "enable_localization_and_mapping:=true"
    "localize_on_startup:=${LOCALIZE_ON_STARTUP}"
    "publish_map_to_odom_tf:=${PUBLISH_MAP_TO_ODOM_TF}"
)

if [ -n "${VSLAM_MAP_DIR}" ]; then
    VSLAM_ARGS+=("load_map_path:=${VSLAM_MAP_DIR}")
fi

launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
    ros2 launch system_launch vslam.launch.xml \
    "${VSLAM_ARGS[@]}" \
    > "${VSLAM_LOG_PATH}" 2>&1

sleep 2

VIRTUAL_SCAN_ARGS=(
    "container_name:=${VIRTUAL_SCAN_CONTAINER_NAME}"
    "create_container:=true"
    "hd_map_yaml_path:=${HD_MAP_YAML}"
    "use_sim_time:=true"
    "odometry_topic:=${ODOM_TOPIC}"
    "scan_topic:=${VIRTUAL_SCAN_TOPIC}"
    "debug_markers_topic:=${VIRTUAL_SCAN_DEBUG_TOPIC}"
)

if [ -n "${VIRTUAL_SCAN_PARAM}" ]; then
    VIRTUAL_SCAN_ARGS+=("param_file:=${VIRTUAL_SCAN_PARAM}")
fi

launch_background_process "VIRTUAL_SCAN_LAUNCH_PID" "VIRTUAL_SCAN_LAUNCH_USES_SETSID" \
    ros2 launch hd_map_virtual_scan hd_map_virtual_scan.launch.xml \
    "${VIRTUAL_SCAN_ARGS[@]}" \
    > "${VIRTUAL_SCAN_LOG_PATH}" 2>&1

echo "[2/5] Wait for nodes"
if ! wait_for_service "/visual_slam/save_map" 60; then
    die "Visual SLAM service did not become ready. Check ${VSLAM_LOG_PATH}"
fi

if ! wait_for_topic "${VIRTUAL_SCAN_TOPIC}" 30; then
    die "Virtual scan topic did not appear. Check ${VIRTUAL_SCAN_LOG_PATH}"
fi

echo "[3/5] Start rosbag record"
RECORD_TOPICS=(
    "${VIRTUAL_SCAN_TOPIC}"
    "${ODOM_TOPIC}"
    "/visual_slam/tracking/slam_path"
    "/tf"
    "/tf_static"
    "/clock"
)

if [ "${INCLUDE_SOURCE_SCAN}" = true ]; then
    RECORD_TOPICS+=("${SOURCE_SCAN_TOPIC}")
fi

if [ "${INCLUDE_CMD}" = true ]; then
    RECORD_TOPICS+=("${CMD_TOPIC}")
fi

if [ "${ENABLE_DEBUG_MARKERS}" = true ]; then
    RECORD_TOPICS+=("${VIRTUAL_SCAN_DEBUG_TOPIC}")
fi

echo "  - output bag: ${OUTPUT_BAG_DIR}"
echo "  - topics    : ${RECORD_TOPICS[*]}"
launch_background_process "RECORDER_PID" "RECORDER_USES_SETSID" \
    ros2 bag record \
    -o "${OUTPUT_BAG_DIR}" \
    "${RECORD_TOPICS[@]}" \
    > "${RECORDER_LOG_PATH}" 2>&1

sleep 2

echo "[4/5] Replay source rosbag"
if [ "${PLAY_ALL_TOPICS}" = true ]; then
    echo "  - replay all topics"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" > "${PLAYER_LOG_PATH}" 2>&1
else
    build_filtered_play_topics
    echo "  - replay topics: ${PLAY_TOPICS[*]}"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}" > "${PLAYER_LOG_PATH}" 2>&1
fi

echo "[5/5] Stop recorder and background processes"
sleep 2
stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
stop_background_process "VIRTUAL_SCAN_LAUNCH_PID" "VIRTUAL_SCAN_LAUNCH_USES_SETSID"
stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"

[ -f "${OUTPUT_BAG_DIR}/metadata.yaml" ] || die "Virtual scan bag was not written. Check ${RECORDER_LOG_PATH}, ${VSLAM_LOG_PATH}, ${VIRTUAL_SCAN_LOG_PATH}"

trap - EXIT INT TERM

echo ""
echo "Virtual scan offline generation complete:"
echo "  - output dir : ${OUTPUT_DIR}"
echo "  - output bag : ${OUTPUT_BAG_DIR}"
echo "  - logs       : ${TF_LOG_PATH}, ${VSLAM_LOG_PATH}, ${VIRTUAL_SCAN_LOG_PATH}, ${RECORDER_LOG_PATH}, ${PLAYER_LOG_PATH}"
