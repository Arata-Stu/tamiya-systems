#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_PWD="${PWD}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_vslam_map/process_utils.sh"

BAG_PATH=""
RECORD_ROOT="/record"
OUTPUT_DIR=""
OUTPUT_ROOT="/tmp/vslam_speed_debug"
PLAY_RATE="1.0"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
USE_IMU=false
PLAY_ALL_TOPICS=false
USE_IMAGE_PREPROCESSORS="true"
CAMERA_CONTAINER_NAME="speed_vslam_container_$$"
ODOM_TOPIC="/visual_slam/tracking/odometry"
PLOT_SCRIPT_PATH=""

OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false
CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_CANDIDATES=()

usage() {
    cat <<'EOF'
Usage:
  analyze_vslam_speed_from_bag.sh [OPTIONS]

Options:
  --bag-path DIR              input camera rosbag2 directory
  --record-root DIR           rosbag search root for interactive selection (default: /record)
  --output-dir DIR            output directory. Default: /tmp/vslam_speed_debug/<bag>_<timestamp>
  --rate RATE                 ros2 bag play rate (default: 1.0)
  --image-width PX            offline VSLAM image width (default: 424)
  --image-height PX           offline VSLAM image height (default: 240)
  --use-image-preprocessors true|false
                              pass through to vslam.launch.xml (default: true)
  --with-imu                  replay /camera/imu too
  --play-all-topics           replay the entire source bag instead of filtered camera topics
  --odom-topic TOPIC          VSLAM odom topic to record/analyze
  --plot-script PATH          explicit plot_odom_speed.py path
  -h, --help                  show this help

Outputs:
  <output-dir>/odom_bag/
  <output-dir>/vslam_speed.csv
  <output-dir>/vslam_speed.png
  <output-dir>/vslam_speed_summary.json
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
    if [[ ${nounset_was_enabled} -eq 1 ]]; then
        set -u
    fi
}

source_setup_if_available() {
    if [ -n "${CREATE_VSLAM_SPEED_SETUP:-}" ]; then
        source_setup_script "${CREATE_VSLAM_SPEED_SETUP}"
    elif [ -f "/workspaces/install/setup.bash" ]; then
        source_setup_script "/workspaces/install/setup.bash"
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
            read -r -p "速度解析する camera rosbag を番号で選択 (1-${#ROSBAG_CANDIDATES[@]}): " choice
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

resolve_plot_script() {
    local candidate
    if [ -n "${PLOT_SCRIPT_PATH}" ]; then
        [ -f "${PLOT_SCRIPT_PATH}" ] && echo "${PLOT_SCRIPT_PATH}" && return 0
        return 1
    fi

    for candidate in \
        "${CREATE_VSLAM_SPEED_PYTHON_WS_ROOT:-}/data_analysis/plot_odom_speed.py" \
        "/python_ws/data_analysis/plot_odom_speed.py" \
        "${SCRIPT_DIR}/../python_ws/data_analysis/plot_odom_speed.py" \
        "/workspaces/python_ws/data_analysis/plot_odom_speed.py" \
        "/workspace/python_ws/data_analysis/plot_odom_speed.py" \
        "${INITIAL_PWD}/python_ws/data_analysis/plot_odom_speed.py"; do
        [ -n "${candidate}" ] || continue
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

wait_for_service() {
    local service_name="$1"
    local timeout_sec="$2"
    local count=0

    while [ "${count}" -lt "${timeout_sec}" ]; do
        if ros2 service list | grep -Fxq "${service_name}"; then
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    return 1
}

cleanup_all() {
    stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
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
        --output-dir)
            OUTPUT_DIR="$2"
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
            USE_IMAGE_PREPROCESSORS="$2"
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
        --odom-topic)
            ODOM_TOPIC="$2"
            shift 2
            ;;
        --plot-script)
            PLOT_SCRIPT_PATH="$2"
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

case "${USE_IMAGE_PREPROCESSORS}" in
    true|false)
        ;;
    *)
        die "--use-image-preprocessors must be true or false"
        ;;
esac

source_setup_if_available

if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi

BAG_PATH="${BAG_PATH%/}"
[ -d "${BAG_PATH}" ] || die "Bag directory not found: ${BAG_PATH}"
[ -f "${BAG_PATH}/metadata.yaml" ] || die "metadata.yaml not found: ${BAG_PATH}"

BAG_NAME="$(basename "${BAG_PATH}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${OUTPUT_ROOT%/}/${BAG_NAME}_${TIMESTAMP}"
fi
ODOM_BAG_DIR="${OUTPUT_DIR%/}/odom_bag"
TF_LOG_PATH="${OUTPUT_DIR%/}/offline_tf.log"
VSLAM_LOG_PATH="${OUTPUT_DIR%/}/vslam.log"
RECORDER_LOG_PATH="${OUTPUT_DIR%/}/odom_record.log"
PLAYER_LOG_PATH="${OUTPUT_DIR%/}/bag_play.log"
CSV_PATH="${OUTPUT_DIR%/}/vslam_speed.csv"
PLOT_PATH="${OUTPUT_DIR%/}/vslam_speed.png"
SUMMARY_JSON_PATH="${OUTPUT_DIR%/}/vslam_speed_summary.json"

[ ! -e "${ODOM_BAG_DIR}" ] || die "Output odom bag already exists: ${ODOM_BAG_DIR}"
mkdir -p "${OUTPUT_DIR}"

PLOT_SCRIPT="$(resolve_plot_script)" || die "plot_odom_speed.py was not found"

trap cleanup_all EXIT INT TERM

echo "[1/4] Launch offline TF and VSLAM"
echo "  - output: ${OUTPUT_DIR}"
echo "  - source bag: ${BAG_PATH}"
echo "  - logs: ${TF_LOG_PATH}, ${VSLAM_LOG_PATH}"

launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
    ros2 launch system_launch offline_sensor_tf.launch.xml \
    > "${TF_LOG_PATH}" 2>&1

sleep 2

launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
    ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

sleep 2

launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
    ros2 launch system_launch vslam.launch.xml \
    "use_sim_time:=true" \
    "image_width:=${IMAGE_WIDTH}" \
    "image_height:=${IMAGE_HEIGHT}" \
    "camera_container_name:=${CAMERA_CONTAINER_NAME}" \
    "vslam_map_frame:=map" \
    "vslam_map_parent_frame:=map" \
    "publish_vslam_map_identity_tf:=false" \
    "use_image_preprocessors:=${USE_IMAGE_PREPROCESSORS}" \
    "enable_localization_and_mapping:=true" \
    > "${VSLAM_LOG_PATH}" 2>&1

echo "[2/4] Wait for VSLAM"
if ! wait_for_service "/visual_slam/save_map" 60; then
    die "Visual SLAM service did not become ready. Check ${VSLAM_LOG_PATH}"
fi

echo "[3/4] Record VSLAM odometry while replaying camera bag"
launch_background_process "RECORDER_PID" "RECORDER_USES_SETSID" \
    ros2 bag record \
    -o "${ODOM_BAG_DIR}" \
    "${ODOM_TOPIC}" \
    > "${RECORDER_LOG_PATH}" 2>&1

sleep 2

if [ "${PLAY_ALL_TOPICS}" = true ]; then
    echo "  - replay all topics"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" > "${PLAYER_LOG_PATH}" 2>&1
else
    PLAY_TOPICS=(
        /tf
        /tf_static
        /camera/left/image_raw
        /camera/left/camera_info
        /camera/right/image_raw
        /camera/right/camera_info
    )
    if [ "${USE_IMU}" = true ]; then
        PLAY_TOPICS+=(/camera/imu)
    fi
    echo "  - replay topics: ${PLAY_TOPICS[*]}"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}" > "${PLAYER_LOG_PATH}" 2>&1
fi

sleep 2
stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"

[ -f "${ODOM_BAG_DIR}/metadata.yaml" ] || die "VSLAM odom bag was not written. Check ${RECORDER_LOG_PATH} and ${VSLAM_LOG_PATH}"

echo "[4/4] Analyze speed timeseries"
python3 "${PLOT_SCRIPT}" \
    --bag "${ODOM_BAG_DIR}" \
    --odom-topic "${ODOM_TOPIC}" \
    --csv "${CSV_PATH}" \
    --plot "${PLOT_PATH}" \
    --summary-json "${SUMMARY_JSON_PATH}"

trap - EXIT INT TERM

echo ""
echo "VSLAM speed debug outputs:"
echo "  - directory    : ${OUTPUT_DIR}"
echo "  - odom bag     : ${ODOM_BAG_DIR}"
echo "  - speed CSV    : ${CSV_PATH}"
echo "  - speed plot   : ${PLOT_PATH}"
echo "  - summary JSON : ${SUMMARY_JSON_PATH}"
