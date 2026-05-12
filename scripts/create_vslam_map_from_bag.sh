#!/bin/bash

set -euo pipefail

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
    if [ -n "${CREATE_VSLAM_MAP_SETUP:-}" ]; then
        source_setup_script "${CREATE_VSLAM_MAP_SETUP}"
    elif [ -f "/workspaces/install/setup.bash" ]; then
        source_setup_script "/workspaces/install/setup.bash"
    elif [ -f "install/setup.bash" ]; then
        source_setup_script "install/setup.bash"
    fi
}

source_setup_if_available

usage() {
    cat <<'EOF'
Usage:
  create_vslam_map_from_bag.sh [OPTIONS]

Options:
  --mode NAME         default|vslam_map (default: default)
  --bag-path DIR      input rosbag2 directory (skip interactive selection)
  --map-name NAME     output map name (skip interactive prompt)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --record-root DIR   input rosbag search root (default: /record)
  --map-root DIR      output visual map root (default: /map)
  --bag-root DIR      output lightweight bag root (default: /record/2d_input)
  --lightweight-bag-dir DIR
                     explicit output directory for the lightweight 2D input bag
  --image-width PX    camera width for offline vslam launch (default: 424)
  --image-height PX   camera height for offline vslam launch (default: 240)
  --image-fps FPS     camera fps for offline vslam launch (default: 90.0)
  --with-imu          replay /camera/imu as well (default: disabled)
  --play-all-topics   replay the entire source bag instead of filtered topics
  -h, --help          show this help

Outputs:
  /map/<source_bag>/<MAP_NAME>/cuvslam_map/
  /record/2d_input/<source_bag>/<MAP_NAME>_2d_input_<timestamp>/
    - /visual_slam/tracking/odometry
    - /scan
    - /tf
    - /tf_static
EOF
}

PLAY_RATE="1.0"
MODE="default"
RECORD_ROOT="/record"
MAP_ROOT="/map"
BAG_ROOT="/record/2d_input"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
IMAGE_FPS="90.0"
USE_IMU=false
PLAY_ALL_TOPICS=false
CAMERA_CONTAINER_NAME="offline_camera_container_$$"
OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false

CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_CANDIDATES=()

apply_mode() {
    case "$1" in
        default)
            ;;
        vslam_map)
            IMAGE_WIDTH="1280"
            IMAGE_HEIGHT="720"
            IMAGE_FPS="30.0"
            ;;
        *)
            echo "Unknown mode: $1" >&2
            usage
            exit 1
            ;;
    esac
}

while (($#)); do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --bag-path)
            BAG_PATH="$2"
            shift 2
            ;;
        --map-name)
            MAP_NAME="$2"
            shift 2
            ;;
        --rate)
            PLAY_RATE="$2"
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
        --bag-root)
            BAG_ROOT="$2"
            shift 2
            ;;
        --lightweight-bag-dir)
            EXPLICIT_LIGHTWEIGHT_BAG_DIR="$2"
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
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            echo "Positional arguments are not supported: $1" >&2
            usage
            exit 1
            ;;
    esac
done

apply_mode "${MODE}"

BAG_PATH=""
MAP_NAME=""
BAG_DIR_NAME=""
MAP_DIR=""
VSLAM_MAP_DIR=""
LIGHTWEIGHT_BAG_DIR=""
EXPLICIT_LIGHTWEIGHT_BAG_DIR=""
VSLAM_LOG_PATH=""
TF_LOG_PATH=""

discover_rosbag_candidates() {
    local search_root="$1"
    local metadata_path
    local dir
    local -a discovered_dirs=()

    ROSBAG_CANDIDATES=()
    if [ ! -d "${search_root}" ]; then
        return 1
    fi

    while IFS= read -r -d '' metadata_path; do
        discovered_dirs+=("$(dirname "${metadata_path}")")
    done < <(find "${search_root}" -type f -name metadata.yaml -print0 2>/dev/null)

    if [ "${#discovered_dirs[@]}" -eq 0 ]; then
        return 1
    fi

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
            read -r -p "rosbagを番号で選択 (1-${#ROSBAG_CANDIDATES[@]}): " choice
            if [[ "${choice}" =~ ^[0-9]+$ ]] && \
               [ "${choice}" -ge 1 ] && \
               [ "${choice}" -le "${#ROSBAG_CANDIDATES[@]}" ]; then
                BAG_PATH="${ROSBAG_CANDIDATES[$((choice - 1))]}"
                return 0
            fi
            echo "無効な入力です。番号で選択してください。"
        done
    fi

    echo ""
    echo "Warning: ${RECORD_ROOT} 配下で metadata.yaml を持つ rosbag2 を見つけられませんでした。" >&2
    while :; do
        read -r -p "rosbag2ディレクトリを直接入力してください: " BAG_PATH
        if [ -d "${BAG_PATH}" ] && [ -f "${BAG_PATH%/}/metadata.yaml" ]; then
            BAG_PATH="${BAG_PATH%/}"
            return 0
        fi
        echo "metadata.yaml が見つからないため再入力してください。"
    done
}

prompt_map_name_interactive() {
    while :; do
        read -r -p "作成する map 名を入力してください: " MAP_NAME
        MAP_NAME="${MAP_NAME#"${MAP_NAME%%[![:space:]]*}"}"
        MAP_NAME="${MAP_NAME%"${MAP_NAME##*[![:space:]]}"}"
        if [ -z "${MAP_NAME}" ]; then
            echo "map名が空です。"
            continue
        fi
        if [[ "${MAP_NAME}" == *"/"* ]]; then
            echo "map名に '/' は使えません。"
            continue
        fi
        return 0
    done
}

wait_for_service() {
    local service_name="$1"
    local timeout_sec="$2"
    local count=0

    while [ "$count" -lt "$timeout_sec" ]; do
        if ros2 service list | grep -Fxq "$service_name"; then
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    return 1
}

launch_background_process() {
    local pid_var_name="$1"
    local setsid_var_name="$2"
    shift 2

    if command -v setsid >/dev/null 2>&1; then
        printf -v "${setsid_var_name}" '%s' true
        setsid "$@" &
    else
        printf -v "${setsid_var_name}" '%s' false
        "$@" &
    fi

    printf -v "${pid_var_name}" '%s' "$!"
}

kill_pid_gracefully() {
    local pid="$1"
    local kill_group="${2:-false}"
    local target="$pid"

    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    if [ "$kill_group" = true ]; then
        target="-$pid"
    fi

    kill -INT "$target" 2>/dev/null || true
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$target" 2>/dev/null || true
        sleep 1
    fi

    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$target" 2>/dev/null || true
    fi
}

stop_background_process() {
    local pid_var_name="$1"
    local setsid_var_name="$2"
    local pid="${!pid_var_name:-}"
    local use_setsid="${!setsid_var_name:-false}"

    if [ -n "${pid}" ]; then
        kill_pid_gracefully "${pid}" "${use_setsid}"
        wait "${pid}" 2>/dev/null || true
        printf -v "${pid_var_name}" '%s' ""
        printf -v "${setsid_var_name}" '%s' false
    fi
}

cleanup_all() {
    stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

trap cleanup_all EXIT INT TERM

if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi

if [ -z "${MAP_NAME}" ]; then
    prompt_map_name_interactive
fi

BAG_PATH_CLEAN="${BAG_PATH%/}"
BAG_DIR_NAME="$(basename "${BAG_PATH_CLEAN}")"
MAP_DIR="${MAP_ROOT%/}/${BAG_DIR_NAME}/${MAP_NAME}"
VSLAM_MAP_DIR="${MAP_DIR}/cuvslam_map"
if [ -n "${EXPLICIT_LIGHTWEIGHT_BAG_DIR}" ]; then
    LIGHTWEIGHT_BAG_DIR="${EXPLICIT_LIGHTWEIGHT_BAG_DIR%/}"
else
    LIGHTWEIGHT_BAG_DIR="${BAG_ROOT%/}/${BAG_DIR_NAME}/${MAP_NAME}_2d_input_$(date +%Y%m%d_%H%M%S)"
fi
VSLAM_LOG_PATH="/tmp/offline_vslam_record_$(date +%Y%m%d_%H%M%S).log"
TF_LOG_PATH="/tmp/offline_vslam_tf_$(date +%Y%m%d_%H%M%S).log"

if [ ! -d "${BAG_PATH}" ] || [ ! -f "${BAG_PATH}/metadata.yaml" ]; then
    echo "Invalid BAG_PATH: ${BAG_PATH}" >&2
    exit 1
fi

mkdir -p "${MAP_DIR}"
mkdir -p "$(dirname "${LIGHTWEIGHT_BAG_DIR}")"

echo "[1/5] Launch offline TF + vslam (logs: ${TF_LOG_PATH}, ${VSLAM_LOG_PATH})"
launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
    ros2 launch system_launch offline_sensor_tf.launch.xml \
    > "${TF_LOG_PATH}" 2>&1

sleep 2

launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
    ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

sleep 2

launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
    ros2 launch system_launch vslam.launch.xml \
    "image_width:=${IMAGE_WIDTH}" \
    "image_height:=${IMAGE_HEIGHT}" \
    "camera_container_name:=${CAMERA_CONTAINER_NAME}" \
    "enable_localization_and_mapping:=true" \
    "save_map_path:=${VSLAM_MAP_DIR}" \
    > "${VSLAM_LOG_PATH}" 2>&1

echo "[2/5] Wait for visual slam services"
if ! wait_for_service "/visual_slam/save_map" 60; then
    echo "Visual SLAM service not ready. Check log: ${VSLAM_LOG_PATH}" >&2
    exit 1
fi

echo "[3/5] Start lightweight rosbag record"
launch_background_process "RECORDER_PID" "RECORDER_USES_SETSID" \
    ros2 bag record \
    -o "${LIGHTWEIGHT_BAG_DIR}" \
    /visual_slam/tracking/odometry \
    /scan \
    /tf \
    /tf_static

sleep 2

echo "[4/5] Replay source rosbag"
if [ "${PLAY_ALL_TOPICS}" = true ]; then
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}"
else
    PLAY_TOPICS=(
        /scan
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

    echo "  - topics: ${PLAY_TOPICS[*]}"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}"
fi

echo "[5/5] Save visual map and stop processes"
sleep 2
if ! ros2 service call /visual_slam/save_map \
    isaac_ros_visual_slam_interfaces/srv/FilePath \
    "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null; then
    echo "Warning: visual_slam/save_map failed." >&2
fi

cleanup_all

echo ""
echo "✅ Visual map and lightweight bag generated:"
echo "  - visual map: ${VSLAM_MAP_DIR}"
echo "  - 2d input bag: ${LIGHTWEIGHT_BAG_DIR}"
