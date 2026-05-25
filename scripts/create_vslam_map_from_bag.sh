#!/bin/bash


# --- Source Library Modules ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for lib in "${SCRIPT_DIR}/lib/create_vslam_map/"*.sh; do
    source "${lib}"
done
# ------------------------------

set -euo pipefail



source_setup_if_available


PLAY_RATE="1.0"
MODE="default"
RECORD_ROOT="/record"
MAP_ROOT="/map"
BAG_ROOT="/record/2d_input"
# ==============================================================================
# カメラ解像度設定
# デフォルト値はここで変更できます。--image-width / --image-height で都度上書きも可能。
# RealSense D435 stereo gray (infra) のサポート解像度例:
#   424x240 (デフォルト, 最大 90fps)
#   640x480 (最大 90fps)
#   848x480 (最大 90fps)
#   1280x720 (最大 30fps)
# ==============================================================================
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
IMAGE_FPS="90.0"
USE_IMU=false
PLAY_ALL_TOPICS=false
CAMERA_CONTAINER_NAME="offline_camera_container_$$"
OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false

BAG_PATH=""
MAP_NAME=""
BAG_DIR_NAME=""
MAP_DIR=""
VSLAM_MAP_DIR=""
LIGHTWEIGHT_BAG_DIR=""
EXPLICIT_LIGHTWEIGHT_BAG_DIR=""
VSLAM_LOG_PATH=""
TF_LOG_PATH=""

CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_CANDIDATES=()


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
    "use_sim_time:=true" \
    "image_width:=${IMAGE_WIDTH}" \
    "image_height:=${IMAGE_HEIGHT}" \
    "camera_container_name:=${CAMERA_CONTAINER_NAME}" \
    "vslam_map_frame:=map" \
    "vslam_map_parent_frame:=map" \
    "publish_vslam_map_identity_tf:=false" \
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
