#!/bin/bash


# --- Source Library Modules ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for lib in "${SCRIPT_DIR}/lib/create_2d_map/"*.sh; do
    source "${lib}"
done
# ------------------------------

set -euo pipefail

INITIAL_PWD="${PWD}"


SCRIPT_PATH="$(resolve_real_path "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"




REPO_ROOT="$(resolve_repo_root)"
SYSTEM_LAUNCH_SOURCE_SHARE=""
SYSTEM_LAUNCH_CMD=()








source_setup_if_available

SYSTEM_LAUNCH_SOURCE_SHARE="$(resolve_system_launch_source_share || true)"



# ==========================================
# Default settings
# ==========================================
SCAN_TOPIC="/scan"
PLAY_RATE="1.0"
ODOM_TOPIC=""
CONFIG_BASENAME="cartographer_2d.lua"
DEFAULT_VSLAM_ODOM_TOPIC="/visual_slam/tracking/odometry"
MODE="no_odom_offline_vslam"
MODE_SET_BY_USER=false
CARTOGRAPHER_USE_ODOM=false
RUN_VSLAM_OVERRIDE="auto"
ODOM_READY_WAIT_ENABLED=true
ODOM_READY_WINDOW="10"
ODOM_READY_MIN_RATE_HZ=""
ODOM_READY_TIMEOUT_SEC="45"
VSLAM_VIS_ENABLED=false
PLAY_ALL_TOPICS=false
ENABLE_CENTERLINE=true
ENABLE_RACELINE=true
ENABLE_LINE_PREVIEW=true
MAP_EDIT_MODE="auto"
MAP_EDIT_ENABLED=false
MAP_EDIT_SCRIPT_PATH=""
MAP_EDIT_OUTPUT_PATH=""
VSLAM_LANDMARK_TRACE_MODE="never"
VSLAM_LANDMARK_TRACE_ENABLED=false
VSLAM_LANDMARK_TRACE_COMPLETED=false
VSLAM_LANDMARK_EXPORT_SCRIPT_PATH=""
VSLAM_MAP_ALIGNMENT_CONFIG_PATH=""
VSLAM_MAP_ALIGNMENT_CONFIG_SET_BY_USER=false
VSLAM_REFERENCE_SNAPSHOT_PATH=""
PREPARE_VSLAM_MAP_ALIGNMENT=false
VSLAM_LIVE_ALIGNMENT_MODE="auto"
VSLAM_LIVE_ALIGNMENT_ENABLED=false
VSLAM_LIVE_ALIGNMENT_RVIZ_PATH=""
VSLAM_LANDMARK_IMAGE_PATH=""
VSLAM_LANDMARK_YAML_PATH=""
VSLAM_LANDMARK_REFERENCE_YAML_PATH=""
VSLAM_LANDMARK_TARGET_FRAME="map"
VSLAM_TRACE_OUTPUT_PATH=""
VSLAM_LANDMARK_TOPIC="/visual_slam/vis/landmarks_cloud"
VSLAM_LANDMARK_PATH_TOPIC="/visual_slam/tracking/slam_path"
VSLAM_LANDMARK_EXPORT_TIMEOUT_SEC="120"
CENTERLINE_DEBUG=true
CENTERLINE_DEBUG_DIR=""
CENTERLINE_SCRIPT_PATH=""
CENTERLINE_PRESET="default"
CENTERLINE_DIRECTION="forward"
RACELINE_SCRIPT_PATH=""
RACELINE_PRESET="race-stacks"
RACELINE_BACKEND="global-opt"
RACELINE_OPT_TYPE="mincurv"
RACELINE_DIRECTION="forward"
GLOBAL_OPTIMIZER_ROOT=""
LINE_PREVIEW_SCRIPT_PATH=""
RECORD_ROOT="/record"
PIPELINE_MODE="offline"
PIPELINE_MODE_OVERRIDE="auto"
BAG_PATH=""
MAP_NAME=""
ODOM_TOPIC_SET_BY_USER=false
# ==============================================================================
# カメラ解像度設定
# デフォルト値はここで変更できます。--image-width / --image-height で都度上書きも可能。
# launch_system.sh の SENSOR_IMAGE_WIDTH/HEIGHT と必ず共通の値に展開すること。
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
CAMERA_CONTAINER_NAME="offline_camera_container_$$"
VSLAM_MAP_DIR=""
OFFLINE_TF_PID=""
OFFLINE_TF_USES_SETSID=false
ENABLE_SCP=true

DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.98")
DEFAULT_REMOTE_DIR="/home/tamiya/workspaces/tamiya-systems/map/"
# ==========================================

CARTOGRAPHER_PID=""
CARTOGRAPHER_USES_SETSID=false
CAMERA_CONTAINER_PID=""
CAMERA_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_PLAY_PID=""
ROSBAG_PLAY_USES_SETSID=false
LANDMARK_EXPORT_PID=""
LANDMARK_EXPORT_USES_SETSID=false
RVIZ_PID=""
RVIZ_USES_SETSID=false
VSLAM_REFERENCE_RECORDER_PID=""
VSLAM_REFERENCE_RECORDER_USES_SETSID=false
VSLAM_REFERENCE_CAPTURE_EXPECTED=false
VSLAM_REFERENCE_CAPTURE_STARTED=false
VSLAM_REFERENCE_CAPTURE_WAS_STARTED=false
POST_ALIGNMENT_STACK_PID=""
POST_ALIGNMENT_STACK_USES_SETSID=false
POST_ALIGNMENT_REFERENCE_PUBLISHER_PID=""
POST_ALIGNMENT_REFERENCE_PUBLISHER_USES_SETSID=false
BASE_CARTOGRAPHER_PIDS=()
ROSBAG_CANDIDATES=()
SOURCE_PLAY_TOPICS=()
OFFLINE_ODOM_BAG_DIR=""
OFFLINE_ODOM_BAG_CREATED=false
VSLAM_LOCALIZATION_PARAM_PATH=""















while (($#)); do
    case "$1" in
        --mode)
            MODE="$2"
            MODE_SET_BY_USER=true
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
        --scan-topic)
            SCAN_TOPIC="$2"
            shift 2
            ;;
        --rate)
            PLAY_RATE="$2"
            shift 2
            ;;
        --odom-topic)
            ODOM_TOPIC="$2"
            ODOM_TOPIC_SET_BY_USER=true
            shift 2
            ;;
        --odom-ready-window)
            ODOM_READY_WINDOW="$2"
            shift 2
            ;;
        --odom-ready-min-rate)
            ODOM_READY_MIN_RATE_HZ="$2"
            shift 2
            ;;
        --odom-ready-timeout)
            ODOM_READY_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --no-odom-ready-wait)
            ODOM_READY_WAIT_ENABLED=false
            shift
            ;;
        --run-vslam)
            RUN_VSLAM_OVERRIDE="online"
            shift
            ;;
        --no-vslam)
            RUN_VSLAM_OVERRIDE="offline"
            shift
            ;;
        --vslam-vis)
            VSLAM_VIS_ENABLED=true
            shift
            ;;
        --no-vslam-vis)
            VSLAM_VIS_ENABLED=false
            shift
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
        --vslam-map-dir)
            VSLAM_MAP_DIR="$2"
            shift 2
            ;;
        --pipeline-mode)
            PIPELINE_MODE_OVERRIDE="$2"
            shift 2
            ;;
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --no-scp)
            ENABLE_SCP=false
            shift
            ;;
        --no-centerline)
            ENABLE_CENTERLINE=false
            ENABLE_RACELINE=false
            shift
            ;;
        --no-raceline)
            ENABLE_RACELINE=false
            shift
            ;;
        --no-line-preview)
            ENABLE_LINE_PREVIEW=false
            shift
            ;;
        --edit-map)
            MAP_EDIT_MODE="always"
            shift
            ;;
        --no-edit-map)
            MAP_EDIT_MODE="never"
            shift
            ;;
        --map-edit-mode)
            MAP_EDIT_MODE="$2"
            shift 2
            ;;
        --map-edit-script)
            MAP_EDIT_SCRIPT_PATH="$2"
            shift 2
            ;;
        --map-edit-output)
            MAP_EDIT_OUTPUT_PATH="$2"
            shift 2
            ;;
        --trace-vslam-landmarks)
            VSLAM_LANDMARK_TRACE_MODE="always"
            shift
            ;;
        --no-trace-vslam-landmarks)
            VSLAM_LANDMARK_TRACE_MODE="never"
            shift
            ;;
        --vslam-landmark-trace-mode)
            VSLAM_LANDMARK_TRACE_MODE="$2"
            shift 2
            ;;
        --vslam-map-alignment-config)
            VSLAM_MAP_ALIGNMENT_CONFIG_PATH="$2"
            VSLAM_MAP_ALIGNMENT_CONFIG_SET_BY_USER=true
            shift 2
            ;;
        --prepare-vslam-map-alignment)
            PREPARE_VSLAM_MAP_ALIGNMENT=true
            shift
            ;;
        --no-prepare-vslam-map-alignment)
            PREPARE_VSLAM_MAP_ALIGNMENT=false
            shift
            ;;
        --live-vslam-map-align)
            VSLAM_LIVE_ALIGNMENT_MODE="always"
            shift
            ;;
        --no-live-vslam-map-align)
            VSLAM_LIVE_ALIGNMENT_MODE="never"
            shift
            ;;
        --live-vslam-map-align-mode)
            VSLAM_LIVE_ALIGNMENT_MODE="$2"
            shift 2
            ;;
        --live-vslam-map-align-rviz)
            VSLAM_LIVE_ALIGNMENT_RVIZ_PATH="$2"
            shift 2
            ;;
        --vslam-landmark-export-script)
            VSLAM_LANDMARK_EXPORT_SCRIPT_PATH="$2"
            shift 2
            ;;
        --vslam-landmark-image)
            VSLAM_LANDMARK_IMAGE_PATH="$2"
            shift 2
            ;;
        --vslam-landmark-yaml)
            VSLAM_LANDMARK_YAML_PATH="$2"
            shift 2
            ;;
        --vslam-landmark-reference-yaml)
            VSLAM_LANDMARK_REFERENCE_YAML_PATH="$2"
            shift 2
            ;;
        --vslam-landmark-target-frame)
            VSLAM_LANDMARK_TARGET_FRAME="$2"
            shift 2
            ;;
        --vslam-trace-output)
            VSLAM_TRACE_OUTPUT_PATH="$2"
            shift 2
            ;;
        --centerline-debug)
            CENTERLINE_DEBUG=true
            shift
            ;;
        --centerline-debug-dir)
            CENTERLINE_DEBUG=true
            CENTERLINE_DEBUG_DIR="$2"
            shift 2
            ;;
        --centerline-script)
            CENTERLINE_SCRIPT_PATH="$2"
            shift 2
            ;;
        --line-preset)
            CENTERLINE_PRESET="$2"
            RACELINE_PRESET="$2"
            shift 2
            ;;
        --centerline-direction)
            CENTERLINE_DIRECTION="$2"
            shift 2
            ;;
        --raceline-script)
            RACELINE_SCRIPT_PATH="$2"
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
        --raceline-direction)
            RACELINE_DIRECTION="$2"
            shift 2
            ;;
        --optimizer-root)
            GLOBAL_OPTIMIZER_ROOT="$2"
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
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            echo "Positional arguments are not supported: $1" >&2
            echo "rosbag / map名は実行中に対話入力してください。" >&2
            usage
            exit 1
            ;;
    esac
done

if [ "${MODE_SET_BY_USER}" != true ]; then
    prompt_mode_interactive
fi

apply_mode "${MODE}"

case "${PIPELINE_MODE_OVERRIDE}" in
    auto)
        ;;
    offline|online)
        PIPELINE_MODE="${PIPELINE_MODE_OVERRIDE}"
        ;;
    full|fast)
        echo "Legacy --pipeline-mode ${PIPELINE_MODE_OVERRIDE} has been removed. Use one of the explicit --mode presets instead." >&2
        exit 1
        ;;
    *)
        echo "Invalid --pipeline-mode: ${PIPELINE_MODE_OVERRIDE}" >&2
        exit 1
        ;;
esac

case "${VSLAM_LANDMARK_TRACE_MODE}" in
    auto|always|never)
        ;;
    *)
        echo "Invalid --vslam-landmark-trace-mode: ${VSLAM_LANDMARK_TRACE_MODE}" >&2
        exit 1
        ;;
esac

case "${VSLAM_LIVE_ALIGNMENT_MODE}" in
    auto|always|never)
        ;;
    *)
        echo "Invalid --live-vslam-map-align-mode: ${VSLAM_LIVE_ALIGNMENT_MODE}" >&2
        exit 1
        ;;
esac

BAG_DIR_NAME=""
BAG_OUT_DIR=""
OUT_DIR=""
MAP_STEM=""
PBSTREAM_PATH=""
MAP_YAML_PATH=""
MAP_PGM_PATH=""
MAP_PNG_PATH=""
CENTERLINE_OUTPUT_PATH=""
CENTERLINE_DEBUG_PATH=""
CENTERLINE_CREATED=false
RACELINE_OUTPUT_PATH=""
RACELINE_CREATED=false
LINE_PREVIEW_OUTPUT_PATH=""
LINE_PREVIEW_CREATED=false
MAP_LOG_PATH=""
VSLAM_LOG_PATH=""
TF_LOG_PATH=""
OFFLINE_VSLAM_MAP_LOG_PATH=""
OFFLINE_VSLAM_MAP_TF_LOG_PATH=""
OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH=""
OFFLINE_VSLAM_ODOM_LOG_PATH=""
OFFLINE_VSLAM_ODOM_TF_LOG_PATH=""
OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH=""
OFFLINE_VSLAM_ODOM_RECORD_LOG_PATH=""
VSLAM_LANDMARK_LOG_PATH=""
VSLAM_LANDMARK_TF_LOG_PATH=""
VSLAM_LANDMARK_PLAYER_LOG_PATH=""
VSLAM_LANDMARK_EXPORT_LOG_PATH=""

if [ "${ODOM_TOPIC_SET_BY_USER}" = true ]; then
    CARTOGRAPHER_USE_ODOM=true
fi

case "${RUN_VSLAM_OVERRIDE}" in
    auto)
        ;;
    offline|online)
        PIPELINE_MODE="${RUN_VSLAM_OVERRIDE}"
        ;;
    *)
        echo "Invalid VSLAM override: ${RUN_VSLAM_OVERRIDE}" >&2
        exit 1
        ;;
esac

if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
    if [ -z "${ODOM_TOPIC}" ]; then
        ODOM_TOPIC="${DEFAULT_VSLAM_ODOM_TOPIC}"
    fi
    CONFIG_BASENAME="cartographer_2d_with_odom.lua"
    if odom_ready_wait_applicable && [ -z "${ODOM_READY_MIN_RATE_HZ}" ]; then
        ODOM_READY_MIN_RATE_HZ="$(default_odom_ready_min_rate_hz)"
    fi
else
    ODOM_TOPIC=""
    CONFIG_BASENAME="cartographer_2d.lua"
fi

if [ "${CARTOGRAPHER_USE_ODOM}" = true ] && [ "${ODOM_TOPIC}" != "${DEFAULT_VSLAM_ODOM_TOPIC}" ]; then
    echo "Warning: with_odom modes assume VSLAM publishes odom on ${DEFAULT_VSLAM_ODOM_TOPIC}." >&2
    echo "         current odom topic: ${ODOM_TOPIC}" >&2
fi











if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi

if [ -z "${MAP_NAME}" ]; then
    prompt_map_name_interactive
fi

# bag dir name
BAG_PATH_CLEAN="${BAG_PATH%/}"
BAG_DIR_NAME="$(basename "${BAG_PATH_CLEAN}")"
SOURCE_BAG_PATH="${BAG_PATH}"

if [ "${PIPELINE_MODE}" = "online" ] && [ "${CARTOGRAPHER_USE_ODOM}" = true ] && [ "${PLAY_ALL_TOPICS}" = true ]; then
    echo "--play-all-topics is not supported in with_odom_online_vslam mode." >&2
    echo "Cartographer odom must come from the live VSLAM output, not from the replayed bag." >&2
    exit 1
fi

# output paths
BAG_OUT_DIR="/map/${BAG_DIR_NAME}"
OUT_DIR="${BAG_OUT_DIR}/${MAP_NAME}"
MAP_STEM="${OUT_DIR}/${MAP_NAME}"
PBSTREAM_PATH="${MAP_STEM}.pbstream"
MAP_YAML_PATH="${MAP_STEM}.yaml"
MAP_PGM_PATH="${MAP_STEM}.pgm"
MAP_PNG_PATH="${MAP_STEM}.png"
SECTION_OUTPUT_PATH="${OUT_DIR}/sections_pixels.csv"
SECTION_GATE_OUTPUT_PATH="${OUT_DIR}/sections_pixels_gates.csv"
CENTERLINE_OUTPUT_PATH="${MAP_STEM}_centerline.csv"
RACELINE_OUTPUT_PATH="${MAP_STEM}_raceline.csv"
LINE_PREVIEW_OUTPUT_PATH="${MAP_STEM}_lines.png"
if [ -z "${VSLAM_REFERENCE_SNAPSHOT_PATH}" ]; then
    VSLAM_REFERENCE_SNAPSHOT_PATH="${MAP_STEM}_vslam_reference.json"
fi
if [ -z "${VSLAM_LANDMARK_IMAGE_PATH}" ]; then
    VSLAM_LANDMARK_IMAGE_PATH="${MAP_STEM}_vslam_landmarks.png"
fi
if [ -z "${VSLAM_LANDMARK_YAML_PATH}" ]; then
    VSLAM_LANDMARK_YAML_PATH="${MAP_STEM}_vslam_landmarks.yaml"
fi
if [ -z "${VSLAM_TRACE_OUTPUT_PATH}" ]; then
    VSLAM_TRACE_OUTPUT_PATH="${MAP_STEM}_vslam_traced.png"
fi
if [ -z "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ]; then
    VSLAM_MAP_ALIGNMENT_CONFIG_PATH="${OUT_DIR}/vslam_map_alignment.yaml"
fi
MAP_LOG_PATH="/tmp/cartographer_mapping_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_ODOM_BAG_DIR="${OUT_DIR}/offline_vslam_odom_input_$(date +%Y%m%d_%H%M%S)"

# validate input bags
if [ ! -d "${SOURCE_BAG_PATH}" ] || [ ! -f "${SOURCE_BAG_PATH}/metadata.yaml" ]; then
    echo "Invalid source BAG_PATH: ${SOURCE_BAG_PATH}" >&2
    echo "metadata.yaml not found." >&2
    exit 1
fi

if [ ! -d "$BAG_PATH" ] || [ ! -f "$BAG_PATH/metadata.yaml" ]; then
    echo "Invalid active BAG_PATH: $BAG_PATH" >&2
    echo "metadata.yaml not found." >&2
    exit 1
fi

if [ "${VSLAM_MAP_ALIGNMENT_CONFIG_SET_BY_USER}" = true ] && [ ! -f "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ]; then
    echo "Warning: VSLAM alignment config not found. Continue without saved map->vslam_map alignment: ${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" >&2
fi

mkdir -p "${OUT_DIR}"













































trap cleanup_all EXIT INT TERM

if [ -z "${VSLAM_MAP_DIR}" ]; then
    VSLAM_MAP_DIR="${OUT_DIR}/cuvslam_map"
fi
VSLAM_LOG_PATH="/tmp/create_2d_map_vslam_mapping_$(date +%Y%m%d_%H%M%S).log"
TF_LOG_PATH="/tmp/create_2d_map_vslam_tf_$(date +%Y%m%d_%H%M%S).log"
VSLAM_PLAYER_LOG_PATH="/tmp/create_2d_map_vslam_player_$(date +%Y%m%d_%H%M%S).log"
CARTOGRAPHER_PLAYER_LOG_PATH="/tmp/cartographer_player_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_MAP_LOG_PATH="/tmp/create_2d_map_offline_vslam_map_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_MAP_TF_LOG_PATH="/tmp/create_2d_map_offline_vslam_map_tf_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH="/tmp/create_2d_map_offline_vslam_map_player_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_ODOM_LOG_PATH="/tmp/create_2d_map_offline_vslam_odom_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_ODOM_TF_LOG_PATH="/tmp/create_2d_map_offline_vslam_odom_tf_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH="/tmp/create_2d_map_offline_vslam_odom_player_$(date +%Y%m%d_%H%M%S).log"
OFFLINE_VSLAM_ODOM_RECORD_LOG_PATH="/tmp/create_2d_map_offline_vslam_odom_record_$(date +%Y%m%d_%H%M%S).log"
VSLAM_LANDMARK_LOG_PATH="/tmp/create_2d_map_vslam_landmarks_$(date +%Y%m%d_%H%M%S).log"
VSLAM_LANDMARK_TF_LOG_PATH="/tmp/create_2d_map_vslam_landmarks_tf_$(date +%Y%m%d_%H%M%S).log"
VSLAM_LANDMARK_PLAYER_LOG_PATH="/tmp/create_2d_map_vslam_landmarks_player_$(date +%Y%m%d_%H%M%S).log"
VSLAM_LANDMARK_EXPORT_LOG_PATH="/tmp/create_2d_map_vslam_landmarks_export_$(date +%Y%m%d_%H%M%S).log"
VSLAM_REFERENCE_SNAPSHOT_LOG_PATH="${OUT_DIR}/vslam_reference_capture.log"
POST_ALIGNMENT_LOG_PATH="/tmp/create_2d_map_post_alignment_$(date +%Y%m%d_%H%M%S).log"
POST_ALIGNMENT_REFERENCE_PUBLISHER_LOG_PATH="${OUT_DIR}/post_alignment_reference_publisher.log"

if [ "${PIPELINE_MODE}" = "online" ]; then
    mkdir -p "${VSLAM_MAP_DIR}"
fi

if [ "${PIPELINE_MODE}" = "offline" ] && [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
    prepare_offline_vslam_odom_bag
fi

print_mode_summary
prompt_live_vslam_alignment
ensure_vslam_visualization_requirements

# ==========================================
# 1. Build maps
# ==========================================
if [ "${PIPELINE_MODE}" = "online" ]; then
    MAIN_VSLAM_ENABLE_ALIGNMENT_FROM_CONFIG=true
    if [ "${VSLAM_LIVE_ALIGNMENT_ENABLED}" = true ]; then
        MAIN_VSLAM_ENABLE_ALIGNMENT_FROM_CONFIG=false
    fi
    echo "[1/5] Launch online VSLAM for map creation (logs: ${TF_LOG_PATH}, ${VSLAM_LOG_PATH})"
    launch_vslam_stack \
        "${TF_LOG_PATH}" \
        "${VSLAM_LOG_PATH}" \
        "${VSLAM_MAP_DIR}" \
        "" \
        "" \
        "${MAIN_VSLAM_ENABLE_ALIGNMENT_FROM_CONFIG}"

    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        echo "[2/5] Launch cartographer with live VSLAM odom (log: ${MAP_LOG_PATH})"
    else
        echo "[2/5] Launch cartographer without odom (log: ${MAP_LOG_PATH})"
    fi
    launch_cartographer_mapping

    echo "[3/5] Wait for VSLAM and cartographer services"
    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready. Check log: ${VSLAM_LOG_PATH}" >&2
        exit 1
    fi
    if ! wait_for_service "/write_state" 60; then
        echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
        exit 1
    fi

    start_vslam_reference_capture

    echo "[4/5] Play source rosbag for online VSLAM + Cartographer"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        if [ "${VSLAM_LIVE_ALIGNMENT_ENABLED}" = true ]; then
            play_rosbag_background \
                "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}"
            run_live_vslam_alignment_session
            if ! wait_for_rosbag_playback; then
                exit 1
            fi
        else
            if ! play_rosbag \
                "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}"; then
                exit 1
            fi
        fi
    else
        build_online_source_play_topics
        PLAY_TOPICS=("${SOURCE_PLAY_TOPICS[@]}")

        echo "  - mode: filtered topics"
        echo "  - topics: ${PLAY_TOPICS[*]}"
        if [ "${VSLAM_LIVE_ALIGNMENT_ENABLED}" = true ]; then
            play_rosbag_background \
                "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}" "${PLAY_TOPICS[@]}"
            run_live_vslam_alignment_session
            if ! wait_for_rosbag_playback; then
                exit 1
            fi
        else
            if ! play_rosbag \
                "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}" "${PLAY_TOPICS[@]}"; then
                exit 1
            fi
        fi
    fi

    echo "[5/5] Save VSLAM map and cartographer state"
    sleep 2
    ros2 service call /visual_slam/save_map \
        isaac_ros_visual_slam_interfaces/srv/FilePath \
        "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null

    if ! ros2 service call /finish_trajectory \
        cartographer_ros_msgs/srv/FinishTrajectory \
        "{trajectory_id: 0}" > /dev/null; then
        echo "Warning: /finish_trajectory failed. Continue." >&2
    fi

    WRITE_STATE_REQUEST=$(printf "{filename: '%s', include_unfinished_submaps: true}" "${PBSTREAM_PATH}")
    ros2 service call /write_state \
        cartographer_ros_msgs/srv/WriteState \
        "${WRITE_STATE_REQUEST}" > /dev/null

    stop_vslam_reference_recorder
    stop_vslam
    stop_cartographer
else
    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        if [ "${OFFLINE_ODOM_BAG_CREATED}" = true ]; then
            echo "[1/3] Launch cartographer with offline-generated VSLAM odom (log: ${MAP_LOG_PATH})"
        else
            echo "[1/3] Launch cartographer with replayed odom (log: ${MAP_LOG_PATH})"
        fi
    else
        echo "[1/3] Launch cartographer without odom (log: ${MAP_LOG_PATH})"
    fi
    launch_cartographer_mapping

    echo "[2/3] Wait for /write_state service"
    if ! wait_for_service "/write_state" 60; then
        echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
        exit 1
    fi

    echo "[3/3] Build the provisional 2D map"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        if ! play_rosbag \
            "${BAG_PATH}" "${CARTOGRAPHER_PLAYER_LOG_PATH}"; then
            exit 1
        fi
    else
        # Cartographer only needs the scan, odometry topic, and static sensor TFs.
        # Replaying VSLAM's dynamic /tf here can conflict with Cartographer's own TF output.
        PLAY_TOPICS=("${SCAN_TOPIC}" "/tf_static")
        if [ -n "${ODOM_TOPIC}" ]; then
            PLAY_TOPICS+=("${ODOM_TOPIC}")
        fi

        echo "  - mode: filtered topics"
        echo "  - topics: ${PLAY_TOPICS[*]}"
        if ! play_rosbag \
            "${BAG_PATH}" "${CARTOGRAPHER_PLAYER_LOG_PATH}" "${PLAY_TOPICS[@]}"; then
            exit 1
        fi
    fi

    if ! ros2 service call /finish_trajectory \
        cartographer_ros_msgs/srv/FinishTrajectory \
        "{trajectory_id: 0}" > /dev/null; then
        echo "Warning: /finish_trajectory failed. Continue." >&2
    fi

    WRITE_STATE_REQUEST=$(printf "{filename: '%s', include_unfinished_submaps: true}" "${PBSTREAM_PATH}")

    ros2 service call /write_state \
        cartographer_ros_msgs/srv/WriteState \
        "${WRITE_STATE_REQUEST}" > /dev/null

    stop_cartographer
fi

if ! convert_pbstream_to_map "✅ Provisional map generated:"; then
    exit 1
fi

run_post_vslam_map_alignment_prep

# ==========================================
# 2. Centerline
# ==========================================
prompt_centerline_generation
CENTERLINE_INPUT_MAP="${MAP_PGM_PATH}"
if [ -f "${MAP_PNG_PATH}" ]; then
    CENTERLINE_INPUT_MAP="${MAP_PNG_PATH}"
fi

prompt_map_edit
run_map_edit "${CENTERLINE_INPUT_MAP}"
generate_centerline "${CENTERLINE_INPUT_MAP}"

# ==========================================
# 3. Raceline
# ==========================================
prompt_raceline_generation
generate_raceline "${CENTERLINE_OUTPUT_PATH}"

# ==========================================
# 4. Line preview
# ==========================================
generate_line_preview "${CENTERLINE_INPUT_MAP}"

# ==========================================
# 5. Transfer by scp
# ==========================================
echo ""
if [ "${ENABLE_SCP}" != true ]; then
    echo "scp転送をスキップしました。"
    exit 0
fi

prompt_pre_transfer_action

echo ""
read -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

echo ""
echo "相手のIPアドレスを選択、または直接入力してください:"
i=1
for ip in "${DEFAULT_REMOTE_IPS[@]}"; do
    if [ $i -eq 1 ]; then
        echo "  $i) $ip (Enterのデフォルト)"
    else
        echo "  $i) $ip"
    fi
    ((i++))
done
echo ""

read -p "番号、またはIPを直接入力 (Enterで '${DEFAULT_REMOTE_IPS[0]}'): " IP_CHOICE

if [ -z "$IP_CHOICE" ]; then
    REMOTE_IP="${DEFAULT_REMOTE_IPS[0]}"
elif [[ "$IP_CHOICE" =~ ^[0-9]+$ ]] && \
     [ "$IP_CHOICE" -ge 1 ] && \
     [ "$IP_CHOICE" -le "${#DEFAULT_REMOTE_IPS[@]}" ]; then
    REMOTE_IP="${DEFAULT_REMOTE_IPS[$((IP_CHOICE-1))]}"
else
    REMOTE_IP="$IP_CHOICE"
fi

echo ""
read -p "送信先ディレクトリ (Enterで '${DEFAULT_REMOTE_DIR}'): " REMOTE_DIR
REMOTE_DIR=${REMOTE_DIR:-$DEFAULT_REMOTE_DIR}

echo ""
echo "================ 転送内容確認 ================"
echo "送信元 : ${OUT_DIR}"
echo "送信先 : ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR%/}/${BAG_DIR_NAME}/"
echo "=============================================="
echo ""

read -p "この内容でscp転送しますか？ (Y/n, Enterで実行): " FINAL_CONFIRM
FINAL_CONFIRM=${FINAL_CONFIRM:-y}

if [[ ! "$FINAL_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "転送をキャンセルしました。"
    exit 0
fi

echo "scp転送を開始します..."
ssh "${REMOTE_USER}@${REMOTE_IP}" "mkdir -p '${REMOTE_DIR%/}/${BAG_DIR_NAME}'"
scp -r "${OUT_DIR}" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR%/}/${BAG_DIR_NAME}/"

echo ""
echo "✅ scp転送が完了しました！"
