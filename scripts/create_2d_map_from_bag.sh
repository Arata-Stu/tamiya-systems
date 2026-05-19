#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYSTEM_LAUNCH_SOURCE_SHARE=""
SYSTEM_LAUNCH_CMD=()

resolve_system_launch_source_share() {
    local candidate

    for candidate in \
        "${CREATE_2D_MAP_SYSTEM_LAUNCH_SOURCE_SHARE:-}" \
        "/workspaces/src/launch/system_launch" \
        "${REPO_ROOT}/ros2_ws/src/launch/system_launch" \
        "${REPO_ROOT}" \
        "${PWD}/ros2_ws/src/launch/system_launch" \
        "${PWD}"; do
        [ -n "${candidate}" ] || continue
        if [ -d "${candidate}/launch" ] && [ -d "${candidate}/config" ]; then
            (cd "${candidate}" && pwd)
            return 0
        fi
    done

    return 1
}

resolve_system_launch_config_file() {
    local relative_path="$1"
    local pkg_prefix=""

    if [ -n "${SYSTEM_LAUNCH_SOURCE_SHARE}" ] && \
       [ -f "${SYSTEM_LAUNCH_SOURCE_SHARE}/config/${relative_path}" ]; then
        echo "${SYSTEM_LAUNCH_SOURCE_SHARE}/config/${relative_path}"
        return 0
    fi

    if command -v ros2 >/dev/null 2>&1; then
        pkg_prefix="$(ros2 pkg prefix system_launch 2>/dev/null || true)"
        if [ -n "${pkg_prefix}" ] && \
           [ -f "${pkg_prefix}/share/system_launch/config/${relative_path}" ]; then
            echo "${pkg_prefix}/share/system_launch/config/${relative_path}"
            return 0
        fi
    fi

    return 1
}

resolve_repo_file() {
    local relative_path="$1"
    local candidate_root

    for candidate_root in \
        "${CREATE_2D_MAP_PROJECT_ROOT:-}" \
        "/workspaces/src/launch/system_launch" \
        "${REPO_ROOT}" \
        "${SYSTEM_LAUNCH_SOURCE_SHARE}" \
        "${PWD}"; do
        [ -n "${candidate_root}" ] || continue
        if [ -f "${candidate_root}/${relative_path}" ]; then
            echo "${candidate_root}/${relative_path}"
            return 0
        fi
    done

    return 1
}

resolve_python_ws_file() {
    local relative_path="$1"
    local candidate_root

    for candidate_root in \
        "${CREATE_2D_MAP_PYTHON_WS_ROOT:-}" \
        "/python_ws" \
        "${REPO_ROOT}/python_ws" \
        "${PWD}/python_ws"; do
        [ -n "${candidate_root}" ] || continue
        if [ -f "${candidate_root}/${relative_path}" ]; then
            echo "${candidate_root}/${relative_path}"
            return 0
        fi
    done

    return 1
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
    if [ -n "${CREATE_2D_MAP_SETUP:-}" ]; then
        source_setup_script "${CREATE_2D_MAP_SETUP}"
    elif [ -f "/workspaces/install/setup.bash" ]; then
        source_setup_script "/workspaces/install/setup.bash"
    elif [ -f "install/setup.bash" ]; then
        source_setup_script "install/setup.bash"
    fi
}

source_setup_if_available

SYSTEM_LAUNCH_SOURCE_SHARE="$(resolve_system_launch_source_share || true)"

build_system_launch_cmd() {
    local launch_file="$1"

    SYSTEM_LAUNCH_CMD=(ros2 launch)
    if [ -f "${SYSTEM_LAUNCH_SOURCE_SHARE}/launch/${launch_file}" ]; then
        SYSTEM_LAUNCH_CMD+=("${SYSTEM_LAUNCH_SOURCE_SHARE}/launch/${launch_file}")
    else
        SYSTEM_LAUNCH_CMD+=(system_launch "${launch_file}")
    fi
}

usage() {
    cat <<'EOF'
Usage:
  create_2d_map_from_bag.sh [OPTIONS]

Options:
  --mode NAME         no_odom_offline_vslam|no_odom_online_vslam|
                      with_odom_offline_vslam|with_odom_online_vslam
                      aliases: default=no_odom_offline_vslam,
                               2d_slam=no_odom_online_vslam
  --bag-path DIR      input rosbag2 directory (skip interactive selection)
  --map-name NAME     output map name (skip interactive prompt)
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  enable odometry in cartographer and set the odom topic
                      default for with_odom_* modes: /visual_slam/tracking/odometry
  --odom-ready-window N
                      number of odom messages used for stability check (default: 10)
  --odom-ready-min-rate HZ
                      minimum average odom rate before recording starts
                      default: 90% of --image-fps in with_odom modes
  --odom-ready-timeout SEC
                      timeout while waiting for odom rate stabilization (default: 45)
  --no-odom-ready-wait
                      disable odom-rate stabilization wait in with_odom modes
  --run-vslam         compatibility override: force online_vslam execution
  --no-vslam          compatibility override: force offline_vslam execution
  --vslam-vis         enable VSLAM visualization topics during parallel VSLAM execution
  --no-vslam-vis      disable VSLAM visualization topics (default)
  --image-width PX    camera width for parallel VSLAM launch (default: 424)
  --image-height PX   camera height for parallel VSLAM launch (default: 240)
  --image-fps FPS     camera fps for parallel VSLAM launch (default: 90.0)
  --with-imu          replay /camera/imu as well (default: disabled)
  --vslam-map-dir DIR visual slam map output directory
  --pipeline-mode MODE
                      offline|online|auto
                      compatibility override for VSLAM execution mode
  --play-all-topics   play all topics in bag (default: play only needed topics)
  --record-root DIR   rosbag探索ルート (default: /record)
  --no-scp            skip interactive scp transfer step
  --no-centerline     skip centerline CSV generation
  --no-raceline       skip raceline CSV generation
  --no-line-preview   skip centerline/raceline preview image generation
  --edit-map          always launch GUI map cleanup editor before centerline generation
  --no-edit-map       never launch GUI map cleanup editor
  --map-edit-mode MODE
                      auto|always|never (default: auto)
  --map-edit-script PATH
                      path to map_cleanup_editor.py (auto-detect by default)
  --map-edit-output PATH
                      path to cleaned PNG output (default: <MAP_NAME>_centerline_input.png)
  --centerline-debug  save centerline debug images (default: enabled when centerline is generated)
  --centerline-debug-dir DIR
                      set centerline debug image output directory
  --centerline-script PATH
                      path to generate_centerline.py (auto-detect by default)
  --line-preset PRESET
                      default|race-stacks for centerline/raceline helper scripts (default: default)
  --centerline-direction DIR
                      forward|reverse|both (default: forward)
  --raceline-script PATH
                      path to generate_raceline.py (auto-detect by default)
  --raceline-backend BACKEND
                      heuristic|global-opt|auto (default: heuristic)
  --raceline-opt-type TYPE
                      shortest_path|mincurv|mincurv_iqp for global-opt (default: mincurv_iqp)
  --raceline-direction DIR
                      forward|reverse|both (default: forward)
  --optimizer-root DIR
                      path to global_racetrajectory_optimization checkout
  --line-preview-script PATH
                      path to visualize_race_lines.py (auto-detect by default)
  -h, --help          show this help

When --mode is omitted:
  the script interactively prompts you to choose one of the 4 mode presets

Mode presets:
  no_odom_offline_vslam:
      Cartographer は scan-only。VSLAM はこのスクリプトでは起動しない
  no_odom_online_vslam:
      Cartographer は scan-only。bag replay と同時に VSLAM も起動する
  with_odom_offline_vslam:
      先に VSLAM map を作り、その map で odom bag を生成してから Cartographer に渡す
  with_odom_online_vslam:
      Cartographer は replay 中に起動した VSLAM の live odom を使う

Outputs:
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pbstream
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pgm
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.png (optional; generated if converter is available)
  /map/<bag_name>/<MAP_NAME>/cuvslam_map/ (optional; generated in online VSLAM or offline odom modes)
  /map/<bag_name>/<MAP_NAME>/offline_vslam_odom_input_<timestamp>/ (optional; generated in with_odom_offline_vslam)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline_input.png (optional; hand-edited map cleanup result)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline.csv (optional; generated unless --no-centerline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_raceline.csv (optional; generated unless --no-raceline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_lines.png (optional; generated unless --no-line-preview)

After map creation:
  optionally transfer /map/<bag_name>/<MAP_NAME>/ to remote host by scp

Interactive flow:
  1) mode を選択（--mode 省略時）
  2) /record を再帰探索して metadata.yaml を持つ rosbag2 ディレクトリを一覧表示
  3) 番号で rosbag を選択
  4) map 名を入力
  5) 選択した mode に応じて Cartographer / VSLAM を実行して map を生成
  6) centerline 生成の可否を確認（debug はデフォルト有効）
  7) 必要なら GUI で map PNG/PGM を黒塗り修正して保存
  8) raceline 生成の可否を確認
  9) centerline / raceline preview画像を生成
  10) 転送前メニューで section edit / scp / 終了 を選択
EOF
}

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
CENTERLINE_DEBUG=true
CENTERLINE_DEBUG_DIR=""
CENTERLINE_SCRIPT_PATH=""
CENTERLINE_PRESET="default"
CENTERLINE_DIRECTION="forward"
RACELINE_SCRIPT_PATH=""
RACELINE_PRESET="race-stacks"
RACELINE_BACKEND="global-opt"
RACELINE_OPT_TYPE="mincurv_iqp"
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
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
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
BASE_CARTOGRAPHER_PIDS=()
ROSBAG_CANDIDATES=()
SOURCE_PLAY_TOPICS=()
OFFLINE_ODOM_BAG_DIR=""
OFFLINE_ODOM_BAG_CREATED=false
VSLAM_LOCALIZATION_PARAM_PATH=""

apply_mode() {
    case "$1" in
        default|no_odom_offline_vslam)
            CARTOGRAPHER_USE_ODOM=false
            PIPELINE_MODE="offline"
            ;;
        2d_slam|no_odom_online_vslam)
            CARTOGRAPHER_USE_ODOM=false
            PIPELINE_MODE="online"
            ;;
        with_odom_offline_vslam)
            CARTOGRAPHER_USE_ODOM=true
            PIPELINE_MODE="offline"
            ;;
        with_odom_online_vslam)
            CARTOGRAPHER_USE_ODOM=true
            PIPELINE_MODE="online"
            ;;
        *)
            echo "Unknown mode: $1" >&2
            usage
            exit 1
            ;;
    esac
}

prompt_mode_interactive() {
    local choice

    echo ""
    echo "2D map 作成 mode を選択してください:"
    echo "  1) no_odom_offline_vslam   Cartographer=scan-only / VSLAMは別実行"
    echo "  2) no_odom_online_vslam    Cartographer=scan-only / 今回の実行でVSLAMも起動"
    echo "  3) with_odom_offline_vslam 先にVSLAM map + odom bagを作ってからCartographer"
    echo "  4) with_odom_online_vslam  Cartographer=live VSLAM odom使用 / 今回の実行でVSLAMも起動"
    echo ""

    while :; do
        read -r -p "番号で選択してください (1-4, Enterで '1'): " choice
        choice="${choice:-1}"
        case "${choice}" in
            1)
                MODE="no_odom_offline_vslam"
                return 0
                ;;
            2)
                MODE="no_odom_online_vslam"
                return 0
                ;;
            3)
                MODE="with_odom_offline_vslam"
                return 0
                ;;
            4)
                MODE="with_odom_online_vslam"
                return 0
                ;;
            *)
                echo "無効な入力です。1-4 を選択してください。"
                ;;
        esac
    done
}

resolve_vslam_param_file() {
    resolve_system_launch_config_file "localization/vslam.param.yaml"
}

resolve_timeout_cmd() {
    command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true
}

float_ge() {
    local lhs="$1"
    local rhs="$2"
    awk -v lhs="${lhs}" -v rhs="${rhs}" 'BEGIN { exit !((lhs + 0.0) >= (rhs + 0.0)) }'
}

default_odom_ready_min_rate_hz() {
    awk -v image_fps="${IMAGE_FPS}" 'BEGIN {
        rate = image_fps * 0.90
        if (rate < 1.0) {
            rate = 1.0
        }
        printf "%.1f", rate
    }'
}

odom_ready_wait_applicable() {
    [ "${CARTOGRAPHER_USE_ODOM}" = true ] && [ "${PIPELINE_MODE}" = "offline" ]
}

create_vslam_localization_param() {
    local base_param
    local temp_param

    base_param="$(resolve_vslam_param_file || true)"
    if [ -z "${base_param}" ] || [ ! -f "${base_param}" ]; then
        echo "Failed to resolve vslam.param.yaml for offline localization." >&2
        return 1
    fi

    temp_param="$(mktemp /tmp/create_2d_map_vslam_param_XXXXXX.yaml)"
    sed 's/^\([[:space:]]*localize_on_startup:\).*/\1 true/' "${base_param}" > "${temp_param}"

    if ! grep -Eq '^[[:space:]]*localize_on_startup:[[:space:]]*true[[:space:]]*$' "${temp_param}"; then
        echo "Failed to enable localize_on_startup in ${temp_param}." >&2
        return 1
    fi

    VSLAM_LOCALIZATION_PARAM_PATH="${temp_param}"
}

launch_vslam_stack() {
    local tf_log_path="$1"
    local vslam_log_path="$2"
    local save_map_path="${3:-}"
    local load_map_path="${4:-}"
    local vslam_param_path="${5:-}"
    local -a launch_args=(
        "image_width:=${IMAGE_WIDTH}"
        "image_height:=${IMAGE_HEIGHT}"
        "camera_container_name:=${CAMERA_CONTAINER_NAME}"
        "enable_localization_and_mapping:=true"
        "enable_slam_visualization:=${VSLAM_VIS_ENABLED}"
        "enable_observations_view:=${VSLAM_VIS_ENABLED}"
        "enable_landmarks_view:=${VSLAM_VIS_ENABLED}"
    )

    if [ -n "${save_map_path}" ]; then
        launch_args+=("save_map_path:=${save_map_path}")
    fi

    if [ -n "${load_map_path}" ]; then
        launch_args+=("load_map_path:=${load_map_path}")
    fi

    if [ -n "${vslam_param_path}" ]; then
        launch_args+=("vslam_param:=${vslam_param_path}")
    fi

    build_system_launch_cmd "offline_sensor_tf.launch.xml"
    launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        > "${tf_log_path}" 2>&1

    sleep 2

    launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
        ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

    sleep 2

    build_system_launch_cmd "vslam.launch.xml"
    launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        "${launch_args[@]}" \
        > "${vslam_log_path}" 2>&1
}

list_cartographer_pids() {
    {
        pgrep -f "cartographer_node" 2>/dev/null || true
        pgrep -f "cartographer_occupancy_grid_node" 2>/dev/null || true
    } | sort -u
}

capture_base_cartographer_pids() {
    BASE_CARTOGRAPHER_PIDS=()
    local pid
    while IFS= read -r pid; do
        [ -n "$pid" ] && BASE_CARTOGRAPHER_PIDS+=("$pid")
    done < <(list_cartographer_pids)
}

is_base_cartographer_pid() {
    local target="$1"
    local pid
    for pid in "${BASE_CARTOGRAPHER_PIDS[@]}"; do
        if [ "$pid" = "$target" ]; then
            return 0
        fi
    done
    return 1
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

cleanup_new_cartographer_processes() {
    local pid
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if ! is_base_cartographer_pid "$pid"; then
            kill_pid_gracefully "$pid" false
        fi
    done < <(list_cartographer_pids)
}

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

resolve_effective_mode() {
    local odom_label="no_odom"
    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        odom_label="with_odom"
    fi

    echo "${odom_label}_${PIPELINE_MODE}_vslam"
}

describe_odom_source() {
    if [ "${CARTOGRAPHER_USE_ODOM}" != true ]; then
        echo "disabled"
        return 0
    fi

    if [ "${PIPELINE_MODE}" = "online" ]; then
        echo "live VSLAM output (${ODOM_TOPIC})"
    elif [ "${OFFLINE_ODOM_BAG_CREATED}" = true ]; then
        echo "offline-generated VSLAM odom bag (${ODOM_TOPIC})"
    else
        echo "pre-recorded odom bag (${ODOM_TOPIC})"
    fi
}

print_mode_summary() {
    local effective_mode
    effective_mode="$(resolve_effective_mode)"

    echo ""
    echo "================ Map build mode ================"
    echo "mode            : ${effective_mode}"
    echo "source bag      : ${SOURCE_BAG_PATH}"
    echo "cartographer bag: ${BAG_PATH}"
    echo "scan topic      : ${SCAN_TOPIC}"
    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        echo "cartographer odom: enabled (${ODOM_TOPIC})"
        if odom_ready_wait_applicable && [ "${ODOM_READY_WAIT_ENABLED}" = true ]; then
            echo "odom ready wait : window=${ODOM_READY_WINDOW}, min_rate=${ODOM_READY_MIN_RATE_HZ} Hz, timeout=${ODOM_READY_TIMEOUT_SEC}s"
        elif odom_ready_wait_applicable; then
            echo "odom ready wait : disabled"
        else
            echo "odom ready wait : n/a for live online odom"
        fi
    else
        echo "cartographer odom: disabled"
    fi
    echo "vslam execution : ${PIPELINE_MODE}"
    echo "odom source     : $(describe_odom_source)"
    echo "================================================"
    echo ""
}

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

build_online_source_play_topics() {
    SOURCE_PLAY_TOPICS=(
        "${SCAN_TOPIC}"
        "/tf_static"
        "/camera/left/image_raw"
        "/camera/left/camera_info"
        "/camera/right/image_raw"
        "/camera/right/camera_info"
    )

    if [ "${USE_IMU}" = true ]; then
        SOURCE_PLAY_TOPICS+=("/camera/imu")
    fi
}

prepare_offline_vslam_odom_bag() {
    echo "[prep 1/2] Build offline VSLAM map (logs: ${OFFLINE_VSLAM_MAP_TF_LOG_PATH}, ${OFFLINE_VSLAM_MAP_LOG_PATH})"
    mkdir -p "${VSLAM_MAP_DIR}"

    launch_vslam_stack \
        "${OFFLINE_VSLAM_MAP_TF_LOG_PATH}" \
        "${OFFLINE_VSLAM_MAP_LOG_PATH}" \
        "${VSLAM_MAP_DIR}"

    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready for offline map build. Check log: ${OFFLINE_VSLAM_MAP_LOG_PATH}" >&2
        exit 1
    fi

    echo "  - replay source bag to create VSLAM map"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH}"; then
            exit 1
        fi
    else
        build_online_source_play_topics
        echo "  - mode: filtered topics"
        echo "  - topics: ${SOURCE_PLAY_TOPICS[*]}"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"; then
            exit 1
        fi
    fi

    sleep 2
    if ! ros2 service call /visual_slam/save_map \
        isaac_ros_visual_slam_interfaces/srv/FilePath \
        "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null; then
        echo "Failed to save offline VSLAM map. Check log: ${OFFLINE_VSLAM_MAP_LOG_PATH}" >&2
        exit 1
    fi

    stop_vslam

    echo "[prep 2/2] Create offline odom bag from saved VSLAM map"
    create_vslam_localization_param

    launch_vslam_stack \
        "${OFFLINE_VSLAM_ODOM_TF_LOG_PATH}" \
        "${OFFLINE_VSLAM_ODOM_LOG_PATH}" \
        "" \
        "${VSLAM_MAP_DIR}" \
        "${VSLAM_LOCALIZATION_PARAM_PATH}"

    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready for offline localization. Check log: ${OFFLINE_VSLAM_ODOM_LOG_PATH}" >&2
        exit 1
    fi

    echo "  - replay source bag to record odom input bag"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        play_rosbag_background \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH}"
    else
        build_online_source_play_topics
        echo "  - mode: filtered topics"
        echo "  - topics: ${SOURCE_PLAY_TOPICS[*]}"
        play_rosbag_background \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"
    fi

    if [ "${ODOM_READY_WAIT_ENABLED}" = true ]; then
        echo "  - wait for ${ODOM_TOPIC} to stabilize before recording"
        if ! wait_for_topic_rate_ready \
            "${ODOM_TOPIC}" \
            "${ODOM_READY_MIN_RATE_HZ}" \
            "${ODOM_READY_WINDOW}" \
            "${ODOM_READY_TIMEOUT_SEC}"; then
            echo "Timed out waiting for ${ODOM_TOPIC} to reach ${ODOM_READY_MIN_RATE_HZ} Hz." >&2
            stop_rosbag_playback
            exit 1
        fi
    else
        sleep 2
    fi

    echo "  - start odom bag recording"
    start_offline_odom_bag_recording "${OFFLINE_ODOM_BAG_DIR}" "${OFFLINE_VSLAM_ODOM_RECORD_LOG_PATH}"
    sleep 1

    if ! wait_for_rosbag_playback; then
        echo "rosbag replay failed while recording offline odom input bag." >&2
        exit 1
    fi

    sleep 2
    stop_recorder
    stop_vslam

    if [ ! -f "${OFFLINE_ODOM_BAG_DIR}/metadata.yaml" ]; then
        echo "Offline odom bag was not created correctly: ${OFFLINE_ODOM_BAG_DIR}" >&2
        exit 1
    fi

    BAG_PATH="${OFFLINE_ODOM_BAG_DIR}"
    OFFLINE_ODOM_BAG_CREATED=true

    echo "✅ Offline odom bag generated: ${OFFLINE_ODOM_BAG_DIR}"
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

mkdir -p "${OUT_DIR}"

stop_cartographer() {
    if [ -n "${CARTOGRAPHER_PID:-}" ]; then
        kill_pid_gracefully "${CARTOGRAPHER_PID}" "${CARTOGRAPHER_USES_SETSID}"
        wait "${CARTOGRAPHER_PID}" 2>/dev/null || true
        CARTOGRAPHER_PID=""
        CARTOGRAPHER_USES_SETSID=false
    fi

    # Safety net:
    # Kill only cartographer processes that appeared after this script started.
    cleanup_new_cartographer_processes
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

stop_vslam() {
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

stop_recorder() {
    stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
}

stop_rosbag_playback() {
    stop_background_process "ROSBAG_PLAY_PID" "ROSBAG_PLAY_USES_SETSID"
}

start_offline_odom_bag_recording() {
    local bag_dir="$1"
    local log_path="$2"

    launch_background_process "RECORDER_PID" "RECORDER_USES_SETSID" \
        ros2 bag record \
        -o "${bag_dir}" \
        "${ODOM_TOPIC}" \
        "${SCAN_TOPIC}" \
        /tf_static \
        > "${log_path}" 2>&1
}

play_rosbag() {
    local bag_path="$1"
    local log_path="$2"
    shift 2

    local -a player_cmd=(
        ros2 bag play "${bag_path}" --clock --rate "${PLAY_RATE}"
    )

    if [ "$#" -gt 0 ]; then
        player_cmd+=(--topics "$@")
    fi

    "${player_cmd[@]}" > "${log_path}" 2>&1
}

play_rosbag_background() {
    local bag_path="$1"
    local log_path="$2"
    shift 2

    local -a player_cmd=(
        ros2 bag play "${bag_path}" --clock --rate "${PLAY_RATE}"
    )

    if [ "$#" -gt 0 ]; then
        player_cmd+=(--topics "$@")
    fi

    launch_background_process "ROSBAG_PLAY_PID" "ROSBAG_PLAY_USES_SETSID" \
        "${player_cmd[@]}" > "${log_path}" 2>&1
}

wait_for_rosbag_playback() {
    local status=0

    if [ -n "${ROSBAG_PLAY_PID:-}" ]; then
        wait "${ROSBAG_PLAY_PID}" || status=$?
        ROSBAG_PLAY_PID=""
        ROSBAG_PLAY_USES_SETSID=false
    fi

    return "${status}"
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

convert_pgm_to_png() {
    local pgm_path="$1"
    local png_path="$2"

    if command -v magick >/dev/null 2>&1; then
        magick "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v convert >/dev/null 2>&1; then
        convert "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -loglevel error -i "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - "${pgm_path}" "${png_path}" <<'PY'
import sys

try:
    from PIL import Image
except Exception as exc:
    raise SystemExit(f"Pillow not available: {exc}")

src, dst = sys.argv[1], sys.argv[2]
Image.open(src).save(dst, format="PNG")
PY
        return $?
    fi

    return 1
}

update_yaml_image_path() {
    local yaml_path="$1"
    local image_path="$2"
    local tmp_yaml_path

    if [ ! -f "${yaml_path}" ]; then
        return 1
    fi

    tmp_yaml_path="${yaml_path}.tmp.$$"

    if ! awk -v image_path="${image_path}" '
        BEGIN { updated = 0 }
        /^[[:space:]]*image:[[:space:]]*/ && updated == 0 {
            match($0, /^[[:space:]]*/)
            indent = substr($0, 1, RLENGTH)
            print indent "image: " image_path
            updated = 1
            next
        }
        { print }
        END {
            if (updated == 0) {
                exit 2
            }
        }
    ' "${yaml_path}" > "${tmp_yaml_path}"; then
        rm -f "${tmp_yaml_path}" 2>/dev/null || true
        return 1
    fi

    mv "${tmp_yaml_path}" "${yaml_path}"
}

resolve_centerline_script() {
    if [ -n "${CENTERLINE_SCRIPT_PATH}" ]; then
        if [ -f "${CENTERLINE_SCRIPT_PATH}" ]; then
            echo "${CENTERLINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/generate_centerline.py"
}

resolve_raceline_script() {
    if [ -n "${RACELINE_SCRIPT_PATH}" ]; then
        if [ -f "${RACELINE_SCRIPT_PATH}" ]; then
            echo "${RACELINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/generate_raceline.py"
}

resolve_line_preview_script() {
    if [ -n "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
        if [ -f "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
            echo "${LINE_PREVIEW_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/visualize_race_lines.py"
}

resolve_map_edit_script() {
    if [ -n "${MAP_EDIT_SCRIPT_PATH}" ]; then
        if [ -f "${MAP_EDIT_SCRIPT_PATH}" ]; then
            echo "${MAP_EDIT_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "map_section_editor/map_cleanup_editor.py"
}

resolve_section_editor_script() {
    resolve_python_ws_file "map_section_editor/section_editor.py"
}

prompt_map_edit() {
    local edit_choice

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        MAP_EDIT_ENABLED=false
        return 0
    fi

    case "${MAP_EDIT_MODE}" in
        always)
            MAP_EDIT_ENABLED=true
            ;;
        never)
            MAP_EDIT_ENABLED=false
            ;;
        auto)
            MAP_EDIT_ENABLED=false
            if [ ! -t 0 ]; then
                return 0
            fi
            echo ""
            read -r -p "centerline前に map を手修正しますか？ (y/N, Enterでスキップ): " edit_choice
            if [[ "${edit_choice:-n}" =~ ^[Yy]$ ]]; then
                MAP_EDIT_ENABLED=true
            fi
            ;;
        *)
            echo "Invalid --map-edit-mode: ${MAP_EDIT_MODE}" >&2
            exit 1
            ;;
    esac
}

run_map_edit() {
    local input_map_path="$1"
    local map_edit_script_path
    local -a map_edit_cmd

    if [ "${MAP_EDIT_ENABLED}" != true ]; then
        echo "[prep] Skip GUI map cleanup"
        return 0
    fi

    echo "[prep] Launch GUI map cleanup"

    if [ ! -f "${input_map_path}" ]; then
        echo "Warning: map input not found for cleanup: ${input_map_path}" >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip GUI map cleanup." >&2
        return 0
    fi

    if ! map_edit_script_path="$(resolve_map_edit_script)"; then
        if [ -n "${MAP_EDIT_SCRIPT_PATH}" ]; then
            echo "Warning: map cleanup editor not found: ${MAP_EDIT_SCRIPT_PATH}" >&2
        else
            echo "Warning: map_cleanup_editor.py not found. Skip GUI map cleanup." >&2
        fi
        return 0
    fi

    if [ -z "${MAP_EDIT_OUTPUT_PATH}" ]; then
        MAP_EDIT_OUTPUT_PATH="${MAP_STEM}_centerline_input.png"
    fi

    map_edit_cmd=(
        python3 "${map_edit_script_path}"
        --input "${input_map_path}"
        --output "${MAP_EDIT_OUTPUT_PATH}"
    )

    if ! "${map_edit_cmd[@]}"; then
        echo "Warning: map cleanup editor failed. Keep original map for centerline." >&2
        return 0
    fi

    if [ -f "${MAP_EDIT_OUTPUT_PATH}" ]; then
        CENTERLINE_INPUT_MAP="${MAP_EDIT_OUTPUT_PATH}"
        echo "  - cleaned map: ${MAP_EDIT_OUTPUT_PATH}"
    else
        echo "Warning: cleaned map was not saved. Keep original map for centerline." >&2
    fi
}

run_section_edit() {
    local section_editor_script_path
    local -a section_editor_cmd

    echo "[post] Launch section editor"

    if [ ! -f "${MAP_YAML_PATH}" ]; then
        echo "Warning: map yaml not found for section edit: ${MAP_YAML_PATH}" >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip section edit." >&2
        return 0
    fi

    if ! section_editor_script_path="$(resolve_section_editor_script)"; then
        echo "Warning: section_editor.py not found. Skip section edit." >&2
        return 0
    fi

    section_editor_cmd=(
        python3 "${section_editor_script_path}"
        --map-yaml "${MAP_YAML_PATH}"
        --output "${SECTION_OUTPUT_PATH}"
    )

    if ! "${section_editor_cmd[@]}"; then
        echo "Warning: section editor failed." >&2
        return 0
    fi

    if [ -f "${SECTION_OUTPUT_PATH}" ]; then
        echo "  - sections: ${SECTION_OUTPUT_PATH}"
    else
        echo "Warning: section CSV was not saved." >&2
    fi

    if [ -f "${SECTION_GATE_OUTPUT_PATH}" ]; then
        echo "  - gates: ${SECTION_GATE_OUTPUT_PATH}"
    fi
}

prompt_pre_transfer_action() {
    local action_choice

    while true; do
        echo ""
        echo "転送前の操作を選んでください:"
        echo "  1) section edit を開く"
        echo "  2) scp 転送へ進む"
        echo "  3) 何もせず終了"
        read -r -p "選択 [2]: " action_choice

        case "${action_choice:-2}" in
            1|section|sections|edit|s)
                run_section_edit
                ;;
            2|transfer|scp|t)
                return 0
                ;;
            3|skip|exit|quit|q)
                echo "転送をスキップしました。"
                exit 0
                ;;
            *)
                echo "無効な選択です: ${action_choice}" >&2
                ;;
        esac
    done
}

generate_centerline() {
    local input_map_path="$1"
    local centerline_script_path
    local -a centerline_cmd

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        echo "[5/8] Skip centerline generation"
        return 0
    fi

    echo "[5/8] Generate centerline"

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip centerline generation." >&2
        return 0
    fi

    if ! centerline_script_path="$(resolve_centerline_script)"; then
        if [ -n "${CENTERLINE_SCRIPT_PATH}" ]; then
            echo "Warning: centerline script not found: ${CENTERLINE_SCRIPT_PATH}" >&2
        else
            echo "Warning: generate_centerline.py not found. Skip centerline generation." >&2
        fi
        return 0
    fi

    if [ "${CENTERLINE_DEBUG}" = true ] && [ -z "${CENTERLINE_DEBUG_DIR}" ]; then
        CENTERLINE_DEBUG_PATH="${MAP_STEM}_centerline_debug"
    else
        CENTERLINE_DEBUG_PATH="${CENTERLINE_DEBUG_DIR}"
    fi

    centerline_cmd=(
        python3 "${centerline_script_path}"
        --map "${input_map_path}"
        --output "${CENTERLINE_OUTPUT_PATH}"
        --yaml "${MAP_YAML_PATH}"
        --preset "${CENTERLINE_PRESET}"
        --direction "${CENTERLINE_DIRECTION}"
    )
    if [ -n "${CENTERLINE_DEBUG_PATH}" ]; then
        centerline_cmd+=(--debug-dir "${CENTERLINE_DEBUG_PATH}")
    fi

    if ! "${centerline_cmd[@]}"; then
        echo "Warning: centerline generation failed. Skip centerline output." >&2
        return 0
    fi

    CENTERLINE_CREATED=true
    echo "  - ${CENTERLINE_OUTPUT_PATH}"
    if [ "${CENTERLINE_DIRECTION}" = "both" ]; then
        echo "  - ${CENTERLINE_OUTPUT_PATH%.*}_reverse.${CENTERLINE_OUTPUT_PATH##*.}"
    fi
    if [ -n "${CENTERLINE_DEBUG_PATH}" ]; then
        echo "  - ${CENTERLINE_DEBUG_PATH}/"
    fi
}

generate_raceline() {
    local centerline_path="$1"
    local raceline_script_path
    local -a raceline_cmd

    if [ "${ENABLE_RACELINE}" != true ]; then
        echo "[6/8] Skip raceline generation"
        return 0
    fi

    echo "[6/8] Generate raceline"

    if [ "${CENTERLINE_CREATED}" != true ] || [ ! -f "${centerline_path}" ]; then
        echo "Warning: centerline CSV not found. Skip raceline generation." >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip raceline generation." >&2
        return 0
    fi

    if ! raceline_script_path="$(resolve_raceline_script)"; then
        if [ -n "${RACELINE_SCRIPT_PATH}" ]; then
            echo "Warning: raceline script not found: ${RACELINE_SCRIPT_PATH}" >&2
        else
            echo "Warning: generate_raceline.py not found. Skip raceline generation." >&2
        fi
        return 0
    fi

    raceline_cmd=(
        python3 "${raceline_script_path}"
        --preset "${RACELINE_PRESET}"
        --backend "${RACELINE_BACKEND}"
        --opt-type "${RACELINE_OPT_TYPE}"
        --centerline "${centerline_path}"
        --output "${RACELINE_OUTPUT_PATH}"
        --direction "${RACELINE_DIRECTION}"
    )
    if [ -n "${GLOBAL_OPTIMIZER_ROOT}" ]; then
        raceline_cmd+=(--optimizer-root "${GLOBAL_OPTIMIZER_ROOT}")
    fi

    if ! "${raceline_cmd[@]}"; then
        echo "Warning: raceline generation failed. Skip raceline output." >&2
        return 0
    fi

    RACELINE_CREATED=true
    echo "  - ${RACELINE_OUTPUT_PATH}"
    if [ "${RACELINE_DIRECTION}" = "both" ]; then
        echo "  - ${RACELINE_OUTPUT_PATH%.*}_reverse.${RACELINE_OUTPUT_PATH##*.}"
    fi
}

generate_line_preview() {
    local input_map_path="$1"
    local preview_script_path
    local -a preview_cmd

    if [ "${ENABLE_LINE_PREVIEW}" != true ]; then
        echo "[7/8] Skip line preview generation"
        return 0
    fi

    echo "[7/8] Generate line preview"

    if [ "${CENTERLINE_CREATED}" != true ] && [ "${RACELINE_CREATED}" != true ]; then
        echo "Warning: no centerline/raceline CSV found. Skip line preview generation." >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip line preview generation." >&2
        return 0
    fi

    if ! preview_script_path="$(resolve_line_preview_script)"; then
        if [ -n "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
            echo "Warning: line preview script not found: ${LINE_PREVIEW_SCRIPT_PATH}" >&2
        else
            echo "Warning: visualize_race_lines.py not found. Skip line preview generation." >&2
        fi
        return 0
    fi

    preview_cmd=(
        python3 "${preview_script_path}"
        --map "${input_map_path}"
        --yaml "${MAP_YAML_PATH}"
        --output "${LINE_PREVIEW_OUTPUT_PATH}"
    )
    if [ "${CENTERLINE_CREATED}" = true ] && [ -f "${CENTERLINE_OUTPUT_PATH}" ]; then
        preview_cmd+=(--centerline "${CENTERLINE_OUTPUT_PATH}")
    fi
    if [ "${RACELINE_CREATED}" = true ] && [ -f "${RACELINE_OUTPUT_PATH}" ]; then
        preview_cmd+=(--raceline "${RACELINE_OUTPUT_PATH}")
    fi

    if ! "${preview_cmd[@]}"; then
        echo "Warning: line preview generation failed. Skip line preview output." >&2
        return 0
    fi

    LINE_PREVIEW_CREATED=true
    echo "  - ${LINE_PREVIEW_OUTPUT_PATH}"
}

prompt_centerline_generation() {
    local generate_choice
    local debug_choice

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        return 0
    fi

    echo ""
    read -r -p "centerlineを生成しますか？ (Y/n, Enterで生成): " generate_choice
    generate_choice=${generate_choice:-y}

    if [[ ! "${generate_choice}" =~ ^[Yy]$ ]]; then
        ENABLE_CENTERLINE=false
        return 0
    fi

    echo ""
    read -r -p "debug画像も保存しますか？ (Y/n, Enterで保存): " debug_choice
    debug_choice=${debug_choice:-y}

    if [[ "${debug_choice}" =~ ^[Nn]$ ]]; then
        CENTERLINE_DEBUG=false
    else
        CENTERLINE_DEBUG=true
    fi
}

prompt_raceline_generation() {
    local generate_choice

    if [ "${ENABLE_CENTERLINE}" != true ] || [ "${ENABLE_RACELINE}" != true ]; then
        return 0
    fi

    echo ""
    read -r -p "racelineも生成しますか？ (Y/n, Enterで生成): " generate_choice
    generate_choice=${generate_choice:-y}

    if [[ ! "${generate_choice}" =~ ^[Yy]$ ]]; then
        ENABLE_RACELINE=false
    fi
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

measure_topic_rate_hz() {
    local topic_name="$1"
    local window_size="$2"
    local sample_timeout_sec="$3"
    local timeout_cmd
    local hz_output

    timeout_cmd="$(resolve_timeout_cmd)"
    if [ -z "${timeout_cmd}" ]; then
        return 1
    fi

    hz_output="$("${timeout_cmd}" "${sample_timeout_sec}s" \
        ros2 topic hz "${topic_name}" -w "${window_size}" 2>/dev/null || true)"

    printf '%s\n' "${hz_output}" | awk '/average rate:/ {print $3}' | tail -n1
}

wait_for_topic_rate_ready() {
    local topic_name="$1"
    local min_rate_hz="$2"
    local window_size="$3"
    local timeout_sec="$4"
    local start_ts
    local elapsed=0
    local sample_timeout_sec
    local current_timeout_sec
    local measured_rate

    if [ "${ODOM_READY_WAIT_ENABLED}" != true ]; then
        return 0
    fi

    if ! wait_for_topic "${topic_name}" "${timeout_sec}"; then
        return 1
    fi

    sample_timeout_sec="$(awk -v window_size="${window_size}" 'BEGIN {
        sample = int(window_size / 5.0) + 2
        if (sample < 3) {
            sample = 3
        }
        if (sample > 8) {
            sample = 8
        }
        print sample
    }')"

    start_ts="$(date +%s)"
    while [ "${elapsed}" -lt "${timeout_sec}" ]; do
        current_timeout_sec="${sample_timeout_sec}"
        if [ $((timeout_sec - elapsed)) -lt "${current_timeout_sec}" ]; then
            current_timeout_sec=$((timeout_sec - elapsed))
        fi
        if [ "${current_timeout_sec}" -lt 1 ]; then
            current_timeout_sec=1
        fi

        measured_rate="$(measure_topic_rate_hz "${topic_name}" "${window_size}" "${current_timeout_sec}" || true)"
        if [ -n "${measured_rate}" ]; then
            echo "  - odom rate sample: ${measured_rate} Hz (target >= ${min_rate_hz} Hz)"
            if float_ge "${measured_rate}" "${min_rate_hz}"; then
                return 0
            fi
        else
            echo "  - odom rate sample: waiting for ${window_size} messages on ${topic_name}"
        fi

        sleep 1
        elapsed=$(( $(date +%s) - start_ts ))
    done

    return 1
}

launch_cartographer_mapping() {
    local -a launch_args=(
        "use_sim_time:=true"
        "scan_topic:=${SCAN_TOPIC}"
        "configuration_basename:=${CONFIG_BASENAME}"
    )

    if [ -n "${ODOM_TOPIC}" ]; then
        launch_args+=("odom_topic:=${ODOM_TOPIC}")
    fi

    capture_base_cartographer_pids

    if command -v setsid >/dev/null 2>&1; then
        build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
        CARTOGRAPHER_USES_SETSID=true
        setsid "${SYSTEM_LAUNCH_CMD[@]}" \
            "${launch_args[@]}" \
            > "${MAP_LOG_PATH}" 2>&1 &
    else
        build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
        CARTOGRAPHER_USES_SETSID=false
        "${SYSTEM_LAUNCH_CMD[@]}" \
            "${launch_args[@]}" \
            > "${MAP_LOG_PATH}" 2>&1 &
    fi

    CARTOGRAPHER_PID="$!"
}

convert_pbstream_to_map() {
    local success_label="$1"
    local png_filename
    local png_created=false
    local yaml_image_updated=false

    if ! ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
        -pbstream_filename "${PBSTREAM_PATH}" \
        -map_filestem "${MAP_STEM}" \
        -resolution 0.05; then
        echo "pbstream generated: ${PBSTREAM_PATH}" >&2
        echo "Failed to convert pbstream to occupancy map." >&2
        return 1
    fi

    png_filename="$(basename "${MAP_PNG_PATH}")"
    if convert_pgm_to_png "${MAP_PGM_PATH}" "${MAP_PNG_PATH}"; then
        png_created=true
        if update_yaml_image_path "${MAP_YAML_PATH}" "${png_filename}"; then
            yaml_image_updated=true
        else
            echo "Warning: PNG was generated, but failed to update image path in ${MAP_YAML_PATH}." >&2
        fi
    else
        echo "Warning: PNG conversion skipped. (Need one of: magick/convert/ffmpeg/python3+Pillow)" >&2
    fi

    echo ""
    echo "${success_label}"
    echo "  - ${MAP_YAML_PATH}"
    if [ "${yaml_image_updated}" = true ]; then
        echo "    image: ${png_filename}"
    fi
    echo "  - ${MAP_PGM_PATH}"
    if [ "${png_created}" = true ]; then
        echo "  - ${MAP_PNG_PATH}"
    fi
    echo "  - ${PBSTREAM_PATH}"
    if [ -d "${VSLAM_MAP_DIR}" ]; then
        echo "  - ${VSLAM_MAP_DIR}/"
    fi
    if [ "${OFFLINE_ODOM_BAG_CREATED}" = true ]; then
        echo "  - ${OFFLINE_ODOM_BAG_DIR}/"
    fi
}

cleanup_all() {
    stop_rosbag_playback
    stop_recorder
    stop_vslam
    stop_cartographer
    if [ -n "${VSLAM_LOCALIZATION_PARAM_PATH:-}" ] && [ -f "${VSLAM_LOCALIZATION_PARAM_PATH}" ]; then
        rm -f "${VSLAM_LOCALIZATION_PARAM_PATH}"
        VSLAM_LOCALIZATION_PARAM_PATH=""
    fi
}

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

if [ "${PIPELINE_MODE}" = "online" ]; then
    mkdir -p "${VSLAM_MAP_DIR}"
fi

if [ "${PIPELINE_MODE}" = "offline" ] && [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
    prepare_offline_vslam_odom_bag
fi

print_mode_summary

# ==========================================
# 1. Build maps
# ==========================================
if [ "${PIPELINE_MODE}" = "online" ]; then
    echo "[1/5] Launch online VSLAM for map creation (logs: ${TF_LOG_PATH}, ${VSLAM_LOG_PATH})"
    launch_vslam_stack \
        "${TF_LOG_PATH}" \
        "${VSLAM_LOG_PATH}" \
        "${VSLAM_MAP_DIR}"

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

    echo "[4/5] Play source rosbag for online VSLAM + Cartographer"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}"; then
            exit 1
        fi
    else
        build_online_source_play_topics
        PLAY_TOPICS=("${SOURCE_PLAY_TOPICS[@]}")

        echo "  - mode: filtered topics"
        echo "  - topics: ${PLAY_TOPICS[*]}"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${VSLAM_PLAYER_LOG_PATH}" "${PLAY_TOPICS[@]}"; then
            exit 1
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
