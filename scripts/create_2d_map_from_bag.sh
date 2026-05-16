#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYSTEM_LAUNCH_SOURCE_SHARE="${REPO_ROOT}/ros2_ws/src/launch/system_launch"
LOCALIZATION_MANAGER_SOURCE_SHARE="${REPO_ROOT}/ros2_ws/src/localization/localization_manager"
SYSTEM_LAUNCH_CMD=()

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
  --mode NAME         default|2d_slam (default: default)
  --bag-path DIR      input rosbag2 directory (skip interactive selection)
  --map-name NAME     output map name (skip interactive prompt)
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  use odometry topic and enable odometry in cartographer
  --use-vslam-odom    shorthand of --odom-topic /visual_slam/tracking/odometry
  --run-vslam         replay stereo/imu topics and generate /visual_slam/tracking/odometry offline
  --no-vslam          do not launch offline vslam even if --use-vslam-odom is specified
  --vslam-vis         enable VSLAM visualization topics and record them into the lightweight bag
  --no-vslam-vis      disable VSLAM visualization topics (default)
  --image-width PX    camera width for offline vslam launch (default: 424)
  --image-height PX   camera height for offline vslam launch (default: 240)
  --image-fps FPS     camera fps for offline vslam launch (default: 90.0)
  --with-imu          replay /camera/imu as well (default: disabled)
  --vslam-map-dir DIR visual slam map output directory
  --lightweight-bag-root DIR
                      2D map用 lightweight bag の保存先ルート (default: /record/2d_input)
  --pipeline-mode MODE
                      full|fast|auto (default: auto)
  --play-all-topics   play all topics in bag (default: play only needed topics)
  --vslam-hint-remap  after the provisional 2D map, replay the source bag once more,
                      feed scan global-localization result into /visual_slam/initial_pose,
                      then rewind the same rosbag player and rebuild the map (default)
  --no-vslam-hint-remap
                      skip the hinted VSLAM remap pass
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

Outputs:
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pbstream
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pgm
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.png (optional; generated if converter is available)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline_input.png (optional; hand-edited map cleanup result)
  /record/2d_input/<bag_name>/<MAP_NAME>_2d_input_<timestamp>/ (when --run-vslam or --use-vslam-odom)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline.csv (optional; generated unless --no-centerline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_raceline.csv (optional; generated unless --no-raceline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_lines.png (optional; generated unless --no-line-preview)

After map creation:
  optionally transfer /map/<bag_name>/<MAP_NAME>/ to remote host by scp

Interactive flow:
  1) /record を再帰探索して metadata.yaml を持つ rosbag2 ディレクトリを一覧表示
  2) 番号で rosbag を選択
  3) map 名を入力
  4) Cartographer -> PNG変換
  5) centerline 生成の可否を確認（debug はデフォルト有効）
  6) 必要なら GUI で map PNG/PGM を黒塗り修正して保存
  7) raceline 生成の可否を確認
  8) centerline / raceline preview画像を生成
  9) 転送前メニューで section edit / scp / 終了 を選択
EOF
}

# ==========================================
# Default settings
# ==========================================
SCAN_TOPIC="/scan"
PLAY_RATE="1.0"
ODOM_TOPIC=""
CONFIG_BASENAME="cartographer_2d.lua"
MODE="default"
RUN_VSLAM=false
VSLAM_VIS_ENABLED=false
PLAY_ALL_TOPICS=false
RUN_VSLAM_HINT_REMAP=true
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
CENTERLINE_DIRECTION="forward"
RACELINE_SCRIPT_PATH=""
RACELINE_BACKEND="auto"
RACELINE_OPT_TYPE="mincurv_iqp"
RACELINE_DIRECTION="forward"
GLOBAL_OPTIMIZER_ROOT=""
LINE_PREVIEW_SCRIPT_PATH=""
RECORD_ROOT="/record"
LIGHTWEIGHT_BAG_ROOT="/record/2d_input"
PIPELINE_MODE="auto"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
IMAGE_FPS="90.0"
USE_IMU=false
CAMERA_CONTAINER_NAME="offline_camera_container_$$"
LOCALIZATION_CONTAINER_NAME="offline_localization_container_$$"
VSLAM_MAP_DIR=""
LIGHTWEIGHT_BAG_DIR=""
USING_LIGHTWEIGHT_BAG=false
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
LIDAR_CONTAINER_PID=""
LIDAR_CONTAINER_USES_SETSID=false
VSLAM_LAUNCH_PID=""
VSLAM_LAUNCH_USES_SETSID=false
LOCALIZATION_LAUNCH_PID=""
LOCALIZATION_LAUNCH_USES_SETSID=false
LOCALIZATION_MANAGER_PID=""
LOCALIZATION_MANAGER_USES_SETSID=false
RECORDER_PID=""
RECORDER_USES_SETSID=false
ROSBAG_PLAYER_PID=""
ROSBAG_PLAYER_USES_SETSID=false
BASE_CARTOGRAPHER_PIDS=()
ROSBAG_CANDIDATES=()
LIGHTWEIGHT_BAG_CANDIDATES=()
LIGHTWEIGHT_RECORD_TOPICS=()
SOURCE_PLAY_TOPICS=()

apply_mode() {
    case "$1" in
        default)
            ;;
        2d_slam)
            ODOM_TOPIC="/visual_slam/tracking/odometry"
            CONFIG_BASENAME="cartographer_2d_with_odom.lua"
            RUN_VSLAM=true
            ;;
        *)
            echo "Unknown mode: $1" >&2
            usage
            exit 1
            ;;
    esac
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
            CONFIG_BASENAME="cartographer_2d_with_odom.lua"
            shift 2
            ;;
        --use-vslam-odom)
            ODOM_TOPIC="/visual_slam/tracking/odometry"
            CONFIG_BASENAME="cartographer_2d_with_odom.lua"
            RUN_VSLAM=true
            shift
            ;;
        --run-vslam)
            RUN_VSLAM=true
            shift
            ;;
        --no-vslam)
            RUN_VSLAM=false
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
        --lightweight-bag-root)
            LIGHTWEIGHT_BAG_ROOT="$2"
            shift 2
            ;;
        --pipeline-mode)
            PIPELINE_MODE="$2"
            shift 2
            ;;
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --vslam-hint-remap)
            RUN_VSLAM_HINT_REMAP=true
            shift
            ;;
        --no-vslam-hint-remap)
            RUN_VSLAM_HINT_REMAP=false
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

apply_mode "${MODE}"

BAG_PATH=""
MAP_NAME=""
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

discover_lightweight_bag_candidates() {
    local search_root="$1"
    local bag_name="$2"
    local map_name="$3"
    local metadata_path
    local dir
    local expected_prefix="${map_name}_2d_input_"

    LIGHTWEIGHT_BAG_CANDIDATES=()
    if [ ! -d "${search_root%/}/${bag_name}" ]; then
        return 1
    fi

    while IFS= read -r -d '' metadata_path; do
        dir="$(dirname "${metadata_path}")"
        if [[ "$(basename "${dir}")" == "${expected_prefix}"* ]]; then
            LIGHTWEIGHT_BAG_CANDIDATES+=("${dir}")
        fi
    done < <(find "${search_root%/}/${bag_name}" -type f -name metadata.yaml -print0 2>/dev/null)

    if [ "${#LIGHTWEIGHT_BAG_CANDIDATES[@]}" -eq 0 ]; then
        return 1
    fi

    mapfile -t LIGHTWEIGHT_BAG_CANDIDATES < <(printf '%s\n' "${LIGHTWEIGHT_BAG_CANDIDATES[@]}" | sort -r)
    [ "${#LIGHTWEIGHT_BAG_CANDIDATES[@]}" -gt 0 ]
}

build_lightweight_record_topics() {
    LIGHTWEIGHT_RECORD_TOPICS=(
        /visual_slam/tracking/odometry
        "${SCAN_TOPIC}"
        /tf
        /tf_static
    )

    if [ "${VSLAM_VIS_ENABLED}" = true ]; then
        LIGHTWEIGHT_RECORD_TOPICS+=(
            /visual_slam/status
            /visual_slam/tracking/vo_pose
            /visual_slam/tracking/vo_path
            /visual_slam/tracking/slam_path
            /visual_slam/vis/slam_odometry
            /visual_slam/vis/velocity
            /visual_slam/vis/gravity
            /visual_slam/vis/landmarks_cloud
            /visual_slam/vis/observations_cloud
            /visual_slam/vis/loop_closure_cloud
            /visual_slam/vis/pose_graph_nodes
            /visual_slam/vis/pose_graph_edges
            /visual_slam/vis/pose_graph_edges2
            /visual_slam/vis/localizer
            /visual_slam/vis/localizer_map_cloud
        )
    fi
}

build_source_play_topics() {
    SOURCE_PLAY_TOPICS=(
        "${SCAN_TOPIC}"
        "/tf"
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

prompt_pipeline_mode_interactive() {
    local choice

    if [ "${PIPELINE_MODE}" = "full" ] || [ "${PIPELINE_MODE}" = "fast" ]; then
        return 0
    fi

    if [ "${#LIGHTWEIGHT_BAG_CANDIDATES[@]}" -eq 0 ]; then
        PIPELINE_MODE="full"
        return 0
    fi

    echo ""
    echo "既存の lightweight bag が見つかりました:"
    printf "  1) full  source bag から VSLAM と 2D map を作り直す\n"
    printf "  2) fast  既存の lightweight bag を使って 2D map から再開\n"
    echo "      latest: ${LIGHTWEIGHT_BAG_CANDIDATES[0]}"
    echo ""

    while :; do
        read -r -p "実行モードを選択 (1-2, Enterで fast): " choice
        choice=${choice:-2}
        case "${choice}" in
            1)
                PIPELINE_MODE="full"
                return 0
                ;;
            2)
                PIPELINE_MODE="fast"
                return 0
                ;;
        esac
        echo "無効な入力です。1 または 2 を選択してください。"
    done
}

select_lightweight_bag_interactive() {
    local choice
    local i

    if [ "${#LIGHTWEIGHT_BAG_CANDIDATES[@]}" -eq 0 ]; then
        echo "No lightweight bag candidates found." >&2
        return 1
    fi

    echo ""
    echo "利用する lightweight bag を選択してください:"
    for i in "${!LIGHTWEIGHT_BAG_CANDIDATES[@]}"; do
        if [ "$i" -eq 0 ]; then
            printf "  %2d) %s (latest, Enterのデフォルト)\n" "$((i + 1))" "${LIGHTWEIGHT_BAG_CANDIDATES[$i]}"
        else
            printf "  %2d) %s\n" "$((i + 1))" "${LIGHTWEIGHT_BAG_CANDIDATES[$i]}"
        fi
    done
    echo ""

    while :; do
        read -r -p "lightweight bag を番号で選択 (1-${#LIGHTWEIGHT_BAG_CANDIDATES[@]}, Enterで1): " choice
        choice=${choice:-1}
        if [[ "${choice}" =~ ^[0-9]+$ ]] && \
           [ "${choice}" -ge 1 ] && \
           [ "${choice}" -le "${#LIGHTWEIGHT_BAG_CANDIDATES[@]}" ]; then
            LIGHTWEIGHT_BAG_DIR="${LIGHTWEIGHT_BAG_CANDIDATES[$((choice - 1))]}"
            return 0
        fi
        echo "無効な入力です。番号で選択してください。"
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

if [ "${RUN_VSLAM}" = true ] && [ "${ODOM_TOPIC}" = "/visual_slam/tracking/odometry" ]; then
    discover_lightweight_bag_candidates "${LIGHTWEIGHT_BAG_ROOT}" "${BAG_DIR_NAME}" "${MAP_NAME}" || true
    prompt_pipeline_mode_interactive
    if [ "${PIPELINE_MODE}" = "fast" ]; then
        select_lightweight_bag_interactive
        RUN_VSLAM=false
        USING_LIGHTWEIGHT_BAG=true
        BAG_PATH="${LIGHTWEIGHT_BAG_DIR}"
    fi
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
    local script_dir
    local candidate

    if [ -n "${CENTERLINE_SCRIPT_PATH}" ]; then
        if [ -f "${CENTERLINE_SCRIPT_PATH}" ]; then
            echo "${CENTERLINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "/python_ws/data_analysis/generate_centerline.py" \
        "${script_dir}/../python_ws/data_analysis/generate_centerline.py" \
        "${PWD}/python_ws/data_analysis/generate_centerline.py"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

resolve_raceline_script() {
    local script_dir
    local candidate

    if [ -n "${RACELINE_SCRIPT_PATH}" ]; then
        if [ -f "${RACELINE_SCRIPT_PATH}" ]; then
            echo "${RACELINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "/python_ws/data_analysis/generate_raceline.py" \
        "${script_dir}/../python_ws/data_analysis/generate_raceline.py" \
        "${PWD}/python_ws/data_analysis/generate_raceline.py"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

resolve_line_preview_script() {
    local script_dir
    local candidate

    if [ -n "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
        if [ -f "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
            echo "${LINE_PREVIEW_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "/python_ws/data_analysis/visualize_race_lines.py" \
        "${script_dir}/../python_ws/data_analysis/visualize_race_lines.py" \
        "${PWD}/python_ws/data_analysis/visualize_race_lines.py"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

resolve_map_edit_script() {
    local script_dir
    local candidate

    if [ -n "${MAP_EDIT_SCRIPT_PATH}" ]; then
        if [ -f "${MAP_EDIT_SCRIPT_PATH}" ]; then
            echo "${MAP_EDIT_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "/python_ws/map_section_editor/map_cleanup_editor.py" \
        "${script_dir}/../python_ws/map_section_editor/map_cleanup_editor.py" \
        "${PWD}/python_ws/map_section_editor/map_cleanup_editor.py"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

resolve_section_editor_script() {
    local script_dir
    local candidate

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "/python_ws/map_section_editor/section_editor.py" \
        "${script_dir}/../python_ws/map_section_editor/section_editor.py" \
        "${PWD}/python_ws/map_section_editor/section_editor.py"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
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

wait_for_topic_message() {
    local topic_name="$1"
    local timeout_sec="$2"
    local output_path="$3"
    local topic_echo_pid=""
    local count=0

    rm -f "${output_path}" 2>/dev/null || true

    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout_sec}s" ros2 topic echo --once "${topic_name}" > "${output_path}" 2>/dev/null
        return $?
    fi

    ros2 topic echo --once "${topic_name}" > "${output_path}" 2>/dev/null &
    topic_echo_pid="$!"

    while kill -0 "${topic_echo_pid}" 2>/dev/null; do
        if [ "${count}" -ge "${timeout_sec}" ]; then
            kill "${topic_echo_pid}" 2>/dev/null || true
            wait "${topic_echo_pid}" 2>/dev/null || true
            return 1
        fi
        sleep 1
        count=$((count + 1))
    done

    wait "${topic_echo_pid}" 2>/dev/null
}

call_rosbag_player_service() {
    local service_suffix="$1"
    local service_type="$2"
    local request="${3:-{}}"

    ros2 service call "/rosbag2_player/${service_suffix}" "${service_type}" \
        "${request}" > /dev/null
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
    if [ -d "${LIGHTWEIGHT_BAG_DIR}" ]; then
        echo "  - ${LIGHTWEIGHT_BAG_DIR}"
    fi
}

run_vslam_hint_remap_pass() {
    local player_log_path="/tmp/offline_vslam_hint_player_$(date +%Y%m%d_%H%M%S).log"
    local localization_log_path="/tmp/offline_vslam_hint_localization_$(date +%Y%m%d_%H%M%S).log"
    local localization_manager_log_path="/tmp/offline_vslam_hint_localization_manager_$(date +%Y%m%d_%H%M%S).log"
    local vslam_log_path="/tmp/offline_vslam_hint_vslam_$(date +%Y%m%d_%H%M%S).log"
    local tf_log_path="/tmp/offline_vslam_hint_tf_$(date +%Y%m%d_%H%M%S).log"
    local map_log_path="/tmp/offline_vslam_hint_cartographer_$(date +%Y%m%d_%H%M%S).log"
    local initial_pose_wait_log="/tmp/offline_vslam_hint_initial_pose_$(date +%Y%m%d_%H%M%S).log"
    local localization_manager_launch
    local hint_wait_pid=""

    if [ "${RUN_VSLAM_HINT_REMAP}" != true ]; then
        return 0
    fi

    if [ "${ODOM_TOPIC}" != "/visual_slam/tracking/odometry" ]; then
        echo "[hint-remap] Skip: odom topic is not VSLAM odometry (${ODOM_TOPIC:-<empty>})"
        return 0
    fi

    if [ ! -d "${VSLAM_MAP_DIR}" ]; then
        echo "[hint-remap] Skip: VSLAM map directory not found: ${VSLAM_MAP_DIR}" >&2
        return 0
    fi

    if [ ! -f "${MAP_YAML_PATH}" ]; then
        echo "[hint-remap] Skip: provisional map yaml not found: ${MAP_YAML_PATH}" >&2
        return 0
    fi

    echo "[hint-remap] Launch VSLAM relocalization + scan global localization"

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
        "image_width:=${IMAGE_WIDTH}" \
        "image_height:=${IMAGE_HEIGHT}" \
        "camera_container_name:=${CAMERA_CONTAINER_NAME}" \
        "enable_localization_and_mapping:=true" \
        "enable_slam_visualization:=${VSLAM_VIS_ENABLED}" \
        "enable_observations_view:=${VSLAM_VIS_ENABLED}" \
        "enable_landmarks_view:=${VSLAM_VIS_ENABLED}" \
        "load_map_path:=${VSLAM_MAP_DIR}" \
        "save_map_path:=${VSLAM_MAP_DIR}" \
        > "${vslam_log_path}" 2>&1

    echo "[hint-remap] Wait for VSLAM services"
    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Warning: hint remap skipped because VSLAM service was not ready. Check log: ${vslam_log_path}" >&2
        stop_vslam
        return 0
    fi

    launch_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID" \
        ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${LOCALIZATION_CONTAINER_NAME}"
    sleep 2

    build_system_launch_cmd "localization.launch.xml"
    launch_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        "lidar_container_name:=${LOCALIZATION_CONTAINER_NAME}" \
        "map_yaml_path:=${MAP_YAML_PATH}" \
        "scan_topic:=${SCAN_TOPIC}" \
        "use_sim_time:=true" \
        "publish_map:=false" \
        "use_localization_manager:=false" \
        "publish_localization_tf:=false" \
        > "${localization_log_path}" 2>&1

    localization_manager_launch="${LOCALIZATION_MANAGER_SOURCE_SHARE}/launch/localization_manager.launch.xml"
    if [ -f "${localization_manager_launch}" ]; then
        launch_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID" \
            ros2 launch "${localization_manager_launch}" \
            "use_sim_time:=true" \
            "publish_localization_tf:=false" \
            "publish_initialpose_to_amcl:=true" \
            "initial_pose_topic:=/visual_slam/initial_pose" \
            "localization_result_topic:=/localization_result" \
            > "${localization_manager_log_path}" 2>&1
    else
        launch_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID" \
            ros2 launch localization_manager localization_manager.launch.xml \
            "use_sim_time:=true" \
            "publish_localization_tf:=false" \
            "publish_initialpose_to_amcl:=true" \
            "initial_pose_topic:=/visual_slam/initial_pose" \
            "localization_result_topic:=/localization_result" \
            > "${localization_manager_log_path}" 2>&1
    fi

    echo "[hint-remap] Wait for localization and rosbag-player services"
    if ! wait_for_service "/trigger_grid_search_localization" 60; then
        echo "Warning: hint remap skipped because /trigger_grid_search_localization was not ready. Check log: ${localization_log_path}" >&2
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    echo "[hint-remap] Start paused source rosbag player"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        launch_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID" \
            ros2 bag play "${SOURCE_BAG_PATH}" --clock --rate "${PLAY_RATE}" --start-paused \
            > "${player_log_path}" 2>&1
    else
        build_source_play_topics
        launch_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID" \
            ros2 bag play "${SOURCE_BAG_PATH}" --clock --rate "${PLAY_RATE}" --start-paused --topics "${SOURCE_PLAY_TOPICS[@]}" \
            > "${player_log_path}" 2>&1
    fi

    if ! wait_for_service "/rosbag2_player/resume" 30 || \
       ! wait_for_service "/rosbag2_player/pause" 30 || \
       ! wait_for_service "/rosbag2_player/seek" 30; then
        echo "Warning: hint remap skipped because rosbag2 player services were not ready. Check log: ${player_log_path}" >&2
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    echo "[hint-remap] Resume briefly, trigger scan global localization, and wait for /visual_slam/initial_pose"
    wait_for_topic_message "/visual_slam/initial_pose" 45 "${initial_pose_wait_log}" &
    hint_wait_pid="$!"

    if ! call_rosbag_player_service "resume" "rosbag2_interfaces/srv/Resume" "{}"; then
        echo "Warning: failed to resume paused rosbag player for hint remap." >&2
        kill "${hint_wait_pid}" 2>/dev/null || true
        wait "${hint_wait_pid}" 2>/dev/null || true
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    sleep 1
    if ! ros2 service call /trigger_grid_search_localization std_srvs/srv/Empty "{}" > /dev/null; then
        echo "Warning: failed to trigger scan global localization for hint remap." >&2
        kill "${hint_wait_pid}" 2>/dev/null || true
        wait "${hint_wait_pid}" 2>/dev/null || true
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    if ! wait "${hint_wait_pid}"; then
        echo "Warning: did not receive /visual_slam/initial_pose during hint remap. Keeping provisional map." >&2
        call_rosbag_player_service "pause" "rosbag2_interfaces/srv/Pause" "{}" 2>/dev/null || true
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    call_rosbag_player_service "pause" "rosbag2_interfaces/srv/Pause" "{}" || true
    sleep 2

    echo "[hint-remap] Rewind paused rosbag player to start and rebuild the map"
    if ! call_rosbag_player_service "seek" "rosbag2_interfaces/srv/Seek" "{time: {sec: 0, nanosec: 0}}"; then
        echo "Warning: failed to rewind rosbag player for hint remap. Keeping provisional map." >&2
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        return 0
    fi

    capture_base_cartographer_pids
    if command -v setsid >/dev/null 2>&1; then
        build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
        CARTOGRAPHER_USES_SETSID=true
        setsid "${SYSTEM_LAUNCH_CMD[@]}" \
            "use_sim_time:=true" \
            "scan_topic:=${SCAN_TOPIC}" \
            "configuration_basename:=${CONFIG_BASENAME}" \
            "odom_topic:=${ODOM_TOPIC}" \
            > "${map_log_path}" 2>&1 &
    else
        build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
        CARTOGRAPHER_USES_SETSID=false
        "${SYSTEM_LAUNCH_CMD[@]}" \
            "use_sim_time:=true" \
            "scan_topic:=${SCAN_TOPIC}" \
            "configuration_basename:=${CONFIG_BASENAME}" \
            "odom_topic:=${ODOM_TOPIC}" \
            > "${map_log_path}" 2>&1 &
    fi
    CARTOGRAPHER_PID=$!

    if ! wait_for_service "/write_state" 60; then
        echo "Warning: hinted cartographer pass skipped because /write_state was not ready. Check log: ${map_log_path}" >&2
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        stop_cartographer
        return 0
    fi

    if ! call_rosbag_player_service "resume" "rosbag2_interfaces/srv/Resume" "{}"; then
        echo "Warning: failed to resume rewound rosbag player for hinted remap." >&2
        stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
        stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
        stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
        stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
        stop_vslam
        stop_cartographer
        return 0
    fi

    wait "${ROSBAG_PLAYER_PID}" 2>/dev/null || true
    ROSBAG_PLAYER_PID=""
    ROSBAG_PLAYER_USES_SETSID=false

    if ! ros2 service call /finish_trajectory \
        cartographer_ros_msgs/srv/FinishTrajectory \
        "{trajectory_id: 0}" > /dev/null; then
        echo "Warning: /finish_trajectory failed after hinted remap. Continue." >&2
    fi

    WRITE_STATE_REQUEST=$(printf "{filename: '%s', include_unfinished_submaps: true}" "${PBSTREAM_PATH}")
    ros2 service call /write_state \
        cartographer_ros_msgs/srv/WriteState \
        "${WRITE_STATE_REQUEST}" > /dev/null

    if ! ros2 service call /visual_slam/save_map \
        isaac_ros_visual_slam_interfaces/srv/FilePath \
        "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null; then
        echo "Warning: failed to save hinted VSLAM map. Continue with updated 2D map only." >&2
    fi

    stop_cartographer
    stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
    stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
    stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
    stop_vslam

    if ! convert_pbstream_to_map "✅ Hinted remap generated:"; then
        exit 1
    fi
}

cleanup_all() {
    stop_background_process "ROSBAG_PLAYER_PID" "ROSBAG_PLAYER_USES_SETSID"
    stop_recorder
    stop_background_process "LOCALIZATION_MANAGER_PID" "LOCALIZATION_MANAGER_USES_SETSID"
    stop_background_process "LOCALIZATION_LAUNCH_PID" "LOCALIZATION_LAUNCH_USES_SETSID"
    stop_background_process "LIDAR_CONTAINER_PID" "LIDAR_CONTAINER_USES_SETSID"
    stop_vslam
    stop_cartographer
}

trap cleanup_all EXIT INT TERM

if [ -z "${VSLAM_MAP_DIR}" ]; then
    VSLAM_MAP_DIR="${OUT_DIR}/cuvslam_map"
fi
if [ -z "${LIGHTWEIGHT_BAG_DIR}" ]; then
    if [ "${VSLAM_VIS_ENABLED}" = true ]; then
        LIGHTWEIGHT_BAG_DIR="${LIGHTWEIGHT_BAG_ROOT%/}/${BAG_DIR_NAME}/${MAP_NAME}_2d_input_vis_$(date +%Y%m%d_%H%M%S)"
    else
        LIGHTWEIGHT_BAG_DIR="${LIGHTWEIGHT_BAG_ROOT%/}/${BAG_DIR_NAME}/${MAP_NAME}_2d_input_$(date +%Y%m%d_%H%M%S)"
    fi
fi
VSLAM_LOG_PATH="/tmp/offline_vslam_mapping_$(date +%Y%m%d_%H%M%S).log"
TF_LOG_PATH="/tmp/offline_vslam_tf_$(date +%Y%m%d_%H%M%S).log"

if [ "${RUN_VSLAM}" = true ]; then
    mkdir -p "${VSLAM_MAP_DIR}"
    mkdir -p "$(dirname "${LIGHTWEIGHT_BAG_DIR}")"
fi

if [ "${RUN_VSLAM}" = true ] && [ "${ODOM_TOPIC}" != "/visual_slam/tracking/odometry" ]; then
    echo "Warning: --run-vslam is enabled but odom topic is '${ODOM_TOPIC}'. Cartographer may not consume vslam odometry." >&2
fi

# ==========================================
# 1. Generate VSLAM odom + lightweight bag
# ==========================================
if [ "${RUN_VSLAM}" = true ]; then
    echo "[1/8] Launch offline TF + vslam (logs: ${TF_LOG_PATH}, ${VSLAM_LOG_PATH})"
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
        "image_width:=${IMAGE_WIDTH}" \
        "image_height:=${IMAGE_HEIGHT}" \
        "camera_container_name:=${CAMERA_CONTAINER_NAME}" \
        "enable_localization_and_mapping:=true" \
        "enable_slam_visualization:=${VSLAM_VIS_ENABLED}" \
        "enable_observations_view:=${VSLAM_VIS_ENABLED}" \
        "enable_landmarks_view:=${VSLAM_VIS_ENABLED}" \
        "save_map_path:=${VSLAM_MAP_DIR}" \
        > "${VSLAM_LOG_PATH}" 2>&1

    echo "[2/8] Wait for /visual_slam/save_map service"
    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready. Check log: ${VSLAM_LOG_PATH}" >&2
        exit 1
    fi

    echo "[3/8] Start lightweight rosbag record: ${LIGHTWEIGHT_BAG_DIR}"
    build_lightweight_record_topics
    echo "  - record vslam vis: ${VSLAM_VIS_ENABLED}"
    echo "  - topics: ${LIGHTWEIGHT_RECORD_TOPICS[*]}"
    launch_background_process "RECORDER_PID" "RECORDER_USES_SETSID" \
        ros2 bag record \
        -o "${LIGHTWEIGHT_BAG_DIR}" \
        "${LIGHTWEIGHT_RECORD_TOPICS[@]}"

    sleep 2

    echo "[4/8] Play source rosbag for offline VSLAM"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        ros2 bag play "${SOURCE_BAG_PATH}" --clock --rate "${PLAY_RATE}"
    else
        build_source_play_topics
        PLAY_TOPICS=("${SOURCE_PLAY_TOPICS[@]}")

        echo "  - mode: filtered topics"
        echo "  - topics: ${PLAY_TOPICS[*]}"
        ros2 bag play "${SOURCE_BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}"
    fi

    echo "[5/8] Save VSLAM map and stop VSLAM-side processes"
    sleep 2
    ros2 service call /visual_slam/save_map \
        isaac_ros_visual_slam_interfaces/srv/FilePath \
        "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null

    stop_recorder
    stop_vslam
    USING_LIGHTWEIGHT_BAG=true
    BAG_PATH="${LIGHTWEIGHT_BAG_DIR}"
else
    echo "[1/8] Skip offline VSLAM"
fi

# ==========================================
# 6. Launch cartographer with finalized odom bag
# ==========================================
echo "[6/8] Launch cartographer (log: ${MAP_LOG_PATH})"
capture_base_cartographer_pids

LAUNCH_ARGS=(
    "use_sim_time:=true"
    "scan_topic:=${SCAN_TOPIC}"
    "configuration_basename:=${CONFIG_BASENAME}"
)

if [ -n "${ODOM_TOPIC}" ]; then
    LAUNCH_ARGS+=("odom_topic:=${ODOM_TOPIC}")
fi

if command -v setsid >/dev/null 2>&1; then
    build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
    CARTOGRAPHER_USES_SETSID=true
    setsid "${SYSTEM_LAUNCH_CMD[@]}" \
        "${LAUNCH_ARGS[@]}" \
        > "${MAP_LOG_PATH}" 2>&1 &
else
    build_system_launch_cmd "cartographer_2d_mapping.launch.xml"
    CARTOGRAPHER_USES_SETSID=false
    "${SYSTEM_LAUNCH_CMD[@]}" \
        "${LAUNCH_ARGS[@]}" \
        > "${MAP_LOG_PATH}" 2>&1 &
fi

CARTOGRAPHER_PID=$!

echo "[7/8] Wait for /write_state service"
if ! wait_for_service "/write_state" 60; then
    echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
    exit 1
fi

echo "[8/8] Play bag for Cartographer and save 2D map"
if [ "${USING_LIGHTWEIGHT_BAG}" = true ]; then
    echo "  - using lightweight bag: ${BAG_PATH}"
fi

if [ "${PLAY_ALL_TOPICS}" = true ]; then
    echo "  - mode: all topics"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}"
else
    # Cartographer only needs the scan, odometry topic, and static sensor TFs.
    # Replaying VSLAM's dynamic /tf here can conflict with Cartographer's own TF output.
    PLAY_TOPICS=("${SCAN_TOPIC}" "/tf_static")
    if [ -n "${ODOM_TOPIC}" ]; then
        PLAY_TOPICS+=("${ODOM_TOPIC}")
    fi

    echo "  - mode: filtered topics"
    echo "  - topics: ${PLAY_TOPICS[*]}"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}"
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

if ! convert_pbstream_to_map "✅ Provisional map generated:"; then
    exit 1
fi

if [ "${RUN_VSLAM_HINT_REMAP}" = true ] && \
   [ "${ODOM_TOPIC}" = "/visual_slam/tracking/odometry" ]; then
    run_vslam_hint_remap_pass
fi

# ==========================================
# 5. Centerline
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
# 6. Raceline
# ==========================================
prompt_raceline_generation
generate_raceline "${CENTERLINE_OUTPUT_PATH}"

# ==========================================
# 7. Line preview
# ==========================================
generate_line_preview "${CENTERLINE_INPUT_MAP}"

# ==========================================
# 8. Transfer by scp
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
