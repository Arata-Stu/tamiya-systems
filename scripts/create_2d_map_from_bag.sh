#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_2d_map_from_bag.sh [--scan-topic TOPIC] [--rate RATE] [--odom-topic TOPIC] [--use-vslam-odom] BAG_PATH MAP_NAME

Arguments:
  BAG_PATH   rosbag2 directory path (metadata.yaml must exist)
  MAP_NAME   output map name (will be saved in /workspaces/map/<bag_name>/<MAP_NAME>)
             e.g. corridor

Options:
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  use odometry topic and enable odometry in cartographer
  --use-vslam-odom    shorthand of --odom-topic /visual_slam/tracking/odometry
  -h, --help          show this help

Outputs:
  /workspaces/map/<bag_name>/<MAP_NAME>.pbstream
  /workspaces/map/<bag_name>/<MAP_NAME>.yaml
  /workspaces/map/<bag_name>/<MAP_NAME>.pgm
EOF
}

SCAN_TOPIC="/scan"
PLAY_RATE="1.0"
ODOM_TOPIC=""
CONFIG_BASENAME="cartographer_2d.lua"
POSITIONAL=()

while (($#)); do
    case "$1" in
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
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -ne 2 ]; then
    usage
    exit 1
fi

BAG_PATH="${POSITIONAL[0]}"
MAP_NAME="${POSITIONAL[1]}"

# 末尾のスラッシュを削除してからベース名（フォルダ名）を取得
BAG_PATH_CLEAN="${BAG_PATH%/}"
BAG_DIR_NAME="$(basename "${BAG_PATH_CLEAN}")"

# 出力先ディレクトリとファイルパスの生成
OUT_DIR="/workspaces/map/${BAG_DIR_NAME}"
MAP_STEM="${OUT_DIR}/${MAP_NAME}"
PBSTREAM_PATH="${MAP_STEM}.pbstream"
MAP_LOG_PATH="/tmp/cartographer_mapping_$(date +%Y%m%d_%H%M%S).log"

if [ ! -d "$BAG_PATH" ] || [ ! -f "$BAG_PATH/metadata.yaml" ]; then
    echo "Invalid BAG_PATH: $BAG_PATH" >&2
    echo "metadata.yaml not found." >&2
    exit 1
fi

# 保存先ディレクトリを作成
mkdir -p "${OUT_DIR}"

stop_cartographer() {
    if [ -n "${CARTOGRAPHER_PID:-}" ] && kill -0 "${CARTOGRAPHER_PID}" 2>/dev/null; then
        kill "${CARTOGRAPHER_PID}" 2>/dev/null || true
        wait "${CARTOGRAPHER_PID}" 2>/dev/null || true
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

trap stop_cartographer EXIT INT TERM

echo "[1/4] Launch cartographer (log: ${MAP_LOG_PATH})"
LAUNCH_ARGS=(
    "use_sim_time:=true"
    "scan_topic:=${SCAN_TOPIC}"
    "configuration_basename:=${CONFIG_BASENAME}"
)
if [ -n "${ODOM_TOPIC}" ]; then
    LAUNCH_ARGS+=("odom_topic:=${ODOM_TOPIC}")
fi

ros2 launch system_launch cartographer_2d_mapping.launch.xml \
    "${LAUNCH_ARGS[@]}" \
    > "${MAP_LOG_PATH}" 2>&1 &
CARTOGRAPHER_PID=$!

echo "[2/4] Wait for /write_state service"
if ! wait_for_service "/write_state" 60; then
    echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
    exit 1
fi

echo "[3/4] Play rosbag"
ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}"

echo "[4/4] Save trajectory and map"
if ! ros2 service call /finish_trajectory \
    cartographer_ros_msgs/srv/FinishTrajectory \
    "{trajectory_id: 0}" > /dev/null; then
    echo "Warning: /finish_trajectory failed. Continue to save state." >&2
fi

WRITE_STATE_REQUEST=$(printf "{filename: '%s', include_unfinished_submaps: true}" "${PBSTREAM_PATH}")
ros2 service call /write_state \
    cartographer_ros_msgs/srv/WriteState \
    "${WRITE_STATE_REQUEST}" > /dev/null

stop_cartographer

if ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
    -pbstream_filename "${PBSTREAM_PATH}" \
    -map_filestem "${MAP_STEM}" \
    -resolution 0.05; then
    echo "Map generated:"
    echo "  - ${MAP_STEM}.yaml"
    echo "  - ${MAP_STEM}.pgm"
    echo "  - ${PBSTREAM_PATH}"
else
    echo "pbstream generated: ${PBSTREAM_PATH}" >&2
    echo "Failed to convert pbstream to occupancy map. Check cartographer_ros installation." >&2
fi