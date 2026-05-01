#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_2d_map_from_bag.sh [--scan-topic TOPIC] [--rate RATE] [--odom-topic TOPIC] [--use-vslam-odom] BAG_PATH MAP_NAME

Arguments:
  BAG_PATH   rosbag2 directory path (metadata.yaml must exist)
  MAP_NAME   output map name (saved in /map/<bag_name>/<MAP_NAME>.*)

Options:
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  use odometry topic and enable odometry in cartographer
  --use-vslam-odom    shorthand of --odom-topic /visual_slam/tracking/odometry
  -h, --help          show this help

Outputs:
  /map/<bag_name>/<MAP_NAME>.pbstream
  /map/<bag_name>/<MAP_NAME>.yaml
  /map/<bag_name>/<MAP_NAME>.pgm
  /map/<bag_name>/<MAP_NAME>.png (optional; generated if converter is available)

After map creation:
  optionally transfer /map/<bag_name>/ to remote host by rsync
EOF
}

# ==========================================
# Default settings
# ==========================================
SCAN_TOPIC="/scan"
PLAY_RATE="1.0"
ODOM_TOPIC=""
CONFIG_BASENAME="cartographer_2d.lua"

DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspace/tamiya-systems/map/"
# ==========================================

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

# bag dir name
BAG_PATH_CLEAN="${BAG_PATH%/}"
BAG_DIR_NAME="$(basename "${BAG_PATH_CLEAN}")"

# output paths
OUT_DIR="/map/${BAG_DIR_NAME}"
MAP_STEM="${OUT_DIR}/${MAP_NAME}"
PBSTREAM_PATH="${MAP_STEM}.pbstream"
MAP_LOG_PATH="/tmp/cartographer_mapping_$(date +%Y%m%d_%H%M%S).log"

# validate bag
if [ ! -d "$BAG_PATH" ] || [ ! -f "$BAG_PATH/metadata.yaml" ]; then
    echo "Invalid BAG_PATH: $BAG_PATH" >&2
    echo "metadata.yaml not found." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

stop_cartographer() {
    if [ -n "${CARTOGRAPHER_PID:-}" ] && kill -0 "${CARTOGRAPHER_PID}" 2>/dev/null; then
        kill "${CARTOGRAPHER_PID}" 2>/dev/null || true
        wait "${CARTOGRAPHER_PID}" 2>/dev/null || true
    fi
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

# ==========================================
# 1. Launch cartographer
# ==========================================
echo "[1/5] Launch cartographer (log: ${MAP_LOG_PATH})"

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

# ==========================================
# 2. Wait service
# ==========================================
echo "[2/5] Wait for /write_state service"

if ! wait_for_service "/write_state" 60; then
    echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
    exit 1
fi

# ==========================================
# 3. Play bag
# ==========================================
echo "[3/5] Play rosbag"

ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}"

# ==========================================
# 4. Save map
# ==========================================
echo "[4/5] Save trajectory and map"

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

# convert map
if ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
    -pbstream_filename "${PBSTREAM_PATH}" \
    -map_filestem "${MAP_STEM}" \
    -resolution 0.05; then

    PNG_PATH="${MAP_STEM}.png"
    PNG_CREATED=false
    if convert_pgm_to_png "${MAP_STEM}.pgm" "${PNG_PATH}"; then
        PNG_CREATED=true
    else
        echo "Warning: PNG conversion skipped. (Need one of: magick/convert/ffmpeg/python3+Pillow)" >&2
    fi

    echo ""
    echo "✅ Map generated:"
    echo "  - ${MAP_STEM}.yaml"
    echo "  - ${MAP_STEM}.pgm"
    if [ "${PNG_CREATED}" = true ]; then
        echo "  - ${MAP_STEM}.png"
    fi
    echo "  - ${PBSTREAM_PATH}"

else
    echo "pbstream generated: ${PBSTREAM_PATH}" >&2
    echo "Failed to convert pbstream to occupancy map." >&2
    exit 1
fi

# ==========================================
# 5. Transfer by rsync
# ==========================================
echo ""
read -p "2D mapを作成しました。送信しますか？ (Y/n, Enterで送信): " SEND_CONFIRM
SEND_CONFIRM=${SEND_CONFIRM:-y}

if [[ ! "$SEND_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "送信をスキップしました。"
    exit 0
fi

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
echo "送信元 : ${OUT_DIR}/"
echo "送信先 : ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"
echo "=============================================="
echo ""

read -p "この内容でrsync転送しますか？ (Y/n, Enterで実行): " FINAL_CONFIRM
FINAL_CONFIRM=${FINAL_CONFIRM:-y}

if [[ ! "$FINAL_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "転送をキャンセルしました。"
    exit 0
fi

echo "rsync転送を開始します..."

rsync -avP \
    "${OUT_DIR}/" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}${BAG_DIR_NAME}/"

echo ""
echo "✅ rsync転送が完了しました！"
