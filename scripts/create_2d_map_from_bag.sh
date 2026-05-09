#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_2d_map_from_bag.sh [OPTIONS]

Options:
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  use odometry topic and enable odometry in cartographer
  --use-vslam-odom    shorthand of --odom-topic /visual_slam/tracking/odometry
  --play-all-topics   play all topics in bag (default: play only needed topics)
  --record-root DIR   rosbag探索ルート (default: /record)
  --no-centerline     skip centerline CSV generation
  --centerline-debug  save centerline debug images (default: enabled when centerline is generated)
  --centerline-debug-dir DIR
                      set centerline debug image output directory
  --centerline-script PATH
                      path to generate_centerline.py (auto-detect by default)
  -h, --help          show this help

Outputs:
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pbstream
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pgm
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.png (optional; generated if converter is available)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline.csv (optional; generated unless --no-centerline)

After map creation:
  optionally transfer /map/<bag_name>/<MAP_NAME>/ to remote host by scp

Interactive flow:
  1) /record を再帰探索して metadata.yaml を持つ rosbag2 ディレクトリを一覧表示
  2) 番号で rosbag を選択
  3) map 名を入力
  4) Cartographer -> PNG変換
  5) centerline 生成の可否を確認（debug はデフォルト有効）
  6) scp 転送可否を確認
EOF
}

# ==========================================
# Default settings
# ==========================================
SCAN_TOPIC="/scan"
PLAY_RATE="1.0"
ODOM_TOPIC=""
CONFIG_BASENAME="cartographer_2d.lua"
PLAY_ALL_TOPICS=false
ENABLE_CENTERLINE=true
CENTERLINE_DEBUG=true
CENTERLINE_DEBUG_DIR=""
CENTERLINE_SCRIPT_PATH=""
RECORD_ROOT="/record"

DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspace/tamiya-systems/map/"
# ==========================================

CARTOGRAPHER_PID=""
CARTOGRAPHER_USES_SETSID=false
BASE_CARTOGRAPHER_PIDS=()
ROSBAG_CANDIDATES=()

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
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --no-centerline)
            ENABLE_CENTERLINE=false
            shift
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
MAP_LOG_PATH=""

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

select_rosbag_path_interactive
prompt_map_name_interactive

# bag dir name
BAG_PATH_CLEAN="${BAG_PATH%/}"
BAG_DIR_NAME="$(basename "${BAG_PATH_CLEAN}")"

# output paths
BAG_OUT_DIR="/map/${BAG_DIR_NAME}"
OUT_DIR="${BAG_OUT_DIR}/${MAP_NAME}"
MAP_STEM="${OUT_DIR}/${MAP_NAME}"
PBSTREAM_PATH="${MAP_STEM}.pbstream"
MAP_YAML_PATH="${MAP_STEM}.yaml"
MAP_PGM_PATH="${MAP_STEM}.pgm"
MAP_PNG_PATH="${MAP_STEM}.png"
CENTERLINE_OUTPUT_PATH="${MAP_STEM}_centerline.csv"
MAP_LOG_PATH="/tmp/cartographer_mapping_$(date +%Y%m%d_%H%M%S).log"

# validate bag
if [ ! -d "$BAG_PATH" ] || [ ! -f "$BAG_PATH/metadata.yaml" ]; then
    echo "Invalid BAG_PATH: $BAG_PATH" >&2
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

generate_centerline() {
    local input_map_path="$1"
    local centerline_script_path
    local -a centerline_cmd

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        echo "[5/6] Skip centerline generation"
        return 0
    fi

    echo "[5/6] Generate centerline"

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
    if [ -n "${CENTERLINE_DEBUG_PATH}" ]; then
        echo "  - ${CENTERLINE_DEBUG_PATH}/"
    fi
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
echo "[1/6] Launch cartographer (log: ${MAP_LOG_PATH})"
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
    CARTOGRAPHER_USES_SETSID=true
    setsid ros2 launch system_launch cartographer_2d_mapping.launch.xml \
        "${LAUNCH_ARGS[@]}" \
        > "${MAP_LOG_PATH}" 2>&1 &
else
    CARTOGRAPHER_USES_SETSID=false
    ros2 launch system_launch cartographer_2d_mapping.launch.xml \
        "${LAUNCH_ARGS[@]}" \
        > "${MAP_LOG_PATH}" 2>&1 &
fi

CARTOGRAPHER_PID=$!

# ==========================================
# 2. Wait service
# ==========================================
echo "[2/6] Wait for /write_state service"

if ! wait_for_service "/write_state" 60; then
    echo "Cartographer service not ready. Check log: ${MAP_LOG_PATH}" >&2
    exit 1
fi

# ==========================================
# 3. Play bag
# ==========================================
echo "[3/6] Play rosbag"

if [ "${PLAY_ALL_TOPICS}" = true ]; then
    echo "  - mode: all topics"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}"
else
    PLAY_TOPICS=("${SCAN_TOPIC}" "/tf_static")
    if [ -n "${ODOM_TOPIC}" ]; then
        PLAY_TOPICS+=("${ODOM_TOPIC}")
    fi

    echo "  - mode: filtered topics"
    echo "  - topics: ${PLAY_TOPICS[*]}"
    ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}" --topics "${PLAY_TOPICS[@]}"
fi

# ==========================================
# 4. Save map
# ==========================================
echo "[4/6] Save trajectory and map"

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

    PNG_FILENAME="$(basename "${MAP_PNG_PATH}")"
    PNG_CREATED=false
    YAML_IMAGE_UPDATED=false
    if convert_pgm_to_png "${MAP_PGM_PATH}" "${MAP_PNG_PATH}"; then
        PNG_CREATED=true
        if update_yaml_image_path "${MAP_YAML_PATH}" "${PNG_FILENAME}"; then
            YAML_IMAGE_UPDATED=true
        else
            echo "Warning: PNG was generated, but failed to update image path in ${MAP_YAML_PATH}." >&2
        fi
    else
        echo "Warning: PNG conversion skipped. (Need one of: magick/convert/ffmpeg/python3+Pillow)" >&2
    fi

    echo ""
    echo "✅ Map generated:"
    echo "  - ${MAP_YAML_PATH}"
    if [ "${YAML_IMAGE_UPDATED}" = true ]; then
        echo "    image: ${PNG_FILENAME}"
    fi
    echo "  - ${MAP_PGM_PATH}"
    if [ "${PNG_CREATED}" = true ]; then
        echo "  - ${MAP_PNG_PATH}"
    fi
    echo "  - ${PBSTREAM_PATH}"

else
    echo "pbstream generated: ${PBSTREAM_PATH}" >&2
    echo "Failed to convert pbstream to occupancy map." >&2
    exit 1
fi

# ==========================================
# 5. Centerline
# ==========================================
prompt_centerline_generation
CENTERLINE_INPUT_MAP="${MAP_PGM_PATH}"
if [ -f "${MAP_PNG_PATH}" ]; then
    CENTERLINE_INPUT_MAP="${MAP_PNG_PATH}"
fi
generate_centerline "${CENTERLINE_INPUT_MAP}"

# ==========================================
# 6. Transfer by scp
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
