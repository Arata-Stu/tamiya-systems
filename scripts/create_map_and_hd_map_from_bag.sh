#!/bin/bash

# One-shot local map build flow:
#   sensor bag -> online cuVSLAM + Cartographer 2D map -> HD map editor/raceline

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
INITIAL_PWD="${PWD}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/core_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/rosbag_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/create_2d_map/map_processing.sh"

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_map_and_hd_map_from_bag.sh [OPTIONS]

Default flow:
  1. Pick one rosbag and one map name.
  2. Replay it once while Cartographer and online cuVSLAM run together.
  3. Save the 2D map PNG for GL use and the cuVSLAM map/reference snapshot.
  4. Export the VSLAM landmark raster, open the HD lane editor, then generate raceline.
  5. Optionally continue to the HD section-gate editor or scp transfer.

Options:
  --bag-path DIR          input rosbag2 directory (skip interactive selection)
  --map-name NAME         output map name (skip interactive prompt)
  --record-root DIR       rosbag search root for interactive selection (default: /record)
  --mode NAME             online 2D mode: with_odom_online_vslam or no_odom_online_vslam
                          (default: with_odom_online_vslam)
  --rate RATE             ros2 bag play rate for 2D/VSLAM mapping (default: 1.0)
  --image-width PX        online VSLAM image width (default: 424)
  --image-height PX       online VSLAM image height (default: 240)
  --image-fps FPS         online VSLAM image fps (default: 90.0)
  --with-imu              replay /camera/imu during mapping (default)
  --no-imu                do not replay /camera/imu
  --play-all-topics       replay every source-bag topic instead of filtered topics
  --use-image-preprocessors run rectify/mono preprocessing before VSLAM
  --no-image-preprocessors make VSLAM subscribe to recorded camera topics directly (default)
  --launch-offline-tf     publish fallback base_link TFs instead of using only bag TFs
  --skip-2d-map           reuse existing <map_dir>/<map_name>.yaml and VSLAM snapshot
  --allow-missing-2d-png  continue even if <map_name>.png was not generated
  --editor-scale SCALE    initial HD editor zoom; 0 fits the whole raster (default: 0)
  --no-editor             only create 2D/VSLAM/HD raster outputs; do not open HD editor
  --no-raceline           skip raceline generation after the HD editor exits
  --no-line-preview       skip centerline/raceline overlay PNG generation
  --open-section-editor   open the HD section-gate editor after raceline generation
  --section-editor-scale SCALE
                          initial HD section-gate editor zoom (default: 0)
  --scp-after             start scp transfer after raceline/section editing
  --no-post-menu          skip the final section-gate/scp menu
  -h, --help              show this help

Outputs:
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>.png
  /map/<source_bag>/<MAP_NAME>/cuvslam_map/
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_reference.json
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.png
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_hd_map.yaml
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_centerline.csv
  /map/<source_bag>/<MAP_NAME>/<MAP_NAME>_raceline.csv
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

ensure_online_2d_mode() {
    case "${MODE}" in
        2d_slam)
            MODE="no_odom_online_vslam"
            ;;
        with_odom_online_vslam|no_odom_online_vslam)
            ;;
        *)
            die "this integrated flow runs 2D SLAM and VSLAM together online. Use --mode with_odom_online_vslam or no_odom_online_vslam."
            ;;
    esac
}

run_hd_section_gate_editor() {
    local hd_map_yaml_path="${MAP_STEM}_hd_map.yaml"
    local -a section_cmd

    if [ ! -f "${hd_map_yaml_path}" ]; then
        echo "Warning: HD map YAML not found for section-gate editor: ${hd_map_yaml_path}" >&2
        return 0
    fi
    if [ ! -f "${EDIT_HD_MAP_SECTIONS_SH}" ]; then
        echo "Warning: HD section-gate editor launcher not found: ${EDIT_HD_MAP_SECTIONS_SH}" >&2
        return 0
    fi

    section_cmd=(
        bash "${EDIT_HD_MAP_SECTIONS_SH}"
        --hd-map-yaml "${hd_map_yaml_path}"
        --scale "${SECTION_EDITOR_SCALE}"
    )

    echo ""
    echo "[post] Open HD section-gate editor"
    printf '  %q' "${section_cmd[@]}"
    echo ""
    "${section_cmd[@]}" || echo "Warning: HD section-gate editor failed." >&2
}

run_scp_transfer() {
    local remote_user
    local ip_choice
    local remote_ip
    local remote_dir
    local final_confirm
    local i

    if [ ! -t 0 ]; then
        echo "Warning: no interactive TTY is available. Skip scp transfer." >&2
        return 0
    fi

    echo ""
    read -r -p "相手のユーザー名 (Enterで '${DEFAULT_REMOTE_USER}'): " remote_user
    remote_user=${remote_user:-$DEFAULT_REMOTE_USER}

    echo ""
    echo "相手のIPアドレスを選択、または直接入力してください:"
    i=1
    for ip in "${DEFAULT_REMOTE_IPS[@]}"; do
        if [ "${i}" -eq 1 ]; then
            echo "  ${i}) ${ip} (Enterのデフォルト)"
        else
            echo "  ${i}) ${ip}"
        fi
        ((i++))
    done
    echo ""

    read -r -p "番号、またはIPを直接入力 (Enterで '${DEFAULT_REMOTE_IPS[0]}'): " ip_choice

    if [ -z "${ip_choice}" ]; then
        remote_ip="${DEFAULT_REMOTE_IPS[0]}"
    elif [[ "${ip_choice}" =~ ^[0-9]+$ ]] && \
         [ "${ip_choice}" -ge 1 ] && \
         [ "${ip_choice}" -le "${#DEFAULT_REMOTE_IPS[@]}" ]; then
        remote_ip="${DEFAULT_REMOTE_IPS[$((ip_choice-1))]}"
    else
        remote_ip="${ip_choice}"
    fi

    echo ""
    read -r -p "送信先ディレクトリ (Enterで '${DEFAULT_REMOTE_DIR}'): " remote_dir
    remote_dir=${remote_dir:-$DEFAULT_REMOTE_DIR}

    echo ""
    echo "================ 転送内容確認 ================"
    echo "送信元 : ${MAP_DIR}"
    echo "送信先 : ${remote_user}@${remote_ip}:${remote_dir%/}/${BAG_DIR_NAME}/"
    echo "=============================================="
    echo ""

    read -r -p "この内容でscp転送しますか？ (Y/n, Enterで実行): " final_confirm
    final_confirm=${final_confirm:-y}

    if [[ ! "${final_confirm}" =~ ^[Yy]$ ]]; then
        echo "転送をキャンセルしました。"
        return 0
    fi

    echo "scp転送を開始します..."
    ssh "${remote_user}@${remote_ip}" "mkdir -p '${remote_dir%/}/${BAG_DIR_NAME}'"
    scp -r "${MAP_DIR}" "${remote_user}@${remote_ip}:${remote_dir%/}/${BAG_DIR_NAME}/"

    echo ""
    echo "✅ scp転送が完了しました！"
}

run_post_menu() {
    local action_choice

    if [ "${POST_MENU_ENABLED}" != true ] || [ ! -t 0 ]; then
        return 0
    fi

    while true; do
        echo ""
        echo "次の操作を選んでください:"
        echo "  1) HD section-gate editor を開く"
        echo "  2) scp 転送へ進む"
        echo "  3) 終了"
        read -r -p "選択 [3]: " action_choice

        case "${action_choice:-3}" in
            1|section|sections|gate|edit|s)
                run_hd_section_gate_editor
                ;;
            2|transfer|scp|t)
                run_scp_transfer
                return 0
                ;;
            3|skip|exit|quit|q)
                return 0
                ;;
            *)
                echo "無効な選択です: ${action_choice}" >&2
                ;;
        esac
    done
}

CREATE_2D_MAP_SH="${SCRIPT_DIR}/create_2d_map_from_bag.sh"
CREATE_HD_MAP_SH="${SCRIPT_DIR}/create_hd_map_from_vslam_bag.sh"
EDIT_HD_MAP_SECTIONS_SH="${SCRIPT_DIR}/edit_hd_map_sections.sh"

BAG_PATH=""
MAP_NAME=""
RECORD_ROOT="/record"
MODE="with_odom_online_vslam"
PLAY_RATE="1.0"
IMAGE_WIDTH="424"
IMAGE_HEIGHT="240"
IMAGE_FPS="90.0"
USE_IMU=true
PLAY_ALL_TOPICS=false
USE_IMAGE_PREPROCESSORS=false
LAUNCH_OFFLINE_TF=false
SKIP_2D_MAP=false
REQUIRE_2D_PNG=true
OPEN_EDITOR=true
GENERATE_RACELINE=true
GENERATE_LINE_PREVIEW=true
EDITOR_SCALE="0"
SECTION_EDITOR_SCALE="0"
POST_MENU_ENABLED=true
OPEN_SECTION_EDITOR_AFTER=false
SCP_AFTER=false
ROSBAG_CANDIDATES=()
SCAN_TOPIC="/scan"
SOURCE_PLAY_TOPICS=()

DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspaces/tamiya-systems/map/"

while (($#)); do
    case "$1" in
        --bag-path)
            BAG_PATH="$2"
            shift 2
            ;;
        --map-name)
            MAP_NAME="$2"
            shift 2
            ;;
        --record-root)
            RECORD_ROOT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
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
        --image-fps)
            IMAGE_FPS="$2"
            shift 2
            ;;
        --with-imu)
            USE_IMU=true
            shift
            ;;
        --no-imu)
            USE_IMU=false
            shift
            ;;
        --play-all-topics)
            PLAY_ALL_TOPICS=true
            shift
            ;;
        --use-image-preprocessors)
            USE_IMAGE_PREPROCESSORS=true
            shift
            ;;
        --no-image-preprocessors)
            USE_IMAGE_PREPROCESSORS=false
            shift
            ;;
        --launch-offline-tf)
            LAUNCH_OFFLINE_TF=true
            shift
            ;;
        --skip-2d-map)
            SKIP_2D_MAP=true
            shift
            ;;
        --allow-missing-2d-png)
            REQUIRE_2D_PNG=false
            shift
            ;;
        --editor-scale)
            EDITOR_SCALE="$2"
            shift 2
            ;;
        --no-editor)
            OPEN_EDITOR=false
            shift
            ;;
        --no-raceline)
            GENERATE_RACELINE=false
            shift
            ;;
        --no-line-preview)
            GENERATE_LINE_PREVIEW=false
            shift
            ;;
        --open-section-editor)
            OPEN_SECTION_EDITOR_AFTER=true
            POST_MENU_ENABLED=false
            shift
            ;;
        --section-editor-scale)
            SECTION_EDITOR_SCALE="$2"
            shift 2
            ;;
        --scp-after)
            SCP_AFTER=true
            POST_MENU_ENABLED=false
            shift
            ;;
        --no-post-menu)
            POST_MENU_ENABLED=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "Unknown option: $1"
            ;;
        *)
            die "Positional arguments are not supported: $1"
            ;;
    esac
done

if [ -z "${BAG_PATH}" ]; then
    select_rosbag_path_interactive
fi
BAG_PATH="${BAG_PATH%/}"
if [ ! -d "${BAG_PATH}" ] || [ ! -f "${BAG_PATH}/metadata.yaml" ]; then
    die "Invalid rosbag2 directory: ${BAG_PATH}"
fi

if [ -z "${MAP_NAME}" ]; then
    prompt_map_name_interactive
fi
if [[ "${MAP_NAME}" == *"/"* ]]; then
    die "map name must not contain '/'"
fi
if [ "${SKIP_2D_MAP}" != true ]; then
    ensure_online_2d_mode
fi

BAG_DIR_NAME="$(basename "${BAG_PATH}")"
MAP_DIR="/map/${BAG_DIR_NAME}/${MAP_NAME}"
MAP_STEM="${MAP_DIR}/${MAP_NAME}"
MAP_YAML_PATH="${MAP_STEM}.yaml"
MAP_PNG_PATH="${MAP_STEM}.png"
VSLAM_MAP_DIR="${MAP_DIR}/cuvslam_map"
SNAPSHOT_PATH="${MAP_STEM}_vslam_reference.json"

echo ""
echo "================ One-shot map build ================"
echo "source bag : ${BAG_PATH}"
echo "map name   : ${MAP_NAME}"
echo "map dir    : ${MAP_DIR}"
echo "2D mode    : ${MODE}"
echo "VSLAM map  : ${VSLAM_MAP_DIR}"
echo "IMU replay : ${USE_IMU}"
echo "image prep : ${USE_IMAGE_PREPROCESSORS}"
echo "===================================================="

if [ "${SKIP_2D_MAP}" != true ]; then
    map_cmd=(
        bash "${CREATE_2D_MAP_SH}"
        --mode "${MODE}"
        --bag-path "${BAG_PATH}"
        --map-name "${MAP_NAME}"
        --rate "${PLAY_RATE}"
        --image-width "${IMAGE_WIDTH}"
        --image-height "${IMAGE_HEIGHT}"
        --image-fps "${IMAGE_FPS}"
        --record-root "${RECORD_ROOT}"
        --save-vslam-reference
        --vslam-vis
        --no-live-vslam-map-align
        --no-centerline
        --no-raceline
        --no-line-preview
        --no-scp
    )
    if [ "${USE_IMU}" = true ]; then
        map_cmd+=(--with-imu)
    else
        map_cmd+=(--no-imu)
    fi
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        map_cmd+=(--play-all-topics)
    fi
    if [ "${USE_IMAGE_PREPROCESSORS}" = true ]; then
        map_cmd+=(--use-image-preprocessors)
    else
        map_cmd+=(--no-image-preprocessors)
    fi
    if [ "${LAUNCH_OFFLINE_TF}" = true ]; then
        map_cmd+=(--launch-offline-tf)
    fi

    echo ""
    echo "[1/2] Build 2D map, cuVSLAM map, and VSLAM reference"
    printf '  %q' "${map_cmd[@]}"
    echo ""
    "${map_cmd[@]}"
else
    echo ""
    echo "[1/2] Reuse existing 2D map and VSLAM reference"
fi

if [ ! -f "${MAP_YAML_PATH}" ]; then
    die "2D map YAML was not created: ${MAP_YAML_PATH}"
fi
if [ "${REQUIRE_2D_PNG}" = true ] && [ ! -f "${MAP_PNG_PATH}" ]; then
    die "2D map PNG was not created: ${MAP_PNG_PATH}. Install ImageMagick/ffmpeg/Pillow or rerun with --allow-missing-2d-png."
fi
if [ ! -d "${VSLAM_MAP_DIR}" ] || [ -z "$(find "${VSLAM_MAP_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    die "cuVSLAM map was not created or is empty: ${VSLAM_MAP_DIR}"
fi
if [ ! -f "${SNAPSHOT_PATH}" ]; then
    die "VSLAM reference snapshot was not created: ${SNAPSHOT_PATH}"
fi

hd_cmd=(
    bash "${CREATE_HD_MAP_SH}"
    --skip-vslam
    --bag-path "${BAG_PATH}"
    --map-name "${MAP_NAME}"
    --map-dir "${MAP_DIR}"
    --snapshot "${SNAPSHOT_PATH}"
    --reference-yaml "${MAP_YAML_PATH}"
    --editor-scale "${EDITOR_SCALE}"
)
if [ "${OPEN_EDITOR}" != true ]; then
    hd_cmd+=(--no-editor)
fi
if [ "${GENERATE_RACELINE}" != true ]; then
    hd_cmd+=(--no-raceline)
fi
if [ "${GENERATE_LINE_PREVIEW}" != true ]; then
    hd_cmd+=(--no-line-preview)
fi

echo ""
echo "[2/2] Open HD map editor from VSLAM reference"
printf '  %q' "${hd_cmd[@]}"
echo ""
"${hd_cmd[@]}"

echo ""
echo "✅ Map bundle ready:"
echo "  - map dir    : ${MAP_DIR}"
echo "  - 2D map     : ${MAP_YAML_PATH}"
if [ -f "${MAP_PNG_PATH}" ]; then
    echo "  - 2D PNG     : ${MAP_PNG_PATH}"
fi
echo "  - VSLAM map  : ${VSLAM_MAP_DIR}"
echo "  - snapshot   : ${SNAPSHOT_PATH}"
echo "  - HD map     : ${MAP_STEM}_hd_map.yaml"
echo "  - raceline   : ${MAP_STEM}_raceline.csv"

if [ "${OPEN_SECTION_EDITOR_AFTER}" = true ]; then
    run_hd_section_gate_editor
fi
if [ "${SCP_AFTER}" = true ]; then
    run_scp_transfer
fi
run_post_menu
