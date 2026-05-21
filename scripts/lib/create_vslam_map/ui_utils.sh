#!/bin/bash

usage() {
    cat <<'EOF'
Usage:
  create_vslam_map_from_bag.sh [OPTIONS]

Options:
  --mode NAME         default|vslam|vslam_map (default: default)
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

