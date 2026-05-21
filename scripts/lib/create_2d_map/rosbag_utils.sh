#!/bin/bash

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

wait_for_topic() {
    local topic_name="$1"
    local timeout_sec="$2"
    local count=0
    local topic_list_output=""

    while [ "${count}" -lt "${timeout_sec}" ]; do
        topic_list_output="$(ros2 topic list 2>/dev/null || true)"
        if printf '%s\n' "${topic_list_output}" | grep -Fxq -- "${topic_name}"; then
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

