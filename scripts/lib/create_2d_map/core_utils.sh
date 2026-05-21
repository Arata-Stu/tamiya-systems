#!/bin/bash

resolve_real_path() {
    local input_path="$1"

    if command -v python3 >/dev/null 2>&1; then
        python3 - "$input_path" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
        return 0
    fi

    if command -v perl >/dev/null 2>&1; then
        perl -MCwd=realpath -e 'print realpath($ARGV[0])' "$input_path"
        return 0
    fi

    printf '%s\n' "$input_path"
}

is_repo_root() {
    local candidate="$1"

    [ -d "${candidate}" ] || return 1
    [ -d "${candidate}/scripts" ] || return 1
    [ -d "${candidate}/ros2_ws" ] || return 1
    [ -f "${candidate}/scripts/create_2d_map_from_bag.sh" ] || return 1
}

find_repo_root_from() {
    local current="$1"

    [ -n "${current}" ] || return 1

    if [ -f "${current}" ]; then
        current="$(dirname "${current}")"
    fi

    if [ ! -d "${current}" ]; then
        return 1
    fi

    while true; do
        if is_repo_root "${current}"; then
            (cd "${current}" && pwd)
            return 0
        fi

        if [ "${current}" = "/" ]; then
            break
        fi

        current="$(dirname "${current}")"
    done

    return 1
}

resolve_repo_root() {
    local candidate

    for candidate in \
        "${CREATE_2D_MAP_PROJECT_ROOT:-}" \
        "${INITIAL_PWD}" \
        "${SCRIPT_DIR}" \
        "${SCRIPT_DIR}/.." \
        "/workspaces" \
        "/workspace"; do
        [ -n "${candidate}" ] || continue
        if find_repo_root_from "${candidate}"; then
            return 0
        fi
    done

    (cd "${SCRIPT_DIR}/.." && pwd)
}

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
    local resolved_path

    if resolved_path="$(resolve_system_launch_share_file "config" "${relative_path}")"; then
        echo "${resolved_path}"
        return 0
    fi

    return 1
}

resolve_system_launch_share_file() {
    local share_subdir="$1"
    local relative_path="$2"
    local pkg_prefix=""

    if [ -n "${SYSTEM_LAUNCH_SOURCE_SHARE}" ] && \
       [ -f "${SYSTEM_LAUNCH_SOURCE_SHARE}/${share_subdir}/${relative_path}" ]; then
        echo "${SYSTEM_LAUNCH_SOURCE_SHARE}/${share_subdir}/${relative_path}"
        return 0
    fi

    if command -v ros2 >/dev/null 2>&1; then
        pkg_prefix="$(ros2 pkg prefix system_launch 2>/dev/null || true)"
        if [ -n "${pkg_prefix}" ] && \
           [ -f "${pkg_prefix}/share/system_launch/${share_subdir}/${relative_path}" ]; then
            echo "${pkg_prefix}/share/system_launch/${share_subdir}/${relative_path}"
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
        "${REPO_ROOT}" \
        "${SYSTEM_LAUNCH_SOURCE_SHARE}" \
        "/workspaces" \
        "/workspace" \
        "${PWD}"; do
        [ -n "${candidate_root}" ] || continue
        if [ -f "${candidate_root}/${relative_path}" ]; then
            echo "${candidate_root}/${relative_path}"
            return 0
        fi
        
        # If relative_path starts with ros2_ws/ but we are in docker where src is mounted directly
        if [[ "${relative_path}" == ros2_ws/* ]]; then
            local stripped_path="${relative_path#ros2_ws/}"
            if [ -f "${candidate_root}/${stripped_path}" ]; then
                echo "${candidate_root}/${stripped_path}"
                return 0
            fi
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

build_system_launch_cmd() {
    local launch_file="$1"

    SYSTEM_LAUNCH_CMD=(ros2 launch)
    if [ -f "${SYSTEM_LAUNCH_SOURCE_SHARE}/launch/${launch_file}" ]; then
        SYSTEM_LAUNCH_CMD+=("${SYSTEM_LAUNCH_SOURCE_SHARE}/launch/${launch_file}")
    else
        SYSTEM_LAUNCH_CMD+=(system_launch "${launch_file}")
    fi
}

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

resolve_effective_mode() {
    local odom_label="no_odom"
    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        odom_label="with_odom"
    fi

    echo "${odom_label}_${PIPELINE_MODE}_vslam"
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

stop_post_alignment_stack() {
    stop_background_process "POST_ALIGNMENT_STACK_PID" "POST_ALIGNMENT_STACK_USES_SETSID"
    stop_background_process "POST_ALIGNMENT_REFERENCE_PUBLISHER_PID" "POST_ALIGNMENT_REFERENCE_PUBLISHER_USES_SETSID"
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

cleanup_all() {
    stop_rosbag_playback
    stop_background_process "LANDMARK_EXPORT_PID" "LANDMARK_EXPORT_USES_SETSID"
    stop_background_process "RVIZ_PID" "RVIZ_USES_SETSID"
    stop_vslam_reference_recorder
    stop_post_alignment_stack
    stop_recorder
    stop_vslam
    stop_cartographer
    if [ -n "${VSLAM_LOCALIZATION_PARAM_PATH:-}" ] && [ -f "${VSLAM_LOCALIZATION_PARAM_PATH}" ]; then
        rm -f "${VSLAM_LOCALIZATION_PARAM_PATH}"
        VSLAM_LOCALIZATION_PARAM_PATH=""
    fi
}

