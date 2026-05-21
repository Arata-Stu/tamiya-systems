#!/bin/bash

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
    if [ -n "${CREATE_VSLAM_MAP_SETUP:-}" ]; then
        source_setup_script "${CREATE_VSLAM_MAP_SETUP}"
    elif [ -f "/workspaces/install/setup.bash" ]; then
        source_setup_script "/workspaces/install/setup.bash"
    elif [ -f "install/setup.bash" ]; then
        source_setup_script "install/setup.bash"
    fi
}

apply_mode() {
    case "$1" in
        default)
            ;;
        vslam|vslam_map)
            # vslam_map プリセット: launch_system.sh の sensor_data_recording と
            # 同じ解像度/FPS で揃えること。上記 IMAGE_WIDTH/IMAGE_HEIGHT に合わせて変更する。
            IMAGE_WIDTH="424"
            IMAGE_HEIGHT="240"
            IMAGE_FPS="30.0"
            ;;
        *)
            echo "Unknown mode: $1" >&2
            usage
            exit 1
            ;;
    esac
}

cleanup_all() {
    stop_background_process "RECORDER_PID" "RECORDER_USES_SETSID"
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

