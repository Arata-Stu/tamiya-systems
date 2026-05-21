#!/bin/bash

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

cleanup_new_cartographer_processes() {
    local pid
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if ! is_base_cartographer_pid "$pid"; then
            kill_pid_gracefully "$pid" false
        fi
    done < <(list_cartographer_pids)
}

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

