#!/bin/bash

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

