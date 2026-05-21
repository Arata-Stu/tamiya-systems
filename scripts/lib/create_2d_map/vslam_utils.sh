#!/bin/bash

resolve_vslam_param_file() {
    resolve_system_launch_config_file "localization/vslam.param.yaml"
}

create_vslam_localization_param() {
    local base_param
    local temp_param

    base_param="$(resolve_vslam_param_file || true)"
    if [ -z "${base_param}" ] || [ ! -f "${base_param}" ]; then
        echo "Failed to resolve vslam.param.yaml for offline localization." >&2
        return 1
    fi

    temp_param="$(mktemp /tmp/create_2d_map_vslam_param_XXXXXX.yaml)"
    sed 's/^\([[:space:]]*localize_on_startup:\).*/\1 true/' "${base_param}" > "${temp_param}"

    if ! grep -Eq '^[[:space:]]*localize_on_startup:[[:space:]]*true[[:space:]]*$' "${temp_param}"; then
        echo "Failed to enable localize_on_startup in ${temp_param}." >&2
        return 1
    fi

    VSLAM_LOCALIZATION_PARAM_PATH="${temp_param}"
}

launch_vslam_stack() {
    local tf_log_path="$1"
    local vslam_log_path="$2"
    local save_map_path="${3:-}"
    local load_map_path="${4:-}"
    local vslam_param_path="${5:-}"
    local enable_alignment_from_config="${6:-true}"
    local -a launch_args=(
        "use_sim_time:=true"
        "image_width:=${IMAGE_WIDTH}"
        "image_height:=${IMAGE_HEIGHT}"
        "camera_container_name:=${CAMERA_CONTAINER_NAME}"
        "enable_localization_and_mapping:=true"
        "enable_slam_visualization:=${VSLAM_VIS_ENABLED}"
        "enable_observations_view:=${VSLAM_VIS_ENABLED}"
        "enable_landmarks_view:=${VSLAM_VIS_ENABLED}"
    )

    if [ "${enable_alignment_from_config}" = true ] && [ -n "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ] && [ -f "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ]; then
        launch_args+=(
            "use_vslam_map_alignment_node:=true"
            "vslam_map_alignment_config:=${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}"
            "vslam_map_alignment_enable_keyboard:=false"
        )
    fi

    if [ -n "${save_map_path}" ]; then
        launch_args+=("save_map_path:=${save_map_path}")
    fi

    if [ -n "${load_map_path}" ]; then
        launch_args+=("load_map_path:=${load_map_path}")
    fi

    if [ -n "${vslam_param_path}" ]; then
        launch_args+=("vslam_param:=${vslam_param_path}")
    fi

    build_system_launch_cmd "offline_sensor_tf.launch.xml"
    launch_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        > "${tf_log_path}" 2>&1

    sleep 2

    launch_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID" \
        ros2 run rclcpp_components component_container_mt --ros-args -r "__node:=${CAMERA_CONTAINER_NAME}"

    sleep 2

    build_system_launch_cmd "vslam.launch.xml"
    launch_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        "${launch_args[@]}" \
        > "${vslam_log_path}" 2>&1
}

ensure_vslam_visualization_requirements() {
    if [ "${VSLAM_LIVE_ALIGNMENT_ENABLED}" = true ] && [ "${VSLAM_VIS_ENABLED}" != true ]; then
        VSLAM_VIS_ENABLED=true
        echo "[align] Enable VSLAM visualization automatically for live alignment."
        echo "[align] landmarks / slam_path を見るため enable_slam_visualization=true, enable_landmarks_view=true で起動します。"
    fi
}

start_vslam_reference_capture() {
    local recorder_script_path
    local -a recorder_cmd

    VSLAM_REFERENCE_CAPTURE_EXPECTED=false
    VSLAM_REFERENCE_CAPTURE_STARTED=false
    VSLAM_REFERENCE_CAPTURE_WAS_STARTED=false

    if [ "${PIPELINE_MODE}" != "online" ] || [ "${PREPARE_VSLAM_MAP_ALIGNMENT}" != true ]; then
        return 0
    fi

    VSLAM_REFERENCE_CAPTURE_EXPECTED=true

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip VSLAM reference snapshot capture." >&2
        return 0
    fi

    if ! recorder_script_path="$(resolve_vslam_reference_recorder_script)"; then
        echo "Warning: VSLAM reference recorder script not found. Skip path snapshot capture." >&2
        return 0
    fi

    recorder_cmd=(
        python3 "${recorder_script_path}"
        --path-topic "${VSLAM_LANDMARK_PATH_TOPIC}"
        --odom-topic "${DEFAULT_VSLAM_ODOM_TOPIC}"
        --output "${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    )

    mkdir -p "$(dirname "${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}")"
    {
        echo "[info] Launch saved VSLAM reference recorder"
        echo "[info] path topic : ${VSLAM_LANDMARK_PATH_TOPIC}"
        echo "[info] odom topic : ${DEFAULT_VSLAM_ODOM_TOPIC}"
        echo "[info] output     : ${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    } > "${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}"

    launch_background_process "VSLAM_REFERENCE_RECORDER_PID" "VSLAM_REFERENCE_RECORDER_USES_SETSID" \
        "${recorder_cmd[@]}" \
        >> "${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}" 2>&1

    if [ -n "${VSLAM_REFERENCE_RECORDER_PID}" ] && kill -0 "${VSLAM_REFERENCE_RECORDER_PID}" 2>/dev/null; then
        VSLAM_REFERENCE_CAPTURE_STARTED=true
        VSLAM_REFERENCE_CAPTURE_WAS_STARTED=true
        echo "[ref] Started VSLAM reference snapshot recorder"
        echo "  - pid  : ${VSLAM_REFERENCE_RECORDER_PID}"
        echo "  - log  : ${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}"
        echo "  - json : ${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    else
        echo "Warning: failed to start VSLAM reference snapshot recorder." >&2
        echo "Warning: expected log path: ${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}" >&2
    fi
}

prepare_offline_vslam_odom_bag() {
    echo "[prep 1/2] Build offline VSLAM map (logs: ${OFFLINE_VSLAM_MAP_TF_LOG_PATH}, ${OFFLINE_VSLAM_MAP_LOG_PATH})"
    mkdir -p "${VSLAM_MAP_DIR}"

    launch_vslam_stack \
        "${OFFLINE_VSLAM_MAP_TF_LOG_PATH}" \
        "${OFFLINE_VSLAM_MAP_LOG_PATH}" \
        "${VSLAM_MAP_DIR}"

    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready for offline map build. Check log: ${OFFLINE_VSLAM_MAP_LOG_PATH}" >&2
        exit 1
    fi

    echo "  - replay source bag to create VSLAM map"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH}"; then
            exit 1
        fi
    else
        build_online_source_play_topics
        echo "  - mode: filtered topics"
        echo "  - topics: ${SOURCE_PLAY_TOPICS[*]}"
        if ! play_rosbag \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_MAP_PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"; then
            exit 1
        fi
    fi

    sleep 2
    if ! ros2 service call /visual_slam/save_map \
        isaac_ros_visual_slam_interfaces/srv/FilePath \
        "$(printf "{file_path: '%s'}" "${VSLAM_MAP_DIR}")" > /dev/null; then
        echo "Failed to save offline VSLAM map. Check log: ${OFFLINE_VSLAM_MAP_LOG_PATH}" >&2
        exit 1
    fi

    stop_vslam

    echo "[prep 2/2] Create offline odom bag from saved VSLAM map"
    create_vslam_localization_param

    launch_vslam_stack \
        "${OFFLINE_VSLAM_ODOM_TF_LOG_PATH}" \
        "${OFFLINE_VSLAM_ODOM_LOG_PATH}" \
        "" \
        "${VSLAM_MAP_DIR}" \
        "${VSLAM_LOCALIZATION_PARAM_PATH}"

    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Visual SLAM service not ready for offline localization. Check log: ${OFFLINE_VSLAM_ODOM_LOG_PATH}" >&2
        exit 1
    fi

    echo "  - replay source bag to record odom input bag"
    if [ "${PLAY_ALL_TOPICS}" = true ]; then
        echo "  - mode: all topics"
        play_rosbag_background \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH}"
    else
        build_online_source_play_topics
        echo "  - mode: filtered topics"
        echo "  - topics: ${SOURCE_PLAY_TOPICS[*]}"
        play_rosbag_background \
            "${SOURCE_BAG_PATH}" "${OFFLINE_VSLAM_ODOM_PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"
    fi

    if [ "${ODOM_READY_WAIT_ENABLED}" = true ]; then
        echo "  - wait for ${ODOM_TOPIC} to stabilize before recording"
        if ! wait_for_topic_rate_ready \
            "${ODOM_TOPIC}" \
            "${ODOM_READY_MIN_RATE_HZ}" \
            "${ODOM_READY_WINDOW}" \
            "${ODOM_READY_TIMEOUT_SEC}"; then
            echo "Timed out waiting for ${ODOM_TOPIC} to reach ${ODOM_READY_MIN_RATE_HZ} Hz." >&2
            stop_rosbag_playback
            exit 1
        fi
    else
        sleep 2
    fi

    echo "  - start odom bag recording"
    start_offline_odom_bag_recording "${OFFLINE_ODOM_BAG_DIR}" "${OFFLINE_VSLAM_ODOM_RECORD_LOG_PATH}"
    sleep 1

    if ! wait_for_rosbag_playback; then
        echo "rosbag replay failed while recording offline odom input bag." >&2
        exit 1
    fi

    sleep 2
    stop_recorder
    stop_vslam

    if [ ! -f "${OFFLINE_ODOM_BAG_DIR}/metadata.yaml" ]; then
        echo "Offline odom bag was not created correctly: ${OFFLINE_ODOM_BAG_DIR}" >&2
        exit 1
    fi

    BAG_PATH="${OFFLINE_ODOM_BAG_DIR}"
    OFFLINE_ODOM_BAG_CREATED=true

    echo "✅ Offline odom bag generated: ${OFFLINE_ODOM_BAG_DIR}"
}

stop_vslam() {
    stop_background_process "VSLAM_LAUNCH_PID" "VSLAM_LAUNCH_USES_SETSID"
    stop_background_process "CAMERA_CONTAINER_PID" "CAMERA_CONTAINER_USES_SETSID"
    stop_background_process "OFFLINE_TF_PID" "OFFLINE_TF_USES_SETSID"
}

stop_vslam_reference_recorder() {
    if [ "${VSLAM_REFERENCE_CAPTURE_STARTED}" = true ]; then
        echo "[ref] Stop VSLAM reference snapshot recorder"
    fi
    stop_background_process "VSLAM_REFERENCE_RECORDER_PID" "VSLAM_REFERENCE_RECORDER_USES_SETSID"
    VSLAM_REFERENCE_CAPTURE_STARTED=false
}

resolve_vslam_landmark_export_script() {
    if [ -n "${VSLAM_LANDMARK_EXPORT_SCRIPT_PATH}" ]; then
        if [ -f "${VSLAM_LANDMARK_EXPORT_SCRIPT_PATH}" ]; then
            echo "${VSLAM_LANDMARK_EXPORT_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_repo_file "ros2_ws/src/tools/vslam_map_tools/vslam_map_tools/export_landmarks_png.py"
}

resolve_vslam_reference_recorder_script() {
    resolve_repo_file "ros2_ws/src/tools/vslam_map_tools/vslam_map_tools/record_vslam_reference_snapshot.py"
}

resolve_vslam_reference_publisher_script() {
    resolve_repo_file "ros2_ws/src/tools/vslam_map_tools/vslam_map_tools/publish_saved_vslam_reference.py"
}

resolve_vslam_live_alignment_rviz_config() {
    if [ -n "${VSLAM_LIVE_ALIGNMENT_RVIZ_PATH}" ]; then
        if [ -f "${VSLAM_LIVE_ALIGNMENT_RVIZ_PATH}" ]; then
            echo "${VSLAM_LIVE_ALIGNMENT_RVIZ_PATH}"
            return 0
        fi
        return 1
    fi

    if resolve_system_launch_share_file "rviz" "vslam_map_alignment.rviz"; then
        return 0
    fi

    resolve_repo_file "ros2_ws/src/launch/system_launch/rviz/vslam_map_alignment.rviz"
}

prompt_live_vslam_alignment() {
    local align_choice
    local default_choice="n"

    if [ "${PIPELINE_MODE}" != "online" ]; then
        VSLAM_LIVE_ALIGNMENT_ENABLED=false
        return 0
    fi

    case "${VSLAM_LIVE_ALIGNMENT_MODE}" in
        always)
            VSLAM_LIVE_ALIGNMENT_ENABLED=true
            ;;
        never)
            VSLAM_LIVE_ALIGNMENT_ENABLED=false
            ;;
        auto)
            VSLAM_LIVE_ALIGNMENT_ENABLED=false
            if [ ! -t 0 ]; then
                return 0
            fi
            if [ "${VSLAM_LANDMARK_TRACE_MODE}" != "never" ]; then
                default_choice="y"
            fi
            echo ""
            if [ "${default_choice}" = "y" ]; then
                read -r -p "mapping 中に RViz2 で live alignment session を開きますか？ (Y/n, Enterで開く): " align_choice
                align_choice=${align_choice:-y}
            else
                read -r -p "mapping 中に RViz2 で live alignment session を開きますか？ (y/N, Enterでスキップ): " align_choice
                align_choice=${align_choice:-n}
            fi
            if [[ "${align_choice}" =~ ^[Yy]$ ]]; then
                VSLAM_LIVE_ALIGNMENT_ENABLED=true
            fi
            ;;
    esac
}

prompt_vslam_landmark_trace() {
    local trace_choice

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        VSLAM_LANDMARK_TRACE_ENABLED=false
        return 0
    fi

    case "${VSLAM_LANDMARK_TRACE_MODE}" in
        always)
            VSLAM_LANDMARK_TRACE_ENABLED=true
            ;;
        never)
            VSLAM_LANDMARK_TRACE_ENABLED=false
            ;;
        auto)
            VSLAM_LANDMARK_TRACE_ENABLED=false
            if [ ! -t 0 ]; then
                return 0
            fi
            echo ""
            read -r -p "VSLAM landmarks から tracing 用 map を作りますか？ (y/N, Enterでスキップ): " trace_choice
            if [[ "${trace_choice:-n}" =~ ^[Yy]$ ]]; then
                VSLAM_LANDMARK_TRACE_ENABLED=true
            fi
            ;;
    esac
}

run_live_vslam_alignment_session() {
    local rviz_config_path
    local alignment_status=0
    local -a alignment_cmd

    if [ "${VSLAM_LIVE_ALIGNMENT_ENABLED}" != true ]; then
        return 0
    fi

    if [ ! -t 0 ]; then
        echo "Warning: no interactive TTY is available. Skip live VSLAM alignment session." >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip live VSLAM alignment session." >&2
        return 0
    fi

    if ! command -v ros2 >/dev/null 2>&1; then
        echo "Warning: ros2 not found. Skip live VSLAM alignment session." >&2
        return 0
    fi

    if ! command -v rviz2 >/dev/null 2>&1; then
        echo "Warning: rviz2 not found. Skip live VSLAM alignment session." >&2
        return 0
    fi

    if ! rviz_config_path="$(resolve_vslam_live_alignment_rviz_config)"; then
        echo "Warning: RViz config not found for live alignment session." >&2
        return 0
    fi

    echo "[align] Launch RViz2 for live map/VSLAM alignment"
    launch_background_process "RVIZ_PID" "RVIZ_USES_SETSID" \
        rviz2 -d "${rviz_config_path}" --ros-args -p use_sim_time:=true

    sleep 3
    wait_for_topic "/map" 5 || true
    wait_for_topic "${VSLAM_LANDMARK_PATH_TOPIC}" 5 || true

    echo "[align] Use RViz2 to compare /map, landmarks, and slam_path."
    echo "[align] Adjust in this terminal, then press 'p' to save and continue."

    alignment_cmd=(
        ros2 run vslam_map_tools manual_tf_alignment_node.py
        --ros-args
        -p use_sim_time:=true
        -p parent_frame:=map
        -p child_frame:=vslam_map
        -p config_path:="${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}"
        -p enable_keyboard:=true
        -p exit_on_save:=true
    )

    "${alignment_cmd[@]}" || alignment_status=$?

    stop_background_process "RVIZ_PID" "RVIZ_USES_SETSID"

    if [ "${alignment_status}" -ne 0 ]; then
        echo "Warning: live VSLAM alignment session exited with status ${alignment_status}." >&2
        return 0
    fi

    if [ -f "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ]; then
        echo "[align] Saved alignment config: ${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}"
    else
        echo "Warning: live alignment session ended without saving ${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}." >&2
    fi
}

run_post_vslam_map_alignment_prep() {
    local rviz_config_path=""
    local reference_publisher_script_path=""
    local manual_alignment_cmd=""
    local manual_alignment_cmd_single_line=""
    local manual_alignment_helper_path=""
    local rviz_cmd=""
    local dummy=""
    local -a launch_args
    local -a reference_publish_cmd

    if [ "${PREPARE_VSLAM_MAP_ALIGNMENT}" != true ]; then
        return 0
    fi

    if [ ! -t 0 ]; then
        echo "Warning: no interactive TTY is available. Skip post-map map/vslam_map alignment prep." >&2
        return 0
    fi

    if [ ! -f "${MAP_YAML_PATH}" ]; then
        echo "Warning: map YAML not found for post-map map/vslam_map alignment prep: ${MAP_YAML_PATH}" >&2
        return 0
    fi

    echo "[prep] Launch 2D map publisher for manual map/vslam_map alignment"

    launch_args=(
        "use_sim_time:=false"
        "map_yaml_path:=${MAP_YAML_PATH}"
        "map_server_use_sim_time:=false"
    )

    build_system_launch_cmd "map_publisher.launch.xml"
    launch_background_process "POST_ALIGNMENT_STACK_PID" "POST_ALIGNMENT_STACK_USES_SETSID" \
        "${SYSTEM_LAUNCH_CMD[@]}" \
        "${launch_args[@]}" \
        > "${POST_ALIGNMENT_LOG_PATH}" 2>&1

    if wait_for_topic "/map" 20; then
        echo "  - /map is available"
    else
        echo "Warning: /map did not appear. Check log: ${POST_ALIGNMENT_LOG_PATH}" >&2
    fi

    if [ -f "${VSLAM_REFERENCE_SNAPSHOT_PATH}" ]; then
        if ! command -v python3 >/dev/null 2>&1; then
            echo "Warning: python3 not found. Skip saved VSLAM reference republisher." >&2
        elif reference_publisher_script_path="$(resolve_vslam_reference_publisher_script 2>/dev/null || true)"; then
            if [ -n "${reference_publisher_script_path}" ]; then
                reference_publish_cmd=(
                    python3 "${reference_publisher_script_path}"
                    --input "${VSLAM_REFERENCE_SNAPSHOT_PATH}"
                    --path-topic "${VSLAM_LANDMARK_PATH_TOPIC}"
                    --odom-topic "${DEFAULT_VSLAM_ODOM_TOPIC}"
                    --publish-rate-hz 5.0
                )
                launch_background_process "POST_ALIGNMENT_REFERENCE_PUBLISHER_PID" "POST_ALIGNMENT_REFERENCE_PUBLISHER_USES_SETSID" \
                    "${reference_publish_cmd[@]}" \
                    > "${POST_ALIGNMENT_REFERENCE_PUBLISHER_LOG_PATH}" 2>&1
                if wait_for_topic "${VSLAM_LANDMARK_PATH_TOPIC}" 10; then
                    echo "  - ${VSLAM_LANDMARK_PATH_TOPIC} is active (saved reference replay)"
                fi
            fi
        fi
    else
        if [ "${VSLAM_REFERENCE_CAPTURE_EXPECTED}" != true ]; then
            echo "[prep] Saved VSLAM reference snapshot was not requested in this mode."
        elif [ "${VSLAM_REFERENCE_CAPTURE_WAS_STARTED}" = true ]; then
            echo "Warning: saved VSLAM reference snapshot not found: ${VSLAM_REFERENCE_SNAPSHOT_PATH}" >&2
            echo "Warning: recorder did start. Check log for QoS / initialization issues: ${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}" >&2
        else
            echo "Warning: saved VSLAM reference snapshot not found: ${VSLAM_REFERENCE_SNAPSHOT_PATH}" >&2
            echo "Warning: recorder never reported a successful start in this run." >&2
            echo "Warning: expected recorder log path: ${VSLAM_REFERENCE_SNAPSHOT_LOG_PATH}" >&2
        fi
    fi

    manual_alignment_cmd="$(cat <<EOF
ros2 run vslam_map_tools manual_tf_alignment_node.py --ros-args \\
  -p use_sim_time:=false \\
  -p parent_frame:=map \\
  -p child_frame:=vslam_map \\
  -p config_path:=${VSLAM_MAP_ALIGNMENT_CONFIG_PATH} \\
  -p enable_keyboard:=true
EOF
)"
    manual_alignment_cmd_single_line="ros2 run vslam_map_tools manual_tf_alignment_node.py --ros-args -p use_sim_time:=false -p parent_frame:=map -p child_frame:=vslam_map -p config_path:=${VSLAM_MAP_ALIGNMENT_CONFIG_PATH} -p enable_keyboard:=true"
    manual_alignment_helper_path="${OUT_DIR}/run_manual_tf_alignment.sh"
    cat > "${manual_alignment_helper_path}" <<EOF
#!/bin/bash
set -eo pipefail

for candidate in \\
  "${REPO_ROOT}/install/setup.bash" \\
  "/workspaces/install/setup.bash" \\
  "install/setup.bash"; do
  if [ -f "\${candidate}" ]; then
    # shellcheck disable=SC1090
    set +u
    source "\${candidate}"
    set -u
    break
  fi
done

exec ros2 run vslam_map_tools manual_tf_alignment_node.py --ros-args \\
  -p use_sim_time:=false \\
  -p parent_frame:=map \\
  -p child_frame:=vslam_map \\
  -p config_path:=${VSLAM_MAP_ALIGNMENT_CONFIG_PATH} \\
  -p enable_keyboard:=true
EOF
    chmod +x "${manual_alignment_helper_path}"

    echo ""
    echo "[prep] Alignment helper stack is running."
    echo "  - map yaml     : ${MAP_YAML_PATH}"
    echo "  - path snapshot: ${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    echo "  - map log      : ${POST_ALIGNMENT_LOG_PATH}"
    if [ -n "${POST_ALIGNMENT_REFERENCE_PUBLISHER_PID}" ]; then
        echo "  - path log     : ${POST_ALIGNMENT_REFERENCE_PUBLISHER_LOG_PATH}"
    fi
    echo ""
    echo "[prep] In another pane/window, first source your workspace if needed:"
    echo "  source install/setup.bash"
    echo ""
    echo "[prep] Easiest option:"
    echo "  bash ${manual_alignment_helper_path}"
    echo ""
    echo "[prep] Then run either of these:"
    echo "[prep] Multiline:"
    printf '%s\n' "${manual_alignment_cmd}"
    echo ""
    echo "[prep] Single line:"
    echo "  ${manual_alignment_cmd_single_line}"
    echo ""
    echo "[prep] No VSLAM node is launched in this second step."
    echo "[prep] Fixed Frame は vslam_map にして、saved slam_path を基準に map を合わせてください。"
    echo "[prep] map は manual_tf_alignment_node の map -> vslam_map で動きます。"

    if rviz_config_path="$(resolve_vslam_live_alignment_rviz_config 2>/dev/null || true)"; then
        if [ -n "${rviz_config_path}" ]; then
            rviz_cmd="rviz2 -d ${rviz_config_path}"
            echo "[prep] Suggested RViz command if you need another viewer:"
            echo "  ${rviz_cmd}"
        fi
    fi

    echo "[prep] Stop manual_tf_alignment_node yourself when you're done adjusting."
    read -r -p "[prep] Press Enter here to stop the helper stack and continue: " dummy

    stop_post_alignment_stack
}

run_vslam_landmark_trace() {
    local export_script_path
    local map_edit_script_path
    local trace_load_map_path=""
    local export_status=0
    local old_vslam_vis_enabled="${VSLAM_VIS_ENABLED}"
    local -a export_cmd
    local -a trace_edit_cmd

    if [ "${VSLAM_LANDMARK_TRACE_ENABLED}" != true ]; then
        echo "[prep] Skip VSLAM landmark tracing helper"
        return 0
    fi

    echo "[prep] Export VSLAM landmarks for tracing"

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip VSLAM landmark tracing helper." >&2
        return 0
    fi

    if ! export_script_path="$(resolve_vslam_landmark_export_script)"; then
        if [ -n "${VSLAM_LANDMARK_EXPORT_SCRIPT_PATH}" ]; then
            echo "Warning: landmark export script not found: ${VSLAM_LANDMARK_EXPORT_SCRIPT_PATH}" >&2
        else
            echo "Warning: export_landmarks_png.py not found. Skip VSLAM landmark tracing helper." >&2
        fi
        return 0
    fi

    if ! map_edit_script_path="$(resolve_map_edit_script)"; then
        echo "Warning: map cleanup editor not found. Skip VSLAM landmark tracing helper." >&2
        return 0
    fi

    if [ -z "${VSLAM_LANDMARK_REFERENCE_YAML_PATH}" ]; then
        VSLAM_LANDMARK_REFERENCE_YAML_PATH="${MAP_YAML_PATH}"
    fi

    if [ ! -f "${VSLAM_LANDMARK_REFERENCE_YAML_PATH}" ]; then
        echo "Warning: reference YAML not found for landmark export: ${VSLAM_LANDMARK_REFERENCE_YAML_PATH}" >&2
        return 0
    fi

    if [ -d "${VSLAM_MAP_DIR}" ]; then
        trace_load_map_path="${VSLAM_MAP_DIR}"
    fi

    VSLAM_VIS_ENABLED=true

    launch_vslam_stack \
        "${VSLAM_LANDMARK_TF_LOG_PATH}" \
        "${VSLAM_LANDMARK_LOG_PATH}" \
        "" \
        "${trace_load_map_path}"

    if ! wait_for_service "/visual_slam/save_map" 60; then
        echo "Warning: Visual SLAM service not ready for landmark export. Check log: ${VSLAM_LANDMARK_LOG_PATH}" >&2
        stop_vslam
        VSLAM_VIS_ENABLED="${old_vslam_vis_enabled}"
        return 0
    fi

    export_cmd=(
        python3 "${export_script_path}"
        --landmarks-topic "${VSLAM_LANDMARK_TOPIC}"
        --path-topic "${VSLAM_LANDMARK_PATH_TOPIC}"
        --target-frame "${VSLAM_LANDMARK_TARGET_FRAME}"
        --reference-yaml "${VSLAM_LANDMARK_REFERENCE_YAML_PATH}"
        --output-image "${VSLAM_LANDMARK_IMAGE_PATH}"
        --output-yaml "${VSLAM_LANDMARK_YAML_PATH}"
        --timeout-sec "${VSLAM_LANDMARK_EXPORT_TIMEOUT_SEC}"
    )

    launch_background_process "LANDMARK_EXPORT_PID" "LANDMARK_EXPORT_USES_SETSID" \
        "${export_cmd[@]}" \
        > "${VSLAM_LANDMARK_EXPORT_LOG_PATH}" 2>&1

    sleep 2

    build_online_source_play_topics
    echo "  - replay source bag for VSLAM landmark export"
    echo "  - topics: ${SOURCE_PLAY_TOPICS[*]}"
    if ! play_rosbag \
        "${SOURCE_BAG_PATH}" "${VSLAM_LANDMARK_PLAYER_LOG_PATH}" "${SOURCE_PLAY_TOPICS[@]}"; then
        echo "Warning: source bag replay failed during VSLAM landmark export." >&2
        stop_background_process "LANDMARK_EXPORT_PID" "LANDMARK_EXPORT_USES_SETSID"
        stop_vslam
        VSLAM_VIS_ENABLED="${old_vslam_vis_enabled}"
        return 0
    fi

    if [ -n "${LANDMARK_EXPORT_PID:-}" ]; then
        wait "${LANDMARK_EXPORT_PID}" || export_status=$?
        LANDMARK_EXPORT_PID=""
        LANDMARK_EXPORT_USES_SETSID=false
    fi

    stop_vslam
    VSLAM_VIS_ENABLED="${old_vslam_vis_enabled}"

    if [ "${export_status}" -ne 0 ] || [ ! -f "${VSLAM_LANDMARK_IMAGE_PATH}" ]; then
        echo "Warning: failed to export VSLAM landmarks. Check log: ${VSLAM_LANDMARK_EXPORT_LOG_PATH}" >&2
        return 0
    fi

    echo "  - landmark PNG: ${VSLAM_LANDMARK_IMAGE_PATH}"
    if [ -f "${VSLAM_LANDMARK_YAML_PATH}" ]; then
        echo "  - landmark YAML: ${VSLAM_LANDMARK_YAML_PATH}"
    fi

    echo "[prep] Launch tracing editor from VSLAM landmarks"
    trace_edit_cmd=(
        python3 "${map_edit_script_path}"
        --input "${VSLAM_LANDMARK_IMAGE_PATH}"
        --output "${VSLAM_TRACE_OUTPUT_PATH}"
        --initialize-mode blank_black
    )

    if ! "${trace_edit_cmd[@]}"; then
        echo "Warning: landmark tracing editor failed. Keep original map for centerline." >&2
        return 0
    fi

    if [ -f "${VSLAM_TRACE_OUTPUT_PATH}" ]; then
        CENTERLINE_INPUT_MAP="${VSLAM_TRACE_OUTPUT_PATH}"
        VSLAM_LANDMARK_TRACE_COMPLETED=true
        echo "  - traced map: ${VSLAM_TRACE_OUTPUT_PATH}"
    else
        echo "Warning: traced map was not saved. Keep original map for centerline." >&2
    fi
}

