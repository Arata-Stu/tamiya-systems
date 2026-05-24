#!/bin/bash

usage() {
    cat <<'EOF'
Usage:
  create_2d_map_from_bag.sh [OPTIONS]

Options:
  --mode NAME         no_odom_offline_vslam|no_odom_online_vslam|
                      with_odom_offline_vslam|with_odom_online_vslam
                      aliases: default=no_odom_offline_vslam,
                               2d_slam=no_odom_online_vslam
  --bag-path DIR      input rosbag2 directory (skip interactive selection)
  --map-name NAME     output map name (skip interactive prompt)
  --scan-topic TOPIC  scan topic for cartographer (default: /scan)
  --rate RATE         ros2 bag play rate (default: 1.0)
  --odom-topic TOPIC  enable odometry in cartographer and set the odom topic
                      default for with_odom_* modes: /visual_slam/tracking/odometry
  --odom-ready-window N
                      number of odom messages used for stability check (default: 10)
  --odom-ready-min-rate HZ
                      minimum average odom rate before recording starts
                      default: 90% of --image-fps in with_odom modes
  --odom-ready-timeout SEC
                      timeout while waiting for odom rate stabilization (default: 45)
  --no-odom-ready-wait
                      disable odom-rate stabilization wait in with_odom modes
  --run-vslam         compatibility override: force online_vslam execution
  --no-vslam          compatibility override: force offline_vslam execution
  --vslam-vis         enable VSLAM visualization topics during parallel VSLAM execution
  --no-vslam-vis      disable VSLAM visualization topics (default)
  --image-width PX    camera width for parallel VSLAM launch (default: 424)
  --image-height PX   camera height for parallel VSLAM launch (default: 240)
  --image-fps FPS     camera fps for parallel VSLAM launch (default: 90.0)
  --with-imu          replay /camera/imu as well (default: enabled)
  --no-imu            do not replay /camera/imu
  --use-image-preprocessors
                      run rectify/mono preprocessing before VSLAM
  --no-image-preprocessors
                      make VSLAM subscribe to recorded camera topics directly (default)
  --launch-offline-tf publish fallback base_link TFs instead of using only bag TFs
  --vslam-map-dir DIR visual slam map output directory
  --pipeline-mode MODE
                      offline|online|auto
                      compatibility override for VSLAM execution mode
  --play-all-topics   play all topics in bag (default: play only needed topics)
  --record-root DIR   rosbag探索ルート (default: /record)
  --no-scp            skip interactive scp transfer step
  --no-centerline     skip centerline CSV generation
  --no-raceline       skip raceline CSV generation
  --no-line-preview   skip centerline/raceline preview image generation
  --edit-map          always launch GUI map cleanup editor before centerline generation
  --no-edit-map       never launch GUI map cleanup editor
  --map-edit-mode MODE
                      auto|always|never (default: auto)
  --map-edit-script PATH
                      path to map_cleanup_editor.py (auto-detect by default)
  --map-edit-output PATH
                      path to cleaned PNG output (default: <MAP_NAME>_centerline_input.png)
  --trace-vslam-landmarks
                      replay source bag with VSLAM landmarks, export a tracing PNG,
                      and launch the editor in blank-canvas tracing mode
  --no-trace-vslam-landmarks
                      disable the VSLAM landmark tracing helper
  --vslam-landmark-trace-mode MODE
                      auto|always|never (default: never)
  --vslam-map-alignment-config PATH
                      saved map->vslam_map alignment YAML used when replaying VSLAM
  --prepare-vslam-map-alignment
                      after map generation, launch 2D map publishing plus saved
                      VSLAM path/odom republish for external manual_tf_alignment_node work
  --no-prepare-vslam-map-alignment
                      skip the post-map alignment helper stack (default)
  --live-vslam-map-align
                      during online mapping, open RViz2 and run a live map->vslam_map
                      alignment session before saving outputs
  --no-live-vslam-map-align
                      disable the live alignment session
  --live-vslam-map-align-mode MODE
                      auto|always|never (default: auto)
  --live-vslam-map-align-rviz PATH
                      rviz config for the live alignment session
  --vslam-landmark-export-script PATH
                      path to export_landmarks_png.py (auto-detect by default)
  --vslam-landmark-image PATH
                      output landmark PNG path (default: <MAP_NAME>_vslam_landmarks.png)
  --vslam-landmark-yaml PATH
                      output landmark YAML path (default: <MAP_NAME>_vslam_landmarks.yaml)
  --vslam-landmark-reference-yaml PATH
                      reference YAML used for resolution/origin/image size
                      (default: generated provisional map YAML)
  --vslam-landmark-target-frame FRAME
                      target frame passed to the landmark exporter (default: map)
  --vslam-trace-output PATH
                      traced PNG path used for centerline generation
                      (default: <MAP_NAME>_vslam_traced.png)
  --centerline-debug  save centerline debug images (default: enabled when centerline is generated)
  --centerline-debug-dir DIR
                      set centerline debug image output directory
  --centerline-script PATH
                      path to generate_centerline.py (auto-detect by default)
  --line-preset PRESET
                      default|race-stacks for centerline/raceline helper scripts (default: default)
  --centerline-direction DIR
                      forward|reverse|both (default: forward)
  --raceline-script PATH
                      path to generate_raceline.py (auto-detect by default)
  --raceline-backend BACKEND
                      heuristic|global-opt|auto (default: heuristic)
  --raceline-opt-type TYPE
                      shortest_path|mincurv|mincurv_iqp for global-opt (default: mincurv_iqp)
  --raceline-direction DIR
                      forward|reverse|both (default: forward)
  --optimizer-root DIR
                      path to global_racetrajectory_optimization checkout
  --line-preview-script PATH
                      path to visualize_race_lines.py (auto-detect by default)
  -h, --help          show this help

When --mode is omitted:
  the script interactively prompts you to choose one of the 4 mode presets

Mode presets:
  no_odom_offline_vslam:
      Cartographer は scan-only。VSLAM はこのスクリプトでは起動しない
  no_odom_online_vslam:
      Cartographer は scan-only。bag replay と同時に VSLAM も起動する
  with_odom_offline_vslam:
      先に VSLAM map を作り、その map で odom bag を生成してから Cartographer に渡す
  with_odom_online_vslam:
      Cartographer は replay 中に起動した VSLAM の live odom を使う

Outputs:
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pbstream
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.yaml
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.pgm
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>.png (optional; generated if converter is available)
  /map/<bag_name>/<MAP_NAME>/cuvslam_map/ (optional; generated in online VSLAM or offline odom modes)
  /map/<bag_name>/<MAP_NAME>/vslam_map_alignment.yaml (optional; generated by manual_tf_alignment_node)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_vslam_reference.json (optional; saved path/odom reference)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.png (optional; generated with --trace-vslam-landmarks)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_vslam_landmarks.yaml (optional; generated with --trace-vslam-landmarks)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_vslam_traced.png (optional; generated with --trace-vslam-landmarks)
  /map/<bag_name>/<MAP_NAME>/offline_vslam_odom_input_<timestamp>/ (optional; generated in with_odom_offline_vslam)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline_input.png (optional; hand-edited map cleanup result)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_centerline.csv (optional; generated unless --no-centerline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_raceline.csv (optional; generated unless --no-raceline)
  /map/<bag_name>/<MAP_NAME>/<MAP_NAME>_lines.png (optional; generated unless --no-line-preview)

After map creation:
  optionally transfer /map/<bag_name>/<MAP_NAME>/ to remote host by scp

Interactive flow:
  1) mode を選択（--mode 省略時）
  2) /record を再帰探索して metadata.yaml を持つ rosbag2 ディレクトリを一覧表示
  3) 番号で rosbag を選択
  4) map 名を入力
  5) 選択した mode に応じて Cartographer / VSLAM を実行して map を生成
  6) 必要なら 2D map publish + 保存した VSLAM path/odom を再publishし、別paneで map/vslam_map を調整
  7) 必要なら landmarks replay から tracing 用 PNG を作って手描き map を保存
  8) centerline 生成の可否を確認（debug はデフォルト有効）
  9) 必要なら GUI で map PNG/PGM を黒塗り修正して保存
  10) raceline 生成の可否を確認
  11) centerline / raceline preview画像を生成
  12) 転送前メニューで section edit / scp / 終了 を選択
EOF
}

prompt_mode_interactive() {
    local choice

    echo ""
    echo "2D map 作成 mode を選択してください:"
    echo "  1) no_odom_offline_vslam   Cartographer=scan-only / VSLAMは別実行"
    echo "  2) no_odom_online_vslam    Cartographer=scan-only / 今回の実行でVSLAMも起動"
    echo "  3) with_odom_offline_vslam 先にVSLAM map + odom bagを作ってからCartographer"
    echo "  4) with_odom_online_vslam  Cartographer=live VSLAM odom使用 / 今回の実行でVSLAMも起動"
    echo ""

    while :; do
        read -r -p "番号で選択してください (1-4, Enterで '1'): " choice
        choice="${choice:-1}"
        case "${choice}" in
            1)
                MODE="no_odom_offline_vslam"
                return 0
                ;;
            2)
                MODE="no_odom_online_vslam"
                return 0
                ;;
            3)
                MODE="with_odom_offline_vslam"
                return 0
                ;;
            4)
                MODE="with_odom_online_vslam"
                return 0
                ;;
            *)
                echo "無効な入力です。1-4 を選択してください。"
                ;;
        esac
    done
}

describe_odom_source() {
    if [ "${CARTOGRAPHER_USE_ODOM}" != true ]; then
        echo "disabled"
        return 0
    fi

    if [ "${PIPELINE_MODE}" = "online" ]; then
        echo "live VSLAM output (${ODOM_TOPIC})"
    elif [ "${OFFLINE_ODOM_BAG_CREATED}" = true ]; then
        echo "offline-generated VSLAM odom bag (${ODOM_TOPIC})"
    else
        echo "pre-recorded odom bag (${ODOM_TOPIC})"
    fi
}

print_mode_summary() {
    local effective_mode
    effective_mode="$(resolve_effective_mode)"

    echo ""
    echo "================ Map build mode ================"
    echo "mode            : ${effective_mode}"
    echo "source bag      : ${SOURCE_BAG_PATH}"
    echo "cartographer bag: ${BAG_PATH}"
    echo "scan topic      : ${SCAN_TOPIC}"
    if [ "${CARTOGRAPHER_USE_ODOM}" = true ]; then
        echo "cartographer odom: enabled (${ODOM_TOPIC})"
        if odom_ready_wait_applicable && [ "${ODOM_READY_WAIT_ENABLED}" = true ]; then
            echo "odom ready wait : window=${ODOM_READY_WINDOW}, min_rate=${ODOM_READY_MIN_RATE_HZ} Hz, timeout=${ODOM_READY_TIMEOUT_SEC}s"
        elif odom_ready_wait_applicable; then
            echo "odom ready wait : disabled"
        else
            echo "odom ready wait : n/a for live online odom"
        fi
    else
        echo "cartographer odom: disabled"
    fi
    echo "vslam execution : ${PIPELINE_MODE}"
    echo "odom source     : $(describe_odom_source)"
    echo "vslam vis       : ${VSLAM_VIS_ENABLED}"
    echo "align prep      : ${PREPARE_VSLAM_MAP_ALIGNMENT}"
    if [ "${PREPARE_VSLAM_MAP_ALIGNMENT}" = true ]; then
        echo "vslam ref snap  : ${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    fi
    if [ -n "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ] && [ -f "${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}" ]; then
        echo "vslam alignment : ${VSLAM_MAP_ALIGNMENT_CONFIG_PATH}"
    else
        echo "vslam alignment : identity / no saved config"
    fi
    echo "live alignment  : ${VSLAM_LIVE_ALIGNMENT_MODE}"
    echo "================================================"
    echo ""
}

prompt_pre_transfer_action() {
    local action_choice

    while true; do
        echo ""
        echo "転送前の操作を選んでください:"
        echo "  1) section edit を開く"
        echo "  2) scp 転送へ進む"
        echo "  3) 何もせず終了"
        read -r -p "選択 [2]: " action_choice

        case "${action_choice:-2}" in
            1|section|sections|edit|s)
                run_section_edit
                ;;
            2|transfer|scp|t)
                return 0
                ;;
            3|skip|exit|quit|q)
                echo "転送をスキップしました。"
                exit 0
                ;;
            *)
                echo "無効な選択です: ${action_choice}" >&2
                ;;
        esac
    done
}
