#!/bin/bash

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [mode] [options] [-- extra_ros2_launch_args...]

Modes:
  production              Full production run with VSLAM + localization + section localizer
  sensor_data_recording   Sensor-recording preset for initial mapping runs
  identification          VSLAM odom + final cmd_drive recording preset for MAP lookup generation
  localization_eval       Lean localization evaluation preset (VSLAM ref + global localization)
  perception_eval         Lean LiDAR-camera perception evaluation preset
  vslam_eval              Lean VSLAM-only evaluation preset
  rule_base               Rule-base autonomous driving (Localization + Planning + Map Controller)
  base                    Alias of production
  mapping                 Alias of sensor_data_recording
  vslam_map               Alias of sensor_data_recording
  map_lookup_recording    Alias of identification
  map_lookup              Alias of identification

Options:
  -i, --interactive       Toggle launch arguments interactively before running
  --legacy-interactive    Use the old line-based interactive prompt
  -n, --dry-run           Print the command without running it
  --bag-manager NAME      Select bag manager yaml: default|mapping|e2e|PATH
  --e2e VARIANT           Enable unified E2E: camera|lidar|camera_trajectory|lidar_trajectory
  --set KEY=VALUE         Override an argument (also accepts KEY:=VALUE)
  -h, --help              Show this help

Examples:
  ${SCRIPT_NAME} production -- map_dir:=/map/mybag/mycourse
  ${SCRIPT_NAME} sensor_data_recording
  ${SCRIPT_NAME} localization_eval --set map_dir=/map/mybag/mycourse
  ${SCRIPT_NAME} perception_eval
  ${SCRIPT_NAME} vslam_eval --set map_dir=/map/mybag/mycourse
  ${SCRIPT_NAME} sensor_data_recording -i
  ${SCRIPT_NAME} sensor_data_recording --bag-manager mapping
  ${SCRIPT_NAME} sensor_data_recording --set record=false --dry-run
  ${SCRIPT_NAME} production --set map_dir=/map/mybag/mycourse
  ${SCRIPT_NAME} production --set use_perception=true
  ${SCRIPT_NAME} production --set use_perception=true --set use_perception_classifier=true
  ${SCRIPT_NAME} production --set map_dir=/map/mybag/mycourse --set use_control_filter=true
  ${SCRIPT_NAME} vslam_eval --set map_dir=/map/mybag/mycourse --set use_camera=false --set use_hd_map=true --set use_planning=true --set localize_on_startup=true --set planning_publish_local_path=false --set planning_publish_local_reference=false
  ${SCRIPT_NAME} production --e2e lidar_trajectory
  ${SCRIPT_NAME} production --set use_drive_mode_manager=true
  ${SCRIPT_NAME} identification
  ${SCRIPT_NAME} -i -- map_dir:=/map/mybag/mycourse
EOF
}

bag_manager_label() {
  local idx

  for idx in "${!BAG_MANAGER_PATHS[@]}"; do
    if [[ "$ARG_bag_manager_param" == "${BAG_MANAGER_PATHS[$idx]}" ]]; then
      echo "${BAG_MANAGER_PRESETS[$idx]}"
      return
    fi
  done

  echo "custom"
}

vehicle_profile_label() {
  local current

  if ! mapping_vehicle_profile_applicable; then
    echo "(n/a)"
    return 0
  fi

  current="$(current_vehicle_param)"
  case "$current" in
    "$VEHICLE_MAPPING_PARAM_PATH")
      echo "slow (throttle_gain=0.1)"
      ;;
    ""|"$VEHICLE_DEFAULT_PARAM_PATH")
      echo "normal (throttle_gain=1.0)"
      ;;
    *)
      echo "custom (${current})"
      ;;
  esac
}

map_dir_label() {
  local current
  current="$(current_map_dir)"
  if [[ -n "$current" ]]; then
    echo "$current"
  else
    echo "(unset)"
  fi
}

choose_map_dir_interactive() {
  local answer
  local idx
  local current

  current="$(current_map_dir)"
  echo ""
  echo "Map directory:"

  if discover_map_candidates; then
    for idx in "${!MAP_CANDIDATES[@]}"; do
      printf "  %2d) %s\n" "$((idx + 1))" "${MAP_CANDIDATES[$idx]}"
    done
    echo "   c) custom path"
    echo "   x) clear"
    read -r -p "Select map dir [${current:-none}]: " answer

    [[ -z "$answer" ]] && return

    case "$answer" in
      c|C)
        read -r -p "map_dir path: " answer
        [[ -n "$answer" ]] && set_extra_arg_value "map_dir" "$answer"
        ;;
      x|X)
        clear_extra_arg_value "map_dir"
        ;;
      *)
        if [[ "$answer" =~ ^[0-9]+$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le "${#MAP_CANDIDATES[@]}" ]; then
          set_extra_arg_value "map_dir" "${MAP_CANDIDATES[$((answer - 1))]}"
        else
          echo "Invalid map dir selection: $answer" >&2
        fi
        ;;
    esac
    return
  fi

  echo "  No map candidates found under ${MAP_SEARCH_ROOT}"
  echo "  c) custom path"
  echo "  x) clear"
  read -r -p "Select map dir [${current:-none}]: " answer

  case "$answer" in
    c|C)
      read -r -p "map_dir path: " answer
      [[ -n "$answer" ]] && set_extra_arg_value "map_dir" "$answer"
      ;;
    x|X)
      clear_extra_arg_value "map_dir"
      ;;
  esac
}

choose_vehicle_profile_interactive() {
  local answer
  local current

  if ! mapping_vehicle_profile_applicable; then
    return 0
  fi

  current="$(vehicle_profile_label)"
  echo ""
  echo "Vehicle speed profile:"
  echo "  1) slow     throttle_gain=0.1  (mapping向け, 推奨)"
  echo "  2) normal   throttle_gain=1.0"
  echo "  c) custom vehicle_param path"
  read -r -p "Select vehicle profile [${current}]: " answer

  [[ -z "$answer" ]] && return 0

  case "$answer" in
    1|slow|Slow)
      set_extra_arg_value "vehicle_param" "$VEHICLE_MAPPING_PARAM_PATH"
      ;;
    2|normal|Normal)
      set_extra_arg_value "vehicle_param" "$VEHICLE_DEFAULT_PARAM_PATH"
      ;;
    c|C)
      read -r -p "vehicle_param path: " answer
      [[ -n "$answer" ]] && set_extra_arg_value "vehicle_param" "$answer"
      ;;
    *)
      echo "Invalid vehicle profile selection: $answer" >&2
      ;;
  esac
}

choose_bag_manager_interactive() {
  local answer
  local idx

  echo ""
  echo "Bag manager param:"
  for idx in "${!BAG_MANAGER_PRESETS[@]}"; do
    printf "  %d) %-8s %s\n" "$((idx + 1))" "${BAG_MANAGER_PRESETS[$idx]}" "${BAG_MANAGER_PATHS[$idx]}"
  done
  echo "  c) custom path"
  read -r -p "Select bag manager [$(bag_manager_label)]: " answer

  [[ -z "$answer" ]] && return

  case "$answer" in
    1|default)
      set_bag_manager_preset default
      ;;
    2|mapping)
      set_bag_manager_preset mapping
      ;;
    3|e2e)
      set_bag_manager_preset e2e
      ;;
    c|C)
      read -r -p "bag_manager_param path: " answer
      [[ -n "$answer" ]] && ARG_bag_manager_param="$answer"
      ;;
    *)
      echo "Invalid bag manager selection: $answer" >&2
      ;;
  esac
}

choose_mode_interactive() {
  # モード定義: 表示ラベルと内部名
  local -a MODE_LABELS=(
    "production       (= base)              本番走行: VSLAM + 自己位置推定 + セクションローカライザ"
    "mapping          (= sensor_data_recording, vslam_map)  センサ録画: マッピング用rosbag収集"
    "identification  (= map_lookup_recording, map_lookup)  MAP lookup生成用: VSLAM odom + 実車cmd_drive録画"
    "localization_eval                       自己位置推定の精度検証 (VSLAM + 全体位置推定)"
    "perception_eval                         LiDAR/カメラ知覚の評価"
    "vslam_eval                              VSLAM単体の評価"
    "rule_base                               ルールベース自動運転 (Localization + Planning + Map Controller)"
  )
  local -a MODE_VALUES=(
    "production"
    "sensor_data_recording"
    "identification"
    "localization_eval"
    "perception_eval"
    "vslam_eval"
    "rule_base"
  )
  local n_modes="${#MODE_LABELS[@]}"
  local cursor=0
  local key
  local idx

  # 現在の MODE にカーソルを合わせる
  for idx in "${!MODE_VALUES[@]}"; do
    if [[ "${MODE_VALUES[$idx]}" == "$MODE" ]]; then
      cursor="$idx"
      break
    fi
  done

  while true; do
    tui_clear_screen
    echo "モードの選択"
    echo ""
    for idx in "${!MODE_LABELS[@]}"; do
      local marker=" "
      if [[ "$idx" -eq "$cursor" ]]; then marker=">"; fi
      printf " %s  %s\n" "$marker" "${MODE_LABELS[$idx]}"
    done
    echo ""
    echo "j/k または ↑↓: 移動  Enter: 決定  q: 終了"

    key="$(tui_read_key)"
    case "$key" in
      j|$'\033[B') cursor=$(( (cursor + 1) % n_modes )) ;;
      k|$'\033[A') cursor=$(( (cursor + n_modes - 1) % n_modes )) ;;
      ""|$'\n'|$'\r')
        MODE="${MODE_VALUES[$cursor]}"
        tui_clear_screen
        return 0
        ;;
      q|Q)
        tui_clear_screen
        return 0
        ;;
    esac
  done
}

toggle_interactive_legacy() {
  local answer
  local key
  local idx

  while true; do
    echo ""
    echo "Launch arguments:"
    for idx in "${!BOOL_KEYS[@]}"; do
      key="${BOOL_KEYS[$idx]}"
      printf "  %2d) %-32s %s\n" "$((idx + 1))" "$key" "$(get_arg "$key")"
    done
    printf "      %-32s %s (%s)\n" "bag_manager_param" "$ARG_bag_manager_param" "$(bag_manager_label)"
    printf "      %-32s %s\n" "map_dir" "$(map_dir_label)"
    if mapping_vehicle_profile_applicable; then
      printf "      %-32s %s\n" "vehicle_speed_profile" "$(vehicle_profile_label)"
    fi
    echo ""
    echo "Enter numbers to toggle, 'b' for bag manager, 'm' for map dir, 'v' for vehicle speed, 's KEY=VALUE' to set a value, or Enter to run."
    read -r -p "> " answer

    [[ -z "$answer" ]] && break

    if [[ "$answer" == "b" || "$answer" == "B" ]]; then
      choose_bag_manager_interactive
      continue
    fi

    if [[ "$answer" == "m" || "$answer" == "M" ]]; then
      choose_map_dir_interactive
      continue
    fi

    if [[ "$answer" == "v" || "$answer" == "V" ]]; then
      choose_vehicle_profile_interactive
      continue
    fi

    if [[ "$answer" == s\ * ]]; then
      set_arg "${answer#s }"
      continue
    fi

    for idx in $answer; do
      if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#BOOL_KEYS[@]}" ]; then
        key="${BOOL_KEYS[$((idx - 1))]}"
        if [[ "$(get_arg "$key")" == "true" ]]; then
          set_arg "${key}=false"
        else
          set_arg "${key}=true"
        fi
      else
        echo "Skipped invalid selection: $idx" >&2
      fi
    done
  done
}

toggle_bool_key() {
  local key="$1"

  if [[ "$(get_arg "$key")" == "true" ]]; then
    set_arg "${key}=false"
  else
    set_arg "${key}=true"
  fi
}

render_launch_extra_interactive() {
  local e2e_variant

  e2e_variant="$(current_e2e_variant)"
  if [[ -z "$e2e_variant" ]]; then
    e2e_variant="(unset)"
  fi

  printf "   %-38s %s (%s)\n" "bag_manager_param" "$ARG_bag_manager_param" "$(bag_manager_label)"
  printf "   %-38s %s\n" "map_dir" "$(map_dir_label)"
  printf "   %-38s %s\n" "e2e_variant" "$e2e_variant"
  printf "   %-38s %s\n" "camera_resolution" "$(camera_resolution_label)"
  if mapping_vehicle_profile_applicable; then
    printf "   %-38s %s\n" "vehicle_speed_profile" "$(vehicle_profile_label)"
  fi
}

camera_resolution_label() {
  printf '%s x %s @ %s fps' "$SENSOR_IMAGE_WIDTH" "$SENSOR_IMAGE_HEIGHT" "$SENSOR_IMAGE_FPS"
}

choose_camera_resolution_interactive() {
  local res_cursor=0
  local fps_cursor=0
  local n_res="${#CAMERA_RES_LABELS[@]}"
  local key
  local idx
  local fps_arr
  local n_fps
  local phase="res"  # res | fps

  # 現在値にカーソルを合わせる
  for idx in "${!CAMERA_RES_WIDTHS[@]}"; do
    if [[ "${CAMERA_RES_WIDTHS[$idx]}" == "$SENSOR_IMAGE_WIDTH" \
       && "${CAMERA_RES_HEIGHTS[$idx]}" == "$SENSOR_IMAGE_HEIGHT" ]]; then
      res_cursor="$idx"
      break
    fi
  done

  while true; do
    if [[ "$phase" == "res" ]]; then
      # --- 解像度選択画面 ---
      tui_clear_screen
      echo "カメラ解像度の選択"
      echo ""
      echo "現在: ${SENSOR_IMAGE_WIDTH}x${SENSOR_IMAGE_HEIGHT} @ ${SENSOR_IMAGE_FPS} fps"
      echo ""
      for idx in "${!CAMERA_RES_LABELS[@]}"; do
        local marker=" "
        if [[ "$idx" -eq "$res_cursor" ]]; then marker=">"; fi
        printf " %s  %s\n" "$marker" "${CAMERA_RES_LABELS[$idx]}"
      done
      echo ""
      echo "j/k または ↑↓: 移動  Enter: FPS選択へ  q: キャンセル"

      key="$(tui_read_key)"
      case "$key" in
        j|$'\033[B') res_cursor=$(( (res_cursor + 1) % n_res )) ;;
        k|$'\033[A') res_cursor=$(( (res_cursor + n_res - 1) % n_res )) ;;
        ""|$'\n'|$'\r')
          phase="fps"
          # 選択解像度の FPS リストを決定
          read -r -a fps_arr <<< "${CAMERA_RES_FPS_LIST[$res_cursor]}"
          n_fps="${#fps_arr[@]}"
          # 現在の FPS に最も近い候補にカーソルを合わせる
          fps_cursor=$(( n_fps - 1 ))
          local cur_fps_int="${SENSOR_IMAGE_FPS%%.*}"
          for idx in "${!fps_arr[@]}"; do
            if [[ "${fps_arr[$idx]}" == "$cur_fps_int" ]]; then
              fps_cursor="$idx"
              break
            fi
          done
          ;;
        q|Q)
          tui_clear_screen
          return 0
          ;;
      esac

    else
      # --- FPS 選択画面 ---
      read -r -a fps_arr <<< "${CAMERA_RES_FPS_LIST[$res_cursor]}"
      n_fps="${#fps_arr[@]}"

      tui_clear_screen
      echo "FPSの選択  [解像度: ${CAMERA_RES_WIDTHS[$res_cursor]}x${CAMERA_RES_HEIGHTS[$res_cursor]}]"
      echo ""
      echo "現在: ${SENSOR_IMAGE_WIDTH}x${SENSOR_IMAGE_HEIGHT} @ ${SENSOR_IMAGE_FPS} fps"
      echo ""
      for idx in "${!fps_arr[@]}"; do
        local marker=" "
        if [[ "$idx" -eq "$fps_cursor" ]]; then marker=">"; fi
        printf " %s  %s fps\n" "$marker" "${fps_arr[$idx]}"
      done
      echo ""
      echo "j/k または ↑↓: 移動  Enter: 決定  b: 解像度選択に戻る  q: キャンセル"

      key="$(tui_read_key)"
      case "$key" in
        j|$'\033[B') fps_cursor=$(( (fps_cursor + 1) % n_fps )) ;;
        k|$'\033[A') fps_cursor=$(( (fps_cursor + n_fps - 1) % n_fps )) ;;
        b|B) phase="res" ;;
        ""|$'\n'|$'\r')
          SENSOR_IMAGE_WIDTH="${CAMERA_RES_WIDTHS[$res_cursor]}"
          SENSOR_IMAGE_HEIGHT="${CAMERA_RES_HEIGHTS[$res_cursor]}"
          SENSOR_IMAGE_FPS="${fps_arr[$fps_cursor]}.0"
          # プリセットの EXTRA_ARGS に反映
          set_extra_arg_value "image_width"  "$SENSOR_IMAGE_WIDTH"
          set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
          set_extra_arg_value "image_fps"    "$SENSOR_IMAGE_FPS"
          tui_clear_screen
          return 0
          ;;
        q|Q)
          tui_clear_screen
          return 0
          ;;
      esac
    fi
  done
}

warn_if_mode_incomplete() {
  local normalized_e2e_variant

  if [[ "$MODE" == "production" || "$MODE" == "base" || "$MODE" == "localization_eval" ]]; then
    if [[ -z "$(current_map_dir)" ]]; then
      echo "Warning: ${MODE} mode expects map_dir to be set." >&2
    fi
  fi

  if [[ "$(get_arg use_e2e)" == "true" ]]; then
    local e2e_variant
    e2e_variant="$(current_e2e_variant)"
    normalized_e2e_variant="$(normalize_e2e_variant "$e2e_variant")"

    case "$normalized_e2e_variant" in
      camera|camera_control|camera_e2e|camera_trajectory|camera_traj)
        if [[ "$(get_arg use_camera)" != "true" ]]; then
          echo "Warning: e2e_variant=${e2e_variant} expects use_camera=true." >&2
        fi
        ;;
      lidar|lidar_control|lidar_e2e|lidar_trajectory|lidar_traj|magp_rl_trajectory)
        if [[ "$(get_arg use_lidar)" != "true" ]]; then
          echo "Warning: e2e_variant=${e2e_variant} expects use_lidar=true." >&2
        fi
        ;;
    esac

    if [[ "$(get_arg use_magp_rl_trajectory)" == "true" ]]; then
      echo "Warning: use_e2e=true and use_magp_rl_trajectory=true may launch duplicate E2E pipelines." >&2
    fi
  fi

  if [[ "$(get_arg use_drive_mode_manager)" == "true" ]]; then
    if [[ "$(get_arg use_section_localizer)" != "true" ]]; then
      echo "Warning: use_drive_mode_manager=true but use_section_localizer is false." >&2
    fi
    if [[ "$(get_arg use_perception)" != "true" ]]; then
      echo "Warning: use_drive_mode_manager=true but use_perception is false." >&2
    fi
  fi
}

normalize_e2e_variant() {
  printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]'
}

validate_e2e_configuration() {
  local e2e_variant
  local normalized_e2e_variant

  e2e_variant="$(current_e2e_variant)"
  normalized_e2e_variant="$(normalize_e2e_variant "$e2e_variant")"

  if [[ "$(get_arg use_e2e)" == "true" ]]; then
    if [[ -z "$e2e_variant" ]]; then
      echo "Error: use_e2e=true requires e2e_variant. Use --e2e camera|lidar|camera_trajectory|lidar_trajectory." >&2
      exit 1
    fi

    case "$normalized_e2e_variant" in
      camera|camera_control|camera_e2e|lidar|lidar_control|lidar_e2e|camera_trajectory|camera_traj|lidar_trajectory|lidar_traj|magp_rl_trajectory)
        ;;
      *)
        echo "Error: unsupported e2e_variant=${e2e_variant}. Choose camera|lidar|camera_trajectory|lidar_trajectory." >&2
        exit 1
        ;;
    esac
    return 0
  fi

  if [[ -n "$e2e_variant" ]]; then
    echo "Error: e2e_variant=${e2e_variant} was set while use_e2e=false. Use --e2e ${e2e_variant} or set use_e2e=true." >&2
    exit 1
  fi

  return 0
}

toggle_interactive_checkbox() {
  tui_checkbox_menu \
    "Mode: $MODE" \
    BOOL_KEYS \
    get_arg \
    toggle_bool_key \
    render_launch_extra_interactive \
    choose_bag_manager_interactive \
    prompt_set_arg_interactive \
    "bag manager" \
    "set value" \
    choose_map_dir_interactive \
    "map" \
    choose_vehicle_profile_interactive \
    "vehicle speed"
}

toggle_interactive() {
  if [[ "$LEGACY_INTERACTIVE" == "true" || ! -t 0 ]]; then
    toggle_interactive_legacy
  else
    toggle_interactive_checkbox
  fi
}
