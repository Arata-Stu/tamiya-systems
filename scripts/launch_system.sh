#!/bin/bash

set -eo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEM_LAUNCH_SOURCE_SHARE="$REPO_ROOT/ros2_ws/src/launch/system_launch"
SETUP_SOURCED="false"

# ==============================================================================
# カメラ解像度設定 (全プリセット共通)
# 解像度を変更する場合はここを書き換えるだけで全プリセットに反映されます。
# または起動時に-iフラグでインタラクティブに選択できます。
#
# RealSense D435 stereo gray (infra) のサポート解像度例:
#   424x240  (デフォルト, 最大 90fps)
#   640x480  (最大 90fps)
#   848x480  (最大 90fps)
#   1280x720 (最大 30fps)
# ==============================================================================
SENSOR_IMAGE_WIDTH="424"
SENSOR_IMAGE_HEIGHT="240"
SENSOR_IMAGE_FPS="90.0"

# --- 解像度プリセット定義 ---
# CAMERA_RES_LABELS  : 表示用ラベル
# CAMERA_RES_WIDTHS  : W
# CAMERA_RES_HEIGHTS : H
# CAMERA_RES_FPS_LIST: 利用可能な FPS選択肢 (スペース区切り)
CAMERA_RES_LABELS=(
  "1280x720  [ 6/15/30      fps] Depth最高解像度(負荷大, 90fps不可)"
  "848x480   [ 6/15/30/60/90fps] D435i推奨・ネイティブバランス"
  "640x480   [ 6/15/30/60/90fps] 4:3設定, SLAM等で多用"
  "640x360   [ 6/15/30/60/90fps] 16:9軽量設定"
  "480x270   [ 6/15/30/60/90fps] 低解像度"
  "424x240   [ 6/15/30/60/90fps] 最小クラス, 計算資源の節約用"
)
CAMERA_RES_WIDTHS=(1280 848 640 640 480 424)
CAMERA_RES_HEIGHTS=(720 480 480 360 270 240)
CAMERA_RES_FPS_LIST=(
  "6 15 30"
  "6 15 30 60 90"
  "6 15 30 60 90"
  "6 15 30 60 90"
  "6 15 30 60 90"
  "6 15 30 60 90"
)

# shellcheck source=scripts/common/tui.sh
source "$SCRIPT_DIR/common/tui.sh"

BOOL_KEYS=(
  record
  use_vehicle
  vslam
  localization
  use_lidar
  use_camera
  use_ftg
  use_emergency
  use_perception
  use_perception_classifier
  use_magp_rl_trajectory
  magp_rl_run_pure_pursuit
  use_pure_pursuit
  use_map_controller
  use_sim_time
  publish_map
  map_server_use_sim_time
  use_localization_manager
  publish_localization_tf
  use_section_localizer
  section_localizer_debug_mode
  enable_localization_and_mapping
)

ARG_record="false"
ARG_use_vehicle="true"
ARG_vslam="false"
ARG_localization="false"
ARG_use_lidar="false"
ARG_use_camera="false"
ARG_use_ftg="false"
ARG_use_emergency="false"
ARG_use_perception="false"
ARG_use_perception_classifier="false"
ARG_use_magp_rl_trajectory="false"
ARG_magp_rl_run_pure_pursuit="false"
ARG_use_pure_pursuit="false"
ARG_use_map_controller="false"
ARG_use_sim_time="false"
ARG_publish_map="false"
ARG_map_server_use_sim_time="false"
ARG_use_localization_manager="false"
ARG_publish_localization_tf="false"
ARG_use_section_localizer="false"
ARG_section_localizer_debug_mode="false"
ARG_enable_localization_and_mapping="false"

if [[ -d "$SYSTEM_LAUNCH_SOURCE_SHARE" ]]; then
  BAG_MANAGER_DEFAULT_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager.param.yaml"
  BAG_MANAGER_MAPPING_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager_mapping.param.yaml"
  BAG_MANAGER_E2E_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager_e2e.param.yaml"
else
  BAG_MANAGER_DEFAULT_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager.param.yaml'
  BAG_MANAGER_MAPPING_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager_mapping.param.yaml'
  BAG_MANAGER_E2E_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager_e2e.param.yaml'
fi

ARG_bag_manager_param="$BAG_MANAGER_DEFAULT_PATH"

BAG_MANAGER_PRESETS=(
  default
  mapping
  e2e
)

BAG_MANAGER_PATHS=(
  "$BAG_MANAGER_DEFAULT_PATH"
  "$BAG_MANAGER_MAPPING_PATH"
  "$BAG_MANAGER_E2E_PATH"
)

MODE="production"
INTERACTIVE="false"
LEGACY_INTERACTIVE="false"
DRY_RUN="false"
EXTRA_ARGS=()
OVERRIDES=()
ORIGINAL_ARGC=$#
MAP_SEARCH_ROOT="/map"
MAP_CANDIDATES=()

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
  ${SCRIPT_NAME} identification
  ${SCRIPT_NAME} -i -- map_dir:=/map/mybag/mycourse
EOF
}

normalize_bool() {
  case "$1" in
    true|True|TRUE|1|yes|Yes|YES|y|Y|on|ON)
      echo "true"
      ;;
    false|False|FALSE|0|no|No|NO|n|N|off|OFF)
      echo "false"
      ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

set_arg() {
  local assignment="$1"
  local key
  local value
  local var_name

  assignment="${assignment/:=/=}"
  if [[ "$assignment" != *=* ]]; then
    echo "Expected KEY=VALUE for --set, got: $assignment" >&2
    exit 1
  fi

  key="${assignment%%=*}"
  value="${assignment#*=}"

  if [[ -z "$key" || -z "$value" ]]; then
    echo "Expected non-empty KEY=VALUE for --set, got: $assignment" >&2
    exit 1
  fi

  case "$key" in
    record|use_vehicle|vslam|localization|use_lidar|use_camera|use_ftg|use_emergency|use_perception|use_perception_classifier|use_magp_rl_trajectory|magp_rl_run_pure_pursuit|use_pure_pursuit|use_map_controller|use_sim_time|publish_map|map_server_use_sim_time|use_localization_manager|publish_localization_tf|use_section_localizer|section_localizer_debug_mode|enable_localization_and_mapping)
      value="$(normalize_bool "$value")"
      var_name="ARG_${key}"
      printf -v "$var_name" '%s' "$value"
      ;;
    bag_manager_param)
      set_bag_manager_preset "$value"
      ;;
    *)
      EXTRA_ARGS+=("${key}:=${value}")
      ;;
  esac
}

get_arg() {
  local var_name="ARG_$1"
  printf '%s\n' "${!var_name}"
}

apply_mode() {
  case "$1" in
    production|base)
      ARG_record="false"
      ARG_use_vehicle="true"
      ARG_vslam="true"
      ARG_localization="true"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="true"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="true"
      ARG_publish_localization_tf="true"
      ARG_use_section_localizer="true"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="true"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      ;;
    sensor_data_recording|mapping|vslam_map)
      ARG_record="true"
      ARG_use_vehicle="true"
      ARG_vslam="false"
      ARG_localization="false"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="true"
      ARG_use_perception_classifier="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[1]}"
      # Initial mapping runs keep the stereo-gray left/right sensors at
      # ${SENSOR_IMAGE_WIDTH}x${SENSOR_IMAGE_HEIGHT}x${SENSOR_IMAGE_FPS%%.*}
      # and enable crop perception so recorded bags already contain /perception/crop/*.
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      ;;
    identification|map_lookup_recording|map_lookup)
      ARG_record="true"
      ARG_use_vehicle="true"
      ARG_vslam="true"
      ARG_localization="false"
      ARG_use_lidar="false"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="true"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      ;;
    localization_eval)
      ARG_record="false"
      ARG_use_vehicle="false"
      ARG_vslam="true"
      ARG_localization="true"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="true"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="true"
      ARG_publish_localization_tf="true"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      ;;
    perception_eval)
      ARG_record="false"
      ARG_use_vehicle="false"
      ARG_vslam="false"
      ARG_localization="false"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="true"
      ARG_use_perception_classifier="true"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      ;;
    vslam_eval)
      ARG_record="false"
      ARG_use_vehicle="false"
      ARG_vslam="true"
      ARG_localization="false"
      ARG_use_lidar="false"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="true"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      ;;
    *)
      echo "Unknown mode: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

set_bag_manager_preset() {
  local choice="$1"
  local idx

  for idx in "${!BAG_MANAGER_PRESETS[@]}"; do
    if [[ "$choice" == "${BAG_MANAGER_PRESETS[$idx]}" ]]; then
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[$idx]}"
      return
    fi
  done

  ARG_bag_manager_param="$choice"
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

current_map_dir() {
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      map_dir:=*|map_dir=*)
        echo "${arg#*=}"
        return
        ;;
    esac
  done

  echo ""
}

set_extra_arg_value() {
  local key="$1"
  local value="$2"
  local idx

  for idx in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$idx]}" in
      "${key}:="*|"${key}="*)
        EXTRA_ARGS[$idx]="${key}:=${value}"
        return
        ;;
    esac
  done

  EXTRA_ARGS+=("${key}:=${value}")
}

clear_extra_arg_value() {
  local key="$1"
  local updated=()
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      "${key}:="*|"${key}="*)
        ;;
      *)
        updated+=("$arg")
        ;;
    esac
  done

  EXTRA_ARGS=("${updated[@]}")
}

discover_map_candidates() {
  local metadata
  local map_dir
  local -a discovered=()

  MAP_CANDIDATES=()
  if [[ ! -d "$MAP_SEARCH_ROOT" ]]; then
    return 1
  fi

  while IFS= read -r metadata; do
    [[ -z "$metadata" ]] && continue
    map_dir="$(dirname "$metadata")"
    discovered+=("$map_dir")
  done < <(find "$MAP_SEARCH_ROOT" -type f -name '*.yaml' ! -path '*/cuvslam_map/*' 2>/dev/null)

  if [[ "${#discovered[@]}" -eq 0 ]]; then
    return 1
  fi

  while IFS= read -r map_dir; do
    [[ -n "$map_dir" ]] && MAP_CANDIDATES+=("$map_dir")
  done < <(printf '%s\n' "${discovered[@]}" | sort -u)

  [[ "${#MAP_CANDIDATES[@]}" -gt 0 ]]
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
  )
  local -a MODE_VALUES=(
    "production"
    "sensor_data_recording"
    "identification"
    "localization_eval"
    "perception_eval"
    "vslam_eval"
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
    echo ""
    echo "Enter numbers to toggle, 'b' for bag manager, 'm' for map dir, 's KEY=VALUE' to set a value, or Enter to run."
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

prompt_set_arg_interactive() {
  local answer

  tui_clear_screen
  echo "Set launch argument"
  echo ""
  echo "Examples:"
  echo "  use_lidar=false"
  echo "  map_yaml_path:=/map/course/map.yaml"
  echo ""
  read -r -p "KEY=VALUE: " answer
  [[ -n "$answer" ]] && set_arg "$answer"
}

render_launch_extra_interactive() {
  printf "   %-38s %s (%s)\n" "bag_manager_param" "$ARG_bag_manager_param" "$(bag_manager_label)"
  printf "   %-38s %s\n" "map_dir" "$(map_dir_label)"
  printf "   %-38s %s\n" "camera_resolution" "$(camera_resolution_label)"
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

apply_mode_derived_args() {
  local map_dir
  local section_csv
  local gate_csv

  if [[ "$MODE" != "production" && "$MODE" != "base" && "$(get_arg use_section_localizer)" != "true" ]]; then
    return
  fi

  map_dir="$(current_map_dir)"
  if [[ -z "$map_dir" ]]; then
    return
  fi

  section_csv="${map_dir%/}/sections_pixels.csv"
  gate_csv="${map_dir%/}/sections_pixels_gates.csv"

  if [[ -f "$section_csv" ]]; then
    set_extra_arg_value "section_definition_path" "$section_csv"
  fi

  if [[ -f "$gate_csv" ]]; then
    set_extra_arg_value "gate_definition_path" "$gate_csv"
  fi
}

warn_if_mode_incomplete() {
  if [[ "$MODE" == "production" || "$MODE" == "base" || "$MODE" == "localization_eval" ]]; then
    if [[ -z "$(current_map_dir)" ]]; then
      echo "Warning: ${MODE} mode expects map_dir to be set." >&2
    fi
  fi
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
    "map"
}

toggle_interactive() {
  if [[ "$LEGACY_INTERACTIVE" == "true" || ! -t 0 ]]; then
    toggle_interactive_legacy
  else
    toggle_interactive_checkbox
  fi
}

build_command() {
  local cmd=("ros2" "launch" "system_launch" "system.launch.xml")
  local key

  for key in "${BOOL_KEYS[@]}"; do
    cmd+=("${key}:=$(get_arg "$key")")
  done
  cmd+=("bag_manager_param:=${ARG_bag_manager_param}")

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

source_setup_if_available() {
  if [[ "$SETUP_SOURCED" == "true" ]]; then
    return 0
  fi

  local nounset_was_enabled=0
  if [[ $- == *u* ]]; then
    nounset_was_enabled=1
    set +u
  fi

  if [[ -n "${SYSTEM_LAUNCH_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SYSTEM_LAUNCH_SETUP"
  elif [[ -f "/workspaces/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "/workspaces/install/setup.bash"
  elif [[ -f "install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "install/setup.bash"
  fi

  if [[ ${nounset_was_enabled} -eq 1 ]]; then
    set -u
  fi

  SETUP_SOURCED="true"
}

resolve_package_share_dir() {
  local package_name="$1"
  local share_dir=""
  local prefix=""

  if [[ "$package_name" == "system_launch" && -d "$SYSTEM_LAUNCH_SOURCE_SHARE" ]]; then
    printf '%s\n' "$SYSTEM_LAUNCH_SOURCE_SHARE"
    return 0
  fi

  source_setup_if_available

  if ! command -v ros2 >/dev/null 2>&1; then
    return 1
  fi

  share_dir="$(ros2 pkg prefix --share "$package_name" 2>/dev/null || true)"
  if [[ -n "$share_dir" ]]; then
    printf '%s\n' "$share_dir"
    return 0
  fi

  prefix="$(ros2 pkg prefix "$package_name" 2>/dev/null || true)"
  if [[ -n "$prefix" ]]; then
    printf '%s\n' "${prefix%/}/share/$package_name"
    return 0
  fi

  return 1
}

resolve_find_pkg_share_value() {
  local value="$1"
  local token
  local package_name
  local share_dir

  while [[ "$value" =~ \$\(find-pkg-share[[:space:]]+([[:alnum:]_]+)\) ]]; do
    token="${BASH_REMATCH[0]}"
    package_name="${BASH_REMATCH[1]}"

    if ! share_dir="$(resolve_package_share_dir "$package_name")"; then
      echo "Failed to resolve package share for: $package_name" >&2
      exit 1
    fi

    value="${value/$token/$share_dir}"
  done

  printf '%s\n' "$value"
}

resolve_launch_arg_assignment() {
  local arg="$1"
  local key
  local value

  case "$arg" in
    *:=*)
      key="${arg%%:=*}"
      value="${arg#*:=}"
      printf '%s:=%s\n' "$key" "$(resolve_find_pkg_share_value "$value")"
      ;;
    *=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      printf '%s=%s\n' "$key" "$(resolve_find_pkg_share_value "$value")"
      ;;
    *)
      printf '%s\n' "$(resolve_find_pkg_share_value "$arg")"
      ;;
  esac
}

resolve_launch_paths() {
  local resolved_extra_args=()
  local arg

  ARG_bag_manager_param="$(resolve_find_pkg_share_value "$ARG_bag_manager_param")"

  for arg in "${EXTRA_ARGS[@]}"; do
    resolved_extra_args+=("$(resolve_launch_arg_assignment "$arg")")
  done

  EXTRA_ARGS=("${resolved_extra_args[@]}")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -i|--interactive)
      INTERACTIVE="true"
      shift
      ;;
    --legacy-interactive)
      INTERACTIVE="true"
      LEGACY_INTERACTIVE="true"
      shift
      ;;
    -n|--dry-run)
      DRY_RUN="true"
      shift
      ;;
    --set)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --set" >&2
        exit 1
      fi
      OVERRIDES+=("$2")
      shift 2
      ;;
    --bag-manager)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --bag-manager" >&2
        exit 1
      fi
      OVERRIDES+=("bag_manager_param=$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    production|base|sensor_data_recording|mapping|vslam_map|identification|map_lookup_recording|map_lookup|localization_eval|perception_eval|vslam_eval)
      MODE="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$ORIGINAL_ARGC" -eq 0 && -t 0 ]]; then
  INTERACTIVE="true"
fi

if [[ "$INTERACTIVE" == "true" && -t 0 && "$MODE" == "production" ]]; then
  choose_mode_interactive
fi

apply_mode "$MODE"

for override in "${OVERRIDES[@]}"; do
  set_arg "$override"
done

if [[ "$INTERACTIVE" == "true" ]]; then
  # 解像度・ FPS をインタラクティブに選択
  choose_camera_resolution_interactive
  toggle_interactive
fi

apply_mode_derived_args
warn_if_mode_incomplete
resolve_launch_paths

echo "Mode: $MODE"
echo "Command:"
build_command

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

source_setup_if_available
exec ros2 launch system_launch system.launch.xml \
  "record:=${ARG_record}" \
  "use_vehicle:=${ARG_use_vehicle}" \
  "vslam:=${ARG_vslam}" \
  "localization:=${ARG_localization}" \
  "use_lidar:=${ARG_use_lidar}" \
  "use_camera:=${ARG_use_camera}" \
  "use_ftg:=${ARG_use_ftg}" \
  "use_emergency:=${ARG_use_emergency}" \
  "use_perception:=${ARG_use_perception}" \
  "use_perception_classifier:=${ARG_use_perception_classifier}" \
  "use_magp_rl_trajectory:=${ARG_use_magp_rl_trajectory}" \
  "magp_rl_run_pure_pursuit:=${ARG_magp_rl_run_pure_pursuit}" \
  "use_pure_pursuit:=${ARG_use_pure_pursuit}" \
  "use_map_controller:=${ARG_use_map_controller}" \
  "use_sim_time:=${ARG_use_sim_time}" \
  "publish_map:=${ARG_publish_map}" \
  "map_server_use_sim_time:=${ARG_map_server_use_sim_time}" \
  "use_localization_manager:=${ARG_use_localization_manager}" \
  "publish_localization_tf:=${ARG_publish_localization_tf}" \
  "use_section_localizer:=${ARG_use_section_localizer}" \
  "section_localizer_debug_mode:=${ARG_section_localizer_debug_mode}" \
  "enable_localization_and_mapping:=${ARG_enable_localization_and_mapping}" \
  "bag_manager_param:=${ARG_bag_manager_param}" \
  "${EXTRA_ARGS[@]}"
