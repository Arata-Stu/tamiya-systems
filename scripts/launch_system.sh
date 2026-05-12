#!/bin/bash

set -eo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  use_magp_rl_trajectory
  magp_rl_run_pure_pursuit
  use_pure_pursuit
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
ARG_use_magp_rl_trajectory="false"
ARG_magp_rl_run_pure_pursuit="false"
ARG_use_pure_pursuit="false"
ARG_use_sim_time="false"
ARG_publish_map="false"
ARG_map_server_use_sim_time="false"
ARG_use_localization_manager="false"
ARG_publish_localization_tf="false"
ARG_use_section_localizer="false"
ARG_section_localizer_debug_mode="false"
ARG_enable_localization_and_mapping="false"
ARG_bag_manager_param='$(find-pkg-share system_launch)/config/tools/bag_manager.param.yaml'

BAG_MANAGER_PRESETS=(
  default
  mapping
  e2e
)

BAG_MANAGER_PATHS=(
  '$(find-pkg-share system_launch)/config/tools/bag_manager.param.yaml'
  '$(find-pkg-share system_launch)/config/tools/bag_manager_mapping.param.yaml'
  '$(find-pkg-share system_launch)/config/tools/bag_manager_e2e.param.yaml'
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
  base                    Alias of production
  mapping                 Alias of sensor_data_recording
  vslam_map               Alias of sensor_data_recording

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
  ${SCRIPT_NAME} sensor_data_recording -i
  ${SCRIPT_NAME} sensor_data_recording --bag-manager mapping
  ${SCRIPT_NAME} sensor_data_recording --set record=false --dry-run
  ${SCRIPT_NAME} production --set map_dir=/map/mybag/mycourse
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
    record|use_vehicle|vslam|localization|use_lidar|use_camera|use_ftg|use_emergency|use_magp_rl_trajectory|magp_rl_run_pure_pursuit|use_pure_pursuit|use_sim_time|publish_map|map_server_use_sim_time|use_localization_manager|publish_localization_tf|use_section_localizer|section_localizer_debug_mode|enable_localization_and_mapping)
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
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
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
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[1]}"
      # Initial mapping runs record only lidar + camera sensor data at 1280x720x30.
      set_extra_arg_value "image_width" "1280"
      set_extra_arg_value "image_height" "720"
      set_extra_arg_value "image_fps" "30.0"
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
  local answer

  echo "Mode:"
  echo "  1) production"
  echo "  2) sensor_data_recording"
  read -r -p "Select mode [${MODE}]: " answer

  case "${answer:-$MODE}" in
    1|production|base)
      MODE="production"
      ;;
    2|sensor_data_recording|mapping|vslam_map)
      MODE="sensor_data_recording"
      ;;
    *)
      echo "Invalid mode: $answer" >&2
      exit 1
      ;;
  esac
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
}

apply_mode_derived_args() {
  local map_dir
  local section_csv
  local gate_csv

  if [[ "$MODE" != "production" && "$MODE" != "base" ]]; then
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
  if [[ "$MODE" == "production" || "$MODE" == "base" ]]; then
    if [[ -z "$(current_map_dir)" ]]; then
      echo "Warning: production mode expects map_dir to be set." >&2
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
    production|base|sensor_data_recording|mapping|vslam_map)
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
  toggle_interactive
fi

apply_mode_derived_args
warn_if_mode_incomplete

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
  "use_magp_rl_trajectory:=${ARG_use_magp_rl_trajectory}" \
  "magp_rl_run_pure_pursuit:=${ARG_magp_rl_run_pure_pursuit}" \
  "use_pure_pursuit:=${ARG_use_pure_pursuit}" \
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
