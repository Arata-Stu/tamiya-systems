#!/bin/bash

set -eo pipefail

SCRIPT_NAME="$(basename "$0")"

BOOL_KEYS=(
  use_section_localizer
  localization
  vslam
  use_camera
  use_lidar
  use_ftg
  use_emergency
  record
  use_magp_rl_trajectory
  use_pure_pursuit
  use_localization_manager
  enable_localization_and_mapping
)

ARG_use_section_localizer="false"
ARG_localization="false"
ARG_vslam="true"
ARG_use_camera="true"
ARG_use_lidar="true"
ARG_use_ftg="false"
ARG_use_emergency="false"
ARG_record="false"
ARG_use_magp_rl_trajectory="false"
ARG_use_pure_pursuit="false"
ARG_use_localization_manager="true"
ARG_enable_localization_and_mapping="true"
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

MODE="base"
INTERACTIVE="false"
DRY_RUN="false"
EXTRA_ARGS=()
OVERRIDES=()
ORIGINAL_ARGC=$#

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [mode] [options] [-- extra_ros2_launch_args...]

Modes:
  base       Current simple system launch preset
  mapping    Mapping/recording preset

Options:
  -i, --interactive       Toggle launch arguments interactively before running
  -n, --dry-run           Print the command without running it
  --bag-manager NAME      Select bag manager yaml: default|mapping|e2e|PATH
  --set KEY=VALUE         Override an argument (also accepts KEY:=VALUE)
  -h, --help              Show this help

Examples:
  ${SCRIPT_NAME} mapping
  ${SCRIPT_NAME} mapping -i
  ${SCRIPT_NAME} mapping --bag-manager mapping
  ${SCRIPT_NAME} base --bag-manager e2e
  ${SCRIPT_NAME} mapping --set record=false --dry-run
  ${SCRIPT_NAME} -i -- map_yaml_path:=/map/course/map.yaml
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
    use_section_localizer|localization|vslam|use_camera|use_lidar|use_ftg|use_emergency|record|use_magp_rl_trajectory|use_pure_pursuit|use_localization_manager|enable_localization_and_mapping)
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
    base)
      ARG_use_section_localizer="false"
      ARG_localization="false"
      ARG_vslam="true"
      ARG_use_camera="true"
      ARG_use_lidar="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_record="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      ;;
    mapping)
      ARG_use_section_localizer="false"
      ARG_localization="false"
      ARG_vslam="true"
      ARG_use_camera="true"
      ARG_use_lidar="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_record="true"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[1]}"
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
  echo "  1) base"
  echo "  2) mapping"
  read -r -p "Select mode [${MODE}]: " answer

  case "${answer:-$MODE}" in
    1|base)
      MODE="base"
      ;;
    2|mapping)
      MODE="mapping"
      ;;
    *)
      echo "Invalid mode: $answer" >&2
      exit 1
      ;;
  esac
}

toggle_interactive() {
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
    echo ""
    echo "Enter numbers to toggle, 'b' to choose bag manager yaml, 's KEY=VALUE' to set a value, or Enter to run."
    read -r -p "> " answer

    [[ -z "$answer" ]] && break

    if [[ "$answer" == "b" || "$answer" == "B" ]]; then
      choose_bag_manager_interactive
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
    base|mapping)
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

if [[ "$INTERACTIVE" == "true" && -t 0 && "$MODE" == "base" ]]; then
  choose_mode_interactive
fi

apply_mode "$MODE"

for override in "${OVERRIDES[@]}"; do
  set_arg "$override"
done

if [[ "$INTERACTIVE" == "true" ]]; then
  toggle_interactive
fi

echo "Mode: $MODE"
echo "Command:"
build_command

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

source_setup_if_available
exec ros2 launch system_launch system.launch.xml \
  "use_section_localizer:=${ARG_use_section_localizer}" \
  "localization:=${ARG_localization}" \
  "vslam:=${ARG_vslam}" \
  "use_camera:=${ARG_use_camera}" \
  "use_lidar:=${ARG_use_lidar}" \
  "use_ftg:=${ARG_use_ftg}" \
  "use_emergency:=${ARG_use_emergency}" \
  "record:=${ARG_record}" \
  "use_magp_rl_trajectory:=${ARG_use_magp_rl_trajectory}" \
  "use_pure_pursuit:=${ARG_use_pure_pursuit}" \
  "use_localization_manager:=${ARG_use_localization_manager}" \
  "enable_localization_and_mapping:=${ARG_enable_localization_and_mapping}" \
  "bag_manager_param:=${ARG_bag_manager_param}" \
  "${EXTRA_ARGS[@]}"
