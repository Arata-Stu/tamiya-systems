#!/bin/bash

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
    record|use_vehicle|vslam|localization|use_lidar|use_camera|use_ftg|use_emergency|use_perception|use_perception_classifier|use_planning|use_magp_rl_trajectory|magp_rl_run_pure_pursuit|use_pure_pursuit|use_map_controller|use_control_filter|use_e2e|use_sim_time|publish_map|map_server_use_sim_time|use_localization_manager|publish_localization_tf|use_section_localizer|section_localizer_debug_mode|enable_localization_and_mapping)
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
