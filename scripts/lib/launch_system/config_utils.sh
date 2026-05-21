#!/bin/bash

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

apply_mode() {
  clear_extra_arg_value "vehicle_param"

  case "$1" in
    rule_base)
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
      ARG_use_planning="true"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="true"
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
      ARG_use_planning="false"
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
      ARG_use_planning="false"
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
      set_extra_arg_value "vehicle_param" "$VEHICLE_MAPPING_PARAM_PATH"
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
      ARG_use_planning="false"
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
      ARG_use_planning="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="true"
      ARG_publish_map="true"
      ARG_map_server_use_sim_time="true"
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
      ARG_use_planning="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="true"
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
      ARG_use_planning="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_sim_time="true"
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

current_vehicle_param() {
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      vehicle_param:=*|vehicle_param=*)
        echo "${arg#*=}"
        return
        ;;
    esac
  done

  echo ""
}

mapping_vehicle_profile_applicable() {
  [[ "$(get_arg use_vehicle)" == "true" ]] || return 1

  case "$MODE" in
    sensor_data_recording|mapping|vslam_map)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
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
