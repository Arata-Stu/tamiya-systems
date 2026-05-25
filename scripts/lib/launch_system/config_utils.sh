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
    race_e2e)
      apply_mode production
      ARG_use_hd_map="true"
      ARG_use_e2e="true"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_hd_map_section_localizer="true"
      ARG_publish_map="true"
      set_extra_arg_value "e2e_variant" "camera"
      ;;
    race)
      apply_mode production
      ARG_use_planning="true"
      ARG_use_hd_map="true"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="true"
      ARG_use_speed_controller="true"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="true"
      ;;
    race_pp)
      apply_mode race
      ARG_use_pure_pursuit="true"
      ARG_use_map_controller="false"
      ;;

    e2e_backup)
      apply_mode production
      ARG_vslam="false"
      ARG_localization="false"
      ARG_use_lidar="false"
      ARG_use_camera="true"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_planning="false"
      ARG_use_hd_map="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="true"
      ARG_publish_map="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[2]}"
      set_extra_arg_value "e2e_variant" "camera"
      ;;

    offline_eval)
      apply_mode localization_eval
      ARG_use_lidar="false"
      ARG_use_camera="false"
      ARG_use_hd_map="true"
      ARG_use_planning="true"
      ARG_use_hd_map_section_localizer="true"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      set_extra_arg_value "localize_on_startup" "true"
      ;;

    production)
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
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="false"
      ARG_publish_map="true"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="true"
      ARG_publish_localization_tf="true"
      ARG_use_section_localizer="true"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
      ARG_enable_localization_and_mapping="true"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[0]}"
      ;;
    record_mapping)
      ARG_record="true"
      ARG_use_vehicle="true"
      ARG_vslam="false"
      ARG_localization="false"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_planning="false"
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
      ARG_enable_localization_and_mapping="false"
      ARG_bag_manager_param="${BAG_MANAGER_PATHS[1]}"
      # Map recording keeps the stereo-gray left/right sensors at
      # ${SENSOR_IMAGE_WIDTH}x${SENSOR_IMAGE_HEIGHT}x${SENSOR_IMAGE_FPS%%.*}
      # and records camera, LiDAR, TF, and command topics through the mapping bag preset.
      set_extra_arg_value "image_width" "$SENSOR_IMAGE_WIDTH"
      set_extra_arg_value "image_height" "$SENSOR_IMAGE_HEIGHT"
      set_extra_arg_value "image_fps" "$SENSOR_IMAGE_FPS"
      set_extra_arg_value "vehicle_param" "$VEHICLE_MAPPING_PARAM_PATH"
      ;;

    identification)
      ARG_record="true"
      ARG_use_vehicle="true"
      ARG_vslam="true"
      ARG_localization="false"
      ARG_use_lidar="true"
      ARG_use_camera="true"
      ARG_use_ftg="false"
      ARG_use_emergency="false"
      ARG_use_perception="false"
      ARG_use_perception_classifier="false"
      ARG_use_planning="false"
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="false"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
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
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="true"
      ARG_publish_map="true"
      ARG_map_server_use_sim_time="true"
      ARG_use_localization_manager="true"
      ARG_publish_localization_tf="true"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
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
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="true"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
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
      ARG_use_hd_map="false"
      ARG_use_magp_rl_trajectory="false"
      ARG_magp_rl_run_pure_pursuit="false"
      ARG_use_pure_pursuit="false"
      ARG_use_map_controller="false"
      ARG_use_control_filter="false"
      ARG_use_speed_controller="false"
      ARG_use_e2e="false"
      ARG_use_sim_time="true"
      ARG_publish_map="false"
      ARG_map_server_use_sim_time="false"
      ARG_use_localization_manager="false"
      ARG_publish_localization_tf="false"
      ARG_use_section_localizer="false"
      ARG_use_hd_map_section_localizer="false"
      ARG_section_localizer_debug_mode="false"
      ARG_use_drive_mode_manager="false"
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

current_control_filter_param() {
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      control_filter_param:=*|control_filter_param=*)
        echo "${arg#*=}"
        return
        ;;
    esac
  done

  echo ""
}

current_speed_controller_param() {
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      speed_controller_param:=*|speed_controller_param=*)
        echo "${arg#*=}"
        return
        ;;
    esac
  done

  echo ""
}

current_e2e_variant() {
  local arg

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      e2e_variant:=*|e2e_variant=*)
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
    record_mapping|map_recording|map_record|mapping_record|record_sensors|sensor_data_recording|sensor_recording|record_mapping_debug|record_dataset|mapping|vslam_map)
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
  local control_filter_yaml
  local speed_controller_yaml
  local map_name

  if [[ "$MODE" != "production" && "$MODE" != "base" && "$MODE" != "race" && "$MODE" != "race_map" && "$MODE" != "race_map_controller" && "$MODE" != "race_pp" && "$MODE" != "race_pure_pursuit" && "$MODE" != "hd_map_eval" && "$MODE" != "hd_map_debug" && "$MODE" != "offline_eval" && "$MODE" != "bag_eval" && "$MODE" != "offline_map_eval" && "$(get_arg use_section_localizer)" != "true" && "$(get_arg use_hd_map_section_localizer)" != "true" && "$(get_arg use_control_filter)" != "true" && "$(get_arg use_speed_controller)" != "true" ]]; then
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

  control_filter_yaml="${map_dir%/}/control_filter.param.yaml"
  if [[ -f "$control_filter_yaml" && -z "$(current_control_filter_param)" ]]; then
    set_extra_arg_value "control_filter_param" "$control_filter_yaml"
  fi

  if [[ "$(get_arg use_speed_controller)" == "true" && -z "$(current_speed_controller_param)" ]]; then
    map_name="$(basename "${map_dir%/}")"
    for speed_controller_yaml in \
      "${map_dir%/}/speed_controller_feedforward.param.yaml" \
      "${map_dir%/}/speed_controller.param.yaml" \
      "${map_dir%/}/${map_name}_speed_controller_feedforward.param.yaml"; do
      if [[ -f "$speed_controller_yaml" ]]; then
        set_extra_arg_value "speed_controller_param" "$speed_controller_yaml"
        break
      fi
    done
  fi
}
