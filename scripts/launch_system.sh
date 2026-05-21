#!/bin/bash


# --- Source Library Modules ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for lib in "${SCRIPT_DIR}/lib/launch_system/"*.sh; do
    source "${lib}"
done
# ------------------------------

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
  use_planning
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

if [[ -d "$SYSTEM_LAUNCH_SOURCE_SHARE" ]]; then
  BAG_MANAGER_DEFAULT_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager.param.yaml"
  BAG_MANAGER_MAPPING_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager_mapping.param.yaml"
  BAG_MANAGER_E2E_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/tools/bag_manager_e2e.param.yaml"
  VEHICLE_DEFAULT_PARAM_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/vehicle/jetracer_node.param.yaml"
  VEHICLE_MAPPING_PARAM_PATH="$SYSTEM_LAUNCH_SOURCE_SHARE/config/vehicle/jetracer_node_mapping.param.yaml"
else
  BAG_MANAGER_DEFAULT_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager.param.yaml'
  BAG_MANAGER_MAPPING_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager_mapping.param.yaml'
  BAG_MANAGER_E2E_PATH='$(find-pkg-share system_launch)/config/tools/bag_manager_e2e.param.yaml'
  VEHICLE_DEFAULT_PARAM_PATH='$(find-pkg-share system_launch)/config/vehicle/jetracer_node.param.yaml'
  VEHICLE_MAPPING_PARAM_PATH='$(find-pkg-share system_launch)/config/vehicle/jetracer_node_mapping.param.yaml'
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
  "use_planning:=${ARG_use_planning}" \
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
