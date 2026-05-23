#!/bin/bash

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [SESSION_NAME] [--mode record_mapping|map_build|offline_eval|race|identification|hd_map|e2e_backup|python|map|mapping|localization_eval|perception_eval|vslam_eval|simulator]
  $(basename "$0") [--session SESSION_NAME] [--mode record_mapping|map_build|offline_eval|race|identification|hd_map|e2e_backup|python|map|mapping|localization_eval|perception_eval|vslam_eval|simulator]

Notes:
  - Replace <map_dir> and <bag_path> placeholders in the prepared commands.
  - record_mapping is the Jetson sensor-bag collection layout for map creation.
  - map_build is the note-PC layout for VSLAM/2D map, HD map, section, and eval work.
  - offline_eval replays a camera/LiDAR/TF bag with vehicle disabled for GL/section/HD-map checks.
  - race is the main VSLAM + 2D GL + raceline/controller run layout.
  - mapping/map are legacy aliases for map_build.
  - e2e_backup keeps the camera E2E fallback launch reachable.
EOF
}

choose_mode_interactive() {
  local answer

  if [[ ! -t 0 ]]; then
    echo "$MODE_RACE"
    return
  fi

  while true; do
    echo "Select mode:" >&2
    echo "  1) $MODE_RECORD_MAPPING (Jetson sensor bag: camera + LiDAR + TF + cmd)" >&2
    echo "  2) $MODE_MAP_BUILD (note-PC VSLAM/2D/HD map creation + localization eval)" >&2
    echo "  3) $MODE_OFFLINE_EVAL (bag replay + VSLAM + GL + HD map + sections, no vehicle)" >&2
    echo "  4) $MODE_RACE (Jetson race: VSLAM + 2D GL + HD map + MAP/PP controller)" >&2
    echo "  5) $MODE_IDENTIFICATION (live VSLAM + bag recording for steering/speed identification)" >&2
    echo "  6) $MODE_HD_MAP (HD map editor + section gates + raceline speed tools)" >&2
    echo "  7) $MODE_E2E_BACKUP (camera E2E fallback)" >&2
    echo "  8) $MODE_PYTHON (python workspace)" >&2
    echo "  9) $MODE_LOCALIZATION_EVAL (bag replay + localization eval)" >&2
    echo " 10) $MODE_VSLAM_EVAL (bag replay + VSLAM eval)" >&2
    echo " 11) $MODE_PERCEPTION_EVAL (bag replay + perception eval)" >&2
    echo " 12) $MODE_SIMULATOR (simulator setup)" >&2
    echo " 13) $MODE_TAMIYA (legacy production + monitor)" >&2
    read -r -p "Enter 1-13: " answer

    case "$answer" in
      1|"$MODE_RECORD_MAPPING")
        echo "$MODE_RECORD_MAPPING"
        return
        ;;
      2|"$MODE_MAP_BUILD")
        echo "$MODE_MAP_BUILD"
        return
        ;;
      3|"$MODE_OFFLINE_EVAL")
        echo "$MODE_OFFLINE_EVAL"
        return
        ;;
      4|"$MODE_RACE")
        echo "$MODE_RACE"
        return
        ;;
      5|"$MODE_IDENTIFICATION")
        echo "$MODE_IDENTIFICATION"
        return
        ;;
      6|"$MODE_HD_MAP")
        echo "$MODE_HD_MAP"
        return
        ;;
      7|"$MODE_E2E_BACKUP")
        echo "$MODE_E2E_BACKUP"
        return
        ;;
      8|"$MODE_PYTHON")
        echo "$MODE_PYTHON"
        return
        ;;
      9|"$MODE_LOCALIZATION_EVAL")
        echo "$MODE_LOCALIZATION_EVAL"
        return
        ;;
      10|"$MODE_VSLAM_EVAL")
        echo "$MODE_VSLAM_EVAL"
        return
        ;;
      11|"$MODE_PERCEPTION_EVAL")
        echo "$MODE_PERCEPTION_EVAL"
        return
        ;;
      12|"$MODE_SIMULATOR")
        echo "$MODE_SIMULATOR"
        return
        ;;
      13|"$MODE_TAMIYA")
        echo "$MODE_TAMIYA"
        return
        ;;
      "$MODE_MAPPING"|"$MODE_MAP")
        echo "$MODE_MAP_BUILD"
        return
        ;;
      *)
        echo "Invalid choice: $answer" >&2
        ;;
    esac
  done
}
