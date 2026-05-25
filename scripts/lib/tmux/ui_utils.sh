#!/bin/bash

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [SESSION_NAME] [--mode record|map|race|e2e|lidar_e2e|lidar_e2e_train|identification|hd_map_eval|perception_eval]
  $(basename "$0") [--session SESSION_NAME] [--mode record|map|race|e2e|lidar_e2e|lidar_e2e_train|identification|hd_map_eval|perception_eval]

Notes:
  - Replace <map_dir> and <bag_path> placeholders in the prepared commands.
  - record is the Jetson sensor-bag collection layout for map creation.
  - map is the note-PC layout for VSLAM/HD map creation, editing, and hd_map_eval.
  - race is the main VSLAM + 2D GL + HD map controller run layout.
  - e2e keeps the camera E2E fallback launch reachable.
  - lidar_e2e_train prepares virtual-scan generation, dataset extraction, train, transfer, and deploy commands.
  - identification records VSLAM odom + cmd_drive for steering/speed identification.
  - Legacy aliases still work: record_mapping, map_build, mapping, offline_eval, hd_map, e2e_backup.
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
    echo "  1) $MODE_RECORD (sensor record: camera + LiDAR + TF + cmd)" >&2
    echo "  2) $MODE_MAP (VSLAM/HD map creation, editing, and hd_map_eval)" >&2
    echo "  3) $MODE_RACE (production run)" >&2
    echo "  4) $MODE_E2E (camera E2E fallback run)" >&2
    echo "  5) $MODE_IDENTIFICATION (steering/speed identification)" >&2
    echo "  6) $MODE_LIDAR_E2E (LiDAR E2E run)" >&2
    echo "  7) $MODE_RECORD_VIRTUAL_SCAN (record virtual scan offline)" >&2
    echo "  8) $MODE_PERCEPTION_EVAL (LiDAR camera crop perception evaluation)" >&2
    echo "  9) $MODE_LIDAR_E2E_TRAIN (LiDAR E2E dataset/train/deploy)" >&2
    read -r -p "Enter 1-9: " answer

    case "$answer" in
      "$MODE_OFFLINE_EVAL"|"$MODE_HD_MAP_EVAL")
        echo "$MODE_OFFLINE_EVAL"
        return
        ;;
      1|"$MODE_RECORD"|"$MODE_RECORD_MAPPING")
        echo "$MODE_RECORD"
        return
        ;;
      2|"$MODE_MAP"|"$MODE_MAPPING"|"$MODE_MAP_BUILD"|"$MODE_HD_MAP")
        echo "$MODE_MAP"
        return
        ;;
      3|"$MODE_RACE"|"$MODE_TAMIYA"|"$MODE_PRODUCTION")
        echo "$MODE_RACE"
        return
        ;;
      4|"$MODE_E2E"|"$MODE_E2E_BACKUP")
        echo "$MODE_E2E"
        return
        ;;
      5|"$MODE_IDENTIFICATION")
        echo "$MODE_IDENTIFICATION"
        return
        ;;
      6|"$MODE_LIDAR_E2E")
        echo "$MODE_LIDAR_E2E"
        return
        ;;
      7|"$MODE_RECORD_VIRTUAL_SCAN")
        echo "$MODE_RECORD_VIRTUAL_SCAN"
        return
        ;;
      8|"$MODE_PERCEPTION_EVAL")
        echo "$MODE_PERCEPTION_EVAL"
        return
        ;;
      9|"$MODE_LIDAR_E2E_TRAIN")
        echo "$MODE_LIDAR_E2E_TRAIN"
        return
        ;;
      *)
        echo "Invalid choice: $answer" >&2
        ;;
    esac
  done
}
