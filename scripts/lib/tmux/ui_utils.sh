#!/bin/bash

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [SESSION_NAME] [--mode tamiya|python|map|identification|localization_eval|perception_eval|vslam_eval|simulator]
  $(basename "$0") [--session SESSION_NAME] [--mode tamiya|python|map|identification|localization_eval|perception_eval|vslam_eval|simulator]

Notes:
  - Replace <map_dir> and <bag_path> placeholders in the prepared commands.
  - map mode prepares online map creation, then post-map map-only alignment prep + landmark tracing.
EOF
}

choose_mode_interactive() {
  local answer

  if [[ ! -t 0 ]]; then
    echo "$MODE_TAMIYA"
    return
  fi

  while true; do
    echo "Select mode:" >&2
    echo "  1) $MODE_TAMIYA (production + monitor)" >&2
    echo "  2) $MODE_PYTHON (python workspace)" >&2
    echo "  3) $MODE_MAP (post-map alignment prep + localization eval)" >&2
    echo "  4) $MODE_IDENTIFICATION (live VSLAM + bag recording for MAP lookup)" >&2
    echo "  5) $MODE_LOCALIZATION_EVAL (bag replay + localization eval)" >&2
    echo "  6) $MODE_PERCEPTION_EVAL (bag replay + perception eval)" >&2
    echo "  7) $MODE_VSLAM_EVAL (bag replay + VSLAM eval)" >&2
    echo "  8) $MODE_SIMULATOR (simulator setup)" >&2
    read -r -p "Enter 1-8: " answer

    case "$answer" in
      1|"$MODE_TAMIYA")
        echo "$MODE_TAMIYA"
        return
        ;;
      2|"$MODE_PYTHON")
        echo "$MODE_PYTHON"
        return
        ;;
      3|"$MODE_MAP")
        echo "$MODE_MAP"
        return
        ;;
      4|"$MODE_IDENTIFICATION")
        echo "$MODE_IDENTIFICATION"
        return
        ;;
      5|"$MODE_LOCALIZATION_EVAL")
        echo "$MODE_LOCALIZATION_EVAL"
        return
        ;;
      6|"$MODE_PERCEPTION_EVAL")
        echo "$MODE_PERCEPTION_EVAL"
        return
        ;;
      7|"$MODE_VSLAM_EVAL")
        echo "$MODE_VSLAM_EVAL"
        return
        ;;
      8|"$MODE_SIMULATOR")
        echo "$MODE_SIMULATOR"
        return
        ;;
      *)
        echo "Invalid choice: $answer" >&2
        ;;
    esac
  done
}

