#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HD_MAP_YAML=""
WINDOW_WIDTH="1600"
WINDOW_HEIGHT="1000"
SCALE="1.0"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --hd-map-yaml PATH [OPTIONS]
  $(basename "$0") --map-dir DIR [OPTIONS]

Options:
  --hd-map-yaml PATH  editable HD map YAML
  --map-dir DIR       derive <DIR>/<basename DIR>_hd_map.yaml
  --window-width PX   editor window width (default: ${WINDOW_WIDTH})
  --window-height PX  editor window height (default: ${WINDOW_HEIGHT})
  --scale SCALE       initial zoom; 0 fits raster (default: ${SCALE})
  -h, --help          show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hd-map-yaml)
      HD_MAP_YAML="$2"
      shift 2
      ;;
    --map-dir)
      map_dir="${2%/}"
      map_name="$(basename "$map_dir")"
      HD_MAP_YAML="${map_dir}/${map_name}_hd_map.yaml"
      shift 2
      ;;
    --window-width)
      WINDOW_WIDTH="$2"
      shift 2
      ;;
    --window-height)
      WINDOW_HEIGHT="$2"
      shift 2
      ;;
    --scale)
      SCALE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HD_MAP_YAML" ]]; then
  echo "Missing --hd-map-yaml or --map-dir." >&2
  usage >&2
  exit 1
fi

python3 "${REPO_ROOT}/python_ws/map_section_editor/hd_map_section_gate_editor.py" \
  --hd-map-yaml "$HD_MAP_YAML" \
  --window-width "$WINDOW_WIDTH" \
  --window-height "$WINDOW_HEIGHT" \
  --scale "$SCALE"
