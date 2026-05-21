#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_TOOL="${REPO_ROOT}/python_ws/map_section_editor/control_filter_config_editor.py"
DEFAULT_BASE_CONFIG="${REPO_ROOT}/ros2_ws/src/control/control_filter/config/control_filter.param.yaml"

MAP_DIR=""
SECTIONS_CSV=""
OUTPUT_PATH=""
BASE_CONFIG=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--map-dir DIR] [--sections-csv PATH] [--output PATH] [--base-config PATH]

Examples:
  $(basename "$0") --map-dir /map/course_a
  $(basename "$0") --sections-csv /map/course_a/sections_pixels.csv --output /map/course_a/control_filter.param.yaml

Defaults:
  --map-dir DIR      -> sections CSV: DIR/sections_pixels.csv
                     -> output YAML: DIR/control_filter.param.yaml
  --base-config      -> output file if it already exists, otherwise:
                        ${DEFAULT_BASE_CONFIG}
EOF
}

discover_map_dirs() {
  find /map -name sections_pixels.csv -type f 2>/dev/null | sort
}

choose_map_dir_interactive() {
  local candidates=()
  local idx=1
  local choice
  while IFS= read -r path; do
    candidates+=("$path")
  done < <(discover_map_dirs)

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "No sections_pixels.csv found under /map" >&2
    return 1
  fi

  echo "Available section maps:"
  for path in "${candidates[@]}"; do
    printf "  %2d) %s\n" "$idx" "$(dirname "$path")"
    idx=$((idx + 1))
  done
  echo ""
  read -r -p "Select map dir [1]: " choice
  choice="${choice:-1}"
  if [[ ! "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#candidates[@]} )); then
    echo "Invalid selection: ${choice}" >&2
    return 1
  fi
  MAP_DIR="$(dirname "${candidates[$((choice - 1))]}")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map-dir)
      MAP_DIR="$2"
      shift 2
      ;;
    --sections-csv)
      SECTIONS_CSV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
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

if [[ -z "$MAP_DIR" && -z "$SECTIONS_CSV" ]]; then
  choose_map_dir_interactive
fi

if [[ -n "$MAP_DIR" ]]; then
  MAP_DIR="${MAP_DIR%/}"
  [[ -z "$SECTIONS_CSV" ]] && SECTIONS_CSV="${MAP_DIR}/sections_pixels.csv"
  [[ -z "$OUTPUT_PATH" ]] && OUTPUT_PATH="${MAP_DIR}/control_filter.param.yaml"
fi

if [[ -z "$SECTIONS_CSV" ]]; then
  echo "--sections-csv or --map-dir is required" >&2
  exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$(dirname "$SECTIONS_CSV")/control_filter.param.yaml"
fi

if [[ ! -f "$SECTIONS_CSV" ]]; then
  echo "sections CSV not found: $SECTIONS_CSV" >&2
  exit 1
fi

if [[ ! -f "$PY_TOOL" ]]; then
  echo "editor tool not found: $PY_TOOL" >&2
  exit 1
fi

if [[ -z "$BASE_CONFIG" ]]; then
  if [[ -f "$OUTPUT_PATH" ]]; then
    BASE_CONFIG="$OUTPUT_PATH"
  else
    BASE_CONFIG="$DEFAULT_BASE_CONFIG"
  fi
fi

exec python3 "$PY_TOOL" \
  --sections-csv "$SECTIONS_CSV" \
  --output "$OUTPUT_PATH" \
  --base-config "$BASE_CONFIG"
