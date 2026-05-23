#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RECORD_ROOT="/record"
BAG_PATH=""
MAP_DIR=""
OUTPUT_YAML=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra_build_speed_controller_feedforward_args...]

Options:
  --bag PATH           rosbag2 directory or metadata.yaml
  --record-root PATH   root used for interactive bag selection (default: /record)
  --map-dir PATH       write speed_controller_feedforward.param.yaml under this map dir
  --output-yaml PATH   explicit output YAML path
  -h, --help           show this help

Examples:
  $(basename "$0") --bag /record/session/take/metadata.yaml --map-dir /map/course_a
  $(basename "$0") --record-root /record --map-dir /map/course_a -- --max-abs-steer 0.10
EOF
}

choose_bag_interactive() {
  local -a candidates=()
  local metadata
  local idx
  local answer

  if [[ ! -d "$RECORD_ROOT" ]]; then
    echo "Record root not found: $RECORD_ROOT" >&2
    exit 1
  fi

  while IFS= read -r metadata; do
    candidates+=("$metadata")
  done < <(find "$RECORD_ROOT" -type f -name metadata.yaml 2>/dev/null | sort -r | head -n 30)

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "No metadata.yaml found under $RECORD_ROOT" >&2
    exit 1
  fi

  echo "Select rosbag:"
  for idx in "${!candidates[@]}"; do
    printf "  %2d) %s\n" "$((idx + 1))" "${candidates[$idx]}"
  done
  read -r -p "bag [1]: " answer
  answer="${answer:-1}"
  if ! [[ "$answer" =~ ^[0-9]+$ ]] || [[ "$answer" -lt 1 || "$answer" -gt "${#candidates[@]}" ]]; then
    echo "Invalid selection: $answer" >&2
    exit 1
  fi
  BAG_PATH="${candidates[$((answer - 1))]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --bag|--bag-path)
      BAG_PATH="$2"
      shift 2
      ;;
    --record-root)
      RECORD_ROOT="$2"
      shift 2
      ;;
    --map-dir)
      MAP_DIR="$2"
      shift 2
      ;;
    --output-yaml|--param-yaml)
      OUTPUT_YAML="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$BAG_PATH" ]]; then
  choose_bag_interactive
fi

if [[ -z "$OUTPUT_YAML" && -n "$MAP_DIR" ]]; then
  OUTPUT_YAML="${MAP_DIR%/}/speed_controller_feedforward.param.yaml"
fi

cmd=(
  python3
  "$REPO_ROOT/python_ws/data_analysis/build_speed_controller_feedforward.py"
  --bag
  "$BAG_PATH"
)

if [[ -n "$OUTPUT_YAML" ]]; then
  cmd+=(--param-yaml "$OUTPUT_YAML")
fi

cmd+=("${EXTRA_ARGS[@]}")

printf 'Command: '
printf '%q ' "${cmd[@]}"
echo

exec "${cmd[@]}"
