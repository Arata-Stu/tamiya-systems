#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Isaac ROS Camera E2E Control Deployment Script
# 1. PyTorch (.pth) -> ONNX (.onnx) via export_onnx.py
# 2. ONNX -> TensorRT (.plan) via trtexec
# 3. Isaac ROS Assets (Triton) directory setup
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ONNX_PATH=""
CHECKPOINT_BASE_DIR="${SCRIPT_DIR}/ckpts"
MODEL_NAME="pilotnet"
INPUT_TENSOR_NAME="image_input"
PROJECT_NAME="isaac_ros_camera_e2e_control"
CONFIG_BASENAME="pilotnet_config"
PRECISION="fp16"
MAX_BATCH_SIZE="1"
KEEP_VERSIONS="false"
PYTHON_CONVERT_SCRIPT="${SCRIPT_DIR}/export_onnx.py"

CHANNELS="3"
HEIGHT="120"
WIDTH="212"

show_help() {
  cat <<EOF
Usage:
  $0 [CHECKPOINT_BASE_DIR] [options]

Description:
  Search best_model.pth -> export ONNX -> build TensorRT engine -> deploy to Triton model repo.

Options:
  --onnx PATH                 Use existing ONNX and skip export_onnx.py
  --precision {fp16|fp32}     TensorRT compute precision (I/O stays FP32, default: fp16)
  --max-batch-size N          Max batch size for TensorRT profile (default: 1)
  --keep-versions             Keep existing Triton numeric version dirs and deploy to next version
  -h, --help                  Show this help message

Examples:
  $0
  $0 ./ckpts --precision fp32
  $0 --onnx ./ckpts/train/run/best_model.onnx --precision fp16
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

find_trtexec() {
  if [[ -x "/usr/src/tensorrt/bin/trtexec" ]]; then
    echo "/usr/src/tensorrt/bin/trtexec"
    return
  fi
  if command -v trtexec >/dev/null 2>&1; then
    command -v trtexec
    return
  fi
  die "trtexec not found. Install TensorRT or add trtexec to PATH."
}

resolve_config_source() {
  local config_file="$1"
  local pkg_share_path=""
  local config_source_path=""

  if [[ -f "${SCRIPT_DIR}/config/${config_file}" ]]; then
    config_source_path="${SCRIPT_DIR}/config/${config_file}"
  fi

  if [[ -z "${config_source_path}" ]] && pkg_share_path="$(ros2 pkg prefix "${PROJECT_NAME}" --share 2>/dev/null)"; then
    if [[ -f "${pkg_share_path}/config/${config_file}" ]]; then
      config_source_path="${pkg_share_path}/config/${config_file}"
    fi
  fi

  [[ -n "${config_source_path}" ]] || die "'${config_file}' not found in ${SCRIPT_DIR}/config or package share"
  echo "${config_source_path}"
}

select_checkpoint_interactive() {
  local options=()
  while IFS= read -r pth; do
    options+=("${pth}")
  done < <(find "${CHECKPOINT_BASE_DIR}" -type f -name "best_model.pth" | sort)

  if [[ ${#options[@]} -eq 0 ]]; then
    die "no 'best_model.pth' found in ${CHECKPOINT_BASE_DIR}"
  fi
  options+=("Quit")

  echo "--- Select checkpoint file ---" >&2
  PS3="Select number: "
  select opt in "${options[@]}"; do
    if [[ "${opt}" == "Quit" ]]; then
      exit 1
    elif [[ -n "${opt}" && -f "${opt}" ]]; then
      echo "${opt}"
      return
    else
      echo "Error: invalid selection" >&2
    fi
  done
}

print_parameters() {
  local config_file="${CONFIG_BASENAME}_fp32.pbtxt"
  echo "==================================================="
  echo "Camera Model Deployment Configuration"
  echo "==================================================="
  echo "CHECKPOINT_BASE   : ${CHECKPOINT_BASE_DIR}"
  echo "INPUT_ONNX_PATH   : ${INPUT_ONNX_PATH:-<auto export>}"
  echo "MODEL_NAME        : ${MODEL_NAME}"
  echo "INPUT_TENSOR_NAME : ${INPUT_TENSOR_NAME}"
  echo "INPUT_SHAPE (TRT) : 1x${CHANNELS}x${HEIGHT}x${WIDTH}"
  echo "PRECISION         : ${PRECISION}"
  echo "KEEP_VERSIONS     : ${KEEP_VERSIONS}"
  echo "CONFIG_FILE       : ${config_file}"
  echo "PROJECT_NAME      : ${PROJECT_NAME}"
  echo "==================================================="
}

clear_model_versions() {
  local model_root="$1"
  local version_dir
  local version_name

  [[ -d "${model_root}" ]] || return 0

  for version_dir in "${model_root}"/[0-9]*; do
    [[ -d "${version_dir}" ]] || continue
    version_name="$(basename "${version_dir}")"
    [[ "${version_name}" =~ ^[0-9]+$ ]] || continue
    rm -rf -- "${version_dir}"
  done
}

setup_model() {
  [[ -f "${INPUT_ONNX_PATH}" ]] || die "ONNX file not found: ${INPUT_ONNX_PATH}"

  local config_file="${CONFIG_BASENAME}_fp32.pbtxt"
  local config_source_path
  config_source_path="$(resolve_config_source "${config_file}")"

  local assets_base="/workspaces/isaac_ros_assets/models"
  local model_root="${assets_base}/${MODEL_NAME}"

  if [[ "${KEEP_VERSIONS}" != "true" ]]; then
    echo "Removing existing Triton versions under ${model_root}..."
    clear_model_versions "${model_root}"
  fi

  local version=1
  while [[ -d "${model_root}/${version}" ]]; do
    version=$((version + 1))
  done

  local version_path="${model_root}/${version}"
  mkdir -p "${version_path}"
  cp "${INPUT_ONNX_PATH}" "${version_path}/model.onnx"

  local trtexec
  trtexec="$(find_trtexec)"
  local precision_args=()
  if [[ "${PRECISION}" == "fp16" ]]; then
    precision_args+=(--fp16)
  fi

  echo "Converting ONNX to TensorRT engine (${PRECISION})..."
  "${trtexec}" \
    --onnx="${version_path}/model.onnx" \
    --saveEngine="${version_path}/model.plan" \
    --minShapes="${INPUT_TENSOR_NAME}:1x${CHANNELS}x${HEIGHT}x${WIDTH}" \
    --optShapes="${INPUT_TENSOR_NAME}:1x${CHANNELS}x${HEIGHT}x${WIDTH}" \
    --maxShapes="${INPUT_TENSOR_NAME}:${MAX_BATCH_SIZE}x${CHANNELS}x${HEIGHT}x${WIDTH}" \
    "${precision_args[@]}" \
    --verbose

  echo "Copying config.pbtxt (${config_file})..."
  cp "${config_source_path}" "${model_root}/config.pbtxt"

  echo "---------------------------------------------------"
  echo "Deploy complete: ${MODEL_NAME} (Version ${version})"
  echo "Location: ${version_path}"
  echo "---------------------------------------------------"
}

parse_args() {
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        show_help
        exit 0
        ;;
      --onnx)
        [[ $# -ge 2 ]] || die "--onnx requires a path"
        INPUT_ONNX_PATH="$2"
        shift 2
        ;;
      --precision)
        [[ $# -ge 2 ]] || die "--precision requires fp16 or fp32"
        PRECISION="$2"
        shift 2
        ;;
      --max-batch-size)
        [[ $# -ge 2 ]] || die "--max-batch-size requires a number"
        MAX_BATCH_SIZE="$2"
        shift 2
        ;;
      --keep-versions)
        KEEP_VERSIONS="true"
        shift
        ;;
      --*)
        die "unknown option: $1"
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional[@]} -gt 1 ]]; then
    die "too many positional arguments"
  fi
  if [[ ${#positional[@]} -eq 1 ]]; then
    CHECKPOINT_BASE_DIR="${positional[0]}"
  fi
}

main() {
  parse_args "$@"
  [[ "${PRECISION}" == "fp16" || "${PRECISION}" == "fp32" ]] || die "precision must be fp16 or fp32"

  if [[ -n "${INPUT_ONNX_PATH}" ]]; then
    INPUT_ONNX_PATH="$(realpath "${INPUT_ONNX_PATH}")"
    [[ -f "${INPUT_ONNX_PATH}" ]] || die "ONNX file not found: ${INPUT_ONNX_PATH}"
  else
    [[ -d "${CHECKPOINT_BASE_DIR}" ]] || die "Directory not found: ${CHECKPOINT_BASE_DIR}"
    CHECKPOINT_BASE_DIR="$(realpath "${CHECKPOINT_BASE_DIR}")"

    local selected_pth
    selected_pth="$(select_checkpoint_interactive)"

    [[ -f "${PYTHON_CONVERT_SCRIPT}" ]] || die "export script not found: ${PYTHON_CONVERT_SCRIPT}"
    echo "Exporting ONNX..."
    python3 "${PYTHON_CONVERT_SCRIPT}" \
      --checkpoint "${selected_pth}" \
      --channels "${CHANNELS}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --input_normalization external

    INPUT_ONNX_PATH="${selected_pth%.*}.onnx"
    [[ -f "${INPUT_ONNX_PATH}" ]] || die "ONNX export output not found: ${INPUT_ONNX_PATH}"
  fi

  print_parameters
  setup_model
}

main "$@"
