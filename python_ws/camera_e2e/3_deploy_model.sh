#!/bin/bash

# ==============================================================================
# Isaac ROS Camera E2E Control Deployment Script
# 1. PyTorch (.pth) -> ONNX (.onnx) via export_onnx.py
# 2. ONNX -> TensorRT (.plan) via trtexec
# 3. Isaac ROS Assets (Triton) directory setup
# ==============================================================================

INPUT_ONNX_PATH=""
MODEL_NAME="pilotnet"
INPUT_TENSOR_NAME="image_input"
PROJECT_NAME="isaac_ros_camera_e2e_control"
CONFIG_FILE="pilotnet_config.pbtxt"
PRECISION="fp16"       # fp16 or fp32
MAX_BATCH_SIZE="1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_CONVERT_SCRIPT="${SCRIPT_DIR}/export_onnx.py"
CHECKPOINT_BASE_DIR="${SCRIPT_DIR}/ckpts"

CHANNELS="3"
HEIGHT="66"
WIDTH="200"

print_parameters() {
  echo "==================================================="
  echo "Camera Model Deployment Configuration"
  echo "==================================================="
  echo "MODEL_NAME        : $MODEL_NAME"
  echo "INPUT_TENSOR_NAME : $INPUT_TENSOR_NAME"
  echo "INPUT_SHAPE (TRT) : 1x${CHANNELS}x${HEIGHT}x${WIDTH}"
  echo "PRECISION         : $PRECISION"
  echo "CONFIG_FILE       : $CONFIG_FILE"
  echo "PROJECT_NAME      : $PROJECT_NAME"
  echo "==================================================="
}

select_checkpoint_interactive() {
  local options=()
  while IFS= read -r pth; do
    options+=("${pth}")
  done < <(find "${CHECKPOINT_BASE_DIR}" -type f -name "best_model.pth" | sort)

  if [[ ${#options[@]} -eq 0 ]]; then
    echo "Error: no checkpoint directories found in ${CHECKPOINT_BASE_DIR}" >&2
    exit 1
  fi
  options+=("Quit")

  echo "--- Select checkpoint directory ---" >&2
  PS3="Select number: "
  select opt in "${options[@]}"; do
    if [[ "$opt" == "Quit" ]]; then
      exit 1
    elif [[ -n "$opt" ]]; then
      if [[ -f "$opt" ]]; then
        echo "$opt"
        return 0
      else
        echo "Error: file not found: $opt" >&2
      fi
    fi
  done
}

setup_model() {
  if [[ ! -f "$INPUT_ONNX_PATH" ]]; then
    echo "Error: ONNX file not found: ${INPUT_ONNX_PATH}"
    exit 1
  fi

  local assets_base="/workspaces/isaac_ros_assets/models"
  local model_root="${assets_base}/${MODEL_NAME}"

  local version=1
  while [[ -d "${model_root}/${version}" ]]; do
    version=$((version + 1))
  done

  local version_path="${model_root}/${version}"
  mkdir -p "${version_path}"
  cp "${INPUT_ONNX_PATH}" "${version_path}/model.onnx"

  echo "Converting ONNX to TensorRT engine..."
  /usr/src/tensorrt/bin/trtexec \
    --onnx="${version_path}/model.onnx" \
    --saveEngine="${version_path}/model.plan" \
    --minShapes=${INPUT_TENSOR_NAME}:1x${CHANNELS}x${HEIGHT}x${WIDTH} \
    --optShapes=${INPUT_TENSOR_NAME}:1x${CHANNELS}x${HEIGHT}x${WIDTH} \
    --maxShapes=${INPUT_TENSOR_NAME}:${MAX_BATCH_SIZE}x${CHANNELS}x${HEIGHT}x${WIDTH} \
    --${PRECISION} \
    --verbose

  if [[ $? -ne 0 ]]; then
    echo "Error: trtexec failed."
    exit 1
  fi

  echo "Copying config.pbtxt..."
  local pkg_share_path
  pkg_share_path=$(ros2 pkg prefix ${PROJECT_NAME} --share 2>/dev/null)
  local config_source_path="${pkg_share_path}/config/${CONFIG_FILE}"

  if [[ ! -f "$config_source_path" ]]; then
    if [[ -f "${SCRIPT_DIR}/config/${CONFIG_FILE}" ]]; then
      config_source_path="${SCRIPT_DIR}/config/${CONFIG_FILE}"
    else
      echo "Error: '${CONFIG_FILE}' not found."
      exit 1
    fi
  fi

  cp "${config_source_path}" "${model_root}/config.pbtxt"

  echo "Deploy complete: ${MODEL_NAME} (Version ${version})"
}

if [[ -z "$INPUT_ONNX_PATH" ]]; then
  SELECTED_PTH=$(select_checkpoint_interactive)
  if [[ $? -ne 0 ]]; then exit 1; fi

  if [[ ! -f "$PYTHON_CONVERT_SCRIPT" ]]; then
    echo "Error: $PYTHON_CONVERT_SCRIPT not found."
    exit 1
  fi

  echo "Exporting ONNX..."
  python3 "$PYTHON_CONVERT_SCRIPT" \
    --checkpoint "$SELECTED_PTH" \
    --channels "$CHANNELS" \
    --height "$HEIGHT" \
    --width "$WIDTH"

  if [[ $? -ne 0 ]]; then
    echo "Error: ONNX export failed."
    exit 1
  fi

  INPUT_ONNX_PATH="${SELECTED_PTH%.*}.onnx"
fi

print_parameters
setup_model
