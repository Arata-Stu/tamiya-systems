#!/bin/bash

# ==============================================================================
# Isaac ROS Lidar E2E Control Deployment Script
# 1. PyTorch (.pth) -> ONNX (.onnx) via export_onnx.py
# 2. ONNX -> TensorRT (.plan) via trtexec
# 3. Isaac ROS Assets (Triton) Directory Setup
# ==============================================================================

# --- デフォルト設定 ---
INPUT_ONNX_PATH=""
MODEL_NAME="tinylidarnet"
SCAN_POINTS="1081"           # LiDARの点数（トレーニング時と一致させる）
INPUT_TENSOR_NAME="scan_input"
PROJECT_NAME="isaac_ros_lidar_e2e_control"
CONFIG_FILE="tinylidarnet_config.pbtxt"
PRECISION="fp16"             # fp16 or fp32
MAX_BATCH_SIZE="1"
PYTHON_CONVERT_SCRIPT="export_onnx.py"
CHECKPOINT_BASE_DIR="../ckpts/tinylidarnet"

# --- 関数の定義 ---

function print_parameters() {
  echo "==================================================="
  echo "🚀 LiDAR Model Deployment Configuration"
  echo "==================================================="
  echo "MODEL_NAME        : $MODEL_NAME"
  echo "SCAN_POINTS       : $SCAN_POINTS"
  echo "INPUT_TENSOR_NAME : $INPUT_TENSOR_NAME"
  echo "INPUT_SHAPE (TRT) : 1x1x$SCAN_POINTS (Batch x History x Points)"
  echo "PRECISION         : $PRECISION"
  echo "CONFIG_FILE       : $CONFIG_FILE"
  echo "PROJECT_NAME      : $PROJECT_NAME"
  echo "==================================================="
}

function select_checkpoint_interactive() {
  local options=()
  # チェックポイントディレクトリを検索
  while IFS= read -r dir; do
    options+=("$(basename "$dir")")
  done < <(find "${CHECKPOINT_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ ${#options[@]} -eq 0 ]]; then
    echo "❌ Error: ${CHECKPOINT_BASE_DIR} 内にディレクトリが見つかりません。" >&2
    exit 1
  fi
  options+=("終了 (Quit)")

  echo "--- 1. チェックポイントディレクトリを選択してください ---" >&2
  PS3="番号を入力してください: "
  select opt in "${options[@]}"; do
    if [[ "$opt" == "終了 (Quit)" ]]; then
      exit 1
    elif [[ -n "$opt" ]]; then
      local pth_file="${CHECKPOINT_BASE_DIR}/${opt}/best_model.pth"
      if [[ -f "$pth_file" ]]; then
        echo "$pth_file"
        return 0
      else
        echo "❌ Error: $opt 内に best_model.pth が見つかりません。" >&2
      fi
    fi
  done
}

function setup_model() {
  # ONNXの存在確認
  if [[ ! -f "$INPUT_ONNX_PATH" ]]; then
    echo "❌ Error: ONNXファイルが見つかりません: ${INPUT_ONNX_PATH}"
    exit 1
  fi

  # デプロイ先パスの設定（Isaac ROSの標準パス）
  local assets_base="/workspaces/isaac_ros_assets/models"
  local model_root="${assets_base}/${MODEL_NAME}"
  
  # バージョニング（既存のフォルダを避けて新しい番号を作成）
  local version=1
  while [[ -d "${model_root}/${version}" ]]; do
    version=$((version + 1))
  done
  
  local version_path="${model_root}/${version}"
  echo "📂 作成中: ${version_path}"
  mkdir -p "${version_path}"

  # ONNXをコピー（Tritonの慣習でmodel.onnxにリネーム）
  cp "${INPUT_ONNX_PATH}" "${version_path}/model.onnx"

  echo "🔄 TensorRTエンジン (.plan) へ変換中..."
  # trtexecによる変換。ScanEncoderNodeの [1, 1, Points] に合わせて形状を指定。
  /usr/src/tensorrt/bin/trtexec \
    --onnx="${version_path}/model.onnx" \
    --saveEngine="${version_path}/model.plan" \
    --minShapes=${INPUT_TENSOR_NAME}:1x1x${SCAN_POINTS} \
    --optShapes=${INPUT_TENSOR_NAME}:1x1x${SCAN_POINTS} \
    --maxShapes=${INPUT_TENSOR_NAME}:${MAX_BATCH_SIZE}x1x${SCAN_POINTS} \
    --${PRECISION} \
    --verbose
  
  if [[ $? -ne 0 ]]; then
    echo "❌ Error: trtexec の実行に失敗しました。"
    exit 1
  fi

  # Triton設定ファイル (pbtxt) のコピー
  echo "📄 設定ファイル (config.pbtxt) を配置中..."
  
  # 1. ROS 2パッケージ内を検索
  local pkg_share_path=$(ros2 pkg prefix ${PROJECT_NAME} --share 2>/dev/null)
  local config_source_path="${pkg_share_path}/config/${CONFIG_FILE}"

  # 2. パッケージ内にない場合、カレントディレクトリの config/ を検索
  if [[ ! -f "$config_source_path" ]]; then
    if [[ -f "./config/${CONFIG_FILE}" ]]; then
      config_source_path="./config/${CONFIG_FILE}"
    else
      echo "❌ Error: '${CONFIG_FILE}' が見つかりません。"
      exit 1
    fi
  fi
  
  # デプロイ先に config.pbtxt という名前でコピー
  cp "${config_source_path}" "${model_root}/config.pbtxt"
  
  echo "==================================================="
  echo "✨ デプロイ完了: ${MODEL_NAME} (Version ${version})"
  echo "==================================================="
}

# --- メイン実行フロー ---

# 1. 引数でONNXが指定されていない場合、対話形式で生成
if [[ -z "$INPUT_ONNX_PATH" ]]; then
  SELECTED_PTH=$(select_checkpoint_interactive)
  if [[ $? -ne 0 ]]; then exit 1; fi

  echo "✅ チェックポイントを選択しました: $SELECTED_PTH"

  # 2. Pythonエクスポートスクリプトの実行
  if [[ ! -f "$PYTHON_CONVERT_SCRIPT" ]]; then
    echo "❌ Error: $PYTHON_CONVERT_SCRIPT が見つかりません。"
    exit 1
  fi

  echo "🔄 ONNXを生成中 (Scan Points: $SCAN_POINTS)..."
  python3 "$PYTHON_CONVERT_SCRIPT" --checkpoint "$SELECTED_PTH" --scan_points "$SCAN_POINTS"
  
  if [[ $? -ne 0 ]]; then
    echo "❌ Error: ONNXのエクスポートに失敗しました。"
    exit 1
  fi

  # 生成されたONNXのパスを設定（.pth を .onnx に置換）
  INPUT_ONNX_PATH="${SELECTED_PTH%.*}.onnx"
fi

# 3. パラメータの表示とデプロイ実行
print_parameters
setup_model