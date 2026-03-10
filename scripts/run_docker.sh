#!/bin/bash

# デフォルトのワークスペース設定
SCRIPT_DIR="${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts"

# 引数の解析
USE_BUILD=""
while getopts "b" opt; do
  case $opt in
    b)
      USE_BUILD="-b"
      ;;
    \?)
      echo "Usage: $0 [-b]"
      exit 1
      ;;
  esac
done

# 1. 設定ファイルの作成
if [ -d "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
    cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=ros2_humble.additional_setting
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF
    echo "Configuration file updated."
else
    echo "Error: Directory $SCRIPT_DIR not found."
    exit 1
fi

# 2. run_dev.sh の実行
if [ -d "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
    echo "Running: ./run_dev.sh $USE_BUILD"
    bash ./run_dev.sh $USE_BUILD
else
    echo "Error: Directory $SCRIPT_DIR not found."
    exit 1
fi