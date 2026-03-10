#!/bin/bash

# デフォルトのワークスペース設定
CONFIG_DIR="${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts"
RUN_DIR="${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common"

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
if [ -d "$CONFIG_DIR" ]; then
    cd "$CONFIG_DIR"
    cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=ros2_humble.additional_setting
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF
    echo "Configuration file updated."
else
    echo "Error: Directory $CONFIG_DIR not found."
    exit 1
fi

# 2. run_dev.sh の実行
if [ -d "$RUN_DIR" ]; then
    cd "$RUN_DIR"
    echo "Running: ./run_dev.sh $USE_BUILD"
    bash ./run_dev.sh $USE_BUILD
else
    echo "Error: Directory $RUN_DIR not found."
    exit 1
fi