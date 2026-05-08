#!/bin/bash

# デフォルトのワークスペース設定
SCRIPT_DIR="${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts"
IMAGE_KEY="ros2_humble.additional_setting"

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

container_name() {
    local platform
    platform="$(uname -m)"
    echo "isaac_ros_dev-${platform}-container"
}

is_container_running() {
    local name="$1"
    docker ps --quiet --filter "name=^/${name}$" | grep -q .
}

exec_container() {
    local name="$1"
    local workspace

    workspace="$(docker exec "$name" printenv ISAAC_ROS_WS 2>/dev/null || true)"
    workspace="${workspace:-/workspaces}"

    echo "Exec into running container: $name"
    echo "Preferred workspace: $workspace"
    exec docker exec -it -u admin "$name" /bin/bash -lc '
        for dir in "${ISAAC_ROS_WS:-}" /workspaces /workspaces/tamiya-systems /workspaces/isaac_ros-dev /; do
            if [ -n "$dir" ] && [ -d "$dir" ]; then
                cd "$dir"
                break
            fi
        done
        echo "Workspace: $(pwd)"
        exec /bin/bash
    '
}

ask_exec_running_container() {
    local name="$1"
    local answer

    if [[ ! -t 0 ]]; then
        return 1
    fi

    echo "Container is already running: $name"
    read -r -p "Exec into it? (Y/n): " answer
    answer="${answer:-y}"

    [[ "$answer" =~ ^[Yy]$ ]]
}

# 1. 設定ファイルの作成
if [ -d "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
    cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=${IMAGE_KEY}
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF
    echo "Configuration file updated."
else
    echo "Error: Directory $SCRIPT_DIR not found."
    exit 1
fi

CONTAINER_NAME="$(container_name)"
if command -v docker >/dev/null 2>&1 && is_container_running "$CONTAINER_NAME"; then
    if ask_exec_running_container "$CONTAINER_NAME"; then
        exec_container "$CONTAINER_NAME"
    fi

    echo "Container is still running. Skipping new docker launch."
    exit 0
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
