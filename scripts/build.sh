#!/bin/bash

# ワークスペースのディレクトリへ移動
cd /workspaces || { echo "Directory not found"; exit 1; }

# オプションのチェック
CLEAN_MODE=false
for arg in "$@"; do
  if [ "$arg" = "-c" ] || [ "$arg" = "--clean" ]; then
    CLEAN_MODE=true
    break
  fi
done

# クリーンアップの実行
if [ "$CLEAN_MODE" = true ]; then
  echo "Cleaning up build, install, and log directories..."
  rm -rf build/ install/ log/
fi

# ビルドの実行
echo "Starting colcon build with --symlink-install..."
colcon build --symlink-install