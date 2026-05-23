#!/usr/bin/env bash
set -u

section() {
  printf '\n== %s ==\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

search_files() {
  local pattern="$1"
  shift

  if have_cmd rg; then
    rg -n "$pattern" "$@" 2>/dev/null || true
  else
    grep -R -n -E "$pattern" "$@" 2>/dev/null || true
  fi
}

section "ROS node list"
if have_cmd ros2; then
  node_list="$(ros2 node list 2>&1 || true)"
  printf '%s\n' "$node_list"

  section "Duplicate node names"
  printf '%s\n' "$node_list" \
    | sed '/^WARNING:/d;/^$/d' \
    | sort \
    | uniq -d

  section "/lidar_container info"
  ros2 node info /lidar_container 2>&1 || true
else
  echo "ros2 command not found"
fi

section "Processes that can own LiDAR/container nodes"
ps -eo pid,ppid,pgid,sid,tty,etimes,cmd \
  | grep -E 'component_container|ros2 launch|launch_system|lidar_container|sensor_kit|system.launch' \
  | grep -v grep || true

section "tmux sessions"
if have_cmd tmux; then
  tmux ls 2>/dev/null || true
  section "tmux panes"
  tmux list-panes -a -F '#{session_name}:#{window_name}.#{pane_index} #{pane_pid} #{pane_current_command} #{pane_current_path}' 2>/dev/null || true
else
  echo "tmux command not found"
fi

section "Installed launch files that create a lidar_container"
for root in /workspaces/install "$PWD/install" "$PWD/ros2_ws/install"; do
  if [[ -d "$root" ]]; then
    echo "-- $root"
    search_files 'component_container.*lidar_container|name="\$\(var lidar_container_name\)"|create_lidar_container" value="true"|create_lidar_container" default="true"' "$root"
  fi
done

section "Source launch files that create a lidar_container"
if [[ -d "$PWD/ros2_ws/src" ]]; then
  search_files 'component_container.*lidar_container|name="\$\(var lidar_container_name\)"|create_lidar_container" value="true"|create_lidar_container" default="true"' "$PWD/ros2_ws/src"
fi
